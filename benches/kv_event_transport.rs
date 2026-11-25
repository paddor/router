//! KV event subscriber throughput: libzmq (`zmq` 0.10, the current path)
//! against OMQ (`omq-tokio`) and `zeromq` 0.7 (#607).
//!
//! A child process publishes vLLM-shaped `[topic, seq, payload]` batches from
//! a libzmq XPUB over TCP loopback. `ZMQ_XPUB_NODROP` makes it wait instead of
//! dropping at its HWM, so the reported rate is what the subscriber sustains.
//! Each subscriber runs the live loop of `source/events/subscriber.rs`:
//! receive, check topic and sequence, optionally `decode_batch`, then yield.
//! Index updates are not included. `libzmq-blocking` receives on a plain
//! thread without tokio and marks the publisher and libzmq ceiling. The
//! `-coop` variants replace the per-frame `yield_now` with Tokio's coop
//! budget: `consume_budget` for libzmq, and OMQ, whose `recv` spends it.
//!
//! Payloads follow vLLM's `KVEventBatch` encoding: `[ts, [BlockStored...]]`
//! with u64 block hashes, 16-token blocks and token ids below 150k.
//!
//! Run: `cargo bench --bench kv_event_transport`
//! Options: `-- --rounds N` (default 3), `--scale F` (frame count factor),
//! `--only NAME` (one subscriber).

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rmpv::Value;
use tokio::io::unix::AsyncFd;
use tokio::io::Interest;
use vllm_router_rs::kv_index::decode_batch;

/// vLLM's default topic.
const TOPIC: &[u8] = b"";
/// The router's `MAX_KV_EVENT_BYTES`.
const MAX_EVENT_BYTES: usize = 64 * 1024 * 1024;
/// vLLM `ZmqEventPublisher` default.
const PUBLISHER_HWM: i32 = 100_000;
const BLOCK_SIZE: usize = 16;
const DISTINCT_PAYLOADS: usize = 64;

#[derive(Clone, Copy)]
struct Profile {
    name: &'static str,
    events: usize,
    blocks: usize,
    frames: u64,
}

const PROFILES: &[Profile] = &[
    // One block filled during decode.
    Profile {
        name: "1 event x 1 block",
        events: 1,
        blocks: 1,
        frames: 1_000_000,
    },
    // A scheduler step storing blocks for several requests.
    Profile {
        name: "4 events x 4 blocks",
        events: 4,
        blocks: 4,
        frames: 500_000,
    },
    // A 4096-token prefill.
    Profile {
        name: "1 event x 256 blocks",
        events: 1,
        blocks: 256,
        frames: 50_000,
    },
];

#[derive(Clone, Copy, PartialEq)]
enum Sub {
    LibzmqBlocking,
    Libzmq,
    LibzmqCoop,
    Omq,
    OmqCoop,
    Zeromq,
}

const SUBS: &[Sub] = &[
    Sub::LibzmqBlocking,
    Sub::Libzmq,
    Sub::LibzmqCoop,
    Sub::Omq,
    Sub::OmqCoop,
    Sub::Zeromq,
];

impl Sub {
    fn name(self) -> &'static str {
        match self {
            Sub::LibzmqBlocking => "libzmq-blocking",
            Sub::Libzmq => "libzmq",
            Sub::LibzmqCoop => "libzmq-coop",
            Sub::Omq => "omq",
            Sub::OmqCoop => "omq-coop",
            Sub::Zeromq => "zeromq",
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("publish") {
        let profile = PROFILES[args[3].parse::<usize>().unwrap()];
        publish(&args[2], profile, args[4].parse().unwrap());
        return;
    }

    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let rounds: usize = opt("--rounds").map_or(3, |v| v.parse().unwrap());
    let scale: f64 = opt("--scale").map_or(1.0, |v| v.parse().unwrap());
    let only = opt("--only").cloned();
    let subs: Vec<Sub> = SUBS
        .iter()
        .copied()
        .filter(|s| only.as_deref().is_none_or(|o| o == s.name()))
        .collect();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    for decode in [false, true] {
        println!(
            "\n## {}",
            if decode {
                "receive + decode_batch"
            } else {
                "receive only"
            }
        );
        println!(
            "{:<22} {:>9} {:<16} {:>12} {:>9} {:>10}",
            "batch", "payload", "subscriber", "frames/s", "MB/s", "CPU us/f"
        );
        for (index, profile) in PROFILES.iter().enumerate() {
            let payload_len = payloads(*profile)[0].len();
            let frames = ((profile.frames as f64 * scale) as u64).max(1000);
            let mut results: Vec<(Sub, Vec<Stats>)> =
                subs.iter().map(|s| (*s, Vec::new())).collect();
            for round in 0..rounds {
                // Rotate the order so drift does not favor one subscriber.
                for offset in 0..subs.len() {
                    let k = (offset + round) % subs.len();
                    let stats = run(&rt, subs[k], index, frames, decode);
                    results[k].1.push(stats);
                }
            }
            for (sub, mut stats) in results {
                stats.sort_by(|a, b| a.rate.total_cmp(&b.rate));
                let median = &stats[stats.len() / 2];
                println!(
                    "{:<22} {:>7} B {:<16} {:>12.0} {:>9.1} {:>10.2}",
                    profile.name,
                    payload_len,
                    sub.name(),
                    median.rate,
                    median.rate * payload_len as f64 / 1e6,
                    median.cpu_us_per_frame,
                );
            }
        }
    }
}

struct Stats {
    rate: f64,
    cpu_us_per_frame: f64,
}

fn run(rt: &tokio::runtime::Runtime, sub: Sub, profile: usize, frames: u64, decode: bool) -> Stats {
    let port = portpicker::pick_unused_port().expect("free port");
    let endpoint = format!("tcp://127.0.0.1:{port}");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "publish",
            &endpoint,
            &profile.to_string(),
            &frames.to_string(),
        ])
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn publisher");
    let mut ready = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut ready)
        .unwrap();
    assert_eq!(ready.trim(), "ready");

    let meter = Meter::new(frames, decode);
    let stats = match sub {
        Sub::LibzmqBlocking => libzmq_blocking(&endpoint, meter),
        Sub::Libzmq => rt
            .block_on(rt.spawn(libzmq(endpoint, meter, true)))
            .unwrap(),
        Sub::LibzmqCoop => rt
            .block_on(rt.spawn(libzmq(endpoint, meter, false)))
            .unwrap(),
        Sub::Omq => rt.block_on(rt.spawn(omq(endpoint, meter, true))).unwrap(),
        Sub::OmqCoop => rt.block_on(rt.spawn(omq(endpoint, meter, false))).unwrap(),
        Sub::Zeromq => rt.block_on(rt.spawn(zeromq(endpoint, meter))).unwrap(),
    };
    assert!(child.wait().unwrap().success(), "publisher failed");
    stats
}

/// Counts frames, checks topic and sequence, and times all but the warmup.
struct Meter {
    total: u64,
    warmup: u64,
    next: u64,
    decode: bool,
    start: Option<(Instant, Duration)>,
}

impl Meter {
    fn new(total: u64, decode: bool) -> Self {
        Self {
            total,
            warmup: total / 10,
            next: 0,
            decode,
            start: None,
        }
    }

    /// Returns whether the last frame arrived.
    fn frame(&mut self, parts: usize, topic: &[u8], seq: &[u8], payload: &[u8]) -> bool {
        assert_eq!(parts, 3, "live frames are [topic, seq, payload]");
        assert_eq!(topic, TOPIC);
        let seq = u64::from_be_bytes(seq.try_into().expect("8-byte sequence"));
        assert_eq!(seq, self.next, "sequence gap");
        self.next += 1;
        if self.decode {
            std::hint::black_box(decode_batch(payload).expect("valid batch"));
        }
        if seq == self.warmup {
            self.start = Some((Instant::now(), cpu_time()));
        }
        self.next == self.total
    }

    fn finish(self) -> Stats {
        let (start, cpu) = self.start.expect("warmup finished");
        let frames = (self.total - self.warmup - 1) as f64;
        let elapsed = start.elapsed().as_secs_f64();
        let cpu = (cpu_time() - cpu).as_secs_f64();
        Stats {
            rate: frames / elapsed,
            cpu_us_per_frame: cpu * 1e6 / frames,
        }
    }
}

/// User and system time of this process, which runs only the subscriber.
fn cpu_time() -> Duration {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage fills the struct for RUSAGE_SELF.
    let usage = unsafe {
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()), 0);
        usage.assume_init()
    };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

fn libzmq_blocking(endpoint: &str, mut meter: Meter) -> Stats {
    let ctx = zmq::Context::new();
    let sub = libzmq_sub(&ctx, endpoint);
    loop {
        let frames = sub.recv_multipart(0).unwrap();
        if meter.frame(frames.len(), &frames[0], &frames[1], &frames[2]) {
            return meter.finish();
        }
    }
}

fn libzmq_sub(ctx: &zmq::Context, endpoint: &str) -> zmq::Socket {
    // As `ConnectionMonitor::fresh_sub`, without the monitor.
    let sub = ctx.socket(zmq::SUB).unwrap();
    sub.set_maxmsgsize(MAX_EVENT_BYTES as i64).unwrap();
    sub.set_linger(0).unwrap();
    sub.set_subscribe(TOPIC).unwrap();
    sub.connect(endpoint).unwrap();
    sub
}

/// The router's current path: `ReadableSocket` over `ZMQ_FD`.
async fn libzmq(endpoint: String, mut meter: Meter, yield_each: bool) -> Stats {
    let ctx = zmq::Context::new();
    // As `ReadableSocket::new`, which tokio 1.53.3 deprecates the same way.
    #[allow(deprecated)]
    let mut sub = AsyncFd::with_interest(libzmq_sub(&ctx, &endpoint), Interest::READABLE).unwrap();
    loop {
        match sub.get_ref().recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => {
                if meter.frame(frames.len(), &frames[0], &frames[1], &frames[2]) {
                    return meter.finish();
                }
                if yield_each {
                    tokio::task::yield_now().await;
                } else {
                    tokio::task::coop::consume_budget().await;
                }
            }
            Err(zmq::Error::EAGAIN) => {
                tokio::select! {
                    result = readable(&mut sub) => result.unwrap(),
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {}
                }
            }
            Err(error) => panic!("recv: {error}"),
        }
    }
}

/// `ReadableSocket::wait`: ZMQ_FD is edge-triggered, ZMQ_EVENTS decides.
async fn readable(socket: &mut AsyncFd<zmq::Socket>) -> std::io::Result<()> {
    loop {
        if socket
            .get_ref()
            .get_events()
            .map_err(std::io::Error::other)?
            .contains(zmq::POLLIN)
        {
            return Ok(());
        }
        let mut ready = socket.readable_mut().await?;
        if ready
            .get_inner()
            .get_events()
            .map_err(std::io::Error::other)?
            .contains(zmq::POLLIN)
        {
            return Ok(());
        }
        ready.clear_ready();
    }
}

async fn omq(endpoint: String, mut meter: Meter, yield_each: bool) -> Stats {
    use omq_tokio::{Endpoint, Options, Socket, SocketType};

    let sub = Socket::new(
        SocketType::Sub,
        Options::default().max_message_size(MAX_EVENT_BYTES),
    );
    sub.subscribe(Bytes::from_static(TOPIC)).await.unwrap();
    sub.connect(endpoint.parse::<Endpoint>().unwrap())
        .await
        .unwrap();
    loop {
        let msg = sub.recv().await.unwrap();
        if meter.frame(msg.len(), &msg[0], &msg[1], &msg[2]) {
            return meter.finish();
        }
        if yield_each {
            tokio::task::yield_now().await;
        }
    }
}

async fn zeromq(endpoint: String, mut meter: Meter) -> Stats {
    use zeromq::{Socket, SocketRecv, SubSocket};

    let mut sub = SubSocket::new();
    sub.subscribe(std::str::from_utf8(TOPIC).unwrap())
        .await
        .unwrap();
    sub.connect(&endpoint).await.unwrap();
    loop {
        let msg = sub.recv().await.unwrap();
        let part = |i: usize| msg.get(i).map_or(&[][..], |b| &b[..]);
        if meter.frame(msg.len(), part(0), part(1), part(2)) {
            return meter.finish();
        }
        tokio::task::yield_now().await;
    }
}

/// vLLM-shaped `KVEventBatch` payloads with u64 block hashes.
fn payloads(profile: Profile) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(607);
    (0..DISTINCT_PAYLOADS)
        .map(|_| {
            let events = (0..profile.events)
                .map(|_| block_stored(&mut rng, profile.blocks))
                .collect();
            let batch = Value::Array(vec![Value::F64(1.7e9), Value::Array(events)]);
            let mut buf = Vec::new();
            rmpv::encode::write_value(&mut buf, &batch).unwrap();
            decode_batch(&buf).expect("router accepts the payload");
            buf
        })
        .collect()
}

fn block_stored(rng: &mut StdRng, blocks: usize) -> Value {
    let hashes = (0..blocks)
        .map(|_| Value::from(rng.random::<u64>()))
        .collect();
    let tokens = (0..blocks * BLOCK_SIZE)
        .map(|_| Value::from(rng.random_range(0..150_000u32)))
        .collect();
    let field = |k: &str, v: Value| (Value::from(k), v);
    Value::Map(vec![
        field("type", Value::from("BlockStored")),
        field("block_hashes", Value::Array(hashes)),
        field("parent_block_hash", Value::from(rng.random::<u64>())),
        field("token_ids", Value::Array(tokens)),
        field("block_size", Value::from(BLOCK_SIZE as u64)),
        field("lora_id", Value::Nil),
        field("medium", Value::from("GPU")),
        field("lora_name", Value::Nil),
    ])
}

/// Publisher process: a libzmq XPUB that waits rather than drops.
fn publish(endpoint: &str, profile: Profile, frames: u64) {
    let payloads = payloads(profile);
    let ctx = zmq::Context::new();
    let mut xpub = ctx.socket(zmq::XPUB).unwrap();
    xpub.set_sndhwm(PUBLISHER_HWM).unwrap();
    let on: i32 = 1;
    // SAFETY: a live socket and an int option of the right size.
    let rc = unsafe {
        zmq_sys::zmq_setsockopt(
            xpub.as_mut_ptr(),
            zmq_sys::ZMQ_XPUB_NODROP as i32,
            (&on as *const i32).cast(),
            std::mem::size_of::<i32>(),
        )
    };
    assert_eq!(rc, 0, "ZMQ_XPUB_NODROP");
    xpub.bind(endpoint).unwrap();
    println!("ready");

    // Publish once the subscription arrived.
    let subscription = xpub.recv_bytes(0).unwrap();
    assert_eq!(subscription.first(), Some(&1));
    for seq in 0..frames {
        let payload = &payloads[seq as usize % DISTINCT_PAYLOADS];
        xpub.send_multipart([TOPIC, &seq.to_be_bytes()[..], payload], 0)
            .unwrap();
    }
    // The default linger delivers what is queued before the context ends.
}
