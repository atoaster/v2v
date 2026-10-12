use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use image::{ImageFormat, ImageReader, Limits, RgbImage};
use minifb::{Key, ScaleMode, Window, WindowOptions};
use std::collections::VecDeque;
use std::io::{Cursor, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const PORT: u16 = 47000;
const VIDEO: u8 = 0;
const AUDIO: u8 = 1;
const CHUNK: usize = 1200;
const MAX_CHUNKS: u16 = 256;
const RATE: u32 = 16_000; // wire audio: mono i16 LE
const AUDIO_PKT: usize = 320; // 20 ms at 16 kHz

type Frame = Arc<Mutex<Option<Arc<[u8]>>>>; // latest received JPEG
type Jitter = Arc<Mutex<VecDeque<f32>>>;

enum Packet<'a> {
    Video { id: u32, idx: u16, count: u16, data: &'a [u8] },
    Audio { seq: u32, pcm: &'a [u8] },
}

/// Parses an untrusted datagram; returns None for anything malformed.
fn parse(b: &[u8]) -> Option<Packet<'_>> {
    match *b.first()? {
        VIDEO if b.len() > 9 => {
            let id = u32::from_be_bytes(b[1..5].try_into().ok()?);
            let idx = u16::from_be_bytes([b[5], b[6]]);
            let count = u16::from_be_bytes([b[7], b[8]]);
            (count > 0 && count <= MAX_CHUNKS && idx < count && b.len() - 9 <= CHUNK).then(|| Packet::Video {
                id,
                idx,
                count,
                data: &b[9..],
            })
        }
        AUDIO if b.len() > 5 && b.len() % 2 == 1 => {
            let seq = u32::from_be_bytes(b[1..5].try_into().ok()?);
            Some(Packet::Audio { seq, pcm: &b[5..] })
        }
        _ => None,
    }
}

fn video_packets(id: u32, jpeg: &[u8]) -> Vec<Vec<u8>> {
    let count = jpeg.len().div_ceil(CHUNK) as u16;
    jpeg.chunks(CHUNK)
        .enumerate()
        .map(|(i, c)| {
            let mut p = vec![VIDEO];
            p.extend(id.to_be_bytes());
            p.extend((i as u16).to_be_bytes());
            p.extend(count.to_be_bytes());
            p.extend(c);
            p
        })
        .collect()
}

/// Random-ish start for frame ids / audio seq, so a quick restart does not land in the peer's stale window.
fn rand_u32() -> u32 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos() ^ d.as_secs() as u32)
}

/// True if `new` is a little behind `last` (stale/reordered). Big jumps back mean the peer restarted.
fn is_stale(new: u32, last: u32) -> bool {
    let d = new.wrapping_sub(last) as i32;
    d < 0 && d > -100
}

#[derive(Default)]
struct Reassembler {
    id: u32,
    parts: Vec<Option<Vec<u8>>>,
    have: usize,
    done: bool,
}

impl Reassembler {
    fn push(&mut self, id: u32, idx: u16, count: u16, data: &[u8]) -> Option<Vec<u8>> {
        if self.parts.is_empty() || id != self.id {
            if !self.parts.is_empty() && is_stale(id, self.id) {
                return None;
            }
            // newer frame: drop whatever was incomplete
            *self = Reassembler { id, parts: vec![None; count as usize], have: 0, done: false };
        }
        if self.done || self.parts.len() != count as usize {
            return None;
        }
        let slot = self.parts.get_mut(idx as usize)?;
        if slot.is_none() {
            *slot = Some(data.to_vec());
            self.have += 1;
        }
        if self.have == self.parts.len() {
            self.done = true; // ignore late duplicates of this frame
            return Some(self.parts.iter().flatten().flatten().copied().collect());
        }
        None
    }
}

/// Linear resampler, push style. shortcut: no low-pass filter, so downsampling aliases a bit; fine for voice.
struct Resampler {
    step: f64,
    pos: f64,
    prev: f32,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        Resampler { step: from as f64 / to as f64, pos: 0.0, prev: 0.0 }
    }
    fn push(&mut self, x: f32, mut out: impl FnMut(f32)) {
        while self.pos <= 1.0 {
            out(self.prev + (x - self.prev) * self.pos as f32);
            self.pos += self.step;
        }
        self.pos -= 1.0;
        self.prev = x;
    }
}

fn decode(jpeg: &[u8]) -> Option<(usize, usize, Vec<u32>)> {
    let mut r = ImageReader::with_format(Cursor::new(jpeg), ImageFormat::Jpeg);
    let mut lim = Limits::default();
    lim.max_image_width = Some(1920);
    lim.max_image_height = Some(1920);
    r.limits(lim);
    let img = r.decode().ok()?.to_rgb8();
    let px = img.pixels().map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32);
    Some((img.width() as usize, img.height() as usize, px.collect()))
}

fn recv_loop(sock: UdpSocket, peer: IpAddr, frame: Frame, jitter: Jitter, out_rate: u32) {
    let mut buf = [0u8; 2048];
    let mut re = Reassembler::default();
    let mut rs = Resampler::new(RATE, out_rate);
    let mut last_seq: Option<u32> = None;
    let max_buf = out_rate as usize / 8; // cap playback latency at ~125 ms so it recovers after bursts
    loop {
        let Ok((n, src)) = sock.recv_from(&mut buf) else {
            continue;
        };
        if src.ip() != peer {
            continue;
        }
        match parse(&buf[..n]) {
            Some(Packet::Video { id, idx, count, data }) => {
                if let Some(jpeg) = re.push(id, idx, count, data) {
                    *frame.lock().unwrap() = Some(jpeg.into());
                }
            }
            Some(Packet::Audio { seq, pcm }) => {
                if last_seq.is_some_and(|l| l == seq || is_stale(seq, l)) {
                    continue;
                }
                last_seq = Some(seq);
                let mut q = jitter.lock().unwrap();
                for s in pcm.chunks_exact(2) {
                    rs.push(i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0, |v| q.push_back(v));
                }
                let excess = q.len().saturating_sub(max_buf);
                q.drain(..excess);
            }
            None => {}
        }
    }
}

fn camera_loop(sock: UdpSocket, dest: SocketAddr) -> Result<(), String> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
    let want = CameraFormat::new_from(640, 480, FrameFormat::MJPEG, 30);
    let open = |t| nokhwa::Camera::new(CameraIndex::Index(0), RequestedFormat::new::<RgbFormat>(t));
    let mut cam = open(RequestedFormatType::Closest(want))
        .or_else(|_| open(RequestedFormatType::AbsoluteHighestFrameRate))
        .map_err(|e| e.to_string())?;
    cam.open_stream().map_err(|e| e.to_string())?;
    let (mut id, mut fails, mut last) = (rand_u32(), 0, Instant::now() - Duration::from_secs(1));
    loop {
        // skip bad/truncated frames; give up only if the camera keeps failing
        let mut img: RgbImage = match cam.frame().and_then(|f| f.decode_image::<RgbFormat>()) {
            Ok(img) => img,
            Err(e) if fails >= 100 => return Err(format!("camera stopped: {e}")),
            Err(_) => {
                fails += 1;
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        fails = 0;
        // shortcut: fixed ~15 fps cap (~1-3 Mbit/s), no congestion control; add adaptive bitrate if calls run over slow links
        if last.elapsed() < Duration::from_millis(66) {
            continue;
        }
        last = Instant::now();
        if img.width() > 640 || img.height() > 480 {
            let k = f64::min(640.0 / img.width() as f64, 480.0 / img.height() as f64);
            img = image::imageops::thumbnail(&img, (img.width() as f64 * k) as u32, (img.height() as f64 * k) as u32);
        }
        let mut jpeg = Vec::new();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 60);
        if img.write_with_encoder(enc).is_err() || jpeg.len() > CHUNK * MAX_CHUNKS as usize {
            continue;
        }
        for p in video_packets(id, &jpeg) {
            let _ = sock.send_to(&p, dest);
        }
        id = id.wrapping_add(1);
    }
}

fn mic_stream(sock: UdpSocket, dest: SocketAddr) -> Result<cpal::Stream, String> {
    let dev = cpal::default_host().default_input_device().ok_or("no input device")?;
    let cfg = dev.default_input_config().map_err(|e| e.to_string())?;
    let (ch, rate) = (cfg.channels() as usize, cfg.sample_rate());
    let mut rs = Resampler::new(rate, RATE);
    let mut pending: Vec<i16> = Vec::new();
    let mut seq = rand_u32();
    // send from a plain thread so the real-time callback never blocks in send_to
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(50);
    thread::spawn(move || {
        for p in rx {
            let _ = sock.send_to(&p, dest);
        }
    });
    let on_data = move |mono: &mut dyn Iterator<Item = f32>| {
        for x in mono {
            rs.push(x, |v| pending.push((v.clamp(-1.0, 1.0) * 32767.0) as i16));
        }
        while pending.len() >= AUDIO_PKT {
            let mut p = vec![AUDIO];
            p.extend(seq.to_be_bytes());
            p.extend(pending.drain(..AUDIO_PKT).flat_map(i16::to_le_bytes));
            let _ = tx.try_send(p);
            seq = seq.wrapping_add(1);
        }
    };
    fn build<T: SizedSample>(
        dev: &cpal::Device,
        cfg: cpal::StreamConfig,
        ch: usize,
        mut f: impl FnMut(&mut dyn Iterator<Item = f32>) + Send + 'static,
    ) -> Result<cpal::Stream, cpal::Error>
    where
        f32: FromSample<T>,
    {
        dev.build_input_stream(
            cfg,
            move |d: &[T], _: &_| f(&mut d.chunks(ch).map(|fr| fr.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / ch as f32)),
            |e| eprintln!("v2v: mic error: {e}"),
            None,
        )
    }
    let s = match cfg.sample_format() {
        SampleFormat::F32 => build::<f32>(&dev, cfg.config(), ch, on_data),
        SampleFormat::I16 => build::<i16>(&dev, cfg.config(), ch, on_data),
        SampleFormat::U16 => build::<u16>(&dev, cfg.config(), ch, on_data),
        SampleFormat::I32 => build::<i32>(&dev, cfg.config(), ch, on_data),
        SampleFormat::I24 => build::<cpal::I24>(&dev, cfg.config(), ch, on_data),
        SampleFormat::F64 => build::<f64>(&dev, cfg.config(), ch, on_data),
        f => return Err(format!("unsupported mic sample format {f}")),
    }
    .map_err(|e| e.to_string())?;
    s.play().map_err(|e| e.to_string())?;
    Ok(s)
}

fn speaker_stream(jitter: Jitter) -> Result<(cpal::Stream, u32), String> {
    let dev = cpal::default_host().default_output_device().ok_or("no output device")?;
    let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
    let (ch, rate) = (cfg.channels() as usize, cfg.sample_rate());
    fn build<T: SizedSample + FromSample<f32>>(
        dev: &cpal::Device,
        cfg: cpal::StreamConfig,
        ch: usize,
        jitter: Jitter,
    ) -> Result<cpal::Stream, cpal::Error> {
        let prefill = cfg.sample_rate as usize * 3 / 50; // ~60 ms
        let mut playing = false;
        dev.build_output_stream(
            cfg,
            move |out: &mut [T], _: &_| {
                let mut q = jitter.lock().unwrap();
                playing |= q.len() >= prefill; // wait for a cushion before (re)starting playback
                for fr in out.chunks_mut(ch) {
                    let v = if playing { q.pop_front() } else { None };
                    playing &= v.is_some(); // underrun: rebuild the cushion
                    fr.fill(T::from_sample(v.unwrap_or(0.0)));
                }
            },
            |e| eprintln!("v2v: speaker error: {e}"),
            None,
        )
    }
    let s = match cfg.sample_format() {
        SampleFormat::F32 => build::<f32>(&dev, cfg.config(), ch, jitter),
        SampleFormat::I16 => build::<i16>(&dev, cfg.config(), ch, jitter),
        SampleFormat::U16 => build::<u16>(&dev, cfg.config(), ch, jitter),
        SampleFormat::I32 => build::<i32>(&dev, cfg.config(), ch, jitter),
        SampleFormat::I24 => build::<cpal::I24>(&dev, cfg.config(), ch, jitter),
        SampleFormat::F64 => build::<f64>(&dev, cfg.config(), ch, jitter),
        f => return Err(format!("unsupported speaker sample format {f}")),
    }
    .map_err(|e| e.to_string())?;
    s.play().map_err(|e| e.to_string())?;
    Ok((s, rate))
}

/// Serves `frame` as an MJPEG HTTP stream (what tplay/ffmpeg reads). Nothing is sent until the first
/// frame arrives: ffmpeg cannot follow a size change, so no placeholder frame; readers just wait.
fn serve_mjpeg(mut s: TcpStream, frame: Frame, path: &str) {
    let mut req = [0; 1024];
    let n = s.read(&mut req).unwrap_or(0);
    // only tplay knows the random path, so other local users cannot watch the call
    if !req[..n].starts_with(format!("GET {path} ").as_bytes()) {
        return;
    }
    let _ = s.set_nodelay(true); // header and JPEG are separate writes; do not let Nagle hold them
    let mut ok = s.write_all(b"HTTP/1.0 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=v2v\r\n\r\n").is_ok();
    let mut cur: Option<Arc<[u8]>> = None;
    while ok {
        let new = frame.lock().unwrap().clone().filter(|j| cur.as_ref().is_none_or(|c| !Arc::ptr_eq(j, c)));
        let Some(jpeg) = new else {
            thread::sleep(Duration::from_millis(5));
            continue;
        };
        let head = format!("--v2v\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpeg.len());
        ok = s.write_all(head.as_bytes()).and_then(|_| s.write_all(&jpeg)).and_then(|_| s.write_all(b"\r\n")).is_ok();
        cur = Some(jpeg);
    }
}

/// Plays the received video in the terminal through tplay; returns when tplay exits.
fn run_tplay(frame: Frame, peer: IpAddr) {
    let fail = |e: String| -> ! {
        eprintln!("v2v: {e}");
        std::process::exit(1);
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| fail(format!("cannot listen for tplay: {e}")));
    let path: Arc<str> =
        format!("/{:016x}", std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 0)).into();
    let url = format!("http://{}{path}", listener.local_addr().unwrap_or_else(|e| fail(e.to_string())));
    // tplay probes the URL with ffprobe twice and then opens it, so every connection gets its own stream
    thread::spawn(move || {
        for s in listener.incoming().flatten() {
            let (f, p) = (frame.clone(), path.clone());
            thread::spawn(move || serve_mjpeg(s, f, &p));
        }
    });
    let mut child = std::process::Command::new("tplay")
        // read frames as fast as they arrive, so a pause or stall does not leave a backlog behind live
        .args(["--fps", "100", &url])
        .spawn()
        .unwrap_or_else(|e| match e.kind() {
            std::io::ErrorKind::NotFound => fail("tplay not found on PATH (install it with: cargo install tplay)".into()),
            _ => fail(format!("cannot start tplay: {e}")),
        });
    eprintln!("v2v: waiting for video from {peer} (Ctrl-C to quit)...");
    match child.wait() {
        Ok(st) if !st.success() => std::process::exit(st.code().unwrap_or(1)),
        Err(e) => fail(format!("tplay: {e}")),
        _ => {}
    }
}

/// `v2v [--tplay] <ip>`, flag before or after the address.
fn parse_args(args: &[String]) -> Option<(IpAddr, bool)> {
    let rest: Vec<&String> = args.iter().filter(|a| *a != "--tplay").collect();
    match (rest.as_slice(), args.len() - rest.len()) {
        ([ip], n @ (0 | 1)) => Some((ip.parse().ok()?, n == 1)),
        _ => None,
    }
}

/// Usage plus the reply command the peer runs for each of our addresses.
fn help() -> String {
    let mut s = String::from("usage: v2v [--tplay] <ip_address>\n\nThe other side responds with one of:\n");
    let ifs = if_addrs::get_if_addrs().unwrap_or_default();
    // link-local v6 would need a %scope suffix that differs per peer, so skip it
    for i in ifs.iter().filter(|i| !i.is_loopback() && !i.is_link_local()) {
        s += &format!("  v2v {:<39} # {}\n", i.ip(), i.name);
    }
    s + "\n--tplay  render the video in the terminal via tplay instead of a window\n"
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", help());
        return;
    }
    let Some((peer, tplay)) = parse_args(&args) else {
        eprintln!("usage: v2v [--tplay] <ip_address>");
        std::process::exit(2);
    };
    let any: IpAddr = if peer.is_ipv4() { Ipv4Addr::UNSPECIFIED.into() } else { Ipv6Addr::UNSPECIFIED.into() };
    let sock = UdpSocket::bind((any, PORT)).unwrap_or_else(|e| {
        eprintln!("v2v: cannot bind UDP port {PORT}: {e}");
        std::process::exit(1);
    });
    let dest = SocketAddr::new(peer, PORT);
    let clone = || sock.try_clone().expect("clone UDP socket");

    let frame: Frame = Arc::default();
    let jitter: Jitter = Arc::default();
    let (_speaker, out_rate) = match speaker_stream(jitter.clone()) {
        Ok((s, r)) => (Some(s), r),
        Err(e) => {
            eprintln!("v2v: warning: no audio output ({e})");
            (None, RATE)
        }
    };
    let _mic = mic_stream(clone(), dest).map_err(|e| eprintln!("v2v: warning: no microphone ({e})")).ok();
    let cam_sock = clone();
    thread::spawn(move || {
        if let Err(e) = camera_loop(cam_sock, dest) {
            eprintln!("v2v: warning: no camera ({e})");
        }
    });
    let (rsock, rframe) = (clone(), frame.clone());
    thread::spawn(move || recv_loop(rsock, peer, rframe, jitter, out_rate));
    if tplay {
        return run_tplay(frame, peer);
    }

    let opts = WindowOptions { resize: true, scale_mode: ScaleMode::AspectRatioStretch, ..Default::default() };
    let mut win = Window::new(&format!("v2v - {peer}"), 640, 480, opts).unwrap_or_else(|e| {
        eprintln!("v2v: cannot open window: {e}");
        std::process::exit(1);
    });
    win.set_target_fps(30);
    let mut shown = (320, 240, vec![0x404040u32; 320 * 240]); // gray "waiting" screen
    while win.is_open() && !win.is_key_down(Key::Escape) {
        let jpeg = frame.lock().unwrap().take(); // decode outside the lock
        if let Some(f) = jpeg.and_then(|j| decode(&j)) {
            shown = f;
        }
        if let Err(e) = win.update_with_buffer(&shown.2, shown.0, shown.1) {
            eprintln!("v2v: window error: {e}");
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_roundtrip() {
        let jpeg: Vec<u8> = (0..5000u32).map(|i| i as u8).collect();
        let mut re = Reassembler::default();
        let mut pkts = video_packets(7, &jpeg);
        pkts.reverse(); // out-of-order delivery
        let mut done = None;
        for p in &pkts {
            let Some(Packet::Video { id, idx, count, data }) = parse(p) else { panic!("bad packet") };
            assert!(done.is_none());
            done = re.push(id, idx, count, data);
        }
        assert_eq!(done.as_deref(), Some(&jpeg[..]));
        // a late duplicate and an older frame are ignored
        let Some(Packet::Video { id, idx, count, data }) = parse(&pkts[0]) else { panic!() };
        assert!(re.push(id, idx, count, data).is_none());
        let old = video_packets(6, &[1, 2, 3]);
        let Some(Packet::Video { id, idx, count, data }) = parse(&old[0]) else { panic!() };
        assert!(re.push(id, idx, count, data).is_none());
    }

    #[test]
    fn rejects_malformed() {
        let bad: [&[u8]; 7] = [
            &[],
            &[9, 0, 0, 0, 0, 0],
            &[VIDEO, 0, 0, 0, 1, 0, 0, 0, 1],    // no payload
            &[VIDEO, 0, 0, 0, 1, 0, 2, 0, 2, 5], // idx >= count
            &[VIDEO, 0, 0, 0, 1, 0, 0, 0, 0, 5], // count 0
            &[AUDIO, 0, 0, 0, 1, 7],             // odd pcm length
            &[AUDIO, 0, 0],                      // truncated
        ];
        for b in bad {
            assert!(parse(b).is_none(), "{b:?}");
        }
        // chunk count mismatch within a frame is rejected, not panicking
        let mut re = Reassembler::default();
        assert!(re.push(1, 0, 3, &[1]).is_none());
        assert!(re.push(1, 5, 9, &[1]).is_none());
    }

    #[test]
    fn args() {
        let p = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let ip: IpAddr = "10.0.0.2".parse().unwrap();
        assert_eq!(p(&["10.0.0.2"]), Some((ip, false)));
        assert_eq!(p(&["--tplay", "10.0.0.2"]), Some((ip, true)));
        assert_eq!(p(&["10.0.0.2", "--tplay"]), Some((ip, true)));
        for bad in [&[][..], &["--tplay"], &["x"], &["--tplay", "--tplay", "10.0.0.2"], &["1.2.3.4", "5.6.7.8"]] {
            assert_eq!(p(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn resampler_rate() {
        let mut n = 0;
        let mut r = Resampler::new(48_000, RATE);
        for _ in 0..48_000 {
            r.push(0.5, |_| n += 1);
        }
        assert!((15_990..=16_010).contains(&n), "{n}");
    }
}
