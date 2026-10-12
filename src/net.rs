//! Wire format and the receive loop.

use crate::Shared;
use crate::audio::{Jitter, Resampler};
use crate::viz::{FFT_N, Tap};
use std::net::{IpAddr, UdpSocket};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

pub const PORT: u16 = 47000;
pub const VIDEO: u8 = 0;
pub const AUDIO: u8 = 1;
pub const CHUNK: usize = 1200;
pub const MAX_CHUNKS: u16 = 256;
pub const RATE: u32 = 16_000; // wire audio: mono i16 LE
pub const AUDIO_PKT: usize = 320; // 20 ms at 16 kHz

pub type Frame = Arc<Mutex<Option<Arc<[u8]>>>>; // latest received JPEG

pub enum Packet<'a> {
    Video { id: u32, idx: u16, count: u16, data: &'a [u8] },
    Audio { seq: u32, pcm: &'a [u8] },
}

/// Parses an untrusted datagram; returns None for anything malformed.
pub fn parse(b: &[u8]) -> Option<Packet<'_>> {
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

pub fn video_packets(id: u32, jpeg: &[u8]) -> Vec<Vec<u8>> {
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
pub fn rand_u32() -> u32 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos() ^ d.as_secs() as u32)
}

/// True if `new` is a little behind `last` (stale/reordered). Big jumps back mean the peer restarted.
pub fn is_stale(new: u32, last: u32) -> bool {
    let d = new.wrapping_sub(last) as i32;
    d < 0 && d > -100
}

/// Audio packets missing between two accepted sequence numbers; big jumps (peer restart) count as none.
pub fn lost_between(last: u32, seq: u32) -> u64 {
    match seq.wrapping_sub(last) {
        d @ 1..100 => (d - 1) as u64,
        _ => 0,
    }
}

/// Counters the TUI samples once a second to show rates.
#[derive(Default)]
pub struct Stats {
    pub tx_bytes: AtomicU64,
    pub rx_bytes: AtomicU64,
    pub rx_frames: AtomicU64,
    pub audio_rx: AtomicU64,
    pub audio_lost: AtomicU64,
    pub buffer_ms: AtomicU32,
    pub last_rx_ms: AtomicU64, // since Shared::start, 0 = never
}

#[derive(Default)]
pub struct Reassembler {
    id: u32,
    parts: Vec<Option<Vec<u8>>>,
    have: usize,
    done: bool,
}

impl Reassembler {
    pub fn push(&mut self, id: u32, idx: u16, count: u16, data: &[u8]) -> Option<Vec<u8>> {
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

/// `frame` is None in audio mode: video from the peer is then dropped unassembled.
pub fn recv_loop(sock: UdpSocket, peer: IpAddr, frame: Option<Frame>, jitter: Jitter, tap: Tap, sh: Arc<Shared>) {
    let mut buf = [0u8; 2048];
    let mut re = Reassembler::default();
    let mut out_rate = 0;
    let mut rs = Resampler::new(RATE, RATE);
    let mut max_buf = 0;
    let mut last_seq: Option<u32> = None;
    let st = &sh.stats;
    let _ = sock.set_read_timeout(Some(Duration::from_millis(200))); // so a hang-up ends this thread
    while !sh.quit.load(Relaxed) {
        let Ok((n, src)) = sock.recv_from(&mut buf) else {
            continue;
        };
        if src.ip() != peer {
            continue;
        }
        st.rx_bytes.fetch_add(n as u64, Relaxed);
        st.last_rx_ms.store(sh.start.elapsed().as_millis().max(1) as u64, Relaxed);
        match parse(&buf[..n]) {
            Some(Packet::Video { id, idx, count, data }) => {
                if let Some(f) = &frame
                    && let Some(jpeg) = re.push(id, idx, count, data)
                {
                    st.rx_frames.fetch_add(1, Relaxed);
                    *f.lock().unwrap() = Some(jpeg.into());
                }
            }
            Some(Packet::Audio { seq, pcm }) => {
                if last_seq.is_some_and(|l| l == seq || is_stale(seq, l)) {
                    continue;
                }
                st.audio_rx.fetch_add(1, Relaxed);
                st.audio_lost.fetch_add(last_seq.map_or(0, |l| lost_between(l, seq)), Relaxed);
                last_seq = Some(seq);
                let samples = pcm.chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0);
                {
                    // never hold jitter and tap together: the speaker callback must not wait on the visualizer
                    let mut q = jitter.lock().unwrap();
                    let r = sh.out_rate.load(Relaxed);
                    if r != out_rate {
                        // speaker switched to a device with another rate: queued samples are at the old rate
                        (out_rate, rs, max_buf) = (r, Resampler::new(RATE, r), r as usize / 8); // cap latency ~125 ms
                        q.clear();
                    }
                    samples.clone().for_each(|x| rs.push(x, |v| q.push_back(v)));
                    let excess = q.len().saturating_sub(max_buf);
                    q.drain(..excess);
                    st.buffer_ms.store((q.len() as u64 * 1000 / out_rate.max(1) as u64) as u32, Relaxed);
                }
                let mut t = tap.lock().unwrap();
                t.0.extend(samples);
                let excess = t.0.len().saturating_sub(FFT_N);
                t.0.drain(..excess);
                t.1 = Some(Instant::now());
            }
            None => {}
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
    fn loss_from_seq_gaps() {
        assert_eq!(lost_between(5, 6), 0);
        assert_eq!(lost_between(5, 9), 3);
        assert_eq!(lost_between(u32::MAX, 1), 1); // wraps
        assert_eq!(lost_between(5, 5_000), 0); // peer restarted
        assert_eq!(lost_between(5, 4), 0);
    }
}
