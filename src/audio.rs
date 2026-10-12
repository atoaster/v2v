//! Microphone and speaker streams, live device switching, test tone, device lists, Opus, playout.

use crate::Shared;
use crate::viz::{self, FFT_N, Tap};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use nnnoiseless::DenoiseState;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use unsafe_libopus as unsafe_opus;

pub type Jitter = Arc<Mutex<VecDeque<f32>>>;

/// Mic and Opus rate: 48 kHz mono, which is also what the denoiser produces.
pub const RATE: u32 = 48_000;
pub const PKT: usize = 960; // 20 ms
pub type Pcm = [i16; PKT];

/// Linear resampler, push style. shortcut: no low-pass filter, so downsampling aliases a bit; fine for voice.
pub struct Resampler {
    step: f64, // input samples per output sample
    pos: f64,
    prev: f32,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        Resampler { step: from as f64 / to as f64, pos: 0.0, prev: 0.0 }
    }
    pub fn push(&mut self, x: f32, mut out: impl FnMut(f32)) {
        while self.pos <= 1.0 {
            out(self.prev + (x - self.prev) * self.pos as f32);
            self.pos += self.step;
        }
        self.pos -= 1.0;
        self.prev = x;
    }
}

/// Noise suppression (RNNoise via nnnoiseless) on the mono mic signal. Upsamples the device rate to
/// 48 kHz and works in 10 ms frames. The model always runs, and "off" outputs the input delayed by the
/// same one frame the model adds, so toggling never shifts or drops samples and the model is warm.
pub struct Denoiser {
    st: Box<DenoiseState<'static>>,
    up: Resampler,
    inbuf: Vec<f32>, // 48 kHz, i16 scale, not yet a full frame
    prev: [f32; FRAME],
    out: Vec<f32>, // 48 kHz, ±1.0
}

const FRAME: usize = DenoiseState::FRAME_SIZE;
const DENOISE_RATE: u32 = RATE;

impl Denoiser {
    pub fn new(rate: u32) -> Self {
        let cap = 4 * FRAME;
        Denoiser {
            st: DenoiseState::new(),
            up: Resampler::new(rate, DENOISE_RATE),
            inbuf: Vec::with_capacity(cap),
            prev: [0.0; FRAME],
            out: Vec::with_capacity(cap),
        }
    }

    /// Takes device-rate samples, returns the 48 kHz samples ready so far (denoised when `on`).
    pub fn process(&mut self, block: &[f32], on: bool) -> &[f32] {
        let inbuf = &mut self.inbuf;
        for &x in block {
            self.up.push(x, |v| inbuf.push(v * 32768.0));
        }
        self.out.clear();
        let mut frame = [0.0; FRAME];
        let whole = self.inbuf.len() / FRAME * FRAME;
        for chunk in self.inbuf[..whole].chunks_exact(FRAME) {
            self.st.process_frame(&mut frame, chunk);
            let src = if on { &frame } else { &self.prev };
            self.out.extend(src.iter().map(|v| v / 32768.0));
            self.prev.copy_from_slice(chunk);
        }
        self.inbuf.drain(..whole);
        &self.out
    }
}

/// A running stream. Dropping it marks it dead first, so the "device disconnected" error
/// PipeWire reports for our own drop is not shown.
struct Live {
    alive: Arc<AtomicBool>,
    _stream: cpal::Stream,
}

impl Drop for Live {
    fn drop(&mut self) {
        self.alive.store(false, Relaxed);
    }
}

/// Owns the mic and speaker streams and rebuilds them when the selected device changes.
pub struct AudioIo {
    sh: Arc<Shared>,
    jitter: Jitter,
    pcm: SyncSender<Pcm>, // 20 ms mic frames to the call (or the "hear yourself" loopback)
    mic: (Option<String>, Option<Live>), // (device id last tried, stream)
    spk: (Option<String>, Option<Live>),
}

impl AudioIo {
    pub fn new(sh: Arc<Shared>, jitter: Jitter, pcm: SyncSender<Pcm>) -> Self {
        AudioIo { sh, jitter, pcm, mic: (None, None), spk: (None, None) }
    }

    /// Opens or reopens streams whose device setting changed. A failed device is not retried until it changes.
    pub fn sync(&mut self) {
        let (mic, spk) = {
            let s = self.sh.settings.lock().unwrap();
            (s.mic.clone(), s.speaker.clone())
        };
        if self.spk.0.as_ref() != Some(&spk) {
            self.spk.1 = None; // close first: some devices only allow one stream
            self.spk.1 = self.speaker(&spk).map_err(|e| self.sh.log(format!("no audio output ({e})"))).ok();
            self.spk.0 = Some(spk);
        }
        if self.mic.0.as_ref() != Some(&mic) {
            self.mic.1 = None;
            self.sh.mic_level.store(0, Relaxed); // the meter must not show the old device's level if this one fails
            self.mic.1 = self.mic(&mic).map_err(|e| self.sh.log(format!("no microphone ({e})"))).ok();
            self.mic.0 = Some(mic);
        }
    }

    /// Keeps the streams in line with the settings until hang-up.
    pub fn spawn(mut self) {
        thread::spawn(move || {
            while !self.sh.quit.load(Relaxed) {
                self.sync();
                thread::sleep(Duration::from_millis(100));
            }
        });
    }

    fn device(&self, id: &str, input: bool) -> Result<cpal::Device, String> {
        let host = cpal::default_host(); // fresh host: picks up devices plugged in since start
        if !id.is_empty() {
            if let Some(d) = id.parse::<cpal::DeviceId>().ok().and_then(|i| host.device_by_id(&i)) {
                return Ok(d);
            }
            self.sh.log(format!("device {id} not found, using the default"));
        }
        let d = if input { host.default_input_device() } else { host.default_output_device() };
        d.ok_or_else(|| format!("no {} device", if input { "input" } else { "output" }))
    }

    fn mic(&self, id: &str) -> Result<Live, String> {
        let dev = self.device(id, true)?;
        let cfg = dev.default_input_config().map_err(|e| e.to_string())?;
        let (ch, rate) = (cfg.channels() as usize, cfg.sample_rate());
        let mut dn = Denoiser::new(rate);
        let mut pending: Vec<i16> = Vec::with_capacity(4 * PKT);
        let mut block: Vec<f32> = Vec::new();
        let (sh, tx) = (self.sh.clone(), self.pcm.clone());
        let (mut muted, mut noise) = (false, false);
        let on_data = move |mono: &mut dyn Iterator<Item = f32>| {
            block.clear();
            block.extend(mono);
            if block.is_empty() {
                return;
            }
            // real-time thread: never wait for the menu's lock, keep the last values instead
            if let Ok(s) = sh.settings.try_lock() {
                (muted, noise) = (s.mute_mic, s.noise);
            }
            let block = dn.process(&block, noise);
            if block.is_empty() {
                return; // less than one 10 ms frame so far
            }
            let rms = (block.iter().map(|x| x * x).sum::<f32>() / block.len() as f32).sqrt();
            sh.mic_level.store(rms.to_bits(), Relaxed);
            // muted: send nothing (no RTP seq used), so the peer hears silence and counts no loss
            if muted || (sh.offline && !sh.loopback.load(Relaxed)) {
                pending.clear();
                return;
            }
            pending.extend(block.iter().map(|v| (v.clamp(-1.0, 1.0) * 32767.0) as i16));
            while pending.len() >= PKT {
                let mut p = [0i16; PKT]; // on the stack: no allocation in the real-time callback
                p.copy_from_slice(&pending[..PKT]);
                pending.drain(..PKT);
                let _ = tx.try_send(p); // Opus encoding happens on the receiving thread
            }
        };
        fn build<T: SizedSample>(
            dev: &cpal::Device,
            cfg: cpal::StreamConfig,
            ch: usize,
            mut f: impl FnMut(&mut dyn Iterator<Item = f32>) + Send + 'static,
            err: impl FnMut(cpal::Error) + Send + 'static,
        ) -> Result<cpal::Stream, cpal::Error>
        where
            f32: FromSample<T>,
        {
            dev.build_input_stream(
                cfg,
                move |d: &[T], _: &_| f(&mut d.chunks(ch).map(|fr| fr.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / ch as f32)),
                err,
                None,
            )
        }
        let alive = Arc::new(AtomicBool::new(true));
        let err = self.on_error("mic", &alive);
        let c = cfg.config();
        let s = match cfg.sample_format() {
            SampleFormat::F32 => build::<f32>(&dev, c, ch, on_data, err),
            SampleFormat::I16 => build::<i16>(&dev, c, ch, on_data, err),
            SampleFormat::U16 => build::<u16>(&dev, c, ch, on_data, err),
            SampleFormat::I32 => build::<i32>(&dev, c, ch, on_data, err),
            SampleFormat::I24 => build::<cpal::I24>(&dev, c, ch, on_data, err),
            SampleFormat::F64 => build::<f64>(&dev, c, ch, on_data, err),
            f => return Err(format!("unsupported mic sample format {f}")),
        }
        .map_err(|e| e.to_string())?;
        s.play().map_err(|e| e.to_string())?;
        Ok(Live { alive, _stream: s })
    }

    fn speaker(&self, id: &str) -> Result<Live, String> {
        let dev = self.device(id, false)?;
        let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
        let (ch, rate) = (cfg.channels() as usize, cfg.sample_rate());
        fn build<T: SizedSample + FromSample<f32>>(
            dev: &cpal::Device,
            cfg: cpal::StreamConfig,
            ch: usize,
            jitter: Jitter,
            sh: Arc<Shared>,
            err: impl FnMut(cpal::Error) + Send + 'static,
        ) -> Result<cpal::Stream, cpal::Error> {
            let rate = cfg.sample_rate;
            let prefill = rate as usize * 3 / 50; // ~60 ms
            let mut playing = false;
            let mut phase = 0u32;
            let mut vol = 1.0;
            dev.build_output_stream(
                cfg,
                move |out: &mut [T], _: &_| {
                    if let Ok(s) = sh.settings.try_lock() {
                        vol = s.volume as f32 / 100.0; // real-time thread: never wait for the menu's lock
                    }
                    let start = sh.tone.load(Relaxed);
                    let mut tone = start;
                    let mut q = jitter.lock().unwrap();
                    playing |= q.len() >= prefill; // wait for a cushion before (re)starting playback
                    for fr in out.chunks_mut(ch) {
                        let v = if playing { q.pop_front() } else { None };
                        playing &= v.is_some(); // underrun: rebuild the cushion
                        let mut x = v.unwrap_or(0.0);
                        if tone > 0 {
                            tone -= 1;
                            phase = (phase + 1) % rate;
                            x += 0.2 * (phase as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin();
                        }
                        fr.fill(T::from_sample((x * vol).clamp(-1.0, 1.0)));
                    }
                    sh.tone.fetch_sub(start - tone, Relaxed); // not store: the TUI may have restarted the tone meanwhile
                    // here, not in Player::push: it must fall to 0 when the peer stops sending
                    sh.stats.buffer_ms.store((q.len() as u64 * 1000 / rate as u64) as u32, Relaxed);
                },
                err,
                None,
            )
        }
        let alive = Arc::new(AtomicBool::new(true));
        let err = self.on_error("speaker", &alive);
        let (c, j, sh) = (cfg.config(), self.jitter.clone(), self.sh.clone());
        let s = match cfg.sample_format() {
            SampleFormat::F32 => build::<f32>(&dev, c, ch, j, sh, err),
            SampleFormat::I16 => build::<i16>(&dev, c, ch, j, sh, err),
            SampleFormat::U16 => build::<u16>(&dev, c, ch, j, sh, err),
            SampleFormat::I32 => build::<i32>(&dev, c, ch, j, sh, err),
            SampleFormat::I24 => build::<cpal::I24>(&dev, c, ch, j, sh, err),
            SampleFormat::F64 => build::<f64>(&dev, c, ch, j, sh, err),
            f => return Err(format!("unsupported speaker sample format {f}")),
        }
        .map_err(|e| e.to_string())?;
        s.play().map_err(|e| e.to_string())?;
        self.sh.out_rate.store(rate, Relaxed);
        Ok(Live { alive, _stream: s })
    }

    fn on_error(&self, what: &'static str, alive: &Arc<AtomicBool>) -> impl FnMut(cpal::Error) + Send + 'static {
        let (sh, alive) = (self.sh.clone(), alive.clone());
        move |e| {
            if alive.load(Relaxed) {
                sh.log(format!("{what} error: {e}"));
            }
        }
    }
}

/// Received (or looped-back) 48 kHz audio into the speaker's jitter queue and the visualizer tap.
pub struct Player {
    sh: Arc<Shared>,
    jitter: Jitter,
    tap: Tap,
    out_rate: u32,
    rs: Resampler,
    tap_rs: Resampler,
    max_buf: usize,
    tap_buf: Vec<f32>,
    low: usize, // least queued before a push in this window
    pushes: u32,
}

impl Player {
    pub fn new(sh: Arc<Shared>, jitter: Jitter, tap: Tap) -> Self {
        let tap_rs = Resampler::new(RATE, viz::RATE);
        Player { sh, jitter, tap, out_rate: 0, rs: Resampler::new(RATE, RATE), tap_rs, max_buf: 0, tap_buf: Vec::new(), low: usize::MAX, pushes: 0 }
    }

    pub fn push(&mut self, pcm: &[i16]) {
        let samples = pcm.iter().map(|&s| s as f32 / 32768.0);
        {
            // never hold jitter and tap together: the speaker callback must not wait on the visualizer
            let mut q = self.jitter.lock().unwrap();
            let r = self.sh.out_rate.load(Relaxed);
            if r != self.out_rate {
                // speaker switched to a device with another rate: queued samples are at the old rate
                (self.out_rate, self.rs, self.max_buf) = (r, Resampler::new(RATE, r), r as usize / 8); // cap latency ~125 ms
                q.clear();
            }
            // drift control: audio the queue never needed during the last second is extra latency
            // (after a burst, or a sender clock running fast), so play it out 1% faster until it is gone
            self.low = self.low.min(q.len());
            self.pushes += 1;
            if self.pushes == 50 {
                let fast = self.low > self.out_rate as usize / 25; // 40 ms
                self.rs.step = RATE as f64 / self.out_rate as f64 * if fast { 1.01 } else { 1.0 };
                (self.low, self.pushes) = (usize::MAX, 0);
            }
            samples.clone().for_each(|x| self.rs.push(x, |v| q.push_back(v)));
            let excess = q.len().saturating_sub(self.max_buf);
            q.drain(..excess); // last resort; the drift control above normally keeps well below
        }
        let buf = &mut self.tap_buf;
        buf.clear();
        samples.for_each(|x| self.tap_rs.push(x, |v| buf.push(v)));
        let mut t = self.tap.lock().unwrap();
        t.0.extend(buf.iter());
        let excess = t.0.len().saturating_sub(FFT_N);
        t.0.drain(..excess);
        t.1 = Some(Instant::now());
    }
}

/// Opus encoder: 48 kHz mono VOIP, 32 kbit/s, in-band FEC for the peer's loss concealment.
pub struct OpusEnc(*mut unsafe_opus::OpusEncoder);
// SAFETY: the encoder state is plain heap memory owned by this value, used from one thread at a time.
unsafe impl Send for OpusEnc {}

impl OpusEnc {
    pub fn new() -> Result<Self, String> {
        let mut err = 0;
        // SAFETY: valid arguments; the result is checked and owned by Self (freed in Drop).
        unsafe {
            let e = unsafe_opus::opus_encoder_create(RATE as i32, 1, unsafe_opus::OPUS_APPLICATION_VOIP, &mut err);
            if err != unsafe_opus::OPUS_OK || e.is_null() {
                return Err(format!("opus encoder error {err}"));
            }
            unsafe_opus::opus_encoder_ctl!(e, unsafe_opus::OPUS_SET_BITRATE_REQUEST, 32_000);
            unsafe_opus::opus_encoder_ctl!(e, unsafe_opus::OPUS_SET_INBAND_FEC_REQUEST, 1);
            unsafe_opus::opus_encoder_ctl!(e, unsafe_opus::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 10);
            Ok(OpusEnc(e))
        }
    }

    /// One 20 ms frame into `out`; returns the packet length (None on an encoder error).
    pub fn encode(&mut self, pcm: &Pcm, out: &mut [u8]) -> Option<usize> {
        // SAFETY: pcm holds PKT samples, out.len() bytes are writable.
        let n = unsafe { unsafe_opus::opus_encode(self.0, pcm.as_ptr(), PKT as i32, out.as_mut_ptr(), out.len() as i32) };
        usize::try_from(n).ok().filter(|&n| n > 0)
    }
}

impl Drop for OpusEnc {
    fn drop(&mut self) {
        // SAFETY: created by opus_encoder_create, freed once
        unsafe { unsafe_opus::opus_encoder_destroy(self.0) }
    }
}

pub struct OpusDec(*mut unsafe_opus::OpusDecoder);
// SAFETY: as for OpusEnc
unsafe impl Send for OpusDec {}

impl OpusDec {
    pub fn new() -> Result<Self, String> {
        let mut err = 0;
        // SAFETY: valid arguments; the result is checked and owned by Self (freed in Drop).
        let d = unsafe { unsafe_opus::opus_decoder_create(RATE as i32, 1, &mut err) };
        if err != unsafe_opus::OPUS_OK || d.is_null() {
            return Err(format!("opus decoder error {err}"));
        }
        Ok(OpusDec(d))
    }

    /// Decodes `pkt` into one 20 ms frame. `fec` recovers the frame *before* `pkt` from its
    /// in-band FEC; `pkt = None` conceals a lost frame (PLC). A bad packet yields silence.
    pub fn decode(&mut self, pkt: Option<&[u8]>, fec: bool, out: &mut Pcm) {
        let (p, len) = pkt.map_or((std::ptr::null(), 0), |p| (p.as_ptr(), p.len() as i32));
        // SAFETY: p/len describe a live slice (or null/0 for PLC); out holds PKT samples.
        let n = unsafe { unsafe_opus::opus_decode(self.0, p, len, out.as_mut_ptr(), PKT as i32, fec as i32) };
        if n != PKT as i32 {
            out.fill(0);
        }
    }
}

impl Drop for OpusDec {
    fn drop(&mut self) {
        // SAFETY: created by opus_decoder_create, freed once
        unsafe { unsafe_opus::opus_decoder_destroy(self.0) }
    }
}

/// `v2v config` "hear yourself": the mic through Opus straight to the speaker, as the peer would hear it.
pub fn loopback(rx: Receiver<Pcm>, mut player: Player) {
    let (Ok(mut enc), Ok(mut dec)) = (OpusEnc::new(), OpusDec::new()) else { return };
    let (mut pkt, mut pcm) = ([0u8; 1500], [0i16; PKT]);
    for p in rx {
        if let Some(n) = enc.encode(&p, &mut pkt) {
            dec.decode(Some(&pkt[..n]), false, &mut pcm);
            player.push(&pcm);
        }
    }
}

/// Selectable devices as (cpal DeviceId string, name), without duplicates and aliases.
pub fn devices(input: bool) -> Vec<(String, String)> {
    let host = cpal::default_host();
    let list = if input { host.input_devices().map(|d| d.collect()) } else { host.output_devices().map(|d| d.collect()) };
    let mut out: Vec<(String, String)> = Vec::new();
    for d in list.unwrap_or_else(|_| Vec::new()) {
        let (Ok(id), name) = (d.id(), d.to_string()) else { continue };
        let id = id.to_string();
        if keep_device(&id, &name, input) && !out.iter().any(|(_, n)| *n == name) {
            out.push((id, name));
        }
    }
    out
}

/// The system default is a separate "" entry, so default aliases are dropped too.
fn keep_device(id: &str, name: &str, input: bool) -> bool {
    if let Some(pw) = id.strip_prefix("pipewire:") {
        // inputs named after outputs/sinks are monitors of the speakers, not microphones
        let monitor = input && (pw.contains("output") || pw.contains("sink"));
        return !monitor && !pw.ends_with("_default") && name != "unknown";
    }
    if let Some(alsa) = id.strip_prefix("alsa:") {
        // ALSA lists each card under many prefixes; plughw converts rate/format, CARD=<n> duplicates CARD=<name>
        return alsa.strip_prefix("plughw:CARD=").is_some_and(|c| !c.starts_with(|ch: char| ch.is_ascii_digit()));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_rate() {
        let mut n = 0;
        let mut r = Resampler::new(48_000, 16_000);
        for _ in 0..48_000 {
            r.push(0.5, |_| n += 1);
        }
        assert!((15_990..=16_010).contains(&n), "{n}");
    }

    #[test]
    fn player_drains_excess_latency() {
        let sh = Arc::new(Shared::new(crate::config::Settings::default(), false));
        let jitter: Jitter = Arc::default();
        let mut p = Player::new(sh, jitter.clone(), Arc::default());
        let ms = |n: usize| n * 1000 / RATE as usize;
        for _ in 0..5 {
            p.push(&[0; PKT]); // a burst leaves 100 ms queued
        }
        // the speaker takes exactly what arrives: without drift control the queue would stay at 100 ms
        for _ in 0..500 {
            p.push(&[0; PKT]);
            let mut q = jitter.lock().unwrap();
            let n = PKT.min(q.len());
            q.drain(..n);
        }
        let left = ms(jitter.lock().unwrap().len());
        assert!((28..=42).contains(&left), "{left} ms queued"); // 40 ms target, minus up to one window of 1% speed-up
    }

    fn white_noise(n: usize) -> Vec<f32> {
        let mut x = 0x2545_f491_u32;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 - 0.5) * 0.05
            })
            .collect()
    }

    /// Runs `input` through a 48 kHz denoiser in odd-sized blocks, like a cpal callback would.
    fn run(input: &[f32], on: impl Fn(usize) -> bool) -> Vec<f32> {
        let mut d = Denoiser::new(DENOISE_RATE);
        let mut out = Vec::new();
        for (i, b) in input.chunks(333).enumerate() {
            out.extend_from_slice(d.process(b, on(i)));
        }
        out
    }

    #[test]
    fn denoiser_framing_and_suppression() {
        let input = white_noise(3 * 48_000);
        let energy = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        let on = run(&input, |_| true);
        let off = run(&input, |_| false);
        // every whole 10 ms frame comes out, at 48 kHz, whatever the block size
        assert_eq!(on.len(), input.len() / FRAME * FRAME);
        assert_eq!(off.len(), on.len());
        // off = input delayed by one frame (+1 sample from the resampler), the model's own delay
        assert!(off[FRAME + 1..].iter().zip(&input).all(|(a, b)| (a - b).abs() < 1e-6));
        let ratio = energy(&on[48_000..]) / energy(&input[48_000 - FRAME..on.len() - FRAME]);
        assert!(ratio < 0.1, "white noise only {ratio} quieter");
        // toggling mid-stream keeps the sample count
        assert_eq!(run(&input, |i| i % 7 < 3).len(), on.len());
    }

    #[test]
    fn opus_fec_and_plc() {
        let (mut enc, mut dec) = (OpusEnc::new().unwrap(), OpusDec::new().unwrap());
        let tone = |k: usize| -> Pcm { std::array::from_fn(|i| (((k * PKT + i) as f32 * 0.0577).sin() * 8000.0) as i16) };
        let pkts: Vec<Vec<u8>> = (0..20)
            .map(|k| {
                let mut b = [0u8; 1500];
                let n = enc.encode(&tone(k), &mut b).unwrap();
                b[..n].to_vec()
            })
            .collect();
        assert!(pkts.iter().all(|p| p.len() < 200), "~32 kbit/s");
        let energy = |p: &Pcm| p.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        let mut out = [0i16; PKT];
        for p in &pkts[..10] {
            dec.decode(Some(p), false, &mut out);
        }
        // packet 10 lost: recovered from packet 11's FEC, then 11 itself; packet 13 lost: PLC
        dec.decode(Some(&pkts[11]), true, &mut out);
        assert!(energy(&out) > energy(&tone(10)) * 0.1, "FEC frame has signal");
        dec.decode(Some(&pkts[11]), false, &mut out);
        dec.decode(None, false, &mut out);
        assert!(energy(&out) > 0.0, "PLC continues the signal");
        dec.decode(Some(&[0xff; 3]), false, &mut out); // garbage never panics
    }

    #[test]
    fn device_filter() {
        assert!(keep_device("pipewire:alsa_input.usb-046d_0825-02.mono-fallback", "Webcam C270 Mono", true));
        assert!(!keep_device("pipewire:alsa_output.usb-x.analog-stereo", "Monitor", true));
        assert!(keep_device("pipewire:alsa_output.usb-x.analog-stereo", "StreamX", false));
        assert!(!keep_device("pipewire:input_default", "default_input", true));
        assert!(!keep_device("pipewire:sink_default", "default_sink", false));
        assert!(!keep_device("pipewire:speech-dispatcher-dummy", "unknown", false));
        assert!(keep_device("alsa:plughw:CARD=StreamX,DEV=0", "StreamX", true));
        for id in ["alsa:plughw:CARD=1,DEV=0", "alsa:default", "alsa:hw:CARD=StreamX,DEV=0", "alsa:pulse"] {
            assert!(!keep_device(id, "x", true), "{id}");
        }
    }
}
