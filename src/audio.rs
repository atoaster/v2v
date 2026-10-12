//! Microphone and speaker streams, live device switching, test tone, device lists.

use crate::Shared;
use crate::net::{AUDIO, AUDIO_PKT, RATE, rand_u32};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use nnnoiseless::DenoiseState;
use std::collections::VecDeque;
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub type Jitter = Arc<Mutex<VecDeque<f32>>>;

/// Linear resampler, push style. shortcut: no low-pass filter, so downsampling aliases a bit; fine for voice.
pub struct Resampler {
    step: f64,
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
pub const DENOISE_RATE: u32 = 48_000;

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
    sock: UdpSocket,
    dest: SocketAddr,
    seq: Arc<AtomicU32>, // shared across mic rebuilds, so a switch is no seq jump for the peer
    mic: (Option<String>, Option<Live>), // (device id last tried, stream)
    spk: (Option<String>, Option<Live>),
}

impl AudioIo {
    pub fn new(sh: Arc<Shared>, jitter: Jitter, sock: UdpSocket, dest: SocketAddr) -> Self {
        let seq = Arc::new(AtomicU32::new(rand_u32()));
        AudioIo { sh, jitter, sock, dest, seq, mic: (None, None), spk: (None, None) }
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
        let mut rs = Resampler::new(DENOISE_RATE, RATE);
        let mut dn = Denoiser::new(rate);
        let mut pending: Vec<i16> = Vec::new();
        let mut block: Vec<f32> = Vec::new();
        let (sh, seq) = (self.sh.clone(), self.seq.clone());
        // send from a plain thread so the real-time callback never blocks in send_to
        let (tx, rx) = std::sync::mpsc::sync_channel::<[u8; 5 + 2 * AUDIO_PKT]>(50);
        let (sock, dest, sh2) = (self.sock.try_clone().map_err(|e| e.to_string())?, self.dest, self.sh.clone());
        thread::spawn(move || {
            for p in rx {
                if sock.send_to(&p, dest).is_ok() {
                    sh2.stats.tx_bytes.fetch_add(p.len() as u64, Relaxed);
                }
            }
        });
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
            // muted: send nothing (and keep seq), so the peer hears silence and counts no loss
            if muted || (sh.offline && !sh.loopback.load(Relaxed)) {
                pending.clear();
                return;
            }
            for &x in block {
                rs.push(x, |v| pending.push((v.clamp(-1.0, 1.0) * 32767.0) as i16));
            }
            while pending.len() >= AUDIO_PKT {
                let mut p = [0u8; 5 + 2 * AUDIO_PKT]; // on the stack: no allocation in the real-time callback
                p[0] = AUDIO;
                p[1..5].copy_from_slice(&seq.fetch_add(1, Relaxed).to_be_bytes());
                for (d, x) in p[5..].chunks_exact_mut(2).zip(pending.drain(..AUDIO_PKT)) {
                    d.copy_from_slice(&x.to_le_bytes());
                }
                let _ = tx.try_send(p);
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
        let mut r = Resampler::new(48_000, RATE);
        for _ in 0..48_000 {
            r.push(0.5, |_| n += 1);
        }
        assert!((15_990..=16_010).contains(&n), "{n}");
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
