//! Camera capture thread, effects hook, H.264 encode/decode, and the window-side helpers (mirror, picture-in-picture).

use crate::Shared;
use crate::blur::Segmenter;
use image::RgbImage;
use nokhwa::utils::CameraIndex;
use openh264::decoder::Decoder;
use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate, Level, Profile, RateControlMode, UsageType};
use openh264::formats::{RgbSliceU8, YUVBuffer, YUVSource};
use openh264::{OpenH264API, Timestamp};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Latest processed (effects applied, not mirrored) own camera frame; None while the camera is off.
pub type SelfView = Arc<Mutex<Option<Arc<RgbImage>>>>;
/// Latest decoded picture from the peer.
pub type Frame = Arc<Mutex<Option<Arc<RgbImage>>>>;
/// One encoded H.264 access unit (Annex-B) and when it was captured.
pub type Encoded = (Vec<u8>, Instant);

pub const FPS: u32 = 30;
/// Below this target the picture is encoded at half size: 640x480 at ~150 kbit/s is mush.
const SMALL_BPS: u32 = 400_000;

/// H.264 (openh264) for the call, rate-controlled by the bitrate the network thread publishes.
#[derive(Default)]
struct H264 {
    enc: Option<Encoder>,
    dims: (u32, u32),
    bps: u32,
    small: bool,
}

impl H264 {
    fn encode(&mut self, sh: &Shared, img: &RgbImage, at: Instant) -> Result<Vec<u8>, String> {
        let target = sh.stats.video_bps.load(Relaxed);
        // hysteresis, so a bitrate hovering around the threshold does not flip sizes every frame
        // wide band: every switch costs a new encoder and an IDR, just when bandwidth is short
        self.small = if self.small { target < SMALL_BPS * 3 / 2 } else { target < SMALL_BPS };
        let k = if self.small { 2 } else { 1 };
        let (w, h) = ((img.width() / k) & !1, (img.height() / k) & !1); // YUV 4:2:0 needs even sizes
        let scaled;
        let img = if (w, h) == img.dimensions() {
            img
        } else {
            scaled = image::imageops::thumbnail(img, w, h);
            &scaled
        };
        let enc = match &mut self.enc {
            Some(enc) if self.dims == (w, h) => {
                if target.abs_diff(self.bps) > self.bps / 10 {
                    self.bps = target;
                    set_bitrate(enc, target);
                }
                enc
            }
            _ => {
                // a new encoder per size, started at the current target (openh264 resets to its config bitrate)
                let cfg = EncoderConfig::new()
                    .bitrate(BitRate::from_bps(target))
                    .max_frame_rate(FrameRate::from_hz(FPS as f32))
                    .usage_type(UsageType::CameraVideoRealTime)
                    .rate_control_mode(RateControlMode::Bitrate)
                    .profile(Profile::Baseline)
                    .level(Level::Level_3_1)
                    .skip_frames(true)
                    .max_slice_len(1100); // one slice per RTP packet: a lost packet costs a slice, not a frame
                (self.dims, self.bps) = ((w, h), target);
                sh.keyframe.store(false, Relaxed); // a new encoder starts with an IDR anyway
                let mut enc = Encoder::with_api_config(OpenH264API::from_source(), cfg).map_err(|e| e.to_string())?;
                let mut quiet = openh264_sys2::WELS_LOG_QUIET;
                // SAFETY: TRACE_LEVEL is accepted before init and takes an int. Without it, init warns on stderr (over the TUI).
                unsafe { enc.raw_api().set_option(openh264_sys2::ENCODER_OPTION_TRACE_LEVEL, (&raw mut quiet).cast()) };
                self.enc.insert(enc)
            }
        };
        if sh.keyframe.swap(false, Relaxed) {
            enc.force_intra_frame();
        }
        let mut yuv = YUVBuffer::new(w as usize, h as usize);
        yuv.read_rgb8(RgbSliceU8::new(img.as_raw(), (w as usize, h as usize)));
        let ts = Timestamp::from_millis((at - sh.start).as_millis() as u64);
        Ok(enc.encode_at(&yuv, ts).map_err(|e| e.to_string())?.to_vec())
    }
}

/// Changes the target bitrate of an encoder that has encoded at least one frame.
fn set_bitrate(enc: &mut Encoder, bps: u32) {
    use openh264_sys2::*;
    // the layer-0 maximum first: openh264 rejects a target above it
    for (opt, layer) in [(ENCODER_OPTION_MAX_BITRATE, SPATIAL_LAYER_0), (ENCODER_OPTION_BITRATE, SPATIAL_LAYER_ALL)] {
        let mut info = SBitrateInfo { iLayer: layer, iBitrate: bps as i32 };
        // SAFETY: the encoder is initialised (first encode done); these options take an SBitrateInfo.
        unsafe { enc.raw_api().set_option(opt, (&raw mut info).cast()) };
    }
}

/// Decodes the peer's H.264 into `frame` until the sender (the network thread) goes away.
/// A decode error asks the network thread for a keyframe.
pub fn decode_loop(rx: Receiver<Arc<[u8]>>, frame: Frame, sh: Arc<Shared>) {
    let mut dec = match Decoder::new() {
        Ok(d) => d,
        Err(e) => return sh.log(format!("no video decoder ({e})")),
    };
    for au in rx {
        match dec.decode(&au) {
            Ok(Some(yuv)) => {
                let (w, h) = yuv.dimensions();
                let mut img = RgbImage::new(w as u32, h as u32);
                yuv.write_rgb8(&mut img);
                sh.stats.rx_frames.fetch_add(1, Relaxed);
                *frame.lock().unwrap() = Some(Arc::new(img));
            }
            Ok(None) => {}
            Err(_) => sh.want_keyframe.store(true, Relaxed),
        }
    }
}

/// 0RGB pixels for minifb, optionally flipped left-right (mirror is for the local preview only).
pub fn to_argb(img: &RgbImage, mirror: bool) -> Vec<u32> {
    let w = img.width() as usize;
    let raw = img.as_raw();
    (0..raw.len() / 3)
        .map(|i| {
            let (y, x) = (i / w, i % w);
            let p = &raw[(y * w + if mirror { w - 1 - x } else { x }) * 3..][..3];
            (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32
        })
        .collect()
}

/// Draws `src` scaled to a quarter of the width into the bottom-right corner of `dst`.
pub fn pip(dst: &mut [u32], dw: usize, dh: usize, src: &[u32], sw: usize, sh: usize) {
    let m = dw / 40; // margin
    let pw = dw / 4;
    let ph = (pw * sh).checked_div(sw).unwrap_or(0);
    if pw == 0 || ph == 0 || pw + m > dw || ph + m > dh || src.len() < sw * sh || dst.len() < dw * dh {
        return;
    }
    let (x0, y0) = (dw - pw - m, dh - ph - m);
    for y in 0..ph {
        let row = &src[(y * sh / ph) * sw..];
        for (x, d) in dst[(y0 + y) * dw + x0..][..pw].iter_mut().enumerate() {
            *d = row[x * sw / pw];
        }
    }
}

/// Effects on each outgoing frame, applied before H.264 encode and self-view.
/// The blur model loads the first time blur is on; any failure turns blur off and says why.
pub fn apply_effects(sh: &Shared, seg: &mut Option<Segmenter>, img: &mut RgbImage, blur: bool) {
    if !blur {
        if let Some(s) = seg {
            s.pause();
        }
        return;
    }
    if seg.is_none() {
        let t = Instant::now();
        match Segmenter::load() {
            Ok(s) => *seg = Some(s),
            Err(e) => return blur_off(sh, e),
        }
        sh.info(format!("background blur model loaded in {} ms", t.elapsed().as_millis()));
    }
    let Some(s) = seg else { return };
    if let Err(e) = s.apply(img) {
        *seg = None;
        return blur_off(sh, e);
    }
    if s.frames == 30 {
        sh.info(format!("background blur: {:.1} ms/frame", s.busy.as_secs_f64() * 1000.0 / s.frames as f64));
    }
}

fn blur_off(sh: &Shared, e: String) {
    sh.log(format!("background blur off: {e}"));
    sh.settings.lock().unwrap().blur = false;
}

/// Real capture devices as (name, name), the shape the TUI's device lists use.
pub fn cameras() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for c in usable_cameras() {
        if !out.iter().any(|(n, _)| *n == c.human_name()) {
            out.push((c.human_name(), c.human_name()));
        }
    }
    out
}

/// Skips V4L2 metadata nodes (sysfs index != 0): they fail to open and would make the list confusing.
fn usable_cameras() -> Vec<nokhwa::utils::CameraInfo> {
    // macOS lists no cameras until the user has granted access: ask once and wait for the answer (no-op elsewhere)
    static ACCESS: std::sync::Once = std::sync::Once::new();
    ACCESS.call_once(|| {
        if !nokhwa::nokhwa_check() {
            let (tx, rx) = channel();
            nokhwa::nokhwa_initialize(move |_| {
                let _ = tx.send(());
            });
            let _ = rx.recv_timeout(Duration::from_secs(60));
        }
    });
    let mut list = nokhwa::query(nokhwa::utils::ApiBackend::Auto).unwrap_or_default();
    list.retain(|c| match c.index() {
        CameraIndex::Index(n) => std::fs::read_to_string(format!("/sys/class/video4linux/video{n}/index")).map_or(true, |s| s.trim() == "0"),
        CameraIndex::String(_) => true,
    });
    list.sort_by_key(|c| c.index().as_index().unwrap_or(u32::MAX));
    list
}

/// shortcut: picks by name, so of two identical camera models only the first is selectable; store the path if that matters.
fn open(name: &str, sh: &Shared) -> Result<nokhwa::Camera, String> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraFormat, FrameFormat, RequestedFormat, RequestedFormatType};
    let list = usable_cameras();
    let found = list.iter().find(|c| c.human_name() == name);
    if !name.is_empty() && found.is_none() {
        sh.log(format!("camera {name} not found, using the default"));
    }
    let idx = found.or(list.first()).map_or(CameraIndex::Index(0), |c| c.index().clone());
    let want = CameraFormat::new_from(640, 480, FrameFormat::MJPEG, 30);
    let open = |t| nokhwa::Camera::new(idx.clone(), RequestedFormat::new::<RgbFormat>(t));
    let mut cam = open(RequestedFormatType::Closest(want))
        .or_else(|_| open(RequestedFormatType::AbsoluteHighestFrameRate))
        .map_err(|e| e.to_string())?;
    cam.open_stream().map_err(|e| e.to_string())?;
    Ok(cam)
}

/// A camera read on its own thread: nokhwa's V4L `frame()` waits with no timeout, so a stalled
/// camera must not stall settings changes or hang-up. Dropping it tells the thread to let go.
struct Capture {
    rx: Receiver<Result<RgbImage, String>>,
    stop: Arc<AtomicBool>,
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
    }
}

/// Opens `name` once `prev` (the last capture thread) has released its device, or after 1 s if it is stuck.
/// The returned receiver closes when this thread has released the camera in turn.
fn capture(name: String, sh: Arc<Shared>, prev: Option<Receiver<()>>) -> (Capture, Receiver<()>) {
    use nokhwa::pixel_format::RgbFormat;
    let (tx, rx) = sync_channel(1);
    let (done_tx, done) = channel::<()>();
    let stop = Arc::new(AtomicBool::new(false));
    let st = stop.clone();
    thread::spawn(move || {
        let _done = done_tx; // dropped last, after the camera
        if let Some(p) = prev {
            let _ = p.recv_timeout(Duration::from_secs(1));
        }
        let mut cam = match open(&name, &sh) {
            Ok(c) => c,
            Err(e) => return sh.log(format!("no camera ({e})")),
        };
        while !st.load(Relaxed) {
            let r = cam.frame().and_then(|f| f.decode_image::<RgbFormat>()).map_err(|e| e.to_string());
            let bad = r.is_err();
            if tx.send(r).is_err() {
                break;
            }
            if bad {
                thread::sleep(Duration::from_millis(10));
            }
        }
    });
    (Capture { rx, stop }, done)
}

/// Captures until hang-up, (re)opening the camera when the selection changes and closing it while video is off.
/// `net` is None in `v2v config`: frames then only go to the self-view.
pub fn camera_loop(sh: Arc<Shared>, net: Option<SyncSender<Encoded>>, selfv: SelfView) {
    let mut cap: Option<Capture> = None;
    let mut released: Option<Receiver<()>> = None; // closes once the last capture thread let go of its camera
    let mut seg: Option<Segmenter> = None;
    let mut applied: Option<Option<String>> = None; // camera wanted when we last (re)opened; Some(None) = video off
    let (mut h264, mut fails, mut last) = (H264::default(), 0, Instant::now() - Duration::from_secs(1));
    while !sh.quit.load(Relaxed) {
        let (want, blur) = {
            let s = sh.settings.lock().unwrap();
            ((!s.video_off).then(|| s.camera.clone()), s.blur)
        };
        if applied.as_ref() != Some(&want) {
            cap = None; // release the device before opening another
            *selfv.lock().unwrap() = None;
            if let Some(name) = &want {
                let (c, r) = capture(name.clone(), sh.clone(), released.take());
                (cap, released, fails) = (Some(c), Some(r), 0);
            }
            applied = Some(want);
        }
        let Some(c) = &cap else {
            thread::sleep(Duration::from_millis(100));
            continue;
        };
        // skip bad/truncated frames; give up (until the selection changes) only if the camera keeps failing
        let mut img: RgbImage = match c.rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(img)) => img,
            Err(RecvTimeoutError::Timeout) => continue,
            Ok(Err(e)) if fails >= 100 => {
                sh.log(format!("camera stopped: {e}"));
                cap = None;
                *selfv.lock().unwrap() = None;
                continue;
            }
            Ok(Err(_)) => {
                fails += 1;
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                cap = None; // could not open; the thread said why
                continue;
            }
        };
        fails = 0;
        // cap the rate: the encoder's rate control assumes at most FPS; the wide margin keeps a jittery FPS camera at full rate
        if last.elapsed() < Duration::from_millis(1000 / FPS as u64 * 2 / 3) {
            continue;
        }
        last = Instant::now();
        if img.width() > 640 || img.height() > 480 {
            let k = f64::min(640.0 / img.width() as f64, 480.0 / img.height() as f64);
            img = image::imageops::thumbnail(&img, (img.width() as f64 * k) as u32, (img.height() as f64 * k) as u32);
        }
        apply_effects(&sh, &mut seg, &mut img, blur);
        let img = Arc::new(img);
        *selfv.lock().unwrap() = Some(img.clone());
        let Some(tx) = &net else { continue };
        match h264.encode(&sh, &img, last) {
            Ok(au) if au.is_empty() => {} // rate control skipped this frame
            Ok(au) => {
                if let Err(TrySendError::Full(_)) = tx.try_send((au, last)) {
                    sh.keyframe.store(true, Relaxed); // the peer's decoder misses a frame: restart from an IDR
                }
            }
            Err(e) => {
                sh.log(format!("video encoder: {e}"));
                h264 = H264::default();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror() {
        let img = RgbImage::from_raw(3, 2, vec![1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0]).unwrap();
        let px = |v: Vec<u32>| v.into_iter().map(|p| p >> 16).collect::<Vec<_>>();
        assert_eq!(px(to_argb(&img, false)), [1, 2, 3, 4, 5, 6]);
        assert_eq!(px(to_argb(&img, true)), [3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn h264_resizes_with_bitrate() {
        let sh = crate::Shared::new(Default::default(), false);
        let (mut h264, mut dec) = (H264::default(), Decoder::new().unwrap());
        let mut sizes = Vec::new();
        let mut at = sh.start;
        for (i, bps) in [2_000_000, 1_000_000, 500_000, 200_000, 1_500_000].into_iter().enumerate() {
            sh.stats.video_bps.store(bps, Relaxed);
            for f in 0..10 {
                // synthetic moving gradient, not a camera frame
                let img = RgbImage::from_fn(640, 480, |x, y| image::Rgb([(x + f * 8) as u8, y as u8, (i * 50) as u8]));
                at += Duration::from_millis(33);
                let au = h264.encode(&sh, &img, at).unwrap();
                if au.is_empty() {
                    continue;
                }
                if let Some(yuv) = dec.decode(&au).unwrap_or_else(|e| panic!("{bps} frame {f}: {e}")) {
                    let d = yuv.dimensions();
                    if sizes.last().is_none_or(|l| *l != d) {
                        sizes.push(d);
                    }
                }
            }
        }
        // 500k stays full size: the switch down is at 400k, and back up only above 600k
        assert_eq!(sizes, [(640, 480), (320, 240), (640, 480)]);
    }

    #[test]
    fn pip_bounds() {
        let (dw, dh) = (640, 480);
        let mut dst = vec![0u32; dw * dh];
        pip(&mut dst, dw, dh, &vec![7u32; 320 * 240], 320, 240);
        let painted: Vec<usize> = (0..dst.len()).filter(|&i| dst[i] == 7).collect();
        assert_eq!(painted.len(), 160 * 120);
        let (first, last) = (painted[0], *painted.last().unwrap());
        assert!(first % dw > dw / 2 && first / dw > dh / 2, "bottom-right");
        assert!(last % dw < dw && last / dw < dh);
        // degenerate sizes never panic or draw
        for (w, h, sw, sh) in [(3, 3, 320, 240), (640, 10, 320, 240), (640, 480, 0, 0), (640, 480, 4000, 1)] {
            let mut d = vec![0u32; w * h];
            pip(&mut d, w, h, &vec![7u32; sw * sh], sw, sh);
        }
        // a tall source that would not fit is skipped
        let mut d = vec![0u32; 640 * 100];
        pip(&mut d, 640, 100, &vec![7u32; 100 * 400], 100, 400);
        assert!(d.iter().all(|&p| p == 0));
    }
}
