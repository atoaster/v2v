//! Camera capture thread, effects hook, and the window-side helpers (decode, mirror, picture-in-picture).

use crate::Shared;
use crate::blur::Segmenter;
use crate::net::{CHUNK, MAX_CHUNKS, rand_u32, video_packets};
use image::{ImageFormat, ImageReader, Limits, RgbImage};
use nokhwa::utils::CameraIndex;
use std::io::Cursor;
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Latest processed (effects applied, not mirrored) own camera frame; None while the camera is off.
pub type SelfView = Arc<Mutex<Option<Arc<RgbImage>>>>;

pub fn decode(jpeg: &[u8]) -> Option<(usize, usize, Vec<u32>)> {
    let mut r = ImageReader::with_format(Cursor::new(jpeg), ImageFormat::Jpeg);
    let mut lim = Limits::default();
    lim.max_image_width = Some(1920);
    lim.max_image_height = Some(1920);
    r.limits(lim);
    let img = r.decode().ok()?.to_rgb8();
    Some((img.width() as usize, img.height() as usize, to_argb(&img, false)))
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

/// Effects on each outgoing frame, applied before JPEG encode and self-view.
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
pub fn camera_loop(sh: Arc<Shared>, net: Option<(UdpSocket, SocketAddr)>, selfv: SelfView) {
    let mut cap: Option<Capture> = None;
    let mut released: Option<Receiver<()>> = None; // closes once the last capture thread let go of its camera
    let mut seg: Option<Segmenter> = None;
    let mut applied: Option<Option<String>> = None; // camera wanted when we last (re)opened; Some(None) = video off
    let (mut id, mut fails, mut last) = (rand_u32(), 0, Instant::now() - Duration::from_secs(1));
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
        // shortcut: fixed ~15 fps cap (~1-3 Mbit/s), no congestion control; add adaptive bitrate if calls run over slow links
        if last.elapsed() < Duration::from_millis(66) {
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
        let Some((sock, dest)) = &net else { continue };
        let mut jpeg = Vec::new();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 60);
        if img.write_with_encoder(enc).is_err() || jpeg.len() > CHUNK * MAX_CHUNKS as usize {
            continue;
        }
        let pkts = video_packets(id, &jpeg);
        // spread a frame over ~40 ms: back-to-back bursts get dropped by VPN/router queues, and one lost chunk loses the frame
        let gap = Duration::from_millis(40) / pkts.len().max(1) as u32;
        for p in pkts {
            if sock.send_to(&p, dest).is_ok() {
                sh.stats.tx_bytes.fetch_add(p.len() as u64, Relaxed);
            }
            thread::sleep(gap);
        }
        id = id.wrapping_add(1);
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
