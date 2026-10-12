//! `--tplay` mode: serve the received video as MJPEG to tplay in the terminal.

use crate::video::Frame;
use std::io::{Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread;
use std::time::Duration;

/// Serves `frame` as an MJPEG HTTP stream (what tplay/ffmpeg reads), JPEG-encoding each new picture.
/// Nothing is sent until the first
/// frame arrives: ffmpeg cannot follow a size change, so no placeholder frame; readers just wait,
/// and later frames (the peer's encoder halves its size on a slow link) are scaled to the first size.
fn serve_mjpeg(mut s: TcpStream, frame: Frame, path: &str) {
    let mut req = [0; 1024];
    let n = s.read(&mut req).unwrap_or(0);
    // only tplay knows the random path, so other local users cannot watch the call
    if !req[..n].starts_with(format!("GET {path} ").as_bytes()) {
        return;
    }
    let _ = s.set_nodelay(true); // header and JPEG are separate writes; do not let Nagle hold them
    let mut ok = s.write_all(b"HTTP/1.0 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=v2v\r\n\r\n").is_ok();
    let mut cur: Option<Arc<image::RgbImage>> = None;
    let mut size = None;
    while ok {
        let new = frame.lock().unwrap().clone().filter(|j| cur.as_ref().is_none_or(|c| !Arc::ptr_eq(j, c)));
        let Some(img) = new else {
            thread::sleep(Duration::from_millis(5));
            continue;
        };
        let (w, h) = *size.get_or_insert(img.dimensions());
        let scaled = (img.dimensions() != (w, h)).then(|| image::imageops::resize(&*img, w, h, image::imageops::FilterType::Triangle));
        let mut jpeg = Vec::new();
        if scaled.as_ref().unwrap_or(&img).write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 80)).is_err() {
            break;
        }
        let head = format!("--v2v\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpeg.len());
        ok = s.write_all(head.as_bytes()).and_then(|_| s.write_all(&jpeg)).and_then(|_| s.write_all(b"\r\n")).is_ok();
        cur = Some(img);
    }
}

/// Plays the received video in the terminal through tplay; returns when tplay exits or the peer hangs up (`quit`).
pub fn run_tplay(frame: Frame, peer: IpAddr, quit: &AtomicBool) {
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
    loop {
        match child.try_wait() {
            Ok(Some(st)) if !st.success() => std::process::exit(st.code().unwrap_or(1)),
            Ok(Some(_)) => return,
            Err(e) => fail(format!("tplay: {e}")),
            Ok(None) if quit.load(Relaxed) => break,
            Ok(None) => thread::sleep(Duration::from_millis(50)),
        }
    }
    // the peer hung up: tplay gets no signal it could clean up after, so restore the terminal for it
    let _ = child.kill();
    let _ = child.wait();
    use ratatui::crossterm::{cursor, execute, style::ResetColor, terminal};
    let _ = terminal::disable_raw_mode();
    let _ = execute!(std::io::stdout(), ResetColor, cursor::Show, terminal::LeaveAlternateScreen);
}
