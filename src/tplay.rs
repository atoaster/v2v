//! `--tplay` mode: serve the received video as MJPEG to tplay in the terminal.

use crate::net::Frame;
use std::io::{Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

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
pub fn run_tplay(frame: Frame, peer: IpAddr) {
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
