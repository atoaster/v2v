//! `--tplay` mode: serve the received video as MJPEG to tplay in the terminal.

use crate::video::Frame;
use std::io::{Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::os::unix::fs::MetadataExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread;
use std::time::{Duration, Instant};

/// Serves `frame` as an MJPEG HTTP stream (what tplay/ffmpeg reads), JPEG-encoding each new picture.
/// Nothing is sent until the first
/// frame arrives: ffmpeg cannot follow a size change, so no placeholder frame; readers just wait,
/// and later frames (the peer's encoder halves its size on a slow link) are scaled to the first size.
/// Like the window, it shows gray once no frame came for 2 s (peer video off or stalled).
fn serve_mjpeg(mut s: TcpStream, frame: Frame, path: &str) {
    let mut req = [0; 1024];
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let n = s.read(&mut req).unwrap_or(0);
    // the path is in tplay's argv, which other local users can read: the uid check in run_tplay keeps them out
    if !req[..n].starts_with(format!("GET {path} ").as_bytes()) {
        return;
    }
    let _ = s.set_nodelay(true); // header and JPEG are separate writes; do not let Nagle hold them
    let mut ok = s.write_all(b"HTTP/1.0 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=v2v\r\n\r\n").is_ok();
    let mut cur: Option<Arc<image::RgbImage>> = None;
    let mut size = None;
    let (mut last, mut gray) = (Instant::now(), false);
    while ok {
        let new = frame.lock().unwrap().clone().filter(|j| cur.as_ref().is_none_or(|c| !Arc::ptr_eq(j, c)));
        let img = match (new, size) {
            (Some(img), _) => {
                (last, gray, cur) = (Instant::now(), false, Some(img.clone()));
                img
            }
            (None, Some((w, h))) if !gray && last.elapsed() > Duration::from_secs(2) => {
                gray = true;
                Arc::new(image::RgbImage::from_pixel(w, h, image::Rgb([0x40; 3])))
            }
            _ => {
                thread::sleep(Duration::from_millis(5));
                continue;
            }
        };
        let (w, h) = *size.get_or_insert(img.dimensions());
        let scaled = (img.dimensions() != (w, h)).then(|| image::imageops::resize(&*img, w, h, image::imageops::FilterType::Triangle));
        let mut jpeg = Vec::new();
        if scaled.as_ref().unwrap_or(&img).write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 80)).is_err() {
            break;
        }
        let head = format!("--v2v\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpeg.len());
        ok = s.write_all(head.as_bytes()).and_then(|_| s.write_all(&jpeg)).and_then(|_| s.write_all(b"\r\n")).is_ok();
    }
}

/// uid of the established TCP socket from local `port` to `peer_port`, from a /proc/net/tcp listing.
fn owner(tcp: &str, port: u16, peer_port: u16) -> Option<u32> {
    let port_of = |a: &str| u16::from_str_radix(a.rsplit(':').next()?, 16).ok();
    tcp.lines().skip(1).find_map(|l| {
        let f: Vec<&str> = l.split_whitespace().collect();
        let hit = f.len() > 7 && port_of(f[1]) == Some(port) && port_of(f[2]) == Some(peer_port) && f[3] == "01";
        hit.then(|| f[7].parse().ok()).flatten()
    })
}

/// Plays the received video in the terminal through tplay until it exits or the peer hangs up (`quit`).
/// Returns the exit code (tplay's, if it failed).
pub fn run_tplay(frame: Frame, peer: IpAddr, quit: &AtomicBool) -> Result<i32, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("cannot listen for tplay: {e}"))?;
    let path: Arc<str> = format!("/{:016x}", crate::rtc::random()).into();
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let url = format!("http://{addr}{path}");
    let uid = std::fs::metadata("/proc/self").map(|m| m.uid()).map_err(|e| format!("cannot read own uid: {e}"))?;
    // tplay probes the URL with ffprobe twice and then opens it, so every connection gets its own stream
    thread::spawn(move || {
        for s in listener.incoming().flatten() {
            // only our own user's processes may watch; unknown owner = refuse
            let client = s.peer_addr().map(|a| a.port()).unwrap_or(0);
            let tcp = std::fs::read_to_string("/proc/net/tcp").unwrap_or_default();
            if owner(&tcp, client, addr.port()) != Some(uid) {
                continue;
            }
            let (f, p) = (frame.clone(), path.clone());
            let _ = thread::Builder::new().spawn(move || serve_mjpeg(s, f, &p));
        }
    });
    let mut child = std::process::Command::new("tplay")
        // read frames as fast as they arrive, so a pause or stall does not leave a backlog behind live
        .args(["--fps", "100", &url])
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => "tplay not found on PATH (install it with: cargo install tplay)".into(),
            _ => format!("cannot start tplay: {e}"),
        })?;
    eprintln!("v2v: waiting for video from {peer} (Ctrl-C to quit)...");
    loop {
        match child.try_wait() {
            Ok(Some(st)) => return Ok(if st.success() { 0 } else { st.code().unwrap_or(1) }),
            Err(e) => return Err(format!("tplay: {e}")),
            Ok(None) if quit.load(Relaxed) => break,
            Ok(None) => thread::sleep(Duration::from_millis(50)),
        }
    }
    // the peer hung up: tplay gets no signal it could clean up after, so restore the terminal for it
    let _ = child.kill();
    let _ = child.wait();
    crate::viz::restore();
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stalled_peer_turns_gray() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(5))).unwrap(); // fail instead of hanging
        let frame: Frame = Default::default();
        *frame.lock().unwrap() = Some(Arc::new(image::RgbImage::from_pixel(64, 48, image::Rgb([200, 0, 0]))));
        let s = l.accept().unwrap().0;
        thread::spawn(move || serve_mjpeg(s, frame, "/x"));
        c.write_all(b"GET /x HTTP/1.0\r\n\r\n").unwrap();
        let mut r = std::io::BufReader::new(c);
        let mut jpeg = || {
            let mut line = String::new();
            let len = loop {
                line.clear();
                std::io::BufRead::read_line(&mut r, &mut line).unwrap();
                if let Some(n) = line.strip_prefix("Content-Length: ") {
                    break n.trim().parse::<usize>().unwrap();
                }
            };
            let mut buf = vec![0; len + 2]; // the blank line after the header, then the JPEG
            r.read_exact(&mut buf).unwrap();
            image::load_from_memory(&buf[2..]).unwrap().to_rgb8()
        };
        assert!(jpeg().get_pixel(10, 10)[0] > 150); // the peer's frame
        let t = Instant::now();
        let g = jpeg(); // no new frame: gray after 2 s, same size
        assert!(t.elapsed() > Duration::from_millis(1500));
        assert_eq!(g.dimensions(), (64, 48));
        assert!(g.pixels().all(|p| p.0.iter().all(|&v| v.abs_diff(0x40) < 4)));
    }

    #[test]
    fn owner_row() {
        let tcp = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:A1B2 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1001        0 1 1 0 100 0 0 10 0
   1: 0100007F:C350 0100007F:A1B2 01 00000000:00000000 00:00000000 00000000  1000        0 2 1 0 20 4 30 10 -1
   2: 0100007F:A1B2 0100007F:C350 01 00000000:00000000 00:00000000 00000000  1001        0 3 1 0 20 4 30 10 -1";
        assert_eq!(owner(tcp, 0xC350, 0xA1B2), Some(1000)); // the client's end, not the server's
        assert_eq!(owner(tcp, 0xC351, 0xA1B2), None);
        assert_eq!(owner(tcp, 0xA1B2, 0), None); // listening, not established
        assert_eq!(owner("", 1, 2), None);
    }
}
