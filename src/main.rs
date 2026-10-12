mod audio;
mod blur;
mod config;
mod net;
mod tplay;
mod tui;
mod video;
mod viz;

use minifb::{Key, ScaleMode, Window, WindowOptions};
use net::{Frame, PORT, RATE, Stats};
use std::io::IsTerminal;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// State shared by the UI, audio callbacks, camera and network threads.
pub struct Shared {
    pub settings: Mutex<config::Settings>,
    pub quit: AtomicBool,
    pub offline: bool,          // `v2v config`: nothing goes to the network
    pub loopback: AtomicBool,   // `v2v config`: mic is played back locally
    pub mic_level: AtomicU32,   // RMS (f32 bits) of the last mic block
    pub out_rate: AtomicU32,    // current speaker sample rate
    pub tone: AtomicU32,        // test-tone samples still to play
    pub stats: Stats,
    pub tui: AtomicBool,        // while set, warnings go to the status line instead of stderr
    pub status: Mutex<String>,
    pub start: Instant,
}

impl Shared {
    fn new(settings: config::Settings, offline: bool) -> Self {
        Shared {
            settings: Mutex::new(settings),
            quit: AtomicBool::new(false),
            offline,
            loopback: AtomicBool::new(false),
            mic_level: AtomicU32::new(0),
            out_rate: AtomicU32::new(RATE),
            tone: AtomicU32::new(0),
            stats: Stats::default(),
            tui: AtomicBool::new(false),
            status: Mutex::new(String::new()),
            start: Instant::now(),
        }
    }

    /// Runtime warning: stderr would smear the TUI, so it goes to its status line while that is up.
    pub fn log(&self, msg: String) {
        if self.tui.load(Relaxed) {
            *self.status.lock().unwrap() = msg;
        } else {
            eprintln!("v2v: {msg}");
        }
    }

    /// Informational: only shown in the menu, never on stderr (where it would smear tplay's picture).
    pub fn info(&self, msg: String) {
        if self.tui.load(Relaxed) {
            *self.status.lock().unwrap() = msg;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Window,
    Tplay,
    Audio,
}

/// `v2v [--tplay | --audio] [<ip>]`, flag before or after the address; no address = answer the first caller.
fn parse_args(args: &[String]) -> Option<(Option<IpAddr>, Mode)> {
    let n = |f: &str| args.iter().filter(|a| *a == f).count();
    let mode = match (n("--tplay"), n("--audio")) {
        (0, 0) => Mode::Window,
        (1, 0) => Mode::Tplay,
        (0, 1) => Mode::Audio,
        _ => return None,
    };
    match args.iter().filter(|a| *a != "--tplay" && *a != "--audio").collect::<Vec<_>>().as_slice() {
        [] => Some((None, mode)),
        [ip] => Some((Some(ip.parse().ok()?), mode)),
        _ => None,
    }
}

/// Usage plus the reply command the peer runs for each of our addresses.
fn help(color: bool) -> String {
    let c = |code: &str, t: &str| if color { format!("\x1b[{code}m{t}\x1b[0m") } else { t.to_string() };
    let head = |t: &str| c("1;33", t);
    let cmd = |t: &str| c("32", t);
    let dim = |t: &str| c("2", t);
    let mut s = format!(
        "{} - peer-to-peer video call over UDP port 47000\n\n\
         {} v2v [--tplay | --audio] [<ip_address>]\n       v2v config\n\n\
         Calls <ip_address> with your camera and mic. Without an address, v2v waits and answers\n\
         the first v2v that calls, replying to wherever its packets come from (works through NAT).\n\
         During a call the terminal shows a menu: {} mute mic, {} stop video, {} blur background,\n\
         {} noise suppression, {} test sound, arrows/Enter for devices, volume, mirror and self-view.\n\
         {} in the menu, Esc in the window or closing it hangs up ({} in --tplay).\n\
         {} opens the same menu without calling, with a camera preview window, to test\n\
         devices before a call. Settings are saved to {} and used by every mode\n\
         (--tplay and --audio start unmuted, with video on).\n\n\
         {}\n",
        c("1;36", "v2v"),
        head("usage:"),
        c("1", "m"),
        c("1", "v"),
        c("1", "b"),
        c("1", "n"),
        c("1", "t"),
        c("1", "q"),
        c("1", "q"),
        c("36", "v2v config"),
        config::path().map_or("$XDG_CONFIG_HOME/v2v/config".into(), |p| p.display().to_string()),
        head("The other side responds with one of:"),
    );
    let ifs = if_addrs::get_if_addrs().unwrap_or_default();
    // link-local v6 would need a %scope suffix that differs per peer, so skip it
    for i in ifs.iter().filter(|i| !i.is_loopback() && !i.is_link_local()) {
        s += &format!("  v2v {} {}\n", cmd(&format!("{:<39}", i.ip())), dim(&format!("# {}", i.name)));
    }
    s + &format!(
        "\n{}\n  {}  render the video in the terminal via tplay instead of a window\n  \
         {}  audio-only call with a live spectrum of the other side in the terminal ({} hangs up);\n           \
         no camera is used, so a peer in video mode only sees its gray waiting screen\n",
        head("options:"),
        c("36", "--tplay"),
        c("36", "--audio"),
        c("1", "q"),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        let out = std::io::stdout();
        print!("{}", help(out.is_terminal() && std::env::var_os("NO_COLOR").is_none()));
        return;
    }
    let tty = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let offline = args == ["config"];
    let (peer, mode) = if offline {
        if !tty {
            eprintln!("v2v: config is an interactive menu, but stdin/stdout is not a terminal");
            std::process::exit(1);
        }
        (Some(Ipv4Addr::LOCALHOST.into()), Mode::Window)
    } else {
        parse_args(&args).unwrap_or_else(|| {
            eprintln!("usage: v2v [--tplay | --audio] [<ip_address>]\n       v2v config\n       v2v -h");
            std::process::exit(2);
        })
    };
    if mode == Mode::Audio && !std::io::stdout().is_terminal() {
        eprintln!("v2v: --audio draws in the terminal, but stdout is not a terminal");
        std::process::exit(1);
    }
    // shortcut: answering listens on IPv4 only; bind [::] and unmap v4-mapped sources if v6 callers matter
    let any: IpAddr = if peer.is_none_or(|p| p.is_ipv4()) { Ipv4Addr::UNSPECIFIED.into() } else { Ipv6Addr::UNSPECIFIED.into() };
    // config mode: a private loopback socket, only used for "hear yourself"
    let bind = match peer {
        Some(p) if offline => SocketAddr::new(p, 0),
        _ => SocketAddr::new(any, PORT),
    };
    let sock = UdpSocket::bind(bind).unwrap_or_else(|e| {
        eprintln!("v2v: cannot bind UDP port {}: {e}", bind.port());
        std::process::exit(1);
    });
    let dest = match peer {
        _ if offline => sock.local_addr().expect("local address"),
        Some(p) => SocketAddr::new(p, PORT),
        None => answer(&sock),
    };
    let peer = dest.ip();
    let clone = || sock.try_clone().expect("clone UDP socket");

    let mut settings = config::load();
    if mode != Mode::Window || !tty {
        // no menu to undo a remembered mute / video off, so these modes always start live
        (settings.mute_mic, settings.video_off) = (false, false);
    }
    let sh = Arc::new(Shared::new(settings, offline));
    let frame: Frame = Arc::default();
    let jitter: audio::Jitter = Arc::default();
    let mut io = audio::AudioIo::new(sh.clone(), jitter.clone(), clone(), dest);
    io.sync(); // first open before any UI, so its warnings land on the normal screen
    io.spawn();
    let selfv = video::SelfView::default();
    if mode != Mode::Audio {
        let (sh, net, selfv) = (sh.clone(), (!offline).then(|| (clone(), dest)), selfv.clone());
        thread::spawn(move || video::camera_loop(sh, net, selfv));
    }
    let tap: viz::Tap = Arc::default();
    let (rsock, rframe, rtap, rsh) = (clone(), (mode != Mode::Audio).then(|| frame.clone()), tap.clone(), sh.clone());
    thread::spawn(move || net::recv_loop(rsock, peer, rframe, jitter, rtap, rsh));
    match mode {
        Mode::Tplay => tplay::run_tplay(frame, peer),
        Mode::Audio => {
            if let Err(e) = viz::run_audio(peer, tap, &sh.mic_level) {
                eprintln!("v2v: terminal error: {e}");
                std::process::exit(1);
            }
        }
        Mode::Window => run_window(sh, frame, selfv, peer, tty),
    }
}

/// Waits for the first valid v2v packet and returns where it came from; replying there gets through NAT.
fn answer(sock: &UdpSocket) -> SocketAddr {
    eprintln!("v2v: waiting for a call on UDP port {PORT} (Ctrl-C to cancel)...");
    let mut buf = [0u8; 2048];
    loop {
        match sock.recv_from(&mut buf) {
            Ok((n, src)) if net::parse(&buf[..n]).is_some() => {
                eprintln!("v2v: answering {src}");
                return src;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("v2v: receive error: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// minifb on the main thread, the menu (if there is a terminal) in its own thread; either one ending hangs up.
fn run_window(sh: Arc<Shared>, frame: Frame, selfv: video::SelfView, peer: IpAddr, tty: bool) {
    let title = if sh.offline { "v2v config - camera preview".to_string() } else { format!("v2v - {peer}") };
    let opts = WindowOptions { resize: true, scale_mode: ScaleMode::AspectRatioStretch, ..Default::default() };
    // window first: minifb may print to stderr while creating it, which must not land in the TUI
    let win = Window::new(&title, 640, 480, opts);
    let ui = tty.then(|| {
        let (sh, t) = (sh.clone(), if sh.offline { "v2v config".to_string() } else { format!("v2v call — {peer}") });
        thread::spawn(move || tui::run(sh, t))
    });
    match win {
        Ok(mut win) => {
            win.set_target_fps(30);
            let gray = (320, 240, vec![0x404040u32; 320 * 240]); // "waiting" / camera off
            let mut shown = gray.clone();
            let mut me: Option<(Arc<image::RgbImage>, bool, Vec<u32>)> = None; // converted self-view cache
            let mut buf = Vec::new();
            while win.is_open() && !win.is_key_down(Key::Escape) && !sh.quit.load(Relaxed) {
                let jpeg = frame.lock().unwrap().take(); // decode outside the lock
                if let Some(f) = jpeg.and_then(|j| video::decode(&j)) {
                    shown = f;
                }
                let (mirror, show_self) = {
                    let s = sh.settings.lock().unwrap();
                    (s.mirror, s.self_view)
                };
                let cur = selfv.lock().unwrap().clone();
                me = cur.map(|img| match me.take() {
                    Some(c) if Arc::ptr_eq(&c.0, &img) && c.1 == mirror => c,
                    _ => {
                        let px = video::to_argb(&img, mirror);
                        (img, mirror, px)
                    }
                });
                let (w, h) = match &me {
                    Some((img, _, px)) if sh.offline => {
                        buf.clone_from(px);
                        (img.width() as usize, img.height() as usize)
                    }
                    _ if sh.offline => {
                        buf.clone_from(&gray.2);
                        (gray.0, gray.1)
                    }
                    _ => {
                        buf.clone_from(&shown.2);
                        if let Some((img, _, px)) = me.as_ref().filter(|_| show_self) {
                            video::pip(&mut buf, shown.0, shown.1, px, img.width() as usize, img.height() as usize);
                        }
                        (shown.0, shown.1)
                    }
                };
                if let Err(e) = win.update_with_buffer(&buf, w, h) {
                    sh.log(format!("window error: {e}"));
                    break;
                }
            }
        }
        Err(e) if ui.is_none() => {
            eprintln!("v2v: cannot open window: {e}");
            std::process::exit(1);
        }
        Err(e) => {
            // straight to the status line: the menu thread may not have taken over stderr yet
            *sh.status.lock().unwrap() = format!("no window ({e}); menu only");
            while !sh.quit.load(Relaxed) {
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
    sh.quit.store(true, Relaxed);
    if let Some(Ok(Err(e))) = ui.map(|t| t.join()) {
        eprintln!("v2v: terminal error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args() {
        let p = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let ip: Option<IpAddr> = "10.0.0.2".parse().ok();
        assert_eq!(p(&[]), Some((None, Mode::Window)));
        assert_eq!(p(&["--audio"]), Some((None, Mode::Audio)));
        assert_eq!(p(&["10.0.0.2"]), Some((ip, Mode::Window)));
        assert_eq!(p(&["--tplay", "10.0.0.2"]), Some((ip, Mode::Tplay)));
        assert_eq!(p(&["10.0.0.2", "--tplay"]), Some((ip, Mode::Tplay)));
        assert_eq!(p(&["--audio", "10.0.0.2"]), Some((ip, Mode::Audio)));
        assert_eq!(p(&["10.0.0.2", "--audio"]), Some((ip, Mode::Audio)));
        for bad in [
            &["x"][..],
            &["x"],
            &["--tplay", "--tplay", "10.0.0.2"],
            &["1.2.3.4", "5.6.7.8"],
            &["--audio", "--tplay", "10.0.0.2"],
            &["--audio", "10.0.0.2", "--audio"],
        ] {
            assert_eq!(p(bad), None, "{bad:?}");
        }
    }
}
