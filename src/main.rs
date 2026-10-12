mod audio;
mod blur;
mod config;
mod rtc;
mod signal;
mod tplay;
mod tui;
mod video;
mod viz;

use minifb::{Key, ScaleMode, Window, WindowOptions};
use rtc::{PORT, Stats};
use std::io::IsTerminal;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use video::Frame;

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
    pub keyframe: AtomicBool,      // the peer needs an IDR from our encoder
    pub want_keyframe: AtomicBool, // our decoder needs an IDR from the peer (sent as a PLI)
    pub peer_bye: AtomicBool,   // the peer hung up (its BYE ended our call)
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
            out_rate: AtomicU32::new(audio::RATE),
            tone: AtomicU32::new(0),
            stats: Stats::default(),
            keyframe: AtomicBool::new(false),
            want_keyframe: AtomicBool::new(false),
            peer_bye: AtomicBool::new(false),
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
        "{} - peer-to-peer video call (WebRTC: H.264 + Opus) over UDP port 47000\n\n\
         {} v2v [--tplay | --audio] [<ip_address>]\n       v2v config\n\n\
         Calls <ip_address> with your camera and mic. Without an address, v2v waits and answers\n\
         the first v2v that calls, replying to wherever its packets come from (works through NAT).\n\
         During a call the terminal shows a menu: {} mute mic, {} stop video, {} blur background,\n\
         {} noise suppression, {} test sound, arrows/Enter for devices, volume, mirror and self-view.\n\
         {} in the menu, Esc in the window or closing it hangs up ({} in --tplay).\n\
         {} opens the same menu without calling, with a camera preview window, to test\n\
         devices before a call. Settings are saved to {} and used by every mode\n\
         (calls without the menu, i.e. --tplay, --audio or no terminal, start unmuted with video on).\n\n\
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
    for i in callable() {
        s += &format!("  v2v {} {}\n", cmd(&format!("{:<15}", i.ip())), dim(&format!("# {}", i.name)));
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
    // minifb's XOpenIM segfaults if XMODIFIERS names an input method not running on this display
    // (ssh -X, nested X); the window takes no text input. Safe: no other thread exists yet.
    unsafe { std::env::set_var("XMODIFIERS", "@im=none") };
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
        (None, Mode::Window)
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
    let mut settings = config::load();
    if mode != Mode::Window || !tty {
        // no menu to undo a remembered mute / video off, so these modes always start live
        (settings.mute_mic, settings.video_off) = (false, false);
    }
    // calling an address of this machine: one socket cannot be both WebRTC peers, so skip the network
    let self_call = peer.is_some_and(|p| p.is_loopback() || if_addrs::get_if_addrs().is_ok_and(|l| l.iter().any(|i| i.ip() == p)));
    // the call socket; `v2v config` sends nothing and has none
    let call = (!offline && !self_call).then(|| {
        // shortcut: answering listens on IPv4 only; bind [::] and unmap v4-mapped sources if v6 callers matter
        let any: IpAddr = if peer.is_none_or(|p| p.is_ipv4()) { Ipv4Addr::UNSPECIFIED.into() } else { Ipv6Addr::UNSPECIFIED.into() };
        let sock = UdpSocket::bind(SocketAddr::new(any, PORT)).unwrap_or_else(|e| {
            eprintln!("v2v: cannot bind UDP port {PORT}: {e}");
            std::process::exit(1);
        });
        let nonce = rtc::random();
        let dest = match peer {
            Some(p) => SocketAddr::new(p, PORT),
            None => answer(&sock, nonce),
        };
        (sock, dest, nonce)
    });
    let peer = call.as_ref().map(|c| c.1.ip()).or(peer).unwrap_or(Ipv4Addr::LOCALHOST.into());

    let sh = Arc::new(Shared::new(settings, offline));
    let frame: Frame = Arc::default();
    let jitter: audio::Jitter = Arc::default();
    let tap: viz::Tap = Arc::default();
    let (pcm_tx, pcm_rx) = sync_channel(50); // 1 s of 20 ms frames
    let mut io = audio::AudioIo::new(sh.clone(), jitter.clone(), pcm_tx);
    io.sync(); // first open before any UI, so its warnings land on the normal screen
    io.spawn();
    let player = audio::Player::new(sh.clone(), jitter, tap.clone());
    let selfv = video::SelfView::default();
    let (video_tx, video_rx) = sync_channel(4);
    if mode != Mode::Audio {
        let (sh, net, selfv) = (sh.clone(), (!offline).then_some(video_tx), selfv.clone());
        thread::spawn(move || video::camera_loop(sh, net, selfv));
    }
    let video_out = (mode != Mode::Audio && !offline).then(|| {
        let (tx, rx) = sync_channel(8);
        let (frame, sh) = (frame.clone(), sh.clone());
        thread::spawn(move || video::decode_loop(rx, frame, sh));
        tx
    });
    let mut net = None;
    if let Some((sock, dest, nonce)) = call {
        let io = rtc::Io { video_in: (mode != Mode::Audio).then_some(video_rx), pcm_in: pcm_rx, video_out, player };
        let sh = sh.clone();
        net = Some(thread::spawn(move || rtc::run(sock, dest, nonce, sh, io)));
    } else {
        if self_call {
            // self-test: own camera and mic through the codecs, as the peer would get them
            sh.stats.video_bps.store(2_000_000, Relaxed);
            *sh.stats.link.lock().unwrap() = "yourself, no network".into();
            if let Some(tx) = video_out {
                thread::spawn(move || video_rx.into_iter().for_each(|(au, _)| _ = tx.send(au.into())));
            }
        }
        thread::spawn(move || audio::loopback(pcm_rx, player));
    }
    // errors still fall through to the hang-up below, so the peer gets its BYE
    let code = match mode {
        Mode::Tplay => tplay::run_tplay(frame, peer, &sh.quit).unwrap_or_else(|e| {
            eprintln!("v2v: {e}");
            1
        }),
        Mode::Audio => viz::run_audio(peer, tap, &sh.mic_level, &sh.quit).map_or_else(
            |e| {
                eprintln!("v2v: terminal error: {e}");
                1
            },
            |_| 0,
        ),
        Mode::Window => run_window(sh.clone(), frame, selfv, peer, tty),
    };
    sh.quit.store(true, Relaxed);
    // the network thread sends the BYE on its way out (within a few ms)
    if let Some(net) = net {
        let _ = net.join();
    }
    if sh.peer_bye.load(Relaxed) {
        eprintln!("v2v: the other side hung up");
    }
    if code != 0 {
        std::process::exit(code);
    }
}

/// The interfaces a peer can call us on (IPv4, not loopback or link-local).
fn callable() -> Vec<if_addrs::Interface> {
    let ifs = if_addrs::get_if_addrs().unwrap_or_default();
    ifs.into_iter().filter(|i| i.ip().is_ipv4() && !i.is_loopback() && !i.is_link_local()).collect()
}

/// Waits for the caller's first signalling packet and returns where it came from; replying there gets through NAT.
/// `nonce` is the one the call will start with, so a HELLO sent from here is not a stale nonce later.
fn answer(sock: &UdpSocket, nonce: u64) -> SocketAddr {
    let color = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let c = |code: &str, t: &str| if color { format!("\x1b[{code}m{t}\x1b[0m") } else { t.to_string() };
    let mut s = format!("{}\n", c("1;33", &format!("v2v listening on UDP port {PORT}:")));
    for i in callable() {
        s += &format!("  {} {}\n", c("32", &format!("{:<15}", i.ip())), c("2", &format!("({})", i.name)));
    }
    eprintln!("{s}{}", c("2", "waiting for a call, Ctrl-C to cancel..."));
    let mut buf = [0u8; 2048];
    let (hello, mut last) = (signal::hello(nonce), None::<Instant>);
    loop {
        match sock.recv_from(&mut buf) {
            // a BYE is a past call hanging up, not a caller
            Ok((n, src)) if signal::decode(&buf[..n]).is_some_and(|m| m.kind != signal::Kind::Bye) => {
                eprintln!("v2v: answering {src}");
                return src;
            }
            // media from a call this side was in before a restart: a HELLO with a new nonce makes
            // that peer renegotiate now instead of after its 5 s silence timeout
            Ok((_, src)) if last.is_none_or(|t| t.elapsed() > Duration::from_millis(300)) => {
                last = Some(Instant::now());
                let _ = sock.send_to(&hello, src);
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
/// Returns the process exit code.
fn run_window(sh: Arc<Shared>, frame: Frame, selfv: video::SelfView, peer: IpAddr, tty: bool) -> i32 {
    let title = if sh.offline { "v2v config - camera preview".to_string() } else { format!("v2v - {peer}") };
    let opts = WindowOptions { resize: true, scale_mode: ScaleMode::AspectRatioStretch, ..Default::default() };
    // window first: minifb may print to stderr while creating it, which must not land in the TUI
    let win = Window::new(&title, 640, 480, opts);
    let ui = tty.then(|| {
        let (sh, t) = (sh.clone(), if sh.offline { "v2v config".to_string() } else { format!("v2v call — {peer}") });
        thread::spawn(move || tui::run(sh, t))
    });
    let code = match win {
        Ok(mut win) => {
            win.set_target_fps(30);
            let gray = (320, 240, vec![0x404040u32; 320 * 240]); // "waiting" / camera off
            let mut shown = gray.clone();
            let mut shown_at = None::<Instant>; // when `shown` arrived; None while it is the gray one
            let mut me: Option<(Arc<image::RgbImage>, bool, Vec<u32>)> = None; // converted self-view cache
            let mut buf = Vec::new();
            while win.is_open() && !win.is_key_down(Key::Escape) && !sh.quit.load(Relaxed) {
                let new = frame.lock().unwrap().take(); // convert outside the lock
                if let Some(img) = new {
                    shown = (img.width() as usize, img.height() as usize, video::to_argb(&img, false));
                    shown_at = Some(Instant::now());
                } else if shown_at.is_some_and(|t| t.elapsed() > Duration::from_secs(2)) {
                    // the peer turned its video off or stalled: do not freeze on its last picture
                    (shown, shown_at) = (gray.clone(), None);
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
            0
        }
        Err(e) if ui.is_none() => {
            eprintln!("v2v: cannot open window: {e}");
            1
        }
        Err(e) => {
            // straight to the status line: the menu thread may not have taken over stderr yet
            *sh.status.lock().unwrap() = format!("no window ({e}); menu only");
            while !sh.quit.load(Relaxed) {
                thread::sleep(Duration::from_millis(50));
            }
            0
        }
    };
    sh.quit.store(true, Relaxed);
    if let Some(Ok(Err(e))) = ui.map(|t| t.join()) {
        eprintln!("v2v: terminal error: {e}");
        return 1;
    }
    code
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

    #[test]
    fn answer_ignores_bye() {
        let bind = || UdpSocket::bind("127.0.0.1:0").unwrap();
        let (a, b, c) = (bind(), bind(), bind());
        let to = a.local_addr().unwrap();
        let waiter = thread::spawn(move || answer(&a, 7));
        b.send_to(b"junk", to).unwrap();
        b.send_to(&signal::bye(9), to).unwrap();
        c.send_to(&signal::hello(5), to).unwrap();
        assert_eq!(waiter.join().unwrap(), c.local_addr().unwrap());
        // the junk got a HELLO (rate-limited, so none for the BYE right after it)
        b.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let mut buf = [0u8; 2048];
        let mut got = Vec::new();
        while let Ok(n) = b.recv(&mut buf) {
            got.push(signal::decode(&buf[..n]).map(|m| (m.kind, m.nonce)));
        }
        assert_eq!(got, [Some((signal::Kind::Hello, 7))]);
    }
}
