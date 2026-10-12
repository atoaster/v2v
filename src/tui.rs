//! In-call / `v2v config` menu: mute, video, effects, devices, volume, test tone, stats.

use crate::config::{self, MAX_VOLUME, Settings};
use crate::{Shared, audio, video};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Gauge, List, ListState, Paragraph};
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Item {
    Mic,
    Video,
    Blur,
    Noise,
    Camera,
    MicDev,
    Speaker,
    Volume,
    Tone,
    Mirror,
    SelfView,
    Loopback,
    Quit,
}

type Devs = Vec<(String, String)>; // (id stored in settings, label)

/// Next/previous entry of `list`, where "" (system default) comes first; an unknown current id restarts at the default.
fn cycle(list: &Devs, cur: &str, dir: i32) -> String {
    let ids: Vec<&str> = std::iter::once("").chain(list.iter().map(|(id, _)| id.as_str())).collect();
    let i = ids.iter().position(|id| *id == cur).unwrap_or(0) as i32;
    ids[(i + dir).rem_euclid(ids.len() as i32) as usize].to_string()
}

fn name(list: &Devs, cur: &str) -> String {
    match list.iter().find(|(id, _)| id == cur) {
        _ if cur.is_empty() => "System default".into(),
        Some((_, n)) => n.clone(),
        None => format!("{cur} (missing)"),
    }
}

struct Ui {
    sh: Arc<Shared>,
    items: Vec<Item>,
    cams: Devs,
    mics: Devs,
    spks: Devs,
}

impl Ui {
    fn refresh(&mut self, it: Item) {
        match it {
            Item::Camera => self.cams = video::cameras(),
            Item::MicDev => self.mics = audio::devices(true),
            Item::Speaker => self.spks = audio::devices(false),
            _ => {}
        }
    }

    /// Enter/space (dir 1) and left/right (dir -1/1) on an item. Returns false to hang up.
    fn act(&mut self, it: Item, dir: i32) -> bool {
        let sh = self.sh.clone();
        let mut s = sh.settings.lock().unwrap();
        match it {
            Item::Mic => s.mute_mic ^= true,
            Item::Video => s.video_off ^= true,
            Item::Blur => s.blur ^= true,
            Item::Noise => s.noise ^= true,
            Item::Mirror => s.mirror ^= true,
            Item::SelfView => s.self_view ^= true,
            Item::Loopback => _ = sh.loopback.fetch_xor(true, Relaxed),
            Item::Volume => s.volume = s.volume.saturating_add_signed(dir * 10).min(MAX_VOLUME),
            Item::Tone => sh.tone.store(sh.out_rate.load(Relaxed), Relaxed), // 1 s at the speaker rate
            Item::Camera | Item::MicDev | Item::Speaker => {
                drop(s); // listing devices can take a moment; don't block the audio callbacks
                self.refresh(it);
                let mut s = sh.settings.lock().unwrap();
                let (list, cur) = match it {
                    Item::Camera => (&self.cams, &mut s.camera),
                    Item::MicDev => (&self.mics, &mut s.mic),
                    _ => (&self.spks, &mut s.speaker),
                };
                *cur = cycle(list, cur, dir);
            }
            Item::Quit => return false,
        }
        true
    }

    fn row(&self, it: Item, s: &Settings) -> Line<'static> {
        let on_off = |b: bool| if b { ("on".to_string(), Color::Green) } else { ("off".to_string(), Color::DarkGray) };
        let (label, (value, color), key): (&str, (String, Color), &str) = match it {
            Item::Mic if s.mute_mic => ("Microphone", ("MUTED".into(), Color::Red), "m"),
            Item::Mic => ("Microphone", ("on".into(), Color::Green), "m"),
            Item::Video if s.video_off => ("Video", ("STOPPED".into(), Color::Red), "v"),
            Item::Video => ("Video", ("on".into(), Color::Green), "v"),
            Item::Blur => ("Blur background", on_off(s.blur), "b"),
            Item::Noise => ("Noise suppression", on_off(s.noise), "n"),
            Item::Camera => ("Camera", (name(&self.cams, &s.camera), Color::Cyan), "⏎"),
            Item::MicDev => ("Mic device", (name(&self.mics, &s.mic), Color::Cyan), "⏎"),
            Item::Speaker => ("Speaker", (name(&self.spks, &s.speaker), Color::Cyan), "⏎"),
            Item::Volume => {
                let bar = "█".repeat((s.volume / 10) as usize) + &"·".repeat(((MAX_VOLUME - s.volume) / 10) as usize);
                ("Speaker volume", (format!("{bar} {}%", s.volume), Color::Cyan), "←→")
            }
            Item::Tone => ("Play test sound", ("440 Hz, 1 s".into(), Color::DarkGray), "t"),
            Item::Mirror => ("Mirror my video", on_off(s.mirror), ""),
            Item::SelfView => ("Show self-view", on_off(s.self_view), ""),
            Item::Loopback => ("Hear yourself", on_off(self.sh.loopback.load(Relaxed)), ""),
            Item::Quit if self.sh.offline => ("Save and quit", (String::new(), Color::Reset), "q"),
            Item::Quit => ("Hang up", (String::new(), Color::Reset), "q"),
        };
        Line::from(vec![
            Span::raw(format!(" {label:<18} ")),
            Span::styled(value, Style::new().fg(color)),
            Span::styled(if key.is_empty() { String::new() } else { format!("  [{key}]") }, Style::new().fg(Color::DarkGray)),
        ])
    }
}

/// Counter snapshot for per-second rates.
#[derive(Clone, Copy, Default)]
struct Snap {
    at: Option<Instant>,
    tx: u64,
    rx: u64,
    frames: u64,
    arx: u64,
    lost: u64,
}

/// Loss in percent of the audio packets expected in an interval.
fn loss_pct(lost: u64, received: u64) -> f64 {
    if lost + received == 0 { 0.0 } else { 100.0 * lost as f64 / (lost + received) as f64 }
}

fn stats_text(sh: &Shared, prev: &Snap, now: &Snap) -> Vec<Line<'static>> {
    // rates need two snapshots; before that, show a placeholder rather than totals since start
    let dt = now.at.zip(prev.at).map(|(a, b)| (a - b).as_secs_f64().max(0.001));
    let rate = |f: &dyn Fn(f64) -> String| dt.map_or("…".to_string(), f);
    let kbit = |a: u64, b: u64| rate(&|dt| format!("{:.0} kbit/s", (a - b) as f64 * 8.0 / 1000.0 / dt));
    let st = &sh.stats;
    let heard = match st.last_rx_ms.load(Relaxed) {
        0 => "never".into(),
        t => format!("{:.1} s ago", (sh.start.elapsed().as_millis() as u64).saturating_sub(t) as f64 / 1000.0),
    };
    let pct = |a: &std::sync::atomic::AtomicU32| format!("{:.1} %", 100.0 * f32::from_bits(a.load(Relaxed)));
    let bwe = match st.bwe_bps.load(Relaxed) {
        0 => "…".into(),
        b => format!("{} kbit/s, video {}", b / 1000, st.video_bps.load(Relaxed) / 1000),
    };
    let row = |k: &str, v: String| Line::from(vec![Span::styled(format!(" {k:<10}"), Style::new().fg(Color::DarkGray)), Span::raw(v)]);
    vec![
        row("link", st.link.lock().unwrap().clone()),
        row("send", kbit(now.tx, prev.tx)),
        row("receive", kbit(now.rx, prev.rx)),
        row("estimate", bwe),
        row("rtt", match st.rtt_ms.load(Relaxed) { 0 => "…".into(), ms => format!("{ms} ms") }),
        row("loss", format!("in {} out {}", pct(&st.loss_in), pct(&st.loss_out))),
        row("video in", rate(&|dt| format!("{:.1} fps", (now.frames - prev.frames) as f64 / dt))),
        row("audio", rate(&|_| format!("{:.1} % lost, {} ms buf", loss_pct(now.lost - prev.lost, now.arx - prev.arx), st.buffer_ms.load(Relaxed)))),
        row("heard", heard),
    ]
}

/// Runs until q/Esc/Ctrl-C or until `sh.quit` is set (window closed); always restores the terminal.
/// Any panic in any thread restores the terminal and exits 101, like `--audio`.
pub fn run(sh: Arc<Shared>, title: String) -> std::io::Result<()> {
    // raw mode + alt screen + a panic hook that restores them; without a menu the call cannot be ended, so hang up
    let mut term = ratatui::try_init().inspect_err(|_| sh.quit.store(true, Relaxed))?;
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // ratatui's restore leaves the cursor hidden, and exit() skips Terminal's drop that would show it
        let _ = ratatui::crossterm::execute!(std::io::stdout(), ratatui::crossterm::cursor::Show);
        hook(info);
        std::process::exit(101);
    }));
    struct Restore<'a>(&'a Shared);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            ratatui::restore();
            self.0.tui.store(false, Relaxed);
            self.0.quit.store(true, Relaxed); // every way out of the menu hangs up, I/O errors included
        }
    }
    sh.tui.store(true, Relaxed);
    let _restore = Restore(&sh);
    use Item::*;
    let mut items = vec![Mic, Video, Blur, Noise, Camera, MicDev, Speaker, Volume, Tone, Mirror];
    items.extend(if sh.offline { [Loopback, Quit] } else { [SelfView, Quit] });
    let mut ui = Ui { sh: sh.clone(), items, cams: video::cameras(), mics: audio::devices(true), spks: audio::devices(false) };
    let mut state = ListState::default().with_selected(Some(0));
    let mut saved = sh.settings.lock().unwrap().clone();
    let (mut prev, mut now) = (Snap::default(), Snap::default());
    while !sh.quit.load(Relaxed) {
        if now.at.is_none_or(|t| t.elapsed() >= Duration::from_secs(1)) {
            let st = &sh.stats;
            prev = now;
            now = Snap {
                at: Some(Instant::now()),
                tx: st.tx_bytes.load(Relaxed),
                rx: st.rx_bytes.load(Relaxed),
                frames: st.rx_frames.load(Relaxed),
                arx: st.audio_rx.load(Relaxed),
                lost: st.audio_lost.load(Relaxed),
            };
        }
        let s = sh.settings.lock().unwrap().clone();
        if s != saved {
            // save on every change, so a crash or a closed terminal keeps the choice
            if let Err(e) = config::save(&s) {
                sh.log(format!("cannot save settings: {e}"));
            }
            saved = s.clone();
        }
        let rows: Vec<Line> = ui.items.iter().map(|&it| ui.row(it, &s)).collect();
        let rms = f32::from_bits(sh.mic_level.load(Relaxed));
        let lvl = ((20.0 * rms.log10() + 60.0) / 60.0).clamp(0.0, 1.0) as f64; // -60..0 dBFS
        let status = sh.status.lock().unwrap().clone();
        *state.offset_mut() = 0; // re-scroll from the top each frame, so growing the terminal shows every row again
        term.draw(|f| {
            let [main, stat, foot] = Layout::vertical([Constraint::Min(5), Constraint::Length(1), Constraint::Length(1)]).areas(f.area());
            let [menu, side] = Layout::horizontal([Constraint::Min(30), Constraint::Length(36)]).areas(main);
            let [meter, info] = Layout::vertical([Constraint::Length(3), Constraint::Min(3)]).areas(side);
            let list = List::new(rows)
                .block(Block::bordered().title(format!(" {title} ")).title_style(Style::new().bold()))
                .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
            f.render_stateful_widget(list, menu, &mut state);
            let (label, color) = if s.mute_mic { ("muted (not sent)", Color::Red) } else { ("", Color::Green) };
            let gauge = Gauge::default().block(Block::bordered().title(" mic level ")).gauge_style(color).ratio(lvl).label(label);
            f.render_widget(gauge, meter);
            let body = if sh.offline {
                ["Not in a call: nothing is sent.", "", "The window previews your camera", "with blur/mirror applied.", "", "Hear yourself plays your mic", "on the speaker: use headphones."]
                    .map(Line::from)
                    .to_vec()
            } else {
                stats_text(&sh, &prev, &now)
            };
            f.render_widget(Paragraph::new(body).block(Block::bordered().title(if sh.offline { " test " } else { " call stats " })), info);
            f.render_widget(Paragraph::new(format!(" {status}")).fg(Color::Yellow), stat);
            let q = if sh.offline { " q quit" } else { " q hang up" };
            let keys = "  ↑↓ move  ⏎ select  ←→ set  m mic  v video  b blur  n noise  t test";
            f.render_widget(Paragraph::new(format!("{q}{keys}")).fg(Color::DarkGray), foot);
        })?;
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(k) = event::read()? else { continue };
        if k.kind == KeyEventKind::Release {
            continue;
        }
        let sel = ui.items[state.selected().unwrap_or(0).min(ui.items.len() - 1)];
        let go_on = match k.code {
            KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => false, // raw mode: no SIGINT
            KeyCode::Char('q' | 'Q') | KeyCode::Esc => false,
            KeyCode::Down | KeyCode::Char('j') => {
                state.select(Some((state.selected().unwrap_or(0) + 1) % ui.items.len()));
                true
            }
            KeyCode::Up | KeyCode::Char('k') => {
                state.select(Some((state.selected().unwrap_or(0) + ui.items.len() - 1) % ui.items.len()));
                true
            }
            KeyCode::Enter | KeyCode::Char(' ') => ui.act(sel, 1),
            KeyCode::Right | KeyCode::Char('l') if matches!(sel, Volume | Camera | MicDev | Speaker) => ui.act(sel, 1),
            KeyCode::Left | KeyCode::Char('h') if matches!(sel, Volume | Camera | MicDev | Speaker) => ui.act(sel, -1),
            KeyCode::Char('m') => ui.act(Mic, 1),
            KeyCode::Char('v') => ui.act(Video, 1),
            KeyCode::Char('b') => ui.act(Blur, 1),
            KeyCode::Char('n') => ui.act(Noise, 1),
            KeyCode::Char('t') => ui.act(Tone, 1),
            _ => true,
        };
        if !go_on {
            break;
        }
    }
    let s = sh.settings.lock().unwrap().clone();
    if s != saved {
        config::save(&s).map_err(std::io::Error::other)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_cycle() {
        let l: Devs = vec![("a".into(), "A".into()), ("b".into(), "B".into())];
        assert_eq!(cycle(&l, "", 1), "a");
        assert_eq!(cycle(&l, "b", 1), "");
        assert_eq!(cycle(&l, "", -1), "b");
        assert_eq!(cycle(&l, "gone", 1), "a");
        assert_eq!(cycle(&Vec::new(), "", 1), "");
        assert_eq!(name(&l, "b"), "B");
        assert_eq!(name(&l, ""), "System default");
    }

    #[test]
    fn loss() {
        assert_eq!(loss_pct(0, 0), 0.0);
        assert_eq!(loss_pct(1, 49), 2.0);
    }
}
