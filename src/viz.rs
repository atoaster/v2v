//! `--audio` mode: full-screen spectrum of the peer's voice.

use std::collections::VecDeque;
use std::io::Write;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Sample rate of the tap: received audio is downsampled to this for the spectrum.
pub const RATE: u32 = 16_000;
pub const FFT_N: usize = 1024; // visualizer window, 64 ms at 16 kHz

pub type Tap = Arc<Mutex<(VecDeque<f32>, Option<Instant>)>>; // last FFT_N received samples, when received

/// In-place iterative radix-2 FFT; the length must be a power of two.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wi, wr) = (ang * k as f32).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                (re[b], im[b]) = (re[a] - xr, im[a] - xi);
                (re[a], im[a]) = (re[a] + xr, im[a] + xi);
            }
        }
        len <<= 1;
    }
}

/// Level 0..1 (-80..0 dBFS) per column, columns log-spaced from 50 Hz to Nyquist (8 kHz).
fn spectrum(samples: &[f32], cols: usize) -> Vec<f32> {
    let n = samples.len();
    let hann = |i: usize| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
    let mut re: Vec<f32> = samples.iter().enumerate().map(|(i, x)| x * hann(i)).collect();
    let mut im = vec![0.0; n];
    fft(&mut re, &mut im);
    // shortcut: below ~360 Hz neighbouring columns share one 15.6 Hz bin and show as steps; interpolate bins if it matters
    let bin = |c: usize| 50.0 * (RATE as f32 / 100.0).powf(c as f32 / cols as f32) * n as f32 / RATE as f32;
    (0..cols)
        .map(|c| {
            let a = bin(c) as usize;
            let b = (bin(c + 1) as usize).max(a + 1).min(n / 2);
            let mag = (a..b).map(|k| re[k].hypot(im[k])).fold(0.0, f32::max);
            // a full-scale sine peaks at n/4 after the Hann window
            ((20.0 * (mag * 4.0 / n as f32).log10() + 80.0) / 80.0).clamp(0.0, 1.0)
        })
        .collect()
}

/// Live spectrum of the received audio until q/Esc/Ctrl-C; the terminal is restored on return or panic.
pub fn run_audio(peer: IpAddr, tap: Tap, mic: &AtomicU32, quit: &AtomicBool) -> std::io::Result<()> {
    use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    use ratatui::crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
    use ratatui::crossterm::{cursor, execute, queue, terminal};
    fn restore() {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(std::io::stdout(), ResetColor, cursor::Show, terminal::LeaveAlternateScreen);
    }
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            restore();
        }
    }
    let hook = std::panic::take_hook();
    // any panic (also in the network thread) hangs up, with the message readable on the normal screen
    std::panic::set_hook(Box::new(move |info| {
        restore();
        hook(info);
        std::process::exit(101);
    }));
    terminal::enable_raw_mode()?;
    let _guard = Guard;
    let mut stdout = std::io::stdout();
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;
    // shortcut: sizes by char count, so terminals drawing ambiguous-width chars (█▁·—…) as 2 cells garble rows; use unicode-width if needed
    // one string of exactly `w` chars, so every row overwrites the previous frame and never wraps
    let fit = |s: &str, w: usize| s.chars().chain(std::iter::repeat(' ')).take(w).collect::<String>();
    let head = format!(" v2v audio call — {peer}   q/Esc to hang up");
    const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut bars: Vec<f32> = Vec::new();
    let mut out = Vec::new();
    while !quit.load(Ordering::Relaxed) {
        let (w, h) = terminal::size()?;
        let (w, h, bh) = (w as usize, h, h.saturating_sub(2)); // rows: header, mic meter, bars
        let (samples, fresh) = {
            let t = tap.lock().unwrap();
            let mut v = vec![0.0; FFT_N - t.0.len()];
            v.extend(&t.0);
            (v, t.1.is_some_and(|i| i.elapsed() < Duration::from_millis(500)))
        };
        let target = if fresh { spectrum(&samples, w) } else { vec![0.0; w] };
        bars.resize(w, 0.0);
        for (b, t) in bars.iter_mut().zip(target) {
            *b = t.max(*b - 0.025); // instant rise, slow fall
        }
        out.clear();
        if h > 0 {
            queue!(out, cursor::MoveTo(0, 0), SetForegroundColor(Color::Cyan), Print(fit(&head, w)))?;
        }
        if h > 1 {
            let rms = f32::from_bits(mic.load(Ordering::Relaxed));
            let lvl = ((20.0 * rms.log10() + 60.0) / 60.0).clamp(0.0, 1.0); // -60..0 dBFS
            let m = w.saturating_sub(5);
            let filled = (lvl * m as f32).round() as usize;
            let meter = format!(" mic {}{}", "█".repeat(filled), "·".repeat(m - filled));
            queue!(out, cursor::MoveTo(0, 1), SetForegroundColor(Color::Green), Print(fit(&meter, w)))?;
        }
        for r in 0..bh {
            let y = (bh - 1 - r) as usize; // cell height above the bottom row
            let frac = (y + 1) as f32 / bh as f32;
            let mut color = if frac <= 0.5 { Color::Green } else if frac <= 0.8 { Color::Yellow } else { Color::Red };
            let mut line: String = bars
                .iter()
                .map(|b| BLOCKS[((b * bh as f32 * 8.0) as usize).saturating_sub(y * 8).min(8)])
                .collect();
            if !fresh && r == bh / 2 {
                let msg = "waiting for audio…";
                line = fit(&format!("{}{msg}", " ".repeat(w.saturating_sub(msg.chars().count()) / 2)), w);
                color = Color::DarkGrey;
            }
            queue!(out, cursor::MoveTo(0, r + 2), SetForegroundColor(color), Print(line))?;
        }
        stdout.write_all(&out)?;
        stdout.flush()?;
        // fixed deadline so key repeat or resize bursts don't speed up the frame rate (and bar decay)
        let until = Instant::now() + Duration::from_millis(33);
        while event::poll(until.saturating_duration_since(Instant::now()))? {
            if let Event::Key(k) = event::read()?
                && k.kind != KeyEventKind::Release
                && (matches!(k.code, KeyCode::Char('q' | 'Q') | KeyCode::Esc)
                    || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL)))
            {
                return Ok(());
            }
        }
    }
    Ok(()) // the peer hung up
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 37 % 11) as f32 - 5.0) / 5.0).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; 64]);
        fft(&mut re, &mut im);
        for k in 0..64 {
            let w = |i: usize| -2.0 * std::f32::consts::PI * (k * i) as f32 / 64.0;
            let dr: f32 = x.iter().enumerate().map(|(i, v)| v * w(i).cos()).sum();
            let di: f32 = x.iter().enumerate().map(|(i, v)| v * w(i).sin()).sum();
            assert!((re[k] - dr).abs() < 1e-3 && (im[k] - di).abs() < 1e-3, "bin {k}");
        }
    }

    #[test]
    fn spectrum_peaks_at_sine() {
        // 1 kHz = bin 64 at 16 kHz / 1024
        let x: Vec<f32> = (0..FFT_N).map(|i| (2.0 * std::f32::consts::PI * 64.0 * i as f32 / FFT_N as f32).sin()).collect();
        let s = spectrum(&x, 80);
        let peak = (0..80).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
        let hz = |c: usize| 50.0 * 160f32.powf(c as f32 / 80.0);
        assert!(hz(peak) <= 1000.0 && 1000.0 < hz(peak + 2), "peak col {peak}");
        assert!(s[peak] > 0.95 && s[0] < 0.3 && s[79] < 0.3, "{s:?}");
        assert!(spectrum(&x, 0).is_empty() && spectrum(&x, 1).len() == 1);
    }
}
