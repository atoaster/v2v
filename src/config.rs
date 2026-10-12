//! Persisted settings: a tiny `key=value` file at $XDG_CONFIG_HOME/v2v/config (or ~/.config/v2v/config).

use std::path::PathBuf;

/// Device fields hold "" for the system default (first usable camera, default cpal device).
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub camera: String,  // nokhwa human name
    pub mic: String,     // cpal DeviceId string, e.g. "pipewire:alsa_input.usb-..."
    pub speaker: String, // cpal DeviceId string
    pub mute_mic: bool,
    pub video_off: bool,
    pub blur: bool,
    pub noise: bool,
    pub volume: u32, // percent, 0..=150
    pub mirror: bool,
    pub self_view: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            camera: String::new(),
            mic: String::new(),
            speaker: String::new(),
            mute_mic: false,
            video_off: false,
            blur: false,
            noise: false,
            volume: 100,
            mirror: true,
            self_view: true,
        }
    }
}

pub const MAX_VOLUME: u32 = 150;

impl Settings {
    /// Unknown keys and bad values are ignored, so an old or hand-edited file never stops a call.
    pub fn parse(text: &str) -> Settings {
        let mut s = Settings::default();
        for line in text.lines().map(str::trim).filter(|l| !l.starts_with('#')) {
            let Some((k, v)) = line.split_once('=') else { continue };
            let (k, v) = (k.trim(), v.trim());
            let flag = |b: &mut bool| {
                if let Ok(x) = v.parse() {
                    *b = x;
                }
            };
            match k {
                "camera" => s.camera = v.into(),
                "mic" => s.mic = v.into(),
                "speaker" => s.speaker = v.into(),
                "mute_mic" => flag(&mut s.mute_mic),
                "video_off" => flag(&mut s.video_off),
                "blur" => flag(&mut s.blur),
                "noise" => flag(&mut s.noise),
                "mirror" => flag(&mut s.mirror),
                "self_view" => flag(&mut s.self_view),
                "volume" => s.volume = v.parse::<u32>().map_or(s.volume, |x| x.min(MAX_VOLUME)),
                _ => {}
            }
        }
        s
    }

    pub fn serialize(&self) -> String {
        // device names never contain newlines in practice; strip them so one bad name can't break the file
        let one = |v: &str| v.replace(['\n', '\r'], " ");
        format!(
            "# v2v settings, rewritten by `v2v config` and the in-call menu\n\
             camera={}\nmic={}\nspeaker={}\nmute_mic={}\nvideo_off={}\nblur={}\nnoise={}\nvolume={}\nmirror={}\nself_view={}\n",
            one(&self.camera),
            one(&self.mic),
            one(&self.speaker),
            self.mute_mic,
            self.video_off,
            self.blur,
            self.noise,
            self.volume,
            self.mirror,
            self.self_view,
        )
    }
}

pub fn path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("v2v").join("config"))
}

/// Missing file means defaults.
pub fn load() -> Settings {
    path().and_then(|p| std::fs::read_to_string(p).ok()).map_or_else(Settings::default, |t| Settings::parse(&t))
}

pub fn save(s: &Settings) -> Result<(), String> {
    let p = path().ok_or("no HOME or XDG_CONFIG_HOME")?;
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    // write-then-rename so a crash mid-write never leaves a truncated config
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, s.serialize()).and_then(|_| std::fs::rename(&tmp, &p)).map_err(|e| format!("{}: {e}", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let s = Settings {
            camera: "UVC Camera (046d:0825)".into(),
            mic: "pipewire:alsa_input.usb-x=y".into(),
            speaker: "alsa:plughw:CARD=StreamX".into(),
            mute_mic: true,
            video_off: true,
            blur: true,
            noise: true,
            volume: 70,
            mirror: false,
            self_view: false,
        };
        assert_eq!(Settings::parse(&s.serialize()), s);
        assert_eq!(Settings::parse(&Settings::default().serialize()), Settings::default());
    }

    #[test]
    fn tolerant_parse() {
        let s = Settings::parse("garbage\nvolume=999\nmirror=maybe\nblur = true\n# mute_mic=true\nnew_key=1\n");
        assert_eq!(s, Settings { volume: MAX_VOLUME, blur: true, ..Settings::default() });
    }
}
