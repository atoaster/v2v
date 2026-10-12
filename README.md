# v2v

Peer-to-peer video call from the command line.

```sh
cargo build --release
# on machine A (talking to B)
./target/release/v2v <ip_of_B>
# on machine B (talking to A)
./target/release/v2v <ip_of_A>
# or B just answers: waits for the first caller and replies to wherever its packets come from
./target/release/v2v
```

Answering (no address) is what to use when the caller's packets arrive NATed, e.g. a WireGuard peer
masqueraded by the router. Flags still apply: `v2v --audio` answers with an audio-only call.

`v2v -h` prints the exact command the other side should run, one line per local interface address.

A window opens showing the other side's camera (gray until video arrives) with your own camera
picture-in-picture in the bottom-right corner, and plays their microphone. The terminal turns into a
call menu:

| Key | Action |
| --- | --- |
| `↑`/`↓` or `j`/`k` | move |
| `Enter`/`Space` | toggle the item, or pick the next camera / mic / speaker |
| `←`/`→` | speaker volume (0-150 %), previous/next device |
| `m` / `v` | mute mic / stop video (the camera is closed; the peer keeps your last frame) |
| `b` / `n` | background blur / noise suppression |
| `t` | play a 1 s test tone on the speaker |
| `q`, `Esc`, `Ctrl-C` | hang up |

The menu also toggles mirroring of your own preview (never of what is sent) and the self-view, and shows
call stats: send/receive rate, received video fps, audio packet loss, playback buffer and when the peer was last heard.
Device changes apply live without dropping the call. Close the window or press Esc in it to hang up too.
Without a terminal (stdin/stdout redirected) only the window opens, as before.

`v2v config` opens the same menu without calling anyone, plus a window previewing your camera with
the current effects, so you can try devices before a call: the mic meter, the test tone and
"Hear yourself" (plays your mic on the speaker; use headphones). `q` saves and quits.

Settings are saved on every change to `$XDG_CONFIG_HOME/v2v/config` (default `~/.config/v2v/config`),
a plain `key=value` file, and every mode uses them, including `--tplay` and `--audio`
(devices, volume, noise suppression, blur). Muting or stopping video is remembered, so the next windowed call
starts that way; `--tplay`, `--audio` and window-only runs have no menu to undo it, so they always start live.

`v2v 127.0.0.1` calls yourself, which is a handy self-test (use headphones, or the mic feeds back into the speakers).

`v2v --tplay <ip>` shows the video as ASCII art in the terminal with [tplay](https://github.com/maxcurzi/tplay)
instead of a window (audio still plays). It needs `tplay` (`cargo install tplay`) and `ffprobe` on PATH.
The terminal stays as-is until the first frame arrives; press `q` to hang up.

`v2v --audio <ip>` is an audio-only call: no camera and no window. The terminal shows a live spectrum
analyzer of the other side's voice (log-spaced bars, 50 Hz to 8 kHz) and a meter for your own mic; press
`q` or `Esc` to hang up. Both sides can use `--audio`; if the other side runs in video mode it still hears
you but only sees its gray waiting screen, and its video is ignored on your side. Needs a real terminal (not a pipe).

Background blur finds you in each frame with Google's MediaPipe selfie segmentation model (embedded, run on
the CPU with [tract](https://github.com/sonos/tract)) and blurs everything else, in what is sent and in your preview.
It costs about 16 ms per frame on one core (the menu's status line shows the measured time); the model loads the
first time blur is turned on. Objects next to you, such as a chair back, can stay partly sharp.

Noise suppression runs RNNoise ([nnnoiseless](https://github.com/jneem/nnnoiseless)) on your microphone before
sending: steady background noise (fans, hum, hiss) is removed, voice passes. It adds 10 ms of latency (always,
so toggling it mid-call does not glitch) and well under 1% of one core.

Video is capped at about 15 fps / 640x480 JPEG, roughly 1-3 Mbit/s each way; slower or lossy links will drop frames.

## Network

Both sides send and receive on **UDP port 47000**. That port has to be reachable on each machine:
same LAN/VPN, or a port forward on each router plus a firewall rule. There is no NAT traversal,
no encryption and no authentication beyond ignoring packets that do not come from the given IP.

## Third-party model

`assets/selfie_segmenter_landscape.onnx`: MediaPipe selfie_segmenter_landscape © Google, Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>), converted from TFLite to ONNX.

## Linux build deps

`libpipewire-0.3-dev` and `libasound2-dev` (audio; PipeWire is used when running, ALSA otherwise), `clang`/`libclang-dev` and Linux kernel headers (V4L2 camera bindings via bindgen), X11/Wayland dev libs for the window.
