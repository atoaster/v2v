# v2v

Peer-to-peer video call from the command line.

```sh
cargo build --release
# on machine A (talking to B)
./target/release/v2v <ip_of_B>
# on machine B (talking to A)
./target/release/v2v <ip_of_A>
```

`v2v -h` prints the exact command the other side should run, one line per local interface address.

A window opens showing the other side's camera (gray until video arrives) and plays their microphone.
Close the window or press Esc to hang up. `v2v 127.0.0.1` calls yourself, which is a handy self-test (use headphones, or the mic feeds back into the speakers).

`v2v --tplay <ip>` shows the video as ASCII art in the terminal with [tplay](https://github.com/maxcurzi/tplay)
instead of a window (audio still plays). It needs `tplay` (`cargo install tplay`) and `ffprobe` on PATH.
The terminal stays as-is until the first frame arrives; press `q` to hang up.

Video is capped at about 15 fps / 640x480 JPEG, roughly 1-3 Mbit/s each way; slower or lossy links will drop frames.

## Network

Both sides send and receive on **UDP port 47000**. That port has to be reachable on each machine:
same LAN/VPN, or a port forward on each router plus a firewall rule. There is no NAT traversal,
no encryption and no authentication beyond ignoring packets that do not come from the given IP.

## Linux build deps

`libasound2-dev` (ALSA), `clang`/`libclang-dev` and Linux kernel headers (V4L2 camera bindings via bindgen), X11/Wayland dev libs for the window.
