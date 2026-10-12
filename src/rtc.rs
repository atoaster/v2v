//! The call: a WebRTC session (str0m, sans-IO) on our own UDP socket, port 47000.
//!
//! One thread owns the socket. It does the in-band signalling (see `signal`), feeds str0m every
//! datagram and timeout, sends what str0m's pacer releases, writes our encoded media and hands the
//! peer's media on: H.264 to the decoder thread, Opus decoded (with FEC/PLC) to the speaker.

use crate::Shared;
use crate::audio::{OpusDec, OpusEnc, PKT, Pcm, Player};
use crate::signal::{self, Kind, Reassembler};
use crate::video::Encoded;
use std::hash::BuildHasher;
use std::io::ErrorKind;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::time::{Duration, Instant};
use str0m::bwe::{Bitrate, BweKind};
use str0m::change::{SdpAnswer, SdpOffer, SdpPendingOffer};
use str0m::format::Codec;
use str0m::media::{Direction, Frequency, KeyframeRequestKind, MediaData, MediaKind, MediaTime, Mid};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

pub const PORT: u16 = 47000;
const RESEND: Duration = Duration::from_millis(300);
/// The longest we sit in recv_from before looking at the media channels again (adds at most this much latency).
const POLL: Duration = Duration::from_millis(5);
/// ICE disconnected this long: give up on the session and renegotiate (the peer may have restarted).
const RECONNECT: Duration = Duration::from_secs(5);
const BWE_START: u64 = 1_000_000;
const BWE_MIN: u64 = 150_000;
// 640x480 H.264 looks no better above this, and a noisy camera would otherwise fill any estimate
const BWE_MAX: u64 = 2_000_000;
const AUDIO_BPS: u64 = 40_000; // Opus 32 kbit/s plus RTP overhead, kept out of the video budget
const PLI_GAP: Duration = Duration::from_millis(300);

/// Counters and gauges the TUI shows; written by the network, decoder and audio threads.
#[derive(Default)]
pub struct Stats {
    pub tx_bytes: AtomicU64,
    pub rx_bytes: AtomicU64,
    pub rx_frames: AtomicU64,
    pub audio_rx: AtomicU64,
    pub audio_lost: AtomicU64,
    pub buffer_ms: AtomicU32,
    pub last_rx_ms: AtomicU64, // since Shared::start, 0 = never
    pub rtt_ms: AtomicU32,     // 0 = not measured yet
    pub bwe_bps: AtomicU64,    // str0m's send-side estimate, 0 = none yet
    pub video_bps: AtomicU32,  // encoder target derived from the estimate
    pub loss_out: AtomicU32,   // f32 bits: fraction lost on the way to the peer (its receiver reports)
    pub loss_in: AtomicU32,    // f32 bits: fraction lost on the way to us
    pub link: std::sync::Mutex<String>, // connection state for the stats panel
}

/// Media to send (from the camera thread and the mic) and where received media goes.
pub struct Io {
    pub video_in: Option<Receiver<Encoded>>,
    pub pcm_in: Receiver<Pcm>,
    pub video_out: Option<SyncSender<Arc<[u8]>>>, // None in --audio mode: the peer's video is dropped
    pub player: Player,
}

#[derive(Default)]
enum Sig {
    /// Sending HELLO until we know the peer's nonce (or get its OFFER).
    #[default]
    Hello,
    /// We won the tie-break: resend the OFFER until a full ANSWER arrives.
    Offering(Vec<Vec<u8>>, Option<SdpPendingOffer>),
    /// We answered: re-send the cached ANSWER whenever the OFFER shows up again.
    Answered(Vec<Vec<u8>>),
    Done,
}

struct Session {
    sh: Arc<Shared>,
    sock: UdpSocket,
    peer: SocketAddr,
    /// the configured peer (or first caller); HELLOs go back here after a reset, a NAT mapping may be gone
    dial: SocketAddr,
    /// our one ICE host candidate, also the destination of every datagram we hand str0m
    local: SocketAddr,
    rtc: Rtc,
    nonce: u64,
    peer_nonce: Option<u64>,
    sig: Sig,
    reasm: Reassembler,
    next_sig: Instant,
    desired: u64,
    vmid: Option<Mid>,
    amid: Option<Mid>,
    connected: bool,
    lost_since: Option<Instant>,
    last_pli: Option<Instant>,
    self_call: bool,
    io: Io,
    opus_enc: OpusEnc,
    opus_dec: OpusDec,
    audio_n: u64,
    last_audio: Option<u64>,
}

pub fn random() -> u64 {
    std::collections::hash_map::RandomState::new().hash_one(Instant::now())
}

/// Runs the call until hang-up, then tells the peer. `peer` is where the peer's packets come from (its address after SNAT).
pub fn run(sock: UdpSocket, peer: SocketAddr, nonce: u64, sh: Arc<Shared>, io: Io) {
    call(sock, peer, nonce, sh, io, true);
}

/// `bye`: false only in tests, to look like a crashed peer.
fn call(sock: UdpSocket, peer: SocketAddr, nonce: u64, sh: Arc<Shared>, io: Io, bye: bool) {
    let Some(mut s) = Session::new(sock, peer, sh, io) else { return };
    s.nonce = nonce; // later resets pick fresh ones
    s.run();
    if bye && !s.sh.peer_bye.load(Relaxed) {
        for _ in 0..3 {
            let _ = s.sock.send_to(&signal::bye(s.nonce), s.peer);
        }
    }
}

/// None = duplicate or late; otherwise how many packets are missing before `seq`.
fn audio_gap(last: Option<u64>, seq: u64) -> Option<u64> {
    last.map_or(Some(0), |l| (seq > l).then(|| seq - l - 1))
}

impl Session {
    fn new(sock: UdpSocket, peer: SocketAddr, sh: Arc<Shared>, io: Io) -> Option<Self> {
        let codecs = OpusEnc::new().and_then(|e| Ok((e, OpusDec::new()?)));
        let (opus_enc, opus_dec) = match codecs {
            Ok(c) => c,
            Err(e) => {
                sh.log(e);
                return None;
            }
        };
        let mut s = Session {
            rtc: Rtc::new(Instant::now()), // replaced by reset()
            sh,
            sock,
            peer,
            dial: peer,
            local: peer,
            nonce: 0,
            peer_nonce: None,
            sig: Sig::Hello,
            reasm: Reassembler::default(),
            next_sig: Instant::now(),
            desired: 0,
            vmid: None,
            amid: None,
            connected: false,
            lost_since: None,
            last_pli: None,
            self_call: false,
            io,
            opus_enc,
            opus_dec,
            audio_n: 0,
            last_audio: None,
        };
        s.reset("calling");
        Some(s)
    }

    /// A fresh str0m session and signalling state with a new nonce, so the peer notices and resets too.
    fn reset(&mut self, why: &str) {
        self.peer = self.dial;
        self.new_rtc();
        self.nonce = random();
        self.peer_nonce = None;
        self.next_sig = Instant::now();
        self.set_link(format!("{why}…"));
    }

    /// A fresh str0m session in the HELLO state, with our host candidate.
    fn new_rtc(&mut self) {
        let mut cfg = RtcConfig::new()
            .clear_codecs()
            .enable_opus(true, false)
            .enable_bwe(Some(Bitrate::bps(BWE_START)))
            .set_stats_interval(Some(Duration::from_secs(1)))
            // str0m waits for 15 packets (300 ms) on an audio gap by default; Opus conceals it instead
            .set_reordering_size_audio(2)
            .set_reordering_timeout_audio(Some(Duration::from_millis(40)))
            .set_reordering_timeout_video(Some(Duration::from_millis(500)));
        // one constrained-baseline, packetization-mode=1 type (+RTX 109 for NACK) keeps the SDP small
        // str0m defaults (RTO up to 3 s, 9 retries) take ~20 s to notice a dead peer; this is ~5 s
        cfg.set_max_stun_rto(Duration::from_secs(1));
        cfg.set_max_stun_retransmits(6);
        cfg.codec_config().add_h264(108.into(), Some(109.into()), true, 0x42e01f);
        self.rtc = cfg.build(Instant::now());
        self.desired = 0; // set from what we send, in run()
        // one host candidate: the address the kernel uses toward the peer. Every interface would put
        // docker/VPN/loopback addresses in the SDP, and a 0.0.0.0 socket cannot tell them apart anyway.
        // Looked up per session, so a reconnect picks up a changed local address.
        let local = self.route();
        self.local = local;
        match Candidate::host(local, "udp") {
            Ok(c) => _ = self.rtc.add_local_candidate(c),
            Err(e) => self.sh.log(format!("no local ICE candidate for {local}: {e}")),
        }
        (self.sig, self.reasm) = (Sig::Hello, Reassembler::default());
        (self.vmid, self.amid, self.connected, self.lost_since, self.last_audio) = (None, None, false, None, None);
        let st = &self.sh.stats;
        st.video_bps.store((BWE_START - AUDIO_BPS) as u32, Relaxed);
        // no stale numbers next to "reconnecting" (the TUI shows 0 as unknown)
        st.rtt_ms.store(0, Relaxed);
        st.bwe_bps.store(0, Relaxed);
        st.loss_in.store(0, Relaxed);
        st.loss_out.store(0, Relaxed);
    }

    fn set_link(&self, s: String) {
        *self.sh.stats.link.lock().unwrap() = s;
    }

    fn port(&self) -> u16 {
        self.sock.local_addr().map_or(PORT, |a| a.port())
    }

    /// Our address as the kernel routes toward the peer (symmetric routing assumed; IP_PKTINFO would be exact).
    fn route(&self) -> SocketAddr {
        let to = self.peer.ip();
        let ip = self.sock.local_addr().ok().map(|a| a.ip()).filter(|ip| !ip.is_unspecified()).unwrap_or_else(|| {
            let any = SocketAddr::new(if to.is_ipv4() { [0u8; 4].into() } else { [0u16; 8].into() }, 0);
            let s = UdpSocket::bind(any).and_then(|s| s.connect((to, PORT)).map(|_| s));
            s.and_then(|s| s.local_addr()).map_or(to, |a| a.ip())
        });
        SocketAddr::new(ip, self.port())
    }

    fn run(&mut self) {
        let mut buf = vec![0u8; 2048];
        let mut timeout = self.drain();
        while !self.sh.quit.load(Relaxed) {
            let now = Instant::now();
            if now >= timeout {
                timeout = self.input(Input::Timeout(now));
            }
            if now >= self.next_sig {
                self.send_signal();
                self.next_sig = now + RESEND;
            }
            // ICE alone needs >15 s to declare a dead peer (is crate's RECENT_BINDING_REQUEST); STUN keepalives
            // (max RTO 1 s) mean a live peer is never silent for RECONNECT
            let silent = self.sh.start.elapsed().as_millis() as u64 > self.sh.stats.last_rx_ms.load(Relaxed) + RECONNECT.as_millis() as u64;
            if self.lost_since.is_some_and(|t| t.elapsed() > RECONNECT) || (self.connected && silent) {
                self.reset("connection lost, reconnecting");
                timeout = self.drain();
            }
            while let Some(f) = self.io.video_in.as_ref().and_then(|r| r.try_recv().ok()) {
                self.write_video(f);
                timeout = self.drain();
            }
            while let Ok(p) = self.io.pcm_in.try_recv() {
                self.write_audio(&p);
                timeout = self.drain();
            }
            if self.sh.want_keyframe.load(Relaxed) && self.request_keyframe() {
                timeout = self.drain();
            }
            // str0m probes toward the desired rate: only ask for video bandwidth while video is on
            let video = self.io.video_in.is_some() && !self.sh.settings.lock().unwrap().video_off;
            let desired = if video { BWE_MAX } else { AUDIO_BPS };
            if desired != self.desired {
                self.desired = desired;
                self.rtc.bwe().set_desired_bitrate(Bitrate::bps(desired));
            }
            let wait = timeout.min(self.next_sig).saturating_duration_since(Instant::now());
            let _ = self.sock.set_read_timeout(Some(wait.clamp(Duration::from_millis(1), POLL)));
            let (n, source) = match self.sock.recv_from(&mut buf) {
                Ok(r) => r,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => continue,
                // e.g. ICMP port unreachable surfacing as ConnectionRefused while the peer is not up yet
                Err(_) => continue,
            };
            let pkt = &buf[..n];
            if let Some(m) = signal::decode(pkt) {
                // only from the peer's IP; a peer behind NAT is called with plain `v2v` (answer mode), whose dial is its source
                // shortcut: IPv6 matches the /64 so privacy (temporary) source addresses work; a host on that /64 could still interfere
                let ok = match (source.ip(), self.dial.ip()) {
                    (IpAddr::V6(a), IpAddr::V6(b)) => a.segments()[..4] == b.segments()[..4],
                    (a, b) => a == b,
                };
                if ok {
                    self.on_signal(m, source);
                    timeout = self.drain();
                }
                continue;
            }
            // str0m's ICE agent needs the remote credentials first: drop media until the SDP is applied
            if matches!(self.sig, Sig::Hello | Sig::Offering(..)) {
                continue;
            }
            let Ok(contents) = pkt.try_into() else { continue };
            // str0m drops STUN whose destination is not exactly one of our candidates
            let input = Input::Receive(Instant::now(), Receive { proto: Protocol::Udp, source, destination: self.local, contents });
            // strangers' datagrams must neither reach DTLS nor count as "peer alive"
            if !self.rtc.accepts(&input) {
                continue;
            }
            self.sh.stats.rx_bytes.fetch_add(n as u64, Relaxed);
            self.sh.stats.last_rx_ms.store(self.sh.start.elapsed().as_millis().max(1) as u64, Relaxed);
            timeout = self.input(input);
        }
    }

    fn input(&mut self, i: Input) -> Instant {
        if let Err(e) = self.rtc.handle_input(i) {
            self.sh.log(format!("webrtc: {e}"));
        }
        self.drain()
    }

    /// Polls str0m until it wants to wait; sends what the pacer releases now.
    fn drain(&mut self) -> Instant {
        loop {
            match self.rtc.poll_output() {
                Ok(Output::Timeout(t)) => return t,
                Ok(Output::Transmit(t)) => {
                    if self.sock.send_to(&t.contents, t.destination).is_ok() {
                        self.sh.stats.tx_bytes.fetch_add(t.contents.len() as u64, Relaxed);
                    }
                }
                Ok(Output::Event(e)) => self.on_event(e),
                Err(e) => {
                    self.reset(&format!("session error ({e}), reconnecting"));
                    return Instant::now();
                }
            }
        }
    }

    fn on_event(&mut self, e: Event) {
        let sh = self.sh.clone();
        let st = &sh.stats;
        match e {
            Event::IceConnectionStateChange(s) => {
                self.lost_since = (s == IceConnectionState::Disconnected).then(Instant::now);
                if s == IceConnectionState::Disconnected {
                    self.set_link("no packets from peer…".into());
                }
            }
            Event::Connected => {
                self.connected = true;
                self.sh.keyframe.store(true, Relaxed); // the peer's decoder starts from an IDR
                self.set_link("connected".into());
                self.sh.info("call connected (WebRTC: ICE, DTLS-SRTP)".into());
            }
            Event::MediaAdded(m) => match m.kind {
                MediaKind::Video => self.vmid = Some(m.mid),
                MediaKind::Audio => self.amid = Some(m.mid),
            },
            Event::MediaData(d) if Some(d.mid) == self.vmid => self.on_video(d),
            Event::MediaData(d) if Some(d.mid) == self.amid => self.on_audio(&d),
            Event::KeyframeRequest(_) => self.sh.keyframe.store(true, Relaxed),
            Event::EgressBitrateEstimate(BweKind::Twcc { estimate, .. } | BweKind::Remb { estimate, .. }) => {
                let bps = estimate.as_u64().clamp(BWE_MIN, BWE_MAX);
                st.bwe_bps.store(bps, Relaxed);
                st.video_bps.store(bps.saturating_sub(AUDIO_BPS).max(BWE_MIN - AUDIO_BPS) as u32, Relaxed);
            }
            Event::PeerStats(p) => {
                st.rtt_ms.store(p.rtt.map_or(0, |d| d.as_millis().max(1) as u32), Relaxed);
                st.loss_out.store(p.egress_loss_fraction.unwrap_or(0.0).to_bits(), Relaxed);
                st.loss_in.store(p.ingress_loss_fraction.unwrap_or(0.0).to_bits(), Relaxed);
                if let Some(c) = p.selected_candidate_pair.filter(|_| self.connected && self.lost_since.is_none()) {
                    self.set_link(format!("via {}", c.remote.addr));
                }
            }
            _ => {}
        }
    }

    fn on_video(&mut self, d: MediaData) {
        if self.io.video_out.is_none() {
            return; // --audio: no PLIs for video we do not show
        }
        if !d.contiguous {
            self.sh.want_keyframe.store(true, Relaxed); // a frame is missing even after NACK/RTX
        }
        if let Some(tx) = &self.io.video_out
            && let Err(TrySendError::Full(_)) = tx.try_send(d.data)
        {
            self.sh.want_keyframe.store(true, Relaxed); // decoder behind: frames dropped
        }
    }

    /// Opus with loss concealment: one missing packet is rebuilt from the next one's in-band FEC,
    /// longer gaps are filled by PLC (at most 100 ms of it; a longer gap is a pause, not loss).
    fn on_audio(&mut self, d: &MediaData) {
        let seq = **d.seq_range.start();
        let st = &self.sh.stats;
        let Some(gap) = audio_gap(self.last_audio, seq) else { return };
        self.last_audio = Some(seq);
        st.audio_rx.fetch_add(1, Relaxed);
        st.audio_lost.fetch_add(gap.min(50), Relaxed);
        let mut pcm = [0i16; PKT];
        let fill = gap.min(5);
        for i in 0..fill {
            let fec = i + 1 == fill;
            self.opus_dec.decode(fec.then_some(&d.data[..]), fec, &mut pcm);
            self.io.player.push(&pcm);
        }
        self.opus_dec.decode(Some(&d.data), false, &mut pcm);
        self.io.player.push(&pcm);
    }

    /// Sends a PLI, at most one per PLI_GAP (str0m would coalesce them anyway). True if one was sent.
    fn request_keyframe(&mut self) -> bool {
        if self.last_pli.is_some_and(|t| t.elapsed() < PLI_GAP) {
            return false;
        }
        let Some(mut w) = self.vmid.and_then(|m| self.rtc.writer(m)) else { return false };
        if !w.is_request_keyframe_possible(KeyframeRequestKind::Pli) {
            return false;
        }
        self.sh.want_keyframe.store(false, Relaxed);
        self.last_pli = Some(Instant::now());
        w.request_keyframe(None, KeyframeRequestKind::Pli).is_ok()
    }

    fn write_video(&mut self, (au, at): Encoded) {
        let Some(mid) = self.vmid.filter(|_| self.connected) else { return };
        let rtp = MediaTime::new((at - self.sh.start).as_micros() as u64 * 9 / 100, Frequency::NINETY_KHZ);
        self.write(mid, Codec::H264, at, rtp, au);
    }

    fn write_audio(&mut self, pcm: &Pcm) {
        let Some(mid) = self.amid.filter(|_| self.connected) else { return };
        let mut pkt = [0u8; 1500];
        let Some(n) = self.opus_enc.encode(pcm, &mut pkt) else { return };
        // the RTP clock keeps running while muted (no packets then): skip ahead after a pause
        let now_n = (self.sh.start.elapsed().as_millis() / 20) as u64;
        self.audio_n = if now_n > self.audio_n + 5 { now_n } else { self.audio_n + 1 };
        let rtp = MediaTime::new(self.audio_n * PKT as u64, Frequency::FORTY_EIGHT_KHZ);
        self.write(mid, Codec::Opus, Instant::now(), rtp, pkt[..n].to_vec());
    }

    fn write(&mut self, mid: Mid, codec: Codec, at: Instant, rtp: MediaTime, data: Vec<u8>) {
        if !self.rtc.media(mid).is_some_and(|m| m.direction().is_sending()) {
            return; // e.g. video toward an --audio peer, which answered it inactive
        }
        let Some(w) = self.rtc.writer(mid) else { return };
        let Some(pt) = w.payload_params().find(|p| p.spec().codec == codec).map(|p| p.pt()) else { return };
        if let Err(e) = w.write(pt, at, rtp, data) {
            self.sh.log(format!("send {codec:?}: {e}"));
        }
    }

    fn send_signal(&mut self) {
        let pkts = match &self.sig {
            Sig::Hello if self.peer != self.dial => {
                let _ = self.sock.send_to(&signal::hello(self.nonce), self.dial);
                vec![signal::hello(self.nonce)]
            }
            Sig::Hello => vec![signal::hello(self.nonce)],
            Sig::Offering(p, _) => p.clone(),
            Sig::Answered(_) | Sig::Done => return,
        };
        for p in pkts {
            let _ = self.sock.send_to(&p, self.peer);
        }
    }

    fn on_signal(&mut self, m: signal::Msg, source: SocketAddr) {
        if m.kind == Kind::Bye {
            if self.peer_nonce.is_none_or(|n| n == m.nonce) {
                self.set_link("peer hung up".into());
                self.sh.peer_bye.store(true, Relaxed);
                self.sh.quit.store(true, Relaxed);
            }
            return;
        }
        if self.peer_nonce.is_some_and(|n| n != m.nonce) && m.nonce != self.nonce {
            self.reset("peer restarted, reconnecting");
        }
        self.peer = source; // always reply to the port the peer's packets come from (SNAT)
        match (m.kind, &mut self.sig) {
            (Kind::Hello, Sig::Hello) => match signal::we_offer(self.nonce, m.nonce) {
                Some(true) => {
                    self.peer_nonce = Some(m.nonce);
                    let mut api = self.rtc.sdp_api();
                    self.amid = Some(api.add_media(MediaKind::Audio, Direction::SendRecv, None, None, None));
                    // --audio offers no video at all, so the peer does not send any
                    if self.io.video_out.is_some() {
                        self.vmid = Some(api.add_media(MediaKind::Video, Direction::SendRecv, None, None, None));
                    }
                    let Some((offer, pending)) = api.apply() else { return };
                    let Some(pkts) = signal::chunks(Kind::Offer, self.nonce, &offer.to_sdp_string()) else { return };
                    self.sig = Sig::Offering(pkts, Some(pending));
                    self.set_link("offer sent…".into());
                    self.send_signal();
                }
                Some(false) => {
                    self.peer_nonce = Some(m.nonce);
                    self.send_signal(); // make sure the winner has our nonce
                }
                None if !self.self_call => {
                    self.self_call = true;
                    self.sh.log(format!("{source} is this v2v itself: calling yourself is not supported"));
                }
                None => {}
            },
            // a late HELLO from a peer that has not seen our OFFER yet
            (Kind::Hello, Sig::Offering(..)) => self.send_signal(),
            (Kind::Offer, Sig::Hello | Sig::Answered(_)) => {
                let Some(sdp) = self.reasm.push(&m) else { return };
                if let Sig::Hello = self.sig {
                    self.peer_nonce = Some(m.nonce);
                    let answer = SdpOffer::from_sdp_string(&sdp).map_err(|e| e.to_string()).and_then(|o| {
                        self.rtc.sdp_api().accept_offer(o).map_err(|e| e.to_string())
                    });
                    let pkts = match answer {
                        Ok(a) if self.io.video_out.is_none() => signal::chunks(Kind::Answer, self.nonce, &video_inactive(&a.to_sdp_string())),
                        Ok(a) => signal::chunks(Kind::Answer, self.nonce, &a.to_sdp_string()),
                        Err(e) => return self.sh.log(format!("bad offer from peer: {e}")),
                    };
                    let Some(pkts) = pkts else { return };
                    self.sig = Sig::Answered(pkts);
                    self.set_link("answered, connecting…".into());
                    self.add_peer_candidate(source);
                }
                if let Sig::Answered(p) = &self.sig {
                    for p in p {
                        let _ = self.sock.send_to(p, source);
                    }
                }
            }
            (Kind::Answer, Sig::Offering(_, pending)) => {
                let Some(sdp) = self.reasm.push(&m) else { return };
                let Some(pending) = pending.take() else { return };
                let r = SdpAnswer::from_sdp_string(&sdp).map_err(|e| e.to_string());
                if let Err(e) = r.and_then(|a| self.rtc.sdp_api().accept_answer(pending, a).map_err(|e| e.to_string())) {
                    return self.reset(&format!("bad answer from peer ({e}), retrying"));
                }
                self.sig = Sig::Done;
                self.set_link("connecting…".into());
                self.add_peer_candidate(source);
            }
            _ => {}
        }
    }

    /// Where signalling comes from is a known-good remote candidate (the SNAT address, for one).
    fn add_peer_candidate(&mut self, source: SocketAddr) {
        if let Ok(c) = Candidate::host(source, "udp") {
            self.rtc.add_remote_candidate(c);
        }
    }
}

/// --audio answering a video offer: tell the peer not to send video (str0m cannot answer with another direction).
fn video_inactive(sdp: &str) -> String {
    let mut video = false;
    let lines = sdp.split_inclusive('\n').map(|l| {
        if l.starts_with("m=") {
            video = l.starts_with("m=video");
        }
        match l.trim_end() {
            "a=sendrecv" | "a=sendonly" | "a=recvonly" if video => "a=inactive\r\n",
            _ => l,
        }
    });
    lines.collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;
    use openh264::encoder::Encoder;
    use openh264::formats::YUVBuffer;
    use std::sync::mpsc::sync_channel;
    use std::thread;

    struct Side {
        sh: Arc<Shared>,
        video: Receiver<Arc<[u8]>>,
        done: thread::JoinHandle<()>,
        feed: thread::JoinHandle<()>,
    }

    /// A session on `sock` calling `peer`, fed with a test tone and an encoded test pattern.
    fn side(sock: UdpSocket, peer: SocketAddr, bye: bool) -> Side {
        let sh = Arc::new(Shared::new(Settings::default(), false));
        let (pcm_tx, pcm_in) = sync_channel(50);
        let (vid_tx, video_in) = sync_channel(4);
        let (video_out, video) = sync_channel(100);
        let player = Player::new(sh.clone(), Arc::default(), Arc::default());
        let io = Io { video_in: Some(video_in), pcm_in, video_out: Some(video_out), player };
        let s2 = sh.clone();
        let done = thread::spawn(move || call(sock, peer, random(), s2, io, bye));
        let s2 = sh.clone();
        let feed = thread::spawn(move || {
            let mut enc = Encoder::new().unwrap();
            let mut k = 0usize;
            while !s2.quit.load(Relaxed) {
                let _ = pcm_tx.try_send(std::array::from_fn(|i| (((k * PKT + i) as f32 * 0.05).sin() * 8000.0) as i16));
                if k.is_multiple_of(3) {
                    if s2.keyframe.swap(false, Relaxed) {
                        enc.force_intra_frame();
                    }
                    let yuv = YUVBuffer::from_vec((0..160 * 120 * 3 / 2).map(|i| (i + k) as u8).collect(), 160, 120);
                    let _ = vid_tx.try_send((enc.encode(&yuv).unwrap().to_vec(), Instant::now()));
                }
                k += 1;
                thread::sleep(Duration::from_millis(20));
            }
        });
        Side { sh, video, done, feed }
    }

    /// Waits for decodable video (the peer must start with an IDR) and audio.
    fn wait_media(s: &Side, what: &str) {
        let (t, mut dec) = (Instant::now(), openh264::decoder::Decoder::new().unwrap());
        let mut frames = 0;
        while t.elapsed() < Duration::from_secs(10) {
            frames += s.video.try_iter().filter(|au| matches!(dec.decode(au), Ok(Some(_)))).count();
            if frames >= 5 && s.sh.stats.audio_rx.load(Relaxed) >= 20 {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("{what}: {frames} video frames, {} audio packets", s.sh.stats.audio_rx.load(Relaxed));
    }

    fn stop(s: Side) {
        s.sh.quit.store(true, Relaxed);
        s.done.join().unwrap();
        s.feed.join().unwrap();
    }

    #[test]
    fn call_both_ways_and_peer_restart() {
        let a = UdpSocket::bind("127.0.0.1:0").unwrap();
        let b = UdpSocket::bind("127.0.0.1:0").unwrap();
        let (aa, ba) = (a.local_addr().unwrap(), b.local_addr().unwrap());
        // symmetric: both sides call each other, the nonce tie-break picks the offerer
        let a = side(a, ba, true);
        let b = side(b, aa, false); // b's first run "crashes": no BYE
        wait_media(&a, "a");
        wait_media(&b, "b");
        // b restarts on the same address: a must notice the new nonce, renegotiate, and carry media again
        stop(b);
        a.sh.stats.audio_rx.store(0, Relaxed);
        let _ = a.video.try_iter().count();
        let b = side(UdpSocket::bind(ba).unwrap(), aa, true);
        wait_media(&b, "b after restart");
        wait_media(&a, "a after b restarted");
        // a hangs up: b ends its call too
        stop(a);
        let t = Instant::now();
        while !b.sh.quit.load(Relaxed) && t.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(b.sh.peer_bye.load(Relaxed), "b did not hear the hang-up");
        stop(b);
    }

    #[test]
    fn stranger_cannot_take_over() {
        let a = UdpSocket::bind("127.0.0.1:0").unwrap();
        let b = UdpSocket::bind("127.0.0.1:0").unwrap();
        let (aa, ba) = (a.local_addr().unwrap(), b.local_addr().unwrap());
        let x = UdpSocket::bind("127.0.0.2:0").unwrap();
        x.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let offer = signal::chunks(Kind::Offer, u64::MAX, "v=0\r\n").unwrap().remove(0);
        let spam = || {
            for _ in 0..10 {
                let _ = x.send_to(&signal::hello(u64::MAX), aa);
                let _ = x.send_to(&offer, aa);
                thread::sleep(Duration::from_millis(20));
            }
        };
        // first while a is still waiting for b (the window before the peer's nonce is known), then mid-call
        let a = side(a, ba, true);
        spam();
        let b = side(b, aa, true);
        wait_media(&a, "a");
        wait_media(&b, "b");
        spam();
        a.sh.stats.audio_rx.store(0, Relaxed);
        let _ = a.video.try_iter().count();
        b.sh.keyframe.store(true, Relaxed); // wait_media's fresh decoder needs an IDR
        wait_media(&a, "a after the stranger");
        assert!(x.recv_from(&mut [0u8; 2048]).is_err(), "the stranger got a reply");
        stop(a);
        stop(b);
    }

    #[test]
    fn offering_ignores_offer_with_own_nonce() {
        let sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        let peer = UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let sh = Arc::new(Shared::new(Settings::default(), false));
        let (_pcm_tx, pcm_in) = sync_channel(1);
        let player = Player::new(sh.clone(), Arc::default(), Arc::default());
        let mut s = Session::new(sock, peer, sh, Io { video_in: None, pcm_in, video_out: None, player }).unwrap();
        s.nonce = u64::MAX;
        s.on_signal(signal::decode(&signal::hello(1)).unwrap(), peer);
        assert!(matches!(s.sig, Sig::Offering(..)));
        let offer = signal::chunks(Kind::Offer, u64::MAX, "v=0\r\n").unwrap().remove(0);
        s.on_signal(signal::decode(&offer).unwrap(), peer);
        assert!(matches!(s.sig, Sig::Offering(..)));
    }

    #[test]
    fn audio_gap_cases() {
        for (last, seq, want) in [(None, 7, Some(0)), (Some(7), 8, Some(0)), (Some(7), 9, Some(1)), (Some(7), 7, None), (Some(7), 6, None), (Some(7), 100, Some(92))] {
            assert_eq!(audio_gap(last, seq), want, "{last:?} {seq}");
        }
    }

    #[test]
    fn audio_answer_turns_video_off() {
        let sdp = "v=0\r\na=sendrecv\r\nm=audio 9 UDP\r\na=sendrecv\r\nm=video 9 UDP\r\na=mid:1\r\na=sendrecv\r\n";
        let want = "v=0\r\na=sendrecv\r\nm=audio 9 UDP\r\na=sendrecv\r\nm=video 9 UDP\r\na=mid:1\r\na=inactive\r\n";
        assert_eq!(video_inactive(sdp), want);
    }
}
