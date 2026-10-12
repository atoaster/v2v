//! In-band signalling on the media UDP socket: HELLO (role tie-break), then the SDP offer/answer.
//!
//! RFC 7983 demux: a first byte of 0-3 is STUN, 20-63 DTLS, 128-191 RTP/RTCP; 64-127 is unused,
//! so every signalling datagram starts with b'V' (0x56).
//!
//! Wire: [b'V', kind, sender_nonce u64 BE, idx u8, count u8, payload...]
//! HELLO and BYE (hang-up, sent a few times, unacknowledged) have idx=count=0 and no payload. OFFER/ANSWER carry the SDP split into <=CHUNK bytes.
//! No ACKs: the offerer resends the OFFER until a full ANSWER arrives; the answerer
//! (re)sends its cached ANSWER whenever it sees a complete OFFER. Idempotent both ways.

pub const MAGIC: u8 = b'V';
pub const CHUNK: usize = 1100;
const HDR: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Hello = 1,
    Offer = 2,
    Answer = 3,
    Bye = 4,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Msg<'a> {
    pub kind: Kind,
    pub nonce: u64,
    pub idx: u8,
    pub count: u8,
    pub payload: &'a [u8],
}

pub fn hello(nonce: u64) -> Vec<u8> {
    encode(Kind::Hello, nonce, 0, 0, &[])
}

pub fn bye(nonce: u64) -> Vec<u8> {
    encode(Kind::Bye, nonce, 0, 0, &[])
}

fn encode(kind: Kind, nonce: u64, idx: u8, count: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(HDR + payload.len());
    v.extend([MAGIC, kind as u8]);
    v.extend(nonce.to_be_bytes());
    v.extend([idx, count]);
    v.extend(payload);
    v
}

/// Splits an SDP into datagrams. None past 255 chunks (~280 KB), far beyond any SDP.
pub fn chunks(kind: Kind, nonce: u64, sdp: &str) -> Option<Vec<Vec<u8>>> {
    let parts: Vec<&[u8]> = sdp.as_bytes().chunks(CHUNK).collect();
    let n = u8::try_from(parts.len()).ok()?;
    Some(parts.iter().enumerate().map(|(i, p)| encode(kind, nonce, i as u8, n, p)).collect())
}

/// Parses an untrusted datagram; None for anything that is not well-formed signalling.
pub fn decode(pkt: &[u8]) -> Option<Msg<'_>> {
    if pkt.len() < HDR || pkt[0] != MAGIC {
        return None;
    }
    let kind = match pkt[1] {
        1 => Kind::Hello,
        2 => Kind::Offer,
        3 => Kind::Answer,
        4 => Kind::Bye,
        _ => return None,
    };
    let (idx, count) = (pkt[10], pkt[11]);
    if matches!(kind, Kind::Offer | Kind::Answer) && (idx >= count || pkt.len() > HDR + CHUNK) {
        return None;
    }
    Some(Msg { kind, nonce: u64::from_be_bytes(pkt[2..10].try_into().ok()?), idx, count, payload: &pkt[HDR..] })
}

/// Collects the chunks of one OFFER or ANSWER. A chunk with another nonce/count (peer restarted)
/// discards what was collected so far.
#[derive(Default)]
pub struct Reassembler {
    key: Option<(Kind, u64, u8)>,
    parts: Vec<Option<Vec<u8>>>,
}

impl Reassembler {
    /// Feeds a chunk; returns the full SDP once all chunks are present (again on later duplicates).
    pub fn push(&mut self, m: &Msg) -> Option<String> {
        if self.key != Some((m.kind, m.nonce, m.count)) {
            self.key = Some((m.kind, m.nonce, m.count));
            self.parts = vec![None; m.count as usize];
        }
        *self.parts.get_mut(m.idx as usize)? = Some(m.payload.to_vec());
        if self.parts.iter().any(Option::is_none) {
            return None;
        }
        String::from_utf8(self.parts.iter().flatten().flatten().copied().collect()).ok()
    }
}

/// The higher nonce makes the offer and is ICE-controlling (str0m: the offerer controls).
/// Equal nonces (p = 2^-64 between two processes) means we are hearing our own HELLO: a self-call.
pub fn we_offer(mine: u64, theirs: u64) -> Option<bool> {
    (mine != theirs).then_some(mine > theirs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_roundtrip_and_demux() {
        let p = hello(42);
        assert!((64..=127).contains(&p[0]), "RFC 7983 unused range");
        let m = decode(&p).unwrap();
        assert_eq!((m.kind, m.nonce, m.payload.len()), (Kind::Hello, 42, 0));
        let b = bye(43);
        let m = decode(&b).unwrap();
        assert_eq!((m.kind, m.nonce, m.payload.len()), (Kind::Bye, 43, 0));
        // STUN / DTLS / RTP first bytes are never signalling
        for b in [[0x00, 1], [22, 1], [0x80, 1]] {
            assert!(decode(&[&b[..], &[0; 12]].concat()).is_none());
        }
        assert!(decode(&[b'V', 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).is_none()); // unknown kind
        assert!(decode(&[b'V', 2]).is_none()); // truncated
        assert!(decode(&[b'V', 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1]).is_none()); // idx >= count
        assert!(decode(&[b'V', 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).is_none()); // count 0
    }

    #[test]
    fn chunk_reassemble_out_of_order_and_restart() {
        let sdp: String = (0..5000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        let pkts = chunks(Kind::Offer, 7, &sdp).unwrap();
        assert_eq!(pkts.len(), 5);
        assert!(pkts.iter().all(|p| p.len() <= CHUNK + HDR && decode(p).is_some()));
        let mut r = Reassembler::default();
        // a stale offer of an earlier process, then the new one out of order with a duplicate
        let old = chunks(Kind::Offer, 6, "old sdp").unwrap();
        assert_eq!(r.push(&decode(&old[0]).unwrap()), Some("old sdp".into()));
        assert_eq!(r.push(&decode(&pkts[4]).unwrap()), None);
        for p in pkts.iter().rev() {
            let got = r.push(&decode(p).unwrap());
            assert_eq!(got.is_some(), std::ptr::eq(p, &pkts[0]));
        }
        // a duplicate after completion yields it again (the answerer re-sends its answer)
        assert_eq!(r.push(&decode(&pkts[2]).unwrap()), Some(sdp));
        assert!(chunks(Kind::Offer, 1, &"x".repeat(CHUNK * 256)).is_none());
    }

    #[test]
    fn tie_break_is_antisymmetric() {
        for (a, b) in [(5, 3), (0, u64::MAX), (1 << 63, 1)] {
            assert_eq!(we_offer(a, b).map(|x| !x), we_offer(b, a));
            assert!(we_offer(a, b).is_some());
        }
        assert_eq!(we_offer(5, 3), Some(true));
        assert_eq!(we_offer(4, 4), None);
    }
}
