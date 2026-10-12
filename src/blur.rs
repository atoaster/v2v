//! Background blur: person segmentation with MediaPipe's selfie_segmenter_landscape (ONNX, run by tract),
//! then the frame is mixed with a cheap low-res blur of itself outside the person mask.

use image::RgbImage;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tract_onnx::prelude::*;

/// selfie_segmenter_landscape © Google, Apache-2.0, converted from TFLite to ONNX.
static MODEL: &[u8] = include_bytes!("../assets/selfie_segmenter_landscape.onnx");

pub struct Segmenter {
    plan: Arc<TypedSimplePlan>,
    mw: usize,
    mh: usize,
    prev: Vec<f32>, // last mask, for temporal smoothing; empty after a pause
    pub frames: u32,
    pub busy: Duration,
}

impl Segmenter {
    /// Parses and optimizes the embedded model (~50 ms): call on the camera thread, never the UI thread.
    pub fn load() -> Result<Self, String> {
        let load = || -> TractResult<_> {
            let m = tract_onnx::onnx().model_for_read(&mut &MODEL[..])?.into_typed()?;
            let shape = m.input_fact(0)?.shape.as_concrete().ok_or_else(|| TractError::msg("dynamic input shape"))?.to_vec();
            if shape.len() != 4 || shape[3] != 3 {
                return Err(TractError::msg(format!("unexpected model input {shape:?}"))); // want NHWC RGB
            }
            Ok((m.into_optimized()?.into_runnable()?, shape[2], shape[1]))
        };
        let (plan, mw, mh) = load().map_err(|e| format!("cannot load segmentation model: {e}"))?;
        Ok(Segmenter { plan, mw, mh, prev: Vec::new(), frames: 0, busy: Duration::ZERO })
    }

    /// Forget the last mask (blur was off for a while, so it is stale).
    pub fn pause(&mut self) {
        self.prev.clear();
    }

    /// Person mask (0..1, 1 = person) at model resolution, feathered and smoothed over time.
    pub fn mask(&mut self, img: &RgbImage) -> Result<Vec<f32>, String> {
        let (fw, fh, rgb) = (img.width() as usize, img.height() as usize, img.as_raw());
        let (mw, mh) = (self.mw, self.mh);
        // whole frame squashed to model size (no letterbox, like MediaPipe), RGB in [0,1]
        let input = tract_ndarray::Array4::from_shape_fn((1, mh, mw, 3), |(_, y, x, c)| rgb[((y * fh / mh) * fw + x * fw / mw) * 3 + c] as f32 / 255.0);
        let run = || -> TractResult<Vec<f32>> {
            let out = self.plan.run(tvec!(Tensor::from(input).into()))?;
            Ok(out[0].to_plain_array_view::<f32>()?.iter().copied().collect())
        };
        let mut m = feather(&run().map_err(|e| e.to_string())?, mw, mh);
        if m.len() != mw * mh {
            return Err(format!("segmentation output has {} values, expected {}", m.len(), mw * mh));
        }
        if self.prev.len() == m.len() {
            // damps edge flicker between frames
            m.iter_mut().zip(&self.prev).for_each(|(a, b)| *a = 0.6 * *a + 0.4 * b);
        }
        self.prev.clone_from(&m);
        Ok(m)
    }

    pub fn apply(&mut self, img: &mut RgbImage) -> Result<(), String> {
        let t = Instant::now();
        let (fw, fh) = (img.width() as usize, img.height() as usize);
        if fw < 8 || fh < 8 {
            return Ok(());
        }
        let mask = self.mask(img)?;
        let (bg, bw, bh) = blur_small(img.as_raw(), fw, fh, 4, 3);
        composite(img, (&mask, self.mw, self.mh), (&bg, bw, bh));
        self.frames += 1;
        self.busy += t.elapsed();
        Ok(())
    }
}

/// 3x3 box blur of the low-res mask = cheap feathering.
fn feather(m: &[f32], w: usize, h: usize) -> Vec<f32> {
    if m.len() != w * h {
        return m.to_vec();
    }
    let at = |x: isize, y: isize| m[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))).map(|(dx, dy)| at(x + dx, y + dy)).sum::<f32>() / 9.0
        })
        .collect()
}

/// Background: box-average downscale by `k`, then 3 rounds of separable box blur radius `r` (≈ gaussian).
fn blur_small(rgb: &[u8], fw: usize, fh: usize, k: usize, r: usize) -> (Vec<f32>, usize, usize) {
    let (w, h) = (fw / k, fh / k);
    let mut s: Vec<f32> = (0..w * h * 3)
        .map(|i| {
            let (p, c) = (i / 3, i % 3);
            let (x, y) = (p % w * k, p / w * k);
            let acc: u32 = (0..k).flat_map(|yy| (0..k).map(move |xx| (yy, xx))).map(|(yy, xx)| rgb[((y + yy) * fw + x + xx) * 3 + c] as u32).sum();
            acc as f32 / (k * k) as f32
        })
        .collect();
    let mut t = vec![0f32; s.len()];
    for _ in 0..3 {
        box_pass(&s, &mut t, w, h, r, true);
        box_pass(&t, &mut s, w, h, r, false);
    }
    (s, w, h)
}

fn box_pass(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize, horiz: bool) {
    let (n, lines) = if horiz { (w, h) } else { (h, w) };
    let idx = |line: usize, i: usize| if horiz { line * w + i } else { i * w + line };
    let norm = 1.0 / (2 * r + 1) as f32;
    for line in 0..lines {
        for c in 0..3 {
            let px = |i: isize| src[idx(line, i.clamp(0, n as isize - 1) as usize) * 3 + c];
            let mut acc: f32 = (-(r as isize)..=r as isize).map(px).sum();
            for i in 0..n {
                dst[idx(line, i) * 3 + c] = acc * norm;
                acc += px(i as isize + r as isize + 1) - px(i as isize - r as isize);
            }
        }
    }
}

/// Bilinear upsampling taps from `src` to `dst` pixels (centre aligned): (i0, i1, frac).
fn taps(src: usize, dst: usize) -> Vec<(usize, usize, f32)> {
    (0..dst)
        .map(|x| {
            let f = ((x as f32 + 0.5) * src as f32 / dst as f32 - 0.5).clamp(0.0, (src - 1) as f32);
            let i = f as usize;
            (i, (i + 1).min(src - 1), f - i as f32)
        })
        .collect()
}

/// out = a*fg + (1-a)*bg, with mask and background bilinearly upsampled on the fly.
fn composite(img: &mut RgbImage, (mask, mw, mh): (&[f32], usize, usize), (bg, bw, bh): (&[f32], usize, usize)) {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    if mw == 0 || mh == 0 || bw == 0 || bh == 0 || mask.len() < mw * mh || bg.len() < bw * bh * 3 {
        return;
    }
    let (mx, my, bx, by) = (taps(mw, fw), taps(mh, fh), taps(bw, fw), taps(bh, fh));
    let lerp = |a: &[f32], b: &[f32], f: f32, out: &mut [f32]| out.iter_mut().zip(a.iter().zip(b)).for_each(|(o, (a, b))| *o = a + (b - a) * f);
    let (mut mrow, mut brow) = (vec![0f32; mw], vec![0f32; bw * 3]);
    for (y, row) in img.chunks_exact_mut(fw * 3).enumerate() {
        let (m0, m1, ma) = my[y];
        lerp(&mask[m0 * mw..][..mw], &mask[m1 * mw..][..mw], ma, &mut mrow);
        let (b0, b1, ba) = by[y];
        lerp(&bg[b0 * bw * 3..][..bw * 3], &bg[b1 * bw * 3..][..bw * 3], ba, &mut brow);
        for (x, px) in row.chunks_exact_mut(3).enumerate() {
            let (i0, i1, f) = mx[x];
            let a = mrow[i0] + (mrow[i1] - mrow[i0]) * f;
            // smoothstep sharpens the soft model edge a bit without aliasing
            let a = ((a - 0.3) / 0.4).clamp(0.0, 1.0);
            let a = a * a * (3.0 - 2.0 * a);
            if a >= 0.999 {
                continue;
            }
            let (j0, j1, g) = bx[x];
            for (c, p) in px.iter_mut().enumerate() {
                let b = brow[j0 * 3 + c] + (brow[j1 * 3 + c] - brow[j0 * 3 + c]) * g;
                *p = (*p as f32 * a + b * (1.0 - a)) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feather_and_composite() {
        // feathering a single hot pixel spreads it evenly over its 3x3 neighbourhood
        let mut m = vec![0f32; 25];
        m[12] = 9.0;
        let f = feather(&m, 5, 5);
        assert!((f[6] - 1.0).abs() < 1e-6 && (f[12] - 1.0).abs() < 1e-6 && f[0] == 0.0);

        // left half "person" keeps its pixels, right half becomes the (flat gray) background
        let mut img = RgbImage::from_fn(16, 8, |x, _| image::Rgb([if x % 2 == 0 { 255 } else { 0 }, 0, 0]));
        let orig = img.clone();
        let mask: Vec<f32> = (0..8 * 4).map(|i| if i % 8 < 4 { 1.0 } else { 0.0 }).collect();
        let bg = vec![100f32; 4 * 2 * 3];
        composite(&mut img, (&mask, 8, 4), (&bg, 4, 2));
        for y in 0..8 {
            for x in 0..6 {
                assert_eq!(img.get_pixel(x, y), orig.get_pixel(x, y), "person pixel {x},{y} changed");
            }
            for x in 10..16 {
                assert_eq!(img.get_pixel(x, y).0, [100, 100, 100], "background pixel {x},{y} not replaced");
            }
        }
        // bad sizes are ignored, not a panic
        composite(&mut img, (&mask, 0, 4), (&bg, 4, 2));
        composite(&mut img, (&mask[..3], 8, 4), (&bg, 4, 2));

        // background blur of a flat image stays flat
        let (b, w, h) = blur_small(&[50u8; 16 * 8 * 3], 16, 8, 4, 3);
        assert_eq!((w, h), (4, 2));
        assert!(b.iter().all(|v| (v - 50.0).abs() < 1e-3));
    }

    #[test]
    fn model_loads_and_masks() {
        let mut s = Segmenter::load().unwrap();
        assert_eq!((s.mw, s.mh), (256, 144));
        let mut img = RgbImage::from_pixel(640, 480, image::Rgb([128, 128, 128]));
        let m = s.mask(&img).unwrap();
        assert_eq!(m.len(), 256 * 144);
        assert!(m.iter().all(|v| (0.0..=1.0).contains(v)));
        s.apply(&mut img).unwrap();
        assert_eq!(s.frames, 1);
    }
}
