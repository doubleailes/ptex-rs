//! Pixel utility functions (ported from `PtexUtils.cpp`).
//!
//! All functions operate on raw little-endian byte buffers and are
//! dispatched on [`DataType`].

#![allow(clippy::too_many_arguments)] // signatures mirror the C++ PtexUtils API

use crate::types::{float_to_half, half_to_float, DataType, FaceInfo};

/// Element accessor used by the generic reduction kernels.
///
/// Integer types accumulate in a wider integer and shift (matching the C++
/// implementation); float types accumulate in f32.
trait Texel: Copy {
    const SIZE: usize;
    fn get(b: &[u8]) -> Self;
    fn put(self, b: &mut [u8]);
    fn halve_sum(a: Self, b: Self) -> Self;
    fn quarter_sum(a: Self, b: Self, c: Self, d: Self) -> Self;
}

impl Texel for u8 {
    const SIZE: usize = 1;
    fn get(b: &[u8]) -> Self {
        b[0]
    }
    fn put(self, b: &mut [u8]) {
        b[0] = self;
    }
    fn halve_sum(a: Self, b: Self) -> Self {
        ((a as u32 + b as u32) >> 1) as u8
    }
    fn quarter_sum(a: Self, b: Self, c: Self, d: Self) -> Self {
        ((a as u32 + b as u32 + c as u32 + d as u32) >> 2) as u8
    }
}

impl Texel for u16 {
    const SIZE: usize = 2;
    fn get(b: &[u8]) -> Self {
        u16::from_le_bytes([b[0], b[1]])
    }
    fn put(self, b: &mut [u8]) {
        b[..2].copy_from_slice(&self.to_le_bytes());
    }
    fn halve_sum(a: Self, b: Self) -> Self {
        ((a as u32 + b as u32) >> 1) as u16
    }
    fn quarter_sum(a: Self, b: Self, c: Self, d: Self) -> Self {
        ((a as u32 + b as u32 + c as u32 + d as u32) >> 2) as u16
    }
}

/// Half-precision float stored as its 16-bit representation.
#[derive(Clone, Copy)]
struct Half(u16);

impl Texel for Half {
    const SIZE: usize = 2;
    fn get(b: &[u8]) -> Self {
        Half(u16::from_le_bytes([b[0], b[1]]))
    }
    fn put(self, b: &mut [u8]) {
        b[..2].copy_from_slice(&self.0.to_le_bytes());
    }
    fn halve_sum(a: Self, b: Self) -> Self {
        Half(float_to_half(
            0.5 * (half_to_float(a.0) + half_to_float(b.0)),
        ))
    }
    fn quarter_sum(a: Self, b: Self, c: Self, d: Self) -> Self {
        Half(float_to_half(
            0.25 * (half_to_float(a.0)
                + half_to_float(b.0)
                + half_to_float(c.0)
                + half_to_float(d.0)),
        ))
    }
}

impl Texel for f32 {
    const SIZE: usize = 4;
    fn get(b: &[u8]) -> Self {
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn put(self, b: &mut [u8]) {
        b[..4].copy_from_slice(&self.to_le_bytes());
    }
    fn halve_sum(a: Self, b: Self) -> Self {
        0.5 * (a + b)
    }
    fn quarter_sum(a: Self, b: Self, c: Self, d: Self) -> Self {
        0.25 * (a + b + c + d)
    }
}

/// Convert channel-planar data (as stored in the file) to
/// pixel-interleaved data.
///
/// `src` holds `nchan` planes of `uw * vw` elements each; `dst` receives
/// interleaved pixels, one row every `dstride` bytes.
pub fn interleave(
    src: &[u8],
    uw: usize,
    vw: usize,
    dst: &mut [u8],
    dstride: usize,
    dt: DataType,
    nchan: usize,
) {
    let esize = dt.size();
    let plane = uw * vw * esize;
    for c in 0..nchan {
        let splane = &src[c * plane..(c + 1) * plane];
        for v in 0..vw {
            let srow = &splane[v * uw * esize..(v + 1) * uw * esize];
            let drow = &mut dst[v * dstride..];
            for u in 0..uw {
                let doff = (u * nchan + c) * esize;
                drow[doff..doff + esize].copy_from_slice(&srow[u * esize..(u + 1) * esize]);
            }
        }
    }
}

/// Decode difference-encoded data in place (used by the `diffzipped`
/// encoding).  Only applies to `uint8` and `uint16` data.
pub fn decode_difference(data: &mut [u8], dt: DataType) {
    match dt {
        DataType::UInt8 => {
            let mut prev = 0u8;
            for p in data.iter_mut() {
                *p = p.wrapping_add(prev);
                prev = *p;
            }
        }
        DataType::UInt16 => {
            let mut prev = 0u16;
            for p in data.chunks_exact_mut(2) {
                let v = u16::from_le_bytes([p[0], p[1]]).wrapping_add(prev);
                p.copy_from_slice(&v.to_le_bytes());
                prev = v;
            }
        }
        _ => {}
    }
}

/// Generate the mapping from face id to "reduction face id".
///
/// Reduction levels in the file store faces sorted by decreasing detail;
/// `rfaceids[faceid]` gives the position of a face within a reduction
/// level.
pub fn gen_rfaceids(faces: &[FaceInfo]) -> Vec<u32> {
    let nfaces = faces.len();
    let mut faceids: Vec<u32> = (0..nfaces as u32).collect();
    let key = |id: u32| -> i8 {
        let f = &faces[id as usize];
        if f.is_constant() {
            1
        } else {
            f.res.ulog2.min(f.res.vlog2)
        }
    };
    // stable sort by smaller dimension (u or v) in descending order;
    // constant faces are treated as having a res of 1
    faceids.sort_by(|&a, &b| key(b).cmp(&key(a)));

    let mut rfaceids = vec![0u32; nfaces];
    for (rfaceid, &faceid) in faceids.iter().enumerate() {
        rfaceids[faceid as usize] = rfaceid as u32;
    }
    rfaceids
}

/// Multiply color channels by the alpha channel, in place.
pub fn multalpha(
    data: &mut [u8],
    npixels: usize,
    dt: DataType,
    nchannels: usize,
    alphachan: usize,
) {
    let scale = dt.one_value_inv();
    match dt {
        DataType::UInt8 => multalpha_t::<u8>(
            data,
            npixels,
            nchannels,
            alphachan,
            scale,
            |v| v as f32,
            |f| f as u8,
        ),
        DataType::UInt16 => multalpha_t::<u16>(
            data,
            npixels,
            nchannels,
            alphachan,
            scale,
            |v| v as f32,
            |f| f as u16,
        ),
        DataType::Half => multalpha_t::<Half>(
            data,
            npixels,
            nchannels,
            alphachan,
            scale,
            |v| half_to_float(v.0),
            |f| Half(float_to_half(f)),
        ),
        DataType::Float => {
            multalpha_t::<f32>(data, npixels, nchannels, alphachan, scale, |v| v, |f| f)
        }
    }
}

fn multalpha_t<T: Texel>(
    data: &mut [u8],
    npixels: usize,
    nchannels: usize,
    alphachan: usize,
    scale: f32,
    to_f: impl Fn(T) -> f32,
    from_f: impl Fn(f32) -> T,
) {
    // when alpha is the first channel, multiply the remaining channels;
    // otherwise multiply the channels preceding the alpha channel
    let (first, nchanmult) = if alphachan == 0 {
        (1usize, nchannels - 1)
    } else {
        (0usize, alphachan)
    };
    let esize = T::SIZE;
    for pix in data.chunks_exact_mut(nchannels * esize).take(npixels) {
        let alpha = to_f(T::get(&pix[alphachan * esize..]));
        let aval = scale * alpha;
        for c in first..first + nchanmult {
            let off = c * esize;
            let v = to_f(T::get(&pix[off..]));
            from_f(v * aval).put(&mut pix[off..]);
        }
    }
}

/// Fill a `ures * vres` region of `dst` with a single pixel value.
pub fn fill(
    src_pixel: &[u8],
    dst: &mut [u8],
    dstride: usize,
    ures: usize,
    vres: usize,
    pixelsize: usize,
) {
    let rowlen = ures * pixelsize;
    for p in dst[..rowlen].chunks_exact_mut(pixelsize) {
        p.copy_from_slice(src_pixel);
    }
    // fill remaining rows from the first row
    for v in 1..vres {
        let (head, tail) = dst.split_at_mut(v * dstride);
        tail[..rowlen].copy_from_slice(&head[..rowlen]);
    }
}

/// Copy `vres` rows of `rowlen` bytes from `src` (rows every `sstride`
/// bytes) to `dst` (rows every `dstride` bytes).
pub fn copy_rows(
    src: &[u8],
    sstride: usize,
    dst: &mut [u8],
    dstride: usize,
    vres: usize,
    rowlen: usize,
) {
    if sstride == rowlen && dstride == rowlen {
        dst[..vres * rowlen].copy_from_slice(&src[..vres * rowlen]);
    } else {
        for v in 0..vres {
            dst[v * dstride..v * dstride + rowlen]
                .copy_from_slice(&src[v * sstride..v * sstride + rowlen]);
        }
    }
}

macro_rules! dispatch {
    ($dt:expr, $f:ident, $src:expr, $sstride:expr, $uw:expr, $vw:expr, $dst:expr, $dstride:expr, $nchan:expr) => {
        match $dt {
            DataType::UInt8 => $f::<u8>($src, $sstride, $uw, $vw, $dst, $dstride, $nchan),
            DataType::UInt16 => $f::<u16>($src, $sstride, $uw, $vw, $dst, $dstride, $nchan),
            DataType::Half => $f::<Half>($src, $sstride, $uw, $vw, $dst, $dstride, $nchan),
            DataType::Float => $f::<f32>($src, $sstride, $uw, $vw, $dst, $dstride, $nchan),
        }
    };
}

/// 2x2 box-filter reduction of a `uw x vw` image into a `uw/2 x vw/2` image.
pub fn reduce(
    src: &[u8],
    sstride: usize,
    uw: usize,
    vw: usize,
    dst: &mut [u8],
    dstride: usize,
    dt: DataType,
    nchan: usize,
) {
    dispatch!(dt, reduce_t, src, sstride, uw, vw, dst, dstride, nchan)
}

/// Reduction in the u direction only (`uw x vw` -> `uw/2 x vw`).
pub fn reduce_u(
    src: &[u8],
    sstride: usize,
    uw: usize,
    vw: usize,
    dst: &mut [u8],
    dstride: usize,
    dt: DataType,
    nchan: usize,
) {
    dispatch!(dt, reduce_u_t, src, sstride, uw, vw, dst, dstride, nchan)
}

/// Reduction in the v direction only (`uw x vw` -> `uw x vw/2`).
pub fn reduce_v(
    src: &[u8],
    sstride: usize,
    uw: usize,
    vw: usize,
    dst: &mut [u8],
    dstride: usize,
    dt: DataType,
    nchan: usize,
) {
    dispatch!(dt, reduce_v_t, src, sstride, uw, vw, dst, dstride, nchan)
}

/// Reduction of a packed-triangle texture (`w x w` -> `w/2 x w/2`).
pub fn reduce_tri(
    src: &[u8],
    sstride: usize,
    w: usize,
    _vw: usize,
    dst: &mut [u8],
    dstride: usize,
    dt: DataType,
    nchan: usize,
) {
    dispatch!(dt, reduce_tri_t, src, sstride, w, 0, dst, dstride, nchan)
}

fn reduce_t<T: Texel>(
    src: &[u8],
    sstride: usize,
    uw: usize,
    vw: usize,
    dst: &mut [u8],
    dstride: usize,
    nchan: usize,
) {
    let es = T::SIZE;
    for v in 0..vw / 2 {
        let r0 = &src[2 * v * sstride..];
        let r1 = &src[(2 * v + 1) * sstride..];
        let drow = &mut dst[v * dstride..];
        for u in 0..uw / 2 {
            for c in 0..nchan {
                let s0 = (2 * u * nchan + c) * es;
                let s1 = ((2 * u + 1) * nchan + c) * es;
                let d = (u * nchan + c) * es;
                T::quarter_sum(
                    T::get(&r0[s0..]),
                    T::get(&r0[s1..]),
                    T::get(&r1[s0..]),
                    T::get(&r1[s1..]),
                )
                .put(&mut drow[d..]);
            }
        }
    }
}

fn reduce_u_t<T: Texel>(
    src: &[u8],
    sstride: usize,
    uw: usize,
    vw: usize,
    dst: &mut [u8],
    dstride: usize,
    nchan: usize,
) {
    let es = T::SIZE;
    for v in 0..vw {
        let srow = &src[v * sstride..];
        let drow = &mut dst[v * dstride..];
        for u in 0..uw / 2 {
            for c in 0..nchan {
                let s0 = (2 * u * nchan + c) * es;
                let s1 = ((2 * u + 1) * nchan + c) * es;
                let d = (u * nchan + c) * es;
                T::halve_sum(T::get(&srow[s0..]), T::get(&srow[s1..])).put(&mut drow[d..]);
            }
        }
    }
}

fn reduce_v_t<T: Texel>(
    src: &[u8],
    sstride: usize,
    uw: usize,
    vw: usize,
    dst: &mut [u8],
    dstride: usize,
    nchan: usize,
) {
    let es = T::SIZE;
    for v in 0..vw / 2 {
        let r0 = &src[2 * v * sstride..];
        let r1 = &src[(2 * v + 1) * sstride..];
        let drow = &mut dst[v * dstride..];
        for u in 0..uw {
            for c in 0..nchan {
                let s = (u * nchan + c) * es;
                T::halve_sum(T::get(&r0[s..]), T::get(&r1[s..])).put(&mut drow[s..]);
            }
        }
    }
}

fn reduce_tri_t<T: Texel>(
    src: &[u8],
    sstride: usize,
    w: usize,
    _vw: usize,
    dst: &mut [u8],
    dstride: usize,
    nchan: usize,
) {
    // A triangle texture of size w x w packs two triangles per quad:
    // texel (u, v) with u + v < w belongs to the "upright" triangle and
    // texel (w-1-u, w-1-v) is its mirrored counterpart.  The reduction
    // averages the 2x2 block of upright texels with the one mirrored texel
    // that completes the triangle (see PtexUtils::reduceTri).
    let es = T::SIZE;
    for v in 0..w / 2 {
        let r0 = &src[2 * v * sstride..];
        let r1 = &src[(2 * v + 1) * sstride..];
        let drow = &mut dst[v * dstride..];
        for u in 0..w / 2 {
            let rm = &src[(w - 1 - 2 * u) * sstride..];
            for c in 0..nchan {
                let s0 = (2 * u * nchan + c) * es;
                let s1 = ((2 * u + 1) * nchan + c) * es;
                let sm = ((w - 1 - 2 * v) * nchan + c) * es;
                let d = (u * nchan + c) * es;
                T::quarter_sum(
                    T::get(&r0[s0..]),
                    T::get(&r0[s1..]),
                    T::get(&r1[s0..]),
                    T::get(&rm[sm..]),
                )
                .put(&mut drow[d..]);
            }
        }
    }
}

/// Convert `nchan` values starting at `src` to f32.
pub fn convert_to_float(dst: &mut [f32], src: &[u8], dt: DataType, nchan: usize) {
    match dt {
        DataType::UInt8 => {
            // multiply by the reciprocal (not divide) to match the C++
            // conversion bit-for-bit
            let scale = 1.0f32 / 255.0;
            for (d, s) in dst[..nchan].iter_mut().zip(src.iter()) {
                *d = *s as f32 * scale;
            }
        }
        DataType::UInt16 => {
            let scale = 1.0f32 / 65535.0;
            for (d, s) in dst[..nchan].iter_mut().zip(src.chunks_exact(2)) {
                *d = u16::from_le_bytes([s[0], s[1]]) as f32 * scale;
            }
        }
        DataType::Half => {
            for (d, s) in dst[..nchan].iter_mut().zip(src.chunks_exact(2)) {
                *d = half_to_float(u16::from_le_bytes([s[0], s[1]]));
            }
        }
        DataType::Float => {
            for (d, s) in dst[..nchan].iter_mut().zip(src.chunks_exact(4)) {
                *d = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Res;

    #[test]
    fn decode_difference_u8() {
        let mut data = vec![10u8, 5, 250, 10];
        decode_difference(&mut data, DataType::UInt8);
        assert_eq!(data, vec![10, 15, 9, 19]);
    }

    #[test]
    fn decode_difference_u16() {
        let mut data = Vec::new();
        for v in [1000u16, 500, 65000] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        decode_difference(&mut data, DataType::UInt16);
        let vals: Vec<u16> = data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(vals, vec![1000, 1500, 964]); // 1500 + 65000 wraps
    }

    #[test]
    fn interleave_planar() {
        // 2x2, 2 channels of u8: planes [1,2,3,4] and [5,6,7,8]
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut dst = [0u8; 8];
        interleave(&src, 2, 2, &mut dst, 4, DataType::UInt8, 2);
        assert_eq!(dst, [1, 5, 2, 6, 3, 7, 4, 8]);
    }

    #[test]
    fn reduce_2x2_u8() {
        // 2x2 single channel -> 1x1
        let src = [10u8, 20, 30, 41];
        let mut dst = [0u8; 1];
        reduce(&src, 2, 2, 2, &mut dst, 1, DataType::UInt8, 1);
        assert_eq!(dst[0], (10 + 20 + 30 + 41) / 4);
    }

    #[test]
    fn rfaceids_sorted_by_min_res() {
        let mk = |ul: i8, vl: i8, constant: bool| FaceInfo {
            res: Res::new(ul, vl),
            flags: if constant {
                crate::types::face_flags::CONSTANT
            } else {
                0
            },
            ..Default::default()
        };
        let faces = vec![
            mk(1, 1, false),
            mk(4, 4, false),
            mk(3, 5, false),
            mk(8, 8, true),
        ];
        let r = gen_rfaceids(&faces);
        // order by min dim desc: face1 (4), face2 (3), face3 (const->1), face0 (1)
        // stable: face3 comes before face0? both key 1, original order 3 after 0.
        assert_eq!(r[1], 0);
        assert_eq!(r[2], 1);
        assert_eq!(r[0], 2);
        assert_eq!(r[3], 3);
    }
}
