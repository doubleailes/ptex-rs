//! Stateless decoding of Ptex data blocks.
//!
//! Every function here turns raw bytes read from the file into pixels or
//! index structures.  Nothing performs I/O and nothing borrows a reader, so
//! the single-threaded and shared readers share one implementation — and the
//! shared reader can decompress outside its I/O lock.

use std::io::Read;

use crate::error::{Error, Result};
use crate::file_info::{FileInfo, ReduceKind};
use crate::format::{self, Encoding, FaceDataHeader, FACE_DATA_HEADER_SIZE};
use crate::types::Res;
use crate::utils;

/// Index of one reduction level: one face data header and absolute file
/// offset per face stored in the level.
#[derive(Debug, Clone)]
pub(crate) struct Level {
    pub fdh: Vec<FaceDataHeader>,
    pub offsets: Vec<u64>,
}

/// Inflated tile directory of one tiled face data block.
#[derive(Debug, Clone)]
pub(crate) struct TileDir {
    pub tile_res: Res,
    pub fdh: Vec<FaceDataHeader>,
    pub offsets: Vec<u64>,
}

/// Zlib-decompress `comp` into exactly `unzipsize` bytes.
pub(crate) fn unzip(comp: &[u8], unzipsize: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(unzipsize);
    let mut decoder = flate2::read::ZlibDecoder::new(comp);
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::Corrupt(format!("unzip failed: {e}")))?;
    if out.len() != unzipsize {
        return Err(Error::Corrupt(format!(
            "unzip size mismatch (expected {unzipsize}, got {})",
            out.len()
        )));
    }
    Ok(out)
}

/// Parse the face data headers of a level from its inflated level header.
pub(crate) fn parse_face_data_headers(buf: &[u8]) -> Vec<FaceDataHeader> {
    buf.chunks_exact(FACE_DATA_HEADER_SIZE)
        .map(|c| FaceDataHeader {
            data: format::u32_at(c, 0),
        })
        .collect()
}

/// Positions of the faces in a level that are flagged "large" and therefore
/// carry a 64-bit size in an extra table after the level header.
pub(crate) fn large_face_positions(fdh: &[FaceDataHeader]) -> Vec<usize> {
    (0..fdh.len()).filter(|&f| fdh[f].is_large_face()).collect()
}

/// Compute the absolute file offset of every face in a level.
///
/// `base` is the offset just past the level header; `large_table` holds the
/// `8 * nlarge` bytes of 64-bit sizes that follow it (empty when no face is
/// flagged large).
pub(crate) fn level_offsets(
    fdh: &[FaceDataHeader],
    base: u64,
    large_table: &[u8],
) -> Result<Vec<u64>> {
    let nfaces = fdh.len();
    let mut offsets = vec![0u64; nfaces];
    let mut large_faces = Vec::new();
    let mut offset = base;
    for (f, off) in offsets.iter_mut().enumerate() {
        *off = offset;
        if fdh[f].is_large_face() {
            large_faces.push(f);
        } else {
            offset += fdh[f].blocksize() as u64;
        }
    }
    if !large_faces.is_empty() {
        let nlarge = large_faces.len();
        if large_table.len() < 8 * nlarge {
            return Err(Error::Corrupt("truncated large face table".into()));
        }
        let mut extra = (8 * nlarge) as u64;
        let mut f = 0usize;
        for (i, &lf) in large_faces.iter().enumerate() {
            while f <= lf {
                offsets[f] += extra;
                f += 1;
            }
            extra += format::u64_at(large_table, 8 * i);
        }
        while f < nfaces {
            offsets[f] += extra;
            f += 1;
        }
    }
    Ok(offsets)
}

/// Parse the 6-byte tile block header into the tile resolution and the
/// compressed size of the tile directory that follows it.
pub(crate) fn parse_tile_header(head: &[u8], res: Res) -> Result<(Res, usize)> {
    let tile_res = Res {
        ulog2: head[0] as i8,
        vlog2: head[1] as i8,
    };
    let tileheadersize = format::u32_at(head, 2) as usize;
    if tile_res.ulog2 < 0 || tile_res.vlog2 < 0 {
        return Err(Error::Corrupt("negative tile res".into()));
    }
    if tile_res.ulog2 > res.ulog2 || tile_res.vlog2 > res.vlog2 {
        return Err(Error::Corrupt("tile res larger than face res".into()));
    }
    Ok((tile_res, tileheadersize))
}

/// Build a tile directory from the inflated tile header block.
///
/// `base` is the absolute file offset of the first tile's data, i.e. the
/// offset of the tiled block plus 6 plus the compressed directory size.
pub(crate) fn parse_tile_dir(dir: &[u8], tile_res: Res, res: Res, base: u64) -> Result<TileDir> {
    let ntiles = res.ntiles(tile_res);
    let fdh = parse_face_data_headers(dir);
    if fdh.len() != ntiles {
        return Err(Error::Corrupt("tile directory size mismatch".into()));
    }
    let mut offsets = vec![0u64; ntiles];
    let mut offset = base;
    for (t, off) in offsets.iter_mut().enumerate() {
        // A tile directory has no 64-bit size table, so a tile flagged
        // "large" would silently break the offset chain for every tile
        // after it.  Reject rather than mis-read.
        if fdh[t].is_large_face() {
            return Err(Error::Unsupported(
                "large tile data blocks are not supported".into(),
            ));
        }
        if fdh[t].encoding() == Encoding::Tiled {
            return Err(Error::Corrupt("nested tiled face data".into()));
        }
        *off = offset;
        offset += fdh[t].blocksize() as u64;
    }
    Ok(TileDir {
        tile_res,
        fdh,
        offsets,
    })
}

/// Decode a constant block: a single pixel value.
///
/// Alpha premultiplication is applied only at level 0, matching the C++
/// library (reduction levels are premultiplied already).
pub(crate) fn decode_constant(raw: &[u8], info: &FileInfo, levelid: usize) -> Vec<u8> {
    let mut pixel = raw[..info.pixel_size].to_vec();
    if levelid == 0 && info.premultiply && info.header.has_alpha() {
        utils::multalpha(
            &mut pixel,
            1,
            info.data_type,
            info.num_channels(),
            info.header.alphachan as usize,
        );
    }
    pixel
}

/// Decode a zipped or difference-zipped block into interleaved pixels.
pub(crate) fn decode_packed(
    raw: &[u8],
    res: Res,
    encoding: Encoding,
    info: &FileInfo,
    levelid: usize,
) -> Result<Vec<u8>> {
    let uw = res.u();
    let vw = res.v();
    let npixels = uw * vw;
    let unpacked_size = info.pixel_size * npixels;
    let mut tmp = unzip(raw, unpacked_size)?;
    if encoding == Encoding::DiffZipped {
        utils::decode_difference(&mut tmp, info.data_type);
    }
    let mut data = vec![0u8; unpacked_size];
    utils::interleave(
        &tmp,
        uw,
        vw,
        &mut data,
        uw * info.pixel_size,
        info.data_type,
        info.num_channels(),
    );
    if levelid == 0 && info.premultiply && info.header.has_alpha() {
        utils::multalpha(
            &mut data,
            npixels,
            info.data_type,
            info.num_channels(),
            info.header.alphachan as usize,
        );
    }
    Ok(data)
}

/// Apply one reduction step, producing `dst_res` from packed `src_res` data.
pub(crate) fn reduce_step(
    src: &[u8],
    src_res: Res,
    dst_res: Res,
    kind: ReduceKind,
    info: &FileInfo,
) -> Vec<u8> {
    let sstride = src_res.u() * info.pixel_size;
    let dstride = dst_res.u() * info.pixel_size;
    let mut data = vec![0u8; dst_res.size() * info.pixel_size];
    let f = match kind {
        ReduceKind::U => utils::reduce_u,
        ReduceKind::V => utils::reduce_v,
        ReduceKind::Tri => utils::reduce_tri,
    };
    f(
        src,
        sstride,
        src_res.u(),
        src_res.v(),
        &mut data,
        dstride,
        info.data_type,
        info.num_channels(),
    );
    data
}
