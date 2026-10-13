//! On-disk file format structures (ported from `PtexIO.h`).
//!
//! All multi-byte values in a Ptex file are stored little-endian.

use crate::error::{Error, Result};

/// File magic: "Ptex" as a little-endian u32.
pub const MAGIC: u32 =
    (b'P' as u32) | ((b't' as u32) << 8) | ((b'e' as u32) << 16) | ((b'x' as u32) << 24);

/// Size of the fixed file header in bytes.
pub const HEADER_SIZE: usize = 64;
/// Size of the extended header in bytes.
pub const EXT_HEADER_SIZE: usize = 40;
/// Size of a level info record in bytes.
pub const LEVEL_INFO_SIZE: usize = 16;
/// Size of a face data header in bytes.
pub const FACE_DATA_HEADER_SIZE: usize = 4;
/// Size of a face info record in bytes.
pub const FACE_INFO_SIZE: usize = 20;

/// Fixed file header.
#[derive(Debug, Clone, Copy, Default)]
pub struct Header {
    /// File magic; must equal [`MAGIC`].
    pub magic: u32,
    /// Major format version; only version 1 is supported.
    pub version: u32,
    /// Raw [`MeshType`](crate::MeshType) value.
    pub meshtype: u32,
    /// Raw [`DataType`](crate::DataType) value.
    pub datatype: u32,
    /// Index of the alpha channel, or -1 if there is none.
    pub alphachan: i32,
    /// Number of channels per texel.
    pub nchannels: u16,
    /// Number of stored resolution levels, including level 0.
    pub nlevels: u16,
    /// Number of faces in the mesh.
    pub nfaces: u32,
    /// Size in bytes of the extended header that follows this one.
    pub extheadersize: u32,
    /// Size in bytes of the zipped face info table.
    pub faceinfosize: u32,
    /// Size in bytes of the zipped constant-data block.
    pub constdatasize: u32,
    /// Size in bytes of the (uncompressed) level info table.
    pub levelinfosize: u32,
    /// Minor format version.
    pub minorversion: u32,
    /// Total size in bytes of all level data blocks.
    pub leveldatasize: u64,
    /// Size in bytes of the zipped small meta data block.
    pub metadatazipsize: u32,
    /// Size in bytes of the small meta data block once unzipped.
    pub metadatamemsize: u32,
}

impl Header {
    /// Decode a header from its 64-byte on-disk representation.
    pub fn parse(b: &[u8; HEADER_SIZE]) -> Header {
        Header {
            magic: u32_at(b, 0),
            version: u32_at(b, 4),
            meshtype: u32_at(b, 8),
            datatype: u32_at(b, 12),
            alphachan: u32_at(b, 16) as i32,
            nchannels: u16_at(b, 20),
            nlevels: u16_at(b, 22),
            nfaces: u32_at(b, 24),
            extheadersize: u32_at(b, 28),
            faceinfosize: u32_at(b, 32),
            constdatasize: u32_at(b, 36),
            levelinfosize: u32_at(b, 40),
            minorversion: u32_at(b, 44),
            leveldatasize: u64_at(b, 48),
            metadatazipsize: u32_at(b, 56),
            metadatamemsize: u32_at(b, 60),
        }
    }

    /// Size of a pixel in bytes (data size * number of channels).
    pub fn pixel_size(&self) -> Result<usize> {
        let dt = crate::types::DataType::from_u32(self.datatype)?;
        Ok(dt.size() * self.nchannels as usize)
    }

    /// True if `alphachan` names a valid channel.
    pub fn has_alpha(&self) -> bool {
        self.alphachan >= 0 && self.alphachan < self.nchannels as i32
    }
}

/// Extended header (may be truncated or absent in older files;
/// missing fields read as zero).
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)] // unused fields kept to document the on-disk layout
pub struct ExtHeader {
    /// Raw [`BorderMode`](crate::BorderMode) in u.
    pub ubordermode: u16,
    /// Padding; unused.
    pub pad: u16,
    /// Raw [`BorderMode`](crate::BorderMode) in v.
    pub vbordermode: u16,
    /// Raw [`EdgeFilterMode`](crate::EdgeFilterMode).
    pub edgefiltermode: u16,
    /// Size in bytes of the zipped large meta data header.
    pub lmdheaderzipsize: u32,
    /// Size in bytes of the large meta data header once unzipped.
    pub lmdheadermemsize: u32,
    /// Total size in bytes of the large meta data blocks.
    pub lmddatasize: u64,
    /// Total size of the edit blocks appended after the main data, or 0 if
    /// the file has never been edited.
    pub editdatasize: u64,
    /// Absolute file offset of the first edit block; equal to the file
    /// length when there are no edits.
    pub editdatapos: u64,
}

impl ExtHeader {
    /// Decode an extended header from its 40-byte on-disk representation.
    pub fn parse(b: &[u8; EXT_HEADER_SIZE]) -> ExtHeader {
        ExtHeader {
            ubordermode: u16_at(b, 0),
            pad: u16_at(b, 2),
            vbordermode: u16_at(b, 4),
            edgefiltermode: u16_at(b, 6),
            lmdheaderzipsize: u32_at(b, 8),
            lmdheadermemsize: u32_at(b, 12),
            lmddatasize: u64_at(b, 16),
            editdatasize: u64_at(b, 24),
            editdatapos: u64_at(b, 32),
        }
    }
}

/// Per-level info record.
#[derive(Debug, Clone, Copy, Default)]
pub struct LevelInfo {
    /// Size in bytes of this level's data, including its header.
    pub leveldatasize: u64,
    /// Size in bytes of the zipped face data header table at the start
    /// of the level.
    pub levelheadersize: u32,
    /// Number of faces stored in this level.
    pub nfaces: u32,
}

impl LevelInfo {
    /// Decode a level info record from its 16-byte on-disk representation.
    pub fn parse(b: &[u8]) -> LevelInfo {
        LevelInfo {
            leveldatasize: u64_at(b, 0),
            levelheadersize: u32_at(b, 8),
            nfaces: u32_at(b, 12),
        }
    }
}

/// How face data is encoded on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// A single pixel repeated across the whole face.
    Constant,
    /// Raw pixels, deflate-compressed.
    Zipped,
    /// Pixels delta-encoded along each row, then deflate-compressed.
    DiffZipped,
    /// Split into independently encoded tiles behind a tile header.
    Tiled,
}

/// Face data header: block size (bits 0..29) and encoding (bits 30..31).
#[derive(Debug, Clone, Copy, Default)]
pub struct FaceDataHeader {
    /// Packed block size and encoding.
    pub data: u32,
}

/// A face whose block size reaches this sentinel is a "large face"; its
/// actual size is stored in a separate 64-bit large-face header.
pub const BLOCKSIZE_MAX: u32 = 0x3fff_ffff;

impl FaceDataHeader {
    /// Size in bytes of the face's data block.
    pub fn blocksize(self) -> u32 {
        self.data & BLOCKSIZE_MAX
    }

    /// How the face's data block is encoded.
    pub fn encoding(self) -> Encoding {
        match (self.data >> 30) & 3 {
            0 => Encoding::Constant,
            1 => Encoding::Zipped,
            2 => Encoding::DiffZipped,
            _ => Encoding::Tiled,
        }
    }

    /// True if the real block size lives in a separate large-face header.
    pub fn is_large_face(self) -> bool {
        self.blocksize() == BLOCKSIZE_MAX
    }
}

/// Parse a `FaceInfo` record from its 20-byte on-disk representation.
pub fn parse_face_info(b: &[u8]) -> crate::types::FaceInfo {
    crate::types::FaceInfo {
        res: crate::types::Res {
            ulog2: b[0] as i8,
            vlog2: b[1] as i8,
        },
        adjedges: b[2],
        flags: b[3],
        adjfaces: [
            u32_at(b, 4) as i32,
            u32_at(b, 8) as i32,
            u32_at(b, 12) as i32,
            u32_at(b, 16) as i32,
        ],
    }
}

/// Read a little-endian `u16` at `off`; panics if out of bounds.
pub fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

/// Read a little-endian `u32` at `off`; panics if out of bounds.
pub fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Read a little-endian `u64` at `off`; panics if out of bounds.
pub fn u64_at(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

/// Checked variant of [`u32_at`] for parsing untrusted buffers.
pub fn checked_u32_at(b: &[u8], off: usize) -> Result<u32> {
    if off + 4 > b.len() {
        return Err(Error::Corrupt("truncated block".into()));
    }
    Ok(u32_at(b, off))
}
