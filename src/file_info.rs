//! Immutable, parsed-at-open state of a Ptex file.
//!
//! Everything in [`FileInfo`] is read once when the file is opened and never
//! changes afterwards, so it can be shared freely between threads.  Both the
//! single-threaded [`crate::PtexReader`] and the shared reader hold one.

use std::io::{Read, Seek, SeekFrom};

use crate::error::{Error, Result};
use crate::format::{
    self, ExtHeader, Header, LevelInfo, EXT_HEADER_SIZE, FACE_INFO_SIZE, HEADER_SIZE,
    LEVEL_INFO_SIZE, MAGIC,
};
use crate::types::{BorderMode, DataType, EdgeFilterMode, FaceInfo, MeshType, Res};
use crate::utils;

/// How a request for a face at a particular resolution resolves against the
/// file.
///
/// This is the routing that C++ `PtexReader::getData(faceid, res)` performs;
/// it is shared by every entry point so that the tile API, the whole-face
/// API and the single-texel API can never disagree about where data lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaceSource {
    /// A single pixel taken from the file's constant-data block.
    Constant,
    /// A data block stored in the file.
    Stored {
        /// Reduction level the block lives in (0 == full resolution).
        levelid: usize,
        /// Index of the face within that level's face table.
        facepos: usize,
    },
    /// Not stored at this resolution; must be computed by reduction.
    Reduced,
}

/// Which reduction kernel produces a resolution from the next larger one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReduceKind {
    /// Blend pairs of texels in u.
    U,
    /// Blend pairs of texels in v.
    V,
    /// Triangle 2x2 reduction.
    Tri,
}

/// Parsed header, face table and level table of a Ptex file.
pub(crate) struct FileInfo {
    pub premultiply: bool,
    pub header: Header,
    pub ext_header: ExtHeader,
    pub mesh_type: MeshType,
    pub data_type: DataType,
    pub pixel_size: usize,

    pub metadata_pos: u64,
    pub lmdheader_pos: u64,

    pub face_info: Vec<FaceInfo>,
    pub rfaceids: Vec<u32>,
    pub const_data: Vec<u8>,
    pub level_info: Vec<LevelInfo>,
    pub level_pos: Vec<u64>,
}

impl FileInfo {
    /// Read and parse the header, face table, constant data and level table.
    ///
    /// Leaves the stream position unspecified; callers seek explicitly.
    pub fn read<R: Read + Seek>(io: &mut R, premultiply: bool) -> Result<Self> {
        let mut hbuf = [0u8; HEADER_SIZE];
        io.read_exact(&mut hbuf)?;
        let header = Header::parse(&hbuf);
        if header.magic != MAGIC {
            return Err(Error::NotAPtexFile);
        }
        if header.version != 1 {
            return Err(Error::UnsupportedVersion(header.version));
        }
        let mesh_type = MeshType::from_u32(header.meshtype)?;
        let data_type = DataType::from_u32(header.datatype)?;
        let pixel_size = header.pixel_size()?;

        // read extended header (may be shorter or longer than we know;
        // extra bytes are skipped, missing bytes read as zero)
        let mut ebuf = [0u8; EXT_HEADER_SIZE];
        let ext_read = (header.extheadersize as usize).min(EXT_HEADER_SIZE);
        io.read_exact(&mut ebuf[..ext_read])?;
        let ext_header = ExtHeader::parse(&ebuf);

        // compute offsets of the various blocks
        let mut pos = (HEADER_SIZE as u64) + header.extheadersize as u64;
        let faceinfo_pos = pos;
        pos += header.faceinfosize as u64;
        let constdata_pos = pos;
        pos += header.constdatasize as u64;
        let levelinfo_pos = pos;
        pos += header.levelinfosize as u64;
        let leveldata_pos = pos;
        pos += header.leveldatasize;
        let metadata_pos = pos;
        pos += header.metadatazipsize as u64;
        pos += 8; // compatibility barrier
        let lmdheader_pos = pos;

        // face info table
        let nfaces = header.nfaces as usize;
        io.seek(SeekFrom::Start(faceinfo_pos))?;
        let buf = read_zip_from(io, header.faceinfosize as usize, FACE_INFO_SIZE * nfaces)?;
        let face_info: Vec<FaceInfo> = buf
            .chunks_exact(FACE_INFO_SIZE)
            .map(format::parse_face_info)
            .collect();
        let rfaceids = utils::gen_rfaceids(&face_info);

        // constant data block
        io.seek(SeekFrom::Start(constdata_pos))?;
        let mut const_data = read_zip_from(io, header.constdatasize as usize, pixel_size * nfaces)?;
        if premultiply && header.has_alpha() {
            utils::multalpha(
                &mut const_data,
                nfaces,
                data_type,
                header.nchannels as usize,
                header.alphachan as usize,
            );
        }

        // level info table (not compressed)
        let nlevels = header.nlevels as usize;
        if header.levelinfosize as usize != LEVEL_INFO_SIZE * nlevels {
            return Err(Error::Corrupt("level info size mismatch".into()));
        }
        io.seek(SeekFrom::Start(levelinfo_pos))?;
        let mut lbuf = vec![0u8; LEVEL_INFO_SIZE * nlevels];
        io.read_exact(&mut lbuf)?;
        let level_info: Vec<LevelInfo> = lbuf
            .chunks_exact(LEVEL_INFO_SIZE)
            .map(LevelInfo::parse)
            .collect();

        let mut level_pos = Vec::with_capacity(nlevels);
        let mut pos = leveldata_pos;
        for li in &level_info {
            level_pos.push(pos);
            pos += li.leveldatasize;
        }

        Ok(FileInfo {
            premultiply,
            header,
            ext_header,
            mesh_type,
            data_type,
            pixel_size,
            metadata_pos,
            lmdheader_pos,
            face_info,
            rfaceids,
            const_data,
            level_info,
            level_pos,
        })
    }

    pub fn num_faces(&self) -> usize {
        self.header.nfaces as usize
    }

    pub fn num_channels(&self) -> usize {
        self.header.nchannels as usize
    }

    pub fn num_levels(&self) -> usize {
        self.header.nlevels as usize
    }

    pub fn u_border_mode(&self) -> BorderMode {
        BorderMode::from_u16(self.ext_header.ubordermode)
    }

    pub fn v_border_mode(&self) -> BorderMode {
        BorderMode::from_u16(self.ext_header.vbordermode)
    }

    pub fn edge_filter_mode(&self) -> EdgeFilterMode {
        EdgeFilterMode::from_u16(self.ext_header.edgefiltermode)
    }

    pub fn face_info(&self, faceid: usize) -> Result<&FaceInfo> {
        self.face_info.get(faceid).ok_or(Error::FaceOutOfRange {
            faceid: faceid as i32,
            nfaces: self.header.nfaces,
        })
    }

    pub fn constant_data(&self, faceid: usize) -> Result<&[u8]> {
        if faceid >= self.num_faces() {
            return Err(Error::FaceOutOfRange {
                faceid: faceid as i32,
                nfaces: self.header.nfaces,
            });
        }
        Ok(&self.const_data[faceid * self.pixel_size..(faceid + 1) * self.pixel_size])
    }

    /// Number of mipmap levels a face has, counting level 0 (full
    /// resolution).  The last level reduces the smaller dimension to one
    /// texel.
    pub fn face_num_levels(&self, faceid: usize) -> Result<usize> {
        let fi = self.face_info(faceid)?;
        Ok(fi.res.ulog2.min(fi.res.vlog2).max(0) as usize + 1)
    }

    /// Resolution of a face reduced by `level` mipmap levels.
    pub fn face_level_res(&self, faceid: usize, level: usize) -> Result<Res> {
        let fi = *self.face_info(faceid)?;
        if level >= self.face_num_levels(faceid)? {
            return Err(Error::Unsupported(
                "reductions below 1 pixel are not supported".into(),
            ));
        }
        let level = level as i8;
        Ok(Res::new(fi.res.ulog2 - level, fi.res.vlog2 - level))
    }

    /// True if a resolution is stored in the file for this face, and so can
    /// be read (and streamed tile by tile) directly rather than computed by
    /// reduction.  Performs no I/O.
    pub fn is_res_stored(&self, faceid: usize, res: Res) -> Result<bool> {
        let fi = *self.face_info(faceid)?;
        if fi.is_constant() || res.is_one() {
            return Ok(true);
        }
        let redu = fi.res.ulog2 - res.ulog2;
        let redv = fi.res.vlog2 - res.vlog2;
        if redu == 0 && redv == 0 {
            return Ok(faceid < self.level_nfaces(0));
        }
        if redu == redv && redu > 0 && (redu as usize) < self.num_levels() {
            let levelid = redu as usize;
            return Ok((self.rfaceids[faceid] as usize) < self.level_nfaces(levelid));
        }
        Ok(false)
    }

    /// Number of leading mipmap levels of a face that are stored in the file,
    /// counting level 0.
    pub fn num_stored_levels(&self, faceid: usize) -> Result<usize> {
        let nlevels = self.face_num_levels(faceid)?;
        let mut n = 0;
        for level in 0..nlevels {
            let res = self.face_level_res(faceid, level)?;
            if !self.is_res_stored(faceid, res)? {
                break;
            }
            n += 1;
        }
        Ok(n)
    }

    fn level_nfaces(&self, levelid: usize) -> usize {
        self.level_info
            .get(levelid)
            .map_or(0, |li| li.nfaces as usize)
    }

    /// Resolve a `(faceid, res)` request to its location in the file.
    ///
    /// Mirrors the dispatch of C++ `PtexReader::getData(faceid, res)`.  The
    /// validity checks for negative resolutions, enlargements and
    /// anisotropic triangle reductions are performed here so that every
    /// entry point rejects them identically.
    pub fn resolve(&self, faceid: usize, res: Res) -> Result<FaceSource> {
        let fi = *self.face_info(faceid)?;
        if fi.is_constant() || res.is_one() {
            return Ok(FaceSource::Constant);
        }

        let redu = fi.res.ulog2 - res.ulog2;
        let redv = fi.res.vlog2 - res.vlog2;

        if redu == 0 && redv == 0 {
            if faceid < self.level_nfaces(0) {
                return Ok(FaceSource::Stored {
                    levelid: 0,
                    facepos: faceid,
                });
            }
            return Err(Error::Corrupt("face missing from level 0".into()));
        }

        if redu == redv && redu > 0 && (redu as usize) < self.num_levels() {
            // symmetric reduction - it may be stored on disk
            let levelid = redu as usize;
            let facepos = self.rfaceids[faceid] as usize;
            if facepos < self.level_nfaces(levelid) {
                return Ok(FaceSource::Stored { levelid, facepos });
            }
        }

        // dynamic reduction required
        if res.ulog2 < 0 || res.vlog2 < 0 {
            return Err(Error::Unsupported(
                "reductions below 1 pixel are not supported".into(),
            ));
        }
        if redu < 0 || redv < 0 {
            return Err(Error::Unsupported("enlargements are not supported".into()));
        }
        if self.mesh_type == MeshType::Triangle && redu != redv {
            return Err(Error::Unsupported(
                "anisotropic reductions are not supported for triangle meshes".into(),
            ));
        }
        Ok(FaceSource::Reduced)
    }

    /// Resolution to reduce from, and the kernel to use, when `res` has to be
    /// computed dynamically.  Only valid after [`FileInfo::resolve`] returned
    /// [`FaceSource::Reduced`].
    pub fn reduction_source(&self, faceid: usize, res: Res) -> Result<(Res, ReduceKind)> {
        let fi = *self.face_info(faceid)?;
        if self.mesh_type == MeshType::Triangle {
            return Ok((Res::new(res.ulog2 + 1, res.vlog2 + 1), ReduceKind::Tri));
        }
        let redu = fi.res.ulog2 - res.ulog2;
        let redv = fi.res.vlog2 - res.vlog2;
        // determine which direction to blend: for symmetric face blends,
        // alternate u and v blending
        let blendu = if redu == redv {
            res.ulog2 & 1 != 0
        } else {
            redu > redv
        };
        if blendu {
            Ok((Res::new(res.ulog2 + 1, res.vlog2), ReduceKind::U))
        } else {
            Ok((Res::new(res.ulog2, res.vlog2 + 1), ReduceKind::V))
        }
    }
}

/// Read `zipsize` bytes from `io` and zlib-decompress them into exactly
/// `unzipsize` bytes.
pub(crate) fn read_zip_from<R: Read>(
    io: &mut R,
    zipsize: usize,
    unzipsize: usize,
) -> Result<Vec<u8>> {
    let mut comp = vec![0u8; zipsize];
    io.read_exact(&mut comp)?;
    crate::decode::unzip(&comp, unzipsize)
}
