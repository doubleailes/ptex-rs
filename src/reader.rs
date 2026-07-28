//! The Ptex file reader.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};
use crate::format::{
    self, Encoding, ExtHeader, FaceDataHeader, Header, LevelInfo, EXT_HEADER_SIZE,
    FACE_DATA_HEADER_SIZE, FACE_INFO_SIZE, HEADER_SIZE, LEVEL_INFO_SIZE, MAGIC,
};
use crate::metadata::{self, MetaData};
use crate::types::{BorderMode, DataType, EdgeFilterMode, FaceInfo, MeshType, Res};
use crate::utils;

/// Cached per-level index: one face data header and file offset per face
/// stored in the level.
#[derive(Debug, Clone)]
struct Level {
    fdh: Vec<FaceDataHeader>,
    offsets: Vec<u64>,
}

/// Face data as read from the file.
enum FaceData {
    /// A single pixel value.
    Constant(Vec<u8>),
    /// Interleaved pixel data at the given resolution.
    Packed { res: Res, data: Vec<u8> },
    /// A face split into tiles, each tile encoded separately.
    Tiled {
        res: Res,
        tileres: Res,
        fdh: Vec<FaceDataHeader>,
        offsets: Vec<u64>,
        levelid: usize,
    },
}

/// Reader for Ptex texture files.
///
/// ```no_run
/// let mut tx = ptex::PtexReader::open("model.ptx")?;
/// println!("faces: {}", tx.num_faces());
/// let data = tx.get_data(0)?; // full-res pixel data for face 0
/// # Ok::<(), ptex::Error>(())
/// ```
///
/// Reading methods take `&mut self` because data is read from the
/// underlying stream on demand (level indexes are cached; pixel data is
/// not).
pub struct PtexReader<R = BufReader<File>> {
    io: R,
    premultiply: bool,
    header: Header,
    ext_header: ExtHeader,
    mesh_type: MeshType,
    data_type: DataType,
    pixel_size: usize,

    faceinfo_pos: u64,
    constdata_pos: u64,
    levelinfo_pos: u64,
    leveldata_pos: u64,
    metadata_pos: u64,
    lmdheader_pos: u64,

    face_info: Vec<FaceInfo>,
    rfaceids: Vec<u32>,
    const_data: Vec<u8>,
    level_info: Vec<LevelInfo>,
    level_pos: Vec<u64>,
    levels: Vec<Option<Level>>,
    metadata: Option<MetaData>,
}

impl PtexReader<BufReader<File>> {
    /// Open a Ptex file from the file system.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, false)
    }

    /// Open a Ptex file, optionally premultiplying color channels by the
    /// alpha channel (matching the `premultiply` flag of the C++ API).
    pub fn open_with_options(path: impl AsRef<Path>, premultiply: bool) -> Result<Self> {
        Self::new_with_options(BufReader::new(File::open(path)?), premultiply)
    }
}

impl<R: Read + Seek> PtexReader<R> {
    /// Read a Ptex file from any seekable stream.
    pub fn new(io: R) -> Result<Self> {
        Self::new_with_options(io, false)
    }

    /// Read a Ptex file from any seekable stream, optionally premultiplying
    /// color channels by the alpha channel.
    pub fn new_with_options(mut io: R, premultiply: bool) -> Result<Self> {
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

        let mut reader = PtexReader {
            io,
            premultiply,
            header,
            ext_header,
            mesh_type,
            data_type,
            pixel_size,
            faceinfo_pos,
            constdata_pos,
            levelinfo_pos,
            leveldata_pos,
            metadata_pos,
            lmdheader_pos,
            face_info: Vec::new(),
            rfaceids: Vec::new(),
            const_data: Vec::new(),
            level_info: Vec::new(),
            level_pos: Vec::new(),
            levels: Vec::new(),
            metadata: None,
        };
        reader.read_face_info()?;
        reader.read_const_data()?;
        reader.read_level_info()?;
        Ok(reader)
    }

    /// Type of the base mesh (quad or triangle).
    pub fn mesh_type(&self) -> MeshType {
        self.mesh_type
    }

    /// Type of the pixel data.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Index of the alpha channel, or -1 if the file has no alpha channel.
    pub fn alpha_channel(&self) -> i32 {
        self.header.alphachan
    }

    /// Number of channels per pixel.
    pub fn num_channels(&self) -> usize {
        self.header.nchannels as usize
    }

    /// Number of faces in the file.
    pub fn num_faces(&self) -> usize {
        self.header.nfaces as usize
    }

    /// True if the file has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.header.has_alpha()
    }

    /// True if the file stores precomputed mipmap (reduction) levels.
    pub fn has_mip_maps(&self) -> bool {
        self.header.nlevels > 1
    }

    /// Number of stored resolution levels (level 0 is full resolution).
    pub fn num_levels(&self) -> usize {
        self.header.nlevels as usize
    }

    /// Size of a single pixel, in bytes.
    pub fn pixel_size(&self) -> usize {
        self.pixel_size
    }

    /// Minor version of the file format the file was written with.
    pub fn minor_version(&self) -> u32 {
        self.header.minorversion
    }

    /// Border mode in the u direction.
    pub fn u_border_mode(&self) -> BorderMode {
        BorderMode::from_u16(self.ext_header.ubordermode)
    }

    /// Border mode in the v direction.
    pub fn v_border_mode(&self) -> BorderMode {
        BorderMode::from_u16(self.ext_header.vbordermode)
    }

    /// Edge filter mode.
    pub fn edge_filter_mode(&self) -> EdgeFilterMode {
        EdgeFilterMode::from_u16(self.ext_header.edgefiltermode)
    }

    /// Access resolution and adjacency information for a face.
    pub fn face_info(&self, faceid: usize) -> Result<&FaceInfo> {
        self.face_info.get(faceid).ok_or(Error::FaceOutOfRange {
            faceid: faceid as i32,
            nfaces: self.header.nfaces,
        })
    }

    /// All per-face info records.
    pub fn face_infos(&self) -> &[FaceInfo] {
        &self.face_info
    }

    /// Access the constant (average) pixel value of a face as raw data.
    pub fn constant_data(&self, faceid: usize) -> Result<&[u8]> {
        if faceid >= self.num_faces() {
            return Err(Error::FaceOutOfRange {
                faceid: faceid as i32,
                nfaces: self.header.nfaces,
            });
        }
        Ok(&self.const_data[faceid * self.pixel_size..(faceid + 1) * self.pixel_size])
    }

    /// Access the file's meta data (loaded and cached on first access).
    pub fn metadata(&mut self) -> Result<&MetaData> {
        if self.metadata.is_none() {
            let md = self.read_metadata()?;
            self.metadata = Some(md);
        }
        Ok(self.metadata.as_ref().unwrap())
    }

    /// Read the full-resolution pixel data for a face.
    ///
    /// Returns interleaved pixel data in v-major order (`res.v()` rows of
    /// `res.u()` pixels of `pixel_size()` bytes).
    pub fn get_data(&mut self, faceid: usize) -> Result<Vec<u8>> {
        let res = self.face_info(faceid)?.res;
        self.get_data_at_res(faceid, res)
    }

    /// Read pixel data for a face at the given resolution.
    ///
    /// If `res` matches a stored resolution (the full resolution or a
    /// stored mipmap level) the data is read directly; otherwise a
    /// reduction is computed from the next larger resolution, matching the
    /// behavior of the C++ library.  Enlargements are not supported.
    pub fn get_data_at_res(&mut self, faceid: usize, res: Res) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; res.size() * self.pixel_size];
        self.get_data_into(faceid, res, &mut buf, 0)?;
        Ok(buf)
    }

    /// Read pixel data for a face at the given resolution into a
    /// caller-provided buffer.
    ///
    /// `stride` is the row stride of `buffer` in bytes; 0 means tightly
    /// packed (`res.u() * pixel_size()`).
    pub fn get_data_into(
        &mut self,
        faceid: usize,
        res: Res,
        buffer: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let resu = res.u();
        let resv = res.v();
        let rowlen = self.pixel_size * resu;
        let stride = if stride == 0 { rowlen } else { stride };
        if stride < rowlen || buffer.len() < stride * (resv - 1) + rowlen {
            return Err(Error::Unsupported(
                "buffer too small for requested res".into(),
            ));
        }

        let data = self.face_data(faceid, res)?;
        match data {
            FaceData::Constant(pixel) => {
                utils::fill(&pixel, buffer, stride, resu, resv, self.pixel_size);
            }
            FaceData::Packed { data, .. } => {
                utils::copy_rows(&data, rowlen, buffer, stride, resv, rowlen);
            }
            FaceData::Tiled {
                tileres,
                fdh,
                offsets,
                levelid,
                ..
            } => {
                let ntilesu = res.ntilesu(tileres);
                let ntilesv = res.ntilesv(tileres);
                let tileures = tileres.u();
                let tilevres = tileres.v();
                let tilerowlen = self.pixel_size * tileures;
                let mut tile = 0usize;
                for i in 0..ntilesv {
                    for j in 0..ntilesu {
                        let t = self.read_face_data(offsets[tile], fdh[tile], tileres, levelid)?;
                        let dst_off = i * stride * tilevres + j * tilerowlen;
                        let dst = &mut buffer[dst_off..];
                        match t {
                            FaceData::Constant(pixel) => {
                                utils::fill(
                                    &pixel,
                                    dst,
                                    stride,
                                    tileures,
                                    tilevres,
                                    self.pixel_size,
                                );
                            }
                            FaceData::Packed { data, .. } => {
                                utils::copy_rows(
                                    &data, tilerowlen, dst, stride, tilevres, tilerowlen,
                                );
                            }
                            FaceData::Tiled { .. } => {
                                return Err(Error::Corrupt("nested tiled face data".into()));
                            }
                        }
                        tile += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Read a single texel, converted to f32 channel values.
    ///
    /// `first_chan` selects the first channel to return and `nchannels`
    /// how many; the result is clipped to the channels available.
    /// Integer pixel formats are normalized to the 0..1 range.
    pub fn get_pixel(
        &mut self,
        faceid: usize,
        u: usize,
        v: usize,
        first_chan: usize,
        nchannels: usize,
    ) -> Result<Vec<f32>> {
        let nchan = nchannels.min(self.num_channels().saturating_sub(first_chan));
        if nchan == 0 {
            return Ok(Vec::new());
        }
        let fi = *self.face_info(faceid)?;
        let pixel: Vec<u8> = if fi.is_constant() {
            self.constant_data(faceid)?.to_vec()
        } else {
            let data = self.face_data(faceid, fi.res)?;
            self.pixel_from_face_data(&data, u, v)?
        };
        let mut out = vec![0f32; nchan];
        let dsize = self.data_type.size();
        utils::convert_to_float(
            &mut out,
            &pixel[first_chan * dsize..],
            self.data_type,
            nchan,
        );
        Ok(out)
    }

    // ---- internal reading helpers ----

    fn seek(&mut self, pos: u64) -> Result<()> {
        self.io.seek(SeekFrom::Start(pos))?;
        Ok(())
    }

    fn read_exact_vec(&mut self, size: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; size];
        self.io.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read `zipsize` bytes and zlib-decompress them into exactly
    /// `unzipsize` bytes.
    fn read_zip_block(&mut self, zipsize: usize, unzipsize: usize) -> Result<Vec<u8>> {
        let comp = self.read_exact_vec(zipsize)?;
        let mut out = Vec::with_capacity(unzipsize);
        let mut decoder = flate2::read::ZlibDecoder::new(&comp[..]);
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

    fn read_face_info(&mut self) -> Result<()> {
        let nfaces = self.header.nfaces as usize;
        self.seek(self.faceinfo_pos)?;
        let buf =
            self.read_zip_block(self.header.faceinfosize as usize, FACE_INFO_SIZE * nfaces)?;
        self.face_info = buf
            .chunks_exact(FACE_INFO_SIZE)
            .map(format::parse_face_info)
            .collect();
        self.rfaceids = utils::gen_rfaceids(&self.face_info);
        Ok(())
    }

    fn read_const_data(&mut self) -> Result<()> {
        let size = self.pixel_size * self.header.nfaces as usize;
        self.seek(self.constdata_pos)?;
        self.const_data = self.read_zip_block(self.header.constdatasize as usize, size)?;
        if self.premultiply && self.header.has_alpha() {
            let nchannels = self.num_channels();
            utils::multalpha(
                &mut self.const_data,
                self.header.nfaces as usize,
                self.data_type,
                nchannels,
                self.header.alphachan as usize,
            );
        }
        Ok(())
    }

    fn read_level_info(&mut self) -> Result<()> {
        let nlevels = self.header.nlevels as usize;
        if self.header.levelinfosize as usize != LEVEL_INFO_SIZE * nlevels {
            return Err(Error::Corrupt("level info size mismatch".into()));
        }
        self.seek(self.levelinfo_pos)?;
        let buf = self.read_exact_vec(LEVEL_INFO_SIZE * nlevels)?;
        self.level_info = buf
            .chunks_exact(LEVEL_INFO_SIZE)
            .map(LevelInfo::parse)
            .collect();

        self.levels = vec![None; nlevels];
        self.level_pos = Vec::with_capacity(nlevels);
        let mut pos = self.leveldata_pos;
        for li in &self.level_info {
            self.level_pos.push(pos);
            pos += li.leveldatasize;
        }
        Ok(())
    }

    /// Read and cache the face index (headers + offsets) of a level.
    fn ensure_level(&mut self, levelid: usize) -> Result<()> {
        if self.levels[levelid].is_some() {
            return Ok(());
        }
        let li = self.level_info[levelid];
        let nfaces = li.nfaces as usize;
        self.seek(self.level_pos[levelid])?;
        let buf =
            self.read_zip_block(li.levelheadersize as usize, FACE_DATA_HEADER_SIZE * nfaces)?;
        let fdh: Vec<FaceDataHeader> = buf
            .chunks_exact(FACE_DATA_HEADER_SIZE)
            .map(|c| FaceDataHeader {
                data: format::u32_at(c, 0),
            })
            .collect();

        // compute face offsets; faces marked "large" have 64-bit sizes
        // stored in an extra header after the level header
        let mut offsets = vec![0u64; nfaces];
        let mut large_faces = Vec::new();
        let mut offset = self.level_pos[levelid] + li.levelheadersize as u64;
        for f in 0..nfaces {
            offsets[f] = offset;
            if fdh[f].is_large_face() {
                large_faces.push(f);
            } else {
                offset += fdh[f].blocksize() as u64;
            }
        }
        if !large_faces.is_empty() {
            let nlarge = large_faces.len();
            let lf_header = self.read_exact_vec(8 * nlarge)?;
            let mut extra = (8 * nlarge) as u64;
            let mut f = 0usize;
            for (i, &lf) in large_faces.iter().enumerate() {
                while f <= lf {
                    offsets[f] += extra;
                    f += 1;
                }
                extra += format::u64_at(&lf_header, 8 * i);
            }
            while f < nfaces {
                offsets[f] += extra;
                f += 1;
            }
        }

        self.levels[levelid] = Some(Level { fdh, offsets });
        Ok(())
    }

    /// Get the (header, offset) pair for a face within a level, if stored.
    fn level_face_entry(
        &mut self,
        levelid: usize,
        facepos: usize,
    ) -> Result<Option<(FaceDataHeader, u64)>> {
        self.ensure_level(levelid)?;
        let level = self.levels[levelid].as_ref().unwrap();
        Ok(level
            .fdh
            .get(facepos)
            .map(|&fdh| (fdh, level.offsets[facepos])))
    }

    /// Read the data of a single face (or tile) from the file.
    fn read_face_data(
        &mut self,
        pos: u64,
        fdh: FaceDataHeader,
        res: Res,
        levelid: usize,
    ) -> Result<FaceData> {
        self.seek(pos)?;
        match fdh.encoding() {
            Encoding::Constant => {
                let mut pixel = self.read_exact_vec(self.pixel_size)?;
                if levelid == 0 && self.premultiply && self.header.has_alpha() {
                    utils::multalpha(
                        &mut pixel,
                        1,
                        self.data_type,
                        self.num_channels(),
                        self.header.alphachan as usize,
                    );
                }
                Ok(FaceData::Constant(pixel))
            }
            Encoding::Tiled => {
                let head = self.read_exact_vec(6)?;
                let tileres = Res {
                    ulog2: head[0] as i8,
                    vlog2: head[1] as i8,
                };
                let tileheadersize = format::u32_at(&head, 2) as usize;
                if tileres.ulog2 > res.ulog2 || tileres.vlog2 > res.vlog2 {
                    return Err(Error::Corrupt("tile res larger than face res".into()));
                }
                let ntiles = res.ntiles(tileres);
                let buf = self.read_zip_block(tileheadersize, FACE_DATA_HEADER_SIZE * ntiles)?;
                let fdh: Vec<FaceDataHeader> = buf
                    .chunks_exact(FACE_DATA_HEADER_SIZE)
                    .map(|c| FaceDataHeader {
                        data: format::u32_at(c, 0),
                    })
                    .collect();
                let mut offsets = vec![0u64; ntiles];
                let mut offset = pos + 6 + tileheadersize as u64;
                for t in 0..ntiles {
                    offsets[t] = offset;
                    offset += fdh[t].blocksize() as u64;
                }
                Ok(FaceData::Tiled {
                    res,
                    tileres,
                    fdh,
                    offsets,
                    levelid,
                })
            }
            Encoding::Zipped | Encoding::DiffZipped => {
                if fdh.is_large_face() {
                    return Err(Error::Corrupt("non-tiled large face".into()));
                }
                let uw = res.u();
                let vw = res.v();
                let npixels = uw * vw;
                let unpacked_size = self.pixel_size * npixels;
                let mut tmp = self.read_zip_block(fdh.blocksize() as usize, unpacked_size)?;
                if fdh.encoding() == Encoding::DiffZipped {
                    utils::decode_difference(&mut tmp, self.data_type);
                }
                let mut data = vec![0u8; unpacked_size];
                utils::interleave(
                    &tmp,
                    uw,
                    vw,
                    &mut data,
                    uw * self.pixel_size,
                    self.data_type,
                    self.num_channels(),
                );
                if levelid == 0 && self.premultiply && self.header.has_alpha() {
                    utils::multalpha(
                        &mut data,
                        npixels,
                        self.data_type,
                        self.num_channels(),
                        self.header.alphachan as usize,
                    );
                }
                Ok(FaceData::Packed { res, data })
            }
        }
    }

    /// Get face data at the requested resolution, mirroring the logic of
    /// `PtexReader::getData(faceid, res)` in the C++ library.
    fn face_data(&mut self, faceid: usize, res: Res) -> Result<FaceData> {
        let fi = *self.face_info(faceid)?;
        if fi.is_constant() || res.is_one() {
            return Ok(FaceData::Constant(self.constant_data(faceid)?.to_vec()));
        }

        // determine how many reduction levels are needed
        let redu = fi.res.ulog2 - res.ulog2;
        let redv = fi.res.vlog2 - res.vlog2;

        if redu == 0 && redv == 0 {
            // no reduction - get level zero (full) res face
            let (fdh, pos) = self
                .level_face_entry(0, faceid)?
                .ok_or_else(|| Error::Corrupt("face missing from level 0".into()))?;
            return self.read_face_data(pos, fdh, res, 0);
        }

        if redu == redv && redu > 0 && (redu as usize) < self.levels.len() {
            // symmetric reduction - it may be stored on disk
            let levelid = redu as usize;
            let rfaceid = self.rfaceids[faceid] as usize;
            if let Some((fdh, pos)) = self.level_face_entry(levelid, rfaceid)? {
                return self.read_face_data(pos, fdh, res, levelid);
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

        if self.mesh_type == MeshType::Triangle {
            if redu != redv {
                return Err(Error::Unsupported(
                    "anisotropic reductions are not supported for triangle meshes".into(),
                ));
            }
            let src_res = Res::new(res.ulog2 + 1, res.vlog2 + 1);
            let src = self.packed_face_data(faceid, src_res)?;
            let sstride = src_res.u() * self.pixel_size;
            let mut data = vec![0u8; res.size() * self.pixel_size];
            utils::reduce_tri(
                &src,
                sstride,
                src_res.u(),
                src_res.v(),
                &mut data,
                res.u() * self.pixel_size,
                self.data_type,
                self.num_channels(),
            );
            return Ok(FaceData::Packed { res, data });
        }

        // determine which direction to blend: for symmetric face blends,
        // alternate u and v blending
        let blendu = if redu == redv {
            res.ulog2 & 1 != 0
        } else {
            redu > redv
        };
        let src_res = if blendu {
            Res::new(res.ulog2 + 1, res.vlog2)
        } else {
            Res::new(res.ulog2, res.vlog2 + 1)
        };
        let src = self.packed_face_data(faceid, src_res)?;
        let sstride = src_res.u() * self.pixel_size;
        let mut data = vec![0u8; res.size() * self.pixel_size];
        let reduce_fn = if blendu {
            utils::reduce_u
        } else {
            utils::reduce_v
        };
        reduce_fn(
            &src,
            sstride,
            src_res.u(),
            src_res.v(),
            &mut data,
            res.u() * self.pixel_size,
            self.data_type,
            self.num_channels(),
        );
        Ok(FaceData::Packed { res, data })
    }

    /// Get face data at the requested res as a contiguous packed image
    /// (expanding constant and tiled data).
    fn packed_face_data(&mut self, faceid: usize, res: Res) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; res.size() * self.pixel_size];
        self.get_data_into(faceid, res, &mut buf, 0)?;
        Ok(buf)
    }

    /// Extract a single pixel from face data.
    fn pixel_from_face_data(&mut self, data: &FaceData, u: usize, v: usize) -> Result<Vec<u8>> {
        match data {
            FaceData::Constant(pixel) => Ok(pixel.clone()),
            FaceData::Packed { res, data } => {
                let off = (v * res.u() + u) * self.pixel_size;
                data.get(off..off + self.pixel_size)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| Error::Unsupported("pixel coordinates out of range".into()))
            }
            FaceData::Tiled {
                res,
                tileres,
                fdh,
                offsets,
                levelid,
            } => {
                let tileu = u >> tileres.ulog2;
                let tilev = v >> tileres.vlog2;
                let ntilesu = res.ntilesu(*tileres);
                let tile = tilev * ntilesu + tileu;
                if tile >= fdh.len() {
                    return Err(Error::Unsupported("pixel coordinates out of range".into()));
                }
                let t = self.read_face_data(offsets[tile], fdh[tile], *tileres, *levelid)?;
                self.pixel_from_face_data(
                    &t,
                    u - (tileu << tileres.ulog2),
                    v - (tilev << tileres.vlog2),
                )
            }
        }
    }

    fn read_metadata(&mut self) -> Result<MetaData> {
        let mut md = MetaData::default();

        // primary (small) meta data block
        if self.header.metadatamemsize > 0 {
            self.seek(self.metadata_pos)?;
            let buf = self.read_zip_block(
                self.header.metadatazipsize as usize,
                self.header.metadatamemsize as usize,
            )?;
            md.parse_block(&buf)?;
        }

        // large meta data: a zipped header block describing entries, each
        // entry's data stored as its own zip block following the header
        if self.ext_header.lmdheadermemsize > 0 {
            self.seek(self.lmdheader_pos)?;
            let buf = self.read_zip_block(
                self.ext_header.lmdheaderzipsize as usize,
                self.ext_header.lmdheadermemsize as usize,
            )?;
            let mut datapos = self.lmdheader_pos + self.ext_header.lmdheaderzipsize as u64;
            let mut ptr = 0usize;
            while ptr < buf.len() {
                let (key, data_type, datasize) = metadata::parse_entry_header(&buf, &mut ptr)?;
                let zipsize = format::checked_u32_at(&buf, ptr)? as usize;
                ptr += 4;
                self.seek(datapos)?;
                let data = self.read_zip_block(zipsize, datasize)?;
                md.add_entry(key, data_type, data);
                datapos += zipsize as u64;
            }
        }
        Ok(md)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_f32_face0_is_tiled() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quad_f32.ptx");
        let mut tx = PtexReader::open(path).unwrap();
        let (fdh, pos) = tx.level_face_entry(0, 0).unwrap().unwrap();
        assert_eq!(fdh.encoding(), Encoding::Tiled);
        let res = tx.face_info(0).unwrap().res;
        let fd = tx.read_face_data(pos, fdh, res, 0).unwrap();
        match fd {
            FaceData::Tiled { fdh, .. } => assert!(fdh.len() > 1),
            _ => panic!("expected tiled face data"),
        }
    }

    #[test]
    fn fixture_u8_has_diffzipped_or_zipped_faces() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quad_u8.ptx");
        let mut tx = PtexReader::open(path).unwrap();
        let (fdh, _) = tx.level_face_entry(0, 0).unwrap().unwrap();
        assert!(matches!(
            fdh.encoding(),
            Encoding::Zipped | Encoding::DiffZipped
        ));
    }
}
