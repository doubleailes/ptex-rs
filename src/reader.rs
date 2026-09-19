//! The Ptex file reader.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::decode::{self, Level, TileDir};
use crate::error::{Error, Result};
use crate::file_info::{FaceSource, FileInfo};
use crate::format::{Encoding, FaceDataHeader, FACE_DATA_HEADER_SIZE};
use crate::metadata::{self, MetaData};
use crate::tile::{TileInfo, TileLayout};
use crate::types::{BorderMode, DataType, EdgeFilterMode, FaceInfo, MeshType, Res};
use crate::utils;

/// Number of tile directories kept in memory by default.
const DEFAULT_TILE_CACHE_CAPACITY: usize = 16;

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
/// underlying stream on demand (level indexes and tile directories are
/// cached; pixel data is not).  For a reader that can be shared between
/// render threads and that caches decoded pixels, see
/// [`crate::SharedReader`].
///
/// # Streaming tiles
///
/// Large faces are stored as independently compressed tiles, and reduction
/// levels are stored in the file, so a renderer can read exactly the tile it
/// needs at exactly the resolution it needs:
///
/// ```no_run
/// use ptex::PtexReader;
///
/// let mut tx = PtexReader::open("model.ptx")?;
/// let res = tx.res_for_level(0, 2)?;          // two mip levels down
/// let layout = tx.tile_layout(0, res)?;
/// let tile = tx.get_tile(0, res, layout.tile_index(96, 40))?;
/// # Ok::<(), ptex::Error>(())
/// ```
pub struct PtexReader<R = BufReader<File>> {
    io: R,
    info: FileInfo,
    levels: Vec<Option<Level>>,
    tile_cache: Vec<(u64, Res, TileDir)>,
    tile_cache_cap: usize,
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
        let info = FileInfo::read(&mut io, premultiply)?;
        let nlevels = info.num_levels();
        Ok(PtexReader {
            io,
            info,
            levels: vec![None; nlevels],
            tile_cache: Vec::new(),
            tile_cache_cap: DEFAULT_TILE_CACHE_CAPACITY,
            metadata: None,
        })
    }

    /// Type of the base mesh (quad or triangle).
    pub fn mesh_type(&self) -> MeshType {
        self.info.mesh_type
    }

    /// Type of the pixel data.
    pub fn data_type(&self) -> DataType {
        self.info.data_type
    }

    /// Index of the alpha channel, or -1 if the file has no alpha channel.
    pub fn alpha_channel(&self) -> i32 {
        self.info.header.alphachan
    }

    /// Number of channels per pixel.
    pub fn num_channels(&self) -> usize {
        self.info.num_channels()
    }

    /// Number of faces in the file.
    pub fn num_faces(&self) -> usize {
        self.info.num_faces()
    }

    /// True if the file has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.info.header.has_alpha()
    }

    /// True if the file stores precomputed mipmap (reduction) levels.
    pub fn has_mip_maps(&self) -> bool {
        self.info.num_levels() > 1
    }

    /// Number of stored resolution levels (level 0 is full resolution).
    pub fn num_levels(&self) -> usize {
        self.info.num_levels()
    }

    /// Size of a single pixel, in bytes.
    pub fn pixel_size(&self) -> usize {
        self.info.pixel_size
    }

    /// Minor version of the file format the file was written with.
    pub fn minor_version(&self) -> u32 {
        self.info.header.minorversion
    }

    /// Border mode in the u direction.
    pub fn u_border_mode(&self) -> BorderMode {
        self.info.u_border_mode()
    }

    /// Border mode in the v direction.
    pub fn v_border_mode(&self) -> BorderMode {
        self.info.v_border_mode()
    }

    /// Edge filter mode.
    pub fn edge_filter_mode(&self) -> EdgeFilterMode {
        self.info.edge_filter_mode()
    }

    /// Access resolution and adjacency information for a face.
    pub fn face_info(&self, faceid: usize) -> Result<&FaceInfo> {
        self.info.face_info(faceid)
    }

    /// All per-face info records.
    pub fn face_infos(&self) -> &[FaceInfo] {
        &self.info.face_info
    }

    /// Access the constant (average) pixel value of a face as raw data.
    pub fn constant_data(&self, faceid: usize) -> Result<&[u8]> {
        self.info.constant_data(faceid)
    }

    /// Access the file's meta data (loaded and cached on first access).
    pub fn metadata(&mut self) -> Result<&MetaData> {
        if self.metadata.is_none() {
            let md = self.read_metadata()?;
            self.metadata = Some(md);
        }
        Ok(self.metadata.as_ref().unwrap())
    }

    // ---- mipmap level helpers (no I/O) ----

    /// Number of mipmap levels a face has, counting level 0 (the full
    /// resolution).
    ///
    /// Each level halves both dimensions, so the last level reduces the
    /// smaller dimension to a single texel.
    pub fn face_num_levels(&self, faceid: usize) -> Result<usize> {
        self.info.face_num_levels(faceid)
    }

    /// Resolution of a face reduced by `level` mipmap levels.
    ///
    /// Level 0 is the full resolution.  Returns [`Error::Unsupported`] if
    /// `level` would reduce the face below one texel; see
    /// [`PtexReader::face_num_levels`].
    pub fn res_for_level(&self, faceid: usize, level: usize) -> Result<Res> {
        self.info.face_level_res(faceid, level)
    }

    /// True if a resolution is stored in the file for this face, so that it
    /// can be read — and streamed tile by tile — directly, rather than
    /// computed by reducing a larger resolution.
    ///
    /// Performs no I/O.
    pub fn is_res_stored(&self, faceid: usize, res: Res) -> Result<bool> {
        self.info.is_res_stored(faceid, res)
    }

    /// Number of leading mipmap levels of a face that are stored in the
    /// file, counting level 0.
    ///
    /// Levels `0..n` all satisfy [`PtexReader::is_res_stored`]; the count
    /// stops at the first level that is not stored.
    pub fn num_stored_levels(&self, faceid: usize) -> Result<usize> {
        self.info.num_stored_levels(faceid)
    }

    // ---- whole-face reading ----

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
        let mut buf = vec![0u8; res.size() * self.info.pixel_size];
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
        let stride = check_buffer(res, self.info.pixel_size, buffer, stride)?;
        match self.info.resolve(faceid, res)? {
            FaceSource::Constant => {
                let pixel = self.info.constant_data(faceid)?.to_vec();
                utils::fill(
                    &pixel,
                    buffer,
                    stride,
                    res.u(),
                    res.v(),
                    self.info.pixel_size,
                );
            }
            FaceSource::Stored { levelid, facepos } => {
                let (fdh, pos) = self.level_entry(levelid, facepos)?;
                if fdh.encoding() == Encoding::Tiled {
                    let (tile_res, ntiles) = self.tile_dir_shape(pos, res)?;
                    let ntilesu = res.ntilesu(tile_res);
                    let tilerowlen = self.info.pixel_size * tile_res.u();
                    let tilevres = tile_res.v();
                    for tile in 0..ntiles {
                        let i = tile / ntilesu;
                        let j = tile % ntilesu;
                        let dst_off = i * stride * tilevres + j * tilerowlen;
                        self.read_tile_into(
                            pos,
                            res,
                            tile,
                            levelid,
                            &mut buffer[dst_off..],
                            stride,
                        )?;
                    }
                } else {
                    self.read_block_into(pos, fdh, res, levelid, buffer, stride)?;
                }
            }
            FaceSource::Reduced => {
                let (src_res, kind) = self.info.reduction_source(faceid, res)?;
                let src = self.packed_face_data(faceid, src_res)?;
                let data = decode::reduce_step(&src, src_res, res, kind, &self.info);
                let rowlen = res.u() * self.info.pixel_size;
                utils::copy_rows(&data, rowlen, buffer, stride, res.v(), rowlen);
            }
        }
        Ok(())
    }

    /// Read a single texel, converted to f32 channel values.
    ///
    /// `first_chan` selects the first channel to return and `nchannels`
    /// how many; the result is clipped to the channels available.
    /// Integer pixel formats are normalized to the 0..1 range.
    ///
    /// For a tiled face only the tile containing the texel is read.
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
        let res = self.face_info(faceid)?.res;
        let pixel = self.read_texel(faceid, res, u, v)?;
        let mut out = vec![0f32; nchan];
        let dsize = self.info.data_type.size();
        utils::convert_to_float(
            &mut out,
            &pixel[first_chan * dsize..],
            self.info.data_type,
            nchan,
        );
        Ok(out)
    }

    // ---- tile streaming ----

    /// Describe how a face is laid out at the given resolution.
    ///
    /// `res` is interpreted exactly as by [`PtexReader::get_data_at_res`]:
    /// the full resolution and stored mipmap levels are read directly,
    /// other resolutions are computed by reduction and report a single
    /// tile.  Only index data is read, never pixel data.
    pub fn tile_layout(&mut self, faceid: usize, res: Res) -> Result<TileLayout> {
        match self.info.resolve(faceid, res)? {
            FaceSource::Constant => Ok(TileLayout::untiled(res, true, true)),
            FaceSource::Reduced => Ok(TileLayout::untiled(res, false, false)),
            FaceSource::Stored { levelid, facepos } => {
                let (fdh, pos) = self.level_entry(levelid, facepos)?;
                match fdh.encoding() {
                    Encoding::Constant => Ok(TileLayout::untiled(res, true, true)),
                    Encoding::Zipped | Encoding::DiffZipped => {
                        Ok(TileLayout::untiled(res, false, true))
                    }
                    Encoding::Tiled => {
                        let (tile_res, _) = self.tile_dir_shape(pos, res)?;
                        Ok(TileLayout {
                            res,
                            tile_res,
                            ntilesu: res.ntilesu(tile_res),
                            ntilesv: res.ntilesv(tile_res),
                            is_tiled: true,
                            is_constant: false,
                            is_stored: true,
                        })
                    }
                }
            }
        }
    }

    /// Information about a single tile, without reading its pixel data.
    pub fn tile_info(&mut self, faceid: usize, res: Res, tile: usize) -> Result<TileInfo> {
        let layout = self.tile_layout(faceid, res)?;
        let ntiles = layout.ntiles();
        if tile >= ntiles {
            return Err(Error::TileOutOfRange { tile, ntiles });
        }
        let origin = layout.tile_origin(tile);
        if !layout.is_stored || layout.is_constant {
            return Ok(TileInfo {
                res: layout.tile_res,
                origin,
                is_constant: layout.is_constant,
                compressed_size: 0,
                file_offset: None,
            });
        }
        let FaceSource::Stored { levelid, facepos } = self.info.resolve(faceid, res)? else {
            return Err(Error::Corrupt("inconsistent tile layout".into()));
        };
        let (fdh, pos) = self.level_entry(levelid, facepos)?;
        let (fdh, pos) = if layout.is_tiled {
            let (tfdh, toff, _) = self.tile_entry(pos, res, tile)?;
            (tfdh, toff)
        } else {
            (fdh, pos)
        };
        Ok(TileInfo {
            res: layout.tile_res,
            origin,
            is_constant: fdh.encoding() == Encoding::Constant,
            compressed_size: fdh.blocksize() as u64,
            file_offset: Some(pos),
        })
    }

    /// Read the pixel data of a single tile.
    ///
    /// Returns interleaved pixels in v-major order: `tile_res.v()` rows of
    /// `tile_res.u()` pixels of `pixel_size()` bytes, where `tile_res` is
    /// [`TileLayout::tile_res`] for the same face and resolution.  For a
    /// face that is not tiled this is the whole face and `tile` must be 0.
    pub fn get_tile(&mut self, faceid: usize, res: Res, tile: usize) -> Result<Vec<u8>> {
        let layout = self.tile_layout(faceid, res)?;
        let ntiles = layout.ntiles();
        if tile >= ntiles {
            return Err(Error::TileOutOfRange { tile, ntiles });
        }
        let mut buf = vec![0u8; layout.tile_size_bytes(self.info.pixel_size)];
        self.get_tile_into(faceid, res, tile, &mut buf, 0)?;
        Ok(buf)
    }

    /// Read the pixel data of a single tile into a caller-provided buffer.
    ///
    /// `stride` is the row stride of `buffer` in bytes; 0 means tightly
    /// packed (`tile_res.u() * pixel_size()`).  A non-zero stride writes the
    /// tile straight into the middle of a larger image.
    pub fn get_tile_into(
        &mut self,
        faceid: usize,
        res: Res,
        tile: usize,
        buffer: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let layout = self.tile_layout(faceid, res)?;
        let ntiles = layout.ntiles();
        if tile >= ntiles {
            return Err(Error::TileOutOfRange { tile, ntiles });
        }
        if !layout.is_tiled {
            return self.get_data_into(faceid, res, buffer, stride);
        }
        let stride = check_buffer(layout.tile_res, self.info.pixel_size, buffer, stride)?;
        let FaceSource::Stored { levelid, facepos } = self.info.resolve(faceid, res)? else {
            return Err(Error::Corrupt("inconsistent tile layout".into()));
        };
        let (_, pos) = self.level_entry(levelid, facepos)?;
        self.read_tile_into(pos, res, tile, levelid, buffer, stride)
    }

    /// Maximum number of tile directories kept in memory (16 by default).
    pub fn tile_cache_capacity(&self) -> usize {
        self.tile_cache_cap
    }

    /// Set the maximum number of tile directories kept in memory.
    ///
    /// A tile directory is the index of one tiled face at one resolution
    /// (four bytes plus an offset per tile); caching it keeps repeated
    /// [`PtexReader::get_tile`] calls on the same face from re-reading and
    /// re-inflating it.  Values below one are clamped to one.
    pub fn set_tile_cache_capacity(&mut self, capacity: usize) {
        self.tile_cache_cap = capacity.max(1);
        self.tile_cache.truncate(self.tile_cache_cap);
    }

    // ---- internal reading helpers ----

    fn seek(&mut self, pos: u64) -> Result<()> {
        self.io.seek(SeekFrom::Start(pos))?;
        Ok(())
    }

    /// Read `size` raw bytes from an absolute file offset.
    fn read_raw(&mut self, pos: u64, size: usize) -> Result<Vec<u8>> {
        self.seek(pos)?;
        let mut buf = vec![0u8; size];
        self.io.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read `zipsize` bytes at the current position and zlib-decompress them
    /// into exactly `unzipsize` bytes.
    fn read_zip_block(&mut self, zipsize: usize, unzipsize: usize) -> Result<Vec<u8>> {
        let mut comp = vec![0u8; zipsize];
        self.io.read_exact(&mut comp)?;
        decode::unzip(&comp, unzipsize)
    }

    /// Read and cache the face index (headers + offsets) of a level.
    fn ensure_level(&mut self, levelid: usize) -> Result<()> {
        if self.levels[levelid].is_some() {
            return Ok(());
        }
        let li = self.info.level_info[levelid];
        let nfaces = li.nfaces as usize;
        let base = self.info.level_pos[levelid];
        let raw = self.read_raw(base, li.levelheadersize as usize)?;
        let buf = decode::unzip(&raw, FACE_DATA_HEADER_SIZE * nfaces)?;
        let fdh = decode::parse_face_data_headers(&buf);

        // faces marked "large" have 64-bit sizes stored in an extra table
        // immediately after the level header
        let data_base = base + li.levelheadersize as u64;
        let nlarge = decode::large_face_positions(&fdh).len();
        let large_table = if nlarge > 0 {
            self.read_raw(data_base, 8 * nlarge)?
        } else {
            Vec::new()
        };
        let offsets = decode::level_offsets(&fdh, data_base, &large_table)?;

        self.levels[levelid] = Some(Level { fdh, offsets });
        Ok(())
    }

    /// Header and absolute file offset of a face within a level.
    fn level_entry(&mut self, levelid: usize, facepos: usize) -> Result<(FaceDataHeader, u64)> {
        self.ensure_level(levelid)?;
        let level = self.levels[levelid].as_ref().unwrap();
        match level.fdh.get(facepos) {
            Some(&fdh) => Ok((fdh, level.offsets[facepos])),
            None => Err(Error::Corrupt("face missing from level".into())),
        }
    }

    /// Read and inflate the tile directory of the tiled block at `pos`.
    fn read_tile_dir(&mut self, pos: u64, res: Res) -> Result<TileDir> {
        let head = self.read_raw(pos, 6)?;
        let (tile_res, tileheadersize) = decode::parse_tile_header(&head, res)?;
        let ntiles = res.ntiles(tile_res);
        let raw = self.read_raw(pos + 6, tileheadersize)?;
        let dir = decode::unzip(&raw, FACE_DATA_HEADER_SIZE * ntiles)?;
        decode::parse_tile_dir(&dir, tile_res, res, pos + 6 + tileheadersize as u64)
    }

    /// Get the tile directory of the block at `pos`, reading it if it is not
    /// already cached.  Uses move-to-front, so the result is always the first
    /// cache entry.
    fn tile_dir(&mut self, pos: u64, res: Res) -> Result<&TileDir> {
        match self
            .tile_cache
            .iter()
            .position(|e| e.0 == pos && e.1 == res)
        {
            Some(0) => {}
            Some(i) => {
                let e = self.tile_cache.remove(i);
                self.tile_cache.insert(0, e);
            }
            None => {
                let dir = self.read_tile_dir(pos, res)?;
                self.tile_cache.insert(0, (pos, res, dir));
                self.tile_cache.truncate(self.tile_cache_cap.max(1));
            }
        }
        Ok(&self.tile_cache[0].2)
    }

    /// Tile resolution and tile count of a tiled block.
    fn tile_dir_shape(&mut self, pos: u64, res: Res) -> Result<(Res, usize)> {
        let dir = self.tile_dir(pos, res)?;
        Ok((dir.tile_res, dir.fdh.len()))
    }

    /// Header, absolute file offset and resolution of a single tile.
    ///
    /// Returns only `Copy` values, so the tile cache is no longer borrowed
    /// when the caller goes on to read from the stream.
    fn tile_entry(
        &mut self,
        pos: u64,
        res: Res,
        tile: usize,
    ) -> Result<(FaceDataHeader, u64, Res)> {
        let dir = self.tile_dir(pos, res)?;
        match dir.fdh.get(tile) {
            Some(&fdh) => Ok((fdh, dir.offsets[tile], dir.tile_res)),
            None => Err(Error::TileOutOfRange {
                tile,
                ntiles: dir.fdh.len(),
            }),
        }
    }

    /// Read one non-tiled data block into `dst[dst_off..]`.
    fn read_block_into(
        &mut self,
        pos: u64,
        fdh: FaceDataHeader,
        res: Res,
        levelid: usize,
        dst: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let pixel_size = self.info.pixel_size;
        match fdh.encoding() {
            Encoding::Constant => {
                let raw = self.read_raw(pos, pixel_size)?;
                let pixel = decode::decode_constant(&raw, &self.info, levelid);
                utils::fill(&pixel, dst, stride, res.u(), res.v(), pixel_size);
            }
            Encoding::Zipped | Encoding::DiffZipped => {
                if fdh.is_large_face() {
                    return Err(Error::Corrupt("non-tiled large face".into()));
                }
                let raw = self.read_raw(pos, fdh.blocksize() as usize)?;
                let data = decode::decode_packed(&raw, res, fdh.encoding(), &self.info, levelid)?;
                let rowlen = res.u() * pixel_size;
                utils::copy_rows(&data, rowlen, dst, stride, res.v(), rowlen);
            }
            Encoding::Tiled => {
                return Err(Error::Corrupt("nested tiled face data".into()));
            }
        }
        Ok(())
    }

    /// Read one tile of the tiled block at `pos` into `dst[dst_off..]`.
    fn read_tile_into(
        &mut self,
        pos: u64,
        res: Res,
        tile: usize,
        levelid: usize,
        dst: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let (fdh, off, tile_res) = self.tile_entry(pos, res, tile)?;
        self.read_block_into(off, fdh, tile_res, levelid, dst, stride)
    }

    /// Get face data at the requested res as a contiguous packed image
    /// (expanding constant and tiled data).
    fn packed_face_data(&mut self, faceid: usize, res: Res) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; res.size() * self.info.pixel_size];
        self.get_data_into(faceid, res, &mut buf, 0)?;
        Ok(buf)
    }

    /// Read a single texel of a face, reading only the tile that holds it.
    fn read_texel(&mut self, faceid: usize, res: Res, u: usize, v: usize) -> Result<Vec<u8>> {
        if u >= res.u() || v >= res.v() {
            return Err(Error::Unsupported("pixel coordinates out of range".into()));
        }
        let pixel_size = self.info.pixel_size;
        match self.info.resolve(faceid, res)? {
            FaceSource::Constant => Ok(self.info.constant_data(faceid)?.to_vec()),
            FaceSource::Stored { levelid, facepos } => {
                let (fdh, pos) = self.level_entry(levelid, facepos)?;
                if fdh.encoding() == Encoding::Tiled {
                    let (tile_res, _) = self.tile_dir_shape(pos, res)?;
                    let tileu = u >> tile_res.ulog2;
                    let tilev = v >> tile_res.vlog2;
                    let tile = tilev * res.ntilesu(tile_res) + tileu;
                    let (tfdh, toff, tres) = self.tile_entry(pos, res, tile)?;
                    self.texel_from_block(
                        toff,
                        tfdh,
                        tres,
                        levelid,
                        u - (tileu << tile_res.ulog2),
                        v - (tilev << tile_res.vlog2),
                    )
                } else {
                    self.texel_from_block(pos, fdh, res, levelid, u, v)
                }
            }
            FaceSource::Reduced => {
                let data = self.get_data_at_res(faceid, res)?;
                let off = (v * res.u() + u) * pixel_size;
                Ok(data[off..off + pixel_size].to_vec())
            }
        }
    }

    /// Read a single texel out of one non-tiled data block.
    fn texel_from_block(
        &mut self,
        pos: u64,
        fdh: FaceDataHeader,
        res: Res,
        levelid: usize,
        u: usize,
        v: usize,
    ) -> Result<Vec<u8>> {
        let pixel_size = self.info.pixel_size;
        let mut buf = vec![0u8; res.size() * pixel_size];
        self.read_block_into(pos, fdh, res, levelid, &mut buf, res.u() * pixel_size)?;
        let off = (v * res.u() + u) * pixel_size;
        Ok(buf[off..off + pixel_size].to_vec())
    }

    fn read_metadata(&mut self) -> Result<MetaData> {
        let mut md = MetaData::default();

        // primary (small) meta data block
        if self.info.header.metadatamemsize > 0 {
            self.seek(self.info.metadata_pos)?;
            let buf = self.read_zip_block(
                self.info.header.metadatazipsize as usize,
                self.info.header.metadatamemsize as usize,
            )?;
            md.parse_block(&buf)?;
        }

        // large meta data: a zipped header block describing entries, each
        // entry's data stored as its own zip block following the header
        if self.info.ext_header.lmdheadermemsize > 0 {
            self.seek(self.info.lmdheader_pos)?;
            let buf = self.read_zip_block(
                self.info.ext_header.lmdheaderzipsize as usize,
                self.info.ext_header.lmdheadermemsize as usize,
            )?;
            let mut datapos =
                self.info.lmdheader_pos + self.info.ext_header.lmdheaderzipsize as u64;
            let mut ptr = 0usize;
            while ptr < buf.len() {
                let (key, data_type, datasize) = metadata::parse_entry_header(&buf, &mut ptr)?;
                let zipsize = crate::format::checked_u32_at(&buf, ptr)? as usize;
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

/// Validate a destination buffer and resolve `stride == 0` to a packed row.
pub(crate) fn check_buffer(
    res: Res,
    pixel_size: usize,
    buffer: &[u8],
    stride: usize,
) -> Result<usize> {
    let rowlen = pixel_size * res.u();
    let stride = if stride == 0 { rowlen } else { stride };
    if stride < rowlen || buffer.len() < stride * (res.v() - 1) + rowlen {
        return Err(Error::Unsupported(
            "buffer too small for requested res".into(),
        ));
    }
    Ok(stride)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn fixture_f32_face0_is_tiled() {
        let mut tx = PtexReader::open(fixture("quad_f32.ptx")).unwrap();
        let res = tx.face_info(0).unwrap().res;
        let layout = tx.tile_layout(0, res).unwrap();
        assert!(layout.is_tiled);
        assert!(layout.ntiles() > 1);
    }

    #[test]
    fn fixture_u8_has_diffzipped_or_zipped_faces() {
        let mut tx = PtexReader::open(fixture("quad_u8.ptx")).unwrap();
        let (fdh, _) = tx.level_entry(0, 0).unwrap();
        assert!(matches!(
            fdh.encoding(),
            Encoding::Zipped | Encoding::DiffZipped
        ));
    }

    /// The zero-I/O `is_res_stored` predicate must agree with an actual
    /// level lookup for every face and level of every fixture.
    #[test]
    fn is_res_stored_matches_level_entry() {
        for name in ["quad_u8.ptx", "quad_f32.ptx", "quad_f16.ptx", "tri_u16.ptx"] {
            let tx = PtexReader::open(fixture(name)).unwrap();
            for faceid in 0..tx.num_faces() {
                let fi = *tx.face_info(faceid).unwrap();
                for level in 0..tx.face_num_levels(faceid).unwrap() {
                    let res = tx.res_for_level(faceid, level).unwrap();
                    let predicted = tx.is_res_stored(faceid, res).unwrap();
                    let actual = if fi.is_constant() || res.is_one() || level == 0 {
                        true
                    } else if level < tx.num_levels() {
                        let facepos = tx.info.rfaceids[faceid] as usize;
                        facepos < tx.info.level_info[level].nfaces as usize
                    } else {
                        false
                    };
                    assert_eq!(predicted, actual, "{name} face {faceid} level {level}");
                }
            }
        }
    }

    #[test]
    fn reader_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PtexReader<std::io::Cursor<Vec<u8>>>>();
        assert_send::<PtexReader<BufReader<File>>>();
    }
}
