//! A thread-safe reader with a bounded cache of decoded pixels.
//!
//! # Locking
//!
//! Two mutexes, both leaves: the I/O stream and the pixel cache.  No code
//! path holds both at once, so no lock order can be violated.  The I/O lock
//! is taken only around `seek` + `read_exact`; inflating, de-planarising,
//! premultiplying and reducing all happen with nothing held, so render
//! threads decode in parallel.
//!
//! Lazily filled state (level indexes, meta data) uses double-checked
//! initialisation rather than a lock held across I/O.  Two threads that miss
//! on the same entry may both decode it; that is deliberate — decoding is a
//! pure function of the bytes, so the results are identical, and duplicate
//! work is cheaper than blocking a reader behind another reader's I/O.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use crate::decode::{self, Level, TileDir};
use crate::error::{Error, Result};
use crate::file_info::{FaceSource, FileInfo};
use crate::format::{Encoding, FaceDataHeader, FACE_DATA_HEADER_SIZE};
use crate::metadata::{self, MetaData};
use crate::reader::check_buffer;
use crate::tile::{TileInfo, TileLayout};
use crate::types::{BorderMode, DataType, EdgeFilterMode, FaceInfo, MeshType, Res};
use crate::utils;

/// Default size of the decoded-pixel cache, in bytes (64 MiB).
pub const DEFAULT_CACHE_BUDGET: usize = 64 << 20;

/// Bytes charged to every cache entry on top of its payload, so that a flood
/// of tiny entries is still bounded by the budget.
const ENTRY_OVERHEAD: usize = 96;

/// Options for opening a [`SharedReader`].
#[derive(Debug, Clone, Copy)]
pub struct CacheOptions {
    /// Premultiply color channels by the alpha channel, matching the
    /// `premultiply` flag of the C++ API.
    pub premultiply: bool,
    /// Maximum number of bytes of decoded pixel data to keep resident.
    /// Zero disables caching entirely.
    pub budget_bytes: usize,
}

impl Default for CacheOptions {
    fn default() -> Self {
        CacheOptions {
            premultiply: false,
            budget_bytes: DEFAULT_CACHE_BUDGET,
        }
    }
}

/// A snapshot of a [`SharedReader`]'s cache counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups served from the cache.
    pub hits: u64,
    /// Lookups that had to read and decode.
    pub misses: u64,
    /// Entries dropped to stay within the budget.
    pub evictions: u64,
    /// Blocks returned to the caller but not cached, because a single one
    /// would not fit in the budget.
    pub oversized: u64,
    /// Entries currently resident.
    pub entries: usize,
    /// Accounted bytes currently resident.
    pub bytes_resident: usize,
    /// The current budget, in bytes.
    pub bytes_budget: usize,
}

/// Decoded pixel data, reference-counted so that a cache hit copies nothing.
///
/// Dereferences to the interleaved bytes, in v-major order.
#[derive(Debug, Clone)]
pub struct PixelData(Arc<Vec<u8>>);

impl PixelData {
    fn new(data: Vec<u8>) -> Self {
        PixelData(Arc::new(data))
    }

    /// The pixel bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// Copy the pixels into a freshly allocated buffer.
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.as_ref().clone()
    }
}

impl Deref for PixelData {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for PixelData {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CacheKey {
    /// Decoded pixels of a block stored in the file — a whole face or a
    /// single tile — identified by its absolute file offset.  An offset
    /// determines the block, and so also the level it belongs to, which is
    /// what decides whether alpha premultiplication was applied.
    Block { pos: u64 },
    /// A dynamically reduced image, which has no file offset of its own.
    Reduced { faceid: u32, res: u16 },
    /// The tile directory of the tiled block at this file offset.
    TileDir { pos: u64 },
}

#[derive(Clone)]
enum CacheValue {
    Pixels(PixelData),
    Dir(Arc<TileDir>),
}

impl CacheValue {
    fn size(&self) -> usize {
        ENTRY_OVERHEAD
            + match self {
                CacheValue::Pixels(p) => p.len(),
                CacheValue::Dir(d) => d.fdh.len() * 12,
            }
    }
}

/// A byte-budgeted LRU over `lru::LruCache`, which is count-bounded only.
struct PixelCache {
    lru: lru::LruCache<CacheKey, CacheValue>,
    bytes: usize,
    budget: usize,
    evictions: u64,
    oversized: u64,
}

impl PixelCache {
    fn new(budget: usize) -> Self {
        PixelCache {
            lru: lru::LruCache::unbounded(),
            bytes: 0,
            budget,
            evictions: 0,
            oversized: 0,
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<CacheValue> {
        self.lru.get(key).cloned()
    }

    /// Insert `value`, or return the entry a racing thread already inserted.
    fn insert_or_get(&mut self, key: CacheKey, value: CacheValue) -> CacheValue {
        if let Some(existing) = self.lru.get(&key) {
            return existing.clone();
        }
        let size = value.size();
        if size > self.budget {
            // Caching it would evict everything and then itself.
            self.oversized += 1;
            return value;
        }
        while self.bytes + size > self.budget {
            match self.lru.pop_lru() {
                Some((_, evicted)) => {
                    self.bytes -= evicted.size();
                    self.evictions += 1;
                }
                None => break,
            }
        }
        self.bytes += size;
        self.lru.put(key, value.clone());
        value
    }

    fn clear(&mut self) {
        self.lru.clear();
        self.bytes = 0;
    }

    fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        while self.bytes > self.budget {
            match self.lru.pop_lru() {
                Some((_, evicted)) => {
                    self.bytes -= evicted.size();
                    self.evictions += 1;
                }
                None => break,
            }
        }
    }
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
}

struct Shared<R> {
    info: FileInfo,
    io: Mutex<R>,
    levels: Vec<OnceLock<Level>>,
    metadata: OnceLock<MetaData>,
    pixels: Mutex<PixelCache>,
    counters: Counters,
}

/// A thread-safe, cheaply clonable reader for Ptex texture files.
///
/// Cloning is an `Arc` bump: every clone shares one open stream, one set of
/// parsed indexes and one bounded cache of decoded pixels.  All read methods
/// take `&self`, so a single handle serves any number of render threads.
///
/// Compared with [`crate::PtexReader`], reads return [`PixelData`] — a
/// reference-counted handle into the cache — so a repeated request for the
/// same tile copies nothing.
///
/// ```no_run
/// use ptex::SharedReader;
///
/// let tx = SharedReader::open("model.ptx")?;
/// std::thread::scope(|s| {
///     for t in 0..4 {
///         let tx = tx.clone();
///         s.spawn(move || {
///             for faceid in (t..tx.num_faces()).step_by(4) {
///                 let res = tx.res_for_level(faceid, 1).unwrap();
///                 let layout = tx.tile_layout(faceid, res).unwrap();
///                 for tile in 0..layout.ntiles() {
///                     let _pixels = tx.get_tile(faceid, res, tile).unwrap();
///                 }
///             }
///         });
///     }
/// });
/// # Ok::<(), ptex::Error>(())
/// ```
///
/// The default stream type is [`File`], not `BufReader<File>`: every read is
/// an absolute seek of an exactly known length, and `BufReader` discards its
/// buffer on each seek, so buffering would only add a copy.
pub struct SharedReader<R = File> {
    inner: Arc<Shared<R>>,
}

// A derived `Clone` would wrongly require `R: Clone`.
impl<R> Clone for SharedReader<R> {
    fn clone(&self) -> Self {
        SharedReader {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl SharedReader<File> {
    /// Open a Ptex file from the file system with the default cache budget.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, CacheOptions::default())
    }

    /// Open a Ptex file from the file system with explicit options.
    pub fn open_with_options(path: impl AsRef<Path>, options: CacheOptions) -> Result<Self> {
        Self::new_with_options(File::open(path)?, options)
    }
}

impl<R: Read + Seek + Send> SharedReader<R> {
    /// Read a Ptex file from any seekable stream.
    pub fn new(io: R) -> Result<Self> {
        Self::new_with_options(io, CacheOptions::default())
    }

    /// Read a Ptex file from any seekable stream with explicit options.
    pub fn new_with_options(mut io: R, options: CacheOptions) -> Result<Self> {
        let info = FileInfo::read(&mut io, options.premultiply)?;
        let nlevels = info.num_levels();
        Ok(SharedReader {
            inner: Arc::new(Shared {
                info,
                io: Mutex::new(io),
                levels: (0..nlevels).map(|_| OnceLock::new()).collect(),
                metadata: OnceLock::new(),
                pixels: Mutex::new(PixelCache::new(options.budget_bytes)),
                counters: Counters::default(),
            }),
        })
    }

    // ---- file information ----

    /// Type of the base mesh (quad or triangle).
    pub fn mesh_type(&self) -> MeshType {
        self.inner.info.mesh_type
    }

    /// Type of the pixel data.
    pub fn data_type(&self) -> DataType {
        self.inner.info.data_type
    }

    /// Index of the alpha channel, or -1 if the file has no alpha channel.
    pub fn alpha_channel(&self) -> i32 {
        self.inner.info.header.alphachan
    }

    /// Number of channels per pixel.
    pub fn num_channels(&self) -> usize {
        self.inner.info.num_channels()
    }

    /// Number of faces in the file.
    pub fn num_faces(&self) -> usize {
        self.inner.info.num_faces()
    }

    /// True if the file has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.inner.info.header.has_alpha()
    }

    /// True if the file stores precomputed mipmap (reduction) levels.
    pub fn has_mip_maps(&self) -> bool {
        self.inner.info.num_levels() > 1
    }

    /// Number of stored resolution levels (level 0 is full resolution).
    pub fn num_levels(&self) -> usize {
        self.inner.info.num_levels()
    }

    /// Size of a single pixel, in bytes.
    pub fn pixel_size(&self) -> usize {
        self.inner.info.pixel_size
    }

    /// Minor version of the file format the file was written with.
    pub fn minor_version(&self) -> u32 {
        self.inner.info.header.minorversion
    }

    /// Border mode in the u direction.
    pub fn u_border_mode(&self) -> BorderMode {
        self.inner.info.u_border_mode()
    }

    /// Border mode in the v direction.
    pub fn v_border_mode(&self) -> BorderMode {
        self.inner.info.v_border_mode()
    }

    /// Edge filter mode.
    pub fn edge_filter_mode(&self) -> EdgeFilterMode {
        self.inner.info.edge_filter_mode()
    }

    /// Access resolution and adjacency information for a face.
    pub fn face_info(&self, faceid: usize) -> Result<&FaceInfo> {
        self.inner.info.face_info(faceid)
    }

    /// All per-face info records.
    pub fn face_infos(&self) -> &[FaceInfo] {
        &self.inner.info.face_info
    }

    /// Access the constant (average) pixel value of a face as raw data.
    pub fn constant_data(&self, faceid: usize) -> Result<&[u8]> {
        self.inner.info.constant_data(faceid)
    }

    /// Access the file's meta data (parsed and cached on first access).
    pub fn metadata(&self) -> Result<&MetaData> {
        if let Some(md) = self.inner.metadata.get() {
            return Ok(md);
        }
        let md = self.read_metadata()?;
        let _ = self.inner.metadata.set(md);
        Ok(self.inner.metadata.get().expect("metadata was just set"))
    }

    // ---- mipmap level helpers (no I/O) ----

    /// Number of mipmap levels a face has, counting level 0.
    pub fn face_num_levels(&self, faceid: usize) -> Result<usize> {
        self.inner.info.face_num_levels(faceid)
    }

    /// Resolution of a face reduced by `level` mipmap levels.
    pub fn res_for_level(&self, faceid: usize, level: usize) -> Result<Res> {
        self.inner.info.face_level_res(faceid, level)
    }

    /// True if a resolution is stored in the file for this face, so it can
    /// be streamed rather than computed by reduction.
    pub fn is_res_stored(&self, faceid: usize, res: Res) -> Result<bool> {
        self.inner.info.is_res_stored(faceid, res)
    }

    /// Number of leading mipmap levels of a face that are stored in the
    /// file, counting level 0.
    pub fn num_stored_levels(&self, faceid: usize) -> Result<usize> {
        self.inner.info.num_stored_levels(faceid)
    }

    // ---- cache administration ----

    /// Snapshot the cache counters.
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.lock_pixels();
        CacheStats {
            hits: self.inner.counters.hits.load(Ordering::Relaxed),
            misses: self.inner.counters.misses.load(Ordering::Relaxed),
            evictions: cache.evictions,
            oversized: cache.oversized,
            entries: cache.lru.len(),
            bytes_resident: cache.bytes,
            bytes_budget: cache.budget,
        }
    }

    /// Drop all cached pixel data and tile directories.
    ///
    /// Parsed level indexes and meta data are kept: they are small and every
    /// read needs them.
    pub fn clear_cache(&self) {
        self.lock_pixels().clear();
    }

    /// Change the cache budget, evicting immediately if it shrank.
    pub fn set_cache_budget(&self, bytes: usize) {
        self.lock_pixels().set_budget(bytes);
    }

    /// The current cache budget, in bytes.
    pub fn cache_budget(&self) -> usize {
        self.lock_pixels().budget
    }

    // ---- reading ----

    /// Read the full-resolution pixel data for a face.
    pub fn get_data(&self, faceid: usize) -> Result<PixelData> {
        let res = self.face_info(faceid)?.res;
        self.get_data_at_res(faceid, res)
    }

    /// Read pixel data for a face at the given resolution.
    ///
    /// Stored resolutions are read directly; others are computed by
    /// reduction, exactly as [`crate::PtexReader::get_data_at_res`].
    pub fn get_data_at_res(&self, faceid: usize, res: Res) -> Result<PixelData> {
        let pixel_size = self.inner.info.pixel_size;
        match self.inner.info.resolve(faceid, res)? {
            FaceSource::Constant => {
                let pixel = self.inner.info.constant_data(faceid)?;
                let mut buf = vec![0u8; res.size() * pixel_size];
                utils::fill(
                    pixel,
                    &mut buf,
                    res.u() * pixel_size,
                    res.u(),
                    res.v(),
                    pixel_size,
                );
                Ok(PixelData::new(buf))
            }
            FaceSource::Stored { levelid, facepos } => {
                let (fdh, pos) = self.level_entry(levelid, facepos)?;
                if fdh.encoding() == Encoding::Tiled {
                    // Assembled from the (cached) tiles rather than cached
                    // itself, so the same pixels are not held twice.
                    let mut buf = vec![0u8; res.size() * pixel_size];
                    let stride = res.u() * pixel_size;
                    self.assemble_tiles(pos, res, levelid, &mut buf, stride)?;
                    Ok(PixelData::new(buf))
                } else {
                    match self.block(pos, fdh, res, levelid)? {
                        Block::Constant(pixel) => {
                            let mut buf = vec![0u8; res.size() * pixel_size];
                            utils::fill(
                                &pixel,
                                &mut buf,
                                res.u() * pixel_size,
                                res.u(),
                                res.v(),
                                pixel_size,
                            );
                            Ok(PixelData::new(buf))
                        }
                        Block::Image(data) => Ok(data),
                    }
                }
            }
            FaceSource::Reduced => self.reduced(faceid, res),
        }
    }

    /// Read pixel data for a face into a caller-provided buffer.
    ///
    /// `stride` is the row stride of `buffer` in bytes; 0 means tightly
    /// packed.  Pixels are copied straight from the cache; no intermediate
    /// buffer is allocated, including for tiled faces.
    pub fn get_data_into(
        &self,
        faceid: usize,
        res: Res,
        buffer: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let pixel_size = self.inner.info.pixel_size;
        let stride = check_buffer(res, pixel_size, buffer, stride)?;
        match self.inner.info.resolve(faceid, res)? {
            FaceSource::Constant => {
                let pixel = self.inner.info.constant_data(faceid)?;
                utils::fill(pixel, buffer, stride, res.u(), res.v(), pixel_size);
                Ok(())
            }
            FaceSource::Stored { levelid, facepos } => {
                let (fdh, pos) = self.level_entry(levelid, facepos)?;
                if fdh.encoding() == Encoding::Tiled {
                    self.assemble_tiles(pos, res, levelid, buffer, stride)
                } else {
                    self.block_into(pos, fdh, res, levelid, buffer, stride)
                }
            }
            FaceSource::Reduced => {
                let data = self.reduced(faceid, res)?;
                let rowlen = res.u() * pixel_size;
                utils::copy_rows(&data, rowlen, buffer, stride, res.v(), rowlen);
                Ok(())
            }
        }
    }

    /// Read a single texel, converted to f32 channel values.
    ///
    /// For a tiled face only the tile containing the texel is read.
    pub fn get_pixel(
        &self,
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
        if u >= res.u() || v >= res.v() {
            return Err(Error::Unsupported("pixel coordinates out of range".into()));
        }
        let pixel_size = self.inner.info.pixel_size;
        let layout = self.tile_layout(faceid, res)?;
        let pixel: Vec<u8> = if layout.is_constant && !layout.is_tiled {
            self.inner.info.constant_data(faceid)?.to_vec()
        } else {
            let tile = layout.tile_index(u, v);
            let (ou, ov) = layout.tile_origin(tile);
            let data = self.get_tile(faceid, res, tile)?;
            let tres = layout.tile_res;
            let off = ((v - ov) * tres.u() + (u - ou)) * pixel_size;
            data[off..off + pixel_size].to_vec()
        };
        let mut out = vec![0f32; nchan];
        let dsize = self.inner.info.data_type.size();
        utils::convert_to_float(
            &mut out,
            &pixel[first_chan * dsize..],
            self.inner.info.data_type,
            nchan,
        );
        Ok(out)
    }

    // ---- tile streaming ----

    /// Describe how a face is laid out at the given resolution.
    pub fn tile_layout(&self, faceid: usize, res: Res) -> Result<TileLayout> {
        match self.inner.info.resolve(faceid, res)? {
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
                        let dir = self.tile_dir(pos, res)?;
                        Ok(TileLayout {
                            res,
                            tile_res: dir.tile_res,
                            ntilesu: res.ntilesu(dir.tile_res),
                            ntilesv: res.ntilesv(dir.tile_res),
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
    pub fn tile_info(&self, faceid: usize, res: Res, tile: usize) -> Result<TileInfo> {
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
        let FaceSource::Stored { levelid, facepos } = self.inner.info.resolve(faceid, res)? else {
            return Err(Error::Corrupt("inconsistent tile layout".into()));
        };
        let (fdh, pos) = self.level_entry(levelid, facepos)?;
        let (fdh, pos) = if layout.is_tiled {
            let dir = self.tile_dir(pos, res)?;
            (dir.fdh[tile], dir.offsets[tile])
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
    /// For a face that is not tiled this is the whole face and `tile` must
    /// be 0.
    pub fn get_tile(&self, faceid: usize, res: Res, tile: usize) -> Result<PixelData> {
        let layout = self.tile_layout(faceid, res)?;
        let ntiles = layout.ntiles();
        if tile >= ntiles {
            return Err(Error::TileOutOfRange { tile, ntiles });
        }
        if !layout.is_tiled {
            return self.get_data_at_res(faceid, res);
        }
        let pixel_size = self.inner.info.pixel_size;
        let FaceSource::Stored { levelid, facepos } = self.inner.info.resolve(faceid, res)? else {
            return Err(Error::Corrupt("inconsistent tile layout".into()));
        };
        let (_, pos) = self.level_entry(levelid, facepos)?;
        let dir = self.tile_dir(pos, res)?;
        match self.block(dir.offsets[tile], dir.fdh[tile], dir.tile_res, levelid)? {
            Block::Image(data) => Ok(data),
            Block::Constant(pixel) => {
                let tres = dir.tile_res;
                let mut buf = vec![0u8; tres.size() * pixel_size];
                utils::fill(
                    &pixel,
                    &mut buf,
                    tres.u() * pixel_size,
                    tres.u(),
                    tres.v(),
                    pixel_size,
                );
                Ok(PixelData::new(buf))
            }
        }
    }

    /// Read the pixel data of a single tile into a caller-provided buffer.
    ///
    /// `stride` is the row stride of `buffer` in bytes; 0 means tightly
    /// packed to the tile's row length.
    pub fn get_tile_into(
        &self,
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
        let pixel_size = self.inner.info.pixel_size;
        let stride = check_buffer(layout.tile_res, pixel_size, buffer, stride)?;
        let FaceSource::Stored { levelid, facepos } = self.inner.info.resolve(faceid, res)? else {
            return Err(Error::Corrupt("inconsistent tile layout".into()));
        };
        let (_, pos) = self.level_entry(levelid, facepos)?;
        self.tile_into(pos, res, tile, levelid, buffer, stride)
    }

    // ---- internals ----

    fn lock_io(&self) -> MutexGuard<'_, R> {
        self.inner.io.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lock_pixels(&self) -> MutexGuard<'_, PixelCache> {
        self.inner
            .pixels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Read `len` raw bytes from an absolute file offset.
    ///
    /// This is the only place the I/O lock is taken, and nothing is decoded
    /// while it is held.
    fn read_raw(&self, pos: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        {
            let mut io = self.lock_io();
            io.seek(SeekFrom::Start(pos))?;
            io.read_exact(&mut buf)?;
        }
        Ok(buf)
    }

    fn cache_get(&self, key: &CacheKey) -> Option<CacheValue> {
        let hit = self.lock_pixels().get(key);
        let counter = if hit.is_some() {
            &self.inner.counters.hits
        } else {
            &self.inner.counters.misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
        hit
    }

    /// Read and cache the face index of a level.
    fn level(&self, levelid: usize) -> Result<&Level> {
        if let Some(level) = self.inner.levels[levelid].get() {
            return Ok(level);
        }
        let li = self.inner.info.level_info[levelid];
        let nfaces = li.nfaces as usize;
        let base = self.inner.info.level_pos[levelid];
        let raw = self.read_raw(base, li.levelheadersize as usize)?;
        let buf = decode::unzip(&raw, FACE_DATA_HEADER_SIZE * nfaces)?;
        let fdh = decode::parse_face_data_headers(&buf);

        let data_base = base + li.levelheadersize as u64;
        let nlarge = decode::large_face_positions(&fdh).len();
        let large_table = if nlarge > 0 {
            self.read_raw(data_base, 8 * nlarge)?
        } else {
            Vec::new()
        };
        let offsets = decode::level_offsets(&fdh, data_base, &large_table)?;

        // A racing thread may have set it first; its value is identical.
        let _ = self.inner.levels[levelid].set(Level { fdh, offsets });
        Ok(self.inner.levels[levelid]
            .get()
            .expect("level index was just set"))
    }

    fn level_entry(&self, levelid: usize, facepos: usize) -> Result<(FaceDataHeader, u64)> {
        let level = self.level(levelid)?;
        match level.fdh.get(facepos) {
            Some(&fdh) => Ok((fdh, level.offsets[facepos])),
            None => Err(Error::Corrupt("face missing from level".into())),
        }
    }

    /// Read, or take from the cache, the tile directory of a tiled block.
    fn tile_dir(&self, pos: u64, res: Res) -> Result<Arc<TileDir>> {
        let key = CacheKey::TileDir { pos };
        if let Some(CacheValue::Dir(dir)) = self.cache_get(&key) {
            return Ok(dir);
        }
        let head = self.read_raw(pos, 6)?;
        let (tile_res, tileheadersize) = decode::parse_tile_header(&head, res)?;
        let ntiles = res.ntiles(tile_res);
        let raw = self.read_raw(pos + 6, tileheadersize)?;
        let inflated = decode::unzip(&raw, FACE_DATA_HEADER_SIZE * ntiles)?;
        let dir = Arc::new(decode::parse_tile_dir(
            &inflated,
            tile_res,
            res,
            pos + 6 + tileheadersize as u64,
        )?);
        match self
            .lock_pixels()
            .insert_or_get(key, CacheValue::Dir(dir.clone()))
        {
            CacheValue::Dir(d) => Ok(d),
            CacheValue::Pixels(_) => Ok(dir),
        }
    }

    /// Read, or take from the cache, one non-tiled data block.
    fn block(&self, pos: u64, fdh: FaceDataHeader, res: Res, levelid: usize) -> Result<Block> {
        let key = CacheKey::Block { pos };
        match fdh.encoding() {
            Encoding::Constant => {
                // Cheap to reproduce and potentially huge once expanded, so
                // constant blocks are never cached.
                let raw = self.read_raw(pos, self.inner.info.pixel_size)?;
                Ok(Block::Constant(decode::decode_constant(
                    &raw,
                    &self.inner.info,
                    levelid,
                )))
            }
            Encoding::Zipped | Encoding::DiffZipped => {
                if let Some(CacheValue::Pixels(data)) = self.cache_get(&key) {
                    return Ok(Block::Image(data));
                }
                if fdh.is_large_face() {
                    return Err(Error::Corrupt("non-tiled large face".into()));
                }
                let raw = self.read_raw(pos, fdh.blocksize() as usize)?;
                let data =
                    decode::decode_packed(&raw, res, fdh.encoding(), &self.inner.info, levelid)?;
                Ok(Block::Image(self.store(key, PixelData::new(data))))
            }
            Encoding::Tiled => Err(Error::Corrupt("nested tiled face data".into())),
        }
    }

    fn store(&self, key: CacheKey, data: PixelData) -> PixelData {
        match self
            .lock_pixels()
            .insert_or_get(key, CacheValue::Pixels(data.clone()))
        {
            CacheValue::Pixels(p) => p,
            CacheValue::Dir(_) => data,
        }
    }

    /// Write one non-tiled block into a strided destination buffer.
    fn block_into(
        &self,
        pos: u64,
        fdh: FaceDataHeader,
        res: Res,
        levelid: usize,
        dst: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let pixel_size = self.inner.info.pixel_size;
        match self.block(pos, fdh, res, levelid)? {
            Block::Constant(pixel) => {
                utils::fill(&pixel, dst, stride, res.u(), res.v(), pixel_size);
            }
            Block::Image(data) => {
                let rowlen = res.u() * pixel_size;
                utils::copy_rows(&data, rowlen, dst, stride, res.v(), rowlen);
            }
        }
        Ok(())
    }

    /// Write one tile of a tiled block into a strided destination buffer.
    fn tile_into(
        &self,
        pos: u64,
        res: Res,
        tile: usize,
        levelid: usize,
        dst: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let dir = self.tile_dir(pos, res)?;
        self.block_into(
            dir.offsets[tile],
            dir.fdh[tile],
            dir.tile_res,
            levelid,
            dst,
            stride,
        )
    }

    /// Fill a face-sized buffer by streaming every tile of a tiled block.
    fn assemble_tiles(
        &self,
        pos: u64,
        res: Res,
        levelid: usize,
        buffer: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let pixel_size = self.inner.info.pixel_size;
        let dir = self.tile_dir(pos, res)?;
        let tile_res = dir.tile_res;
        let ntilesu = res.ntilesu(tile_res);
        let tilerowlen = pixel_size * tile_res.u();
        let tilevres = tile_res.v();
        for tile in 0..dir.fdh.len() {
            let i = tile / ntilesu;
            let j = tile % ntilesu;
            let dst_off = i * stride * tilevres + j * tilerowlen;
            self.block_into(
                dir.offsets[tile],
                dir.fdh[tile],
                tile_res,
                levelid,
                &mut buffer[dst_off..],
                stride,
            )?;
        }
        Ok(())
    }

    /// Compute, or take from the cache, a dynamically reduced resolution.
    fn reduced(&self, faceid: usize, res: Res) -> Result<PixelData> {
        let key = CacheKey::Reduced {
            faceid: faceid as u32,
            res: res.val(),
        };
        if let Some(CacheValue::Pixels(data)) = self.cache_get(&key) {
            return Ok(data);
        }
        let (src_res, kind) = self.inner.info.reduction_source(faceid, res)?;
        let src = self.get_data_at_res(faceid, src_res)?;
        let data = decode::reduce_step(&src, src_res, res, kind, &self.inner.info);
        Ok(self.store(key, PixelData::new(data)))
    }

    fn read_metadata(&self) -> Result<MetaData> {
        let info = &self.inner.info;
        let mut md = MetaData::default();

        if info.header.metadatamemsize > 0 {
            let raw = self.read_raw(info.metadata_pos, info.header.metadatazipsize as usize)?;
            let buf = decode::unzip(&raw, info.header.metadatamemsize as usize)?;
            md.parse_block(&buf)?;
        }

        if info.ext_header.lmdheadermemsize > 0 {
            let raw = self.read_raw(
                info.lmdheader_pos,
                info.ext_header.lmdheaderzipsize as usize,
            )?;
            let buf = decode::unzip(&raw, info.ext_header.lmdheadermemsize as usize)?;
            let mut datapos = info.lmdheader_pos + info.ext_header.lmdheaderzipsize as u64;
            let mut ptr = 0usize;
            while ptr < buf.len() {
                let (key, data_type, datasize) = metadata::parse_entry_header(&buf, &mut ptr)?;
                let zipsize = crate::format::checked_u32_at(&buf, ptr)? as usize;
                ptr += 4;
                let raw = self.read_raw(datapos, zipsize)?;
                md.add_entry(key, data_type, decode::unzip(&raw, datasize)?);
                datapos += zipsize as u64;
            }
        }
        Ok(md)
    }
}

enum Block {
    Constant(Vec<u8>),
    Image(PixelData),
}
