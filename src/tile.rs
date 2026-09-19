//! Tile layout description for streaming face data.

use crate::types::Res;

/// How a face's pixel data is laid out at a given resolution.
///
/// Every face has a tiling at every resolution.  A face that is *not* split
/// into tiles on disk — a constant face, a zipped or difference-encoded
/// face, or a resolution that has to be computed by reduction — reports a
/// single tile covering the whole face, so a renderer needs only one code
/// path.
///
/// Tiles are numbered in v-major order, `tile = tile_v * ntilesu + tile_u`,
/// with tile 0 at the origin of the face.  Ptex always subdivides a face by
/// powers of two, so every tile has exactly [`TileLayout::tile_res`] and
/// there are no partial tiles along the edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileLayout {
    /// Resolution of the face at this level.
    pub res: Res,
    /// Resolution of a single tile; equal to `res` when the face is not
    /// tiled.
    pub tile_res: Res,
    /// Number of tiles in the u direction (at least 1).
    pub ntilesu: usize,
    /// Number of tiles in the v direction (at least 1).
    pub ntilesv: usize,
    /// True if the face really is stored as separate tiles in the file.
    ///
    /// When false, the single reported tile is the whole face and reading it
    /// is equivalent to [`crate::PtexReader::get_data_at_res`].
    pub is_tiled: bool,
    /// True if the whole face is a single constant value at this resolution.
    pub is_constant: bool,
    /// True if this resolution is stored in the file and can be streamed
    /// tile by tile.
    ///
    /// When false the resolution is produced by reducing a larger stored
    /// resolution, so reading its single tile reads and reduces the whole
    /// face.
    pub is_stored: bool,
}

impl TileLayout {
    /// Total number of tiles, always at least 1.
    pub fn ntiles(&self) -> usize {
        self.ntilesu * self.ntilesv
    }

    /// Index of the tile containing the face texel at (`u`, `v`).
    ///
    /// The coordinates must lie inside the face
    /// (`u < res.u()`, `v < res.v()`).
    pub fn tile_index(&self, u: usize, v: usize) -> usize {
        debug_assert!(u < self.res.u() && v < self.res.v(), "texel out of range");
        (v >> self.tile_res.vlog2) * self.ntilesu + (u >> self.tile_res.ulog2)
    }

    /// Tile coordinates (`tile_u`, `tile_v`) of a linear tile index.
    ///
    /// The index must be less than [`TileLayout::ntiles`].
    pub fn tile_coords(&self, tile: usize) -> (usize, usize) {
        debug_assert!(tile < self.ntiles(), "tile index out of range");
        (tile % self.ntilesu, tile / self.ntilesu)
    }

    /// Origin of a tile in face texel coordinates, as (`u`, `v`).
    ///
    /// The index must be less than [`TileLayout::ntiles`].
    pub fn tile_origin(&self, tile: usize) -> (usize, usize) {
        let (tu, tv) = self.tile_coords(tile);
        (tu << self.tile_res.ulog2, tv << self.tile_res.vlog2)
    }

    /// Number of bytes a fully unpacked tile occupies for the given pixel
    /// size (see [`crate::PtexReader::pixel_size`]).
    pub fn tile_size_bytes(&self, pixel_size: usize) -> usize {
        self.tile_res.size() * pixel_size
    }

    /// Build the descriptor for a face that is not split into tiles.
    pub(crate) fn untiled(res: Res, is_constant: bool, is_stored: bool) -> TileLayout {
        TileLayout {
            res,
            tile_res: res,
            ntilesu: 1,
            ntilesv: 1,
            is_tiled: false,
            is_constant,
            is_stored,
        }
    }
}

/// Facts about a single tile that can be answered from the tile directory
/// alone, without reading or decompressing the tile's pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileInfo {
    /// Resolution of this tile.
    pub res: Res,
    /// Origin of the tile in face texel coordinates, as (`u`, `v`).
    pub origin: (usize, usize),
    /// True if the tile is stored as a single constant value.
    ///
    /// Reading it still expands the value over the whole tile; a renderer
    /// that only needs the colour can shortcut on this flag.
    pub is_constant: bool,
    /// Compressed size of the tile's data block in the file, in bytes.
    ///
    /// Zero when the tile has no block of its own: a constant *face*, whose
    /// value lives once in the file's constant-data block, or a resolution
    /// that is computed rather than stored.
    pub compressed_size: u64,
    /// Absolute offset of the tile's data block in the file, or `None` when
    /// the tile has no block (see [`TileInfo::compressed_size`]).
    ///
    /// Useful for ordering reads by file position when prefetching.
    pub file_offset: Option<u64>,
}
