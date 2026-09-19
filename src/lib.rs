//! A pure-Rust reader for Disney's [Ptex] per-face texture file format.
//!
//! This crate is a port of the reading side of the original C++
//! [wdas/ptex](https://github.com/wdas/ptex) library.  It can open `.ptx`
//! files and read:
//!
//! - header information (mesh type, data type, channels, alpha channel, …)
//! - per-face resolution and adjacency information ([`FaceInfo`])
//! - full-resolution pixel data for each face, including constant,
//!   zip-compressed, difference-encoded, and tiled faces
//! - stored mipmap (reduction) levels, and dynamically computed
//!   reductions for resolutions not stored in the file
//! - meta data (including large meta data blocks)
//!
//! Faces larger than 64 KB are stored as independently compressed tiles,
//! and [`PtexReader`] exposes them: [`PtexReader::tile_layout`] describes
//! how a face is laid out at a chosen resolution and
//! [`PtexReader::get_tile`] reads one tile, so a render engine can stream
//! exactly what it needs instead of materializing whole faces.  For reading
//! from several threads against one open file, with a bounded cache of
//! decoded pixels, see [`SharedReader`] (enabled by the default `cache`
//! feature).
//!
//! Writing and filtered sampling are out of scope for now.
//!
//! # Example
//!
//! ```no_run
//! use ptex::PtexReader;
//!
//! let mut tx = PtexReader::open("teapot.ptx")?;
//! println!(
//!     "{} faces, {} channels of {}",
//!     tx.num_faces(),
//!     tx.num_channels(),
//!     tx.data_type().name(),
//! );
//! for faceid in 0..tx.num_faces() {
//!     let info = *tx.face_info(faceid)?;
//!     let data = tx.get_data(faceid)?; // interleaved pixels, v-major
//!     println!("face {faceid}: {}x{} -> {} bytes", info.res.u(), info.res.v(), data.len());
//! }
//! # Ok::<(), ptex::Error>(())
//! ```
//!
//! # Streaming one tile of one mipmap level
//!
//! ```no_run
//! use ptex::PtexReader;
//!
//! let mut tx = PtexReader::open("teapot.ptx")?;
//! let res = tx.res_for_level(0, 2)?;            // two levels down
//! let layout = tx.tile_layout(0, res)?;         // one tile if untiled
//! let tile = layout.tile_index(96, 40);         // the tile holding a texel
//! let pixels = tx.get_tile(0, res, tile)?;      // one seek, one inflate
//! println!("{} bytes for {}x{}", pixels.len(), layout.tile_res.u(), layout.tile_res.v());
//! # Ok::<(), ptex::Error>(())
//! ```
//!
//! [Ptex]: https://ptex.us/

#![warn(missing_docs)]

#[cfg(feature = "cache")]
mod cache;
mod decode;
mod error;
mod file_info;
mod format;
mod metadata;
mod reader;
mod tile;
mod types;
pub mod utils;

#[cfg(feature = "cache")]
pub use cache::{CacheOptions, CacheStats, PixelData, SharedReader, DEFAULT_CACHE_BUDGET};
pub use error::{Error, Result};
pub use metadata::{MetaData, MetaDataEntry};
pub use reader::PtexReader;
pub use tile::{TileInfo, TileLayout};
pub use types::{
    face_flags, float_to_half, half_to_float, BorderMode, DataType, EdgeFilterMode, EdgeId,
    FaceInfo, MeshType, MetaDataType, Res,
};
