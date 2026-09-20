# ptex-rs

A pure-Rust reader for Disney's [Ptex](https://ptex.us/) per-face texture
file format — a port of the reading side of the original C++
[wdas/ptex](https://github.com/wdas/ptex) library.

## Features

- Open `.ptx` files (Ptex file format version 1, quad and triangle meshes)
- Header information: mesh type, data type (`uint8`, `uint16`, `float16`,
  `float32`), channels, alpha channel, border/edge filter modes
- Per-face resolution and adjacency information (`FaceInfo`)
- Full-resolution pixel data for every face, covering all on-disk
  encodings: constant, zip-compressed, difference-encoded, and tiled
  faces (including "large face" offsets)
- Stored mipmap (reduction) levels, plus dynamically computed reductions
  for resolutions not stored in the file (matching the C++ reduction
  kernels bit-for-bit, including triangle reductions)
- **Tile streaming**: ask how a face is tiled at a chosen mipmap level and
  read exactly the tiles you need, instead of materializing whole faces
  (`tile_layout`, `tile_info`, `get_tile`, `get_tile_into`)
- **Thread-safe shared reading**: `SharedReader` is a cheaply clonable
  handle over one open file, with a byte-budgeted cache of decoded pixels
- Single-texel access with float conversion (`get_pixel`)
- Optional alpha premultiplication (like the C++ `premultiply` flag)
- Meta data of all types, including large (LMD) entries
- No unsafe code; `flate2` is the only required dependency (`lru`, used by
  the default-on `cache` feature, is the only other one)

Writing files and filtered sampling (`PtexFilter`) are out of scope for
now. **Edit blocks are not applied**: Ptex files can be modified by
appending edit records instead of being rewritten, and this reader ignores
them, so an edited file reads back in its pre-edit state. `has_edits()`
reports whether a file carries them, so a caller can refuse or fall back
rather than silently using stale data.

## Usage

```rust
use ptex::PtexReader;

let mut tx = PtexReader::open("model.ptx")?;
println!(
    "{} faces, {} channels of {}",
    tx.num_faces(),
    tx.num_channels(),
    tx.data_type().name(),
);

for faceid in 0..tx.num_faces() {
    let info = *tx.face_info(faceid)?;
    // interleaved pixels, v-major, res.v() rows of res.u() pixels
    let data = tx.get_data(faceid)?;
    println!(
        "face {faceid}: {}x{}, {} bytes",
        info.res.u(),
        info.res.v(),
        data.len()
    );
}

// read at a reduced resolution (stored mipmap or computed reduction)
let info = *tx.face_info(0)?;
let half_res = ptex::Res::new(info.res.ulog2 - 1, info.res.vlog2 - 1);
let reduced = tx.get_data_at_res(0, half_res)?;

// single texel as f32 channel values
let texel = tx.get_pixel(0, 3, 5, 0, tx.num_channels())?;

// meta data
if let Some(name) = tx.metadata()?.get_string("PtexFaceVertCounts") {
    println!("{name}");
}
# Ok::<(), ptex::Error>(())
```

### Tile streaming

Ptex stores mipmap levels per face, and splits faces larger than 64 KB into
independently compressed tiles. A renderer can read exactly the tile it
needs, at the resolution it needs, rather than paying for a whole face:

```rust
use ptex::PtexReader;

let mut tx = PtexReader::open("model.ptx")?;

// pick a mipmap level, and find out whether it streams from the file or
// has to be computed by reduction
let res = tx.res_for_level(0, 2)?;
println!("stored: {}", tx.is_res_stored(0, res)?);

let layout = tx.tile_layout(0, res)?;
println!(
    "{}x{} tiles of {}x{}",
    layout.ntilesu,
    layout.ntilesv,
    layout.tile_res.u(),
    layout.tile_res.v(),
);

// the tile covering a particular texel, read on its own
let tile = layout.tile_index(96, 40);
let pixels = tx.get_tile(0, res, tile)?;

// or schedule reads by file position without touching pixel data
let info = tx.tile_info(0, res, tile)?;
println!("{:?} {} bytes", info.file_offset, info.compressed_size);
# Ok::<(), ptex::Error>(())
```

A face that is *not* split into tiles — constant, zipped, or a resolution
computed by reduction — reports a single tile covering the whole face, so
renderer code needs only one path. Tiles are numbered v-major
(`tile = tile_v * ntilesu + tile_u`) and, because Ptex subdivides by powers
of two, every tile has the same resolution: there are no partial edge tiles.

### Shared, multi-threaded reading

`SharedReader` (the default-on `cache` feature) is `Send + Sync` and clones
for the price of an `Arc` bump. Every clone shares one open stream, one set
of parsed indexes, and one cache of decoded pixels bounded by a byte budget.
The I/O lock is held only around the seek and read, so threads decompress in
parallel:

```rust
use ptex::SharedReader;

let tx = SharedReader::open("model.ptx")?;
std::thread::scope(|s| {
    for t in 0..4 {
        let tx = tx.clone();
        s.spawn(move || {
            for faceid in (t..tx.num_faces()).step_by(4) {
                let res = tx.res_for_level(faceid, 1).unwrap();
                let layout = tx.tile_layout(faceid, res).unwrap();
                for tile in 0..layout.ntiles() {
                    // reference-counted: a repeated request copies nothing
                    let _pixels = tx.get_tile(faceid, res, tile).unwrap();
                }
            }
        });
    }
});
# Ok::<(), ptex::Error>(())
```

Use `CacheOptions` to set the budget (64 MiB by default) and
`cache_stats()` to see hits, misses, evictions and resident bytes. Building
with `--no-default-features` drops `lru` and leaves the single-threaded
`PtexReader`, with `flate2` as the only dependency.

Two small CLIs are included as examples:

```sh
cargo run --example ptxinfo -- model.ptx
cargo run --example ptxtile -- model.ptx 0 2      # list tiles of a level
cargo run --example ptxtile -- model.ptx 0 2 1    # read one tile
```

## Testing

The integration tests in `tests/reference.rs` validate the reader against
reference data produced by the original C++ library: the fixture `.ptx`
files in `tests/fixtures/` were written with `PtexWriter`, and the
expected header info, face data (full-res and reduced), pixel samples and
meta data were dumped with the C++ `PtexReader`. The Rust reader's output
is required to match byte-for-byte.

`tests/tiles.rs` and `tests/shared.rs` build on that guarantee: for every
fixture, face and level, tiles fetched individually and reassembled must
equal the whole-face read, and `SharedReader` must agree with `PtexReader`
under every cache budget and across threads. The `quad_tiled` fixture is a
1024x512 face that the writer splits into a 4x2 grid of tiles and whose
first mipmap level is tiled too; its face data would be half a megabyte, so
only the `.ptx` and a dense grid of reference pixel samples are committed.

## License

MIT (this crate). The Ptex file format and the original library are by
Walt Disney Animation Studios; the original C++ library is licensed under
the [Ptex license](https://github.com/wdas/ptex/blob/main/LICENSE).
