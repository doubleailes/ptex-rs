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
- Single-texel access with float conversion (`get_pixel`)
- Optional alpha premultiplication (like the C++ `premultiply` flag)
- Meta data of all types, including large (LMD) entries
- No unsafe code; the only dependency is `flate2`

Writing files and filtered sampling (`PtexFilter`) are out of scope for
now.

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

A small CLI mirroring `ptxinfo` is included as an example:

```sh
cargo run --example ptxinfo -- model.ptx
```

## Testing

The integration tests in `tests/reference.rs` validate the reader against
reference data produced by the original C++ library: the fixture `.ptx`
files in `tests/fixtures/` were written with `PtexWriter`, and the
expected header info, face data (full-res and reduced), pixel samples and
meta data were dumped with the C++ `PtexReader`. The Rust reader's output
is required to match byte-for-byte.

## License

MIT (this crate). The Ptex file format and the original library are by
Walt Disney Animation Studios; the original C++ library is licensed under
the [Ptex license](https://github.com/wdas/ptex/blob/main/LICENSE).
