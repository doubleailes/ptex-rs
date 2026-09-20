//! Stream a single tile of a single mipmap level out of a Ptex file.
//!
//! ```sh
//! cargo run --example ptxtile -- model.ptx [faceid] [level] [tile]
//! ```
//!
//! With no tile given, every tile of the level is listed and the face's
//! tiling is summarised; with a tile index, that one tile's pixels are read
//! and its first few texels printed.

use std::io::Write;
use std::process::ExitCode;

use ptex::PtexReader;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: ptxtile <file.ptx> [faceid] [level] [tile]");
        return ExitCode::FAILURE;
    };
    let faceid: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let level: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let tile: Option<usize> = args.next().and_then(|a| a.parse().ok());

    match run(&path, faceid, level, tile) {
        Ok(()) => ExitCode::SUCCESS,
        // A closed pipe (`ptxtile ... | head`) is a normal way for a reader to
        // stop listening, not a failure.  Rust ignores SIGPIPE, so the print
        // macros would panic here instead; writing through `?` lets us exit
        // quietly.
        Err(ptex::Error::Io(e)) if e.kind() == std::io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{path}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &str, faceid: usize, level: usize, tile: Option<usize>) -> ptex::Result<()> {
    let mut out = std::io::stdout().lock();
    let mut tx = PtexReader::open(path)?;
    let nlevels = tx.face_num_levels(faceid)?;
    let stored = tx.num_stored_levels(faceid)?;
    writeln!(
        out,
        "face {faceid}: {} levels ({stored} stored in the file)",
        nlevels
    )?;
    if level >= nlevels {
        return Err(ptex::Error::Unsupported(format!(
            "face {faceid} has only {nlevels} levels"
        )));
    }

    let res = tx.res_for_level(faceid, level)?;
    let layout = tx.tile_layout(faceid, res)?;
    writeln!(
        out,
        "level {level}: {}x{}  {}  tiles {}x{} of {}x{}",
        res.u(),
        res.v(),
        if layout.is_stored {
            "streamed from the file"
        } else {
            "computed by reduction"
        },
        layout.ntilesu,
        layout.ntilesv,
        layout.tile_res.u(),
        layout.tile_res.v(),
    )?;

    let Some(tile) = tile else {
        for t in 0..layout.ntiles() {
            let info = tx.tile_info(faceid, res, t)?;
            let (u, v) = info.origin;
            match info.file_offset {
                Some(off) => writeln!(
                    out,
                    "  tile {t}: origin ({u},{v}) at file offset {off}, \
                     {} compressed bytes{}",
                    info.compressed_size,
                    if info.is_constant { ", constant" } else { "" }
                )?,
                None => writeln!(out, "  tile {t}: origin ({u},{v}), not stored")?,
            }
        }
        return Ok(());
    };

    // Read only this tile: one seek and one inflate, whatever the face size.
    let data = tx.get_tile(faceid, res, tile)?;
    let (u, v) = layout.tile_origin(tile);
    writeln!(
        out,
        "tile {tile}: origin ({u},{v}), {}x{}, {} bytes",
        layout.tile_res.u(),
        layout.tile_res.v(),
        data.len()
    )?;

    let nchan = tx.num_channels();
    let dt = tx.data_type();
    let psize = tx.pixel_size();
    for i in 0..data.len().min(4 * psize) / psize {
        let mut texel = vec![0f32; nchan];
        ptex::utils::convert_to_float(&mut texel, &data[i * psize..], dt, nchan);
        let values: Vec<String> = texel.iter().map(|c| format!("{c:.4}")).collect();
        writeln!(out, "  texel {i}: {}", values.join(" "))?;
    }
    Ok(())
}
