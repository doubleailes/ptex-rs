//! Print information about a Ptex file, similar to the `ptxinfo` utility
//! from the original C++ distribution.
//!
//! Usage: `cargo run --example ptxinfo -- <file.ptx>`

use std::io::Write;
use std::process::ExitCode;

use ptex::{MetaDataType, PtexReader};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        // A closed pipe (`ptxinfo ... | head`) is a normal way for a reader to
        // stop listening, not a failure.  Rust ignores SIGPIPE, so the print
        // macros would panic here instead; writing through `?` lets us exit
        // quietly.
        Err(e) if is_broken_pipe(e.as_ref()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn is_broken_pipe(e: &(dyn std::error::Error + 'static)) -> bool {
    e.downcast_ref::<std::io::Error>()
        .is_some_and(|io| io.kind() == std::io::ErrorKind::BrokenPipe)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: ptxinfo <file.ptx>")?;
    let mut tx = PtexReader::open(&path)?;
    let mut out = std::io::stdout().lock();

    writeln!(out, "meshType: {}", tx.mesh_type().name())?;
    writeln!(out, "dataType: {}", tx.data_type().name())?;
    writeln!(out, "numChannels: {}", tx.num_channels())?;
    writeln!(out, "alphaChannel: {}", tx.alpha_channel())?;
    writeln!(out, "uBorderMode: {}", tx.u_border_mode().name())?;
    writeln!(out, "vBorderMode: {}", tx.v_border_mode().name())?;
    writeln!(out, "edgeFilterMode: {}", tx.edge_filter_mode().name())?;
    writeln!(out, "numFaces: {}", tx.num_faces())?;
    writeln!(out, "hasMipMaps: {}", tx.has_mip_maps())?;

    writeln!(out, "faceinfo:")?;
    for f in 0..tx.num_faces() {
        let fi = *tx.face_info(f)?;
        writeln!(
            out,
            "  face {f}: res {}x{}{}{} adjface ({} {} {} {}) adjedge ({} {} {} {})",
            fi.res.u(),
            fi.res.v(),
            if fi.is_constant() { " const" } else { "" },
            if fi.is_subface() { " subface" } else { "" },
            fi.adjfaces[0],
            fi.adjfaces[1],
            fi.adjfaces[2],
            fi.adjfaces[3],
            fi.adjedge(0) as u8,
            fi.adjedge(1) as u8,
            fi.adjedge(2) as u8,
            fi.adjedge(3) as u8,
        )?;
    }

    let meta = tx.metadata()?;
    if !meta.is_empty() {
        writeln!(out, "meta:")?;
        for entry in meta.iter() {
            write!(out, "  {} ({}):", entry.key(), entry.data_type().name())?;
            match entry.data_type() {
                MetaDataType::String => write!(out, " {:?}", entry.as_str().unwrap_or(""))?,
                MetaDataType::Int8 => {
                    for v in entry.as_i8().unwrap() {
                        write!(out, " {v}")?;
                    }
                }
                MetaDataType::Int16 => {
                    for v in entry.as_i16().unwrap() {
                        write!(out, " {v}")?;
                    }
                }
                MetaDataType::Int32 => {
                    for v in entry.as_i32().unwrap() {
                        write!(out, " {v}")?;
                    }
                }
                MetaDataType::Float => {
                    for v in entry.as_f32().unwrap() {
                        write!(out, " {v}")?;
                    }
                }
                MetaDataType::Double => {
                    for v in entry.as_f64().unwrap() {
                        write!(out, " {v}")?;
                    }
                }
            }
            writeln!(out)?;
        }
    }
    Ok(())
}
