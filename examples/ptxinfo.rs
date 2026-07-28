//! Print information about a Ptex file, similar to the `ptxinfo` utility
//! from the original C++ distribution.
//!
//! Usage: `cargo run --example ptxinfo -- <file.ptx>`

use ptex::{MetaDataType, PtexReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: ptxinfo <file.ptx>")?;
    let mut tx = PtexReader::open(&path)?;

    println!("meshType: {}", tx.mesh_type().name());
    println!("dataType: {}", tx.data_type().name());
    println!("numChannels: {}", tx.num_channels());
    println!("alphaChannel: {}", tx.alpha_channel());
    println!("uBorderMode: {}", tx.u_border_mode().name());
    println!("vBorderMode: {}", tx.v_border_mode().name());
    println!("edgeFilterMode: {}", tx.edge_filter_mode().name());
    println!("numFaces: {}", tx.num_faces());
    println!("hasMipMaps: {}", tx.has_mip_maps());

    println!("faceinfo:");
    for f in 0..tx.num_faces() {
        let fi = *tx.face_info(f)?;
        println!(
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
        );
    }

    let meta = tx.metadata()?;
    if !meta.is_empty() {
        println!("meta:");
        for entry in meta.iter() {
            print!("  {} ({}):", entry.key(), entry.data_type().name());
            match entry.data_type() {
                MetaDataType::String => print!(" {:?}", entry.as_str().unwrap_or("")),
                MetaDataType::Int8 => entry.as_i8().unwrap().iter().for_each(|v| print!(" {v}")),
                MetaDataType::Int16 => entry.as_i16().unwrap().iter().for_each(|v| print!(" {v}")),
                MetaDataType::Int32 => entry.as_i32().unwrap().iter().for_each(|v| print!(" {v}")),
                MetaDataType::Float => entry.as_f32().unwrap().iter().for_each(|v| print!(" {v}")),
                MetaDataType::Double => entry.as_f64().unwrap().iter().for_each(|v| print!(" {v}")),
            }
            println!();
        }
    }
    Ok(())
}
