//! Tile streaming tests.
//!
//! The load-bearing property: tiles fetched individually and reassembled
//! must be byte-identical to the whole-face read, which `tests/reference.rs`
//! already pins to C++ ptex v2.4.3 output.

use std::path::PathBuf;

use ptex::{Error, PtexReader, Res};

const FIXTURES: [&str; 4] = ["quad_u8", "quad_f32", "quad_f16", "tri_u16"];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn open(name: &str) -> PtexReader {
    PtexReader::open(fixture(&format!("{name}.ptx"))).unwrap()
}

/// Read a face by streaming every tile into its place in a face-sized image.
fn assemble(tx: &mut PtexReader, faceid: usize, res: Res) -> Vec<u8> {
    let psize = tx.pixel_size();
    let stride = res.u() * psize;
    let mut img = vec![0u8; res.size() * psize];
    let layout = tx.tile_layout(faceid, res).unwrap();
    assert_eq!(layout.res, res);
    for tile in 0..layout.ntiles() {
        let (u, v) = layout.tile_origin(tile);
        let off = v * stride + u * psize;
        tx.get_tile_into(faceid, res, tile, &mut img[off..], stride)
            .unwrap();
    }
    img
}

#[test]
fn tiles_reassemble_full_res() {
    for name in FIXTURES {
        let mut tx = open(name);
        for faceid in 0..tx.num_faces() {
            let res = tx.face_info(faceid).unwrap().res;
            let expected = tx.get_data(faceid).unwrap();
            assert!(
                assemble(&mut tx, faceid, res) == expected,
                "{name} face {faceid}: tile assembly differs from get_data"
            );
        }
    }
}

#[test]
fn tiles_reassemble_stored_levels() {
    for name in FIXTURES {
        let mut tx = open(name);
        for faceid in 0..tx.num_faces() {
            for level in 0..tx.num_stored_levels(faceid).unwrap() {
                let res = tx.res_for_level(faceid, level).unwrap();
                assert!(tx.is_res_stored(faceid, res).unwrap());
                assert!(tx.tile_layout(faceid, res).unwrap().is_stored);
                let expected = tx.get_data_at_res(faceid, res).unwrap();
                assert!(
                    assemble(&mut tx, faceid, res) == expected,
                    "{name} face {faceid} level {level}: tile assembly differs"
                );
            }
        }
    }
}

#[test]
fn tiles_reassemble_every_level() {
    // Including levels that are not stored and so are computed by reduction.
    for name in FIXTURES {
        let mut tx = open(name);
        for faceid in 0..tx.num_faces() {
            for level in 0..tx.face_num_levels(faceid).unwrap() {
                let res = tx.res_for_level(faceid, level).unwrap();
                let expected = tx.get_data_at_res(faceid, res).unwrap();
                assert!(
                    assemble(&mut tx, faceid, res) == expected,
                    "{name} face {faceid} level {level}: tile assembly differs"
                );
            }
        }
    }
}

#[test]
fn computed_resolutions_report_one_tile() {
    // quad_f16 face 0 is 32x16; an anisotropic reduction is never stored.
    let mut tx = open("quad_f16");
    let res = Res::new(3, 4 - 1);
    assert!(!tx.is_res_stored(0, res).unwrap());
    let layout = tx.tile_layout(0, res).unwrap();
    assert!(!layout.is_stored);
    assert!(!layout.is_tiled);
    assert_eq!(layout.ntiles(), 1);
    assert_eq!(layout.tile_res, res);
    assert_eq!(
        tx.get_tile(0, res, 0).unwrap(),
        tx.get_data_at_res(0, res).unwrap()
    );
    let info = tx.tile_info(0, res, 0).unwrap();
    assert_eq!(info.compressed_size, 0);
    assert_eq!(info.file_offset, None);
}

#[test]
fn constant_and_untiled_faces_report_one_tile() {
    let mut tx = open("quad_u8");
    // face 2 of quad_u8 is a constant face
    let const_face = (0..tx.num_faces())
        .find(|&f| tx.face_info(f).unwrap().is_constant())
        .expect("fixture has a constant face");
    let res = tx.face_info(const_face).unwrap().res;
    let layout = tx.tile_layout(const_face, res).unwrap();
    assert!(layout.is_constant);
    assert!(!layout.is_tiled);
    assert!(layout.is_stored);
    assert_eq!(layout.ntiles(), 1);
    let info = tx.tile_info(const_face, res, 0).unwrap();
    assert!(info.is_constant);
    assert_eq!(info.compressed_size, 0);
    assert_eq!(info.file_offset, None);

    // a plain zipped/diffzipped face
    let plain = (0..tx.num_faces())
        .find(|&f| !tx.face_info(f).unwrap().is_constant())
        .unwrap();
    let res = tx.face_info(plain).unwrap().res;
    let layout = tx.tile_layout(plain, res).unwrap();
    assert!(!layout.is_constant);
    assert!(!layout.is_tiled);
    assert!(layout.is_stored);
    assert_eq!(layout.ntiles(), 1);
    let info = tx.tile_info(plain, res, 0).unwrap();
    assert!(!info.is_constant);
    assert!(info.compressed_size > 0);
    assert!(info.file_offset.is_some());
}

#[test]
fn tiled_face_shape_and_info() {
    let mut tx = open("quad_f32");
    let res = tx.face_info(0).unwrap().res;
    assert_eq!(res, Res::new(7, 7));
    let layout = tx.tile_layout(0, res).unwrap();
    assert!(layout.is_tiled);
    assert!(layout.is_stored);
    assert!(!layout.is_constant);
    assert_eq!(layout.tile_res, Res::new(7, 6));
    assert_eq!((layout.ntilesu, layout.ntilesv), (1, 2));
    assert_eq!(layout.ntiles(), 2);

    assert_eq!(layout.tile_origin(0), (0, 0));
    assert_eq!(layout.tile_origin(1), (0, 64));
    assert_eq!(layout.tile_index(0, 0), 0);
    assert_eq!(layout.tile_index(127, 63), 0);
    assert_eq!(layout.tile_index(0, 64), 1);
    assert_eq!(layout.tile_index(127, 127), 1);
    assert_eq!(layout.tile_size_bytes(tx.pixel_size()), 128 * 64 * 12);

    // every texel maps into the tile whose span contains it
    for v in (0..res.v()).step_by(7) {
        for u in (0..res.u()).step_by(5) {
            let tile = layout.tile_index(u, v);
            let (ou, ov) = layout.tile_origin(tile);
            assert!(u >= ou && u < ou + layout.tile_res.u());
            assert!(v >= ov && v < ov + layout.tile_res.v());
        }
    }

    // per-tile info is answerable without reading pixels, and the tiles are
    // laid out consecutively in the file
    let t0 = tx.tile_info(0, res, 0).unwrap();
    let t1 = tx.tile_info(0, res, 1).unwrap();
    assert_eq!(t0.res, layout.tile_res);
    assert_eq!(t0.origin, (0, 0));
    assert_eq!(t1.origin, (0, 64));
    assert!(!t0.is_constant && !t1.is_constant);
    assert!(t0.compressed_size > 0 && t1.compressed_size > 0);
    assert_eq!(
        t1.file_offset.unwrap(),
        t0.file_offset.unwrap() + t0.compressed_size
    );
}

#[test]
fn tile_out_of_range() {
    let mut tx = open("quad_f32");
    let res = tx.face_info(0).unwrap().res;
    assert!(matches!(
        tx.get_tile(0, res, 2),
        Err(Error::TileOutOfRange { tile: 2, ntiles: 2 })
    ));
    assert!(matches!(
        tx.tile_info(0, res, 9),
        Err(Error::TileOutOfRange { tile: 9, ntiles: 2 })
    ));
    // an untiled face has exactly one tile
    let res1 = tx.face_info(1).unwrap().res;
    assert!(matches!(
        tx.get_tile(1, res1, 1),
        Err(Error::TileOutOfRange { tile: 1, ntiles: 1 })
    ));
}

#[test]
fn get_tile_into_with_padded_stride() {
    let mut tx = open("quad_f32");
    let psize = tx.pixel_size();
    let res = tx.face_info(0).unwrap().res;
    let layout = tx.tile_layout(0, res).unwrap();
    let tile = 1;
    let rowlen = layout.tile_res.u() * psize;
    let stride = rowlen + 16;
    let mut buf = vec![0xabu8; stride * layout.tile_res.v()];
    tx.get_tile_into(0, res, tile, &mut buf, stride).unwrap();

    let face = tx.get_data(0).unwrap();
    let (ou, ov) = layout.tile_origin(tile);
    for v in 0..layout.tile_res.v() {
        let got = &buf[v * stride..v * stride + rowlen];
        let src = (ov + v) * res.u() * psize + ou * psize;
        assert_eq!(got, &face[src..src + rowlen], "row {v}");
        // padding bytes must be untouched
        if v + 1 < layout.tile_res.v() {
            assert!(buf[v * stride + rowlen..(v + 1) * stride]
                .iter()
                .all(|&b| b == 0xab));
        }
    }

    // buffers that are too small, or strides below the row length, are rejected
    let mut small = vec![0u8; rowlen * layout.tile_res.v() - 1];
    assert!(tx.get_tile_into(0, res, tile, &mut small, 0).is_err());
    let mut ok = vec![0u8; rowlen * layout.tile_res.v()];
    assert!(tx.get_tile_into(0, res, tile, &mut ok, rowlen - 1).is_err());
}

#[test]
fn repeated_tile_reads_are_stable_under_any_cache_capacity() {
    let mut tx = open("quad_f32");
    let res = tx.face_info(0).unwrap().res;
    assert_eq!(tx.tile_cache_capacity(), 16);
    let baseline: Vec<Vec<u8>> = (0..2).map(|t| tx.get_tile(0, res, t).unwrap()).collect();

    for cap in [0usize, 1, 64] {
        tx.set_tile_cache_capacity(cap);
        assert_eq!(tx.tile_cache_capacity(), cap.max(1));
        for _ in 0..3 {
            for (t, expected) in baseline.iter().enumerate() {
                assert!(&tx.get_tile(0, res, t).unwrap() == expected);
            }
        }
    }
}

#[test]
fn mip_level_helpers() {
    let tx = open("quad_f32");
    // face 0 is 128x128, face 1 is 32x32
    assert_eq!(tx.face_num_levels(0).unwrap(), 8);
    assert_eq!(tx.face_num_levels(1).unwrap(), 6);
    assert_eq!(tx.res_for_level(0, 0).unwrap(), Res::new(7, 7));
    assert_eq!(tx.res_for_level(0, 3).unwrap(), Res::new(4, 4));
    assert!(tx.res_for_level(0, 8).is_err());

    // stored levels are a prefix, and every one of them really is stored
    for faceid in 0..tx.num_faces() {
        let n = tx.num_stored_levels(faceid).unwrap();
        assert!(n >= 1);
        for level in 0..n {
            let res = tx.res_for_level(faceid, level).unwrap();
            assert!(tx.is_res_stored(faceid, res).unwrap());
        }
    }

    // a triangle fixture too
    let tri = open("tri_u16");
    assert_eq!(tri.face_num_levels(0).unwrap(), 6);
    assert_eq!(tri.res_for_level(0, 2).unwrap(), Res::new(3, 3));
}

#[test]
fn get_pixel_agrees_with_streamed_tiles() {
    for name in FIXTURES {
        let mut tx = open(name);
        let nchan = tx.num_channels();
        let psize = tx.pixel_size();
        for faceid in 0..tx.num_faces() {
            let res = tx.face_info(faceid).unwrap().res;
            let face = tx.get_data(faceid).unwrap();
            for (u, v) in [
                (0, 0),
                (res.u() - 1, res.v() - 1),
                (res.u() / 2, res.v() / 2),
            ] {
                let got = tx.get_pixel(faceid, u, v, 0, nchan).unwrap();
                let off = (v * res.u() + u) * psize;
                let mut expected = vec![0f32; nchan];
                ptex::utils::convert_to_float(
                    &mut expected,
                    &face[off..off + psize],
                    tx.data_type(),
                    nchan,
                );
                assert_eq!(got, expected, "{name} face {faceid} texel ({u},{v})");
            }
        }
    }
}
