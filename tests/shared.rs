//! Tests for the thread-safe [`SharedReader`].
//!
//! `PtexReader` is pinned byte-for-byte to C++ ptex v2.4.3 output by
//! `tests/reference.rs`, so comparing against it is comparing against the
//! reference implementation.

#![cfg(feature = "cache")]

use std::io::Cursor;
use std::path::PathBuf;

use ptex::{CacheOptions, Error, PtexReader, Res, SharedReader};

const FIXTURES: [&str; 5] = ["quad_u8", "quad_f32", "quad_f16", "tri_u16", "quad_tiled"];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.ptx"))
}

fn pair(name: &str) -> (PtexReader, SharedReader) {
    (
        PtexReader::open(fixture(name)).unwrap(),
        SharedReader::open(fixture(name)).unwrap(),
    )
}

#[test]
fn handle_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SharedReader>();
    assert_send_sync::<SharedReader<Cursor<Vec<u8>>>>();
    assert_send_sync::<ptex::PixelData>();
    assert_send_sync::<ptex::CacheStats>();
}

#[test]
fn header_and_metadata_match_the_plain_reader() {
    for name in FIXTURES {
        let (mut plain, shared) = pair(name);
        assert_eq!(shared.mesh_type(), plain.mesh_type());
        assert_eq!(shared.data_type(), plain.data_type());
        assert_eq!(shared.num_faces(), plain.num_faces());
        assert_eq!(shared.num_channels(), plain.num_channels());
        assert_eq!(shared.alpha_channel(), plain.alpha_channel());
        assert_eq!(shared.pixel_size(), plain.pixel_size());
        assert_eq!(shared.num_levels(), plain.num_levels());
        assert_eq!(shared.has_mip_maps(), plain.has_mip_maps());
        assert_eq!(shared.minor_version(), plain.minor_version());
        assert_eq!(shared.face_infos(), plain.face_infos());
        for f in 0..shared.num_faces() {
            assert_eq!(
                shared.constant_data(f).unwrap(),
                plain.constant_data(f).unwrap()
            );
        }
        let expected: Vec<(String, usize)> = plain
            .metadata()
            .unwrap()
            .iter()
            .map(|e| (e.key().to_string(), e.raw_data().len()))
            .collect();
        let got: Vec<(String, usize)> = shared
            .metadata()
            .unwrap()
            .iter()
            .map(|e| (e.key().to_string(), e.raw_data().len()))
            .collect();
        assert_eq!(got, expected, "{name}: metadata");
    }
}

#[test]
fn pixel_data_matches_the_plain_reader() {
    for name in FIXTURES {
        let (mut plain, shared) = pair(name);
        let nchan = shared.num_channels();
        for faceid in 0..shared.num_faces() {
            for level in 0..plain.face_num_levels(faceid).unwrap() {
                let res = plain.res_for_level(faceid, level).unwrap();
                let expected = plain.get_data_at_res(faceid, res).unwrap();
                assert!(
                    shared.get_data_at_res(faceid, res).unwrap().as_slice() == expected.as_slice(),
                    "{name} face {faceid} level {level}: get_data_at_res"
                );

                // and through the caller-buffer API, with a padded stride
                let psize = shared.pixel_size();
                let rowlen = res.u() * psize;
                let stride = rowlen + 8;
                let mut buf = vec![0u8; stride * res.v()];
                shared.get_data_into(faceid, res, &mut buf, stride).unwrap();
                for v in 0..res.v() {
                    assert_eq!(
                        &buf[v * stride..v * stride + rowlen],
                        &expected[v * rowlen..(v + 1) * rowlen],
                        "{name} face {faceid} level {level} row {v}"
                    );
                }

                // tile layouts and tiles agree
                let el = plain.tile_layout(faceid, res).unwrap();
                let sl = shared.tile_layout(faceid, res).unwrap();
                assert_eq!(sl, el, "{name} face {faceid} level {level}: layout");
                for tile in 0..sl.ntiles() {
                    assert_eq!(
                        shared.tile_info(faceid, res, tile).unwrap(),
                        plain.tile_info(faceid, res, tile).unwrap()
                    );
                    assert!(
                        shared.get_tile(faceid, res, tile).unwrap().as_slice()
                            == plain.get_tile(faceid, res, tile).unwrap().as_slice(),
                        "{name} face {faceid} level {level} tile {tile}"
                    );
                }
            }

            let res = shared.face_info(faceid).unwrap().res;
            for (u, v) in [
                (0, 0),
                (res.u() - 1, res.v() - 1),
                (res.u() / 2, res.v() / 3),
            ] {
                assert_eq!(
                    shared.get_pixel(faceid, u, v, 0, nchan).unwrap(),
                    plain.get_pixel(faceid, u, v, 0, nchan).unwrap(),
                    "{name} face {faceid} texel ({u},{v})"
                );
            }
        }
    }
}

#[test]
fn premultiply_matches_the_plain_reader() {
    let mut plain = PtexReader::open_with_options(fixture("quad_u8"), true).unwrap();
    let shared = SharedReader::open_with_options(
        fixture("quad_u8"),
        CacheOptions {
            premultiply: true,
            ..CacheOptions::default()
        },
    )
    .unwrap();
    for faceid in 0..shared.num_faces() {
        let res = plain.face_info(faceid).unwrap().res;
        assert!(
            shared.get_data(faceid).unwrap().as_slice()
                == plain.get_data_at_res(faceid, res).unwrap().as_slice()
        );
    }
}

#[test]
fn reads_from_an_in_memory_stream() {
    let bytes = std::fs::read(fixture("quad_f32")).unwrap();
    let shared = SharedReader::new(Cursor::new(bytes)).unwrap();
    let mut plain = PtexReader::open(fixture("quad_f32")).unwrap();
    for faceid in 0..shared.num_faces() {
        assert!(shared.get_data(faceid).unwrap().as_slice() == plain.get_data(faceid).unwrap());
    }
}

#[test]
fn tile_index_out_of_range() {
    let shared = SharedReader::open(fixture("quad_f32")).unwrap();
    let res = shared.face_info(0).unwrap().res;
    assert!(matches!(
        shared.get_tile(0, res, 2),
        Err(Error::TileOutOfRange { tile: 2, ntiles: 2 })
    ));
}

#[test]
fn many_threads_share_one_handle() {
    // Compute every expected answer single-threaded first, so the threaded
    // pass has nothing to race against but itself.
    let mut plain = PtexReader::open(fixture("quad_f32")).unwrap();
    let mut expected: Vec<(usize, Res, usize, Vec<u8>)> = Vec::new();
    for faceid in 0..plain.num_faces() {
        for level in 0..plain.face_num_levels(faceid).unwrap() {
            let res = plain.res_for_level(faceid, level).unwrap();
            let layout = plain.tile_layout(faceid, res).unwrap();
            for tile in 0..layout.ntiles() {
                let data = plain.get_tile(faceid, res, tile).unwrap();
                expected.push((faceid, res, tile, data));
            }
        }
    }
    assert!(expected.len() > 8);

    let shared = SharedReader::open(fixture("quad_f32")).unwrap();
    std::thread::scope(|s| {
        for t in 0..8usize {
            let shared = shared.clone();
            let expected = &expected;
            s.spawn(move || {
                // Two passes: the first staggered so threads spread over the
                // work, the second identical for every thread so they race
                // on the same entries.
                for (i, (faceid, res, tile, want)) in expected.iter().enumerate() {
                    if i % 8 == t {
                        assert!(shared.get_tile(*faceid, *res, *tile).unwrap().as_slice() == want);
                    }
                }
                for (faceid, res, tile, want) in expected.iter() {
                    assert!(shared.get_tile(*faceid, *res, *tile).unwrap().as_slice() == want);
                }
            });
        }
    });

    let stats = shared.cache_stats();
    assert!(stats.hits > 0);
    assert!(stats.bytes_resident <= stats.bytes_budget);
}

#[test]
fn results_are_exact_under_every_cache_budget() {
    for name in FIXTURES {
        let mut plain = PtexReader::open(fixture(name)).unwrap();
        let mut expected = Vec::new();
        for faceid in 0..plain.num_faces() {
            for level in 0..plain.face_num_levels(faceid).unwrap() {
                let res = plain.res_for_level(faceid, level).unwrap();
                expected.push((faceid, res, plain.get_data_at_res(faceid, res).unwrap()));
            }
        }

        for budget in [0usize, 1, 1024, 64 * 1024, usize::MAX] {
            let shared = SharedReader::open_with_options(
                fixture(name),
                CacheOptions {
                    premultiply: false,
                    budget_bytes: budget,
                },
            )
            .unwrap();
            assert_eq!(shared.cache_budget(), budget);
            // read everything twice, so eviction and reuse both happen
            for _ in 0..2 {
                for (faceid, res, want) in &expected {
                    assert!(
                        shared.get_data_at_res(*faceid, *res).unwrap().as_slice()
                            == want.as_slice(),
                        "{name} face {faceid} at budget {budget}"
                    );
                }
            }
            let stats = shared.cache_stats();
            assert!(stats.bytes_resident <= stats.bytes_budget);
            if budget == 0 {
                assert_eq!(stats.entries, 0);
                assert!(stats.oversized > 0);
            }
        }
    }
}

#[test]
fn shrinking_the_budget_evicts_and_clearing_empties() {
    let shared = SharedReader::open(fixture("quad_f32")).unwrap();
    for faceid in 0..shared.num_faces() {
        let res = shared.face_info(faceid).unwrap().res;
        let _ = shared.get_data_at_res(faceid, res).unwrap();
    }
    let before = shared.cache_stats();
    assert!(before.bytes_resident > 0);
    assert!(before.entries > 0);

    shared.set_cache_budget(1024);
    let shrunk = shared.cache_stats();
    assert!(shrunk.bytes_resident <= 1024);
    assert!(shrunk.evictions > 0);

    shared.set_cache_budget(ptex::DEFAULT_CACHE_BUDGET);
    let res = shared.face_info(0).unwrap().res;
    let again = shared.get_data_at_res(0, res).unwrap();
    let mut plain = PtexReader::open(fixture("quad_f32")).unwrap();
    assert!(again.as_slice() == plain.get_data(0).unwrap());

    shared.clear_cache();
    assert_eq!(shared.cache_stats().bytes_resident, 0);
    assert_eq!(shared.cache_stats().entries, 0);
    // still correct after a cold cache
    assert!(shared.get_data_at_res(0, res).unwrap().as_slice() == plain.get_data(0).unwrap());
}
