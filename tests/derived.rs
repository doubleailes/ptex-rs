//! Tests for host-derived mip chains ([`DerivedLevels`]) on [`SharedReader`].

#![cfg(feature = "cache")]

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ptex::{CacheOptions, DerivedLevels, Res, SharedReader};

const FIXTURES: [&str; 5] = ["quad_u8", "quad_f32", "quad_f16", "tri_u16", "quad_tiled"];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.ptx"))
}

/// A host chain: capped at `cap` per axis, "decoded" by widening every byte
/// to an `f32`, and reduced by the mean of each 2x2 block (clamped on a
/// pinned axis).  Any pure function would do; this one makes every level
/// depend on every texel of its parent.
struct Widen {
    cap: i8,
    /// Bytes per pixel in the file.
    pixel_size: usize,
    /// Cap by reducing both axes together, keeping the face's aspect, so the
    /// base is a level the file stores; otherwise each axis clamps alone.
    stored: bool,
}

impl DerivedLevels for Widen {
    fn base_res(&self, _faceid: usize, res: Res) -> Res {
        if self.stored {
            let d = (res.ulog2.max(res.vlog2) - self.cap).max(0);
            Res::new((res.ulog2 - d).max(0), (res.vlog2 - d).max(0))
        } else {
            Res::new(res.ulog2.min(self.cap), res.vlog2.min(self.cap))
        }
    }

    fn decode(&self, _faceid: usize, res: Res, pixels: &[u8]) -> Vec<u8> {
        assert_eq!(pixels.len(), res.size() * self.pixel_size);
        pixels
            .iter()
            .flat_map(|&b| (b as f32).to_le_bytes())
            .collect()
    }

    fn derive(&self, _faceid: usize, k: u8, parent_res: Res, parent: &[u8]) -> Vec<u8> {
        assert!(k >= 1);
        let n = self.pixel_size;
        let (sw, sh) = (parent_res.u(), parent_res.v());
        assert_eq!(parent.len(), sw * sh * n * 4);
        let at = |x: usize, y: usize, c: usize| {
            let i = ((y * sw + x) * n + c) * 4;
            f32::from_le_bytes(parent[i..i + 4].try_into().unwrap())
        };
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut out = Vec::with_capacity(dw * dh * n * 4);
        for y in 0..dh {
            for x in 0..dw {
                let (x0, x1) = ((2 * x).min(sw - 1), (2 * x + 1).min(sw - 1));
                let (y0, y1) = ((2 * y).min(sh - 1), (2 * y + 1).min(sh - 1));
                for c in 0..n {
                    let m = 0.25 * (at(x0, y0, c) + at(x1, y0, c) + at(x0, y1, c) + at(x1, y1, c));
                    out.extend_from_slice(&m.to_le_bytes());
                }
            }
        }
        out
    }
}

fn open(name: &str, cap: i8, budget: usize) -> SharedReader {
    let options = CacheOptions {
        budget_bytes: budget,
        ..Default::default()
    };
    let tx = SharedReader::open_with_options(fixture(name), options).unwrap();
    let pixel_size = tx.pixel_size();
    tx.with_derived(Arc::new(Widen {
        cap,
        pixel_size,
        stored: false,
    }))
    .unwrap()
}

/// Every derived level of every face, in order.
fn chain<R: Read + Seek + Send>(tx: &SharedReader<R>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for f in 0..tx.num_faces() {
        for k in 0..tx.derived_levels(f).unwrap() {
            out.push(tx.get_derived(f, k).unwrap().to_vec());
        }
    }
    out
}

#[test]
fn level_zero_is_the_decoded_base_and_each_level_halves() {
    for name in FIXTURES {
        let tx = open(name, 3, 64 << 20);
        for f in 0..tx.num_faces() {
            let base = tx.derived_base_res(f).unwrap();
            let file = tx.get_data_at_res(f, base).unwrap();
            let level0 = tx.get_derived(f, 0).unwrap();
            let widened: Vec<u8> = file
                .iter()
                .flat_map(|&b| (b as f32).to_le_bytes())
                .collect();
            assert_eq!(&*level0, &widened[..], "{name} face {f}");
            let n = tx.derived_levels(f).unwrap();
            assert_eq!(tx.derived_res(f, n - 1).unwrap(), Res::new(0, 0));
            for k in 1..n {
                let r = tx.derived_res(f, k).unwrap();
                let p = tx.derived_res(f, k - 1).unwrap();
                assert_eq!(r.u(), (p.u() / 2).max(1), "{name} face {f} level {k}");
                assert_eq!(r.v(), (p.v() / 2).max(1), "{name} face {f} level {k}");
                let block = tx.get_derived(f, k).unwrap();
                assert_eq!(block.len(), r.size() * tx.pixel_size() * 4);
            }
            assert!(tx.get_derived(f, n).is_err(), "{name} face {f}");
        }
    }
}

#[test]
fn without_a_host_chain_get_derived_is_refused() {
    let tx = SharedReader::open(fixture("quad_u8")).unwrap();
    assert!(!tx.has_derived());
    assert!(tx.get_derived(0, 0).is_err());
    let tx = open("quad_u8", 2, 64 << 20);
    assert!(tx.has_derived());
    // Once per reader, clones included.
    let again = Widen {
        cap: 1,
        pixel_size: tx.pixel_size(),
        stored: false,
    };
    assert!(tx.clone().with_derived(Arc::new(again)).is_err());
}

#[test]
fn rederiving_after_eviction_is_deterministic() {
    for name in FIXTURES {
        let tx = open(name, 4, 64 << 20);
        let first = chain(&tx);
        let derives = tx.cache_stats().derives;
        assert_eq!(
            derives as usize,
            first.len(),
            "{name}: one derive per block"
        );
        tx.clear_cache();
        assert_eq!(tx.cache_stats().derived_blocks, 0);
        assert_eq!(chain(&tx), first, "{name} after clear");
        // A budget too small to keep a chain evicts as it goes and still
        // answers the same.
        let tight = open(name, 4, 4096);
        assert_eq!(chain(&tight), first, "{name} under a tight budget");
        assert!(
            tight.cache_stats().derives as usize >= first.len(),
            "{name}: evicted parents are derived again"
        );
    }
}

#[test]
fn a_coarse_request_caches_only_the_level_asked_for() {
    for name in FIXTURES {
        let full = open(name, 4, 64 << 20);
        let expected = chain(&full);
        let tx = open(name, 4, 64 << 20);
        let mut i = 0;
        let mut blocks = 0;
        let mut derives = 0;
        for f in 0..tx.num_faces() {
            let n = tx.derived_levels(f).unwrap();
            // The coarsest level first: derived through the whole chain,
            // which is not kept.
            let coarse = tx.get_derived(f, n - 1).unwrap();
            assert_eq!(&*coarse, &expected[i + n - 1][..], "{name} face {f}");
            blocks += 1;
            derives += n;
            let s = tx.cache_stats();
            assert_eq!(
                s.derived_blocks, blocks,
                "{name} face {f}: only the asked level"
            );
            assert_eq!(s.derives as usize, derives, "{name} face {f}");
            // A finer level after it starts again from the base, and a level
            // between the two then starts from that one.
            if n >= 4 {
                assert_eq!(&*tx.get_derived(f, 0).unwrap(), &expected[i][..]);
                derives += 1;
                assert_eq!(&*tx.get_derived(f, 2).unwrap(), &expected[i + 2][..]);
                derives += 2;
                blocks += 2;
                let s = tx.cache_stats();
                assert_eq!(s.derived_blocks, blocks, "{name} face {f}");
                assert_eq!(s.derives as usize, derives, "{name} face {f}");
            }
            i += n;
        }
    }
}

#[test]
fn the_base_block_read_to_derive_is_not_kept() {
    // `quad_u8`'s faces are stored untiled at every level a cap of 4 reads,
    // so the only cache entries a derived chain leaves are its own blocks.
    let tx = open("quad_u8", 4, 64 << 20);
    let expected = chain(&open("quad_u8", 4, 64 << 20));
    assert_eq!(chain(&tx), expected);
    let s = tx.cache_stats();
    assert_eq!(s.entries, s.derived_blocks, "a raw base block was kept");
    // A base block that *is* resident is used rather than read again.
    let tx = open("quad_u8", 4, 64 << 20);
    for f in 0..tx.num_faces() {
        let base = tx.derived_base_res(f).unwrap();
        tx.get_data_at_res(f, base).unwrap();
    }
    let before = tx.cache_stats();
    assert_eq!(chain(&tx), expected);
    let after = tx.cache_stats();
    assert_eq!(after.entries - after.derived_blocks, before.entries);
}

#[test]
fn a_reduced_base_and_its_source_are_not_kept() {
    // Capping each axis of `quad_u8`'s 8x4 face at 2 asks for 4x4, which the
    // file does not store: the reader reduces it from 8x4. Neither block may
    // outlive the derivation, and the chain must equal one read through
    // `get_data_at_res`, which caches both.
    let tx = open("quad_u8", 2, 64 << 20);
    let mut reduced = 0;
    for f in 0..tx.num_faces() {
        let base = tx.derived_base_res(f).unwrap();
        if !tx.is_res_stored(f, base).unwrap() {
            reduced += 1;
        }
    }
    assert!(reduced > 0, "the fixture has a reduced base at this cap");
    let expected = {
        let warm = open("quad_u8", 2, 64 << 20);
        for f in 0..warm.num_faces() {
            let base = warm.derived_base_res(f).unwrap();
            warm.get_data_at_res(f, base).unwrap();
        }
        chain(&warm)
    };
    assert_eq!(chain(&tx), expected);
    let s = tx.cache_stats();
    assert_eq!(s.entries, s.derived_blocks, "a base or its source was kept");
}

#[test]
fn derived_bytes_are_inside_the_budget() {
    for name in FIXTURES {
        let tx = open(name, 4, 64 << 20);
        let blocks = chain(&tx);
        let s = tx.cache_stats();
        assert_eq!(s.derived_blocks, blocks.len(), "{name}");
        let payload: usize = blocks.iter().map(Vec::len).sum();
        assert!(s.derived_bytes > payload, "{name}: overhead is charged");
        assert!(s.derived_bytes <= s.bytes_resident, "{name}");
        assert!(s.derived_blocks <= s.entries, "{name}");

        // Shrinking the budget evicts derived blocks like any other, and
        // their bytes leave the account with them.
        tx.set_cache_budget(s.bytes_resident / 3);
        let after = tx.cache_stats();
        assert!(after.bytes_resident <= after.bytes_budget, "{name}");
        assert!(after.derived_bytes <= after.bytes_resident, "{name}");
        assert!(after.derived_blocks < s.derived_blocks, "{name}");
        tx.set_cache_budget(0);
        let empty = tx.cache_stats();
        assert_eq!(
            (empty.derived_blocks, empty.derived_bytes),
            (0, 0),
            "{name}"
        );
    }
}

/// A file stream that records every read's offset.
struct Recording {
    file: std::fs::File,
    pos: u64,
    reads: Arc<Mutex<Vec<u64>>>,
}

impl Read for Recording {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reads.lock().unwrap().push(self.pos);
        let n = self.file.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Recording {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.pos = self.file.seek(to)?;
        Ok(self.pos)
    }
}

#[test]
fn a_derived_level_never_reads_finer_than_its_base() {
    // The tiled fixture's large face is 1024x512 with stored levels, so a
    // base of 32x16 (cap 5, aspect kept) is a stored level well below the
    // tiled one. A base the file does *not* store (32x32 here) is reduced by
    // the reader from a finer stored level, as `get_data_at_res` always is;
    // the guarantee is about the derived levels, which read only the base.
    let reads = Arc::new(Mutex::new(Vec::new()));
    let io = Recording {
        file: std::fs::File::open(fixture("quad_tiled")).unwrap(),
        pos: 0,
        reads: reads.clone(),
    };
    let tx = SharedReader::new(io).unwrap();
    let pixel_size = tx.pixel_size();
    let tx = tx
        .with_derived(Arc::new(Widen {
            cap: 5,
            pixel_size,
            stored: true,
        }))
        .unwrap();

    // Every block the file holds finer than each face's base, by offset.
    let mut finer = Vec::new();
    for f in 0..tx.num_faces() {
        let base = tx.derived_base_res(f).unwrap();
        for level in 0..tx.face_num_levels(f).unwrap() {
            let res = tx.res_for_level(f, level).unwrap();
            if res.ulog2 <= base.ulog2 && res.vlog2 <= base.vlog2 {
                continue;
            }
            let layout = tx.tile_layout(f, res).unwrap();
            for tile in 0..layout.ntiles() {
                if let Some(pos) = tx.tile_info(f, res, tile).unwrap().file_offset {
                    finer.push(pos);
                }
            }
        }
    }
    assert!(!finer.is_empty(), "the fixture has levels above the base");

    reads.lock().unwrap().clear();
    let blocks = chain(&tx);
    assert!(!blocks.is_empty());
    let reads = reads.lock().unwrap();
    assert!(!reads.is_empty(), "the base was read from the file");
    for pos in reads.iter() {
        assert!(
            !finer.contains(pos),
            "read a block finer than the base at {pos}"
        );
    }
}

#[test]
fn concurrent_get_derived_agrees_with_one_thread() {
    for name in FIXTURES {
        let expected = chain(&open(name, 4, 64 << 20));
        // A budget that evicts under contention, so threads race on misses,
        // on inserts and on re-derives at once.
        let tx = open(name, 4, 16 << 10);
        std::thread::scope(|s| {
            for t in 0..8 {
                let tx = tx.clone();
                let expected = &expected;
                s.spawn(move || {
                    for round in 0..4 {
                        let mut i = 0;
                        for f in 0..tx.num_faces() {
                            let n = tx.derived_levels(f).unwrap();
                            // Walk the chain coarse-first on odd threads, so
                            // a level is often asked for before its parent.
                            let order: Vec<usize> = if (t + round) % 2 == 0 {
                                (0..n).collect()
                            } else {
                                (0..n).rev().collect()
                            };
                            for &k in &order {
                                let got = tx.get_derived(f, k).unwrap();
                                assert_eq!(&*got, &expected[i + k][..], "{name} {f}/{k}");
                            }
                            i += n;
                        }
                    }
                });
            }
        });
        let s = tx.cache_stats();
        assert!(s.bytes_resident <= s.bytes_budget, "{name}");
    }
}
