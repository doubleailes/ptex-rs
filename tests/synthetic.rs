//! Tests over hand-built `.ptx` files.
//!
//! The reference C++ writer never emits a whole-face block with the constant
//! encoding *and* real pixel bytes: `writeConstantFace` puts the value in the
//! file's constant-data block, flags the face constant, and leaves a level
//! entry of zero length. So every fixture written by it takes the
//! constant-data path, and the stored-constant-block branch of the readers is
//! unreachable from the fixture set. Another writer may still produce it, so
//! these files exercise it directly.

use std::io::Write;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use ptex::{Error, PtexReader, Res};

/// Single uint8 channel, so one pixel is one byte.
const PIXEL_SIZE: usize = 1;

fn zlib(data: &[u8]) -> Vec<u8> {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
    e.write_all(data).unwrap();
    e.finish().unwrap()
}

fn put_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// A one-face 2x2 uint8 file whose face is **not** flagged constant but whose
/// level-0 block uses the constant encoding.
///
/// `blocksize` is written verbatim into the face data header, so a caller can
/// build both the well-formed shape (`blocksize == PIXEL_SIZE`, one pixel of
/// data present) and the malformed zero-length shape. `block_pixel` and
/// `const_pixel` differ so a test can tell which one a read used.
///
/// Returns the file bytes and the absolute file offset of the level-0 block.
fn constant_block_ptx(blocksize: u32, block_pixel: u8, const_pixel: u8) -> (Vec<u8>, u64) {
    // face info: res (1,1) == 2x2, no adjacency, flags 0 (NOT constant)
    let mut fi = vec![1u8, 1, 0, 0];
    for _ in 0..4 {
        fi.extend_from_slice(&(-1i32).to_le_bytes());
    }
    assert_eq!(fi.len(), 20);
    let faceinfo = zlib(&fi);
    let constdata = zlib(&[const_pixel]);

    // one face data header: encoding bits 30..31 == 0 == constant
    let levelheader = zlib(&blocksize.to_le_bytes());
    let mut leveldata = levelheader.clone();
    leveldata.extend(std::iter::repeat(block_pixel).take(blocksize as usize));

    let mut levelinfo = Vec::new();
    levelinfo.extend_from_slice(&(leveldata.len() as u64).to_le_bytes());
    levelinfo.extend_from_slice(&(levelheader.len() as u32).to_le_bytes());
    levelinfo.extend_from_slice(&1u32.to_le_bytes());
    assert_eq!(levelinfo.len(), 16);

    let mut header = vec![0u8; 64];
    header[0..4].copy_from_slice(b"Ptex");
    put_u32(&mut header, 4, 1); // version
    put_u32(&mut header, 8, 1); // meshtype: quad
    put_u32(&mut header, 12, 0); // datatype: uint8
    put_u32(&mut header, 16, (-1i32) as u32); // alphachan
    header[20..22].copy_from_slice(&1u16.to_le_bytes()); // nchannels
    header[22..24].copy_from_slice(&1u16.to_le_bytes()); // nlevels
    put_u32(&mut header, 24, 1); // nfaces
    put_u32(&mut header, 28, 0); // extheadersize
    put_u32(&mut header, 32, faceinfo.len() as u32);
    put_u32(&mut header, 36, constdata.len() as u32);
    put_u32(&mut header, 40, levelinfo.len() as u32);
    put_u32(&mut header, 44, 0); // minorversion
    header[48..56].copy_from_slice(&(leveldata.len() as u64).to_le_bytes());
    put_u32(&mut header, 56, 0); // metadatazipsize
    put_u32(&mut header, 60, 0); // metadatamemsize

    let block_pos =
        (header.len() + faceinfo.len() + constdata.len() + levelinfo.len() + levelheader.len())
            as u64;

    let mut file = header;
    file.extend_from_slice(&faceinfo);
    file.extend_from_slice(&constdata);
    file.extend_from_slice(&levelinfo);
    file.extend_from_slice(&leveldata);
    (file, block_pos)
}

fn open(bytes: &[u8]) -> PtexReader<std::io::Cursor<Vec<u8>>> {
    PtexReader::new(std::io::Cursor::new(bytes.to_vec())).unwrap()
}

/// A stored block that merely uses the constant encoding has a header and a
/// file position of its own, and `tile_info` must report them - it is only a
/// constant *face*, whose value lives in the constant-data block, that has no
/// block to describe.
#[test]
fn stored_constant_block_reports_its_header_and_offset() {
    let (bytes, block_pos) = constant_block_ptx(PIXEL_SIZE as u32, 0x7f, 0x2a);
    let mut tx = open(&bytes);
    let res = tx.face_info(0).unwrap().res;
    assert_eq!(res, Res::new(1, 1));
    assert!(!tx.face_info(0).unwrap().is_constant());

    let layout = tx.tile_layout(0, res).unwrap();
    assert!(layout.is_constant);
    assert!(layout.is_stored);
    assert!(!layout.is_tiled);
    assert_eq!(layout.ntiles(), 1);

    let info = tx.tile_info(0, res, 0).unwrap();
    assert!(info.is_constant);
    assert_eq!(info.compressed_size, PIXEL_SIZE as u64);
    assert_eq!(info.file_offset, Some(block_pos));
}

/// The pixels must come from the block, not from the file's constant-data
/// block, which holds a different value here.
#[test]
fn stored_constant_block_pixels_come_from_the_block() {
    let (bytes, _) = constant_block_ptx(PIXEL_SIZE as u32, 0x7f, 0x2a);
    let mut tx = open(&bytes);
    let res = tx.face_info(0).unwrap().res;
    assert_eq!(tx.constant_data(0).unwrap(), &[0x2a]);
    assert_eq!(tx.get_data(0).unwrap(), vec![0x7f; 4]);
    assert_eq!(tx.get_tile(0, res, 0).unwrap(), vec![0x7f; 4]);
}

/// A constant block carries one pixel. A level entry claiming zero bytes has
/// the *next* block's offset, so reading it would yield that block's bytes;
/// it must be reported as corrupt instead.
#[test]
fn zero_length_constant_block_is_rejected() {
    let (bytes, _) = constant_block_ptx(0, 0x7f, 0x2a);
    let mut tx = open(&bytes);
    let res = tx.face_info(0).unwrap().res;

    // the layout and its metadata are still describable
    let info = tx.tile_info(0, res, 0).unwrap();
    assert!(info.is_constant);
    assert_eq!(info.compressed_size, 0);

    match tx.get_data(0) {
        Err(Error::Corrupt(msg)) => assert!(msg.contains("constant block"), "{msg}"),
        other => panic!("expected Corrupt, got {other:?}"),
    }
}

#[cfg(feature = "cache")]
mod shared {
    use super::*;
    use ptex::SharedReader;

    fn open_shared(bytes: &[u8]) -> SharedReader<std::io::Cursor<Vec<u8>>> {
        SharedReader::new(std::io::Cursor::new(bytes.to_vec())).unwrap()
    }

    #[test]
    fn shared_reader_agrees_on_stored_constant_blocks() {
        let (bytes, block_pos) = constant_block_ptx(PIXEL_SIZE as u32, 0x7f, 0x2a);
        let tx = open_shared(&bytes);
        let res = tx.face_info(0).unwrap().res;

        let info = tx.tile_info(0, res, 0).unwrap();
        assert!(info.is_constant);
        assert_eq!(info.compressed_size, PIXEL_SIZE as u64);
        assert_eq!(info.file_offset, Some(block_pos));

        assert_eq!(tx.constant_data(0).unwrap(), &[0x2a]);
        assert_eq!(tx.get_data(0).unwrap().as_slice(), &[0x7f; 4]);
        assert_eq!(tx.get_tile(0, res, 0).unwrap().as_slice(), &[0x7f; 4]);

        // and matches the single-threaded reader exactly
        let mut plain = super::open(&bytes);
        assert_eq!(info, plain.tile_info(0, res, 0).unwrap());
        assert_eq!(
            tx.tile_layout(0, res).unwrap(),
            plain.tile_layout(0, res).unwrap()
        );
    }

    #[test]
    fn shared_reader_rejects_a_zero_length_constant_block() {
        let (bytes, _) = constant_block_ptx(0, 0x7f, 0x2a);
        let tx = open_shared(&bytes);
        match tx.get_data(0) {
            Err(Error::Corrupt(msg)) => assert!(msg.contains("constant block"), "{msg}"),
            other => panic!("expected Corrupt, got {:?}", other.map(|_| ())),
        }
    }
}

/// A minimal one-face file used to exercise edit-block detection.
///
/// With `ext_header`, a full 40-byte extended header is written declaring
/// `editdatasize` and an `editdatapos` pointing just past the main data —
/// the shape modern files have. Without it, `extheadersize` is 0, so both
/// fields read as zero and a reader must fall back to looking for trailing
/// bytes, which is how the C++ reader handles older files. `trailing` bytes
/// are appended after the main data either way.
fn ptx_with_edits(ext_header: bool, editdatasize: u64, trailing: usize) -> Vec<u8> {
    let mut fi = vec![1u8, 1, 0, 0];
    for _ in 0..4 {
        fi.extend_from_slice(&(-1i32).to_le_bytes());
    }
    let faceinfo = zlib(&fi);
    let constdata = zlib(&[0x11]);
    let levelheader = zlib(&(PIXEL_SIZE as u32).to_le_bytes());
    let mut leveldata = levelheader.clone();
    leveldata.push(0x22);
    let mut levelinfo = Vec::new();
    levelinfo.extend_from_slice(&(leveldata.len() as u64).to_le_bytes());
    levelinfo.extend_from_slice(&(levelheader.len() as u32).to_le_bytes());
    levelinfo.extend_from_slice(&1u32.to_le_bytes());

    let ehs = if ext_header { 40usize } else { 0 };
    // 8 = the compatibility barrier that follows the meta data block
    let end_of_main =
        (64 + ehs + faceinfo.len() + constdata.len() + levelinfo.len() + leveldata.len() + 8)
            as u64;

    let mut header = vec![0u8; 64];
    header[0..4].copy_from_slice(b"Ptex");
    put_u32(&mut header, 4, 1);
    put_u32(&mut header, 8, 1);
    put_u32(&mut header, 12, 0);
    put_u32(&mut header, 16, (-1i32) as u32);
    header[20..22].copy_from_slice(&1u16.to_le_bytes());
    header[22..24].copy_from_slice(&1u16.to_le_bytes());
    put_u32(&mut header, 24, 1);
    put_u32(&mut header, 28, ehs as u32);
    put_u32(&mut header, 32, faceinfo.len() as u32);
    put_u32(&mut header, 36, constdata.len() as u32);
    put_u32(&mut header, 40, levelinfo.len() as u32);
    header[48..56].copy_from_slice(&(leveldata.len() as u64).to_le_bytes());

    let mut file = header;
    if ext_header {
        let mut e = vec![0u8; 40];
        e[24..32].copy_from_slice(&editdatasize.to_le_bytes());
        e[32..40].copy_from_slice(&end_of_main.to_le_bytes());
        file.extend_from_slice(&e);
    }
    file.extend_from_slice(&faceinfo);
    file.extend_from_slice(&constdata);
    file.extend_from_slice(&levelinfo);
    file.extend_from_slice(&leveldata);
    file.extend_from_slice(&[0xeeu8; 8]); // compatibility barrier
    assert_eq!(file.len() as u64, end_of_main);
    file.extend(std::iter::repeat(0xedu8).take(trailing));
    file
}

/// A modern file records the edit extent in its extended header. Note that
/// `editdatapos` is non-zero even when there are no edits, so the size is
/// what decides.
#[test]
fn declared_edit_data_is_detected() {
    let none = ptx_with_edits(true, 0, 0);
    assert!(!open(&none).has_edits());

    let some = ptx_with_edits(true, 16, 16);
    assert!(open(&some).has_edits());
}

/// An older file has neither field, so edits can only be found as bytes
/// trailing the main data — the fallback the C++ reader uses.
#[test]
fn trailing_data_in_an_older_file_is_detected() {
    let clean = ptx_with_edits(false, 0, 0);
    assert!(!open(&clean).has_edits());

    let edited = ptx_with_edits(false, 0, 24);
    assert!(open(&edited).has_edits());

    // A record whose size field is zero is where the reference reader stops,
    // so trailing padding must not be mistaken for an edit.
    let mut padded = ptx_with_edits(false, 0, 0);
    padded.extend_from_slice(&[0u8; 16]);
    assert!(!open(&padded).has_edits());

    // Too few bytes to even hold a record preamble.
    let mut stub = ptx_with_edits(false, 0, 0);
    stub.extend_from_slice(&[0xed; 3]);
    assert!(!open(&stub).has_edits());
}

#[test]
fn face_flags_has_edits_reads_the_right_bit() {
    use ptex::{face_flags, FaceInfo, Res};
    let mut fi = FaceInfo {
        res: Res::new(1, 1),
        ..FaceInfo::default()
    };
    assert!(!fi.has_edits());
    fi.flags = face_flags::HAS_EDITS;
    assert!(fi.has_edits());
    assert!(!fi.is_constant());
    // the C++ bit layout: constant=1, hasedits=2, nbconstant=4, subface=8
    assert_eq!(face_flags::HAS_EDITS, 2);
}

/// None of the fixtures are edited, and both readers must agree.
#[test]
fn fixtures_report_no_edits() {
    for name in ["quad_u8", "quad_f32", "quad_f16", "tri_u16", "quad_tiled"] {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("{name}.ptx"));
        let tx = PtexReader::open(&path).unwrap();
        assert!(!tx.has_edits(), "{name}");
        for f in 0..tx.num_faces() {
            assert!(!tx.face_info(f).unwrap().has_edits(), "{name} face {f}");
        }
        #[cfg(feature = "cache")]
        assert!(
            !ptex::SharedReader::open(&path).unwrap().has_edits(),
            "{name}"
        );
    }
}
