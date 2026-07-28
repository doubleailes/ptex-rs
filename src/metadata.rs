//! Meta data access.

use crate::error::{Error, Result};
use crate::format;
use crate::types::MetaDataType;

/// A single meta data entry.
#[derive(Debug, Clone)]
pub struct MetaDataEntry {
    key: String,
    data_type: MetaDataType,
    data: Vec<u8>,
}

impl MetaDataEntry {
    /// The entry's key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The entry's value type.
    pub fn data_type(&self) -> MetaDataType {
        self.data_type
    }

    /// Raw little-endian value bytes.
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

    /// Number of values in the entry (string entries report their byte
    /// length including the terminating NUL, matching the C++ API).
    pub fn len(&self) -> usize {
        let esize = match self.data_type {
            MetaDataType::String | MetaDataType::Int8 => 1,
            MetaDataType::Int16 => 2,
            MetaDataType::Int32 | MetaDataType::Float => 4,
            MetaDataType::Double => 8,
        };
        self.data.len() / esize
    }

    /// True if the entry holds no data.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Value as a string (if the entry is of string type).
    pub fn as_str(&self) -> Option<&str> {
        if self.data_type != MetaDataType::String {
            return None;
        }
        let bytes = match self.data.split_last() {
            Some((0, head)) => head, // strip terminating NUL
            _ => &self.data[..],
        };
        std::str::from_utf8(bytes).ok()
    }

    /// Value as an array of i8.
    pub fn as_i8(&self) -> Option<Vec<i8>> {
        (self.data_type == MetaDataType::Int8).then(|| self.data.iter().map(|&b| b as i8).collect())
    }

    /// Value as an array of i16.
    pub fn as_i16(&self) -> Option<Vec<i16>> {
        (self.data_type == MetaDataType::Int16).then(|| {
            self.data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect()
        })
    }

    /// Value as an array of i32.
    pub fn as_i32(&self) -> Option<Vec<i32>> {
        (self.data_type == MetaDataType::Int32).then(|| {
            self.data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
    }

    /// Value as an array of f32.
    pub fn as_f32(&self) -> Option<Vec<f32>> {
        (self.data_type == MetaDataType::Float).then(|| {
            self.data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
    }

    /// Value as an array of f64.
    pub fn as_f64(&self) -> Option<Vec<f64>> {
        (self.data_type == MetaDataType::Double).then(|| {
            self.data
                .chunks_exact(8)
                .map(|c| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(c);
                    f64::from_le_bytes(a)
                })
                .collect()
        })
    }
}

/// Meta data stored in a Ptex file.
///
/// Meta data entries are key/value pairs; values are arrays of a single
/// [`MetaDataType`].
#[derive(Debug, Default, Clone)]
pub struct MetaData {
    entries: Vec<MetaDataEntry>,
}

impl MetaData {
    /// Number of meta data entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Access an entry by index.
    pub fn entry(&self, index: usize) -> Option<&MetaDataEntry> {
        self.entries.get(index)
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &MetaDataEntry> {
        self.entries.iter()
    }

    /// All keys in the file, in file order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.key())
    }

    /// Look up an entry by key.
    pub fn find(&self, key: &str) -> Option<&MetaDataEntry> {
        self.entries.iter().find(|e| e.key == key)
    }

    /// Look up a string value by key.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.find(key).and_then(|e| e.as_str())
    }

    /// Look up an i8 array value by key.
    pub fn get_i8(&self, key: &str) -> Option<Vec<i8>> {
        self.find(key).and_then(|e| e.as_i8())
    }

    /// Look up an i16 array value by key.
    pub fn get_i16(&self, key: &str) -> Option<Vec<i16>> {
        self.find(key).and_then(|e| e.as_i16())
    }

    /// Look up an i32 array value by key.
    pub fn get_i32(&self, key: &str) -> Option<Vec<i32>> {
        self.find(key).and_then(|e| e.as_i32())
    }

    /// Look up an f32 array value by key.
    pub fn get_f32(&self, key: &str) -> Option<Vec<f32>> {
        self.find(key).and_then(|e| e.as_f32())
    }

    /// Look up an f64 array value by key.
    pub fn get_f64(&self, key: &str) -> Option<Vec<f64>> {
        self.find(key).and_then(|e| e.as_f64())
    }

    /// Parse a decompressed primary meta data block and append its entries.
    pub(crate) fn parse_block(&mut self, buf: &[u8]) -> Result<()> {
        let mut ptr = 0usize;
        while ptr < buf.len() {
            let (key, data_type, datasize) = parse_entry_header(buf, &mut ptr)?;
            if ptr + datasize > buf.len() {
                return Err(Error::Corrupt("meta data entry overruns block".into()));
            }
            let data = buf[ptr..ptr + datasize].to_vec();
            ptr += datasize;
            self.entries.push(MetaDataEntry {
                key,
                data_type,
                data,
            });
        }
        Ok(())
    }

    pub(crate) fn add_entry(&mut self, key: String, data_type: MetaDataType, data: Vec<u8>) {
        self.entries.push(MetaDataEntry {
            key,
            data_type,
            data,
        });
    }
}

/// Parse the shared `[keysize][key][datatype][datasize]` prefix of a meta
/// data entry, advancing `ptr`.
pub(crate) fn parse_entry_header(
    buf: &[u8],
    ptr: &mut usize,
) -> Result<(String, MetaDataType, usize)> {
    let err = || Error::Corrupt("truncated meta data block".into());
    let keysize = *buf.get(*ptr).ok_or_else(err)? as usize;
    *ptr += 1;
    if keysize == 0 || *ptr + keysize > buf.len() {
        return Err(err());
    }
    // key is stored with a terminating NUL included in keysize
    let key_bytes = &buf[*ptr..*ptr + keysize - 1];
    let key = String::from_utf8_lossy(key_bytes).into_owned();
    *ptr += keysize;
    let dt = *buf.get(*ptr).ok_or_else(err)?;
    *ptr += 1;
    let datasize = format::checked_u32_at(buf, *ptr)? as usize;
    *ptr += 4;
    Ok((key, MetaDataType::from_u8(dt)?, datasize))
}
