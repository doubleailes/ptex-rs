//! Error type for the ptex crate.

use std::fmt;

/// Errors that can occur while reading a Ptex file.
#[derive(Debug)]
pub enum Error {
    /// An I/O error occurred while reading the file.
    Io(std::io::Error),
    /// The file does not start with the Ptex magic number.
    NotAPtexFile,
    /// The file version is not supported (only version 1 is defined).
    UnsupportedVersion(u32),
    /// The header contains an invalid mesh type value.
    InvalidMeshType(u32),
    /// The header contains an invalid data type value.
    InvalidDataType(u32),
    /// The file contents are inconsistent or corrupt.
    Corrupt(String),
    /// A face id is out of range.
    FaceOutOfRange {
        /// The requested face id.
        faceid: i32,
        /// The number of faces in the file.
        nfaces: u32,
    },
    /// The requested operation is not supported.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::NotAPtexFile => write!(f, "not a ptex file"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported ptex file version ({v})"),
            Error::InvalidMeshType(v) => write!(f, "invalid mesh type ({v})"),
            Error::InvalidDataType(v) => write!(f, "invalid data type ({v})"),
            Error::Corrupt(msg) => write!(f, "corrupt ptex file: {msg}"),
            Error::FaceOutOfRange { faceid, nfaces } => {
                write!(f, "face id {faceid} out of range (file has {nfaces} faces)")
            }
            Error::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
