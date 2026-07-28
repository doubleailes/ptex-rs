//! Common data structures and enums used throughout the API.
//!
//! These mirror the types defined in `Ptexture.h` of the original C++
//! library.

use crate::error::{Error, Result};

/// Type of base mesh for which the textures are defined.  A mesh
/// can be triangle-based (with triangular textures) or quad-based
/// (with rectangular textures).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshType {
    /// Mesh is triangle-based.
    Triangle,
    /// Mesh is quad-based.
    Quad,
}

impl MeshType {
    pub(crate) fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(MeshType::Triangle),
            1 => Ok(MeshType::Quad),
            _ => Err(Error::InvalidMeshType(v)),
        }
    }

    /// Look up the name of the mesh type ("triangle" or "quad").
    pub fn name(self) -> &'static str {
        match self {
            MeshType::Triangle => "triangle",
            MeshType::Quad => "quad",
        }
    }
}

/// Type of data stored in texture file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// Unsigned, 8-bit integer.
    UInt8,
    /// Unsigned, 16-bit integer.
    UInt16,
    /// Half-precision (16-bit) floating point.
    Half,
    /// Single-precision (32-bit) floating point.
    Float,
}

impl DataType {
    pub(crate) fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(DataType::UInt8),
            1 => Ok(DataType::UInt16),
            2 => Ok(DataType::Half),
            3 => Ok(DataType::Float),
            _ => Err(Error::InvalidDataType(v)),
        }
    }

    /// Size of the data type in bytes.
    pub fn size(self) -> usize {
        match self {
            DataType::UInt8 => 1,
            DataType::UInt16 => 2,
            DataType::Half => 2,
            DataType::Float => 4,
        }
    }

    /// Value of this data type that corresponds to the normalized value of 1.0.
    pub fn one_value(self) -> f32 {
        match self {
            DataType::UInt8 => 255.0,
            DataType::UInt16 => 65535.0,
            DataType::Half | DataType::Float => 1.0,
        }
    }

    /// Inverse of [`DataType::one_value`].
    pub fn one_value_inv(self) -> f32 {
        match self {
            DataType::UInt8 => 1.0 / 255.0,
            DataType::UInt16 => 1.0 / 65535.0,
            DataType::Half | DataType::Float => 1.0,
        }
    }

    /// Look up the name of the data type ("uint8", "uint16", "float16" or
    /// "float32"), matching `Ptex::DataTypeName`.
    pub fn name(self) -> &'static str {
        match self {
            DataType::UInt8 => "uint8",
            DataType::UInt16 => "uint16",
            DataType::Half => "float16",
            DataType::Float => "float32",
        }
    }
}

/// How to handle transformation across edges when filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeFilterMode {
    /// Don't do anything with the values.
    None,
    /// Values are vectors in tangent space; rotate values.
    TangentVector,
}

impl EdgeFilterMode {
    pub(crate) fn from_u16(v: u16) -> Self {
        match v {
            1 => EdgeFilterMode::TangentVector,
            _ => EdgeFilterMode::None,
        }
    }

    /// Look up the name of the edge filter mode.
    pub fn name(self) -> &'static str {
        match self {
            EdgeFilterMode::None => "none",
            EdgeFilterMode::TangentVector => "tanvec",
        }
    }
}

/// How to handle mesh border when filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorderMode {
    /// Texel access is clamped to border.
    Clamp,
    /// Texels beyond border are assumed to be black.
    Black,
    /// Texel access wraps to other side of face.
    Periodic,
}

impl BorderMode {
    pub(crate) fn from_u16(v: u16) -> Self {
        match v {
            1 => BorderMode::Black,
            2 => BorderMode::Periodic,
            _ => BorderMode::Clamp,
        }
    }

    /// Look up the name of the border mode.
    pub fn name(self) -> &'static str {
        match self {
            BorderMode::Clamp => "clamp",
            BorderMode::Black => "black",
            BorderMode::Periodic => "periodic",
        }
    }
}

/// Edge IDs used in adjacency data in the [`FaceInfo`] struct.
/// Edge ID usage for triangle meshes is TBD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeId {
    /// Bottom edge, from UV (0,0) to (1,0).
    Bottom = 0,
    /// Right edge, from UV (1,0) to (1,1).
    Right = 1,
    /// Top edge, from UV (1,1) to (0,1).
    Top = 2,
    /// Left edge, from UV (0,1) to (0,0).
    Left = 3,
}

impl EdgeId {
    pub(crate) fn from_u8(v: u8) -> Self {
        match v & 3 {
            0 => EdgeId::Bottom,
            1 => EdgeId::Right,
            2 => EdgeId::Top,
            _ => EdgeId::Left,
        }
    }

    /// Look up the name of the edge id.
    pub fn name(self) -> &'static str {
        match self {
            EdgeId::Bottom => "bottom",
            EdgeId::Right => "right",
            EdgeId::Top => "top",
            EdgeId::Left => "left",
        }
    }
}

/// Type of a meta data entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetaDataType {
    /// Null-terminated string.
    String,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Single-precision (32-bit) floating point.
    Float,
    /// Double-precision (64-bit) floating point.
    Double,
}

impl MetaDataType {
    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(MetaDataType::String),
            1 => Ok(MetaDataType::Int8),
            2 => Ok(MetaDataType::Int16),
            3 => Ok(MetaDataType::Int32),
            4 => Ok(MetaDataType::Float),
            5 => Ok(MetaDataType::Double),
            _ => Err(Error::Corrupt(format!("invalid meta data type ({v})"))),
        }
    }

    /// Look up the name of the meta data type.
    pub fn name(self) -> &'static str {
        match self {
            MetaDataType::String => "string",
            MetaDataType::Int8 => "int8",
            MetaDataType::Int16 => "int16",
            MetaDataType::Int32 => "int32",
            MetaDataType::Float => "float",
            MetaDataType::Double => "double",
        }
    }
}

/// Pixel resolution of a given texture.
///
/// The resolution is stored in log form: `ulog2 = log2(ures)`,
/// `vlog2 = log2(vres)`.
/// Note: negative `ulog2` or `vlog2` values are reserved for internal use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Res {
    /// log base 2 of u resolution, in texels.
    pub ulog2: i8,
    /// log base 2 of v resolution, in texels.
    pub vlog2: i8,
}

impl Res {
    /// Create a resolution from log2 u and v sizes.
    pub fn new(ulog2: i8, vlog2: i8) -> Self {
        Res { ulog2, vlog2 }
    }

    /// Create a resolution from a packed 16-bit value (as stored on disk).
    pub fn from_val(value: u16) -> Self {
        Res {
            ulog2: (value & 0xff) as i8,
            vlog2: ((value >> 8) & 0xff) as i8,
        }
    }

    /// U resolution in texels.
    pub fn u(self) -> usize {
        1usize << (self.ulog2 as u32 & 31)
    }

    /// V resolution in texels.
    pub fn v(self) -> usize {
        1usize << (self.vlog2 as u32 & 31)
    }

    /// Resolution as a single 16-bit integer value.
    pub fn val(self) -> u16 {
        (self.ulog2 as u8 as u16) | ((self.vlog2 as u8 as u16) << 8)
    }

    /// Total size of the texture in texels (u * v).
    pub fn size(self) -> usize {
        self.u() * self.v()
    }

    /// Get value of resolution with u and v swapped.
    pub fn swappeduv(self) -> Res {
        Res::new(self.vlog2, self.ulog2)
    }

    /// Clamp the resolution value against the given value.
    pub fn clamp(&mut self, r: Res) {
        if self.ulog2 > r.ulog2 {
            self.ulog2 = r.ulog2;
        }
        if self.vlog2 > r.vlog2 {
            self.vlog2 = r.vlog2;
        }
    }

    /// Determine the number of tiles in the u direction for the given tile res.
    pub fn ntilesu(self, tileres: Res) -> usize {
        1usize << (self.ulog2 - tileres.ulog2)
    }

    /// Determine the number of tiles in the v direction for the given tile res.
    pub fn ntilesv(self, tileres: Res) -> usize {
        1usize << (self.vlog2 - tileres.vlog2)
    }

    /// Determine the total number of tiles for the given tile res.
    pub fn ntiles(self, tileres: Res) -> usize {
        self.ntilesu(tileres) * self.ntilesv(tileres)
    }

    /// True if res is 1x1 texel.
    pub fn is_one(self) -> bool {
        self.ulog2 == 0 && self.vlog2 == 0
    }
}

/// Flag bit values used in [`FaceInfo::flags`].
pub mod face_flags {
    /// Face is constant (a single color).
    pub const CONSTANT: u8 = 1;
    /// Obsolete flag (formerly "has edits").
    pub const OBSOLETE: u8 = 2;
    /// The face and all its neighbors are constant with the same color.
    pub const NEIGHBORHOOD_CONSTANT: u8 = 4;
    /// The face is a subface (for non-quad faces stored as quad subfaces).
    pub const SUBFACE: u8 = 8;
}

/// Information about a face, as stored in the Ptex file header.
///
/// The FaceInfo data contains the face resolution and neighboring face
/// adjacency information as well as a set of flags describing the face.
///
/// The `adjfaces` data member contains the face ids of the four neighboring
/// faces.  The neighbors are accessed in [`EdgeId`] order, CCW, starting with
/// the bottom edge.  The `adjedges` data member contains the corresponding
/// edge id for each neighboring face.
///
/// If a face has no neighbor for a given edge, the adjface id should be -1,
/// and the adjedge id doesn't matter (but is typically zero).
///
/// If an adjacent face is a pair of subfaces, the id of the first subface as
/// encountered in a CCW traversal should be stored as the adjface id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceInfo {
    /// Resolution of face.
    pub res: Res,
    /// Adjacent edges, 2 bits per edge.
    pub adjedges: u8,
    /// Flags (see [`face_flags`]).
    pub flags: u8,
    /// Adjacent faces (-1 == no adjacent face).
    pub adjfaces: [i32; 4],
}

impl Default for FaceInfo {
    fn default() -> Self {
        FaceInfo {
            res: Res::default(),
            adjedges: 0,
            flags: 0,
            adjfaces: [-1; 4],
        }
    }
}

impl FaceInfo {
    /// Access an adjacent edge id.  The `eid` value must be 0..3.
    pub fn adjedge(&self, eid: usize) -> EdgeId {
        EdgeId::from_u8((self.adjedges >> (2 * eid)) & 3)
    }

    /// Access an adjacent face id.  The `eid` value must be 0..3.
    pub fn adjface(&self, eid: usize) -> i32 {
        self.adjfaces[eid]
    }

    /// Determine if the face is constant (by checking a flag).
    pub fn is_constant(&self) -> bool {
        self.flags & face_flags::CONSTANT != 0
    }

    /// Determine if the neighborhood of the face is constant (by checking a flag).
    pub fn is_neighborhood_constant(&self) -> bool {
        self.flags & face_flags::NEIGHBORHOOD_CONSTANT != 0
    }

    /// Determine if the face is a subface (by checking a flag).
    pub fn is_subface(&self) -> bool {
        self.flags & face_flags::SUBFACE != 0
    }
}

/// Convert a half-precision (16-bit) float to a 32-bit float.
pub fn half_to_float(h: u16) -> f32 {
    let sign = (h as u32) >> 15;
    let exp = ((h as u32) >> 10) & 0x1f;
    let mant = (h as u32) & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            // signed zero
            sign << 31
        } else {
            // subnormal: normalize it
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        // inf/nan
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Convert a 32-bit float to a half-precision (16-bit) float.
///
/// This matches `PtexHalf::fromFloat` bit-for-bit: rounding is
/// round-half-up, overflow converts to infinity, and both +0.0 and -0.0
/// convert to +0.
pub fn float_to_half(f: f32) -> u16 {
    if f == 0.0 {
        return 0;
    }
    let i = f.to_bits();
    let s = (i >> 16) & 0x8000;
    let e = ((i >> 23) & 0xff) as i32 - 112; // biased half exponent
    if (1..=30).contains(&e) {
        // normal case: round the mantissa (half-up); a carry out of the
        // mantissa correctly bumps the exponent
        (s + ((e as u32) << 10) + (((i & 0x7f_ffff) + 0x1000) >> 13)) as u16
    } else {
        float_to_half_except(i)
    }
}

/// Handle exceptional cases for float-to-half conversion
/// (matches `PtexHalf::fromFloat_except`).
fn float_to_half_except(i: u32) -> u16 {
    let s = ((i >> 16) & 0x8000) as u16;
    let e = ((i >> 13) & 0x3fc00) as i32 - 0x1c000;
    if e <= 0 {
        // denormalized: half subnormal unit is 2^-24
        let f = f32::from_bits(i);
        s | (f.abs() * 1.677_721_6e7 + 0.5) as u16
    } else if e == 0x23c00 {
        // inf/nan, preserve msb bits of the mantissa for the nan code
        s | 0x7c00 | ((i & 0x7f_ffff) >> 13) as u16
    } else {
        // overflow - convert to inf
        s | 0x7c00
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn res_basics() {
        let r = Res::new(3, 2);
        assert_eq!(r.u(), 8);
        assert_eq!(r.v(), 4);
        assert_eq!(r.size(), 32);
        assert_eq!(Res::from_val(r.val()), r);
        assert_eq!(r.swappeduv(), Res::new(2, 3));
        assert_eq!(Res::new(4, 3).ntiles(Res::new(2, 2)), 8);
    }

    #[test]
    fn half_roundtrip() {
        for &f in &[0.0f32, 1.0, -1.0, 0.5, 2.0, 65504.0, 6.1e-5, 3.375] {
            let h = float_to_half(f);
            let g = half_to_float(h);
            assert!((f - g).abs() <= f.abs() * 1e-3 + 1e-7, "{f} -> {g}");
        }
        assert_eq!(half_to_float(float_to_half(f32::INFINITY)), f32::INFINITY);
        assert!(half_to_float(float_to_half(f32::NAN)).is_nan());
        // exact values
        assert_eq!(half_to_float(0x3c00), 1.0);
        assert_eq!(half_to_float(0xc000), -2.0);
        assert_eq!(float_to_half(1.0), 0x3c00);
    }

    #[test]
    fn face_info_flags() {
        let mut fi = FaceInfo::default();
        assert!(!fi.is_constant());
        fi.flags = face_flags::CONSTANT | face_flags::SUBFACE;
        assert!(fi.is_constant());
        assert!(fi.is_subface());
        assert!(!fi.is_neighborhood_constant());
    }
}
