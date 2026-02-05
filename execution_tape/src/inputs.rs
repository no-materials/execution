// Copyright 2026 the Execution Tape Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Program input binding and canonical snapshot encoding.
//!
//! This module is `no_std + alloc` friendly. It provides a resolver interface that allows a VM
//! embedder to supply inputs *by name* without explicitly pushing every value into bytecode.
//! Inputs are resolved once per run into a deterministic snapshot.

use alloc::string::String;
use alloc::vec::Vec;

use crate::format::{DecodeError, Reader, write_sleb128_i64, write_uleb128_u64};
use crate::program::{InputId, Program, SymbolId, ValueType};
use crate::value::Decimal;

const SNAPSHOT_MAGIC: &[u8; 8] = b"EXTINP\0\0";
const SNAPSHOT_VERSION_MAJOR: u16 = 0;
const SNAPSHOT_VERSION_MINOR: u16 = 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum SnapshotTag {
    Unit = 0,
    Bool = 1,
    I64 = 2,
    U64 = 3,
    F64 = 4,
    Decimal = 5,
    Bytes = 6,
    Str = 7,
}

impl SnapshotTag {
    fn from_u8(v: u8) -> Result<Self, DecodeError> {
        match v {
            0 => Ok(Self::Unit),
            1 => Ok(Self::Bool),
            2 => Ok(Self::I64),
            3 => Ok(Self::U64),
            4 => Ok(Self::F64),
            5 => Ok(Self::Decimal),
            6 => Ok(Self::Bytes),
            7 => Ok(Self::Str),
            _ => Err(DecodeError::OutOfBounds),
        }
    }
}

/// A single input value for snapshot binding.
#[derive(Clone, Debug, PartialEq)]
pub enum InputValue {
    /// `()`.
    Unit,
    /// Boolean.
    Bool(bool),
    /// Signed 64-bit integer.
    I64(i64),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// 64-bit float.
    F64(f64),
    /// Decimal.
    Decimal(Decimal),
    /// Byte string.
    Bytes(Vec<u8>),
    /// UTF-8 string.
    Str(String),
}

impl InputValue {
    /// Returns the corresponding [`ValueType`].
    #[must_use]
    pub fn ty(&self) -> ValueType {
        match self {
            Self::Unit => ValueType::Unit,
            Self::Bool(_) => ValueType::Bool,
            Self::I64(_) => ValueType::I64,
            Self::U64(_) => ValueType::U64,
            Self::F64(_) => ValueType::F64,
            Self::Decimal(_) => ValueType::Decimal,
            Self::Bytes(_) => ValueType::Bytes,
            Self::Str(_) => ValueType::Str,
        }
    }
}

/// A resolved input key passed to a resolver.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InputKey<'a> {
    /// Input id (index into the program input table).
    pub id: InputId,
    /// Input symbol string.
    pub symbol: &'a str,
    /// Declared input type.
    pub ty: ValueType,
}

/// An input resolver used to fetch values by name.
pub trait InputResolver {
    /// Resolves a value for `key`.
    fn resolve(&mut self, key: InputKey<'_>) -> Result<InputValue, InputError>;
}

/// An error returned by an [`InputResolver`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputError {
    /// No input value was found.
    NotFound,
    /// The resolver returned the wrong type.
    TypeMismatch {
        /// Expected type.
        expected: ValueType,
        /// Actual type.
        actual: ValueType,
    },
    /// A resolver-specific failure occurred.
    ResolverFailed,
}

/// A bind-time error when resolving or validating inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputBindError {
    /// The input table referenced an invalid symbol id.
    InvalidSymbolId {
        /// Input id.
        id: InputId,
        /// Symbol id.
        symbol: SymbolId,
    },
    /// The input table declared an unsupported value type.
    UnsupportedType {
        /// Input id.
        id: InputId,
        /// Declared type.
        ty: ValueType,
    },
    /// A required input was missing.
    MissingInput {
        /// Input id.
        id: InputId,
    },
    /// A resolved input had the wrong type.
    TypeMismatch {
        /// Input id.
        id: InputId,
        /// Expected type.
        expected: ValueType,
        /// Actual type.
        actual: ValueType,
    },
    /// Resolver failed for this input.
    ResolverFailed {
        /// Input id.
        id: InputId,
    },
    /// Snapshot length did not match the program input table.
    SnapshotLenMismatch {
        /// Expected count.
        expected: usize,
        /// Actual count.
        actual: usize,
    },
    /// Snapshot decoding failed.
    SnapshotDecodeError(DecodeError),
    /// Snapshot had trailing bytes after decoding.
    SnapshotTrailingBytes,
}

/// A resolved input snapshot used for deterministic replay.
#[derive(Clone, Debug, PartialEq)]
pub struct InputSnapshot {
    /// Entries ordered by `InputId`.
    pub entries: Vec<InputValue>,
}

impl InputSnapshot {
    /// Returns the input value for `id`, if present.
    #[must_use]
    pub fn get(&self, id: InputId) -> Option<&InputValue> {
        self.entries.get(id.0 as usize)
    }

    /// Encodes the snapshot in a canonical, deterministic format.
    pub fn encode_canonical(&self, out: &mut Vec<u8>) -> Result<(), InputBindError> {
        out.clear();
        out.extend_from_slice(SNAPSHOT_MAGIC);
        out.extend_from_slice(&SNAPSHOT_VERSION_MAJOR.to_le_bytes());
        out.extend_from_slice(&SNAPSHOT_VERSION_MINOR.to_le_bytes());

        write_uleb128_u64(out, self.entries.len() as u64);
        for entry in &self.entries {
            encode_snapshot_value(out, entry);
        }
        Ok(())
    }

    /// Decodes a snapshot from the canonical format.
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, InputBindError> {
        let mut r = Reader::new(bytes);
        let magic = r
            .read_bytes(SNAPSHOT_MAGIC.len())
            .map_err(InputBindError::SnapshotDecodeError)?;
        if magic != SNAPSHOT_MAGIC {
            return Err(InputBindError::SnapshotDecodeError(DecodeError::BadMagic));
        }
        let major = r
            .read_u16_le()
            .map_err(InputBindError::SnapshotDecodeError)?;
        let minor = r
            .read_u16_le()
            .map_err(InputBindError::SnapshotDecodeError)?;
        if major != SNAPSHOT_VERSION_MAJOR || minor != SNAPSHOT_VERSION_MINOR {
            return Err(InputBindError::SnapshotDecodeError(
                DecodeError::UnsupportedVersion { major, minor },
            ));
        }

        let count = read_usize(&mut r).map_err(InputBindError::SnapshotDecodeError)?;
        let mut entries = Vec::with_capacity(count);
        // We intentionally decode with a single pass and avoid intermediate allocations.
        for _ in 0..count {
            entries.push(decode_snapshot_value(&mut r)?);
        }
        if r.offset() != bytes.len() {
            return Err(InputBindError::SnapshotTrailingBytes);
        }
        Ok(Self { entries })
    }
}

/// A helper for resolving and validating program inputs.
#[derive(Copy, Clone, Debug)]
pub struct InputBinder<'a> {
    program: &'a Program,
}

impl<'a> InputBinder<'a> {
    /// Creates a binder over `program`.
    #[must_use]
    pub fn new(program: &'a Program) -> Self {
        Self { program }
    }

    /// Resolves all inputs in `program` using `resolver` and returns a snapshot.
    pub fn resolve(
        &self,
        resolver: &mut dyn InputResolver,
    ) -> Result<InputSnapshot, InputBindError> {
        let mut entries = Vec::with_capacity(self.program.input_table.len());
        for (i, input) in self.program.input_table.iter().enumerate() {
            let id = InputId(u32::try_from(i).unwrap_or(u32::MAX));
            let symbol = self.program.symbol_str(input.symbol).map_err(|_| {
                InputBindError::InvalidSymbolId {
                    id,
                    symbol: input.symbol,
                }
            })?;
            if !is_supported_type(input.ty) {
                return Err(InputBindError::UnsupportedType { id, ty: input.ty });
            }
            let key = InputKey {
                id,
                symbol,
                ty: input.ty,
            };
            let value = match resolver.resolve(key) {
                Ok(v) => v,
                Err(InputError::NotFound) => return Err(InputBindError::MissingInput { id }),
                Err(InputError::TypeMismatch { expected, actual }) => {
                    return Err(InputBindError::TypeMismatch {
                        id,
                        expected,
                        actual,
                    });
                }
                Err(InputError::ResolverFailed) => {
                    return Err(InputBindError::ResolverFailed { id });
                }
            };
            if value.ty() != input.ty {
                return Err(InputBindError::TypeMismatch {
                    id,
                    expected: input.ty,
                    actual: value.ty(),
                });
            }
            entries.push(value);
        }
        Ok(InputSnapshot { entries })
    }

    /// Validates `snapshot` against the program input table.
    pub fn validate_snapshot(&self, snapshot: &InputSnapshot) -> Result<(), InputBindError> {
        let expected = self.program.input_table.len();
        let actual = snapshot.entries.len();
        if expected != actual {
            return Err(InputBindError::SnapshotLenMismatch { expected, actual });
        }
        for (i, (input, value)) in self
            .program
            .input_table
            .iter()
            .zip(snapshot.entries.iter())
            .enumerate()
        {
            let id = InputId(u32::try_from(i).unwrap_or(u32::MAX));
            if !is_supported_type(input.ty) {
                return Err(InputBindError::UnsupportedType { id, ty: input.ty });
            }
            if value.ty() != input.ty {
                return Err(InputBindError::TypeMismatch {
                    id,
                    expected: input.ty,
                    actual: value.ty(),
                });
            }
        }
        Ok(())
    }
}

fn is_supported_type(ty: ValueType) -> bool {
    matches!(
        ty,
        ValueType::Unit
            | ValueType::Bool
            | ValueType::I64
            | ValueType::U64
            | ValueType::F64
            | ValueType::Decimal
            | ValueType::Bytes
            | ValueType::Str
    )
}

fn encode_snapshot_value(out: &mut Vec<u8>, v: &InputValue) {
    match v {
        InputValue::Unit => out.push(SnapshotTag::Unit as u8),
        InputValue::Bool(b) => {
            out.push(SnapshotTag::Bool as u8);
            out.push(u8::from(*b));
        }
        InputValue::I64(i) => {
            out.push(SnapshotTag::I64 as u8);
            write_sleb128_i64(out, *i);
        }
        InputValue::U64(u) => {
            out.push(SnapshotTag::U64 as u8);
            write_uleb128_u64(out, *u);
        }
        InputValue::F64(f) => {
            out.push(SnapshotTag::F64 as u8);
            out.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        InputValue::Decimal(d) => {
            out.push(SnapshotTag::Decimal as u8);
            write_sleb128_i64(out, d.mantissa);
            out.push(d.scale);
        }
        InputValue::Bytes(b) => {
            out.push(SnapshotTag::Bytes as u8);
            write_uleb128_u64(out, b.len() as u64);
            out.extend_from_slice(b);
        }
        InputValue::Str(s) => {
            out.push(SnapshotTag::Str as u8);
            write_uleb128_u64(out, s.len() as u64);
            out.extend_from_slice(s.as_bytes());
        }
    }
}

fn decode_snapshot_value(r: &mut Reader<'_>) -> Result<InputValue, InputBindError> {
    let tag = SnapshotTag::from_u8(r.read_u8().map_err(InputBindError::SnapshotDecodeError)?)
        .map_err(InputBindError::SnapshotDecodeError)?;
    let v = match tag {
        SnapshotTag::Unit => InputValue::Unit,
        SnapshotTag::Bool => {
            let b = r.read_u8().map_err(InputBindError::SnapshotDecodeError)?;
            InputValue::Bool(b != 0)
        }
        SnapshotTag::I64 => {
            let i = r
                .read_sleb128_i64()
                .map_err(InputBindError::SnapshotDecodeError)?;
            InputValue::I64(i)
        }
        SnapshotTag::U64 => {
            let u = r
                .read_uleb128_u64()
                .map_err(InputBindError::SnapshotDecodeError)?;
            InputValue::U64(u)
        }
        SnapshotTag::F64 => {
            let bits = r
                .read_u64_le()
                .map_err(InputBindError::SnapshotDecodeError)?;
            InputValue::F64(f64::from_bits(bits))
        }
        SnapshotTag::Decimal => {
            let mantissa = r
                .read_sleb128_i64()
                .map_err(InputBindError::SnapshotDecodeError)?;
            let scale = r.read_u8().map_err(InputBindError::SnapshotDecodeError)?;
            InputValue::Decimal(Decimal { mantissa, scale })
        }
        SnapshotTag::Bytes => {
            let len = read_usize(r).map_err(InputBindError::SnapshotDecodeError)?;
            let bytes = r
                .read_bytes(len)
                .map_err(InputBindError::SnapshotDecodeError)?;
            InputValue::Bytes(bytes.to_vec())
        }
        SnapshotTag::Str => {
            let len = read_usize(r).map_err(InputBindError::SnapshotDecodeError)?;
            let s = r
                .read_str(len)
                .map_err(InputBindError::SnapshotDecodeError)?;
            InputValue::Str(String::from(s))
        }
    };
    Ok(v)
}

fn read_usize(r: &mut Reader<'_>) -> Result<usize, DecodeError> {
    let v = r.read_uleb128_u64()?;
    usize::try_from(v).map_err(|_| DecodeError::OutOfBounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{HostSymbol, InputDecl, Program, TypeTableDef};
    use alloc::vec;

    struct MapResolver {
        entries: Vec<(String, InputValue)>,
    }

    impl InputResolver for MapResolver {
        fn resolve(&mut self, key: InputKey<'_>) -> Result<InputValue, InputError> {
            for (k, v) in &self.entries {
                if k.as_str() == key.symbol {
                    if v.ty() != key.ty {
                        return Err(InputError::TypeMismatch {
                            expected: key.ty,
                            actual: v.ty(),
                        });
                    }
                    return Ok(v.clone());
                }
            }
            Err(InputError::NotFound)
        }
    }

    #[test]
    fn snapshot_roundtrips_canonical_encoding() {
        let snapshot = InputSnapshot {
            entries: vec![
                InputValue::Unit,
                InputValue::Bool(true),
                InputValue::I64(-12),
                InputValue::U64(99),
                InputValue::F64(1.5),
                InputValue::Decimal(Decimal {
                    mantissa: 1234,
                    scale: 2,
                }),
                InputValue::Bytes(vec![1, 2, 3, 4]),
                InputValue::Str("hello".into()),
            ],
        };
        let mut bytes = Vec::new();
        snapshot.encode_canonical(&mut bytes).unwrap();
        let back = InputSnapshot::decode_canonical(&bytes).unwrap();
        assert_eq!(back, snapshot);
    }

    #[test]
    fn binder_resolves_and_validates_inputs() {
        let symbols = vec![
            HostSymbol {
                symbol: "env.CONFIG.max_items".into(),
            },
            HostSymbol {
                symbol: "env.FEATURES".into(),
            },
        ];
        let input_table = vec![
            InputDecl {
                symbol: SymbolId(0),
                ty: ValueType::I64,
            },
            InputDecl {
                symbol: SymbolId(1),
                ty: ValueType::Str,
            },
        ];
        let program = Program::new(
            symbols,
            input_table,
            vec![],
            vec![],
            TypeTableDef::default(),
            vec![],
        );

        let mut resolver = MapResolver {
            entries: vec![
                ("env.CONFIG.max_items".into(), InputValue::I64(7)),
                ("env.FEATURES".into(), InputValue::Str("a,b".into())),
            ],
        };
        let binder = InputBinder::new(&program);
        let snapshot = binder.resolve(&mut resolver).unwrap();
        binder.validate_snapshot(&snapshot).unwrap();
        assert_eq!(snapshot.get(InputId(0)), Some(&InputValue::I64(7)));
    }

    #[test]
    fn binder_rejects_missing_input() {
        let symbols = vec![HostSymbol {
            symbol: "env.CONFIG.max_items".into(),
        }];
        let input_table = vec![InputDecl {
            symbol: SymbolId(0),
            ty: ValueType::I64,
        }];
        let program = Program::new(
            symbols,
            input_table,
            vec![],
            vec![],
            TypeTableDef::default(),
            vec![],
        );
        let mut resolver = MapResolver { entries: vec![] };
        let binder = InputBinder::new(&program);
        let err = binder.resolve(&mut resolver).unwrap_err();
        assert!(matches!(err, InputBindError::MissingInput { .. }));
    }
}
