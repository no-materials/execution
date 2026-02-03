// Copyright 2026 the Execution Tape Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generic host catalog for registering host-call signatures.
//!
//! This module provides a lightweight registry abstraction for host-call
//! symbols and their argument/return types. It does **not** implement any
//! host functionality. It only registers signatures into a `ProgramBuilder`
//! and returns a stable lookup table of `HostSigId`s by `(symbol, sig_hash)`.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::asm::ProgramBuilder;
use crate::host::{HostSig, SigHash, sig_hash_slices};
use crate::program::{HostSigId, ValueType};

/// A host-call signature specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostSpec {
    /// Fully-qualified host symbol (stable across toolchains).
    pub symbol: String,
    /// Argument types for the host call.
    pub args: Box<[ValueType]>,
    /// Return types for the host call.
    pub rets: Box<[ValueType]>,
}

impl HostSpec {
    /// Construct a new host spec from symbol + arg/ret types.
    pub fn new(symbol: impl Into<String>, args: &[ValueType], rets: &[ValueType]) -> Self {
        Self {
            symbol: symbol.into(),
            args: Box::from(args),
            rets: Box::from(rets),
        }
    }
}

/// A collection of host-call signatures to be registered.
///
/// # Example
/// ```
/// extern crate alloc;
///
/// use alloc::vec;
///
/// use execution_tape::asm::ProgramBuilder;
/// use execution_tape::host_catalog::{HostCatalog, HostSpec};
/// use execution_tape::program::ValueType;
///
/// let mut cat = HostCatalog::new();
/// cat.push(HostSpec::new(
///     "geom::aabb3_from_size/3/1",
///     &[ValueType::F64, ValueType::F64, ValueType::F64],
///     &[ValueType::I64],
/// ));
///
/// let mut pb = ProgramBuilder::new();
/// let reg = cat.register_all(&mut pb).unwrap();
/// assert!(reg
///     .sig_id_for(
///         "geom::aabb3_from_size/3/1",
///         &[ValueType::F64, ValueType::F64, ValueType::F64],
///         &[ValueType::I64],
///     )
///     .is_some());
/// ```
#[derive(Clone, Debug, Default)]
pub struct HostCatalog {
    specs: Vec<HostSpec>,
}

impl HostCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    /// Add a host spec.
    pub fn push(&mut self, spec: HostSpec) {
        self.specs.push(spec);
    }

    /// Add multiple specs.
    pub fn extend<I: IntoIterator<Item = HostSpec>>(&mut self, iter: I) {
        self.specs.extend(iter);
    }

    /// Iterate over all specs.
    pub fn iter(&self) -> impl Iterator<Item = &HostSpec> {
        self.specs.iter()
    }

    /// Register all specs into `ProgramBuilder` and return a symbol registry.
    ///
    /// This consumes the catalog to avoid cloning argument/return type vectors.
    pub fn register_all(
        self,
        pb: &mut ProgramBuilder,
    ) -> Result<HostSigRegistry, HostCatalogError> {
        let mut by_symbol: BTreeMap<String, BTreeMap<u64, HostSigId>> = BTreeMap::new();
        for HostSpec { symbol, args, rets } in self.specs {
            let sig_hash = sig_hash_slices(args.as_ref(), rets.as_ref());
            if by_symbol
                .get(&symbol)
                .is_some_and(|existing| existing.contains_key(&sig_hash.0))
            {
                return Err(HostCatalogError::DuplicateSignature { symbol, sig_hash });
            }
            let sig_id = pb.host_sig_for(
                &symbol,
                HostSig {
                    args: Vec::from(args),
                    rets: Vec::from(rets),
                },
            );
            by_symbol
                .entry(symbol)
                .or_default()
                .insert(sig_hash.0, sig_id);
        }
        Ok(HostSigRegistry { by_symbol })
    }
}

/// Lookup registry for host signature ids.
#[derive(Clone, Debug, Default)]
pub struct HostSigRegistry {
    by_symbol: BTreeMap<String, BTreeMap<u64, HostSigId>>,
}

impl HostSigRegistry {
    /// Look up a host signature id by `(symbol, sig_hash)`.
    pub fn sig_id(&self, symbol: &str, sig_hash: SigHash) -> Option<HostSigId> {
        self.by_symbol
            .get(symbol)
            .and_then(|sigs| sigs.get(&sig_hash.0))
            .copied()
    }

    /// Look up a host signature id by `(symbol, args, rets)`.
    pub fn sig_id_for(
        &self,
        symbol: &str,
        args: &[ValueType],
        rets: &[ValueType],
    ) -> Option<HostSigId> {
        let sig_hash = sig_hash_slices(args, rets);
        self.sig_id(symbol, sig_hash)
    }
}

/// Errors when registering a host catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostCatalogError {
    /// A `(symbol, sig_hash)` appeared more than once in the catalog.
    DuplicateSignature {
        /// Host symbol string.
        symbol: String,
        /// Signature hash for the duplicate entry.
        sig_hash: SigHash,
    },
}

impl core::fmt::Display for HostCatalogError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateSignature { symbol, sig_hash } => write!(
                f,
                "duplicate host signature for '{symbol}' (sig_hash=0x{:016x})",
                sig_hash.0
            ),
        }
    }
}

impl core::error::Error for HostCatalogError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_all_rejects_duplicate_signatures() {
        let mut cat = HostCatalog::new();
        cat.push(HostSpec::new("foo", &[ValueType::I64], &[ValueType::F64]));
        cat.push(HostSpec::new("foo", &[ValueType::I64], &[ValueType::F64]));

        let mut pb = ProgramBuilder::new();
        let err = cat.register_all(&mut pb).unwrap_err();
        assert_eq!(
            err,
            HostCatalogError::DuplicateSignature {
                symbol: "foo".into(),
                sig_hash: sig_hash_slices(&[ValueType::I64], &[ValueType::F64]),
            }
        );
    }

    #[test]
    fn register_all_allows_overloads() {
        let mut cat = HostCatalog::new();
        cat.push(HostSpec::new("foo", &[ValueType::I64], &[ValueType::F64]));
        cat.push(HostSpec::new("foo", &[ValueType::F64], &[ValueType::I64]));

        let mut pb = ProgramBuilder::new();
        let reg = cat.register_all(&mut pb).expect("register");

        assert!(
            reg.sig_id_for("foo", &[ValueType::I64], &[ValueType::F64])
                .is_some()
        );
        assert!(
            reg.sig_id_for("foo", &[ValueType::F64], &[ValueType::I64])
                .is_some()
        );
    }

    #[test]
    fn register_all_returns_registry() {
        let mut cat = HostCatalog::new();
        cat.push(HostSpec::new("foo", &[ValueType::I64], &[ValueType::F64]));
        cat.push(HostSpec::new("bar", &[ValueType::F64], &[ValueType::I64]));

        let mut pb = ProgramBuilder::new();
        let reg = cat.register_all(&mut pb).expect("register");

        assert!(
            reg.sig_id_for("foo", &[ValueType::I64], &[ValueType::F64])
                .is_some()
        );
        assert!(
            reg.sig_id_for("bar", &[ValueType::F64], &[ValueType::I64])
                .is_some()
        );
        assert!(
            reg.sig_id_for("baz", &[ValueType::I64], &[ValueType::F64])
                .is_none()
        );
    }
}
