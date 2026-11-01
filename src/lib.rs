//! Command-line front end over the [`knowledge`] crate.
//!
//! The engine, storage and MCP server live in `knowledge`; this crate owns the
//! CLI surface and the product-side evaluation harnesses (`eval`, `answer_eval`)
//! that pin this repository's lock file and fixtures.

pub use knowledge::*;

#[cfg(feature = "full")]
pub mod answer_eval;
pub mod eval;
