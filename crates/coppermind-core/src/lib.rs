//! # Coppermind Core
//!
//! Platform-independent library for semantic search, embedding, and text chunking.
//!
//! This crate provides the core algorithms and traits used by the Coppermind
//! search engine, designed to be reusable across different frontends (GUI, CLI, MCP).
//!
//! ## Modules
//!
//! - [`search`] - Hybrid search (HNSW vector + BM25 keyword + RRF fusion)
//! - [`storage`] - Platform-agnostic storage trait
//! - [`config`] - Production configuration constants
//!
//! Modules to be migrated:
//! - `embedding` - ML model inference and text chunking
//! - `crawler` - Web page crawling (feature-gated)
//! - `error` - Error types
//! - `utils` - Utility traits

#![forbid(unsafe_code)]

pub mod config;
pub mod search;
pub mod storage;
