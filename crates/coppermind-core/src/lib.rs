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
//! - [`error`] - Error types for embedding, chunking, and GPU operations
//! - [`embedding`] - ML model inference abstractions and implementations
//! - [`chunking`] - Text chunking strategies (text, markdown, code)
//! - [`gpu`] - GPU scheduler for Metal thread-safe model inference
//! - [`processing`] - Document processing and indexing pipeline
//! - [`metrics`] - Performance metrics collection with rolling averages

// Note: unsafe is used only in processing/pipeline.rs for lifetime transmute
// of tokenizer reference to satisfy the text-splitter crate's 'static requirement.
// This is safe because the tokenizer is stored in an Arc and outlives its usage.
#![allow(unsafe_code)]

pub mod chunking;
pub mod config;
pub mod embedding;
pub mod error;
pub mod gpu;
pub mod metrics;
pub mod processing;
pub mod search;
pub mod storage;
