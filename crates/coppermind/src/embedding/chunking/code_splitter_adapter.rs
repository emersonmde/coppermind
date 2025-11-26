//! Code-specific chunking adapter (native platforms only).
//!
//! This module wraps the `text-splitter` crate's `CodeSplitter` to provide
//! syntax-aware chunking for programming language source files using tree-sitter.
//!
//! # Platform Availability
//!
//! **Native only (desktop/mobile)** - tree-sitter uses C code that doesn't compile
//! to WASM. On web platform, code files fall back to text-based chunking.
//!
//! # Benefits over generic text splitting
//!
//! - **Syntax-aware**: Respects function/class boundaries, scope blocks
//! - **Language-specific**: Uses tree-sitter grammars for each language
//! - **Semantic units**: Chunks preserve complete syntactic elements
//! - **Context preservation**: Keeps related code together (e.g., function + docstring)

use super::{calculate_chunk_boundaries, ChunkingStrategy, TextChunk, TokenizerSizer};
use crate::error::EmbeddingError;
use text_splitter::{ChunkConfig, CodeSplitter};
use tokenizers::Tokenizer;
#[cfg(feature = "profile")]
use tracing::instrument;

/// Supported programming languages for code chunking.
///
/// Each language uses its corresponding tree-sitter grammar for AST-based
/// chunking. Languages are detected from file extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLanguage {
    /// Rust (.rs)
    Rust,
    /// Python (.py)
    Python,
    /// JavaScript (.js, .mjs)
    JavaScript,
    /// TypeScript (.ts)
    TypeScript,
    /// Java (.java)
    Java,
    /// C (.c, .h)
    C,
    /// C++ (.cpp, .hpp, .cc, .cxx)
    Cpp,
    /// Go (.go)
    Go,
}

impl CodeLanguage {
    /// Detects code language from file extension.
    ///
    /// Returns None if extension doesn't match a supported language.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "js" | "mjs" => Some(Self::JavaScript),
            "ts" => Some(Self::TypeScript),
            "java" => Some(Self::Java),
            "c" | "h" => Some(Self::C),
            "cpp" | "hpp" | "cc" | "cxx" | "hxx" => Some(Self::Cpp),
            "go" => Some(Self::Go),
            _ => None,
        }
    }

    /// Returns the tree-sitter language for this code language.
    fn tree_sitter_language(&self) -> tree_sitter::Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
        }
    }
}

/// Code splitter adapter using the `text-splitter` crate with tree-sitter.
///
/// Provides syntax-aware chunking for programming language source files.
///
/// # Platform Support
///
/// **Native platforms only** (desktop, mobile). The web platform doesn't support
/// this chunker because tree-sitter's C code doesn't compile to WASM.
///
/// # Algorithm
///
/// The CodeSplitter uses tree-sitter's AST to:
/// 1. Parse source code into an abstract syntax tree
/// 2. Identify semantic boundaries (functions, classes, blocks)
/// 3. Split at the highest semantic level that fits within chunk size
/// 4. Preserve syntactic completeness (no mid-expression splits)
///
/// # Examples
///
/// ```ignore
/// use crate::embedding::chunking::code_splitter_adapter::{CodeSplitterAdapter, CodeLanguage};
/// use crate::embedding::ensure_tokenizer;
///
/// let tokenizer = ensure_tokenizer(2048)?;
/// let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Rust, tokenizer);
/// let chunks = chunker.chunk("fn main() {\n    println!(\"Hello\");\n}")?;
/// ```
pub struct CodeSplitterAdapter {
    max_tokens: usize,
    language: CodeLanguage,
    tokenizer: &'static Tokenizer,
}

impl CodeSplitterAdapter {
    /// Creates a new code-splitter adapter.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens per chunk
    /// * `language` - Programming language for syntax-aware splitting
    /// * `tokenizer` - Reference to HuggingFace tokenizer (must be static lifetime)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Python, tokenizer);
    /// ```
    pub fn new(max_tokens: usize, language: CodeLanguage, tokenizer: &'static Tokenizer) -> Self {
        Self {
            max_tokens,
            language,
            tokenizer,
        }
    }
}

impl ChunkingStrategy for CodeSplitterAdapter {
    #[cfg_attr(feature = "profile", instrument(skip_all, fields(text_len = text.len(), max_tokens = self.max_tokens, language = ?self.language)))]
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, EmbeddingError> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Create our custom ChunkSizer
        let sizer = TokenizerSizer {
            tokenizer: self.tokenizer,
        };

        // Create chunk config with our tokenizer sizer
        let chunk_config = ChunkConfig::new(self.max_tokens)
            .with_sizer(sizer)
            .with_trim(true);

        // Use text-splitter's code chunking with tree-sitter
        let language = self.language.tree_sitter_language();
        let splitter = CodeSplitter::new(language, chunk_config)
            .map_err(|e| EmbeddingError::ChunkingFailed(e.to_string()))?;

        // Use shared helper to calculate boundaries correctly for duplicate text
        let chunks = calculate_chunk_boundaries(text, splitter.chunks(text));

        Ok(chunks)
    }

    fn name(&self) -> &'static str {
        "code-splitter"
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::tokenizer::ensure_tokenizer;

    // Helper to load test tokenizer
    fn load_test_tokenizer() -> &'static Tokenizer {
        use std::fs;
        let tokenizer_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/models/jina-bert-tokenizer.json"
        );
        let tokenizer_bytes = fs::read(tokenizer_path).expect("Failed to read tokenizer file");
        ensure_tokenizer(tokenizer_bytes, 2048).expect("Failed to load test tokenizer")
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(CodeLanguage::from_extension("rs"), Some(CodeLanguage::Rust));
        assert_eq!(
            CodeLanguage::from_extension("py"),
            Some(CodeLanguage::Python)
        );
        assert_eq!(
            CodeLanguage::from_extension("js"),
            Some(CodeLanguage::JavaScript)
        );
        assert_eq!(
            CodeLanguage::from_extension("ts"),
            Some(CodeLanguage::TypeScript)
        );
        assert_eq!(
            CodeLanguage::from_extension("java"),
            Some(CodeLanguage::Java)
        );
        assert_eq!(CodeLanguage::from_extension("c"), Some(CodeLanguage::C));
        assert_eq!(CodeLanguage::from_extension("cpp"), Some(CodeLanguage::Cpp));
        assert_eq!(CodeLanguage::from_extension("go"), Some(CodeLanguage::Go));
        assert_eq!(CodeLanguage::from_extension("md"), None);
        assert_eq!(CodeLanguage::from_extension("txt"), None);
    }

    #[test]
    fn test_code_splitter_rust() {
        let tokenizer = load_test_tokenizer();
        let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Rust, tokenizer);

        let code = r#"
fn main() {
    println!("Hello, world!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let chunks = chunker.chunk(code).unwrap();

        assert!(!chunks.is_empty());
        // With 512 token limit, this should fit in one chunk
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_code_splitter_python() {
        let tokenizer = load_test_tokenizer();
        let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Python, tokenizer);

        let code = r#"
def hello():
    print("Hello, world!")

def add(a, b):
    return a + b
"#;
        let chunks = chunker.chunk(code).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_code() {
        let tokenizer = load_test_tokenizer();
        let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Rust, tokenizer);

        let chunks = chunker.chunk("").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_whitespace_only_code() {
        let tokenizer = load_test_tokenizer();
        let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Rust, tokenizer);

        let chunks = chunker.chunk("   \n\t  ").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_small_chunk_size() {
        let tokenizer = load_test_tokenizer();
        // Very small chunk size to force multiple chunks
        let chunker = CodeSplitterAdapter::new(5, CodeLanguage::Rust, tokenizer);

        let code = r#"
fn function_one() {
    let x = 1;
    let y = 2;
    println!("{}", x + y);
}

fn function_two() {
    let a = 10;
    let b = 20;
    println!("{}", a * b);
}

fn function_three() {
    let foo = "bar";
    let baz = "qux";
    println!("{} {}", foo, baz);
}
"#;
        let chunks = chunker.chunk(code).unwrap();

        // Should create multiple chunks
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_chunking_strategy_trait() {
        let tokenizer = load_test_tokenizer();
        let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Rust, tokenizer);

        assert_eq!(chunker.name(), "code-splitter");
        assert_eq!(chunker.max_tokens(), 512);
    }

    #[test]
    fn test_chunk_boundaries() {
        let tokenizer = load_test_tokenizer();
        let chunker = CodeSplitterAdapter::new(512, CodeLanguage::Rust, tokenizer);

        let code = "fn main() {\n    println!(\"test\");\n}";
        let chunks = chunker.chunk(code).unwrap();

        // Verify chunks have valid boundaries
        for chunk in &chunks {
            assert!(chunk.end_char >= chunk.start_char);
            assert!(chunk.end_char <= code.len());
        }
    }
}
