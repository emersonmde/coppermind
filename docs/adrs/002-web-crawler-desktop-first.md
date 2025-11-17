# ADR 002: Web Crawler - Desktop-First with Cross-Platform Architecture

**Status:** Accepted
**Date:** 2025-11-16
**Context:** Adding web crawler feature to fetch and index web pages

---

## Summary

Implement web crawler as **desktop-only initially**, but architect the module for easy web platform enablement in the future. Use cross-platform Rust crates (`reqwest` + `scraper`) so web support is a configuration change, not a refactor.

---

## Context

### Feature Requirements

User wants to paste a URL (e.g., `https://example.com/docs`) and have the app:
1. Fetch the HTML page
2. Extract visible text content (strip tags)
3. Find all links (`<a href="...">`)
4. Follow same-origin links recursively (e.g., `example.com/docs/*`)
5. Feed extracted text to existing embedding pipeline for indexing

### The CORS Problem

**CORS blocks cross-origin fetches on web platform:**

- Browser's Same-Origin Policy restricts fetching resources from other domains
- Most websites don't send CORS headers (`Access-Control-Allow-Origin`)
- **Result:** fetch blocked by browser for most external URLs
- Desktop has no CORS restrictions (OS-level HTTP)

---

## Decision

### Phase 1: Desktop-Only Implementation (Current)

**Module structure** (`src/crawler/mod.rs`):
```rust
// Core crawler logic - cross-platform from day one
#[cfg(not(target_arch = "wasm32"))]
pub mod crawler;

pub struct CrawlConfig {
    pub start_url: String,
    pub max_depth: usize,
    pub same_origin_only: bool,
}

pub struct CrawlResult {
    pub url: String,
    pub text: String,
    pub links: Vec<String>,
}

// Implementation uses reqwest + scraper (both work on WASM + native)
```

**UI conditionally shown:**
```rust
// In components/hero.rs or new component
#[cfg(not(target_arch = "wasm32"))]
rsx! {
    WebCrawlerInput { /* ... */ }
}
```

**Dependencies (cross-platform ready):**
```toml
[dependencies]
scraper = "0.22"  # Pure Rust, works on WASM (10.7M downloads)
url = "2.5"       # URL parsing and origin detection

# reqwest with platform-specific TLS
reqwest = { version = "0.12", default-features = false }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls"] }
```

**Why these crates:**
- **reqwest**: Industry standard (54M downloads), unified API across platforms
  - WASM: Uses browser `fetch()` API internally
  - Native: Uses `hyper` with TLS support
- **scraper**: Browser-grade HTML5 parser (Servo project), CSS selectors
  - Pure Rust (no C dependencies like tree-sitter-html)
  - Works on WASM (proven in production)
- **url**: Standard Rust URL parsing (same-origin matching)

### Phase 2: Future Web Support (Optional)

**Option A: Browser Extension**

- Desktop: Direct fetching (no extension needed)
- Web: Optional browser extension that operates in different security context
- Extension can bypass CORS restrictions (background script has broader permissions)
- Webpage includes "Install Extension" button for users who want web crawling
- Preserves local-first architecture (extension runs locally, no external services)

**Option B: Accept web platform limitations**

- Desktop-only feature (current approach)
- Web platform focuses on file upload instead of crawling
- Simplest implementation, no workarounds needed

---

## Consequences

### ✅ Positive

1. **Desktop gets full functionality** - No CORS restrictions, fast HTTP, can write to disk
2. **Clean architecture** - Module is cross-platform ready, just gated by `cfg`
3. **Easy to enable on web later** - Remove `cfg` gate + add CORS proxy option = web support
4. **No tree-sitter complexity** - HTML parsing is pure Rust, works everywhere
5. **Matches philosophy** - Desktop for power features, web for lightweight use

### ⚠️ Neutral

1. **Web platform can't crawl initially** - Acceptable (can still upload files)
2. **Future decision required** - Must choose Option A/B if web crawling is desired

### ❌ Negative

1. **Platform divergence** - Desktop has features web doesn't (mitigated by clear UI)

---

## Implementation Notes

### Module Organization

```
src/
├── crawler/
│   ├── mod.rs           # Public API, CrawlConfig/CrawlResult types
│   ├── fetcher.rs       # HTTP fetching (reqwest wrapper)
│   ├── parser.rs        # HTML parsing (scraper wrapper)
│   └── engine.rs        # Recursive crawl logic, cycle detection
└── components/
    └── web_crawler.rs   # UI component (desktop-only)
```

### Integration Points

1. **Text extraction** → Existing chunking pipeline (`embedding/chunking/mod.rs`)
   - Detect as `FileType::Text` or new `FileType::Html`
   - Use `TextSplitter` for semantic chunking

2. **Embedding** → Existing `embed_text_chunks()` function
   - Each page = document with URL as ID

3. **Storage** → Existing hybrid search index
   - Store crawled pages alongside uploaded files

### Proof of Concept Scope

Minimal implementation to validate architecture:
- [ ] Fetch single HTML page (no recursion)
- [ ] Extract visible text (strip tags)
- [ ] Extract all `<a href>` links
- [ ] Log results to console
- [ ] Verify works on desktop (skip web platform)

**Success criteria:**
- Desktop: `dx serve --platform desktop` → crawler UI visible and functional
- Web: `dx serve` → crawler UI hidden (no errors)

---

## Alternatives Considered

### ❌ Tree-sitter-html for Parsing

**Rejected because:**
- Requires C library (breaks WASM unless using complex workarounds)
- Overkill for HTML (need structure extraction, not syntax analysis)
- `scraper` is purpose-built for this exact use case
- Current codebase already limits tree-sitter to native-only (code chunking)

### ❌ User-Provided CORS Proxy (localhost)

**Rejected because:**
- Only works for local development (`dx serve`)
- Deployed web apps cannot access user's localhost due to Private Network Access (PNA) restrictions
- Browser security blocks `https://example.com` from fetching `http://localhost:9000`
- Would work: `http://localhost:8080` (app) → `http://localhost:9000` (proxy)
- Wouldn't work: `https://errorsignal.dev` (app) → `http://localhost:9000` (proxy)

### ❌ Publicly Hosted CORS Proxy

**Rejected because:**
- Violates local-first architecture (requires external proxy service)
- User privacy concerns (proxy sees all crawled URLs)
- Introduces external dependency (proxy must be maintained, hosted, and available)
- Defeats purpose of local-first design

### ❌ Implement Twice (Separate Web/Desktop Code)

**Rejected because:**
- `reqwest` and `scraper` work on both platforms already
- DRY principle - shared code is easier to maintain
- Future web support would require refactor

---

## References

- **reqwest WASM support:** https://docs.rs/reqwest (automatic platform detection)
- **scraper WASM support:** https://docs.rs/scraper (pure Rust HTML5 parser)
- **CORS and Same-Origin Policy:** https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS

---

## Future Enhancements

Once basic crawler is working on desktop:

1. **Recursive crawling** with depth limit
2. **Progress UI** (X pages crawled, Y queued)
3. **URL filtering** (same-origin matching with `url` crate)
4. **Robots.txt support** (respect crawler directives)
5. **Rate limiting** (politeness delay between requests)
6. **Error handling** (404s, timeouts, redirects)
7. **Sitemap.xml parsing** (crawl seed from sitemap)
8. **Web platform support** (via optional browser extension with "Install Extension" button)
