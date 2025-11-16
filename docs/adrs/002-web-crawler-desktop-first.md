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

**Cross-Origin Isolation (COI) blocks cross-origin fetches on web platform:**

- Service worker (`public/coi-serviceworker.min.js`) sets:
  - `Cross-Origin-Embedder-Policy: require-corp`
  - `Cross-Origin-Opener-Policy: same-origin`
- This requires external sites to send CORS headers (`Access-Control-Allow-Origin` or `Cross-Origin-Resource-Policy: cross-origin`)
- **Most websites don't send these headers** → fetch blocked by browser
- Desktop has no CORS restrictions (OS-level HTTP)

### Why COI Exists (Currently Unused, But Valuable)

**Current Status: COI enabled but NOT actively parallelizing**

Rayon is in the dependency tree (via Candle), but running **single-threaded on WASM**:
```bash
# Rayon is included in WASM build
$ cargo tree --target wasm32-unknown-unknown -p rayon
rayon v1.11.0
├── candle-core v0.8.4
├── instant-distance v0.6.1  # Already using rayon
└── ...

# But web_spin_lock feature is NOT enabled (required for WASM parallelism)
$ cargo tree --target wasm32-unknown-unknown -p rayon --format "{p} {f}"
rayon v1.11.0  # <-- no web_spin_lock feature
```

**Parallel ML Inference (3x speedup - IMPLEMENTED 2025-11-16):**

~~Draft~~ Proven speedup from Hugging Face Candle PR #3063:
- **Before:** 5 tokens/sec (single-threaded WASM)
- **After:** 16 tokens/sec (multi-threaded with `wasm-bindgen-rayon`)
- **Source:** https://github.com/huggingface/candle/pull/3063

**Status: ✅ IMPLEMENTED**
1. ✅ COOP/COEP headers (already had via service worker)
2. ✅ Enable rayon's `web_spin_lock` feature (added to Cargo.toml)
3. ✅ Add `wasm-bindgen-rayon` adapter dependency (v1.2)
4. ✅ Switch to nightly Rust (`rust-toolchain.toml`)
5. ✅ Initialize thread pool in worker (`init_thread_pool(navigator.hardwareConcurrency)`)

**Conclusion:** COI is now ACTIVELY USED for 3x ML speedup. Critical to preserve for performance.

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

**Option A: Dynamic COI Toggle (Recommended for web support)**

Service worker already supports deregistration:
```javascript
// Send message to service worker
navigator.serviceWorker.controller.postMessage({ type: "deregister" });
// → Page reloads without COI
// → Cross-origin fetches now work
```

UI implementation:
```rust
// Settings panel
rsx! {
    div { class: "coi-toggle",
        p { "Cross-Origin Isolation: " {if coi_enabled() { "Enabled" } else { "Disabled" }} }
        p { class: "help-text",
            "Disable to allow web crawling. Disabling prevents future parallel ML inference."
        }
        button {
            onclick: move |_| toggle_coi(),
            {if coi_enabled() { "Disable COI" } else { "Enable COI" }}
        }
    }

    // Show crawler only when COI disabled
    if !coi_enabled() {
        WebCrawlerInput { /* ... */ }
    }
}
```

**Option B: User-Provided CORS Proxy**

- Desktop: Direct fetching (no proxy)
- Web: Optional proxy URL setting
- User runs their own proxy (e.g., `cors-anywhere` on localhost)
- Preserves local-first architecture (user controls proxy)

**Option C: Remove COI Entirely**

- ❌ Loses future 3x ML speedup potential
- ❌ Cannot use SharedArrayBuffer/Rayon in WASM
- Not recommended

---

## Consequences

### ✅ Positive

1. **Desktop gets full functionality** - No CORS restrictions, fast HTTP, can write to disk
2. **Preserves COI for future ML speedup** - 3x performance gain when Rayon WASM matures
3. **Clean architecture** - Module is cross-platform ready, just gated by `cfg`
4. **Easy to enable on web later** - Remove `cfg` gate + add COI toggle = web support
5. **No tree-sitter complexity** - HTML parsing is pure Rust, works everywhere
6. **Matches philosophy** - Desktop for power features, web for lightweight use

### ⚠️ Neutral

1. **Web platform can't crawl initially** - Acceptable (can still upload files)
2. **Future decision required** - Must choose Option A/B/C if web crawling is desired

### ❌ Negative

1. **Platform divergence** - Desktop has features web doesn't (mitigated by clear UI)
2. **Service worker complexity** - Dynamic toggle adds UX complexity (but technically feasible)

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

### ❌ Web-Only with CORS Proxy by Default

**Rejected because:**
- Violates local-first architecture (requires external proxy service)
- Adds complexity for minimal benefit (desktop works fine)
- User privacy concerns (proxy sees all crawled URLs)

### ❌ Implement Twice (Separate Web/Desktop Code)

**Rejected because:**
- `reqwest` and `scraper` work on both platforms already
- DRY principle - shared code is easier to maintain
- Future web support would require refactor

---

## References

- **Candle Rayon WASM PR:** https://github.com/huggingface/candle/pull/3063 (3x speedup)
- **COI Service Worker:** `public/coi-serviceworker.min.js`
- **wasm-bindgen-rayon:** https://github.com/GoogleChromeLabs/wasm-bindgen-rayon
- **COOP/COEP Explainer:** https://web.dev/coop-coep/
- **reqwest WASM support:** https://docs.rs/reqwest (automatic platform detection)
- **scraper WASM support:** https://docs.rs/scraper (pure Rust HTML5 parser)

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
8. **Web platform support** (via COI toggle or proxy)
