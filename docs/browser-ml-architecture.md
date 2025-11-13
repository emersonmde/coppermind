# Browser-Based ML Architecture

This document describes the architecture patterns for running ML inference entirely in the browser using WebAssembly, Web Workers, and WebGPU.

## Overview

Coppermind demonstrates a **zero-installation ML inference** approach where users can run computationally expensive tasks (embedding models, parallel processing) entirely in the browser without installing any software.

### Key Technologies
- **WebAssembly (WASM):** Near-native performance for ML inference
- **Candle:** Rust ML framework with WASM support
- **Web Workers:** Parallel CPU computation
- **WebGPU:** GPU-accelerated compute shaders
- **Cross-Origin Isolation:** Enables SharedArrayBuffer and high-precision timers

---

## Cross-Origin Isolation (COOP/COEP)

### Why It's Required

After the Spectre vulnerability was discovered, browsers restricted access to certain powerful APIs that could be used for timing attacks:
- **SharedArrayBuffer** - Shared memory between threads
- **WebAssembly Threads** - Multi-threaded WASM
- **High-precision timers** - Sub-millisecond timing

To re-enable these features, sites must implement **Cross-Origin Isolation** via HTTP headers.

### Required Headers

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

**COOP (Cross-Origin-Opener-Policy):**
- Isolates your document from other origins' windows
- Prevents cross-origin windows from accessing your global object

**COEP (Cross-Origin-Embedder-Policy):**
- Requires all resources (images, scripts, styles) to explicitly opt-in via CORS or CORP headers
- Ensures no cross-origin resources can read your data

### Service Worker Solution

**Problem:** GitHub Pages and many static hosts don't allow custom HTTP headers.

**Solution:** Use a Service Worker to inject headers at runtime.

#### Implementation

**1. Service Worker (`public/coi-serviceworker.min.js`):**
```javascript
// Intercepts all requests and injects COOP/COEP headers
// Source: https://github.com/gzuidhof/coi-serviceworker
self.addEventListener('fetch', function(event) {
  // ... intercepts and adds headers to responses
});
```

**2. Load Service Worker in App (`src/main.rs`):**
```rust
#[component]
fn App() -> Element {
    rsx! {
        // Configure COI service worker
        document::Script {
            r#"window.coi = {{ coepCredentialless: true, quiet: false }};"#
        }
        // Load service worker from public/ directory
        document::Script { src: "/coppermind/assets/coi-serviceworker.min.js" }

        // ... rest of app
    }
}
```

**3. Verification:**
```javascript
// In browser console:
console.log(crossOriginIsolated); // Should be true
```

### Benefits Unlocked

✅ **SharedArrayBuffer:** Share memory between main thread and workers (future use)
✅ **WASM Threads:** Multi-threaded ML inference (future use)
✅ **High-Precision Timers:** Accurate performance measurements
✅ **Works on Static Hosts:** No server configuration needed

---

## Web Workers for Parallel Processing

### Architecture

```
Main Thread (Dioxus UI)
    │
    ├─> Worker 1 (CPU task)
    ├─> Worker 2 (CPU task)
    ├─> Worker 3 (CPU task)
    └─> Worker N (CPU task)
```

### Implementation (`src/cpu.rs`)

**Key Pattern: Inline Worker Code**
```rust
pub async fn spawn_worker(worker_id: usize, text: &str, iterations: usize)
    -> Result<String, String>
{
    // 1. Define worker code as inline JavaScript string
    let worker_code = r#"
        self.onmessage = function(e) {
            const { workerId, iterations } = e.data;
            // Perform computation
            // ...
            self.postMessage({ workerId, result });
        };
    "#;

    // 2. Create Blob URL from code (avoids external .js file)
    let blob = Blob::new_with_str_sequence(&parts)?;
    let url = Url::create_object_url_with_blob(&blob)?;
    let worker = Worker::new(&url)?;

    // 3. Set up message handler with oneshot channel
    let (sender, receiver) = futures_channel::oneshot::channel();
    let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
        // Extract result and send to receiver
        // ...
    }));
    worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

    // 4. Send message to worker and await result
    worker.post_message(&msg)?;
    receiver.await
}
```

**Advantages:**
- No external `.js` files needed
- Worker code stays in Rust codebase
- Easy to deploy (single WASM bundle)
- No CORS issues with worker scripts

### Usage Pattern (from `components.rs`)

```rust
// Spawn 16 workers in parallel
let num_workers = 16;
for i in 0..num_workers {
    spawn({
        let mut results = cpu_results;
        async move {
            match spawn_worker(i, "test", 10000).await {
                Ok(result) => {
                    results.write().push(result);
                }
                Err(e) => { /* handle error */ }
            }
        }
    });
}
```

---

## WebGPU Compute Shaders

### Architecture

```
Main Thread
    │
    └─> WebGPU Device
            │
            ├─> Compute Pipeline
            ├─> Shader Module (WGSL)
            ├─> Buffer (Input/Output)
            └─> Command Queue
```

### Implementation (`src/wgpu.rs`)

**Pattern: Manual WebGPU via JS Bindings**

Since `wgpu-rs` doesn't fully support WASM/WebGPU yet, we use manual JS bindings:

```rust
pub async fn test_webgpu() -> Result<String, String> {
    // 1. Get GPU adapter
    let navigator = web_sys::window()?.navigator();
    let gpu = Reflect::get(&navigator, &"gpu".into())?;
    let adapter = request_adapter(&gpu).await?;

    // 2. Get GPU device
    let device = request_device(&adapter).await?;

    // 3. Create compute shader (WGSL)
    let shader_code = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            // Compute operations
            data[idx] = /* result */;
        }
    "#;
    let shader_module = create_shader_module(&device, shader_code)?;

    // 4. Create buffer, pipeline, submit work
    // ...
}
```

**Advantages:**
- Direct access to WebGPU features
- Full control over compute pipeline
- Can optimize for specific use cases

**Future Use Cases:**
- Matrix multiplication for embeddings
- Batch inference acceleration
- Vector similarity search (GPU-accelerated)

---

## ML Inference Architecture

### Current: CPU-Only Inference with Candle

```
User Input
    ↓
Tokenizer (tokenizers-rs)
    ↓ token IDs
Candle BertModel (WASM)
    ↓ hidden states
Mean Pooling
    ↓ pooled embeddings
L2 Normalization
    ↓
512-dim embedding vector
```

### Model Loading Pattern

**Key Insight: Singleton Model with Lazy Loading**

```rust
// Global tokenizer (OnceCell for one-time initialization)
static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();

// Thread-local model cache (Rc for shared ownership)
thread_local! {
    static MODEL_CACHE: RefCell<Option<Rc<EmbeddingModel>>> =
        const { RefCell::new(None) };
}

async fn get_or_load_model() -> Result<Rc<EmbeddingModel>, String> {
    // Check cache first
    if let Some(existing) = MODEL_CACHE.with(|cell| cell.borrow().clone()) {
        return Ok(existing);
    }

    // Cold start: load model from assets
    let model_bytes = fetch_asset_bytes(&model_url).await?;
    let model = EmbeddingModel::from_bytes(model_bytes, vocab_size, config)?;

    // Cache for future use
    let model = Rc::new(model);
    MODEL_CACHE.with(|cell| cell.borrow_mut().replace(model.clone()));
    Ok(model)
}
```

**Why This Pattern?**
- **OnceCell:** Thread-safe, one-time initialization for tokenizer (immutable, shareable)
- **thread_local + Rc:** WASM is single-threaded, so thread_local ensures model stays on UI thread
- **Lazy loading:** Model only loads when first needed (262MB download + initialization)
- **Caching:** Subsequent inferences reuse loaded model

### Asset Loading via Dioxus

```rust
// Compile-time asset registration
const MODEL_FILE: Asset = asset!("/assets/models/jina-bert.safetensors");
const TOKENIZER_FILE: Asset = asset!("/assets/models/jina-bert-tokenizer.json");

// Runtime URL resolution (handles base path automatically)
let model_url = MODEL_FILE.to_string();  // "/coppermind/assets/models/..."
```

**Advantages:**
- Assets bundled with app (offline-capable)
- Dioxus handles base path rewriting
- Works with GitHub Pages deployment

---

## Future Architecture: Worker Pool for Embeddings

From `project_plan.md`, the long-term vision:

```
Main Thread (Dioxus UI)
    ↓ file selection
Streaming Reader (wasm_streams)
    ↓ byte chunks
Chunker (main thread or dedicated worker)
    ↓ text chunks (zero-copy transfer)
Worker Pool (N embedding workers)
    ↓ embeddings (Float32Array)
Coordinator/Writer
    ↓ batch writes
Vector Store (SQLite on OPFS)
```

### Benefits
- **Parallel embedding:** Process multiple chunks simultaneously
- **Non-blocking UI:** Embeddings happen in background
- **Streaming:** Process large files without loading entirely in memory
- **OPFS storage:** Persistent vector database in browser

### Challenges
- **Model duplication:** Each worker needs its own model instance (262MB × N workers)
- **Memory pressure:** May need to limit worker pool size
- **SharedArrayBuffer:** Would need COOP/COEP (already implemented ✅)

### Alternative: Sequential with Batching
```rust
// Current approach (sequential batching)
pub fn embed_batch_tokens(&self, batch: Vec<Vec<u32>>)
    -> Result<Vec<Vec<f32>>, String>
{
    // Single model instance, batch inference
    // More memory efficient than N workers
}
```

**Tradeoffs:**
- Sequential batching: Less memory, still reasonably fast
- Worker pool: More parallelism, higher memory cost

---

## Performance Characteristics

### Cold Start (First Load)
```
Download model (262MB):       ~2-5s (varies by connection)
Download tokenizer (695KB):   ~100-500ms
Model initialization:         ~500ms-1s
Total cold start:             ~3-7s
```

### Warm Start (Model Cached)
```
Model already in memory:      0ms (instant)
Tokenization:                 ~1-10ms (varies by text length)
Inference (single):           ~50-200ms (varies by sequence length)
Inference (batch of 10):      ~200-500ms
```

### Memory Footprint
```
Model weights (F32):          ~250MB
Tokenizer:                    ~5MB
ALiBi bias (2048 tokens):     ~128MB
Runtime overhead:             ~100-200MB
Total:                        ~500-600MB
```

---

## Browser Compatibility

### WASM Support
✅ Chrome 57+
✅ Firefox 52+
✅ Safari 11+
✅ Edge 16+

### WASM 4GB Memory
✅ Chrome 83+ (2020)
✅ Firefox 79+ (2020)
✅ Safari 14+ (2020)

### WebGPU Support
✅ Chrome 113+ (2023)
✅ Edge 113+ (2023)
✅ Safari 18+ (September 2024)
⏳ Firefox (experimental, behind `dom.webgpu.enabled` flag)

### Cross-Origin Isolation
✅ All modern browsers (with Service Worker workaround)

---

## Deployment Considerations

### GitHub Pages
- ✅ Static hosting (free)
- ✅ HTTPS by default (required for Service Workers)
- ✅ Service Worker solution works
- ❌ No custom HTTP headers (COOP/COEP)
- ⚠️  Large model files (262MB) may slow initial load

### Optimizations for Production

**1. Model Compression:**
```rust
// Current: F32 weights (~250MB)
// Future: F16 or INT8 quantization (~125MB or ~62MB)
```

**2. Lazy Model Loading:**
```rust
// Don't load model until user actually needs it
// Current: Load on first embedding request ✅
```

**3. Progressive Model Streaming:**
```rust
// Stream model weights in chunks instead of waiting for full download
// Requires custom safetensors loader
```

**4. CDN for Model Files:**
```
// Host large model files on CDN instead of GitHub Pages
// Faster downloads, reduced repository size
```

---

## Security Considerations

### Cross-Origin Isolation
✅ Prevents Spectre-like timing attacks
✅ Isolates application from other origins
✅ Required for SharedArrayBuffer

### Model Integrity
⚠️  Models loaded from assets/ directory (trusted)
⚠️  Consider adding SHA-256 checksum verification
⚠️  Safetensors format prevents arbitrary code execution

### User Data Privacy
✅ All inference happens client-side (no data sent to server)
✅ No API keys or external services required
✅ Embeddings never leave user's browser

---

## References

1. [MDN: Cross-Origin Isolation](https://developer.mozilla.org/en-US/docs/Web/API/crossOriginIsolated)
2. [COI Service Worker Library](https://github.com/gzuidhof/coi-serviceworker)
3. [WebGPU Specification](https://www.w3.org/TR/webgpu/)
4. [Candle ML Framework](https://github.com/huggingface/candle)
5. [WebAssembly Threads Proposal](https://github.com/WebAssembly/threads)
6. [Spectre Attack Paper](https://spectreattack.com/)
