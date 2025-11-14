# Web Worker Implementation

**Date:** 2025-11-13
**Status:** Working for "Test Embedding" button; file upload has known issue

## Problem Statement

Running ML inference (JinaBERT embedding) on the main thread in web browsers causes the UI to freeze for 1-2+ seconds during model loading and inference. This creates a poor user experience.

## Solution Architecture

Implement a **Web Worker** that handles all ML operations on a separate thread, keeping the main UI thread completely responsive.

### High-Level Flow

```
Main Thread (Dioxus UI)
    ↓
    Send: { text, tokenizerUrl, modelUrl }
    ↓
Web Worker (Separate Thread)
    ├─ Download & cache tokenizer (~466KB, once)
    ├─ Download & cache model (~65MB, once)  ← 30-60s, UI stays responsive!
    ├─ Tokenize text (Rust WASM)
    ├─ Run inference (Candle ML)
    └─ Send back: { embedding: Vec<f32>, duration, tokenCount }
    ↓
Main Thread
    └─ Update UI with results
```

## Current Status

### Working Features

**Test Embedding Button:**
- Worker initialization using blob pattern
- WASM module loading in worker context
- Message passing via js_sys::Object
- Model download and caching in worker (65MB, 30-60s first run)
- Tokenization and inference on worker thread
- UI remains responsive during model download and inference

**Architecture:**
- Blob worker avoids asset hashing issues
- Simple interface: text input → embedding vector output
- All ML operations (tokenizer, model, inference) run on worker thread
- Results passed back to main thread via postMessage

### Known Issues

**File Upload:**
- Currently experiencing errors when using file upload path
- Test embedding button works as expected
- Investigation needed to align file upload code path with working implementation

## Implementation Details

### File Structure

```
src/
├── components.rs           # UI + worker coroutine (main thread side)
├── embedding.rs            # WasmEmbeddingModel, WasmTokenizer WASM bindings
└── lib.rs                  # WASM library exports

Created at runtime:
└── blob:<uuid>            # Dynamically generated worker code
```

### Key Code Patterns

#### 1. **Blob Worker Creation** (components.rs:200-309)

```rust
// Inject WASM path at runtime to avoid asset hashing issues
let wasm_path = "/coppermind/wasm/coppermind.js";

let worker_code = format!(r#"
    // Worker JavaScript code with {wasm_path} injected
    async function ensureWasm() {{ ... }}
    async function ensureTokenizer(url) {{ ... }}
    async function ensureModel(url) {{ ... }}
    self.onmessage = async (event) => {{ ... }}
"#, wasm_path = wasm_path);

// Create blob worker
let parts = Array::new();
parts.push(&JsValue::from_str(&worker_code));
let blob = Blob::new_with_str_sequence(&parts)?;
let blob_url = Url::create_object_url_with_blob(&blob)?;
let worker = Worker::new(&blob_url)?;
```

**Why blob worker?**
- Avoids MIME type issues with Service Worker
- Allows runtime injection of WASM path
- No separate `.js` file to manage (code is in Rust string)

#### 2. **Proper JS Interop** (components.rs:398-428)

```rust
// DON'T use serde_json + serde_wasm_bindgen (produces undefined)
// DO use js_sys::Object directly:

use js_sys::{Object, Reflect};

let msg = Object::new();
Reflect::set(&msg, &"id".into(), &JsValue::from(request_id))?;
Reflect::set(&msg, &"command".into(), &JsValue::from_str("embed_text"))?;

let data = Object::new();
Reflect::set(&data, &"text".into(), &JsValue::from_str(&text))?;
Reflect::set(&data, &"tokenizerUrl".into(), &JsValue::from_str(&tokenizer_url))?;
Reflect::set(&data, &"modelUrl".into(), &JsValue::from_str(&model_url))?;

Reflect::set(&msg, &"data".into(), &data)?;
worker.post_message(&msg)?;
```

**Why not serde_json?**
- `serde_wasm_bindgen::to_value(serde_json::Value)` produces invalid JS objects
- Worker receives `undefined` for all fields
- Direct `js_sys::Object` construction works correctly

#### 3. **WASM Bindings** (embedding.rs:330-366, lib.rs)

```rust
// Export tokenizer to WASM for worker use
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmTokenizer {
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl WasmTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(json_bytes: Vec<u8>) -> Result<WasmTokenizer, JsValue> { ... }

    #[wasm_bindgen(js_name = encode)]
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, JsValue> { ... }
}

// Already existed:
#[wasm_bindgen]
pub struct WasmEmbeddingModel { ... }
```

**Why expose tokenizer?**
- Worker needs to tokenize text
- Rust tokenizer is fast and accurate
- Avoids duplicating tokenization logic in JavaScript

#### 4. **Asset URL Management** (embedding.rs:46-52)

```rust
const MODEL_FILE: Asset = asset!("/assets/models/jina-bert.safetensors");
const TOKENIZER_FILE: Asset = asset!("/assets/models/jina-bert-tokenizer.json");

pub fn get_model_url() -> String {
    MODEL_FILE.to_string()  // Returns hashed URL: /coppermind/assets/models/jina-bert-dxhXXX.safetensors
}

pub fn get_tokenizer_url() -> String {
    TOKENIZER_FILE.to_string()
}
```

**Why helper functions?**
- `asset!()` macro handles file hashing
- URLs change on rebuild
- Worker needs actual URLs to fetch files

## Worker JavaScript Logic

### Message Handler (components.rs:264-306)

```javascript
self.onmessage = async (event) => {
    const { id, command, data } = event.data;

    if (command === 'embed_text') {
        const { text, tokenizerUrl, modelUrl } = data;

        // Ensure WASM, tokenizer, model loaded (cached after first use)
        const wasm = await ensureWasm();
        const tok = await ensureTokenizer(tokenizerUrl);
        const mdl = await ensureModel(modelUrl);

        // Tokenize text
        const tokenIds = tok.encode(text);

        // Run inference (CPU-intensive, but off main thread!)
        const startTime = performance.now();
        const embedding = mdl.embedTokens(tokenIds);
        const duration = performance.now() - startTime;

        // Send results back
        self.postMessage({
            id,
            success: true,
            result: {
                embedding: Array.from(embedding),
                duration,
                tokenCount: tokenIds.length
            }
        });
    }
};
```

### Caching Strategy

**First Run:**
1. Load WASM module (~3MB)
2. Download tokenizer (~466KB)
3. Download model (~65MB) ← 30-60 seconds
4. Run inference

**Subsequent Runs:**
1. ~~Load WASM~~ ← cached in `wasmModule` variable
2. ~~Download tokenizer~~ ← cached in `tokenizer` variable
3. ~~Download model~~ ← cached in `model` variable
4. Run inference ← instant!

## Performance Characteristics

### First Inference
- **Model Download:** 30-60 seconds (65MB over network)
- **Tokenization:** <10ms
- **Inference:** ~100-500ms (depends on text length)
- **UI Responsiveness:** 100% responsive throughout ✅

### Subsequent Inferences
- **Model Download:** 0ms (cached)
- **Tokenization:** <10ms
- **Inference:** ~100-500ms
- **UI Responsiveness:** 100% responsive ✅

## Cargo.toml Configuration

```toml
[lib]
crate-type = ["cdylib", "rlib"]  # Required for WASM worker

[dependencies]
web-sys = { version = "0.3", features = [
    "Worker", "WorkerOptions", "WorkerType",  # Worker creation
    "MessageEvent", "ErrorEvent",              # Worker communication
    "Blob", "Url",                             # Blob worker pattern
    # ... other features
] }
```

## Testing Procedure

### Manual UAT Steps

1. Run `dx serve`
2. Open browser to http://127.0.0.1:8080
3. Navigate to "🧪 Web Worker PoC Test" section
4. Click "Test Web Worker Embedding"

**Expected Console Output:**
```
INFO src/components.rs:195 🔧 Initializing Web Worker...
INFO src/components.rs:209 ✓ Worker created successfully
[Worker] JinaBERT worker initializing...
[Worker] Ready for messages
INFO src/components.rs:375 📤 Sending text to worker: 'This is a Web Worker test...'
[Worker] Received command: embed_text (id: 1)
[Worker] Loading WASM module from /coppermind/wasm/coppermind.js
[Worker] ✓ WASM ready
[Worker] Downloading tokenizer from /coppermind/assets/jina-bert-tokenizer-dxhXXX.json...
[Worker] Tokenizer downloaded: 455.28KB
[Worker] Creating tokenizer...
[Worker] ✓ Tokenizer ready
[Worker] Downloading model from /coppermind/assets/jina-bert-dxhXXX.safetensors...
[Worker] Model downloaded: 65.39MB
[Worker] Creating model...
[Worker] ✓ Model ready
[Worker] Tokenizing: "This is a Web Worker test for non-blocking ML..."
[Worker] Tokenized into 15 tokens
[Worker] Running inference...
[Worker] ✓ Embedding complete in 145.23ms, dim: 512
INFO src/components.rs:478 🎉 Worker test complete! Tokens: 15, Dim: 512, Duration: 145.23ms
```

**Expected UI Behavior:**
- During 30-60s model download: UI scrolls, buttons hover, clicks work ✅
- After completion: Shows embedding results with first 10 values

## Known Limitations & Future Work

### Current Limitations
1. **File upload path broken** - File upload triggers errors, needs debugging
2. **No progress indication** - User doesn't see download progress during first run
3. **Single worker instance** - Each component creates its own worker
4. **No batch processing** - Processes one text at a time
5. **No error recovery** - Worker errors require page reload

### Immediate Next Steps
1. **Debug file upload** - Identify why file upload path fails vs test button
2. **Align code paths** - Ensure file processing uses same worker mechanism as test button

### Future Enhancements
1. **Worker pool** - Reuse single worker across app
2. **Progress callbacks** - Report model download progress to UI
3. **Batch API** - Process multiple texts in one call
4. **Persistent cache** - Store model in IndexedDB/OPFS
5. **Desktop optimization** - Use `tokio::spawn_blocking` instead of worker

## Key Implementation Patterns

### Technical Decisions

**Blob Worker Pattern:**
- Avoids issues with static JavaScript files and asset hashing
- Allows runtime injection of WASM path
- All worker code in Rust string for easier maintenance

**Message Passing:**
- js_sys::Object works correctly for worker communication
- serde_json produces invalid objects when passed through serde_wasm_bindgen
- Direct object construction required for proper JavaScript interop

**Architecture Benefits:**
- Main thread isolated from ML implementation details
- Worker is optional, existing code paths unaffected
- Pattern can be adapted for desktop using `tokio::spawn_blocking`
- True parallelism prevents UI blocking during model operations

---

**Last Updated:** 2025-11-13
