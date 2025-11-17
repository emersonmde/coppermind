# Ecosystem, Limitations & Resources

This document covers practical limitations, interesting implementations from the community, and resources about running ML models in browsers with Rust/WASM/WebGPU.

---

## Technology Stack Integration

### Candle + WASM

**Candle** is Hugging Face's Rust ML framework designed with WASM as a first-class target.

#### Key Design Decisions

**1. No `std::thread` in WASM (yet):**
```rust
// Desktop: Can use rayon for parallelism
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// WASM: Sequential or Web Workers
#[cfg(target_arch = "wasm32")]
// Must use Web Workers API via wasm-bindgen
```

**2. Device Selection:**
```rust
// Candle's device selection in WASM
let device = Device::cuda_if_available(0).unwrap();
// In browser: Always returns CPU device
// In future: Could return WebGPU device
```

**3. DType Limitations:**
```rust
// WASM: F32 only (F16 not well-supported)
let vb = VarBuilder::from_buffered_safetensors(
    model_bytes,
    DType::F32,  // Auto-converts F16 weights on load
    &device
)?;
```

#### Official Candle WASM Examples

**Repository:** [huggingface/candle](https://github.com/huggingface/candle)

**Example:** `candle-wasm-examples/`
- **Whisper:** Speech-to-text in browser
- **BERT:** Text classification
- **Segment Anything:** Image segmentation
- **LLaMA:** Large language model inference

**Pattern Used:**
```rust
// All examples use this pattern:
// 1. Load model via fetch
// 2. VarBuilder with DType::F32
// 3. Sequential inference (no threading)
// 4. Results via wasm-bindgen
```

**Key Insight from Examples:**
> "We use F32 everywhere in WASM because F16 operations aren't consistently fast across browsers"

### Dioxus + WASM

**Dioxus** is a React-like framework for Rust that compiles to WASM for web.

#### Asset Handling
```rust
// Compile-time asset registration
const MODEL: Asset = asset!("/assets/model.safetensors");

// Runtime: Dioxus handles base path rewriting
let url = MODEL.to_string();
// Production: "/myapp/assets/model.safetensors"
// Dev: "/assets/model.safetensors"
```

**Why This Matters:**
- GitHub Pages deploys to `/<repo-name>/`
- Dioxus `base_path` config handles this automatically
- No manual path manipulation needed

#### Async in Components
```rust
#[component]
fn MyComponent() -> Element {
    let mut result = use_signal(String::new);

    rsx! {
        button {
            onclick: move |_| {
                // spawn() for async tasks
                spawn(async move {
                    let data = expensive_async_work().await;
                    result.set(data);
                });
            },
            "Run Task"
        }
    }
}
```

**Signal Safety (Critical!):**
```rust
// ❌ DEADLOCK - holding Write across await
spawn(async move {
    let mut data = signal.write();  // Acquire write lock
    data.push(expensive().await);   // ❌ Lock held during await
});

// ✅ CORRECT - release lock before await
spawn(async move {
    let result = expensive().await;
    signal.write().push(result);  // Lock acquired after await
});
```

This is why `clippy.toml` has `await-holding-invalid-types` warnings.

### WebGPU + Rust

#### Current State (2024/2025)

**wgpu-rs:**
- ✅ Works great on desktop (Vulkan/Metal/DX12)
- ⚠️  WASM/WebGPU support is experimental
- ❌ Some features not available in browser

**Manual Approach (Used in Coppermind):**
```rust
// Use web-sys + js-sys directly
use web_sys::{gpu::*, ...};
use wasm_bindgen::prelude::*;

// Manually call WebGPU APIs
let gpu = Reflect::get(&navigator, &"gpu".into())?;
```

**Why Not wgpu-rs?**
1. Simpler for PoC (less abstraction)
2. Direct control over WebGPU features
3. Smaller WASM bundle size
4. Easier to debug (can see exact JS calls)

**When to Use wgpu-rs:**
- Desktop + Web cross-platform
- Need unified rendering pipeline
- Willing to handle platform differences

#### WebGPU Compute for ML

**Example: Matrix Multiplication**
```wgsl
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(8, 8)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    var sum = 0.0;
    for (var i = 0u; i < N; i++) {
        sum += a[row * N + i] * b[i * N + col];
    }
    result[row * N + col] = sum;
}
```

**Potential Speedup:** 10-100x for large matrix ops (varies by GPU)

---

## Known Limitations

### WASM Limitations

**1. No Native Threading**
```rust
// ❌ Doesn't work in WASM
use std::thread;
thread::spawn(|| { /* ... */ });

// ✅ Use Web Workers instead
spawn_worker(worker_code).await?;
```

**Status:** WebAssembly threading with Rust investigated and abandoned - Rust WebAssembly atomics are fundamentally incomplete and unsuitable for production use (see [ADR 003](adrs/003-wasm-threading-workaround.md) for full investigation). Web Workers remain the recommended approach for parallel computation in browsers.

**2. Memory Limit: 4GB (wasm32)**
- Fixed by architecture (32-bit pointers)
- Need Memory64 (wasm64) for >4GB
- Browser support: Chrome/Firefox experimental

**3. No SIMD Intrinsics (Limited)**
```rust
// Desktop: Can use packed_simd
#[cfg(not(target_arch = "wasm32"))]
use packed_simd::*;

// WASM: SIMD128 exists but limited support
// Candle doesn't use WASM SIMD yet
```

**4. Slower Than Native**
- ~1.5-3x slower than native code (varies by workload)
- JIT compilation overhead
- No access to some CPU features (AVX, etc.)

### Candle Limitations in WASM

**1. CPU-Only (For Now)**
```rust
// Always returns CPU device in WASM
let device = Device::cuda_if_available(0)?;
assert!(device.is_cpu());  // Always true in browser
```

**Future:** Candle WebGPU backend is planned but not ready yet.

**2. Large Model Size**
```
BERT-base:     440 MB (too large for quick loading)
BERT-small:    ~100 MB (feasible)
JinaBERT-v2:   262 MB (borderline, slow first load)
```

**Mitigation:**
- Use smaller models
- Quantization (F16/INT8)
- Progressive loading
- CDN hosting

**3. No Quantization in WASM**
```rust
// Desktop: Can use INT8/INT4 quantization
// WASM: F32 only (F16 converts to F32)
```

**Why:** Browser math libraries optimized for F32/F64, not INT8.

**4. Limited Model Support**
- BERT variants: ✅ Yes
- GPT/LLaMA: ✅ Yes (but slow)
- Diffusion models: ⚠️  Possible but very slow
- Large vision models: ❌ Too large/slow

### Dioxus WASM Limitations

**1. No Server Functions in Static Sites**
```rust
// ❌ Doesn't work on GitHub Pages
#[server]
async fn fetch_data() -> Result<String> { ... }

// ✅ Use direct fetch instead
let resp = window.fetch_with_str(url).await?;
```

**2. Routing on Static Hosts**
```rust
// GitHub Pages: Only supports hash routing
// base_path works, but refreshing /myapp/page → 404
// Need to use hash mode or configure server
```

**3. Bundle Size**
```
Minimal Dioxus app:     ~500 KB (gzipped)
With Candle + model:    ~63 MB (model) + ~2 MB (WASM)
Total first load:       ~65 MB
```

**Mitigation:** Lazy load model only when needed ✅ (already implemented)

### WebGPU Limitations

**1. Browser Support (2025)**
- ✅ Chrome 113+
- ✅ Edge 113+
- ⏳ Firefox (flag: `dom.webgpu.enabled`)
- ⏳ Safari (experimental)

**Coverage:** ~70% of desktop browsers, ~30% of mobile

**2. Fallback Required**
```rust
// Always need CPU fallback
match test_webgpu().await {
    Ok(_) => // Use GPU path
    Err(_) => // Use CPU path
}
```

**3. API Still Evolving**
- Spec is stable as of 2023
- But implementations still catching up
- Some features missing in browsers vs. spec

**4. Compute Shader Limits**
```wgsl
// Maximum workgroup size (varies by device)
@workgroup_size(256)  // Safe on most GPUs
@workgroup_size(1024) // May fail on some GPUs
```

Check `device.limits.maxComputeWorkgroupSizeX` at runtime.

---

## Interesting Community Implementations

### 1. Transformers.js (Xenova)

**Repository:** [@xenova/transformers](https://github.com/xenova/transformers.js)

**Language:** JavaScript/TypeScript + ONNX Runtime

**Approach:**
- ONNX Runtime compiled to WASM
- Pre-converted HuggingFace models to ONNX
- Multi-threading via Web Workers (auto-managed)
- WebGPU backend support

**Example:**
```javascript
import { pipeline } from '@xenova/transformers';

// Loads and runs in browser automatically
const embedder = await pipeline('feature-extraction',
    'Xenova/all-MiniLM-L6-v2');
const output = await embedder('Hello world');
```

**Comparison to Candle:**
| Feature | Transformers.js | Candle |
|---------|----------------|---------|
| Language | JS/TS | Rust |
| Runtime | ONNX | Native |
| Models | ONNX (pre-converted) | Safetensors/PyTorch |
| Threading | Auto (Web Workers) | Manual |
| WebGPU | ✅ Yes | ⏳ Planned |
| Size | ~2-5 MB (runtime) | ~1-2 MB (WASM) |

**When to Use:**
- **Transformers.js:** Quick prototyping, JS ecosystem, need WebGPU now
- **Candle:** Rust ecosystem, custom models, fine-grained control

### 2. ONNX Runtime Web

**Repository:** [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

**Backends:**
- WASM (CPU)
- WebGL (GPU - older)
- WebGPU (GPU - modern) ✅

**Example Use:**
```javascript
const session = await ort.InferenceSession.create('model.onnx', {
    executionProviders: ['webgpu', 'wasm']
});
```

**Advantage:** Mature, production-ready, good WebGPU support.

**Disadvantage:** Requires ONNX model format (conversion step).

### 3. Rust WASM LLMs

**Project:** [rustformers/llm](https://github.com/rustformers/llm)

**Models:** LLaMA, GPT-J, MPT, etc.

**WASM Status:** Experimental, slow for large models.

**Key Insight:**
> "Running LLaMA-7B in browser is possible but impractical (~30s per token). Better for small models (<1B params)."

### 4. TensorFlow.js

**Repository:** [tensorflow/tfjs](https://github.com/tensorflow/tfjs)

**Backends:**
- CPU (JS)
- WebGL (mature, good support)
- WebGPU (experimental)
- WASM (via XNNPACK)

**Advantage:** Huge ecosystem, many pre-trained models.

**Disadvantage:** Not Rust, different API paradigm.

### 5. Candle WASM Examples (Official)

**Repository:** [huggingface/candle](https://github.com/huggingface/candle/tree/main/candle-wasm-examples)

**Examples:**
- **whisper:** Speech recognition (3-4s per 30s audio on laptop)
- **segment-anything:** Image segmentation (~5-10s per image)
- **llama2:** Text generation (~2-3 tokens/sec on laptop)
- **bert:** Text embeddings (similar to our use case)

**Learning from BERT Example:**
```rust
// Key patterns:
// 1. F32 everywhere
// 2. Sequential inference (no threading)
// 3. Lazy model loading
// 4. Asset bundling via include_bytes! or fetch
```

---

## Blog Posts & Resources

### Must-Read Articles

**1. "Up to 4GB of memory in WebAssembly" (V8 Blog, 2020)**
- URL: https://v8.dev/blog/4gb-wasm-memory
- Key: Explains memory limit increase and why it matters

**2. "Making JavaScript Run Fast on WebAssembly" (Mozilla, 2019)**
- Explains JIT compilation in WASM
- Performance characteristics

**3. "WebGPU — All of the cores, none of the canvas" (Surma, 2023)**
- URL: https://surma.dev/things/webgpu/
- Excellent compute shader tutorial

**4. "Rust + WASM + ML: A Perfect Match?" (HuggingFace Blog, 2023)**
- Candle announcement and rationale
- Why Rust for browser ML

### Useful Documentation

**Candle:**
- Main repo: https://github.com/huggingface/candle
- Examples: https://github.com/huggingface/candle/tree/main/candle-wasm-examples
- Discord: Active community for questions

**Dioxus:**
- Docs: https://dioxuslabs.com/
- WASM guide: https://dioxuslabs.com/learn/0.5/reference/web
- Discord: Very responsive community

**WebGPU:**
- Spec: https://www.w3.org/TR/webgpu/
- MDN: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- WebGPU Fundamentals: https://webgpufundamentals.org/

**WASM:**
- Spec: https://webassembly.org/
- MDN: https://developer.mozilla.org/en-US/docs/WebAssembly
- Rust + WASM book: https://rustwasm.github.io/docs/book/

### Video Resources

**1. "Building ML Apps in the Browser with Rust" (RustConf 2023)**
- Covers Candle architecture
- WASM performance tips

**2. "WebGPU for Compute" (Chrome Dev Summit 2023)**
- Compute shader patterns
- Performance optimization

**3. "Dioxus 0.5: Full-Stack Rust" (Rust Nation UK 2024)**
- Latest Dioxus features
- WASM deployment patterns

### Community Projects Using Similar Stack

**1. Whisper Web (Speech-to-Text)**
- URL: https://huggingface.co/spaces/Xenova/whisper-web
- Stack: Transformers.js + ONNX Runtime
- Insight: Demonstrates progressive model loading

**2. Segment Anything WASM Demo**
- URL: https://huggingface.co/spaces/radames/segment-anything-webgpu
- Stack: ONNX Runtime + WebGPU
- Insight: Heavy model (~2GB) with WebGPU acceleration

**3. Photopea (Image Editor)**
- URL: https://www.photopea.com/
- Stack: Custom C++ compiled to WASM
- Insight: Proof that complex desktop apps can run in browser

**4. Figma (Design Tool)**
- Uses WASM for performance-critical rendering
- Multi-threaded via Web Workers + SharedArrayBuffer
- Insight: Production-scale WASM deployment

---

## Performance Comparison: Different Approaches

### Embedding 1000 Documents (512 tokens each)

| Approach | Time | Memory | Complexity |
|----------|------|---------|------------|
| **Candle WASM (CPU)** | ~15-30s | ~500MB | Medium |
| **Transformers.js (WASM)** | ~10-20s | ~400MB | Low |
| **ONNX Runtime (WebGPU)** | ~5-10s | ~600MB | Medium |
| **Native Python (CPU)** | ~3-5s | ~800MB | Low |
| **Native Python (GPU)** | ~1-2s | ~2GB | Low |

**Takeaway:** Browser inference is 3-10x slower than native, but "good enough" for many use cases.

### When Browser ML Makes Sense

✅ **Good Use Cases:**
- Privacy-sensitive applications (no data leaves device)
- Zero-installation requirement
- Offline-capable apps
- Edge inference (reduce server costs)
- Real-time user interaction

❌ **Poor Use Cases:**
- Batch processing (native is much faster)
- Very large models (>1GB)
- When GPU is required
- When fast iteration matters (WASM compile time)

---

## Future Directions

### 1. Candle WebGPU Backend (Planned)
```rust
// Future API (proposed):
let device = Device::webgpu()?;
let model = BertModel::new(vb, &config)?;
// Uses GPU automatically
```

**Expected:** 5-20x speedup for matrix operations.

### 2. ~~WASM Threads + SharedArrayBuffer~~
**Status:** ❌ Investigated and abandoned - Rust WebAssembly atomics are fundamentally broken and production-unviable.

After extensive investigation (see [ADR 003](adrs/003-wasm-threading-workaround.md)), determined that Rust WebAssembly threading is unsuitable for production use due to incomplete atomics implementation, toolchain incompatibilities, and browser limitations. Web Workers remain the recommended approach for parallel computation.

### 3. Quantization Support
```rust
// Future: INT8 inference in browser
let vb = VarBuilder::from_quantized_safetensors(
    model_bytes,
    DType::I8,  // 4x smaller than F32
    &device
)?;
```

**Expected:** 4x smaller models, 2-3x faster inference (if browser-optimized).

### 4. Progressive Model Streaming
```rust
// Future: Start inference before full model loaded
let model = ProgressiveModel::from_stream(stream).await?;
// Can run inference on early layers while late layers still loading
```

**Benefit:** Reduce perceived latency for large models.

---

## Debugging Tips

### 1. Monitor Memory Usage
```javascript
// In browser console:
performance.memory.usedJSHeapSize / 1048576  // MB
performance.memory.totalJSHeapSize / 1048576
```

### 2. Profile WASM Performance
```
Chrome DevTools → Performance → Record
Look for long "Evaluate Script" blocks (WASM execution)
```

### 3. WebGPU Debugging
```javascript
// Enable WebGPU validation errors
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredFeatures: [],
    requiredLimits: {},
    // Validation errors go to console
    label: "debug-device"
});
```

### 5. WASM Bundle Size Analysis
```bash
# Check WASM size
wasm-opt --version
wasm-opt -Oz input.wasm -o output.wasm  # Optimize for size

# Analyze what's in the bundle
twiggy top -n 20 output.wasm
```

---

## Contributing to the Ecosystem

### How to Help

**1. Report Browser Issues:**
- WebGPU bugs: https://bugs.chromium.org/
- WASM performance: https://bugzilla.mozilla.org/

**2. Contribute to Candle:**
- WASM examples
- WebGPU backend
- Documentation

**3. Share Your Findings:**
- Blog about your experience
- Open-source your implementations
- Join community discussions (Discord, GitHub)

**4. Benchmark & Document:**
- Real-world performance numbers
- Memory usage patterns
- Browser compatibility tables

---

## References

- [Candle GitHub](https://github.com/huggingface/candle)
- [Dioxus Docs](https://dioxuslabs.com/)
- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [WASM Spec](https://webassembly.org/specs/)
- [Transformers.js](https://github.com/xenova/transformers.js)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
- [Rust WASM Book](https://rustwasm.github.io/docs/book/)
