# Candle Known Issues

This document details issues discovered while using Candle for embedding inference.

---

## Issue: JinaBERT F16 Inference Not Supported

### Summary

Candle's JinaBERT implementation hardcodes F32 for ALiBi positional bias, preventing F16 inference even when model weights are stored as F16.

### Environment

- **Candle version**: 0.9.x
- **Model**: JinaBERT (jinaai/jina-embeddings-v2-small-en)
- **Safetensors dtype**: F16

### Error

When loading model weights as F16 and running inference:

```
dtype mismatch in add, lhs: F16, rhs: F32
```

### Root Cause

In `candle-transformers/src/models/jina_bert.rs`:

```rust
// Line 9 - hardcoded dtype constant
pub const DTYPE: DType = DType::F32;

// Line 270 in build_alibi_bias() - forced F32 conversion
alibi_bias.to_dtype(DType::F32)?.broadcast_mul(&slopes)
```

The ALiBi positional bias is always created as F32, regardless of model weight dtype. When model weights are loaded as F16, the `add` operation during forward pass fails due to dtype mismatch.

### Impact

- Cannot use F16 inference on Metal (which is faster than F32 on Apple Silicon)
- Cannot use BF16 inference on CUDA
- Must convert F16 weights to F32 at load time

### Workaround

Load model weights as F32 (Candle auto-converts from safetensors F16):

```rust
let vb = VarBuilder::from_buffered_safetensors(
    model_bytes,
    DType::F32,  // Force F32, even though safetensors has F16
    &device
)?;
```

The F16→F32 conversion happens at load time (once), not inference time, so performance impact is minimal. GPU acceleration still works for all matrix operations.

### Potential Fix

The `jina_bert.rs` module would need to:
1. Accept dtype as a parameter instead of using `DTYPE` constant
2. Create ALiBi bias in the same dtype as model weights
3. Or cast the bias to match weights before `add` operation

### References

- Source: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/jina_bert.rs
- Related: F16 generally faster than BF16 on Apple Silicon ([Stack Overflow](https://stackoverflow.com/questions/79167526/why-do-bf16-models-have-slower-inference-on-mac-m-series-chips-compared-to-f16-m))

---

## Issue: Metal Command Buffer Threading

This section details a Metal command buffer threading issue discovered while using Candle for embedding inference on macOS.

## Summary

When using Candle's Metal backend from multiple threads (e.g., tokio's `spawn_blocking` thread pool), the application crashes with either:

1. **Candle 0.8.x / 0.9.1**: Command encoder assertion failure
2. **Candle 0.9.2-alpha.1**: SIGSEGV in `allocate_zeros`

## Environment

- **Platform**: macOS 14.3 (Sonoma), Apple Silicon (M-series)
- **Rust**: 1.82+
- **Candle versions tested**: 0.8.4, 0.9.1, 0.9.2-alpha.1
- **Use case**: Embedding inference with JinaBERT from async Rust (tokio)

## Issue 1: Command Encoder Assertion (0.8.x / 0.9.1)

### Error Message

```
-[AGXG15XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]:1015:
failed assertion `A command encoder is already encoding to this command buffer'
```

### Stack Trace

The crash occurs in Apple's Metal GPU driver when multiple threads attempt to encode commands to the same command buffer simultaneously.

### Reproduction

```rust
use candle_core::{Device, Tensor};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Create Metal device (shared across threads)
    let device = Device::new_metal(0).expect("Metal device");
    let device = Arc::new(device);

    // Spawn multiple concurrent embedding tasks
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let device = device.clone();
            tokio::task::spawn_blocking(move || {
                // Each thread tries to create tensors on the same Metal device
                // This causes command buffer race condition
                let tensor = Tensor::zeros((1, 512), candle_core::DType::F32, &device)
                    .expect("create tensor");

                // Any operation that uses the command buffer can trigger the crash
                let _result = tensor.sum_all();
            })
        })
        .collect();

    // One or more of these will crash with the assertion failure
    for handle in handles {
        handle.await.unwrap();
    }
}
```

### Root Cause

Metal's `MTLCommandBuffer` is **not thread-safe** - only one CPU thread can encode commands to a command buffer at a time. In Candle 0.8.x/0.9.1, the Metal backend shares a single command buffer across all threads, causing race conditions when multiple threads perform tensor operations concurrently.

From Apple's Metal documentation:
> "A single command buffer can have only one active command encoder at a time."

## Issue 2: SIGSEGV in allocate_zeros (0.9.2-alpha.1)

### Error Message

```
fatal runtime error: Rust cannot catch foreign exceptions, aborting
```

Or in crash reports:
```
Exception Type:        EXC_BAD_ACCESS (SIGSEGV)
Exception Codes:       KERN_INVALID_ADDRESS at 0x0000000000000004
```

### Stack Trace

```
Thread 28 Crashed:: tokio-runtime-worker
0   AGXMetalG15X_B0    endComputePass + 344
1   AGXMetalG15X_B0    -[AGXG15XFamilyComputeContext deferredEndEncoding] + 176
2   AGXMetalG15X_B0    -[AGXG15XFamilyCommandBuffer commitEncoder] + 96
3   AGXMetalG15X_B0    -[AGXG15XFamilyCommandBuffer blitCommandEncoderCommon:] + 360
4   coppermind         metal::commandbuffer::CommandBufferRef::new_blit_command_encoder
5   coppermind         candle_core::metal_backend::device::MetalDevice::allocate_zeros
6   coppermind         candle_core::tensor::Tensor::zeros
7   coppermind         BertEmbeddings::forward
8   coppermind         BertModel::forward
9   coppermind         JinaBertEmbedder::embed_tokens
```

### Root Cause

Candle 0.9.2-alpha.1 includes a major Metal refactor ([PR #3070](https://github.com/huggingface/candle/pull/3070)) that introduced a pool-based command buffer management system. The crash at address `0x4` (null pointer dereference) suggests a race condition in the pool selection logic when accessed from multiple tokio `spawn_blocking` threads.

The crash occurs during `allocate_zeros` when trying to create a blit command encoder, indicating the selected command buffer is in an invalid state.

## Version Comparison

| Version | Thread Safety Fix | Metal Refactor | Status |
|---------|------------------|----------------|--------|
| 0.8.4 | No | No | Crashes: command encoder assertion |
| 0.9.1 | No | No | Crashes: command encoder assertion |
| 0.9.2-alpha.1 | Yes (PR #3079) | Yes (PR #3070) | Crashes: SIGSEGV in allocate_zeros |

### Relevant PRs

- [PR #3079](https://github.com/huggingface/candle/pull/3079): "[Metal] Ensure tensors are send/sync" - Added thread-isolated command buffers
- [PR #3070](https://github.com/huggingface/candle/pull/3070): "[Metal] Refactor" - Major restructure of candle-metal-kernels
- [PR #3093](https://github.com/huggingface/candle/pull/3093): "[Metal] Buffer improvements" - Changed to StorageModeShared

## Expected Behavior

Metal tensors should be safely usable from multiple threads when the device is wrapped in `Arc`. The thread-safety fix in PR #3079 aimed to achieve this by isolating command buffers per thread.

## Actual Behavior

- **0.8.x/0.9.1**: No thread isolation, command buffer race condition causes immediate crash
- **0.9.2-alpha.1**: Thread isolation implemented but pool-based system has regression causing SIGSEGV

## Workarounds

### 1. Semaphore Serialization (Current)

Serialize all GPU access with a semaphore:

```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

static GPU_SEMAPHORE: once_cell::sync::Lazy<Arc<Semaphore>> =
    once_cell::sync::Lazy::new(|| Arc::new(Semaphore::new(1)));

pub async fn embed_tokens_safe(model: Arc<Model>, tokens: Vec<u32>) -> Result<Vec<f32>> {
    let _permit = GPU_SEMAPHORE.acquire().await?;

    tokio::task::spawn_blocking(move || {
        model.embed_tokens(tokens)
    }).await?
}
```

**Downside**: No parallel GPU utilization.

### 2. Dedicated Worker Thread (Recommended)

Create a single thread that owns the Metal device and process requests via channels:

```rust
use std::sync::mpsc;
use tokio::sync::oneshot;

pub enum EmbedRequest {
    Embed {
        tokens: Vec<u32>,
        response: oneshot::Sender<Result<Vec<f32>, String>>,
    },
    Shutdown,
}

pub struct MetalWorker {
    tx: mpsc::Sender<EmbedRequest>,
}

impl MetalWorker {
    pub fn new(model_bytes: Vec<u8>) -> Self {
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            // All Metal operations happen on this single thread
            let device = Device::new_metal(0).unwrap();
            let model = load_model(model_bytes, &device).unwrap();

            while let Ok(request) = rx.recv() {
                match request {
                    EmbedRequest::Embed { tokens, response } => {
                        let result = model.embed_tokens(tokens);
                        let _ = response.send(result);
                    }
                    EmbedRequest::Shutdown => break,
                }
            }
        });

        MetalWorker { tx }
    }

    pub async fn embed(&self, tokens: Vec<u32>) -> Result<Vec<f32>, String> {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx.send(EmbedRequest::Embed {
            tokens,
            response: response_tx
        }).map_err(|e| e.to_string())?;

        response_rx.await.map_err(|e| e.to_string())?
    }
}
```

**Benefit**: Metal GPU acceleration works, no race conditions, future-proof for multiple models.

### 3. Batch Processing

Instead of concurrent individual embeddings, batch multiple texts in a single call:

```rust
// Instead of spawning multiple concurrent embed_tokens calls:
let embeddings = model.embed_batch_tokens(vec![
    tokens1,
    tokens2,
    tokens3,
    tokens4,
])?;
```

**Benefit**: GPU processes batch in parallel internally, single command buffer used.

## Suggested Fix

The pool-based command buffer system in 0.9.2-alpha.1 (`candle-metal-kernels/src/metal/commands.rs`) appears to have a race condition in `select_entry()` or related pool management when accessed from tokio's `spawn_blocking` threads.

Potential areas to investigate:

1. **Pool entry selection**: The two-phase selection algorithm may return an entry in an invalid state
2. **Command buffer lifecycle**: Buffers may be recycled while still in use
3. **Thread ID handling**: tokio's blocking pool creates/destroys threads dynamically

## References

- [Candle Issue #2637](https://github.com/huggingface/candle/issues/2637) - Original thread safety issue
- [Apple Metal Threading Guide](https://developer.apple.com/documentation/metal/mtlcommandqueue) - Command queue is thread-safe, command buffer is not
- [Metal Best Practices](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Cmd-Submiss/Cmd-Submiss.html) - One encoder per command buffer at a time
