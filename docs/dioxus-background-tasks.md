# Dioxus 0.7 Background Task Mechanisms

Comprehensive research on all methods for running background tasks without blocking UI rendering in Dioxus 0.7.

**Research Date:** 2025-11-13
**Dioxus Version:** 0.7.x
**Source:** Official documentation at https://dioxuslabs.com/learn/0.7/ and docs.rs

---

## Table of Contents

1. [Core Async Primitives](#core-async-primitives)
2. [Reactive Hooks](#reactive-hooks)
3. [Task Management Hooks](#task-management-hooks)
4. [Platform-Specific Threading](#platform-specific-threading)
5. [Web Workers & WASM Threading](#web-workers--wasm-threading)
6. [Comparison Matrix](#comparison-matrix)
7. [Recommendations for ML Inference](#recommendations-for-ml-inference)

---

## Core Async Primitives

### 1. `spawn()`

**Purpose:** Spawn a future in the background that automatically cancels when the component unmounts.

**Signature:**
```rust
pub fn spawn(fut: impl Future<Output = ()> + 'static) -> Task
```

**Key Characteristics:**
- Returns a `Task` handle for control (cancel, pause)
- Future is tied to component lifecycle
- Automatically cancelled on component drop
- Best for event-driven async operations

**When to Use:**
- One-off async operations in event handlers
- Background tasks that should stop when component unmounts
- Network requests, file I/O, or other async operations

**Platform Compatibility:** ✅ Web (WASM), ✅ Desktop, ✅ Mobile

**Code Example:**
```rust
use dioxus::prelude::*;

fn App() -> Element {
    rsx! {
        button {
            onclick: move |_| {
                spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    println!("Hello World");
                });
            },
            "Print hello in one second"
        }
    }
}
```

**Gotchas:**
- Creates a NEW future each call - don't call on every render
- Futures are lazy - won't execute until spawned
- Never hold locks across `.await` points
- Blocking operations will freeze UI - use separate threads instead

---

### 2. `spawn_forever()`

**Purpose:** Spawn a future that persists for the entire application lifetime, not tied to any component.

**Signature:**
```rust
pub fn spawn_forever(fut: impl Future<Output = ()> + 'static)
```

**Key Characteristics:**
- Attaches to root component scope
- Survives component unmounting
- Future runs until app shuts down
- No automatic cleanup

**When to Use:**
- Background sync engines
- Long-lived system I/O listeners
- Global event loops
- Tasks that should never stop

**Platform Compatibility:** ✅ Web (WASM), ✅ Desktop, ✅ Mobile

**Code Example:**
```rust
use dioxus::prelude::*;

fn App() -> Element {
    use_hook(|| {
        spawn_forever(async move {
            loop {
                // Background sync every 5 minutes
                tokio::time::sleep(Duration::from_secs(300)).await;
                sync_data().await;
            }
        });
    });

    rsx! { /* ... */ }
}
```

**Gotchas:**
- ⚠️ Signals used in `spawn_forever` must remain valid for entire app lifecycle
- Using dropped signals will cause panics
- No automatic cleanup - use with `use_hook` to prevent re-spawning on re-renders
- Memory leaks if not careful with resource management

---

### 3. `spawn_isomorphic()`

**Purpose:** Spawn tasks that run during suspense and work identically on server and client (for fullstack apps).

**Signature:**
```rust
pub fn spawn_isomorphic(fut: impl Future<Output = ()> + 'static)
```

**Key Characteristics:**
- Runs during suspense resolution
- Must be deterministic across server/client
- No async I/O (causes hydration issues)
- For logging, state tracking, not data fetching

**When to Use:**
- ✅ Logging or analytics
- ✅ Responding to state changes
- ❌ NOT for API requests
- ❌ NOT for platform-specific I/O

**Platform Compatibility:** ✅ Web (SSR), ✅ Desktop, ⚠️ Fullstack only

**Code Example:**
```rust
// ✅ Good - deterministic logging
spawn_isomorphic(async move {
    log::info!("User action: {}", action);
});

// ❌ Bad - causes hydration issues
spawn_isomorphic(async move {
    let data = fetch_api().await; // DON'T DO THIS
});
```

**Gotchas:**
- ⚠️ Use regular `spawn` instead unless you specifically need suspense integration
- Async I/O will cause different results on server vs client
- Hydration mismatches are hard to debug

---

## Reactive Hooks

### 4. `use_resource()`

**Purpose:** Reactive hook that spawns a future and returns its result, automatically re-running when dependencies change.

**Signature:**
```rust
pub fn use_resource<T, F>(future: impl FnMut() -> F + 'static) -> Resource<T>
where
    T: 'static,
    F: Future<Output = T> + 'static,
```

**Key Characteristics:**
- Returns `Resource<T>` wrapping `Option<T>`
- Automatically tracks signal dependencies
- Cancels and restarts on dependency changes
- Integrates with Suspense boundaries

**When to Use:**
- Data fetching with reactive dependencies
- API calls that depend on props/signals
- Async operations with return values
- Loading states with Suspense

**Platform Compatibility:** ✅ Web (WASM), ✅ Desktop, ✅ Mobile

**Code Example:**
```rust
use dioxus::prelude::*;

#[component]
fn UserProfile(user_id: u32) -> Element {
    // Automatically re-fetches when user_id changes
    let user_data = use_resource(move || async move {
        fetch_user(user_id).await
    });

    rsx! {
        match user_data.value() {
            Some(Ok(user)) => rsx! { "User: {user.name}" },
            Some(Err(e)) => rsx! { "Error: {e}" },
            None => rsx! { "Loading..." }
        }
    }
}
```

**Resource Methods:**
```rust
resource.value()      // ReadSignal<Option<T>> - current value
resource.suspend()?   // MappedSignal<T> - pauses until ready
resource.restart()    // Cancel and restart the future
resource.cancel()     // Force cancel
resource.pause()      // Pause execution
resource.resume()     // Resume execution
resource.pending()    // bool - is running?
resource.finished()   // bool - is done?
resource.state()      // UseResourceState (Pending/Done/Stopped)
```

**Non-Reactive Dependencies:**
```rust
// Use use_reactive! for non-signal values
let result = use_resource(use_reactive!(|(count,)| async move {
    fetch_data(count).await
}));
```

**Gotchas:**
- Resource restarts cancel the current future - handle cancellation gracefully
- Only signals are automatically tracked - use `use_reactive!` for other deps
- Must follow hook rules (no conditionals, loops, etc.)
- ⚠️ Does NOT run on server in fullstack apps - use server functions instead

---

### 5. `use_future()`

**Purpose:** Spawn a future once on component mount, without returning a value.

**Signature:**
```rust
pub fn use_future<F>(future: impl FnMut() -> F + 'static) -> UseFuture
where F: Future + 'static
```

**Key Characteristics:**
- Runs once on initial render
- No return value (use `use_resource` if you need results)
- No automatic reactivity to dependency changes
- Does NOT run on server in fullstack apps

**When to Use:**
- Side effects on component mount
- Fire-and-forget async operations
- Initialization tasks without results

**Platform Compatibility:** ✅ Web (WASM), ✅ Desktop, ❌ Server (use spawn_isomorphic)

**Code Example:**
```rust
use dioxus::prelude::*;

#[component]
fn AutoCounter() -> Element {
    let mut count = use_signal(|| 0);
    let running = use_signal(|| true);

    use_future(move || async move {
        loop {
            if running() {
                count += 1;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
    });

    rsx! {
        button { onclick: move |_| running.toggle(), "Toggle" }
        "Count: {count}"
    }
}
```

**Gotchas:**
- No reactivity - won't restart when signals change
- No return value - use `use_resource` if you need results
- ⚠️ Server-only code will cause issues - use conditional compilation

---

### 6. `use_effect()`

**Purpose:** Reactive side effects that run AFTER component rendering completes.

**Signature:**
```rust
pub fn use_effect(callback: impl FnMut() + 'static) -> Effect
```

**Key Characteristics:**
- Executes AFTER render (post-render timing)
- Automatically tracks signal dependencies
- Reruns when any read signal changes
- Can spawn async tasks inside

**When to Use:**
- DOM manipulation (web-sys, JavaScript)
- Reading rendered element properties
- Side effects based on state changes
- Triggering analytics/logging

**Platform Compatibility:** ✅ Web (WASM), ✅ Desktop, ✅ Mobile

**Code Example:**
```rust
use dioxus::prelude::*;

#[component]
fn TrackedComponent() -> Element {
    let mut count = use_signal(|| 0);

    // Logs every time count changes
    use_effect(move || {
        log::info!("Count changed to: {}", count());
    });

    rsx! {
        button { onclick: move |_| count += 1, "Increment" }
        "Count: {count}"
    }
}
```

**Spawning Async in Effects:**
```rust
use_effect(move || {
    spawn(async move {
        let result = fetch_data().await;
        data_signal.set(result);
    });
});
```

**Gotchas:**
- Effects run AFTER render - DOM is already updated
- If component returns early, effect won't run
- Don't use for derived state - use `use_memo` instead
- Can cause infinite loops if you write to signals you read

---

### 7. `use_memo()`

**Purpose:** Efficiently compute derived values that update when dependencies change.

**Signature:**
```rust
pub fn use_memo<R>(f: impl FnMut() -> R + 'static) -> Memo<R>
where R: PartialEq + 'static,
```

**Key Characteristics:**
- Runs immediately on first call
- Automatically tracks signal dependencies
- Reruns when any read signal changes
- Only triggers dependents if output changes (via `PartialEq`)

**When to Use:**
- Computing derived data from signals
- Expensive calculations that should cache
- Breaking reactivity chains (only update if result differs)
- Transforming signal values

**Platform Compatibility:** ✅ Web (WASM), ✅ Desktop, ✅ Mobile

**Code Example:**
```rust
use dioxus::prelude::*;

#[component]
fn Calculator() -> Element {
    let mut count = use_signal(|| 5);

    // Only recalculates when count changes
    let double = use_memo(move || count() * 2);
    let square = use_memo(move || count() * count());

    rsx! {
        "Count: {count}, Double: {double}, Square: {square}"
        button { onclick: move |_| count += 1, "Increment" }
    }
}
```

**Gotchas:**
- The closure reruns on EVERY signal write, but dependents only update if output differs
- Requires `PartialEq` - can't use with non-comparable types
- Cannot write to a memo - it's read-only
- Use `use_reactive!` for non-signal dependencies

---

## Task Management Hooks

### 8. `use_coroutine()`

**Purpose:** Long-running async task with message-passing channel for communication.

**Signature:**
```rust
pub fn use_coroutine<M, G, F>(init: G) -> Coroutine<M>
where
    M: 'static,
    G: FnMut(UnboundedReceiver<M>) -> F + 'static,
    F: Future<Output = ()> + 'static,
```

**Key Characteristics:**
- Provides `UnboundedReceiver<M>` for message passing
- Returns `Coroutine<M>` handle with `.send()` method
- Can pause, resume, and cancel
- Automatically injected as shared context

**When to Use:**
- Centralized async event loops
- Background services (chat clients, WebSocket handlers)
- State machines with async transitions
- Multi-component coordination

**Platform Compatibility:** ✅ Web (WASM), ✅ Desktop, ✅ Mobile

**Code Example:**
```rust
use dioxus::prelude::*;
use futures_util::StreamExt;

enum ChatAction {
    Connect,
    Disconnect,
    SendMessage(String),
}

#[component]
fn ChatApp() -> Element {
    let chat = use_coroutine(|mut rx: UnboundedReceiver<ChatAction>| async move {
        while let Some(action) = rx.next().await {
            match action {
                ChatAction::Connect => {
                    log::info!("Connecting to chat...");
                    // Async connection logic
                }
                ChatAction::SendMessage(msg) => {
                    log::info!("Sending: {}", msg);
                    // Async send logic
                }
                ChatAction::Disconnect => {
                    log::info!("Disconnecting...");
                    break;
                }
            }
        }
    });

    rsx! {
        button {
            onclick: move |_| chat.send(ChatAction::Connect),
            "Connect"
        }
        button {
            onclick: move |_| chat.send(ChatAction::SendMessage("Hello".into())),
            "Send"
        }
    }
}
```

**Context Injection:**
```rust
// Child components can access parent's coroutine
#[component]
fn ChildComponent() -> Element {
    let chat = use_coroutine_handle::<ChatAction>();

    rsx! {
        button {
            onclick: move |_| chat.send(ChatAction::SendMessage("Hi from child".into())),
            "Send from child"
        }
    }
}
```

**Gotchas:**
- Coroutine runs for component lifetime - use `use_hook` to prevent re-creation
- Channel is unbounded - can accumulate messages if consumer is slow
- Messages processed sequentially - no parallelism within one coroutine
- Must handle all message variants in the loop

---

### 9. `use_action()` (Fullstack/Server)

**Purpose:** Execute server functions with automatic cancellation and result tracking.

**Signature:**
```rust
pub fn use_action<T, F>(action: F) -> Action<T>
where
    F: Fn(T) -> Future<Output = Result<R, E>> + 'static,
```

**Key Characteristics:**
- Designed for server function calls
- Automatically cancels previous task on new call
- Stores result in signal for UI access
- Prevents race conditions

**When to Use:**
- ✅ Server function calls (RPC)
- ✅ Form submissions
- ✅ API mutations
- ❌ NOT for client-side async (use `use_resource`)

**Platform Compatibility:** ⚠️ Fullstack apps only (requires server functions)

**Code Example:**
```rust
use dioxus::prelude::*;

#[server]
async fn get_meaning(input: String) -> Result<String, ServerFnError> {
    Ok(format!("The meaning of '{}' is 42", input))
}

#[component]
fn ServerAction() -> Element {
    let mut meaning = use_action(|| get_meaning("life".into()));

    rsx! {
        h1 { "Meaning: {meaning.value():?}" }
        button {
            onclick: move |_| meaning.call(),
            "Run server function"
        }
    }
}
```

**Action Methods:**
```rust
action.call(input)    // Trigger action with input
action.value()        // Get result (Option<Result<T, E>>)
action.cancel()       // Cancel in-flight request
```

**Gotchas:**
- Only works with server functions (`#[server]`)
- Automatically cancels previous call - handle gracefully
- Not available in pure client-side apps
- Requires fullstack feature enabled

---

## Platform-Specific Threading

### Desktop: Tokio Runtime

**Available on:** ✅ Desktop, ❌ Web (WASM), ❌ Mobile

Dioxus Desktop runs on a **multithreaded Tokio runtime**, enabling true parallelism.

#### `tokio::spawn` - Async Parallelism

```rust
use dioxus::prelude::*;

#[component]
fn DesktopParallel() -> Element {
    let mut result = use_signal_sync(|| String::new());

    rsx! {
        button {
            onclick: move |_| {
                spawn(async move {
                    // Spawn onto Tokio's threadpool
                    let handle = tokio::spawn(async {
                        heavy_async_work().await
                    });

                    match handle.await {
                        Ok(res) => result.set(res),
                        Err(e) => log::error!("Task failed: {}", e),
                    }
                });
            },
            "Run on Tokio pool"
        }
        "{result}"
    }
}
```

#### `tokio::task::spawn_blocking` - CPU-Heavy Computation

**Use for:** Blocking I/O, CPU-intensive sync code (cryptography, compression, ML inference)

```rust
use dioxus::prelude::*;
use tokio::task::spawn_blocking;

#[component]
fn MLInference() -> Element {
    let mut embedding = use_signal_sync(|| Vec::new());

    rsx! {
        button {
            onclick: move |_| {
                spawn(async move {
                    // Move computation to blocking threadpool
                    let result = spawn_blocking(|| {
                        // CPU-intensive ML model inference
                        run_ml_model_sync("input text")
                    }).await.unwrap();

                    embedding.set(result);
                });
            },
            "Run ML model"
        }
    }
}
```

**When to use:**
- ✅ CPU-bound computation (ML, crypto, compression)
- ✅ Blocking I/O (sync file I/O, legacy APIs)
- ❌ NOT for async I/O (use `tokio::spawn` instead)
- ❌ NOT for long-running CPU work (use Rayon instead)

**Rayon for CPU Parallelism:**

For pure CPU-bound work (no I/O), Rayon's fork-join parallelism is more efficient:

```rust
use rayon::prelude::*;

spawn_blocking(move || {
    // Process chunks in parallel using Rayon
    let embeddings: Vec<_> = chunks.par_iter()
        .map(|chunk| compute_embedding(chunk))
        .collect();
    embeddings
})
```

---

### Web: Single-Threaded Async

**Available on:** ✅ Web (WASM), ❌ Desktop, ❌ Mobile

Web platform runs on browser's **single-threaded event loop** - no true parallelism without Web Workers.

**Key Constraints:**
- All async tasks run on main thread
- Long computations WILL block UI rendering
- `tokio::spawn` and `spawn_blocking` are NOT available
- Must use Web Workers for true parallelism

**Example (blocks UI):**
```rust
#[cfg(target_arch = "wasm32")]
{
    // ⚠️ This blocks the UI thread!
    spawn(async move {
        let embedding = compute_heavy_ml_model(&text).await;
        result.set(embedding);
    });
}
```

**Solution: Use Web Workers** (see next section)

---

### Cross-Platform Conditional Compilation

**Pattern for platform-specific async:**

```rust
use dioxus::prelude::*;

#[component]
fn CrossPlatformTask() -> Element {
    let mut result = use_signal(String::new);

    rsx! {
        button {
            onclick: move |_| {
                spawn(async move {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        // Desktop: Use tokio threadpool
                        let res = tokio::task::spawn_blocking(|| {
                            expensive_computation()
                        }).await.unwrap();
                        result.set(res);
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        // Web: Post to Web Worker
                        let res = post_to_worker("compute", data).await;
                        result.set(res);
                    }
                });
            },
            "Compute"
        }
    }
}
```

---

## Web Workers & WASM Threading

### Web Workers with wasm-bindgen

**Purpose:** True parallelism in web browsers by offloading work to separate threads.

**Requirements:**
- Cross-Origin Isolation (COOP/COEP headers)
- Service Worker to inject headers
- Module Worker loading WASM bundle

**Architecture:**

```
Main Thread (Dioxus UI)
    ↓ postMessage
Web Worker (WASM)
    ↓ ML Inference / Heavy Computation
    ↓ postMessage
Main Thread (Update UI)
```

**Setup:**

1. **Service Worker** (`public/coi-serviceworker.min.js`):
```javascript
// Injects COOP/COEP headers for SharedArrayBuffer support
self.addEventListener('fetch', function(e) {
  // Inject headers...
});
```

2. **Module Worker** (`public/worker.js`):
```javascript
// ES6 module worker
import init, { wasm_function } from '/wasm/app.js';

self.onmessage = async (e) => {
  await init(); // Initialize WASM
  const result = wasm_function(e.data);
  self.postMessage(result);
};
```

3. **wasm-bindgen Rust Code** (`lib.rs`):
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct EmbeddingWorker {
    model: Model,
}

#[wasm_bindgen]
impl EmbeddingWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Initialize ML model in worker
        Self { model: load_model() }
    }

    #[wasm_bindgen]
    pub fn compute(&self, text: &str) -> Vec<f32> {
        // Heavy computation in worker thread
        self.model.encode(text)
    }
}
```

4. **Dioxus Integration:**
```rust
use dioxus::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::{Worker, MessageEvent};

#[component]
fn WorkerExample() -> Element {
    let mut result = use_signal(|| Vec::new());

    // Initialize worker once
    let worker = use_hook(|| {
        Worker::new("/worker.js").unwrap()
    });

    use_effect(move || {
        let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
            let data: Vec<f32> = serde_wasm_bindgen::from_value(e.data()).unwrap();
            result.set(data);
        }) as Box<dyn FnMut(_)>);

        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget(); // Keep alive
    });

    rsx! {
        button {
            onclick: move |_| {
                worker.post_message(&JsValue::from_str("compute")).unwrap();
            },
            "Compute in worker"
        }
        "Result: {result():?}"
    }
}
```

**Current Project Implementation:**

This project uses a blob worker pattern for non-blocking ML inference on the web platform. See `docs/web-worker-implementation.md` for implementation details.

**Implementation:**
- Blob worker with dynamically generated code (components.rs:200-309)
- WASM bindings for tokenizer and model (embedding.rs)
- Message passing via js_sys::Object (avoids serde_json serialization issues)
- Model loads in worker thread (65MB, 30-60s initial download)
- UI remains responsive during model download and inference

**Performance:**
- First run: 30-60s (model download) + 100-500ms (inference)
- Subsequent runs: <10ms (tokenization) + 100-500ms (inference)
- Model cached in worker memory after initial load

---

### SharedArrayBuffer & WASM Threads

**Status:** ⚠️ Experimental / Limited browser support

**Requirements:**
- Cross-Origin Isolation (COOP + COEP headers)
- Browser support (Chrome, Firefox with flags)
- `wasm-bindgen` with `atomics` + `bulk-memory` features

**Pattern:**
```rust
// Cargo.toml
[dependencies]
wasm-bindgen = { version = "0.2", features = ["atomics"] }

[target.wasm32-unknown-unknown]
rustflags = ['-C', 'target-feature=+atomics,+bulk-memory']
```

**Use Cases:**
- Rayon parallel iterators in WASM
- `std::thread::spawn` in browser
- Shared memory between workers

**Gotchas:**
- Very limited browser support
- Requires careful memory management
- SharedArrayBuffer has security restrictions
- Not recommended for production (yet)

---

## Comparison Matrix

| Mechanism | Returns Value | Reactive | Lifecycle | Parallelism (Desktop) | Parallelism (Web) | Use Case |
|-----------|--------------|----------|-----------|----------------------|-------------------|----------|
| **spawn** | ❌ | ❌ | Component | Async only | Main thread | Event-driven tasks |
| **spawn_forever** | ❌ | ❌ | App | Async only | Main thread | Global background tasks |
| **spawn_isomorphic** | ❌ | ❌ | Component | Async only | Main thread | Fullstack deterministic tasks |
| **use_resource** | ✅ `Option<T>` | ✅ | Component | Async only | Main thread | Reactive data fetching |
| **use_future** | ❌ | ❌ | Component | Async only | Main thread | Mount-time side effects |
| **use_effect** | ❌ | ✅ | Component | Async only | Main thread | Post-render side effects |
| **use_memo** | ✅ `T` | ✅ | Component | Sync only | Sync only | Derived state |
| **use_coroutine** | ❌ | ❌ (messages) | Component | Async only | Main thread | Message-driven async loops |
| **use_action** | ✅ `Result<T,E>` | ❌ | Component | Server-side | N/A | Server function calls |
| **tokio::spawn** | ❌ | ❌ | Manual | ✅ True threads | ❌ N/A | Desktop async parallelism |
| **tokio::spawn_blocking** | ❌ | ❌ | Manual | ✅ True threads | ❌ N/A | Desktop CPU-bound work |
| **Web Worker** | ✅ | ❌ | Manual | ❌ N/A | ✅ Separate thread | Web CPU-bound work |
| **SharedArrayBuffer** | ✅ | ❌ | Manual | ❌ N/A | ⚠️ Experimental | Web shared memory |

---

## Recommendations for ML Inference

Based on the research, here are the recommended approaches for ML inference in Dioxus:

### Desktop Platform (Recommended)

**Pattern 1: Async Model Loading + Blocking Inference**

```rust
use dioxus::prelude::*;
use tokio::task::spawn_blocking;

#[component]
fn MLInference() -> Element {
    // Load model once using resource
    let model = use_resource(|| async {
        load_ml_model().await // Async model loading
    });

    let mut result = use_signal_sync(|| Vec::new());

    rsx! {
        match model.value() {
            Some(Ok(m)) => rsx! {
                button {
                    onclick: move |_| {
                        let model_clone = m.clone();
                        spawn(async move {
                            // Move CPU-heavy inference to blocking pool
                            let embedding = spawn_blocking(move || {
                                model_clone.encode("input text")
                            }).await.unwrap();

                            result.set(embedding);
                        });
                    },
                    "Run Inference"
                }
                "Result: {result():?}"
            },
            Some(Err(e)) => rsx! { "Error loading model: {e}" },
            None => rsx! { "Loading model..." }
        }
    }
}
```

**Why:**
- ✅ Model loading is async (doesn't block UI)
- ✅ Inference runs on dedicated thread pool
- ✅ UI remains responsive
- ✅ Simple, idiomatic Dioxus code

---

### Web Platform (Current Implementation)

**Pattern 2: Blob Worker with Direct Message Passing**

The project uses a blob worker created from a JavaScript string embedded in Rust. The worker handles model loading, tokenization, and inference on a separate thread.

**Key implementation details:**
- Worker created from blob URL (avoids asset hashing issues)
- WASM path injected at runtime into worker code
- Message passing using js_sys::Object (not serde_json)
- Model and tokenizer initialized in worker context
- Results sent back via postMessage

**Implementation location:** `src/components.rs` (blob worker creation), `src/embedding.rs` (WASM bindings)

**Tradeoffs:**
- True parallelism keeps UI responsive during ML inference
- Requires COOP/COEP headers (already configured via service worker)
- More complex setup than inline async code
- Blob pattern avoids separate JavaScript file management

See `docs/web-worker-implementation.md` for complete implementation details.

---

### Cross-Platform (Best of Both)

**Pattern 3: Conditional Compilation**

```rust
use dioxus::prelude::*;

#[component]
fn UniversalMLInference() -> Element {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Desktop: Use tokio
        use tokio::task::spawn_blocking;

        let mut result = use_signal_sync(Vec::new);

        rsx! {
            button {
                onclick: move |_| {
                    spawn(async move {
                        let embedding = spawn_blocking(|| {
                            run_ml_model("input")
                        }).await.unwrap();
                        result.set(embedding);
                    });
                },
                "Compute"
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        // Web: Use coroutine + Web Worker
        let worker_task = use_coroutine(|mut rx| async move {
            // Web Worker logic...
        });

        rsx! {
            button {
                onclick: move |_| {
                    worker_task.send(ComputeMessage);
                },
                "Compute"
            }
        }
    }
}
```

**Why:**
- ✅ Optimal performance on each platform
- ✅ Single codebase with platform-specific optimizations
- ✅ Type-safe abstractions
- ⚠️ More boilerplate

---

### Key Takeaways for ML Workloads

1. **Desktop:** Use `tokio::spawn_blocking` for CPU-heavy inference
2. **Web:** Use Web Workers for true parallelism (current project approach)
3. **Model Loading:** Use `use_resource` for async, reactive loading
4. **State Management:** Use `use_signal_sync` for thread-safe signals
5. **Task Coordination:** Use `use_coroutine` for message-passing patterns
6. **Avoid:** Never run heavy computation directly in `spawn` on web (blocks UI)

---

## Gotchas & Best Practices

### General Rules

1. **Hook Rules:**
   - Always call hooks at component root level
   - Never use in conditionals, loops, or event handlers
   - Maintain consistent ordering across renders

2. **Async Safety:**
   - Never hold locks across `.await` points
   - Futures are single-threaded by default
   - Use `use_signal_sync` for multi-threaded access

3. **Cancellation:**
   - `spawn` auto-cancels on component drop
   - `use_resource` auto-cancels and restarts on dependency changes
   - `use_action` auto-cancels previous calls
   - Always handle cancellation gracefully (use `Drop` guards)

4. **Performance:**
   - `use_memo` only updates dependents if output changes
   - `use_resource` caches results until dependencies change
   - Avoid spawning in render loops

5. **Platform Differences:**
   - Web: Single-threaded, requires Web Workers for parallelism
   - Desktop: Multi-threaded Tokio runtime
   - Use conditional compilation for platform-specific code

### Signal Safety (from project `clippy.toml`)

**Never hold these across `.await` points:**
- `generational_box::GenerationalRef`
- `generational_box::GenerationalRefMut`
- `dioxus_signals::Write`

**Example (causes deadlock):**
```rust
// ❌ BAD - holds Write lock across .await
{
    let mut write = signal.write();
    some_async_fn().await; // DEADLOCK!
    *write = new_value;
}

// ✅ GOOD - drop lock before .await
{
    let value = {
        let mut write = signal.write();
        *write = new_value;
        *write
    }; // Lock dropped here
    some_async_fn().await; // Safe
}
```

---

## Additional Resources

### Official Documentation
- **Dioxus 0.7 Docs:** https://dioxuslabs.com/learn/0.7/
- **Async & Futures Guide:** https://dioxuslabs.com/learn/0.7/essentials/basics/async
- **Hooks Reference:** https://docs.rs/dioxus-hooks/latest/
- **Dioxus Core:** https://docs.rs/dioxus-core/latest/

### Examples
- **Dioxus Examples Repo:** https://github.com/DioxusLabs/dioxus/tree/main/packages/signals/examples
- **send.rs:** Multi-threaded signal example

### Community
- **Discord:** https://discord.gg/XgGxMSkvUM
- **GitHub Discussions:** https://github.com/DioxusLabs/dioxus/discussions

### Related Technologies
- **wasm-bindgen Guide:** https://rustwasm.github.io/docs/wasm-bindgen/
- **Web Workers Pattern:** https://alexcrichton.github.io/wasm-bindgen/examples/wasm-in-web-worker.html
- **Tokio Docs:** https://docs.rs/tokio/latest/tokio/

---

**Document Version:** 1.0
**Last Updated:** 2025-11-13
**Dioxus Version:** 0.7.1
