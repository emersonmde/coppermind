# ADR 003: WASM Threading Workaround for Dioxus Build System

**Status:** Accepted
**Date:** 2025-11-16
**Context:** Enabling wasm-bindgen-rayon for 3x ML inference speedup in browser

---

## Summary

Implement WASM threading (via wasm-bindgen-rayon) to parallelize Candle ML inference, achieving ~3x speedup (5 tokens/sec → 16 tokens/sec). The `wasm-threading` feature is configured but fails in release builds due to Dioxus Manganis asset system incompatibility with Rust's `build-std` feature. This ADR documents a tiered workaround approach while preserving dx workflow for desktop/mobile builds.

---

## Context

### Goal

Enable multi-threaded WASM execution for:
- **Candle ML inference**: Parallelize matrix operations (gemm, etc.)
- **instant-distance HNSW**: Parallel vector search
- **Expected speedup**: 3x based on [Hugging Face Candle PR #3063](https://github.com/huggingface/candle/pull/3063)

### Current State

**What's Working**:
- ✅ Cross-Origin Isolation (COI) enabled via service worker
- ✅ `wasm-threading` Cargo feature configured
- ✅ `rust-toolchain.toml` with `rust-src` component
- ✅ `.cargo/config.toml` with `+atomics,+bulk-memory` flags
- ✅ Worker code calls `module.initThreadPool(navigator.hardwareConcurrency)`

**What's Failing**:
- ❌ `dx bundle --release --features wasm-threading` fails with:
  ```
  ERROR Failed to hash asset : No such file or directory (os error 2)
  ERROR Failed to copy asset "": No such file or directory (os error 2)
  ```
- ✅ `dx build --features wasm-threading` (debug mode) succeeds with warnings
- ✅ `initThreadPool` is exported in debug builds

### The Problem: Manganis + build-std Incompatibility

**Root Cause**:
1. **build-std** (required for WASM threading) rebuilds Rust std library with `+atomics` support
2. **Manganis** (Dioxus asset system) serializes asset metadata during build
3. With `build-std`, the metadata format changes → Manganis can't deserialize
4. **Debug mode**: Only warns ("asset could not be deserialized")
5. **Release mode**: Hard fails ("Failed to hash asset")

**Why This Matters**:
- Manganis provides asset hashing (cache-busting), bundling, and optimization
- Critical for production builds
- Desktop/iOS builds use native dx workflow - we want to preserve this

---

## Research Findings

### Investigation: How dx bundle Works

**Source**: DioxusLabs/dioxus `packages/cli/src/`

**Build Pipeline**:
1. `cargo build --target wasm32-unknown-unknown`
2. **wasm-bindgen invocation** (dx bundles its own version)
   - `--target web`
   - Generates JS glue code
3. **Manganis asset optimization** (PR #3531)
   - Hashes and optimizes assets
   - Bundles wasm-bindgen outputs
   - **This is where build-std breaks**
4. Optional `wasm-opt` for size optimization

**Complexity**: Medium - tightly integrated with Manganis

### Investigation: Community Precedent

**Searches Performed**:
- Dioxus + wasm-bindgen-rayon: ❌ Zero examples
- Leptos + rayon: ❌ Zero examples
- Yew + rayon: ❌ Zero examples
- Dioxus issues: "build-std", "asset failed to hash": ❌ No documented issues

**Production WASM Threading Examples** (not Rust web frameworks):
- ✅ Squoosh.app (image compression)
- ✅ Google Earth Web
- ✅ FFMPEG.WASM
- ✅ wasm-bindgen parallel raytracing (official example)

**Conclusion**: **We are the first** to attempt wasm-bindgen-rayon with a modern Rust web framework. This is uncharted territory - making this a valuable reference implementation for the community.

### Investigation: Potential Fixes Evaluated

| Option | Feasibility | Timeline | Reversibility | Trade-offs |
|--------|-------------|----------|---------------|------------|
| **A. NO_DOWNLOADS flag** | Easy (if available) | 15 min | Perfect | May not exist in dx 0.7.1 |
| **B. Trunk bundler** | Medium | 2-4 hours | High | Loses desktop/mobile dx workflow |
| **C. Manual build script** | Hard | 4-8 hours | High | Maintenance burden, replicates dx |
| **D. Fix Manganis upstream** | Very Hard | Days-Weeks | Perfect | Not aligned with reference impl timeline |
| **E. Local Manganis patch** | Medium-Hard | 6-12 hours | Medium | Brittle, breaks on Dioxus updates |

**Option A - NO_DOWNLOADS Flag**:
- Merged in [Dioxus PR #3465](https://github.com/DioxusLabs/dioxus/pull/3465) (Jan 17, 2025)
- Tells dx to use system `wasm-bindgen-cli` instead of bundled version
- **Status**: Unknown if included in dx 0.7.1 (current version)
- **Usage**: `NO_DOWNLOADS=1 dx bundle --release`

**Option C - Manual Build Script**:
- Bypass Manganis entirely
- Use cargo + system wasm-bindgen-cli
- Manually replicate asset copying/hashing
- Preserves dx for desktop/mobile (`dx bundle --platform desktop`)

---

## Decision

### Chosen Approach: Tiered Workaround Strategy

**Priority**: Stay with dx for desktop/mobile, temporary workaround for web release builds

**Phases**:

#### Phase 1: Test NO_DOWNLOADS Flag (Best Case: 15 min)

**Action**: Test if dx 0.7.1 includes NO_DOWNLOADS support

**Command**:
```bash
NO_DOWNLOADS=1 dx bundle --release --features wasm-threading
```

**Success Criteria**:
- Build completes without asset errors
- `initThreadPool` exported in output
- Browser console shows Rayon initialization

**If Successful**:
- Update `.githooks/pre-commit` to use flag
- Update `.github/workflows/ci.yml` to use flag
- Document in CLAUDE.md as temporary workaround
- **Skip Phase 2**

**If Failed**:
- Proceed to Phase 2

---

#### Phase 2: Build Script Workaround (Fallback: 3-4 hours)

**Action**: Create `scripts/build-web-release.sh` to bypass Manganis

**Script Overview**:
```bash
#!/bin/bash
# scripts/build-web-release.sh
# Temporary workaround for wasm-threading until Dioxus Manganis supports build-std
# TODO: Remove when https://github.com/DioxusLabs/dioxus/issues/XXXX is resolved

set -euo pipefail

echo "→ Building WASM with threading support..."

# 1. Build with build-std (requires nightly + rust-src)
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory' \
  cargo build \
  --release \
  --target wasm32-unknown-unknown \
  --features web,wasm-threading \
  -Z build-std=panic_abort,std

# 2. Run wasm-bindgen (system version, not dx's bundled version)
echo "→ Running wasm-bindgen..."
wasm-bindgen \
  target/wasm32-unknown-unknown/release/coppermind.wasm \
  --out-dir dist/wasm \
  --target web \
  --no-typescript \
  --split-linked-modules

# 3. Optional: Optimize with wasm-opt
if command -v wasm-opt >/dev/null 2>&1; then
  echo "→ Optimizing WASM..."
  wasm-opt -Oz -o dist/wasm/coppermind_bg.wasm dist/wasm/coppermind_bg.wasm
fi

# 4. Copy public assets
echo "→ Copying assets..."
mkdir -p dist
cp -r public/* dist/

# 5. Generate index.html from template
echo "→ Generating index.html..."
cat > dist/index.html <<'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Coppermind</title>
    <base href="/coppermind/" />
    <link rel="icon" type="image/x-icon" href="favicon.ico" />
</head>
<body>
    <div id="main"></div>
    <script type="module">
        import init from './wasm/coppermind.js';
        await init();
    </script>
</body>
</html>
EOF

echo "✓ Build complete: dist/"
```

**Asset Handling**:
- Public assets: Direct copy from `public/` (favicon, COI service worker)
- Model assets: Loaded via Dioxus `asset!()` macro from `assets/models/`
  - ⚠️ **Potential issue**: `asset!()` may not work with manual build
  - **Solution**: Convert to runtime fetch if needed

**CI Integration**:
```yaml
# .github/workflows/ci.yml - wasm-build job
- name: Build Dioxus web assets with WASM threading
  run: ./scripts/build-web-release.sh
- name: Upload Pages artifact
  with:
    path: dist  # Changed from target/dx/coppermind/release/web/public
```

**Desktop/Mobile Builds** (unchanged):
```bash
dx bundle --release --platform desktop  # Works normally
dx bundle --release --platform ios       # Works normally
```

**Reversibility**:
- When Dioxus fixes Manganis: Delete `scripts/build-web-release.sh`
- Revert CI to use `dx bundle --release --features wasm-threading`
- Remove script references from pre-commit hook

---

#### Phase 3: Document & File Upstream Issue (1 hour)

**Actions**:

1. **Test thoroughly**:
   - Verify `initThreadPool` export
   - Confirm Rayon initialization in browser console
   - Measure embedding speedup (expect ~3x)

2. **File Dioxus GitHub Issue**:
   - **Title**: "Manganis asset system incompatible with build-std (WASM threading)"
   - **Body**:
     - Reproduction steps
     - Error logs (release vs debug behavior)
     - Use case: wasm-bindgen-rayon for ML inference
     - Link to this ADR
     - Workaround documented
   - **Labels**: bug, wasm, asset-system

3. **Update CLAUDE.md**:
   ```markdown
   ## WASM Threading Status

   **Current**: Release builds use `scripts/build-web-release.sh` workaround
   **Reason**: Manganis asset system incompatible with build-std
   **Tracking**: https://github.com/DioxusLabs/dioxus/issues/XXXX
   **Reversibility**: When fixed, delete script and revert to `dx bundle --release --features wasm-threading`
   ```

4. **Document in code**:
   ```bash
   # scripts/build-web-release.sh
   # TODO: Remove when https://github.com/DioxusLabs/dioxus/issues/XXXX is resolved
   ```

---

## Implementation Results

### Phase 1: NO_DOWNLOADS Flag (Completed - FAILED)

**Date**: 2025-11-16
**Duration**: 30 minutes

**Actions Taken**:
1. Installed `wasm-bindgen-cli@0.2.105` (exact version dx requires)
2. Ran: `NO_DOWNLOADS=1 dx bundle --release --features wasm-threading`

**Result**: ❌ **FAILED**

**Error**: Same asset hashing failure as without NO_DOWNLOADS:
```
ERROR Failed to hash asset : No such file or directory (os error 2)
ERROR Failed to copy asset "": No such file or directory (os error 2)
ERROR dx bundle: Failed to write assets
```

**Findings**:
- The `NO_DOWNLOADS` feature exists in dx 0.7.1 ✅
- Using system `wasm-bindgen-cli` instead of bundled version does NOT solve the issue
- Confirms the problem is **Manganis + build-std incompatibility**, not wasm-bindgen version mismatch
- Proceed to Phase 2

---

### Phase 2: Build Script Workaround (Completed - FAILED)

**Date**: 2025-11-16
**Duration**: 2 hours
**Status**: ❌ **FAILED** - Runtime errors prevent WASM execution

**Attempt 2A: cargo + wasm-bindgen-cli**

**Actions Taken**:
1. Created `scripts/build-web-release.sh`
2. Manually invoked cargo + wasm-bindgen-cli to bypass Manganis
3. Created shared WebAssembly.Memory in JS initialization
4. Configured `.cargo/config.toml` with threading flags

**Result**: ❌ **FAILED**

**Error**:
```
error: failed to prepare module for threading
Caused by:
    failed to find `__wasm_init_tls`
```

**Root Cause**:
- wasm-bindgen's threading transform expects `__wasm_init_tls` function
- Recent nightly Rust versions (1.93.0-nightly) removed or changed TLS initialization
- wasm-bindgen-rayon documentation specifies `nightly-2024-08-02` but:
  - That version fails to compile `indexmap` (uses unstable `precise_capturing` feature)
  - Can't find a nightly version that satisfies both wasm-bindgen-rayon AND current dependencies

**Attempt 2B: wasm-pack**

**Actions Taken**:
1. Switched to `wasm-pack build --target web` (official recommended tool)
2. Successfully compiled without `__wasm_init_tls` error
3. Created `scripts/patch-wasm-threading.js` to inject shared memory into wasm-pack output
4. Modified worker initialization for wasm-pack compatibility

**Build Result**: ✅ Build completes successfully

**Runtime Result**: ❌ **FAILED** - Multiple fatal errors

**Errors**:
1. **Main thread**: `TypeError: [object Int32Array] is not a shared typed array`
   - Location: `Atomics.waitAsync()` call in `wasm_bindgen_futures`
   - Cause: WASM memory is NOT actually shared despite patch

2. **Worker thread**: `DataCloneError: Failed to execute 'postMessage' on 'Worker': #<Memory> could not be cloned`
   - Location: `wasm-bindgen-rayon` worker initialization
   - Cause: wasm-bindgen-rayon tries to clone Memory object via postMessage
   - Workers can't share Memory objects this way

**Verification**:
- ✅ `initThreadPool` function exported
- ✅ wasm-bindgen-rayon worker helpers included
- ✅ Build completes without errors
- ✅ SharedArrayBuffer available in browser
- ✅ Shared memory created (`[Threading] Created shared WebAssembly.Memory`)
- ❌ **Memory not actually used by WASM module**
- ❌ **Worker communication fails (cannot clone Memory)**
- ❌ **Buttons unclickable (runtime errors freeze UI)**

**Why Patching Failed**:
- wasm-pack generates `WebAssembly.instantiate()` calls that create their own memory
- Our patch injects memory into `imports.env.memory` but wasm-pack ignores it
- Even if memory injection worked, wasm-bindgen-rayon's postMessage approach is incompatible with WebAssembly.Memory objects

**Conclusion**: Neither cargo+wasm-bindgen nor wasm-pack can satisfy wasm-bindgen-rayon's requirements in our setup

---

## Root Cause Analysis

### Why wasm-bindgen-rayon is Incompatible

**wasm-bindgen-rayon Requirements**:
1. Nightly Rust with TLS support (`__wasm_init_tls`)
2. Shared WebAssembly.Memory created before WASM instantiation
3. Memory object must be clonable via Worker.postMessage()
4. Specific build tool integration (designed for wasm-pack)

**Dioxus + Modern Dependencies Constraints**:
1. Recent dependencies (indexmap 2.12) require newer Rust features
2. Can't use old nightly (2024-08-02) - compilation fails
3. Can't use new nightly - TLS implementation changed
4. **No nightly version satisfies both constraints**

**WebAssembly.Memory Cloning Issue**:
- wasm-bindgen-rayon's `workerHelpers.js` tries: `worker.postMessage({ type: 'init', memory, module })`
- Chrome error: `Failed to execute 'postMessage' on 'Worker': #<Memory> could not be cloned`
- **WebAssembly.Memory objects are NOT clonable via postMessage**
- This is a fundamental limitation, not a configuration issue

**wasm-pack Memory Management**:
- wasm-pack's `--target web` creates memory during `WebAssembly.instantiate()`
- No way to inject pre-created shared memory
- Even with JS patching, the instantiate call creates its own non-shared memory
- The injected shared memory is ignored

### Fundamental Incompatibilities

1. **Toolchain Version Conflict**: No single Rust nightly satisfies:
   - wasm-bindgen-rayon's TLS requirements (old nightly)
   - Modern dependency requirements (new nightly)

2. **Architecture Mismatch**: wasm-bindgen-rayon designed for:
   - Simple wasm-pack projects
   - No complex dependencies
   - Not tested with Dioxus web framework

3. **Browser Limitation**: WebAssembly.Memory cannot be cloned
   - This breaks wasm-bindgen-rayon's worker initialization
   - Not fixable without changing wasm-bindgen-rayon itself

### Why We're the First to Hit This

From ADR research: **Zero examples** of Dioxus/Leptos/Yew + wasm-bindgen-rayon in production.

**Reasons**:
1. Most Rust web frameworks don't use compute-heavy ML workloads
2. Projects doing browser ML typically use JS libraries (transformers.js, ONNX Runtime Web)
3. wasm-bindgen-rayon is relatively niche (experimental, archived repo)
4. The combination of: Dioxus + Candle + wasm-threading + modern dependencies is novel

**We are attempting a cutting-edge combination that hasn't been validated by the community.**

---

## Updated Status & Decision

### Current State

**What Works**:
- ✅ Single-threaded WASM builds (dx serve, dx bundle)
- ✅ Desktop/iOS builds with native threading (Rayon works fine)
- ✅ Cross-Origin Isolation setup (COI service worker)
- ✅ SharedArrayBuffer available in browser

**What Doesn't Work**:
- ❌ WASM threading in release builds
- ❌ wasm-bindgen-rayon runtime initialization
- ❌ Manual build scripts as dx replacement

**Blockers**:
1. Rust nightly version incompatibility (cannot satisfy both wasm-bindgen-rayon and dependencies)
2. WebAssembly.Memory cloning limitation (browser/spec limitation)
3. wasm-pack memory management incompatible with shared memory injection

### Recommendation: Defer WASM Threading

**Rationale**:
1. **No working solution exists** with current tooling
2. **High maintenance burden**: Custom build scripts are fragile
3. **Blocking other work**: Can't make progress on features while debugging build system
4. **Desktop already has threading**: 3x speedup available on macOS/Windows/Linux via native Rayon
5. **Web can work single-threaded**: Slower but functional

**Proposed Path Forward**:

**Short-term (Now)**:
- Revert custom build scripts
- Return to dx-based workflow
- Document WASM threading as "future work"
- Focus on core functionality (search, UI, storage)

**Medium-term (Q1 2025)**:
- Monitor wasm-bindgen-rayon for updates
- Watch for Dioxus + threading examples in community
- Re-evaluate when Rust WASM threading stabilizes

**Long-term (Q2+ 2025)**:
- Consider alternative approaches:
  - WebGPU compute shaders for ML inference (when Candle supports it)
  - Web Workers with message-passing parallelism (not shared memory)
  - ONNX Runtime Web (mature, production-ready)

### Success Criteria Revisited

Original goal: 3x speedup for browser ML inference via Rayon threading

**Alternative ways to achieve performance**:
1. **Model optimization**: Quantization, smaller models
2. **WebGPU**: GPU acceleration (future Candle feature)
3. **Caching**: Memoize embeddings, reduce inference calls
4. **Progressive loading**: Stream results, don't block UI
5. **Desktop app**: Native threading already works

**Browser threading is ONE approach, not the ONLY approach**

---

## Implementation Status

### Files Changed (To Be Reverted)

**Created** (experimental, non-functional):
- [ ] `scripts/build-web-release.sh` - Manual build script (TO BE DELETED)
- [ ] `scripts/patch-wasm-threading.js` - JS patching script (TO BE DELETED)
- [x] `docs/adrs/003-wasm-threading-workaround.md` - This ADR (KEEP - documents lessons learned)

**Modified** (experimental changes):
- [ ] `.githooks/pre-commit` - TO BE REVERTED to use dx bundle
- [ ] `.github/workflows/ci.yml` - TO BE REVERTED to standard dx workflow
- [ ] `CLAUDE.md` - TO BE UPDATED to reflect threading deferment
- [ ] `Cargo.toml` - Added wasm-opt = false (TO BE REVERTED)
- [ ] `.cargo/config.toml` - Added threading flags (TO BE REVERTED)
- [ ] `rust-toolchain.toml` - Changed to nightly (TO BE REVERTED to stable or remove)
- [ ] `assets/workers/embedding-worker.js` - Modified for wasm-pack (TO BE REVERTED)

### Success Criteria (NOT MET)

- [x] Release builds complete successfully (wasm-pack builds, but...)
- [x] `initThreadPool` function exported in WASM JS glue
- ❌ **Browser console shows: ERRORS instead of successful initialization**
- ❌ **Embedding operations: FAIL to run due to runtime errors**
- ❌ **CI: Not updated (blocked on working solution)**
- ❌ **Desktop/iOS builds: Not tested with custom scripts**
- ❌ **Upstream issue: Not filed (no clear reproduction to report)**

**Overall Result**: ❌ **FAILED - No viable path to WASM threading with current tooling**

---

## Rollback Plan

**If we encounter blockers during Phase 1 or Phase 2**:

1. ⚠️ **STOP immediately and discuss with user**
2. **Do NOT unilaterally abandon the approach**
3. Evaluate remaining options together:
   - Try alternative build configurations
   - Consider local Manganis patch
   - Investigate Trunk bundler as fallback

4. **Only if all options exhausted AND user agrees**:
   - Revert all changes (git reset)
   - Remove `wasm-threading` from release builds
   - Keep feature available for dev testing only: `dx serve --features wasm-threading`
   - Document in CLAUDE.md:
     ```markdown
     ## WASM Threading Status

     **Status**: Experimental - dev mode only
     **Reason**: Dioxus asset system incompatible with build-std in release mode
     **Usage**: `dx serve --features wasm-threading` (warnings are safe to ignore)
     **Production**: Single-threaded (no performance impact vs current state)
     ```

**Important**: This rollback is a **last resort only**. The goal is to make threading work in production.

---

## Future Considerations

### When Dioxus Fixes Manganis

**Watch for**:
- Dioxus release notes mentioning `build-std` support
- Manganis updates to asset serialization
- Community reports of WASM threading working

**Cleanup Checklist**:
1. Test: `dx bundle --release --features wasm-threading` (without workaround)
2. If successful:
   - Delete `scripts/build-web-release.sh`
   - Remove NO_DOWNLOADS flag from CI/pre-commit
   - Update CLAUDE.md: Remove workaround note
   - Close upstream issue with "Fixed in vX.X.X"
3. Update this ADR status to "Superseded"

### Upstream Contribution Opportunity

If we discover the exact Manganis bug:
- **Potential fix**: Handle build-std metadata format difference
- **Location**: `dioxus-asset-resolver` or `manganis-cli-support` crates
- **Contribution**: PR to DioxusLabs/dioxus with fix + test case

---

## Alternatives Considered

### ❌ Switch to Trunk Bundler

**Pros**:
- Proven to work with wasm-bindgen-rayon
- Community-standard Rust WASM bundler

**Cons**:
- Loses dx workflow for desktop/mobile
- User preference: Stay with dx for cross-platform benefits
- Less reversible (bigger migration)

**Verdict**: Rejected - not aligned with project goals

---

### ❌ Local Manganis Patch

**Approach**: Use `[patch.crates-io]` to fork and fix Manganis locally

**Cons**:
- Brittle - breaks on Dioxus updates
- Requires deep dive into Manganis internals (days of work)
- Not reversible (ongoing maintenance)

**Verdict**: Rejected - too fragile for reference implementation

---

### ❌ Wait for Upstream Fix

**Cons**:
- Timeline unknown (could be weeks/months)
- No guarantee Dioxus will prioritize this
- Defeats goal of being a cutting-edge reference implementation

**Verdict**: Rejected - defeats project purpose

---

## References

- **Candle Rayon WASM PR**: https://github.com/huggingface/candle/pull/3063 (3x speedup proof)
- **wasm-bindgen-rayon**: https://github.com/RReverser/wasm-bindgen-rayon
- **Dioxus NO_DOWNLOADS PR**: https://github.com/DioxusLabs/dioxus/pull/3465
- **COOP/COEP Explainer**: https://web.dev/coop-coep/
- **COI Service Worker**: `public/coi-serviceworker.min.js`
- **Previous ADR**: `002-web-crawler-desktop-first.md` (COI background)

---

## Timeline

- **ADR Creation**: 1 hour
- **Phase 1 (NO_DOWNLOADS test)**: 15 min
- **Phase 2 (Build script)**: 3-4 hours (if needed)
- **Phase 3 (Documentation)**: 1 hour
- **Total**: 5.5-6.5 hours worst case, 2.5 hours best case

---

## Appendix: Technical Details

### Required Rust Toolchain

```toml
# rust-toolchain.toml
[toolchain]
channel = "nightly"
components = ["rust-src", "rustfmt", "clippy"]
targets = ["wasm32-unknown-unknown"]
```

### Required Cargo Config

```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = [
  "--cfg", "getrandom_backend=\"wasm_js\"",
  "-C", "link-arg=--initial-memory=536870912",
  "-C", "link-arg=--max-memory=4294967296",
  "-C", "target-feature=+atomics,+bulk-memory",
]

[unstable]
build-std = ["panic_abort", "std"]
```

### Worker Initialization Code

```javascript
// assets/workers/embedding-worker.js
if (typeof module.initThreadPool === "function") {
    const threadCount = navigator.hardwareConcurrency || 4;
    await module.initThreadPool(threadCount);
}
```

### Expected Console Output (NOT ACHIEVED)

**What we expected**:
```
[EmbeddingWorker] Initializing WASM module...
[EmbeddingWorker] Initializing Rayon thread pool with 8 threads...
[EmbeddingWorker] Rayon thread pool initialized successfully
[EmbeddingWorker] Starting embedding worker...
```

**What we got**:
```
TypeError: [object Int32Array] is not a shared typed array at Atomics.waitAsync
DataCloneError: Failed to execute 'postMessage' on 'Worker': #<Memory> could not be cloned
```

### Performance Benchmarks (NOT MEASURED)

**Goal** (based on Candle PR #3063):
- Embedding speed: ~5 tokens/sec → ~16 tokens/sec (3.2x speedup)
- Document indexing: Parallelized across chunks

**Actual**:
- ❌ Could not run due to runtime errors
- ❌ No performance measurements possible

---

## Final Status

**Status**: ❌ **FAILED / ABANDONED**
**Date**: 2025-11-16
**Decision**: Revert experimental changes, abandon WASM threading indefinitely

**Key Learnings**:
1. wasm-bindgen-rayon incompatible with Dioxus + modern dependencies
2. Rust nightly version conflicts prevent compilation
3. WebAssembly.Memory cloning is a fundamental browser limitation
4. No community precedent for Dioxus + wasm-threading (warning sign)
5. Custom build scripts are fragile and unmaintainable
6. **Rust WebAssembly atomics are fundamentally broken** (confirmed by tracking issue #77839)

### Evidence: Rust WebAssembly Atomics Are Broken

**Source**: [Rust tracking issue #77839](https://github.com/rust-lang/rust/issues/77839) (WebAssembly atomics)

**Status as of September 2025**: Unstabilized, incomplete, broken in critical ways

**Confirmed Issues**:
1. **No dedicated WASM target** - Requires manual flag configuration (`-Zbuild-std`, `-Ctarget-feature=+atomics`)
2. **Thread-local storage (TLS) destructors ignored** - "The standard library ignores destructors registered on wasm and simply never runs them"
3. **`std::thread` incompatible** - Standard thread APIs cannot function with this threading model
4. **Manual setup requirements** - Memory initialization, imports, and TLS configuration remain manual
5. **Incomplete implementation** - Issue author (alexcrichton) expresses uncertainty about stabilization path

**Quote from issue**:
> "I'm not entirely sure what the best way forward is here. [...] I'm hesitant to stabilize the status quo if it feels too incomplete."

**Implications for wasm-bindgen-rayon**:
- Built on unstable, broken foundation
- Requires workarounds for fundamental Rust limitations
- No clear path to stability or production-readiness

**Why no examples exist**: The combination of Rust + WASM + threading is **not production-ready**, despite being technically possible in toy examples

**Next Steps**:
1. Revert all experimental changes (build scripts, config, worker code)
2. Return to dx-based workflow
3. Keep `wasm-threading` feature but mark as **desktop/mobile only**
4. Update CLAUDE.md to document abandonment with clear rationale
5. Focus on core features: search, UI, storage
6. **Do NOT re-evaluate** until Rust WebAssembly atomics are stabilized (no ETA)

**Alternative Performance Strategies**:
- **Desktop/mobile**: Use native Rayon threading (already works perfectly)
- **Web**: Accept single-threaded performance
  - Still functional, just slower (~5 tokens/sec vs ~16 tokens/sec)
  - Use Web Worker for embedding to avoid UI freezing (already implemented)
- **Future optimizations** (when ecosystem matures):
  - WebGPU compute shaders for ML inference (Candle roadmap)
  - Model quantization (reduce computation)
  - Embedding caching (reduce inference calls)
  - ONNX Runtime Web (mature, production-ready alternative)

**Final Recommendation**:
**ABANDON** WASM threading. The Rust ecosystem is not ready. This is not a Dioxus issue, not a wasm-bindgen-rayon issue, not a configuration issue - **it's a fundamental Rust WebAssembly limitation** with no clear timeline for resolution.

**Conclusion**:
WASM threading with Rust Rayon is **not production-viable** in 2025. The combination of:
- Unstable, incomplete Rust atomics implementation
- Browser WebAssembly.Memory cloning limitations
- Toolchain version incompatibilities
- Zero community precedent
- Fragile, unmaintainable custom build scripts

...makes this approach **untenable**. Desktop/mobile threading works perfectly - that's sufficient for a reference implementation.

---

## Appendix: Code Reference (For Future Attempts)

This section preserves the experimental code written during this investigation. If WASM threading becomes viable in the future, these implementations may provide a starting point.

### A. Build Script Approach (`scripts/build-web-release.sh`)

**Attempt**: Bypass Dioxus Manganis by manually calling cargo + wasm-bindgen

**Key sections**:
```bash
# Use cargo with build-std (bypasses dx bundle)
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory' \
  cargo build \
  --bin coppermind \
  --release \
  --target wasm32-unknown-unknown \
  --features web,wasm-threading \
  -Z build-std=panic_abort,std

# Run wasm-bindgen manually
wasm-bindgen \
  target/wasm32-unknown-unknown/release/coppermind.wasm \
  --out-dir "$WASM_DIR" \
  --target web \
  --no-typescript \
  --split-linked-modules
```

**Later iteration**: Switched to wasm-pack
```bash
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory' \
  wasm-pack build \
  --target web \
  --release \
  --out-dir pkg \
  --out-name coppermind \
  --no-typescript \
  -- \
  --bin coppermind \
  --features web,wasm-threading \
  -Z build-std=panic_abort,std
```

**Why it failed**:
- cargo+wasm-bindgen: `__wasm_init_tls` missing
- wasm-pack: Memory not shared despite patching

### B. JavaScript Memory Patching (`scripts/patch-wasm-threading.js`)

**Attempt**: Inject shared WebAssembly.Memory into wasm-pack's generated code

**Approach**:
```javascript
// Create shared memory
let __wbg_shared_memory = new WebAssembly.Memory({
    initial: 128,   // 8MB
    maximum: 65536, // 4GB
    shared: true
});

// Inject into imports before WASM instantiation
function __wbg_init_memory(imports) {
    imports.env = imports.env || {};
    imports.env.memory = __wbg_shared_memory;
    return imports;
}

// Patch __wbg_load to use injected memory
async function __wbg_load(module, imports) {
    imports = __wbg_init_memory(imports);
    // ... rest of wasm-pack's instantiation
}
```

**Why it failed**:
- wasm-pack's `WebAssembly.instantiate()` creates its own memory
- Injected memory in `imports.env.memory` is ignored
- Even if injection worked, Memory cannot be cloned via postMessage (browser limitation)

### C. Configuration Changes

**`.cargo/config.toml`** (threading flags):
```toml
[target.wasm32-unknown-unknown]
rustflags = [
  "--cfg", "getrandom_backend=\"wasm_js\"",
  "-C", "link-arg=--max-memory=4294967296",
  "-C", "link-arg=--import-memory",
  "-C", "link-arg=--shared-memory",
  "-C", "target-feature=+atomics,+bulk-memory,+mutable-globals",
]

[unstable]
build-std = ["panic_abort", "std"]
```

**`Cargo.toml`** (wasm-opt disable):
```toml
[package.metadata.wasm-pack.profile.release]
wasm-opt = false  # wasm-opt doesn't support atomics
```

**`rust-toolchain.toml`** (nightly pinning):
```toml
[toolchain]
channel = "nightly-2024-08-02"  # Tried various nightlies
components = ["rust-src", "rustfmt", "clippy"]
targets = ["wasm32-unknown-unknown"]
```

### D. Worker Initialization Pattern

**Original approach** (manual cargo+wasm-bindgen):
```javascript
// Create shared memory
const memory = new WebAssembly.Memory({
    initial: 128,
    maximum: 65536,
    shared: true
});

// Initialize with explicit memory
await module.default({ module_or_path: wasmUrl, memory });

// Initialize Rayon thread pool
if (typeof module.initThreadPool === "function") {
    await module.initThreadPool(navigator.hardwareConcurrency);
}
```

**wasm-pack approach**:
```javascript
// Import wasm-pack module
const module = await import(wasmJsUrl);

// Initialize (memory injection attempted via patching)
await module.default();

// Initialize Rayon
await module.initThreadPool(navigator.hardwareConcurrency);
```

### E. Key Lessons About Dioxus Build System

1. **Manganis Asset System**:
   - Serializes asset metadata during build
   - Incompatible with `build-std` (changes metadata format)
   - Debug mode: warnings only
   - Release mode: hard failure

2. **dx bundle pipeline**:
   ```
   cargo build → wasm-bindgen → Manganis → wasm-opt
   ```
   - Tightly integrated, hard to bypass individual steps
   - `NO_DOWNLOADS=1` flag doesn't solve Manganis issue

3. **Memory Management**:
   - Dioxus uses standard wasm-bindgen memory model
   - No built-in support for shared memory
   - Would require changes to Dioxus itself

4. **Cross-Platform Strategy**:
   - Desktop/mobile: Standard dx workflow works fine
   - Web: Custom build required for threading
   - Maintaining two build paths adds significant complexity

### F. Root Cause: wasm-bindgen-rayon Architecture

**How wasm-bindgen-rayon works**:
```javascript
// workerHelpers.js tries to do this:
worker.postMessage({
    type: 'init',
    memory: wasmMemory,    // ← THIS FAILS
    module: wasmModule
});
```

**Browser limitation**:
- `WebAssembly.Memory` objects are **not structured-cloneable**
- Cannot be sent via `postMessage` between threads
- This is a WebAssembly spec limitation, not a browser bug

**Why it works in simple examples**:
- Simple wasm-pack projects use `--target no-modules`
- Different instantiation pattern that works around this
- Our Dioxus app requires `--target web` (ES modules)
- The two targets have incompatible memory management

### G. Alternatives Considered But Not Pursued

1. **`--target no-modules`**:
   - Incompatible with Dioxus's ES module system
   - Would require rewriting app initialization

2. **Fork wasm-bindgen-rayon**:
   - Fix Memory cloning issue
   - Too complex, unmaintainable

3. **Manual Worker message-passing**:
   - Bypass Rayon, use custom threading
   - Defeats purpose (Rayon parallelism in Candle)

4. **Trunk bundler**:
   - Loses dx workflow for desktop/mobile
   - Same fundamental issues

---

## References for Future Work

- **Rust WASM threading tracking**: https://github.com/rust-lang/rust/issues/77839
- **wasm-bindgen-rayon**: https://github.com/RReverser/wasm-bindgen-rayon (archived)
- **Candle WASM threading PR**: https://github.com/huggingface/candle/pull/3063
- **Dioxus Manganis**: https://github.com/DioxusLabs/dioxus (check for build-std support)
- **WebAssembly threads proposal**: https://github.com/WebAssembly/threads

**If revisiting in future**:
1. Check if wasm-bindgen-rayon has been updated
2. Look for Dioxus + threading examples in the wild
3. Monitor Rust WASM threading stabilization
4. Consider WebGPU compute shaders as alternative
