# Repository Guidelines

## Project Structure & Module Organization
Rust application code lives in `src/`, with UI entrypoints in `main.rs` and `components.rs`, ML logic in `embedding.rs` and `cpu.rs`, WebGPU helpers in `wgpu.rs`, and the hybrid search pipeline under `search/` (vector, BM25, fusion, and engine modules). Persistent chunk and embedding handling sits in `storage/`. Browser assets and the COOP/COEP service worker are in `public/`, downloadable model blobs land in `assets/models/` via `download-models.sh`, and long-form architecture notes live in `docs/`. Build outputs stay in `target/`; do not commit anything from there.

## Build, Test, and Development Commands
- `dx serve -p coppermind` — launches the hot-reloading web preview; add `--platform desktop` for the native shell.
- `dx bundle -p coppermind --release [--platform desktop]` — produces optimized WASM or desktop bundles.
- `./download-models.sh` — fetches the JinaBERT weights required for local inference.
- `cargo test --verbose` — runs Rust unit/integration suites across all crates.
- `./.githooks/pre-commit` — executes fmt, clippy, tests, audit, and docs in the same order as CI.
- `cargo fmt --check && cargo clippy --all-targets -- -D warnings` — quick lint gate before pushing.

## Coding Style & Naming Conventions
Use stable Rustfmt defaults (4-space indents, trailing commas, grouped `use` blocks) enforced by `cargo fmt`. Modules, files, and functions follow `snake_case`; structs, enums, and traits use `PascalCase`; constants stay `SCREAMING_SNAKE_CASE`. Prefer explicit types over `impl Trait` in public APIs, avoid `unwrap` outside initialization, and follow `clippy.toml`’s Dioxus signal-safety lints. Keep UI components stateless when possible and co-locate CSS-like constants near the component that uses them.

## Testing Guidelines
Write focused Rust tests alongside the code under test (`src/search/vector.rs` has `#[cfg(test)]` blocks as the pattern). Favor descriptive names such as `search_results_rank_semantic_hits`. For browser-only logic, lean on Dioxus' `dx serve -p coppermind --hot` plus `wasm-bindgen-test` stubs to confirm WASM behaviors. Always run `cargo test --verbose` before opening a PR; add targeted tests for regressions in embeddings, chunking, or scoring, and keep coverage high for storage boundary conditions.

## Commit & Pull Request Guidelines
Commits should be concise, imperative, and scoped (e.g., `feat: tighten bm25 fusion weights`); include issue numbers where relevant. Squash noisy WIP commits before review. Pull requests need: summary of behavioral changes, testing/benchmark notes (`cargo test`, `dx bundle`, etc.), and screenshots or terminal logs when UI or CLI output shifts. Link roadmap checklist items in `docs/roadmap.md` when you close them, and ensure CI (`Dioxus CI`) is green before requesting review.

## Security & Configuration Tips
Keep `.cargo/config.toml` aligned with current WASM memory limits; bump both the file and docs when changing max pages. Never commit downloaded models—`assets/models/` stays in `.gitignore`. When touching service-worker or COOP/COEP settings in `public/`, re-run a cold `dx serve -p coppermind` session to confirm SharedArrayBuffer still works, and document any new headers in `docs/browser-ml-architecture.md`.
