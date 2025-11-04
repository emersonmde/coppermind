# Development

Your new bare-bones project includes minimal organization with a single `main.rs` file and a few assets.

```
project/
├─ assets/ # Any assets that are used by the app should be placed here
├─ src/
│  ├─ main.rs # main.rs is the entry point to your application and currently contains all components for the app
├─ Cargo.toml # The Cargo.toml file defines the dependencies and feature flags for your project
```

### Serving Your App

Run the following command in the root of your project to start developing with the default platform:

```bash
dx serve
```

To run for a different platform, use the `--platform platform` flag. E.g.
```bash
dx serve --platform desktop
```

## CI and Deployment

- GitHub Actions workflow `.github/workflows/ci.yml` runs formatting, Clippy, tests, security audit, and doc builds on each push and pull request.
- A release build of the Dioxus web app is produced with `dx build --release --platform web`; the generated bundle is uploaded as an artifact for preview.
- Pushes to `main` automatically publish the latest web bundle to the `gh-pages` branch through `JamesIves/github-pages-deploy-action`.
- Configure the repository's Pages settings to serve from the `gh-pages` branch so the latest build is available via GitHub Pages.
- A repo-managed pre-commit hook (`.githooks/pre-commit`) mirrors the workflow checks locally; enable it with `git config core.hooksPath .githooks` and ensure `cargo-audit` and `dioxus-cli` are installed via `cargo install cargo-audit --locked` and `cargo install dioxus-cli --locked`.
