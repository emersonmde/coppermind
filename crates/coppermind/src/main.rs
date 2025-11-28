use coppermind::components::App as CoppermindApp;
use dioxus::prelude::*;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const APPLE_TOUCH_ICON: Asset = asset!("/assets/favicon-192.png");
const MAIN_CSS: Asset = asset!("/assets/coppermind.css");

/// Initialize Chrome tracing for performance profiling.
/// Returns a guard that must be held until program exit to flush the trace file.
///
/// IMPORTANT: Only traces spans from our crates (coppermind, coppermind_core) to avoid
/// capturing the massive call stacks from dependencies (dioxus, tokio, candle, etc.)
/// which would create gigabyte-sized trace files.
#[cfg(feature = "profile")]
fn init_profiling() -> tracing_chrome::FlushGuard {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    // Filter to only capture spans from our crates, not dependencies
    // This prevents gigabyte-sized traces from tokio/dioxus/candle internals
    let trace_filter = EnvFilter::new("coppermind=trace,coppermind_core=trace");

    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file("./trace.json")
        .include_args(true)
        .build();

    // Also add a console logger so we still see log output
    let console_layer = fmt::layer()
        .with_target(true)
        .with_level(true)
        .with_filter(EnvFilter::new("coppermind=info,coppermind_core=info"));

    tracing_subscriber::registry()
        .with(chrome_layer.with_filter(trace_filter))
        .with(console_layer)
        .init();

    tracing::info!("Profiling enabled - trace will be written to ./trace.json");
    tracing::info!("Tracing only: coppermind, coppermind_core (dependencies filtered out)");
    guard
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        let window = web_sys::window();
        let has_document = window.as_ref().and_then(|w| w.document()).is_some();

        if window.is_none() || !has_document {
            // Running inside a Web Worker — skip mounting the UI.
            return;
        }
    }

    // Profiling mode: use tracing-chrome instead of dioxus logger
    // The guard must be held until program exit to flush the trace file
    #[cfg(feature = "profile")]
    let _profiling_guard = init_profiling();

    // Standard logging (skip when profiling - tracing-chrome handles it)
    #[cfg(not(feature = "profile"))]
    {
        // Initialize cross-platform logger (web console + desktop stdout)
        // Use DEBUG level for development builds, INFO for release builds
        #[cfg(debug_assertions)]
        dioxus::logger::init(dioxus::logger::tracing::Level::DEBUG).expect("logger failed to init");
        #[cfg(not(debug_assertions))]
        dioxus::logger::init(dioxus::logger::tracing::Level::INFO).expect("logger failed to init");
    }

    // Platform-specific launch configuration
    #[cfg(feature = "desktop")]
    {
        use dioxus::desktop::{Config, LogicalSize, WindowBuilder};

        let config = Config::default().with_window(
            WindowBuilder::new()
                .with_title("Coppermind")
                .with_resizable(true)
                .with_inner_size(LogicalSize::new(1200.0, 900.0))
                .with_min_inner_size(LogicalSize::new(800.0, 600.0))
                // Set dark background to prevent white flash on overscroll
                .with_transparent(false),
        );

        dioxus::LaunchBuilder::desktop()
            .with_cfg(config)
            .launch(App);
    }

    #[cfg(feature = "mobile")]
    {
        dioxus::LaunchBuilder::mobile().launch(App);
    }

    #[cfg(feature = "web")]
    {
        dioxus::launch(App);
    }
}

#[component]
fn App() -> Element {
    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "apple-touch-icon", href: APPLE_TOUCH_ICON }

        // CSS loading: asset! macro has issues on desktop, use include_str! as workaround
        if cfg!(target_arch = "wasm32") {
            document::Stylesheet { href: MAIN_CSS }
        } else {
            style { {include_str!("../assets/coppermind.css")} }
        }

        body { class: "cm-body",
            CoppermindApp {}
        }
    }
}
