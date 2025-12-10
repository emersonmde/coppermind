const workerContext = self;

// Guard against double initialization (can happen on page refresh with cached modules)
let bootStarted = false;
let wasmInitialized = false;

// Wait for init message from main thread with WASM JS path
let wasmJsPath = null;
let initResolve;
const initPromise = new Promise((resolve) => {
    initResolve = resolve;
});

workerContext.addEventListener("message", (event) => {
    const data = event.data;
    if (data && data.type === "init") {
        wasmJsPath = data.wasmJsPath;
        console.log(`[EmbeddingWorker] Received WASM JS path: ${wasmJsPath}`);
        if (initResolve) {
            initResolve();
        }
    }
});

function shimDomForWorker() {
    if (typeof workerContext.window === "undefined") {
        workerContext.window = workerContext;
    }

    if (typeof workerContext.document === "undefined") {
        const noop = () => {};
        const elementStub = () => ({
            setAttribute: noop,
            appendChild: noop,
            addEventListener: noop,
            removeEventListener: noop,
            style: {},
        });

        workerContext.document = {
            createElement: elementStub,
            createElementNS: elementStub,
            head: { appendChild: noop },
            body: { appendChild: noop },
            documentElement: { style: {} },
            querySelector: () => null,
            getElementById: () => null,
            addEventListener: noop,
            removeEventListener: noop,
            currentScript: null,
        };
    }
}

async function boot() {
    // Prevent double initialization
    if (bootStarted) {
        console.warn("[EmbeddingWorker] boot() already started, skipping duplicate call");
        return;
    }
    bootStarted = true;

    try {
        // Wait for init message from main thread
        console.log("[EmbeddingWorker] Waiting for init message...");
        await initPromise;

        if (!wasmJsPath) {
            throw new Error("WASM JS path not provided by main thread");
        }

        console.log(`[EmbeddingWorker] Starting bootstrap with: ${wasmJsPath}`);

        // Shim DOM APIs that wasm-bindgen expects
        shimDomForWorker();

        // Convert to absolute URL for worker context
        const absoluteJsUrl = new URL(wasmJsPath, workerContext.location.href).href;
        console.log(`[EmbeddingWorker] Fetching JS to find WASM path: ${absoluteJsUrl}`);

        // Fetch the JS file to extract the hardcoded WASM path
        // The generated code contains: __wbg_init({module_or_path:"/path/to/file.wasm"})
        const jsResponse = await fetch(absoluteJsUrl);
        const jsText = await jsResponse.text();

        // Extract WASM path from the JS code
        // Handle optional whitespace: module_or_path:"..." or module_or_path: "..."
        const wasmPathMatch = jsText.match(/module_or_path\s*:\s*"([^"]+\.wasm)"/);
        if (!wasmPathMatch) {
            throw new Error("Could not find WASM path in generated JS");
        }
        const wasmPath = wasmPathMatch[1];
        console.log(`[EmbeddingWorker] Found WASM path: ${wasmPath}`);

        // Convert WASM path to absolute URL
        const absoluteWasmUrl = new URL(wasmPath, workerContext.location.href).href;
        console.log(`[EmbeddingWorker] Loading WASM from: ${absoluteWasmUrl}`);

        // Import the WASM JS module
        const module = await import(absoluteJsUrl);
        if (typeof module.default !== "function") {
            throw new Error("coppermind wasm init function missing");
        }

        // Initialize WASM with explicit path (guard against double init)
        if (!wasmInitialized) {
            console.log("[EmbeddingWorker] Initializing WASM module...");
            await module.default(absoluteWasmUrl);
            wasmInitialized = true;
        } else {
            console.warn("[EmbeddingWorker] WASM already initialized, skipping");
        }

        if (typeof module.start_embedding_worker !== "function") {
            throw new Error("start_embedding_worker export not found");
        }

        console.log("[EmbeddingWorker] Starting embedding worker...");
        module.start_embedding_worker();
    } catch (error) {
        const message = (error && error.message) ? error.message : String(error);
        const stack = error && error.stack ? ` | stack: ${error.stack}` : "";
        workerContext.postMessage({ type: "worker-error", error: `init: ${message}` });
        console.error(`[EmbeddingWorker] bootstrap failure: ${message}${stack}`);
    }
}

boot();
