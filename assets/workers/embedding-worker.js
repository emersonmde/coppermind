const workerContext = self;

function resolveBasePath() {
    const path = new URL(workerContext.location.href).pathname;
    const marker = "/assets/";
    const markerIndex = path.indexOf(marker);
    if (markerIndex !== -1) {
        return path.slice(0, markerIndex);
    }
    const lastSlash = path.lastIndexOf("/");
    if (lastSlash <= 0) {
        return "";
    }
    return path.slice(0, lastSlash);
}

function joinBase(base, suffix) {
    const normalized = base === "/" ? "" : base.replace(/\/$/, "");
    return `${normalized}${suffix}`;
}

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
    const basePath = resolveBasePath();
    const wasmJsUrl = joinBase(basePath, "/wasm/coppermind.js");
    const wasmBinUrl = joinBase(basePath, "/wasm/coppermind_bg.wasm");

    try {
        workerContext.__COPPERMIND_ASSET_BASE = basePath || "";
        console.log(
            `[EmbeddingWorker] asset base='${workerContext.__COPPERMIND_ASSET_BASE}'`,
        );
        console.log(
            `[EmbeddingWorker] wasm bootstrap js=${wasmJsUrl}, wasm=${wasmBinUrl}`,
        );

        shimDomForWorker();
        const module = await import(wasmJsUrl);
        if (typeof module.default !== "function") {
            throw new Error("coppermind wasm init function missing");
        }

        await module.default({ module_or_path: wasmBinUrl });

        if (typeof module.start_embedding_worker !== "function") {
            throw new Error("start_embedding_worker export not found");
        }

        module.start_embedding_worker();
    } catch (error) {
        const message = (error && error.message) ? error.message : String(error);
        const stack = error && error.stack ? ` | stack: ${error.stack}` : "";
        workerContext.postMessage({ type: "worker-error", error: `init: ${message}` });
        console.error(`[EmbeddingWorker] bootstrap failure: ${message}${stack}`);
    }
}

boot();
