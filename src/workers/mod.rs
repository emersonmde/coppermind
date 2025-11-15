#[cfg(target_arch = "wasm32")]
pub mod embedding_worker {
    use crate::embedding::{compute_embedding, EmbeddingComputation};
    use dioxus::prelude::*;
    use futures_channel::oneshot;
    use js_sys::{Float32Array, Object, Reflect};
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::spawn_local;
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, Worker, WorkerOptions, WorkerType};

    const EMBEDDING_WORKER_SCRIPT: Asset = asset!("/assets/workers/embedding-worker.js");

    // Type alias to reduce complexity
    type PendingRequests =
        Rc<RefCell<HashMap<u32, oneshot::Sender<Result<EmbeddingComputation, String>>>>>;

    #[derive(Clone)]
    pub struct EmbeddingWorkerClient {
        worker: Worker,
        pending: PendingRequests,
        next_id: Rc<Cell<u32>>,
        ready_state: Rc<RefCell<ReadyState>>,
        _on_message: Rc<Closure<dyn FnMut(MessageEvent)>>,
    }

    enum ReadyState {
        Pending(Vec<oneshot::Sender<Result<(), String>>>),
        Ready,
        Failed(String),
    }

    impl EmbeddingWorkerClient {
        pub fn new() -> Result<Self, String> {
            let script_url = EMBEDDING_WORKER_SCRIPT.to_string();
            let opts = WorkerOptions::new();
            opts.set_type(WorkerType::Module);
            let worker = Worker::new_with_options(&script_url, &opts)
                .map_err(|e| format!("Failed to spawn worker: {:?}", e))?;

            let pending = Rc::new(RefCell::new(HashMap::new()));
            let ready_state = Rc::new(RefCell::new(ReadyState::Pending(Vec::new())));
            let next_id = Rc::new(Cell::new(1u32));

            let pending_clone = pending.clone();
            let ready_clone = ready_state.clone();
            let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
                if let Err(err) = handle_worker_message(event, &pending_clone, &ready_clone) {
                    web_sys::console::error_1(
                        &format!("Embedding worker message error: {}", err).into(),
                    );
                }
            }) as Box<dyn FnMut(_)>);

            worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

            // Find the WASM JS path by querying the DOM for the main coppermind script
            // Then send it to the worker so it knows which WASM module to load
            let wasm_js_path = Self::find_wasm_js_path()?;
            web_sys::console::log_1(&format!("Found WASM JS path: {}", wasm_js_path).into());

            // Send init message to worker with the WASM JS path
            let init_msg = js_sys::Object::new();
            js_sys::Reflect::set(&init_msg, &"type".into(), &"init".into())
                .map_err(|_| "Failed to set type")?;
            js_sys::Reflect::set(&init_msg, &"wasmJsPath".into(), &wasm_js_path.into())
                .map_err(|_| "Failed to set wasmJsPath")?;
            worker
                .post_message(&init_msg)
                .map_err(|e| format!("Failed to post init message: {:?}", e))?;

            Ok(Self {
                worker,
                pending,
                ready_state,
                next_id,
                _on_message: Rc::new(on_message),
            })
        }

        fn find_wasm_js_path() -> Result<String, String> {
            let window = web_sys::window().ok_or("No window object")?;
            let document = window.document().ok_or("No document")?;

            // Find the script tag that loaded our main WASM module
            // Look for scripts with "coppermind" but NOT service worker or embedding worker
            let scripts = document
                .query_selector_all("script[src]")
                .map_err(|_| "Failed to query scripts")?;

            for i in 0..scripts.length() {
                if let Some(script) = scripts.item(i) {
                    if let Some(script_el) = script.dyn_ref::<web_sys::HtmlScriptElement>() {
                        let src = script_el.src();
                        // Must contain "coppermind" and end with ".js"
                        // But must NOT contain "coi-serviceworker" or "embedding-worker"
                        if src.contains("coppermind")
                            && src.ends_with(".js")
                            && !src.contains("coi-serviceworker")
                            && !src.contains("embedding-worker")
                        {
                            // Extract pathname from the full URL
                            if let Ok(url) = web_sys::Url::new(&src) {
                                return Ok(url.pathname());
                            }
                        }
                    }
                }
            }

            Err("Could not find coppermind script tag in document".to_string())
        }

        pub async fn embed(&self, text: String) -> Result<EmbeddingComputation, String> {
            self.wait_until_ready().await?;

            let request_id = self.next_id.get();
            self.next_id.set(request_id.wrapping_add(1));

            let (tx, rx) = oneshot::channel();
            self.pending.borrow_mut().insert(request_id, tx);

            let msg = Object::new();
            Reflect::set(
                &msg,
                &"requestId".into(),
                &JsValue::from_f64(request_id as f64),
            )
            .map_err(|e| format!("Failed to set requestId: {:?}", e))?;
            Reflect::set(&msg, &"text".into(), &JsValue::from_str(&text))
                .map_err(|e| format!("Failed to set text: {:?}", e))?;

            self.worker
                .post_message(&msg)
                .map_err(|e| format!("Failed to post embedding job: {:?}", e))?;

            rx.await
                .map_err(|_| "Embedding worker dropped response".to_string())?
        }

        async fn wait_until_ready(&self) -> Result<(), String> {
            {
                let state = self.ready_state.borrow();
                match &*state {
                    ReadyState::Ready => return Ok(()),
                    ReadyState::Failed(err) => return Err(err.clone()),
                    ReadyState::Pending(_) => {}
                }
            }

            let (tx, rx) = oneshot::channel();
            {
                let mut state = self.ready_state.borrow_mut();
                match &mut *state {
                    ReadyState::Pending(waiters) => waiters.push(tx),
                    ReadyState::Ready => return Ok(()),
                    ReadyState::Failed(err) => return Err(err.clone()),
                }
            }

            rx.await
                .map_err(|_| "Embedding worker readiness channel closed".to_string())?
        }
    }

    fn handle_worker_message(
        event: MessageEvent,
        pending: &PendingRequests,
        ready_state: &Rc<RefCell<ReadyState>>,
    ) -> Result<(), String> {
        let data = event.data();
        let obj = data
            .dyn_into::<Object>()
            .map_err(|_| "Worker message was not an object".to_string())?;

        let msg_type = Reflect::get(&obj, &"type".into()).unwrap_or(JsValue::UNDEFINED);
        if msg_type.is_undefined() {
            return Err("Worker message missing type".into());
        }

        let msg_type_str = msg_type
            .as_string()
            .ok_or_else(|| "Worker message type not a string".to_string())?;

        match msg_type_str.as_str() {
            "ready" => {
                notify_ready(ready_state);
                Ok(())
            }
            "worker-error" => {
                let error_text = Reflect::get(&obj, &"error".into())
                    .unwrap_or(JsValue::from("Unknown worker error"))
                    .as_string()
                    .unwrap_or_else(|| "Unknown worker error".into());
                notify_failure(ready_state, &error_text);
                drain_pending_with_error(pending, &error_text);
                Err(error_text)
            }
            "embedding" => {
                let request_id = Reflect::get(&obj, &"requestId".into())
                    .map_err(|e| format!("Missing requestId: {:?}", e))?
                    .as_f64()
                    .ok_or_else(|| "requestId not numeric".to_string())?
                    as u32;
                let token_count = Reflect::get(&obj, &"tokenCount".into())
                    .map_err(|e| format!("Missing tokenCount: {:?}", e))?
                    .as_f64()
                    .ok_or_else(|| "tokenCount not numeric".to_string())?
                    as usize;
                let embedding_val = Reflect::get(&obj, &"embedding".into())
                    .map_err(|e| format!("Missing embedding: {:?}", e))?;
                let typed_array = embedding_val
                    .dyn_into::<Float32Array>()
                    .map_err(|_| "embedding field was not Float32Array".to_string())?;
                let mut buffer = vec![0f32; typed_array.length() as usize];
                typed_array.copy_to(&mut buffer);

                if let Some(sender) = pending.borrow_mut().remove(&request_id) {
                    let _ = sender.send(Ok(EmbeddingComputation {
                        token_count,
                        embedding: buffer,
                    }));
                }
                Ok(())
            }
            "embedding-error" => {
                let request_id = Reflect::get(&obj, &"requestId".into())
                    .map_err(|e| format!("Missing requestId: {:?}", e))?
                    .as_f64()
                    .ok_or_else(|| "requestId not numeric".to_string())?
                    as u32;
                let error_text = Reflect::get(&obj, &"error".into())
                    .unwrap_or(JsValue::from("Unknown embedding error"))
                    .as_string()
                    .unwrap_or_else(|| "Unknown embedding error".into());

                if let Some(sender) = pending.borrow_mut().remove(&request_id) {
                    let _ = sender.send(Err(error_text.clone()));
                }
                Err(error_text)
            }
            other => Err(format!("Unknown worker message type: {}", other)),
        }
    }

    fn notify_ready(ready_state: &Rc<RefCell<ReadyState>>) {
        let waiters = match std::mem::replace(&mut *ready_state.borrow_mut(), ReadyState::Ready) {
            ReadyState::Pending(waiters) => waiters,
            ReadyState::Ready => return,
            ReadyState::Failed(_) => return,
        };
        for waiter in waiters {
            let _ = waiter.send(Ok(()));
        }
    }

    fn notify_failure(ready_state: &Rc<RefCell<ReadyState>>, error: &str) {
        let error_msg = error.to_string();
        let previous = std::mem::replace(
            &mut *ready_state.borrow_mut(),
            ReadyState::Failed(error_msg.clone()),
        );
        if let ReadyState::Pending(waiters) = previous {
            for waiter in waiters {
                let _ = waiter.send(Err(error_msg.clone()));
            }
        }
    }

    fn drain_pending_with_error(pending: &PendingRequests, error: &str) {
        let mut pending_map = pending.borrow_mut();
        for (_, sender) in pending_map.drain() {
            let _ = sender.send(Err(error.to_string()));
        }
    }

    #[derive(serde::Deserialize)]
    struct WorkerRequest {
        #[serde(rename = "requestId")]
        request_id: u32,
        text: String,
    }

    #[wasm_bindgen]
    pub fn start_embedding_worker() -> Result<(), JsValue> {
        console_error_panic_hook::set_once();
        let scope: DedicatedWorkerGlobalScope = js_sys::global()
            .dyn_into()
            .map_err(|_| JsValue::from_str("Not running inside a worker scope"))?;

        let handler = Closure::wrap(Box::new(move |event: MessageEvent| {
            let request: WorkerRequest = match serde_wasm_bindgen::from_value(event.data()) {
                Ok(req) => req,
                Err(err) => {
                    web_sys::console::error_1(
                        &format!("Malformed worker request: {:?}", err).into(),
                    );
                    return;
                }
            };

            spawn_local(async move {
                let response = match compute_embedding(&request.text).await {
                    Ok(result) => build_success_payload(request.request_id, result),
                    Err(err) => build_error_payload(Some(request.request_id), &err.to_string()),
                };

                if let Err(e) = post_message_to_main(&response) {
                    web_sys::console::error_1(
                        &format!("Failed to post worker response: {:?}", e).into(),
                    );
                }
            });
        }) as Box<dyn FnMut(_)>);

        scope.set_onmessage(Some(handler.as_ref().unchecked_ref()));
        handler.forget();

        let ready_msg = build_ready_payload();
        post_message_to_main(&ready_msg)?;
        Ok(())
    }

    fn post_message_to_main(msg: &JsValue) -> Result<(), JsValue> {
        let global = js_sys::global();
        let scope: DedicatedWorkerGlobalScope = global
            .dyn_into()
            .map_err(|_| JsValue::from_str("Not running inside a worker scope"))?;
        scope.post_message(msg)
    }

    fn build_ready_payload() -> JsValue {
        let payload = Object::new();
        let _ = Reflect::set(&payload, &"type".into(), &JsValue::from_str("ready"));
        payload.into()
    }

    fn build_success_payload(request_id: u32, data: EmbeddingComputation) -> JsValue {
        let payload = Object::new();
        let _ = Reflect::set(&payload, &"type".into(), &JsValue::from_str("embedding"));
        let _ = Reflect::set(
            &payload,
            &"requestId".into(),
            &JsValue::from_f64(request_id as f64),
        );
        let _ = Reflect::set(
            &payload,
            &"tokenCount".into(),
            &JsValue::from_f64(data.token_count as f64),
        );
        let array = Float32Array::from(data.embedding.as_slice());
        let _ = Reflect::set(&payload, &"embedding".into(), array.as_ref());
        payload.into()
    }

    fn build_error_payload(request_id: Option<u32>, message: &str) -> JsValue {
        let payload = Object::new();
        let msg_type = if request_id.is_some() {
            "embedding-error"
        } else {
            "worker-error"
        };
        let _ = Reflect::set(&payload, &"type".into(), &JsValue::from_str(msg_type));
        if let Some(id) = request_id {
            let _ = Reflect::set(&payload, &"requestId".into(), &JsValue::from_f64(id as f64));
        }
        let _ = Reflect::set(&payload, &"error".into(), &JsValue::from_str(message));
        payload.into()
    }
}

#[cfg(target_arch = "wasm32")]
pub use embedding_worker::{start_embedding_worker, EmbeddingWorkerClient};
