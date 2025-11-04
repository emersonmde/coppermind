use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Worker, MessageEvent, Blob, Url};
use js_sys::{Object, Reflect, Array};

pub async fn spawn_worker(worker_id: usize, text: &str, iterations: usize) -> Result<String, String> {
    let worker_code = r#"
        self.onmessage = function(e) {
            const { workerId, iterations } = e.data;
            let result = 0;
            for (let i = 0; i < iterations; i++) {
                for (let j = 0; j < 50000; j++) {
                    result += Math.sqrt(i * j) * Math.sin(i) * Math.cos(j);
                    result += Math.pow(Math.abs(Math.sin(j)), 0.37);
                }
            }
            self.postMessage({ workerId, result });
        };
    "#;

    let parts = Array::new();
    parts.push(&JsValue::from_str(worker_code));
    let blob = Blob::new_with_str_sequence(&parts)
        .map_err(|e| format!("Failed to create blob: {:?}", e))?;
    let url = Url::create_object_url_with_blob(&blob)
        .map_err(|e| format!("Failed to create URL: {:?}", e))?;
    let worker = Worker::new(&url)
        .map_err(|e| format!("Failed to create worker: {:?}", e))?;

    let (sender, receiver) = futures_channel::oneshot::channel();
    let sender = std::rc::Rc::new(std::cell::RefCell::new(Some(sender)));

    let worker_id_clone = worker_id;
    let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
        if let Ok(data) = event.data().dyn_into::<Object>() {
            if let Ok(_) = Reflect::get(&data, &"workerId".into()) {
                if let Some(sender) = sender.borrow_mut().take() {
                    let _ = sender.send(format!("Worker {} completed", worker_id_clone));
                }
            }
        }
    }) as Box<dyn FnMut(_)>);

    worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    let msg = Object::new();
    Reflect::set(&msg, &"workerId".into(), &JsValue::from(worker_id)).unwrap();
    Reflect::set(&msg, &"text".into(), &JsValue::from(text)).unwrap();
    Reflect::set(&msg, &"iterations".into(), &JsValue::from(iterations)).unwrap();

    worker.post_message(&msg).map_err(|e| format!("Failed to post message: {:?}", e))?;

    receiver.await.map_err(|e| format!("Worker timeout: {:?}", e))
}
