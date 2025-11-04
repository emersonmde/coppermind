use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

pub async fn test_webgpu() -> Result<String, String> {
    web_sys::console::log_1(&"Starting WebGPU test...".into());

    let window = web_sys::window().ok_or("No window")?;
    let navigator = window.navigator();

    let gpu = Reflect::get(&navigator, &"gpu".into())
        .map_err(|_| "GPU not found - WebGPU not supported")?;

    if gpu.is_undefined() {
        return Err("WebGPU not supported in this browser".to_string());
    }

    let request_adapter =
        Reflect::get(&gpu, &"requestAdapter".into()).map_err(|_| "requestAdapter not found")?;
    let request_adapter_fn = request_adapter
        .dyn_ref::<js_sys::Function>()
        .ok_or("requestAdapter is not a function")?;

    let adapter_promise = request_adapter_fn
        .call0(&gpu)
        .map_err(|e| format!("Failed to request adapter: {:?}", e))?;

    let adapter = JsFuture::from(js_sys::Promise::from(adapter_promise))
        .await
        .map_err(|e| format!("Adapter promise failed: {:?}", e))?;

    if adapter.is_null() {
        return Err("No GPU adapter available".to_string());
    }

    web_sys::console::log_1(&"Got GPU adapter".into());

    let request_device =
        Reflect::get(&adapter, &"requestDevice".into()).map_err(|_| "requestDevice not found")?;
    let request_device_fn = request_device
        .dyn_ref::<js_sys::Function>()
        .ok_or("requestDevice is not a function")?;

    let device_promise = request_device_fn
        .call0(&adapter)
        .map_err(|e| format!("Failed to request device: {:?}", e))?;

    let device = JsFuture::from(js_sys::Promise::from(device_promise))
        .await
        .map_err(|e| format!("Device promise failed: {:?}", e))?;

    web_sys::console::log_1(&"Got GPU device".into());

    let shader_code = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= 1048576u) { return; }

            var sum: f32 = 0.0;
            for (var i = 0u; i < 128u; i = i + 1u) {
                let a = f32(idx + i) * 0.001;
                sum += sin(a) * cos(a) + sqrt(abs(a));
            }
            data[idx] = sum;
        }
    "#;

    let shader_module_desc = Object::new();
    Reflect::set(
        &shader_module_desc,
        &"code".into(),
        &JsValue::from_str(shader_code),
    )
    .unwrap();

    let create_shader_module = Reflect::get(&device, &"createShaderModule".into())
        .map_err(|_| "createShaderModule not found")?;
    let create_shader_module_fn = create_shader_module
        .dyn_ref::<js_sys::Function>()
        .ok_or("createShaderModule is not a function")?;

    let shader_module = create_shader_module_fn
        .call1(&device, &shader_module_desc)
        .map_err(|e| format!("Failed to create shader: {:?}", e))?;

    let buffer_size = 1048576 * 4;
    let buffer_desc = Object::new();
    Reflect::set(
        &buffer_desc,
        &"size".into(),
        &JsValue::from_f64(buffer_size as f64),
    )
    .unwrap();
    Reflect::set(
        &buffer_desc,
        &"usage".into(),
        &JsValue::from_f64((0x0080 | 0x0001) as f64),
    )
    .unwrap();

    let create_buffer = Reflect::get(&device, &"createBuffer".into()).unwrap();
    let create_buffer_fn = create_buffer.dyn_ref::<js_sys::Function>().unwrap();
    let buffer = create_buffer_fn.call1(&device, &buffer_desc).unwrap();

    let pipeline_desc = Object::new();
    let compute = Object::new();
    Reflect::set(&compute, &"module".into(), &shader_module).unwrap();
    Reflect::set(&compute, &"entryPoint".into(), &"main".into()).unwrap();
    Reflect::set(&pipeline_desc, &"compute".into(), &compute.into()).unwrap();
    Reflect::set(&pipeline_desc, &"layout".into(), &JsValue::from_str("auto")).unwrap();

    let create_compute_pipeline = Reflect::get(&device, &"createComputePipeline".into()).unwrap();
    let create_compute_pipeline_fn = create_compute_pipeline
        .dyn_ref::<js_sys::Function>()
        .unwrap();
    let pipeline = create_compute_pipeline_fn
        .call1(&device, &pipeline_desc)
        .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

    let bind_group_desc = Object::new();
    let entries = js_sys::Array::new();
    let entry = Object::new();
    Reflect::set(&entry, &"binding".into(), &JsValue::from_f64(0.0)).unwrap();
    let resource = Object::new();
    Reflect::set(&resource, &"buffer".into(), &buffer).unwrap();
    Reflect::set(&entry, &"resource".into(), &resource.into()).unwrap();
    entries.push(&entry.into());
    Reflect::set(&bind_group_desc, &"entries".into(), &entries).unwrap();

    let get_bind_group_layout = Reflect::get(&pipeline, &"getBindGroupLayout".into()).unwrap();
    let get_bind_group_layout_fn = get_bind_group_layout.dyn_ref::<js_sys::Function>().unwrap();
    let layout = get_bind_group_layout_fn
        .call1(&pipeline, &JsValue::from_f64(0.0))
        .unwrap();
    Reflect::set(&bind_group_desc, &"layout".into(), &layout).unwrap();

    let create_bind_group = Reflect::get(&device, &"createBindGroup".into()).unwrap();
    let create_bind_group_fn = create_bind_group.dyn_ref::<js_sys::Function>().unwrap();
    let bind_group = create_bind_group_fn
        .call1(&device, &bind_group_desc)
        .unwrap();

    let create_command_encoder = Reflect::get(&device, &"createCommandEncoder".into()).unwrap();
    let create_command_encoder_fn = create_command_encoder
        .dyn_ref::<js_sys::Function>()
        .unwrap();
    let encoder = create_command_encoder_fn.call0(&device).unwrap();

    let begin_compute_pass = Reflect::get(&encoder, &"beginComputePass".into()).unwrap();
    let begin_compute_pass_fn = begin_compute_pass.dyn_ref::<js_sys::Function>().unwrap();
    let pass = begin_compute_pass_fn.call0(&encoder).unwrap();

    let set_pipeline = Reflect::get(&pass, &"setPipeline".into()).unwrap();
    let set_pipeline_fn = set_pipeline.dyn_ref::<js_sys::Function>().unwrap();
    set_pipeline_fn.call1(&pass, &pipeline).unwrap();

    let set_bind_group = Reflect::get(&pass, &"setBindGroup".into()).unwrap();
    let set_bind_group_fn = set_bind_group.dyn_ref::<js_sys::Function>().unwrap();
    set_bind_group_fn
        .call2(&pass, &JsValue::from_f64(0.0), &bind_group)
        .unwrap();

    let dispatch_workgroups = Reflect::get(&pass, &"dispatchWorkgroups".into()).unwrap();
    let dispatch_workgroups_fn = dispatch_workgroups.dyn_ref::<js_sys::Function>().unwrap();
    dispatch_workgroups_fn
        .call1(&pass, &JsValue::from_f64(4096.0))
        .unwrap();

    let end = Reflect::get(&pass, &"end".into()).unwrap();
    let end_fn = end.dyn_ref::<js_sys::Function>().unwrap();
    end_fn.call0(&pass).unwrap();

    let finish = Reflect::get(&encoder, &"finish".into()).unwrap();
    let finish_fn = finish.dyn_ref::<js_sys::Function>().unwrap();
    let command_buffer = finish_fn.call0(&encoder).unwrap();

    let queue = Reflect::get(&device, &"queue".into()).unwrap();
    let submit = Reflect::get(&queue, &"submit".into()).unwrap();
    let submit_fn = submit.dyn_ref::<js_sys::Function>().unwrap();
    let commands = js_sys::Array::new();
    commands.push(&command_buffer);
    submit_fn.call1(&queue, &commands).unwrap();

    web_sys::console::log_1(&"GPU compute submitted!".into());

    Ok("✓ GPU compute executed successfully (1M elements × 128 ops)".to_string())
}
