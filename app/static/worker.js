
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.2.4';

// Skip local model checks since we are running in browser
env.allowLocalModels = false;

// Disable multi-threading to avoid SecurityError with CDN-hosted worker scripts
env.backends.onnx.wasm.numThreads = 1;

// Check if WebGPU is available
async function isWebGPUAvailable() {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
    } catch {
        return false;
    }
}

// Models that should use WebGPU (q4f16 quantized for GPU)
const WEBGPU_MODELS = [
    'onnx-community/Llama-3.2-1B-Instruct-ONNX',
];

class PipelineSingleton {
    static task = 'text-generation';
    static model = null;
    static instance = null;
    static loading = false;
    static webgpuAvailable = null;

    static async getInstance(progress_callback = null, model_id = null) {
        // Check WebGPU availability once
        if (this.webgpuAvailable === null) {
            this.webgpuAvailable = await isWebGPUAvailable();
            console.log("[Worker] WebGPU available:", this.webgpuAvailable);
        }

        // If a specific model is requested and it's different from the loaded one, reset.
        if (model_id && this.model !== model_id) {
            console.log("[Worker] Switching model from", this.model, "to", model_id);
            this.instance = null;
            this.model = model_id;
        }

        // If no model specified, use current or default
        if (!this.model) {
            this.model = 'Xenova/TinyLlama-1.1B-Chat-v1.0';
        }

        // If instance exists AND was awaited (is a generator), return it
        if (this.instance && typeof this.instance.then !== 'function') {
            console.log("[Worker] Returning cached instance for", this.model);
            return this.instance;
        }

        // If currently loading, wait for it
        if (this.loading && this.instance) {
            console.log("[Worker] Waiting for pending load of", this.model);
            return await this.instance;
        }

        // Determine device based on model and WebGPU availability
        const useWebGPU = this.webgpuAvailable && WEBGPU_MODELS.some(m => this.model.includes(m));
        const device = useWebGPU ? 'webgpu' : undefined; // undefined = default WASM

        console.log("[Worker] Loading new model:", this.model, "| Device:", device || 'wasm');
        this.loading = true;

        const pipelineOptions = {
            progress_callback,
        };

        // Only add device if using WebGPU (let WASM models use default quantization)
        if (device) {
            pipelineOptions.device = device;
            pipelineOptions.dtype = 'q4'; // Use 4-bit quantization
        }

        this.instance = pipeline(this.task, this.model, pipelineOptions);

        // Await and store the resolved instance
        const resolved = await this.instance;
        this.instance = resolved;
        this.loading = false;
        return resolved;
    }
}

self.addEventListener('message', async (event) => {
    const { type, data } = event.data;

    switch (type) {
        case 'load':
            await load(data);
            break;
        case 'generate':
            await generate(data);
            break;
    }
});

async function load(data) {
    const model_id = (typeof data === 'string') ? data : data?.model;
    const displayName = model_id ? model_id.split('/').pop() : 'default model';

    try {
        self.postMessage({ status: 'loading', data: "Loading " + displayName + "...", model: model_id });

        await PipelineSingleton.getInstance(x => {
            self.postMessage({
                status: 'progress',
                data: {
                    file: x.file || '',
                    progress: x.progress || 0,
                    loaded: x.loaded || 0,
                    total: x.total || 0,
                    status: x.status || 'unknown',
                    model: displayName
                }
            });
        }, model_id);

        self.postMessage({ status: 'ready', model: model_id || PipelineSingleton.model });
    } catch (e) {
        self.postMessage({ status: 'error', data: e.stack || e.toString() });
    }
}

async function generate(data) {
    const { text, max_new_tokens, model_id } = data;

    try {
        const generator = await PipelineSingleton.getInstance(null, model_id);

        const streamer = new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (token) => {
                console.log("[Worker] Streamer token:", token);
                self.postMessage({
                    status: 'update',
                    output: token,
                    delta: true
                });
            }
        });

        const output = await generator(text, {
            max_new_tokens: max_new_tokens || 256,
            temperature: 0.7,
            do_sample: true,
            top_k: 20,
            streamer: streamer,
        });

        self.postMessage({
            status: 'complete',
            output: output[0].generated_text
        });

    } catch (e) {
        console.error("[Worker] Generate error:", e);
        let errorMsg = "Unknown error";
        if (e && e.message) {
            errorMsg = e.message;
        } else if (e && e.stack) {
            errorMsg = e.stack;
        } else if (typeof e === 'number') {
            errorMsg = "WebGPU OOM or allocation error (code: " + e + "). The model may be too large for GPU memory.";
        } else {
            errorMsg = String(e);
        }
        self.postMessage({ status: 'error', data: errorMsg });
    }
}
