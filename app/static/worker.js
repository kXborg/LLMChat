
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.2.4';

// Skip local model checks since we are running in browser
env.allowLocalModels = false;

// Disable multi-threading to avoid SecurityError with CDN-hosted worker scripts
env.backends.onnx.wasm.numThreads = 1;

class PipelineSingleton {
    static task = 'text-generation';
    static model = 'Xenova/TinyLlama-1.1B-Chat-v1.0';
    static quantized = true;
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            // Assign the promise immediately to lock other calls
            this.instance = pipeline(this.task, this.model, {
                quantized: this.quantized,
                progress_callback,
            });
        }
        return this.instance;
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
    try {
        self.postMessage({ status: 'loading', data: 'Initiating model load...' });
        await PipelineSingleton.getInstance(x => {
            self.postMessage({ status: 'progress', data: x });
        });
        self.postMessage({ status: 'ready' });
    } catch (e) {
        self.postMessage({ status: 'error', data: e.stack || e.toString() });
    }
}

async function generate(data) {
    const { text, max_new_tokens } = data;

    try {
        const generator = await PipelineSingleton.getInstance();

        const streamer = new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (text) => {
                self.postMessage({
                    status: 'update',
                    output: text,
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
        self.postMessage({ status: 'error', data: e.stack || e.toString() });
    }
}
