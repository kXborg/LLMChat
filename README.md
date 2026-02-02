# LLM Chat

![LLM Chat](/tests/llmchat.png)

A chatML interface integrated with RAG, Web search and more for LLMs and VLMs. Simply plugin your v1 endpoint from OpenAI, Google AI Studio, Ollama, vLLM or any OpenAI compatible endpoint. 

Don't have enough resources? No worries, choose from serveral models in the drop down, and run LLMs in your browser locally! 

## Installation and Setup 

Install vllm, fastapi, uvicorn, and ngrok (optional, you can also use any other reverse proxy).

```bash
pip install -r requirements.txt
```

Terminal 1
```bash
uvicorn app.main:app --host 0.0.0.0 --port 3000
```


## Tested Models on vLLM Server (RTX 3080 Ti, 12 GB)
Confused with hosting LLMs locally or from remote instance? Checkout this article on [Self hosting LLMs using Various Serving Engines](https://www.orbital.net.in/blog/self-hosting-llms-vllm-sglang-llamacpp).

The following are the list of models we have successfully tried so far on `vllm==0.12.x` versions. The errors we faced and fixes are also logged within.

| Model Name | Command | Remarks |
|:----------|:----------|:----------|
| [Falcon3-7B-Instruct-GPTQ-Int4](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-GPTQ-Int4) | `vllm serve tiiuae/Falcon3-7B-Instruct-GPTQ-Int4 --max-model-len 4096 --gpu-memory-utilization 0.85` |  |
| [Ministral-3-8B-Instruct-2512-AWQ-4bit](https://huggingface.co/cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit) | `vllm serve cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit  --gpu-memory-utilization 0.85 --max-model-len 6144 --max-num-batched-tokens 1024` |  |
| [OpenGVLab/InternVL3-8B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-8B-AWQ) | `vllm serve OpenGVLab/InternVL3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-batched-tokens 1024 --trust-remote-code --quantization awq` | Note: AWQ quantized model won't work unless <code>--quantization awq</code> flag is set. |
| [OpenGVLab/InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) | `vllm serve OpenGVLab/InternVL3-2B --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-batched-tokens 1024 --trust-remote-code` |  |
| [Nemotron Cascade 8B](https://huggingface.co/cyankiwi/Nemotron-Cascade-8B-AWQ-4bit) | `vllm serve cyankiwi/Nemotron-Cascade-8B-AWQ-4bit --max-model-len 4096  --gpu-memory-utilization 0.85 --max-num-batched-tokens 1024 --trust-remote-code` |  |
| [Nemotron Orchestrator 8B](https://huggingface.co/cyankiwi/Nemotron-Orchestrator-8B-AWQ-4bit) | `vllm serve cyankiwi/Nemotron-Orchestrator-8B-AWQ-4bit --served-model-name Nemotron-orchestrator --max-model-len 4096 --gpu-memory-utilization 0.85 --max-num-batched-tokens 1024 --trust-remote-code `|  |
| [Qwen VL 2B Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | `vllm serve Qwen/Qwen2-VL-2B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-batched-tokens 1024 ` |  |
| [Nvidia Cosmos Reason2 2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | `vllm serve nvidia/Cosmos-Reason2-2B --max-model-len 8192 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.8 ` |  |
| [H20VL Mississipi 2B](https://huggingface.co/h2oai/h2ovl-mississippi-2b) | `vllm serve h2oai/h2ovl-mississippi-2b --max-model-len 4096 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.75` | Does not support system prompt, need to take care of this. (Not yet fixed) |
| [Gemma 3 4B Instruct](https://huggingface.co/ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g) | `vllm serve ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g --max-model-len 4096 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.8` | Original model OOM. Using GPTQ quantized model from community. Max concurrency observed: 9 |
