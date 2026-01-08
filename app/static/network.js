// Network utilities: streaming, models, errors, throughput footer stripping
(function () {
  function stripThroughputFooter(text) {
    try {
      const src = String(text || '');
      const reLine = /(^|\n)\s*\[throughput\][^\n]*\n?/g;
      let lastFooter = null;
      const matches = src.match(/\[throughput\][^\n]*/g);
      if (matches && matches.length) lastFooter = matches[matches.length - 1].trim();
      const clean = src.replace(reLine, (m, g1) => g1 ? g1 : '');
      return { clean: clean.trim(), footer: lastFooter };
    } catch {
      return { clean: text, footer: null };
    }
  }

  async function safeExtractError(response) {
    try {
      const data = await response.json();
      return data.detail || data.error?.message || response.statusText;
    } catch (_) {
      try { return await response.text(); } catch { return response.statusText; }
    }
  }

  async function loadModels() {
    const modelSelect = document.getElementById("modelSelect");
    const modelCaps = document.getElementById("modelCaps");
    const visionToggle = document.getElementById("visionToggle");
    let models = [];
    let currentModel = null;
    try {
      const res = await fetch('/models');
      const data = await res.json();
      models = data.models || [];
      modelSelect.innerHTML = '';
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.id;
        modelSelect.appendChild(opt);
      }

      // Add Local Model option
      const localOpt = document.createElement('option');
      localOpt.value = "Local: Xenova/TinyLlama-1.1B-Chat-v1.0";
      localOpt.textContent = "💻 Local: TinyLlama-1.1B-Chat (Browser)";
      modelSelect.appendChild(localOpt);
      models.push({ id: "Local: Xenova/TinyLlama-1.1B-Chat-v1.0", vision: false });

      // Qwen models removed - incompatible ONNX file structure
      // OpenELM-270M removed - produces only empty tokens
      // LFM2.5 removed - "Unsupported model type: lfm2" in transformers.js

      // WebGPU-accelerated model (requires Chrome/Edge with WebGPU support)
      const localOptGPU = document.createElement('option');
      localOptGPU.value = "Local: onnx-community/Llama-3.2-1B-Instruct-ONNX";
      localOptGPU.textContent = "🚀 Local: Llama-3.2-1B (WebGPU)";
      modelSelect.appendChild(localOptGPU);
      models.push({ id: "Local: onnx-community/Llama-3.2-1B-Instruct-ONNX", vision: false });

      const localOptSmol = document.createElement('option');
      localOptSmol.value = "Local: HuggingFaceTB/SmolLM2-360M-Instruct";
      localOptSmol.textContent = "🚀 Local: SmolLM2-360M (WebGPU)";
      modelSelect.appendChild(localOptSmol);
      models.push({ id: "Local: HuggingFaceTB/SmolLM2-360M-Instruct", vision: false });

      const localOptGemma = document.createElement('option');
      localOptGemma.value = "Local: onnx-community/gemma-3-1b-it-ONNX";
      localOptGemma.textContent = "🚀 Local: Gemma-3-1b (WebGPU)";
      modelSelect.appendChild(localOptGemma);
      models.push({ id: "Local: onnx-community/gemma-3-1b-it-ONNX", vision: false });

      const localOpt3 = document.createElement('option');
      localOpt3.value = "Local: Xenova/Phi-3-mini-4k-instruct";
      localOpt3.textContent = "💻 Local: Phi-3-mini-4k-instruct (Browser)";
      modelSelect.appendChild(localOpt3);
      models.push({ id: "Local: Xenova/Phi-3-mini-4k-instruct", vision: false });

      const defId = data.default || null;
      currentModel = (defId && models.find(x => x.id === defId)?.id) || models[0]?.id || null;
      modelSelect.value = currentModel || '';
      updateModelCaps();
    } catch (e) {
      modelSelect.innerHTML = '<option value="">(failed to load)</option>';
    }
    function updateModelCaps() {
      const item = models.find(m => m.id === currentModel);
      if (!item) { modelCaps.textContent = ''; visionToggle.checked = false; return; }
      // Set toggle to the probed vision capability by default
      visionToggle.checked = item.vision;
      modelCaps.textContent = item.vision ? 'Vision-capable' : 'Text-only';
    }
    // NOTE: modelSelect.onchange is handled by app.js to update currentModel
    // Handle vision toggle changes
    visionToggle.onchange = () => {
      const item = models.find(m => m.id === currentModel);
      if (!item) return;
      // Update display based on toggle state
      modelCaps.textContent = visionToggle.checked ? 'Vision-capable (override)' : 'Text-only (override)';
      // Re-validate selected files based on new vision state
      const fileInput = document.getElementById("fileInput");
      const fileList = document.getElementById("fileList");
      const files = Array.from(fileInput.files || []);
      if (files.length > 0) {
        // Use Attach.validateSelectionForModel to validate with new vision state
        if (typeof Attach !== 'undefined' && Attach.validateSelectionForModel) {
          const ok = Attach.validateSelectionForModel(files, models, currentModel, visionToggle.checked);
          if (!ok.ok) {
            fileInput.value = '';
            fileList.textContent = '';
          }
        }
      }
    };

    return { models, currentModel, updateModelCaps };
  }

  window.Net = {
    stripThroughputFooter,
    safeExtractError,
    loadModels,
    checkWebSearchStatus: async function () {
      try {
        const res = await fetch('/search/status');
        const data = await res.json();
        return data.available || false;
      } catch {
        return false;
      }
    }
  };
})();
