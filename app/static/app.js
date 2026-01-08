// App logic extracted from index.html
(function () {
  const messagesDiv = document.getElementById("messages");
  const inputField = document.getElementById("inputText");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const attachBtn = document.getElementById("attachBtn");
  const fileInput = document.getElementById("fileInput");
  const fileList = document.getElementById("fileList");
  const modelSelect = document.getElementById("modelSelect");
  const modelCaps = document.getElementById("modelCaps");
  const visionToggle = document.getElementById("visionToggle");
  const webSearchToggle = document.getElementById("webSearchToggle");
  const ragToggle = document.getElementById("ragToggle");
  const ragDocList = document.getElementById("ragDocList");
  const ragDocCount = document.getElementById("ragDocCount");
  const statsDiv = document.getElementById("stats");

  let userId = "user-" + Math.random().toString(36).substring(2, 8);
  let models = [];
  let currentModel = null;

  // Worker for local inference
  const worker = new Worker("/static/worker.js", { type: "module" });
  let isWorkerLoaded = false;
  let currentLocalModelId = null;

  worker.onmessage = function (e) {
    const { status, data, model } = e.data;
    if (status === 'ready') {
      isWorkerLoaded = true;
      currentLocalModelId = model;
      console.log("Local model ready:", model);
    } else if (status === 'error') {
      console.error("Worker error:", data);
    }
  };

  let autoScroll = true;
  function updateAutoScroll() {
    autoScroll = (messagesDiv.scrollTop + messagesDiv.clientHeight) >= (messagesDiv.scrollHeight - 4);
  }
  messagesDiv.addEventListener('scroll', updateAutoScroll);
  updateAutoScroll();

  function addMessage(text, sender) {
    const pinned = autoScroll;
    const bubble = document.createElement("div");
    bubble.classList.add("bubble", sender);
    bubble.textContent = text;
    const line = document.createElement("div");
    line.style.display = "flex";
    line.style.justifyContent = sender === "user" ? "flex-end" : "flex-start";
    line.appendChild(bubble);
    messagesDiv.appendChild(line);
    if (pinned) messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  function createAssistantBubble(initialText) {
    const pinned = autoScroll;
    const bubble = document.createElement("div");
    bubble.classList.add("bubble", "assistant");
    bubble.innerText = initialText || "";
    const line = document.createElement("div");
    line.style.display = "flex";
    line.style.justifyContent = "flex-start";
    line.appendChild(bubble);
    messagesDiv.appendChild(line);
    if (pinned) messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return bubble;
  }

  function buildPrompt(modelId, userText) {
    var sysTag = "<" + "|system|" + ">";
    var userTag = "<" + "|user|" + ">";
    var asstTag = "<" + "|assistant|" + ">";
    var endTag = "<" + "/s" + ">";
    var imStart = "<" + "|im_start|" + ">";
    var imEnd = "<" + "|im_end|" + ">";
    var endTok = "<" + "|end|" + ">";

    if (modelId.includes("TinyLlama")) {
      return sysTag + "\nYou are a friendly assistant. Be concise unless asked to elaborate.\n" + endTag + "\n" + userTag + "\n" + userText + "\n" + endTag + "\n" + asstTag + "\n";
    } else if (modelId.includes("Qwen") || modelId.includes("SmolLM2")) {
      return imStart + "system\nYou are a helpful assistant. Be concise unless asked to elaborate." + imEnd + "\n" + imStart + "user\n" + userText + imEnd + "\n" + imStart + "assistant\n";
    } else if (modelId.includes("Phi-3")) {
      return userTag + "\n" + userText + endTok + "\n" + asstTag + "\n";
    } else if (modelId.includes("OpenELM")) {
      // OpenELM uses simple instruction format
      return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + userText + "\n\n### Response:\n";
    } else if (modelId.includes("LFM2.5") || modelId.includes("LiquidAI")) {
      // LiquidAI LFM uses ChatML-like format
      return imStart + "system\nYou are a helpful assistant." + imEnd + "\n" + imStart + "user\n" + userText + imEnd + "\n" + imStart + "assistant\n";
    } else if (modelId.includes("Llama-3") || modelId.includes("llama-3")) {
      // Llama 3.x uses ChatML-like format with special tokens
      var bosToken = "<|begin_of_text|>";
      var headerStart = "<|start_header_id|>";
      var headerEnd = "<|end_header_id|>";
      var eot = "<|eot_id|>";
      var eot = "<|eot_id|>";
      return bosToken + headerStart + "system" + headerEnd + "\n\nYou are a helpful assistant." + eot + headerStart + "user" + headerEnd + "\n\n" + userText + eot + headerStart + "assistant" + headerEnd + "\n\n";
    } else if (modelId.toLowerCase().includes("gemma")) {
      var startTurn = "<start_of_turn>";
      var endTurn = "<end_of_turn>";
      return startTurn + "user\n" + userText + endTurn + "\n" + startTurn + "model\n";
    } else {
      return userTag + "\n" + userText + "\n" + asstTag + "\n";
    }
  }

  let isGenerating = false;
  let abortController = null;

  async function sendMessage() {
    // If currently generating, this button acts as Stop
    if (isGenerating) {
      if (currentModel.startsWith("Local:")) {
        console.log("[App] Interrupting local worker...");
        worker.postMessage({ type: 'interrupt', data: {} });
      } else {
        console.log("[App] Aborting server request...");
        if (abortController) abortController.abort();
      }
      isGenerating = false;
      sendBtn.textContent = "Send";
      sendBtn.style.background = "#007bff";
      return;
    }

    try { await modelsReady; } catch (_) { }
    const text = inputField.value.trim();
    if (!text) return;
    if (!currentModel) {
      const bubble = createAssistantBubble("");
      renderMarkdownInto(bubble, "Error: No model selected.");
      return;
    }

    addMessage(text, "user");
    inputField.value = "";

    // Set to generating state
    isGenerating = true;
    sendBtn.textContent = "Stop";
    sendBtn.style.background = "#dc3545"; // Red color for stop

    // LOCAL INFERENCE PATH
    if (currentModel.startsWith("Local:")) {
      const actualModelId = currentModel.replace("Local: ", "").trim();
      console.log("[App] Selected local model:", currentModel, "-> actualModelId:", actualModelId);
      const bubble = createAssistantBubble("... (Loading model)");
      let generatedText = "";
      const startTime = performance.now();
      const prompt = buildPrompt(actualModelId, text);

      const onWorkerMessage = (e) => {
        const { status, output, data, delta } = e.data;
        if (status === 'loading') {
          const modelName = data.model || actualModelId.split('/').pop();
          bubble.innerHTML = '<span class="spinner"></span> Loading ' + modelName + '...';
        } else if (status === 'progress') {
          const pct = data.progress || 0;
          const modelName = data.model || actualModelId.split('/').pop();
          if (pct > 0) {
            bubble.innerHTML = '<span class="spinner"></span> Downloading ' + modelName + ': ' + Math.round(pct) + '%';
          }
        } else if (status === 'ready') {
          bubble.innerText = "Generating...";
          worker.postMessage({ type: 'generate', data: { text: prompt, max_new_tokens: 2048, model_id: actualModelId } });
        } else if (status === 'update') {
          if (delta) {
            generatedText += output;
          } else {
            generatedText = output;
          }
          const clean = generatedText.replace(/<\|.*?\|>/g, "").replace(/<\/s>/g, "");
          renderMarkdownInto(bubble, clean);
          if (window.Render && window.Render.updateStatsFromStream) {
            window.Render.updateStatsFromStream(null, startTime, generatedText);
          }
        } else if (status === 'complete') {
          const clean = generatedText.replace(/<\|.*?\|>/g, "").replace(/<\/s>/g, "");
          renderMarkdownInto(bubble, clean);
          if (window.Render && window.Render.updateStatsFromStream) {
            window.Render.updateStatsFromStream(null, startTime, generatedText);
          }
          worker.removeEventListener('message', onWorkerMessage);
          // Reset UI
          isGenerating = false;
          sendBtn.textContent = "Send";
          sendBtn.style.background = "#007bff";
        } else if (status === 'error') {
          renderMarkdownInto(bubble, "Error: " + data);
          worker.removeEventListener('message', onWorkerMessage);
          // Reset UI
          isGenerating = false;
          sendBtn.textContent = "Send";
          sendBtn.style.background = "#007bff";
        }
      };

      worker.addEventListener('message', onWorkerMessage);
      worker.postMessage({ type: 'load', data: { model: actualModelId } });
      return;
    }

    // SERVER-SIDE INFERENCE PATH
    abortController = new AbortController();
    try {
      await streamAssistantResponse(text, abortController.signal);
    } catch (e) {
      if (e.name === 'AbortError') {
        // Interrupted by user
        const bubble = createAssistantBubble("");
        renderMarkdownInto(bubble, "_Aborted._");
      } else {
        console.warn("Stream failed; falling back to /chat", e);
        // Fallback or error handling... 
        // Note: Fallback is complex to abort, so we'll skip complex fallback here for brevity or implement similarly
      }
    } finally {
      isGenerating = false;
      sendBtn.textContent = "Send";
      sendBtn.style.background = "#007bff";
      abortController = null;
    }
  }

  sendBtn.onclick = sendMessage;
  attachBtn.onclick = () => fileInput.click();
  fileInput.onchange = async () => {
    const files = Array.from(fileInput.files || []);
    const ok = Attach.validateSelectionForModel(files, models, currentModel, visionToggle.checked);
    if (!ok.ok) { fileInput.value = ""; fileList.textContent = ""; return; }
    const pdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    const otherFiles = files.filter(f => !f.name.toLowerCase().endsWith('.pdf'));
    for (const pdf of pdfFiles) {
      await uploadRagDocument(pdf);
    }
    if (otherFiles.length > 0) {
      const names = otherFiles.map(f => "* " + f.name);
      fileList.textContent = names.join("\n");
    } else if (pdfFiles.length > 0) {
      fileInput.value = "";
    }
  };

  inputField.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
  });

  resetBtn.onclick = async function () {
    try {
      await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId })
      });
      messagesDiv.innerHTML = "";
      const bubble = createAssistantBubble("");
      renderMarkdownInto(bubble, "_Chat reset._");
    } catch (err) {
      console.error("Failed to reset chat", err);
    }
  };

  marked.setOptions({
    highlight: function (code, lang) {
      try {
        if (lang && window.hljs && window.hljs.getLanguage(lang)) {
          return window.hljs.highlight(code, { language: lang }).value;
        }
        return window.hljs ? window.hljs.highlightAuto(code).value : code;
      } catch (e) { return code; }
    }
  });

  function renderMarkdownInto(element, rawText) {
    Render.renderMarkdownInto(element, rawText);
  }

  function showImagePreviews(attachments) { Render.showImagePreviews(attachments); }

  async function safeExtractError(response) {
    try {
      const data = await response.json();
      return data.detail || (data.error && data.error.message) || response.statusText;
    } catch (_) {
      try { return await response.text(); } catch (e) { return response.statusText; }
    }
  }

  let modelsReady = null;
  async function loadModels() {
    const info = await Net.loadModels();
    models = info.models || models;
    currentModel = info.currentModel || currentModel;
  }

  async function initWebSearch() {
    const available = await Net.checkWebSearchStatus();
    if (webSearchToggle) {
      webSearchToggle.disabled = !available;
      if (!available) {
        webSearchToggle.title = "Web search is not available";
        webSearchToggle.parentElement.style.opacity = "0.5";
      }
    }
  }

  modelsReady = loadModels();
  initWebSearch();

  function updateModelCaps() {
    const item = models.find(m => m.id === currentModel);
    if (!item) { modelCaps.textContent = ''; visionToggle.checked = false; return; }
    visionToggle.checked = item.vision;
    modelCaps.textContent = item.vision ? 'Vision-capable' : 'Text-only';
  }

  modelSelect.onchange = () => {
    currentModel = modelSelect.value || null;
    updateModelCaps();
    const item = models.find(m => m.id === currentModel);
    if (item && !item.vision) {
      if (Array.from(fileInput.files || []).some(f => !(f.type || '').startsWith('text/'))) {
        fileInput.value = '';
        fileList.textContent = '';
      }
    }
  };

  async function uploadSelectedFiles() { return Attach.uploadSelectedFiles(fileInput); }

  async function streamAssistantResponse(promptText, signal) {
    const bubble = createAssistantBubble("...");
    let buffer = "";
    const attachments = await uploadSelectedFiles();
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: signal,
      body: JSON.stringify({ user_id: userId, message: promptText, attachments, model: currentModel, vision_enabled: visionToggle.checked, web_search: webSearchToggle.checked, rag_enabled: ragToggle.checked }),
    });

    if (!response.ok || !response.body) {
      const err = await Net.safeExtractError(response);
      const eb = createAssistantBubble("");
      renderMarkdownInto(eb, "Error: " + err);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let startedAt = performance.now();
    let latestFooter = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      const { clean, footer } = Net.stripThroughputFooter(buffer);
      if (footer) latestFooter = footer;
      renderMarkdownInto(bubble, clean);
    }

    const { clean, footer } = Net.stripThroughputFooter(buffer);
    renderMarkdownInto(bubble, clean);
    showImagePreviews(attachments);
    Render.updateStatsFromStream(footer || latestFooter, startedAt, clean);
  }

  function updateStatsFromResponse(data) { Render.updateStatsFromResponse(data); }

  // RAG Document Management
  async function loadRagDocuments() {
    try {
      const res = await fetch("/rag/documents?user_id=" + encodeURIComponent(userId));
      const data = await res.json();
      const docs = data.documents || [];
      if (ragDocCount) {
        ragDocCount.textContent = docs.length > 0 ? "(" + docs.length + " doc" + (docs.length > 1 ? "s" : "") + " indexed)" : "";
      }
      if (docs.length === 0) {
        ragDocList.innerHTML = '<em style="color:#888;">No documents indexed</em>';
        return;
      }
      ragDocList.innerHTML = docs.map(doc =>
        '<div style="display:flex; justify-content:space-between; align-items:center; padding:4px 6px; margin:4px 0; background:#f8f9fb; border-radius:4px; border:1px solid #eee;">' +
        '<span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:160px;" title="' + doc.filename + '">📄 ' + doc.filename + '</span>' +
        '<button onclick="window.deleteRagDoc(\'' + doc.doc_id + '\')" style="padding:2px 6px; font-size:10px; background:#dc3545; color:white; border:none; border-radius:3px; cursor:pointer;">X</button>' +
        '</div>'
      ).join('');
    } catch (e) {
      console.error('Failed to load RAG documents:', e);
    }
  }

  async function uploadRagDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    fileList.textContent = "Indexing " + file.name + "...";
    try {
      const res = await fetch("/rag/upload?user_id=" + encodeURIComponent(userId), {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        fileList.textContent = "Failed: " + (err.detail || 'Unknown error');
        return false;
      }
      const data = await res.json();
      fileList.textContent = "Indexed: " + file.name + " (" + data.chunk_count + " chunks)";
      await loadRagDocuments();
      return true;
    } catch (e) {
      console.error('Upload failed:', e);
      fileList.textContent = 'Upload failed';
      return false;
    }
  }

  window.deleteRagDoc = async function (docId) {
    if (!confirm('Delete this document from RAG index?')) return;
    try {
      const res = await fetch("/rag/documents/" + docId + "?user_id=" + encodeURIComponent(userId), {
        method: 'DELETE',
      });
      if (!res.ok) {
        alert('Failed to delete document');
        return;
      }
      await loadRagDocuments();
    } catch (e) {
      console.error('Delete failed:', e);
    }
  };

  async function initRag() {
    try {
      const res = await fetch('/rag/status');
      const data = await res.json();
      if (ragToggle) {
        ragToggle.disabled = !data.available;
        if (!data.available) {
          ragToggle.title = 'RAG is not available';
          ragToggle.parentElement.style.opacity = '0.5';
        }
      }
      await loadRagDocuments();
    } catch (e) {
      console.error('RAG init failed:', e);
      if (ragToggle) {
        ragToggle.disabled = true;
        ragToggle.parentElement.style.opacity = '0.5';
      }
    }
  }

  initRag();
})();
