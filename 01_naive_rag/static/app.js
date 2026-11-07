// Naive RAG Explorer Frontend Client
document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");
  const statChunks = document.getElementById("stat-chunks");
  const statEmbed = document.getElementById("stat-embed");
  const statLlm = document.getElementById("stat-llm");

  // 1. Fetch system metadata & stats
  async function loadSystemStats() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error("Failed to fetch info");
      const data = await res.json();
      statChunks.textContent = `${data.total_chunks} Chunks Indexed`;
      statEmbed.textContent = `${data.embedding_model.split("/").pop()} (384d)`;
      statLlm.textContent = data.llm_model;
    } catch (err) {
      console.warn("Could not load stats:", err);
      statChunks.textContent = "264 Chunks";
    }
  }
  loadSystemStats();

  // 2. Clickable query chips
  document.querySelectorAll(".query-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      queryInput.value = query;
      queryForm.dispatchEvent(new Event("submit"));
    });
  });

  // 3. Scroll to bottom
  function scrollToBottom() {
    window.scrollTo({
      top: document.body.scrollHeight,
      behavior: "smooth"
    });
  }

  // 4. Form submission handler
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = queryInput.value.trim();
    if (!question) return;

    // Render User Message
    appendUserMessage(question);
    queryInput.value = "";
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // Render Temporary Assistant Loading State
    const loadingId = appendLoadingMessage();
    scrollToBottom();

    try {
      const startTime = performance.now();
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      const latencyMs = Math.round(performance.now() - startTime);

      // Replace loading bubble with full grounded response
      removeMessage(loadingId);
      appendAssistantResponse(data, latencyMs);
    } catch (err) {
      removeMessage(loadingId);
      appendErrorMessage(err.message);
    } finally {
      queryInput.disabled = false;
      sendBtn.disabled = false;
      queryInput.focus();
      scrollToBottom();
    }
  });

  function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user";
    msgDiv.innerHTML = `
      <div class="message-content">
        ${escapeHtml(text)}
      </div>
    `;
    chatMessages.appendChild(msgDiv);
  }

  function appendLoadingMessage() {
    const id = "loading-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.id = id;
    msgDiv.className = "message assistant";
    msgDiv.innerHTML = `
      <div class="message-header">
        <span class="role-badge">Retrieving & Generating</span>
        <span class="time-label"><span class="spinner"></span></span>
      </div>
      <div class="message-content">
        <p style="color: var(--text-muted); font-style: italic;">
          Embedding query &bull; Computing cosine similarities &bull; Grounding response...
        </p>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
    return id;
  }

  function removeMessage(id) {
    const elem = document.getElementById(id);
    if (elem) elem.remove();
  }

  function appendErrorMessage(errorMsg) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    msgDiv.innerHTML = `
      <div class="message-header">
        <span class="role-badge" style="color: #ef4444;">Error</span>
      </div>
      <div class="message-content" style="border-color: rgba(239, 68, 68, 0.4);">
        <p style="color: #fca5a5;">${escapeHtml(errorMsg)}</p>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
  }

  function appendAssistantResponse(data, latencyMs) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";

    // Format Markdown-like bold and chunk citations
    const formattedAnswer = formatAnswer(data.answer);

    // Build chunks inspector cards
    const chunksHtml = data.chunks
      .map((c) => {
        const scoreClass = c.score >= 0.7 ? "score-high" : "score-med";
        return `
          <div class="chunk-card">
            <div class="chunk-meta">
              <span class="chunk-tag">
                <strong>[Chunk ${c.id}]</strong> ${escapeHtml(c.source)}
              </span>
              <span class="score-badge ${scoreClass}">Cosine: ${c.score.toFixed(4)}</span>
            </div>
            <div class="chunk-snippet">${escapeHtml(c.text)}</div>
          </div>
        `;
      })
      .join("");

    msgDiv.innerHTML = `
      <div class="message-header">
        <span class="role-badge">Naive RAG Output</span>
        <span class="time-label">${latencyMs}ms</span>
      </div>
      <div class="message-content">
        ${formattedAnswer}

        <!-- Collapsible Retrieved Chunks Inspector -->
        <div class="inspector-accordion">
          <div class="accordion-header" onclick="this.nextElementSibling.classList.toggle('hidden');">
            <span>🔍 Retrieved Context (${data.chunks.length} chunks used)</span>
            <span style="font-size: 11px; color: var(--text-muted);">Toggle View ▾</span>
          </div>
          <div class="chunks-grid">
            ${chunksHtml}
          </div>
        </div>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
  }

  function formatAnswer(text) {
    // Escape HTML first
    let escaped = escapeHtml(text);

    // Highlight citations like [Chunk 1], [Chunk 2]
    escaped = escaped.replace(/\[Chunk\s+(\d+)\]/gi, (match) => {
      return `<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-family: var(--font-mono); font-size: 12px;">${match}</span>`;
    });

    // Convert newlines to paragraphs
    const paragraphs = escaped
      .split(/\n\n+/)
      .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
      .join("");

    return paragraphs;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

