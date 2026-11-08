// Advanced RAG Frontend Client
document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");
  const toggleExpansion = document.getElementById("toggle-expansion");
  const statChunks = document.getElementById("stat-chunks");

  // 1. Fetch system statistics
  async function loadSystemStats() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error("Failed to fetch info");
      const data = await res.json();
      statChunks.textContent = `${data.total_chunks} Recursive Chunks`;
    } catch (err) {
      console.warn("Could not load stats:", err);
      statChunks.textContent = "Corpus Indexed";
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

    const useExpansion = toggleExpansion.checked;

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
        body: JSON.stringify({ question, use_expansion: useExpansion })
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
        <span class="role-badge">Dual-Stage Pipeline Running</span>
        <span class="time-label"><span class="spinner"></span></span>
      </div>
      <div class="message-content">
        <p style="color: var(--text-muted); font-style: italic;">
          1. Generating search variations &bull; 2. Fusing Dense + BM25 &bull; 3. Neural Cross-Encoder Re-Ranking...
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
        <span class="role-badge" style="color: #ef4444;">Pipeline Error</span>
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

    // Format Answer with citations
    const formattedAnswer = formatAnswer(data.answer);

    // Multi-Query expansion box (if multiple queries used)
    let expansionHtml = "";
    if (data.expanded_queries && data.expanded_queries.length > 1) {
      const items = data.expanded_queries
        .map((q) => `<li>${escapeHtml(q)}</li>`)
        .join("");
      expansionHtml = `
        <div class="expansion-box">
          <div class="expansion-title">⚡ Multi-Query Expansion Generated ${data.expanded_queries.length} Angles:</div>
          <ul class="expansion-list">${items}</ul>
        </div>
      `;
    }

    // Build chunks inspector cards
    const chunksHtml = data.reranked_chunks
      .map((c) => {
        const denseStr = c.dense_rank ? `#${c.dense_rank}` : "—";
        const sparseStr = c.sparse_rank ? `#${c.sparse_rank}` : "—";
        return `
          <div class="chunk-card">
            <div class="chunk-header-row">
              <span class="chunk-source-tag">
                <strong>[Chunk ${c.final_rank}]</strong> ${escapeHtml(c.source)}
              </span>
              <div class="badges-group">
                <span class="badge-ce" title="Neural Cross-Encoder deep pairwise attention score">Cross-Encoder: ${c.cross_encoder_score.toFixed(4)}</span>
                <span class="badge-rrf" title="Reciprocal Rank Fusion score">RRF: ${c.rrf_score.toFixed(4)}</span>
                <span class="badge-dense" title="Stage 1 Dense Semantic Rank">Dense: ${denseStr}</span>
                <span class="badge-bm25" title="Stage 1 BM25 Lexical Rank">BM25: ${sparseStr}</span>
              </div>
            </div>
            <div class="chunk-snippet">${escapeHtml(c.text)}</div>
          </div>
        `;
      })
      .join("");

    msgDiv.innerHTML = `
      <div class="message-header">
        <span class="role-badge">Advanced RAG Output</span>
        <span class="time-label">${latencyMs}ms</span>
      </div>
      <div class="message-content">
        ${expansionHtml}
        ${formattedAnswer}

        <!-- Collapsible Re-Ranked Chunks Inspector -->
        <div class="inspector-accordion">
          <div class="accordion-header" onclick="this.nextElementSibling.classList.toggle('hidden');">
            <span>🎯 Cross-Encoder Re-Ranked Winners (${data.reranked_chunks.length} Top Chunks)</span>
            <span style="font-size: 11px; color: var(--text-muted);">Toggle Provenance ▾</span>
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
    let escaped = escapeHtml(text);

    // Highlight citations like [Chunk 1], [Chunk 2]
    escaped = escaped.replace(/\[Chunk\s+(\d+)\]/gi, (match) => {
      return `<span style="background: rgba(6, 182, 212, 0.2); color: #22d3ee; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-family: var(--font-mono); font-size: 12px;">${match}</span>`;
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

