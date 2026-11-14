// Modular RAG Frontend Client
document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");

  // 1. Fetch system info and update status pills if needed
  async function loadSystemStats() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error("Failed to fetch info");
      const data = await res.json();
      console.log("[Modular RAG] System Info:", data);
    } catch (err) {
      console.warn("Could not load stats:", err);
    }
  }
  loadSystemStats();

  // 2. Clickable query chips to auto-fill and submit
  document.querySelectorAll(".query-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      queryInput.value = query;
      queryForm.dispatchEvent(new Event("submit"));
    });
  });

  // 3. Smooth auto-scroll
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

    // Render Loading State
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
        const errorData = await res.json().catch(() => ({ detail: "Unknown server error" }));
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      const latencyMs = Math.round(performance.now() - startTime);

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
        <span class="role-badge">Semantic Router Inspecting Query...</span>
        <span class="time-label"><span class="spinner"></span></span>
      </div>
      <div class="message-content">
        <p style="color: var(--text-muted); font-style: italic;">
          Classifying intent &bull; Dispatching to specialized engine &bull; Synthesizing grounded output...
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

    const route = data.routing.route;
    const reasoning = data.routing.reasoning;
    const targetEngine = data.routing.target_engine;

    // Route Badge Icon & Labels
    const routeIcons = {
      TABULAR_SQL: "📊",
      GRAPH_RELATIONS: "⚡",
      VECTOR_SEARCH: "📄",
      DIRECT: "💬"
    };
    const icon = routeIcons[route] || "🔍";

    // Format Answer and Evidence
    const formattedAnswer = renderMarkdown(data.answer);
    const formattedEvidence = renderMarkdown(data.evidence_markdown);

    msgDiv.innerHTML = `
      <div class="message-header">
        <span class="role-badge">Modular RAG Result</span>
        <span class="time-label">${latencyMs}ms</span>
      </div>
      <div class="message-content">
        <!-- Dynamic Route Header Badge -->
        <div class="route-indicator-box">
          <div class="route-badge-pill route-pill-${escapeHtml(route)}">
            <span>${icon}</span>
            <span>${escapeHtml(route)}</span>
          </div>
          <div class="route-reasoning">
            ${escapeHtml(reasoning)} (${escapeHtml(targetEngine)})
          </div>
        </div>

        <!-- Synthesized Answer -->
        <div class="answer-body">
          ${formattedAnswer}
        </div>

        <!-- Collapsible Grounded Evidence Inspector -->
        <div class="inspector-accordion">
          <div class="accordion-header" onclick="this.nextElementSibling.classList.toggle('hidden');">
            <span>🔍 Routed Evidence & Provenance Data</span>
            <span style="font-size: 11px; color: var(--text-muted);">Toggle Provenance ▾</span>
          </div>
          <div class="evidence-content hidden">
            ${formattedEvidence}
          </div>
        </div>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
  }

  function renderMarkdown(text) {
    if (!text) return "";
    let s = escapeHtml(text);

    // Code blocks ```sql ... ``` or ```...```
    s = s.replace(/```([a-z]*)\n([\s\S]*?)```/gi, (match, lang, code) => {
      return `<pre style="background: #080c14; padding: 10px; border-radius: 6px; border: 1px solid var(--border-subtle); overflow-x: auto; margin: 8px 0;"><code class="language-${lang}">${code}</code></pre>`;
    });

    // Inline code `code`
    s = s.replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 12px;">$1</code>');

    // Bold **text**
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italics *text*
    s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Markdown tables parsing
    s = parseMarkdownTables(s);

    // Bracketed citations [Chunk 1], [Chunk 2]
    s = s.replace(/\[Chunk\s+(\d+)\]/gi, (match) => {
      return `<span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-family: var(--font-mono); font-size: 12px;">${match}</span>`;
    });

    // Paragraphs
    const paragraphs = s
      .split(/\n\n+/)
      .map((p) => {
        p = p.trim();
        if (p.startsWith("<table") || p.startsWith("<pre")) {
          return p;
        }
        return `<p>${p.replace(/\n/g, "<br>")}</p>`;
      })
      .join("");

    return paragraphs;
  }

  function parseMarkdownTables(text) {
    const tableRegex = /((?:\|[^\n]+\|\r?\n)+)/g;
    return text.replace(tableRegex, (match) => {
      const lines = match.trim().split("\n").map(l => l.trim()).filter(Boolean);
      if (lines.length < 2) return match;

      // Extract header
      const headers = lines[0].split("|").slice(1, -1).map(h => h.trim());
      // Check if line 1 is separator (---|---)
      if (!lines[1].includes("---")) return match;

      let html = '<table><thead><tr>';
      headers.forEach(h => {
        html += `<th>${h}</th>`;
      });
      html += '</tr></thead><tbody>';

      // Rows
      for (let i = 2; i < lines.length; i++) {
        const cells = lines[i].split("|").slice(1, -1).map(c => c.trim());
        html += '<tr>';
        cells.forEach(c => {
          html += `<td>${c}</td>`;
        });
        html += '</tr>';
      }
      html += '</tbody></table>';
      return html;
    });
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
