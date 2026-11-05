"use strict";

const $ = (sel) => document.querySelector(sel);
const chatHistory = []; // chat turns sent back for follow-up questions

// ---------------------------------------------------------------- helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function ago(ts) {
  if (!ts) return "never";
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

async function busy(btn, label, fn) {
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = label;
  try { return await fn(); }
  catch (e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = old; }
}

// Minimal, XSS-safe markdown: escape first, then apply a few inline/block rules.
function md(src) {
  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/_\(([^)]+)\)_/g, "<em>($1)</em>")
    .replace(/\[(\d+)\]/g, '<span class="cite" data-n="$1">$1</span>');
  const out = [];
  let list = null;
  for (const line of src.split("\n")) {
    const m = line.match(/^\s*(?:[-*•]|(\d+)\.)\s+(.*)$/);
    if (m) {
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { if (list) out.push(`</${list.tag}>`); list = { tag }; out.push(`<${tag}>`); }
      out.push(`<li>${inline(m[2])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list.tag}>`); list = null; }
    const h = line.match(/^#{1,4}\s+(.*)$/);
    if (h) out.push(`<p><strong>${inline(h[1])}</strong></p>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`</${list.tag}>`);
  return out.join("");
}

// Word-level diff between two lines (LCS), used to highlight what exactly changed.
function wordDiff(a, b) {
  const A = a.split(/(\s+)/), B = b.split(/(\s+)/);
  if (A.length * B.length > 160000) return [esc(a), esc(b)];
  const dp = Array.from({ length: A.length + 1 }, () => new Uint16Array(B.length + 1));
  for (let i = A.length - 1; i >= 0; i--)
    for (let j = B.length - 1; j >= 0; j--)
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  let i = 0, j = 0, oa = "", ob = "";
  while (i < A.length && j < B.length) {
    if (A[i] === B[j]) { oa += esc(A[i]); ob += esc(B[j]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) oa += `<del>${esc(A[i++])}</del>`;
    else ob += `<ins>${esc(B[j++])}</ins>`;
  }
  while (i < A.length) oa += `<del>${esc(A[i++])}</del>`;
  while (j < B.length) ob += `<ins>${esc(B[j++])}</ins>`;
  return [oa, ob];
}

function renderDiff(lines) {
  const body = lines.filter((l) => !l.startsWith("---") && !l.startsWith("+++"));
  const html = [];
  for (let k = 0; k < body.length; k++) {
    const l = body[k];
    // Pair runs of removed lines with the following added lines for word-level highlights.
    if (l.startsWith("-")) {
      const dels = [], adds = [];
      while (k < body.length && body[k].startsWith("-")) dels.push(body[k++].slice(1));
      while (k < body.length && body[k].startsWith("+")) adds.push(body[k++].slice(1));
      k--;
      const n = Math.max(dels.length, adds.length);
      const dOut = [], aOut = [];
      for (let p = 0; p < n; p++) {
        if (p < dels.length && p < adds.length) {
          const [x, y] = wordDiff(dels[p], adds[p]);
          dOut.push(x); aOut.push(y);
        } else if (p < dels.length) dOut.push(esc(dels[p]));
        else aOut.push(esc(adds[p]));
      }
      dOut.forEach((x) => html.push(`<div class="d">− ${x}</div>`));
      aOut.forEach((x) => html.push(`<div class="a">+ ${x}</div>`));
    } else if (l.startsWith("+")) html.push(`<div class="a">+ ${esc(l.slice(1))}</div>`);
    else if (l.startsWith("@@")) html.push(`<div class="h">⋯</div>`);
    else html.push(`<div>  ${esc(l.slice(1))}</div>`);
  }
  return `<div class="diff">${html.join("")}</div>`;
}

// ---------------------------------------------------------------- sources
async function loadStatus() {
  try {
    const s = await api("/api/status");
    $("#status").innerHTML =
      `Model: <b>${esc(s.llm || "none (retrieval only)")}</b><br>` +
      `Retrieval: ${esc(s.retrieval)}` +
      (s.watch_minutes ? `<br>Auto-check every ${s.watch_minutes} min` : "");
  } catch {}
}

async function loadSources(flashIds = []) {
  const rows = await api("/api/sources");
  $("#source-count").textContent = rows.length;
  $("#sources").innerHTML = rows.map((r) => {
    const stale = r.checked_at && Date.now() / 1000 - r.checked_at > 86400;
    const dot = r.status === "error" ? "error" : stale ? "stale" : "";
    return `<li class="source ${r.status === "error" ? "error" : ""} ${flashIds.includes(r.id) ? "flash" : ""}">
      <div class="title" title="${esc(r.title)}">${esc(r.title || r.url)}</div>
      <a class="url" href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.url)}</a>
      <div class="meta">
        <span class="dot ${dot}"></span>
        <span title="Last checked">checked ${ago(r.checked_at)}</span>·
        <span title="Last content change">changed ${ago(r.changed_at)}</span>·
        <span>${r.n_chunks} chunks</span>
        <span class="actions">
          <button class="ghost" title="Check this page" data-act="refresh" data-id="${r.id}">↻</button>
          <button class="ghost" title="Version history" data-act="history" data-id="${r.id}">🕘 ${r.n_versions}</button>
          <button class="ghost" title="Remove" data-act="delete" data-id="${r.id}">✕</button>
        </span>
      </div>
      ${r.error ? `<div class="small err">${esc(r.error)}</div>` : ""}
    </li>`;
  }).join("");
}

function showReport(events) {
  const el = $("#report");
  const counts = events.reduce((acc, e) => ((acc[e.status] = (acc[e.status] || 0) + 1), acc), {});
  const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(" · ");
  el.innerHTML = `<header><h2>Freshness report <span class="muted small">${esc(summary)}</span></h2>
      <button class="ghost" onclick="this.closest('section').hidden = true">✕</button></header>` +
    events.map((e) => `<div class="event">
      <span class="badge ${e.status}">${e.status}</span>
      <b>${esc(e.title || e.url)}</b> <span class="muted small">${esc(e.url)}</span>
      ${e.via_304 ? '<span class="muted small"> · server said 304 Not Modified</span>' : ""}
      ${e.error ? `<div class="err small">${esc(e.error)}</div>` : ""}
      ${e.status === "new" ? `<div class="muted small">Indexed ${e.chunks_added} chunks.</div>` : ""}
      ${e.status === "changed" ? `
        ${e.summary ? `<div class="summary">${esc(e.summary)}</div>` : ""}
        <div class="muted small">+${e.added} / −${e.removed} lines · ${e.chunks_added} chunks re-indexed, ${e.chunks_removed} dropped (unchanged chunks reused)</div>
        <details open><summary>View diff</summary>${renderDiff(e.diff)}</details>` : ""}
    </div>`).join("");
  el.hidden = false;
}

async function showHistory(id) {
  const data = await api(`/api/sources/${id}/history`);
  $("#history-title").textContent = data.source.title || data.source.url;
  $("#history-body").innerHTML = data.versions.map((v) => `
    <div class="version">
      <div><b>${new Date(v.fetched_at * 1000).toLocaleString()}</b>
        <span class="muted small">${v.initial ? "first indexed" : `+${v.added} / −${v.removed} lines`}</span></div>
      ${v.summary ? `<div class="summary">${esc(v.summary)}</div>` : ""}
      ${v.diff.length ? `<details ${v === data.versions[0] ? "open" : ""}><summary>Diff vs previous</summary>${renderDiff(v.diff)}</details>` : ""}
    </div>`).join("") || '<p class="muted">No versions yet.</p>';
  $("#history").showModal();
}

$("#sources").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const id = Number(btn.dataset.id);
  if (btn.dataset.act === "history") return showHistory(id);
  if (btn.dataset.act === "delete") {
    if (!confirm("Remove this page from the index?")) return;
    await api(`/api/sources/${id}`, { method: "DELETE" });
    return refreshAll();
  }
  if (btn.dataset.act === "refresh") {
    await busy(btn, "…", async () => {
      const events = await api("/api/refresh", { method: "POST", body: JSON.stringify({ ids: [id] }) });
      showReport(events);
      await loadSources(events.filter((e) => e.status === "changed").map((e) => e.source_id));
    });
  }
});

$("#add-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const urls = $("#urls").value.split("\n").map((u) => u.trim()).filter(Boolean);
  if (!urls.length) return;
  await busy(ev.submitter, "Indexing…", async () => {
    const events = await api("/api/sources", { method: "POST", body: JSON.stringify({ urls, depth: Number($("#depth").value) }) });
    showReport(events);
    $("#urls").value = "";
    await refreshAll();
  });
});

$("#refresh-all").addEventListener("click", (ev) =>
  busy(ev.currentTarget, "Checking…", async () => {
    const events = await api("/api/refresh", { method: "POST", body: "{}" });
    showReport(events);
    await loadSources(events.filter((e) => e.status === "changed").map((e) => e.source_id));
  }));

async function refreshAll() { await Promise.all([loadSources(), loadStatus()]); }

// ---------------------------------------------------------------- chat
function addMessage(cls, html) {
  $("#empty").hidden = true;
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.innerHTML = html;
  $("#messages").appendChild(div);
  div.scrollIntoView({ block: "end", behavior: "smooth" });
  return div;
}

function renderCards(box, sources, answer) {
  const used = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  if (!used.size && answer.includes("couldn't find that in the indexed pages")) sources = [];
  box.querySelector(".cards").innerHTML = sources.map((s) => `
    <div class="card ${used.size ? (used.has(s.n) ? "used" : "unused") : ""}" data-n="${s.n}">
      <span class="n">[${s.n}]</span><span class="ct">${esc(s.title)}</span>
      ${s.section ? `<span class="sec">§ ${esc(s.section)}</span>` : ""}
      <span class="snip">${esc(s.text)}</span>
      <div class="small muted" style="margin-top:4px">checked ${ago(s.checked_at)} · <a href="${esc(s.url)}" target="_blank" rel="noopener">open ↗</a></div>
    </div>`).join("");
}

async function ask(question) {
  addMessage("user", esc(question));
  const box = addMessage("bot", `<div class="answer typing"></div><div class="cards"></div><div class="query-note"></div>`);
  const answerEl = box.querySelector(".answer");
  let text = "", sources = [];
  const btn = $("#ask-btn");
  btn.disabled = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: chatHistory }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!chunk.startsWith("data: ")) continue;
        const ev = JSON.parse(chunk.slice(6));
        if (ev.type === "sources") {
          sources = ev.sources;
          if (ev.query && ev.query !== question) box.querySelector(".query-note").textContent = `Searched for: “${ev.query}”`;
        } else if (ev.type === "token") {
          text += ev.text;
          answerEl.innerHTML = md(text);
        } else if (ev.type === "error") {
          answerEl.innerHTML += `<p class="err">${esc(ev.message)}</p>`;
        }
      }
      box.scrollIntoView({ block: "end" });
    }
  } catch (e) {
    answerEl.innerHTML += `<p class="err">${esc(e.message)}</p>`;
  } finally {
    answerEl.classList.remove("typing");
    btn.disabled = false;
  }
  renderCards(box, sources, text);
  chatHistory.push({ role: "user", content: question }, { role: "assistant", content: text });
}

$("#messages").addEventListener("click", (ev) => {
  const cite = ev.target.closest(".cite");
  const card = ev.target.closest(".card");
  if (cite) {
    const target = cite.closest(".msg").querySelector(`.card[data-n="${cite.dataset.n}"]`);
    if (!target) return;
    target.classList.add("hl", "open");
    target.scrollIntoView({ block: "nearest", behavior: "smooth" });
    setTimeout(() => target.classList.remove("hl"), 1600);
  } else if (card && !ev.target.closest("a")) {
    card.classList.toggle("open");
  }
});

const q = $("#question");
q.addEventListener("input", () => { q.style.height = "auto"; q.style.height = q.scrollHeight + "px"; });
q.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#ask-form").requestSubmit(); }
});
$("#ask-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const question = q.value.trim();
  if (!question || $("#ask-btn").disabled) return;
  q.value = ""; q.style.height = "auto";
  ask(question);
});

refreshAll();
