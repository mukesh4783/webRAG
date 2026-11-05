"use strict";

/* WebRAG front-end: vanilla JS, no build step. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const NOT_FOUND = "I couldn't find that in the indexed pages.";
const GLOWS = ["var(--g-blue)", "var(--g-green)", "var(--g-orange)", "var(--g-pink)", "var(--g-purple)"];
const chatHistory = [];
let sources = [];
let status = {};

// ------------------------------------------------------------------ utilities
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const host = (url) => { try { return new URL(url).host.replace(/^www\./, ""); } catch { return url; } };
const where = (url) => { try { const u = new URL(url); return u.host.replace(/^www\./, "") + u.pathname; } catch { return url; } };
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function ago(ts, short = false) {
  if (!ts) return "never";
  const s = Date.now() / 1000 - ts;
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)}${short ? "m" : " min"} ago`;
  if (s < 86400) return `${Math.round(s / 3600)}${short ? "h" : " h"} ago`;
  return `${Math.round(s / 86400)}${short ? "d" : " d"} ago`;
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
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = label;
  try { return await fn(); }
  catch (e) { toast(e.message, null, true); }
  finally { btn.disabled = false; btn.innerHTML = old; }
}

function toast(message, action = null, error = false) {
  const el = document.createElement("div");
  el.className = `toast${error ? " error" : ""}`;
  el.innerHTML = `<span class="dot"></span><span>${esc(message)}</span>${action ? `<a href="${action.href}">${esc(action.label)}</a>` : ""}`;
  $("#toasts").appendChild(el);
  el.querySelector("a")?.addEventListener("click", () => el.remove());
  setTimeout(() => el.remove(), 7000);
}

// XSS-safe mini markdown; [n] citations become clickable chips.
function md(src) {
  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/_\(([^)]+)\)_/g, "<em>($1)</em>")
    .replace(/\[(\d+)\]/g, '<span class="cite" data-n="$1" role="button" tabindex="0">$1</span>');
  const out = [];
  let list = null;
  for (const line of src.split("\n")) {
    const m = line.match(/^\s*(?:[-*•]|(\d+)\.)\s+(.*)$/);
    if (m) {
      const tag = m[1] ? "ol" : "ul";
      if (list !== tag) { if (list) out.push(`</${list}>`); list = tag; out.push(`<${tag}>`); }
      out.push(`<li>${inline(m[2])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list}>`); list = null; }
    const h = line.match(/^#{1,4}\s+(.*)$/);
    if (h) out.push(`<p><strong>${inline(h[1])}</strong></p>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`</${list}>`);
  return out.join("");
}

// ------------------------------------------------------------------ diff → Before / After rows
function wordOps(a, b) {
  const A = a.split(/(\s+)/), B = b.split(/(\s+)/);
  if (A.length * B.length > 200000) return [["del", a], ["ins", b]];
  const dp = Array.from({ length: A.length + 1 }, () => new Uint16Array(B.length + 1));
  for (let i = A.length - 1; i >= 0; i--)
    for (let j = B.length - 1; j >= 0; j--)
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  const push = (t, s) => (ops.length && ops.at(-1)[0] === t ? (ops.at(-1)[1] += s) : ops.push([t, s]));
  let i = 0, j = 0;
  while (i < A.length && j < B.length) {
    if (A[i] === B[j]) { push("eq", A[i]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) push("del", A[i++]);
    else push("ins", B[j++]);
  }
  while (i < A.length) push("del", A[i++]);
  while (j < B.length) push("ins", B[j++]);
  return ops;
}

function changePairs(lines) {
  const body = lines.filter((l) => !l.startsWith("---") && !l.startsWith("+++"));
  const pairs = [];
  for (let k = 0; k < body.length; k++) {
    if (!body[k].startsWith("-") && !body[k].startsWith("+")) continue;
    const dels = [], adds = [];
    while (k < body.length && body[k].startsWith("-")) dels.push(body[k++].slice(1).replace(/^##\s+/, ""));
    while (k < body.length && body[k].startsWith("+")) adds.push(body[k++].slice(1).replace(/^##\s+/, ""));
    k--;
    for (let p = 0; p < Math.max(dels.length, adds.length); p++) pairs.push([dels[p], adds[p]]);
  }
  return pairs;
}

function beforeAfter(lines, max = 4) {
  const pairs = changePairs(lines);
  const rows = pairs.slice(0, max).map(([d, a]) => {
    let before = "", after = "";
    if (d !== undefined && a !== undefined) {
      for (const [t, s] of wordOps(d, a)) {
        if (t === "eq") { before += esc(s); after += esc(s); }
        else if (t === "del") before += `<del>${esc(s)}</del>`;
        else after += `<ins>${esc(s)}</ins>`;
      }
    } else if (d !== undefined) before = `<del>${esc(d)}</del>`;
    else after = `<ins>${esc(a)}</ins>`;
    return `<div class="ba"><span class="k">Before</span><span class="v">${before || '<span class="k">(nothing)</span>'}</span>
            <span class="k after">After</span><span class="v">${after || '<span class="k">(removed)</span>'}</span></div>`;
  });
  if (pairs.length > max) rows.push(`<div class="ba-more">+ ${plural(pairs.length - max, "more edit")}</div>`);
  return rows.join("");
}

// ------------------------------------------------------------------ routing
function show(view) {
  if (!["ask", "sources", "changes"].includes(view)) view = "ask";
  $$(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
  $$(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  if (view === "changes") { loadChanges(); localStorage.setItem("webrag-seen", String(Date.now() / 1000)); $("#tab-changes").hidden = true; }
  if (view === "ask") setTimeout(() => $("#question").focus(), 0);
  window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", () => show(location.hash.slice(1)));

// ------------------------------------------------------------------ data loading
async function loadStatus() {
  try {
    status = await api("/api/status");
    const model = (status.llm || "no model").replace(/^ollama:/, "");
    const emb = (status.embedder || "keywords only").replace(/^ollama:/, "");
    $("#chip-model").textContent = model;
    $("#chip-retrieval").textContent = status.embedder ? "hybrid search" : "keyword search";
    $("#health").textContent = `${model} · ${emb}`;
    $(".health").classList.remove("off");
  } catch {
    $("#health").textContent = "server unreachable";
    $(".health").classList.add("off");
  }
}

async function loadSources(flash = []) {
  sources = await api("/api/sources");
  $("#tab-sources").textContent = sources.length;
  $("#sources-count").textContent = plural(sources.length, "source");
  renderSources(flash);
  renderStats();
  renderSuggestions();
  updateChangeBadge();
}

function updateChangeBadge() {
  const seen = Number(localStorage.getItem("webrag-seen") || 0);
  const fresh = sources.filter((s) => s.n_versions > 1 && s.changed_at > seen).length;
  const badge = $("#tab-changes");
  badge.textContent = fresh;
  badge.hidden = !fresh || location.hash === "#changes";
}

// ------------------------------------------------------------------ stats panel
function renderStats() {
  const ok = sources.filter((s) => s.status !== "error");
  const passages = sources.reduce((n, s) => n + (s.n_chunks || 0), 0);
  const revisions = sources.reduce((n, s) => n + Math.max(0, (s.n_versions || 1) - 1), 0);
  const top = [...sources].sort((a, b) => b.n_chunks - a.n_chunks).slice(0, 3);
  const maxChunks = Math.max(1, ...top.map((s) => s.n_chunks));
  const checks = sources.map((s) => s.checked_at).filter(Boolean);
  const newest = checks.length ? Math.max(...checks) : 0;
  const oldest = checks.length ? Math.min(...checks) : 0;
  const span = Math.max(1, newest - oldest);
  const lastAgo = newest ? ago(newest, true).replace(" ago", "") : "none";

  const squares = sources.slice(0, 24).map((s) => `<i class="${s.status === "error" ? "err" : ""}" title="${esc(s.title || s.url)}"></i>`).join("")
    || '<i class="empty"></i><i class="empty"></i><i class="empty"></i>';

  $("#stats").innerHTML = `
    <div class="stat">
      <div class="stat-label"><i>◎</i>Sources indexed</div>
      <div class="stat-num">${sources.length}</div>
      <div class="stat-desc">${ok.length} healthy${sources.length - ok.length ? `, ${sources.length - ok.length} unreachable` : ""}</div>
      <div class="stat-foot"><div class="squares">${squares}</div><div class="squares-note">one square per page</div></div>
    </div>
    <div class="stat">
      <div class="stat-label"><i>¶</i>Passages searchable</div>
      <div class="stat-num">${passages.toLocaleString()}</div>
      <div class="stat-desc">Section-aware chunks with keyword and vector search</div>
      <div class="stat-foot"><div class="bars">${top.map((s, i) => `
        <span title="${esc(s.title)}">${esc((s.title || host(s.url)).slice(0, 12))}</span>
        <div class="track"><div class="fill ${i === 0 ? "blue" : ""}" style="width:${(s.n_chunks / maxChunks) * 100}%"></div></div>
        <span style="text-align:right">${s.n_chunks}</span>`).join("") || '<span>—</span><div class="track"></div><span></span>'}</div></div>
    </div>
    <div class="stat">
      <div class="stat-label"><i>↻</i>Freshness</div>
      <div class="stat-num">${esc(lastAgo === "just now" ? "now" : lastAgo)}${lastAgo !== "just now" && lastAgo !== "none" ? "<small>ago</small>" : ""}</div>
      <div class="stat-desc">Since the most recent recheck</div>
      <div class="stat-foot">
        <div class="timeline">${checks.map((c) => `<b class="${c === newest ? "now" : ""}" style="left:${checks.length > 1 ? ((c - oldest) / span) * 100 : 100}%"></b>`).join("")}</div>
        <div class="timeline-labels"><span>oldest check</span><span>latest</span></div>
      </div>
    </div>
    <div class="stat">
      <div class="stat-label"><i>Δ</i>Revisions detected</div>
      <div class="stat-num">${revisions}</div>
      <div class="stat-desc">Page edits caught, summarised and re-indexed</div>
      <div class="stat-foot pills">
        <span class="chip blue">${esc((status.llm || "no model").replace(/^ollama:/, ""))}</span>
        <span class="chip">${esc((status.embedder || "keywords").replace(/^ollama:/, ""))}</span>
      </div>
    </div>`;
}

function renderSuggestions() {
  const short = (t) => { t = t.split(/\s+[-–|]\s+/)[0]; return t.length > 36 ? t.slice(0, 34) + "…" : t; };
  const titles = sources.filter((s) => s.n_chunks).map((s) => short(s.title || host(s.url))).slice(0, 2);
  const ideas = titles.length
    ? [`Summarise ${titles[0]}`, ...(titles[1] ? [`Compare ${titles[0]} and ${titles[1]}`] : []), "What changed recently?"]
    : [];
  $("#suggest").innerHTML = ideas.map((q) => `<button type="button">${esc(q)}</button>`).join("")
    || `<button type="button" data-go="sources">Add your first source →</button>`;
}
$("#suggest").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  if (b.dataset.go) { location.hash = b.dataset.go; return; }
  $("#question").value = b.textContent;
  $("#ask-form").requestSubmit();
});

// ------------------------------------------------------------------ sources view
const GLOBE = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>`;

function renderSources(flash = []) {
  if (!sources.length) {
    $("#sources").innerHTML = `<div class="empty" style="grid-column:1/-1"><h3>No sources yet</h3>Paste a URL above. Wikipedia articles, docs and blog posts all work well.</div>`;
    return;
  }
  $("#sources").innerHTML = sources.map((s, i) => {
    const changed = s.n_versions > 1;
    return `<article class="card ${flash.includes(s.id) ? "flash" : ""}" style="--glow:${s.status === "error" ? "var(--magenta)" : GLOWS[i % GLOWS.length]}">
      <div class="badge">${GLOBE}</div>
      <h3>${esc(s.title || host(s.url))}</h3>
      <a class="url" href="${esc(s.url)}" target="_blank" rel="noopener" title="${esc(s.url)}">${esc(where(s.url))}</a>
      <div class="meta">
        <span class="chip">${plural(s.n_chunks, "passage")}</span>
        <span class="chip ${changed ? "blue" : ""}">${plural(s.n_versions, "version")}</span>
        <span class="chip">checked ${ago(s.checked_at, true)}</span>
      </div>
      ${s.error ? `<div class="err">${esc(s.error)}</div>` : ""}
      <div class="actions">
        <button class="btn btn-outline btn-sm" data-act="recheck" data-id="${s.id}">Recheck</button>
        <button class="btn btn-outline btn-sm" data-act="history" data-id="${s.id}">History</button>
        <button class="btn btn-ghost btn-sm" data-act="remove" data-id="${s.id}">Remove</button>
      </div>
    </article>`;
  }).join("");
}

function summarise(events) {
  const c = (st) => events.filter((e) => e.status === st).length;
  const parts = [];
  if (c("new")) parts.push(`${c("new")} added`);
  if (c("changed")) parts.push(`${c("changed")} changed`);
  if (c("unchanged")) parts.push(`${c("unchanged")} unchanged`);
  if (c("error")) parts.push(`${c("error")} failed`);
  return parts.join(" · ") || "nothing to check";
}

async function afterRefresh(events, title) {
  await loadSources(events.filter((e) => e.status === "changed" || e.status === "new").map((e) => e.source_id));
  const changed = events.some((e) => e.status === "changed");
  const failed = events.filter((e) => e.status === "error");
  toast(`${title}: ${summarise(events)}`, changed ? { href: "#changes", label: "View changes" } : null, failed.length > 0 && !changed);
  failed.forEach((e) => toast(`${host(e.url)}: ${e.error}`, null, true));
}

$("#sources").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const id = Number(btn.dataset.id);
  if (btn.dataset.act === "history") return showHistory(id);
  if (btn.dataset.act === "remove") {
    if (!confirm("Remove this source and its history?")) return;
    await api(`/api/sources/${id}`, { method: "DELETE" });
    return loadSources();
  }
  await busy(btn, "Checking…", async () => {
    const events = await api("/api/refresh", { method: "POST", body: JSON.stringify({ ids: [id] }) });
    await afterRefresh(events, "Recheck");
  });
});

$("#add-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const urls = $("#urls").value.split(/\s+/).map((u) => u.trim()).filter(Boolean);
  if (!urls.length) return $("#urls").focus();
  const depth = Number(new FormData(ev.target).get("depth"));
  await busy(ev.submitter || $("button[type=submit]", ev.target), "Indexing…", async () => {
    const events = await api("/api/sources", { method: "POST", body: JSON.stringify({ urls, depth }) });
    $("#urls").value = "";
    await afterRefresh(events, "Indexing done");
    await loadStatus();
  });
});

$("#recheck-all").addEventListener("click", (ev) =>
  busy(ev.currentTarget, "Rechecking…", async () => {
    const events = await api("/api/refresh", { method: "POST", body: "{}" });
    await afterRefresh(events, "Recheck");
    if (location.hash === "#changes") loadChanges();
  }));

// ------------------------------------------------------------------ change log
function changeCard(v, src, compact = false) {
  const d = new Date(v.fetched_at * 1000);
  const side = `<div class="change-side">
      <div class="when">${d.toLocaleDateString(undefined, { day: "numeric", month: "short" })}</div>
      <div class="time">${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${ago(v.fetched_at, true)}</div>
      ${compact ? "" : `<div class="src">${esc(src.title || host(src.url))}</div><div class="src-url">${esc(host(src.url))}</div>`}
    </div>`;
  if (v.initial) {
    return `<div class="change added">${side}<div class="change-main">First indexed. This is the baseline future versions are compared against.</div></div>`;
  }
  return `<div class="change">${side}<div class="change-main">
      <p class="summary ${v.summary ? "" : "muted"}">${esc(v.summary || "Text changed. No summary was generated (the model was unavailable).")}</p>
      <div class="tags"><span class="chip blue">+${v.added} lines</span><span class="chip pink">−${v.removed} lines</span></div>
      ${beforeAfter(v.diff)}
    </div></div>`;
}

async function loadChanges() {
  const box = $("#changes");
  if (!sources.length) {
    box.innerHTML = `<div class="empty"><h3>No sources yet</h3>Add a page, edit it, then press “Recheck sources”. The edit will show up here.</div>`;
    return;
  }
  const histories = await Promise.all(sources.map((s) => api(`/api/sources/${s.id}/history`).catch(() => null)));
  const items = [];
  histories.forEach((h) => h && h.versions.forEach((v) => items.push({ v, src: h.source })));
  items.sort((a, b) => b.v.fetched_at - a.v.fetched_at);
  const edits = items.filter((x) => !x.v.initial);
  box.innerHTML = (edits.length ? "" : `<div class="empty"><h3>No edits detected yet</h3>Your pages haven't changed since they were indexed. Use “Recheck sources” any time.</div>`)
    + items.map((x) => changeCard(x.v, x.src)).join("");
}

async function showHistory(id) {
  const data = await api(`/api/sources/${id}/history`);
  $("#history-title").textContent = data.source.title || host(data.source.url);
  $("#history-url").textContent = data.source.url;
  $("#history-url").href = data.source.url;
  $("#history-body").innerHTML = data.versions.map((v) => changeCard(v, data.source, true)).join("");
  $("#history").showModal();
}
$("#history").addEventListener("click", (ev) => {
  if (ev.target.closest("[data-close]") || ev.target === ev.currentTarget) ev.currentTarget.close();
});

// ------------------------------------------------------------------ ask
function renderRefs(card, refs, answer) {
  const box = $(".refs", card);
  const cited = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  if ((!cited.size && answer.includes(NOT_FOUND)) || !refs.length) { box.remove(); return; }
  const row = (r) => `<div class="ref" data-n="${r.n}">
      <span class="num">${r.n}</span>
      <div><div class="ref-title">${esc(r.title)}</div>${r.section ? `<div class="ref-sec">${esc(r.section.split(" > ").slice(-2).join(" › "))}</div>` : ""}</div>
      <span class="ref-meta">checked ${ago(r.checked_at, true)} · <a href="${esc(r.url)}" target="_blank" rel="noopener">open ↗</a></span>
      <blockquote>${esc(r.text)}</blockquote>
    </div>`;
  const used = cited.size ? refs.filter((r) => cited.has(r.n)) : refs;
  const rest = cited.size ? refs.filter((r) => !cited.has(r.n)) : [];
  box.innerHTML = `<div class="refs-head"><span class="label">References</span><span class="label">click to preview the passage</span></div>
    ${used.map(row).join("")}
    ${rest.length ? `<details class="more-refs"><summary>${plural(rest.length, "more passage")} retrieved but not cited</summary>${rest.map(row).join("")}</details>` : ""}`;
}

async function ask(question) {
  document.body.classList.add("chatting");
  const card = document.createElement("article");
  card.className = "answer";
  card.innerHTML = `
    <div class="answer-head"><h3>${esc(question)}</h3><span class="chip">searching…</span></div>
    <div class="answer-body typing"></div>
    <div class="searched"></div>
    <div class="refs"></div>`;
  $("#thread").appendChild(card);
  card.scrollIntoView({ behavior: "smooth", block: "start" });

  const body = $(".answer-body", card);
  const chip = $(".answer-head .chip", card);
  let text = "", refs = [];
  $("#ask-btn").disabled = true;
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
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!raw.startsWith("data: ")) continue;
        const ev = JSON.parse(raw.slice(6));
        if (ev.type === "sources") {
          refs = ev.sources;
          chip.textContent = `reading ${plural(refs.length, "passage")}…`;
          if (ev.query && ev.query.toLowerCase() !== question.toLowerCase()) $(".searched", card).textContent = `searched for: ${ev.query}`;
        } else if (ev.type === "token") {
          text += ev.text;
          body.innerHTML = md(text);
        } else if (ev.type === "error") {
          body.insertAdjacentHTML("beforeend", `<p class="err">${esc(ev.message)}</p>`);
        }
      }
    }
  } catch (e) {
    body.insertAdjacentHTML("beforeend", `<p class="err">${esc(e.message)}</p>`);
  } finally {
    body.classList.remove("typing");
    $("#ask-btn").disabled = false;
  }

  const nCited = new Set([...text.matchAll(/\[(\d+)\]/g)].map((m) => m[1])).size;
  const refused = !nCited && text.includes(NOT_FOUND);
  if (refused) {
    body.innerHTML = `<div class="notfound"><b>Not in your sources.</b><span>None of the indexed pages answer this, so WebRAG won't guess. Try adding a page that covers it.</span></div><p></p>`;
    chip.className = "chip pink"; chip.textContent = "not in sources";
  } else if (nCited) {
    chip.className = "chip green"; chip.textContent = `grounded in ${plural(nCited, "source")}`;
  } else {
    chip.className = "chip"; chip.textContent = "no citations";
  }
  renderRefs(card, refs, text);
  chatHistory.push({ role: "user", content: question }, { role: "assistant", content: text });
}

$("#thread").addEventListener("click", (ev) => {
  const cite = ev.target.closest(".cite");
  if (cite) {
    const ref = $(`.ref[data-n="${cite.dataset.n}"]`, cite.closest(".answer"));
    if (!ref) return;
    const more = ref.closest("details");
    if (more) more.open = true;
    ref.classList.add("open", "hl");
    ref.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setTimeout(() => ref.classList.remove("hl"), 1400);
    return;
  }
  const ref = ev.target.closest(".ref");
  if (ref && !ev.target.closest("a")) ref.classList.toggle("open");
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
  q.value = "";
  q.style.height = "auto";
  ask(question);
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "/" && !["TEXTAREA", "INPUT"].includes(document.activeElement.tagName)) {
    ev.preventDefault();
    if (location.hash && location.hash !== "#ask") location.hash = "ask";
    q.focus();
  }
});

// ------------------------------------------------------------------ boot
(async () => {
  await loadStatus();
  await loadSources().catch((e) => toast(e.message, null, true));
  show(location.hash.slice(1) || "ask");
  setInterval(renderStats, 30000);
})();
