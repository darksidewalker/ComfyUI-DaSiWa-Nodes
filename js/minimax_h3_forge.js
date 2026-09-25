// MiniMax H3 Forge: the overlay behind the Director's "Forge" button.
//
// Write an idea, pick a local model, get a prompt in the Director's own
// fields. Runs through /dasiwa/h3/forge (nodes/h3_forge.py), outside the
// ComfyUI queue: the LLM writes, unloads, and only then is the workflow run.
// The Director exposes node.__dasiwaH3Forge for reading the timeline and
// writing the result back.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Server addresses live in ComfyUI Settings, never in the workflow, so a
// downloaded workflow cannot point this machine at a server of its choosing.
const SETTING_OLLAMA = "DaSiWa.H3Forge.OllamaURL";
const SETTING_OPENAI = "DaSiWa.H3Forge.OpenAIURL";
app.registerExtension({
  name: "DaSiWa.H3Forge",
  settings: [
    { id: SETTING_OLLAMA, category: ["DaSiWa", "H3 Forge", "Ollama address"], name: "Ollama address", type: "text", defaultValue: "", tooltip: "Leave empty for Ollama on this computer (http://127.0.0.1:11434). Set it to use Ollama on another machine." },
    { id: SETTING_OPENAI, category: ["DaSiWa", "H3 Forge", "OpenAI-compatible server"], name: "OpenAI-compatible server address", type: "text", defaultValue: "", tooltip: "Optional: a llama.cpp server, llama-swap, LM Studio or koboldcpp, e.g. http://127.0.0.1:8080. Empty = off." },
  ],
});
function settingValue(id) {
  try { return app.extensionManager?.setting?.get(id) ?? app.ui?.settings?.getSettingValue(id) ?? ""; } catch { return ""; }
}
const forgeSettings = () => ({ ollama_url: settingValue(SETTING_OLLAMA) || "", openai_url: settingValue(SETTING_OPENAI) || "" });
const SOURCE_NAME = { local: "ComfyUI models/llm (loads inside ComfyUI)", ollama: "Ollama", openai: "OpenAI-compatible server" };
const NO_MODELS = "No models found. Easiest fix: put a vision model folder (for example Qwen3-VL-8B-Instruct from Hugging Face) in ComfyUI/models/llm and reopen Forge. Or install Ollama and run: ollama pull qwen3-vl:8b. Other servers: Settings > DaSiWa > H3 Forge.";

const STORE_KEY = "dasiwa.h3forge";
const briefs = new Map(); // node id -> last brief, for a reroll after closing
const HISTORY_KEY = "dasiwaH3ForgeHistory";
function forgeHistory(node) {
  const saved = node.properties?.[HISTORY_KEY];
  return Array.isArray(saved) ? saved.filter(entry => entry && typeof entry.simple_prompt === "string" && entry.simple_prompt.trim() && typeof entry.mode === "string" && entry.fields && typeof entry.fields === "object").slice(0, 3) : [];
}
function saveForgeResult(node, result, brief) {
  const entry = {
    mode: result.mode, model: result.model, simple_prompt: result.simple_prompt,
    fields: result.fields, brief, createdAt: Date.now(),
  };
  node.properties ||= {};
  node.properties[HISTORY_KEY] = [entry, ...forgeHistory(node)].slice(0, 3);
  node.graph?.setDirtyCanvas(true, true);
  node.__dasiwaH3Render?.(); // the Director Clear button must enable even when only drafts exist
  return entry;
}

function clearForgeHistory(node) {
  if (node.properties && HISTORY_KEY in node.properties) {
    delete node.properties[HISTORY_KEY];
    node.graph?.setDirtyCanvas(true, true);
  }
  briefs.delete(node.id);
}

function remembered() { try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch { return {}; } }
function remember(patch) { try { localStorage.setItem(STORE_KEY, JSON.stringify({ ...remembered(), ...patch })); } catch { /* private window */ } }

function installStyles() {
  if (document.getElementById("ds-h3-forge-styles")) return;
  const style = document.createElement("style");
  style.id = "ds-h3-forge-styles";
  style.textContent = `
  .ds-h3-modebar .ds-h3-forge-btn,.ds-h3-forge-btn{background:rgba(151,91,255,.14)!important;color:#e6d9ff!important;border-color:rgba(177,128,255,.7)!important}
  .ds-h3-modebar .ds-h3-forge-btn:hover,.ds-h3-forge-btn:hover{box-shadow:0 0 10px rgba(151,91,255,.6)}
  .ds-forge-overlay{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center}
  .ds-forge{width:min(760px,94vw);max-height:90vh;overflow:auto;background:#111820;color:#e5eef4;border:1px solid #40515e;border-radius:8px;padding:14px;font:13px system-ui,sans-serif;display:flex;flex-direction:column;gap:10px;box-shadow:0 10px 40px rgba(0,0,0,.6)}
  .ds-forge h3{margin:0;font-size:15px;display:flex;justify-content:space-between;align-items:center}
  .ds-forge label{color:#9fb3c2;font-weight:600;font-size:12px}
  .ds-forge textarea,.ds-forge select,.ds-forge input[type=text]{width:100%;box-sizing:border-box;background:#0d1217;color:#e5eef4;border:1px solid #40515e;border-radius:4px;padding:7px;font:inherit}
  .ds-forge textarea{min-height:90px;resize:vertical}
  .ds-forge .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .ds-forge .field{display:flex;flex-direction:column;gap:4px}
  .ds-forge input[type=range]{width:100%}
  .ds-forge button{background:#202b35;color:#dbe7f0;border:1px solid #40515e;border-radius:4px;padding:6px 12px;cursor:pointer;font:inherit}
  .ds-forge button:hover{background:#2c3c49}
  .ds-forge button.primary{background:rgba(151,91,255,.3);border-color:rgba(177,128,255,.8);color:#fff;font-weight:600}
  .ds-forge button:disabled{opacity:.45;cursor:default}
  .ds-forge .actions{display:flex;gap:8px;justify-content:flex-end;align-items:center}
  .ds-forge .status{flex:1;color:#f3c67a;min-height:16px}
  .ds-forge .status.error{color:#ff8a8a}
  .ds-forge .muted{color:#8fa3b2;font-size:12px}
  .ds-forge .refs{display:flex;flex-direction:column;gap:6px}
  .ds-forge .ref{display:grid;grid-template-columns:48px 90px 130px 1fr;gap:8px;align-items:center}
  .ds-forge .ref img{width:48px;height:36px;object-fit:cover;border-radius:3px;background:#090d11}
  .ds-forge pre{white-space:pre-wrap;background:#0b1015;border:1px solid #344452;border-radius:4px;padding:8px;margin:0;max-height:320px;overflow:auto;font:12px/1.45 ui-monospace,monospace}
  .ds-forge .history{display:flex;flex-direction:column;gap:5px;border-top:1px solid #344452;padding-top:9px}
  .ds-forge .history-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
  .ds-forge .history-head button{padding:3px 8px;font-size:11px}
  .ds-forge .history button{text-align:left;display:flex;flex-direction:column;gap:3px;min-width:0}
  .ds-forge .history button.selected{border-color:#b180ff;background:rgba(151,91,255,.18)}
  .ds-forge .history .excerpt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#9fb3c2;font-size:11px}
  `;
  document.head.append(style);
}

const el = (tag, props = {}, ...children) => { const n = Object.assign(document.createElement(tag), props); n.append(...children); return n; };
const viewUrl = path => api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`);
const BASE_ROLE = { I2VA: "first frame", FL2VA: "first / last frame", L2VA: "last frame" };

function referencesFor(hook) {
  const mode = hook.mode();
  const laneOrder = { image: 0, video: 1, audio: 2 };
  return hook.items()
    .sort((a, b) => (laneOrder[a.lane] - laneOrder[b.lane]) || (a.slot - b.slot))
    .map(item => {
      if (item.lane === "image") return { item, kind: "image", path: item.value, role: mode === "REF2VA" ? (item.forge_role || "subject") : "keyframe" };
      if (item.lane === "audio" && item.type === "audio") return { item, kind: "audio", duration_seconds: item.duration };
      return { item, kind: "video", role: "motion", stream: item.media_mode === "audio" ? "audio" : item.media_mode === "video_audio" ? "both" : "video", duration_seconds: item.duration };
    });
}

async function open(node) {
  const hook = node.__dasiwaH3Forge;
  if (!hook) return;
  installStyles();
  const mode = hook.mode();
  const prefs = remembered();

  const overlay = el("div", { className: "ds-forge-overlay" });
  const box = el("div", { className: "ds-forge" });
  overlay.append(box);
  // A run in flight, so Cancel and closing the pop-out can stop it.
  let running = null;
  const cancelRun = () => { if (running) { const id = running; running = null; api.fetchApi("/dasiwa/h3/forge/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: id }) }).catch(() => {}); } };
  const close = () => { cancelRun(); overlay.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = e => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  overlay.addEventListener("pointerdown", e => { if (e.target === overlay) close(); });

  const closeBtn = el("button", { textContent: "×", title: "Close (Esc)", onclick: close });
  box.append(el("h3", {}, el("span", { textContent: `H3 Forge — ${mode}` }), closeBtn));

  const brief = el("textarea", { placeholder: "What should the clip be? A sentence or two is enough.", value: briefs.get(node.id) || forgeHistory(node)[0]?.brief || "" });
  box.append(el("div", { className: "field" }, el("label", { textContent: "Idea" }), brief));

  // References from the timeline. REF2VA pictures need a role; base-mode
  // pictures are frames by definition.
  const refs = referencesFor(hook);
  if (refs.length) {
    const list = el("div", { className: "refs" });
    let counts = { image: 0, video: 0, audio: 0 };
    for (const ref of refs) {
      counts[ref.kind] += 1;
      const name = `${ref.kind === "image" ? "Picture" : ref.kind === "video" ? "Video" : "Audio"} ${counts[ref.kind]}`;
      const thumb = ref.kind === "image" ? el("img", { src: viewUrl(ref.path) }) : el("span", { className: "muted", textContent: ref.kind });
      let roleCell;
      if (ref.kind === "image" && mode === "REF2VA") {
        roleCell = el("select", { onchange: e => { ref.role = e.target.value; ref.item.forge_role = e.target.value; } });
        for (const r of ["subject", "style", "keyframe"]) roleCell.append(el("option", { value: r, textContent: r, selected: ref.role === r }));
      } else {
        roleCell = el("span", { className: "muted", textContent: ref.kind === "image" ? BASE_ROLE[mode] || "frame" : ref.kind === "video" ? `motion · ${ref.stream}` : "voice" });
      }
      const keep = el("input", { type: "text", placeholder: "keep (optional)", oninput: e => { ref.keep = e.target.value.trim(); } });
      list.append(el("div", { className: "ref" }, thumb, el("span", { textContent: name }), roleCell, ref.kind === "audio" ? el("span") : keep));
    }
    box.append(el("div", { className: "field" }, el("label", { textContent: "References on the timeline" }), list));
  } else if (mode !== "T2VA") {
    box.append(el("div", { className: "muted", textContent: `${mode} expects pictures on the timeline; none are loaded, so the model writes from the idea alone.` }));
  }

  const modelSel = el("select");
  const detail = el("input", { type: "range", min: 1, max: 10, step: 1 });
  const detailLabel = el("span", { className: "muted" });
  const creativity = el("select");
  box.append(el("div", { className: "row" },
    el("div", { className: "field" }, el("label", { textContent: "Model" }), modelSel),
    el("div", { className: "field" }, el("label", { textContent: "Creativity" }), creativity)));
  box.append(el("div", { className: "field" }, el("label", {}, "Detail ", detailLabel), detail));

  const status = el("span", { className: "status" });
  const setStatus = (msg, err = false) => { status.textContent = msg; status.classList.toggle("error", err); };
  const genBtn = el("button", { className: "primary", textContent: "Generate" });
  const applyBtn = el("button", { textContent: "Apply to node", disabled: true });
  box.append(el("div", { className: "actions" }, status, genBtn, applyBtn));
  const output = el("pre", { hidden: true });
  box.append(output);
  const historyBox = el("div", { className: "history" });
  box.append(historyBox);
  let result = null;
  const showResult = entry => {
    result = entry;
    output.hidden = false;
    output.textContent = entry.simple_prompt;
    applyBtn.disabled = !!running || entry.mode !== hook.mode();
    if (entry.mode !== hook.mode()) setStatus(`This draft is for ${entry.mode}; switch the Director to that mode before applying.`, true);
    renderHistory();
  };
  const renderHistory = () => {
    const entries = forgeHistory(node);
    const clear = el("button", { type: "button", textContent: "Clear history", disabled: !entries.length, title: "Remove the three saved Forge prompts from this node" });
    clear.onclick = () => {
      clearForgeHistory(node);
      node.__dasiwaH3Render?.();
      result = null; output.hidden = true; output.textContent = ""; applyBtn.disabled = true;
      renderHistory();
    };
    historyBox.replaceChildren(el("div", { className: "history-head" }, el("label", { textContent: "Last 3 generated prompts (saved with this node)" }), clear));
    if (!entries.length) { historyBox.append(el("span", { className: "muted", textContent: "No prompts generated yet." })); return; }
    entries.forEach((entry, index) => {
      const date = Number.isFinite(entry.createdAt) ? new Date(entry.createdAt).toLocaleString() : "Saved draft";
      const button = el("button", { type: "button", className: result === entry ? "selected" : "", title: "Show this prompt; Apply to node to use it" },
        el("span", { textContent: `${index + 1}. ${entry.mode} · ${entry.model || "model"} · ${date}` }),
        el("span", { className: "excerpt", textContent: entry.simple_prompt.replace(/\s+/g, " ").slice(0, 150) }));
      button.onclick = () => showResult(entry);
      historyBox.append(button);
    });
  };
  const latest = forgeHistory(node)[0];
  if (latest) showResult(latest); else renderHistory();
  document.body.append(overlay);
  brief.focus();

  const notes = el("div", { className: "muted" });
  box.insertBefore(notes, status.parentElement);
  let levels = {};
  try {
    const res = await api.fetchApi("/dasiwa/h3/forge/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ settings: forgeSettings() }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || res.statusText);
    notes.textContent = Object.values(data.errors || {}).join(" ");
    notes.style.color = notes.textContent ? "#ff8a8a" : "";
    const usable = data.models.filter(m => !m.disabled);
    if (!usable.length) throw new Error(NO_MODELS);
    for (const source of ["local", "ollama", "openai"]) {
      const group = data.models.filter(m => m.id.startsWith(source + ":"));
      if (!group.length) continue;
      const og = el("optgroup", { label: SOURCE_NAME[source] });
      for (const m of group) og.append(el("option", { value: m.id, textContent: m.label, disabled: !!m.disabled }));
      modelSel.append(og);
    }
    modelSel.value = usable.some(m => m.id === prefs.model) ? prefs.model : usable[0].id;
    for (const c of data.creativity) creativity.append(el("option", { value: c, textContent: c[0].toUpperCase() + c.slice(1) }));
    creativity.value = prefs.creativity && data.creativity.includes(prefs.creativity) ? prefs.creativity : data.default_creativity;
    levels = data.detail_levels;
    detail.value = prefs.detail || data.default_detail;
  } catch (err) {
    setStatus(err.message, true);
    genBtn.disabled = true;
  }
  const syncDetail = () => { detailLabel.textContent = `${detail.value} of 10 — ${levels[detail.value] || ""}`; };
  detail.oninput = syncDetail; syncDetail();

  genBtn.onclick = async () => {
    if (running) { cancelRun(); genBtn.disabled = true; setStatus("Cancelling… the model stops at its next token, then unloads."); return; }
    const text = brief.value.trim();
    if (!text) { setStatus("Write the idea first.", true); return; }
    briefs.set(node.id, text);
    remember({ model: modelSel.value, creativity: creativity.value, detail: Number(detail.value) });
    applyBtn.disabled = true;
    const requestId = `forge-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    running = requestId;
    genBtn.textContent = "Cancel";
    const started = Date.now();
    const tick = setInterval(() => setStatus(`Writing with ${modelSel.selectedOptions[0]?.textContent || modelSel.value}… ${Math.round((Date.now() - started) / 1000)}s (Cancel stops it; the model unloads either way)`), 500);
    try {
      const res = await api.fetchApi("/dasiwa/h3/forge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: requestId, brief: text, mode, duration: hook.duration(), model: modelSel.value,
          detail: Number(detail.value), creativity: creativity.value,
          references: refs.map(({ item, ...r }) => r), settings: forgeSettings(),
        }),
      });
      const data = await res.json();
      if (!res.ok) { output.hidden = !data.raw; output.textContent = data.raw || ""; throw new Error(data.message || res.statusText); }
      const saved = saveForgeResult(node, data, text);
      showResult(saved);
      const seen = data.saw_images ? ` · looked at ${data.saw_images} picture${data.saw_images === 1 ? "" : "s"}` : refs.some(r => r.kind === "image") && data.vision === false ? " · this model cannot see images, so it wrote from your idea only" : "";
      const warned = [...(data.warnings || []), ...(data.unloaded ? [] : ["WARNING: model still loaded"])];
      setStatus(`Done in ${data.stats.seconds}s${data.stats.output_tokens ? ` · ${data.stats.output_tokens} tokens` : ""}${seen}${warned.length ? " · " + warned.join(" · ") : " · model unloaded"}`, warned.length > 0);
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      clearInterval(tick);
      running = null;
      applyBtn.disabled = !result || result.mode !== hook.mode();
      genBtn.disabled = false;
      genBtn.textContent = "Regenerate";
    }
  };
  applyBtn.onclick = () => {
    if (!result) return;
    hook.apply(result);
    hook.setStatus(`Forge prompt applied (${result.model}).`);
    close();
  };
}

// At load, not on first open: the toolbar button's style lives here too.
installStyles();
window.DaSiWaH3Forge = { open, clearHistory: clearForgeHistory, settings: forgeSettings };
