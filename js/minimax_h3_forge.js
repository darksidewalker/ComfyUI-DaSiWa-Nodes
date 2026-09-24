// MiniMax H3 Forge: the overlay behind the Director's "Forge" button.
//
// Write an idea, pick a local model, get a prompt in the Director's own
// fields. Runs through /dasiwa/h3/forge (nodes/h3_forge.py), outside the
// ComfyUI queue: the LLM writes, unloads, and only then is the workflow run.
// The Director exposes node.__dasiwaH3Forge for reading the timeline and
// writing the result back.
import { api } from "../../scripts/api.js";

const STORE_KEY = "dasiwa.h3forge";
const briefs = new Map(); // node id -> last brief, for a reroll after closing

function remembered() { try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch { return {}; } }
function remember(patch) { try { localStorage.setItem(STORE_KEY, JSON.stringify({ ...remembered(), ...patch })); } catch { /* private window */ } }

function installStyles() {
  if (document.getElementById("ds-h3-forge-styles")) return;
  const style = document.createElement("style");
  style.id = "ds-h3-forge-styles";
  style.textContent = `
  .ds-h3-forge-btn{background:rgba(151,91,255,.14)!important;color:#e6d9ff!important;border-color:rgba(177,128,255,.7)!important}
  .ds-h3-forge-btn:hover{box-shadow:0 0 10px rgba(151,91,255,.6)}
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
  const close = () => { overlay.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = e => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  overlay.addEventListener("pointerdown", e => { if (e.target === overlay) close(); });

  const closeBtn = el("button", { textContent: "×", title: "Close (Esc)", onclick: close });
  box.append(el("h3", {}, el("span", { textContent: `H3 Forge — ${mode}` }), closeBtn));

  const brief = el("textarea", { placeholder: "What should the clip be? A sentence or two is enough.", value: briefs.get(node.id) || "" });
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
    el("div", { className: "field" }, el("label", { textContent: "Model (Ollama)" }), modelSel),
    el("div", { className: "field" }, el("label", { textContent: "Creativity" }), creativity)));
  box.append(el("div", { className: "field" }, el("label", {}, "Detail ", detailLabel), detail));

  const status = el("span", { className: "status" });
  const setStatus = (msg, err = false) => { status.textContent = msg; status.classList.toggle("error", err); };
  const genBtn = el("button", { className: "primary", textContent: "Generate" });
  const applyBtn = el("button", { textContent: "Apply to node", disabled: true });
  box.append(el("div", { className: "actions" }, status, genBtn, applyBtn));
  const output = el("pre", { hidden: true });
  box.append(output);
  document.body.append(overlay);
  brief.focus();

  let levels = {};
  try {
    const res = await api.fetchApi("/dasiwa/h3/forge/models");
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || res.statusText);
    if (!data.models.length) throw new Error("Ollama has no models installed. Pull one first, e.g. ollama pull qwen3-vl:8b");
    for (const m of data.models) modelSel.append(el("option", { value: m.name, textContent: `${m.name}${m.parameters ? ` (${m.parameters})` : ""}` }));
    if (data.models.some(m => m.name === prefs.model)) modelSel.value = prefs.model;
    for (const c of data.creativity) creativity.append(el("option", { value: c, textContent: c[0].toUpperCase() + c.slice(1) }));
    creativity.value = prefs.creativity && data.creativity.includes(prefs.creativity) ? prefs.creativity : data.default_creativity;
    levels = data.detail_levels;
    detail.value = prefs.detail || data.default_detail;
  } catch (err) {
    setStatus(`Cannot list models: ${err.message}`, true);
    genBtn.disabled = true;
  }
  const syncDetail = () => { detailLabel.textContent = `${detail.value} of 10 — ${levels[detail.value] || ""}`; };
  detail.oninput = syncDetail; syncDetail();

  let result = null;
  genBtn.onclick = async () => {
    const text = brief.value.trim();
    if (!text) { setStatus("Write the idea first.", true); return; }
    briefs.set(node.id, text);
    remember({ model: modelSel.value, creativity: creativity.value, detail: Number(detail.value) });
    genBtn.disabled = true; applyBtn.disabled = true; result = null;
    const started = Date.now();
    const tick = setInterval(() => setStatus(`Writing with ${modelSel.value}… ${Math.round((Date.now() - started) / 1000)}s (the model unloads when it finishes)`), 500);
    try {
      const res = await api.fetchApi("/dasiwa/h3/forge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brief: text, mode, duration: hook.duration(), model: modelSel.value,
          detail: Number(detail.value), creativity: creativity.value,
          references: refs.map(({ item, ...r }) => r),
        }),
      });
      const data = await res.json();
      if (!res.ok) { output.hidden = !data.raw; output.textContent = data.raw || ""; throw new Error(data.message || res.statusText); }
      result = data;
      output.hidden = false;
      output.textContent = data.simple_prompt;
      const seen = data.saw_images ? ` · looked at ${data.saw_images} picture${data.saw_images === 1 ? "" : "s"}` : refs.some(r => r.kind === "image") && !data.vision ? " · this model cannot see images" : "";
      setStatus(`Done in ${data.stats.seconds}s · ${data.stats.output_tokens} tokens${seen} · ${data.unloaded ? "model unloaded" : "WARNING: model still loaded"}`, !data.unloaded);
      applyBtn.disabled = false;
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      clearInterval(tick);
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

window.DaSiWaH3Forge = { open };
