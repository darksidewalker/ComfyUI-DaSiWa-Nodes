// MiniMax H3 Forge: the overlay behind the Director's "Forge" button.
//
// Write an idea, pick a local model, get a prompt in the Director's own
// fields. Runs through /dasiwa/h3/forge (nodes/h3_forge.py), outside the
// ComfyUI queue: the LLM writes, unloads, and only then is the workflow run.
// The Director exposes node.__dasiwaH3Forge for reading the timeline and
// writing the result back.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { forgeReferences, labelRole, refPromptFields, referenceSnapshot, referenceTags } from "./minimax_h3_forge_state.js";

// Server addresses live in ComfyUI Settings, never in the workflow, so a
// downloaded workflow cannot point this machine at a server of its choosing.
const SETTING_OLLAMA = "DaSiWa.H3Forge.OllamaURL";
const SETTING_OPENAI = "DaSiWa.H3Forge.OpenAIURL";
const SETTING_OPENAI_KEY = "DaSiWa.H3Forge.OpenAIKey";
app.registerExtension({
  name: "DaSiWa.H3Forge",
  settings: [
    { id: SETTING_OLLAMA, category: ["DaSiWa", "H3 Forge", "Ollama address"], name: "Ollama address", type: "text", defaultValue: "", tooltip: "Leave empty for Ollama on this computer (http://127.0.0.1:11434). Set it to use Ollama on another machine." },
    { id: SETTING_OPENAI, category: ["DaSiWa", "H3 Forge", "OpenAI-compatible server"], name: "OpenAI-compatible server address", type: "text", defaultValue: "", tooltip: "Optional: a llama.cpp server, llama-swap, LM Studio or koboldcpp, e.g. http://127.0.0.1:8080. Empty = off." },
    { id: SETTING_OPENAI_KEY, category: ["DaSiWa", "H3 Forge", "OpenAI-compatible API key"], name: "OpenAI-compatible API key", type: "text", defaultValue: "", tooltip: "Only if that server asks for one (llama-server --api-key, llama-swap apiKeys, LM Studio with authentication). Sent only to the address above. Stored in ComfyUI's settings file like every other setting." },
  ],
});
function settingValue(id) {
  try { return app.extensionManager?.setting?.get(id) ?? app.ui?.settings?.getSettingValue(id) ?? ""; } catch { return ""; }
}
const forgeSettings = () => ({ ollama_url: settingValue(SETTING_OLLAMA) || "", openai_url: settingValue(SETTING_OPENAI) || "", openai_api_key: settingValue(SETTING_OPENAI_KEY) || "" });
const SOURCE_NAME = { local: "ComfyUI models/llm (loads inside ComfyUI)", ollama: "Ollama", openai: "OpenAI-compatible server" };
const NO_MODELS = "No models found. Easiest fix: put a vision model folder (for example Qwen3-VL-8B-Instruct from Hugging Face) in ComfyUI/models/llm and reopen Forge. Or install Ollama and run: ollama pull qwen3-vl:8b. Other servers: Settings > DaSiWa > H3 Forge.";

const STORE_KEY = "dasiwa.h3forge";
const openDialogs = new WeakMap();
const briefs = new Map(); // node id -> last brief, for a reroll after closing
const shotTexts = new WeakMap(); // node -> what was typed in each shot box
const HISTORY_KEY = "dasiwaH3ForgeHistory";
const GROUPS_KEY = "dasiwaH3ForgeSubjectGroups";
// A REF2VA picture's label is picked in two steps: what it is, then - for
// characters only - which one, or which ones left to right. The saved value
// is one string ("character-2", "group-21", "place"), which is what the
// server reads.
const PICTURE_KINDS = [["character", "Character"], ["group", "Several characters"], ["place", "Place"], ["style", "Style"], ["first-frame", "First frame"], ["last-frame", "Last frame"], ["pose", "Pose"], ["custom", "Custom"]];
const PICTURE_WHO = {
  character: Array.from({ length: 32 }, (_, i) => [`character-${i + 1}`, `Character ${i + 1}`]),
  group: [["group-12", "1 + 2 (1 on the left)"], ["group-21", "2 + 1 (2 on the left)"], ["group-13", "1 + 3 (1 on the left)"], ["group-31", "3 + 1 (3 on the left)"],
    ["group-23", "2 + 3 (2 on the left)"], ["group-32", "3 + 2 (3 on the left)"], ["group-123", "1 + 2 + 3 (left to right)"]],
};
const pictureKind = label => label.startsWith("character-") ? "character" : label.startsWith("group-") ? "group" : label;
const INSTRUCTIONS_HINT = {
  pose: "Pose only; identity, clothes and background stay unchanged. Add details if needed.",
  custom: "Describe what to use from this image (required).",
};
function forgeHistory(node) {
  const saved = node.properties?.[HISTORY_KEY];
  return Array.isArray(saved) ? saved.filter(entry => entry && typeof entry.simple_prompt === "string" && entry.simple_prompt.trim() && typeof entry.mode === "string" && entry.fields && typeof entry.fields === "object").slice(0, 3) : [];
}
function saveForgeResult(node, result, brief) {
  const entry = {
    mode: result.mode, model: result.model, simple_prompt: result.simple_prompt,
    draftOptions: result.draftOptions, fields: result.fields, continuity: !!result.continuity, contextKey: result.contextKey, forgeInputKey: result.forgeInputKey, structured: result.structured, existing_definitions: result.existing_definitions, reference_snapshot: result.reference_snapshot, brief, createdAt: Date.now(),
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
  briefs.delete(`${node.id}:new`); briefs.delete(`${node.id}:continuity`);
  shotTexts.delete(node);
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
  .ds-forge [hidden]{display:none!important}
  .ds-forge option,.ds-forge optgroup{background:#0d1217;color:#e5eef4}
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
  .ds-forge .ref{display:grid;grid-template-columns:48px 80px minmax(130px,auto) 1fr;gap:8px;align-items:center}
  .ds-forge .pick{display:flex;flex-direction:column;gap:4px}
  .ds-forge .ref-notes textarea{min-height:48px;font-size:12px}
  .ds-forge .ref-notes details{font-size:11px;color:#9fb3c2}
  .ds-forge .ref-notes details input{margin-top:4px}
  .ds-forge .ref{align-items:start}
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

function referencesFor(hook, node) {
  return hook.references?.() || forgeReferences(hook.items(), hook.mode(), node.properties?.[GROUPS_KEY]);
}

async function open(node) {
  const hook = node.__dasiwaH3Forge;
  if (!hook) return;
  try { await hook.prepareReferences?.(); } catch (error) { hook.setStatus(error.message, true); return; }
  openDialogs.get(node)?.();
  installStyles();
  const mode = hook.mode();
  const prefs = remembered();
  const continuity = hook.continuity?.();
  let openedKey = hook.contextKey?.();
  const compatible = entry => entry.mode === hook.mode() && !!entry.continuity === !!hook.continuity?.() && (!entry.contextKey || entry.contextKey === hook.contextKey?.()) && (!entry.forgeInputKey || entry.forgeInputKey === inputKey());

  const overlay = el("div", { className: "ds-forge-overlay" });
  const box = el("div", { className: "ds-forge" });
  overlay.append(box);
  // A run in flight, so Cancel and closing the pop-out can stop it.
  let running = null, closed = false, loadingModels = true, statusTimer = null;
  const cancelRun = () => { clearInterval(statusTimer); statusTimer = null; if (running) { const id = running; running = null; api.fetchApi("/dasiwa/h3/forge/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: id }) }).catch(() => {}); } };
  const close = () => { if (closed) return; closed = true; cancelRun(); overlay.remove(); document.removeEventListener("keydown", onKey); openDialogs.delete(node); };
  openDialogs.set(node, close);
  const onKey = e => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  overlay.addEventListener("pointerdown", e => { if (e.target === overlay) close(); });

  const closeBtn = el("button", { textContent: "×", title: "Close (Esc)", onclick: close });
  box.append(el("h3", {}, el("span", { textContent: `H3 Forge — ${mode}${continuity ? " · Continuity Active" : ""}` }), closeBtn));

  const briefKey = `${node.id}:${continuity ? "continuity" : "new"}`;
  const brief = el("textarea", { placeholder: continuity ? "What happens next? Leave empty to continue naturally." : "What should the clip be? A sentence or two is enough.", value: briefs.get(briefKey) || (!continuity ? forgeHistory(node).find(e => !e.continuity)?.brief : "") || "" });
  brief.addEventListener("input", () => briefs.set(briefKey, brief.value));
  box.append(el("div", { className: "field" }, el("label", { textContent: continuity ? "Next action" : "Idea" }), brief));

  // References from the timeline. REF2VA pictures need a role; base-mode
  // pictures are frames by definition.
  const refs = continuity && (mode !== "REF2VA" || !continuity.use_references) ? [] : referencesFor(hook, node);
  const persistReference = (ref, patch) => {
    if (hook.contextKey?.() !== openedKey) return;
    if (ref.item && hook.updateReference?.(ref.item.id, patch) !== false) {
      Object.assign(ref.item, patch);
      openedKey = hook.contextKey?.();
    }
  };
  let includeReferences = null;
  if (continuity) box.append(el("div", { className: "muted", textContent: "The source ending and Duration guide this draft. A vision model uses tail frames internally; audio is not analyzed. Review the result, then Apply to node." }));
  if (continuity && mode === "REF2VA" && hook.setUseReferences) {
    includeReferences = el("input", { type: "checkbox", checked: continuity.use_references, onchange: e => {
      if (hook.contextKey?.() !== openedKey) { hook.setStatus("Director changed. Reopen Forge.", true); return; }
      briefs.set(briefKey, brief.value); hook.setUseReferences(e.target.checked); void open(node);
    } });
    box.append(el("label", {}, includeReferences, " Include timeline references"));
  }
  // REF2VA pictures: one label each - Character 1-4, several characters in
    // one picture, Place, Style, First or Last frame, Pose, Custom. Pictures
    // with the same Character number are one subject (what subject groups
    // did), and the idea names the labels. For a new draft the node writes
    // who is who and no picture goes to the model; H3 sees them itself.
  // Base-mode pictures are frames by definition and need no label.
  const labelled = mode === "REF2VA" && refs.some(r => r.kind === "image" && r.easy_role);
  if (refs.length) {
    if (!continuity && labelled) brief.placeholder = 'Name the labels: "Character 1 sits on the bed in the place. Character 2 walks in and waves."';
    const list = el("div", { className: "refs" });
    const tags = referenceTags(refs);
    for (const [index, ref] of refs.entries()) {
      const name = tags[index].map(tag => tag.slice(1, -1)).join(" + ");
      const thumb = ref.kind === "image" && ref.path ? el("img", { src: viewUrl(ref.path) }) : el("span", { className: "muted", textContent: ref.saved_reference ? "saved" : ref.kind });
      let roleCell;
      let instructions = null;
      if (ref.kind === "image" && mode === "REF2VA" && ref.item && ref.easy_role) {
        const save = value => {
          ref.easy_role = value;
          const role = labelRole(value);
          ref.role = role.forge_role; ref.subject_group = role.forge_subject_group;
          persistReference(ref, { forge_label: value, ...role });
          if (instructions) instructions.placeholder = INSTRUCTIONS_HINT[value] || "What should this reference contribute? (optional)";
          node.graph?.setDirtyCanvas(true, true);
        };
        const kindSel = el("select", { title: "What this picture is." });
        kindSel.setAttribute("aria-label", `${name} label`);
        for (const [value, label] of PICTURE_KINDS) kindSel.append(el("option", { value, textContent: label, selected: pictureKind(ref.easy_role) === value }));
        const whoSel = el("select", { title: "Which character. Pictures with the same Character number are one character." });
        whoSel.setAttribute("aria-label", `${name} character`);
        const fillWho = () => {
          const kind = pictureKind(ref.easy_role);
          const maxCharacter = Math.max(4, refs.filter(r => r.kind === "image").length, ...refs.map(r => Number(r.easy_role?.match(/^character-(\d+)$/)?.[1]) || 0));
          const choices = kind === "character" ? PICTURE_WHO.character.slice(0, maxCharacter) : PICTURE_WHO[kind];
          whoSel.replaceChildren(...(choices || []).map(([value, label]) => el("option", { value, textContent: label, selected: value === ref.easy_role })));
          whoSel.hidden = !choices;
        };
        // Switching to Character picks a number no other picture uses.
        const freeCharacter = () => {
          const used = new Set(refs.filter(r => r !== ref && r.easy_role?.startsWith("character-")).map(r => r.easy_role));
          return PICTURE_WHO.character.find(([value]) => !used.has(value))?.[0] || "character-1";
        };
        kindSel.onchange = e => {
          const kind = e.target.value;
          save(kind === "character" ? freeCharacter() : PICTURE_WHO[kind]?.[0][0] || kind);
          fillWho();
        };
        whoSel.onchange = e => save(e.target.value);
        fillWho();
        roleCell = el("span", { className: "pick" }, kindSel, whoSel);
      } else {
        roleCell = el("span", { className: "muted", textContent: ref.saved_reference ? "saved reference" : ref.kind === "image" ? BASE_ROLE[mode] || "frame" : ref.kind === "video" ? `motion · ${ref.stream}` : "voice" });
      }
      const notes = el("div", { className: "field ref-notes" });
      instructions = el("textarea", { value: ref.instructions || "", maxLength: 4000, rows: 2, placeholder: INSTRUCTIONS_HINT[ref.easy_role] || (ref.role === "pose" ? INSTRUCTIONS_HINT.pose : "What should this reference contribute? (optional)"), oninput: e => { ref.instructions = e.target.value; persistReference(ref, { forge_instructions: ref.instructions }); } });
      instructions.setAttribute("aria-label", `${name} reference instructions`);
      if (ref.saved_reference) instructions.readOnly = true;
      notes.append(instructions);
      if (!ref.saved_reference) {
        const extra = el("details", {}, el("summary", { textContent: "Keep / ignore (optional)" }));
        for (const [key, label] of [["keep", "Keep"], ["drop", "Ignore"]]) {
          const input = el("input", { type: "text", maxLength: 2000, value: ref[key] || "", placeholder: label, oninput: e => { ref[key] = e.target.value; persistReference(ref, { [`forge_${key}`]: ref[key] }); } });
          input.setAttribute("aria-label", `${name} ${label.toLowerCase()}`);
          extra.append(input);
        }
        notes.append(extra);
      }
      list.append(el("div", { className: "ref" }, thumb, el("span", { textContent: name }), roleCell, notes));
    }
    box.append(el("div", { className: "field" }, el("label", { textContent: "References on the timeline" }), list));
    if (labelled) {
      box.append(el("span", { className: "muted", textContent: continuity
        ? "Pictures with the same Character number are one character."
        : 'Pictures with the same Character number are one character. A picture with two or three of them: pick "Several characters" and who stands where, left to right. In the idea, write "Character 1", "Character 2" and "the place". Image-only labelled drafts need no writer vision; mixed media and saved references use the full REF2VA path.' }));
    }
  } else if (mode !== "T2VA" && !continuity) {
    box.append(el("div", { className: "muted", textContent: `${mode} expects pictures on the timeline; none are loaded, so the model writes from the idea alone.` }));
  }

  const inherited = continuity && mode === "REF2VA" ? hook.existingDefinitions?.() || { text: "", warning: "" } : { text: "", warning: "" };
  // Continuity identities belong to the Director prompt, not a second Forge editor.
  const definitions = { value: inherited.text };
  const structured = el("input", { type: "checkbox", checked: !!continuity && mode === "REF2VA" && (continuity.use_references || !!refPromptFields(hook.currentPrompt?.())) });
  if (continuity && mode === "REF2VA") {
    box.append(el("label", {}, structured, " Structured REF2VA draft"));
    box.append(el("span", { className: "muted", textContent: continuity.use_references ? "Timeline references are included in both Forge and video generation. Existing identities are carried forward; new subjects are defined for review." : "Timeline references are off. Enable Include timeline references above to introduce a character or scene reference; this also enables them for video generation." }));
  }
  const modelSel = el("select");
  const detail = el("input", { type: "range", min: 1, max: 10, step: 1 });
  const detailLabel = el("span", { className: "muted" });
  const creativity = el("select");
  // Auto lets the model choose; a number is an instruction the server checks.
  const shots = el("select", { title: "How many shots. Auto lets the model choose." });
  box.append(el("div", { className: "row" },
    el("div", { className: "field" }, el("label", { textContent: "Model" }), modelSel),
    el("div", { className: "field" }, el("label", { textContent: "Creativity" }), creativity),
    // A continuation is one uninterrupted shot, so it has no Shots choice.
    el("div", { className: "field", hidden: !!continuity }, el("label", { textContent: "Shots" }), shots)));
  // One box per picked shot, as in PromptForge: what is typed joins the idea
  // as "Shot N: ..." lines on the server. None on Auto or in a continuation.
  const shotBox = el("div", { className: "field", hidden: true });
  box.append(shotBox);
  const typedShots = () => { if (!shotTexts.has(node)) shotTexts.set(node, []); return shotTexts.get(node); };
  const shotRows = () => (shotBox.hidden ? [] : Array.from(shotBox.querySelectorAll("textarea"), t => t.value));
  const renderShots = () => {
    const n = Number(shots.value);
    const count = !continuity && Number.isInteger(n) && n > 0 ? n : 0;
    const typed = typedShots();
    shotBox.replaceChildren(...Array.from({ length: count }, (_, i) => {
      const input = el("textarea", { rows: 2, value: typed[i] || "", disabled: loadingModels || !!running,
        placeholder: count === 1 ? "What happens in the shot (optional)" : `What happens in shot ${i + 1} (optional)` });
      input.setAttribute("aria-label", `Shot ${i + 1}`);
      input.addEventListener("input", () => { typed[i] = input.value; clearDraft(); });
      return el("div", { className: "field" }, el("label", { textContent: `Shot ${i + 1}` }), input);
    }));
    shotBox.hidden = !count;
  };
  box.append(el("div", { className: "field" }, el("label", {}, "Detail ", detailLabel), detail));
  const status = el("span", { className: "status" });
  const setStatus = (msg, err = false) => { status.textContent = msg; status.classList.toggle("error", err); };
  const genBtn = el("button", { className: "primary", textContent: "Generate", disabled: true });
  const applyBtn = el("button", { textContent: "Apply to node", disabled: true });
  box.append(el("div", { className: "actions" }, status, genBtn, applyBtn));
  const output = el("pre", { hidden: true });
  box.append(output);
  const historyBox = el("div", { className: "history" });
  box.append(historyBox);
  // Shot boxes join the key only when something is typed, so drafts saved
  // before they existed still match.
  const inputKey = () => {
    const rows = shotRows().map(r => r.trim());
    return JSON.stringify([brief.value.trim(), structured.checked, definitions.value, refs.map(({ item, ...r }) => r), ...(rows.some(Boolean) ? [rows] : [])]);
  };
  const referenceControls = Array.from(box.querySelectorAll(".refs input, .refs select, .refs textarea"));
  const controls = [brief, modelSel, detail, creativity, shots, structured, ...referenceControls];
  const setControlsDisabled = disabled => {
    controls.forEach(c => { c.disabled = disabled; });
    shotBox.querySelectorAll("textarea").forEach(c => { c.disabled = disabled; });
    if (includeReferences) includeReferences.disabled = !!running;
  };
  setControlsDisabled(true);
  let result = null;
  const showResult = entry => {
    if (closed) return;
    brief.value = entry.brief || "";
    if (typeof entry.structured === "boolean") structured.checked = entry.structured;
    // Saved drafts are previews, not a source of identities for the next request.
    if (entry.draftOptions) {
      const { model, detail: level, creativity: preset, shots: count } = entry.draftOptions;
      if (Array.from(modelSel.options).some(o => o.value === model)) modelSel.value = model;
      detail.value = level; creativity.value = preset;
      // Drafts saved before the Shots control have none: they were Auto.
      const shotsValue = String(count ?? "Auto");
      if (Array.from(shots.options).some(o => o.value === shotsValue)) shots.value = shotsValue;
      shotTexts.set(node, Array.isArray(entry.draftOptions.shot_briefs) ? [...entry.draftOptions.shot_briefs] : []);
      renderShots();
      if (detail.oninput) detail.oninput();
    }
    result = entry;
    output.hidden = false;
    output.textContent = entry.simple_prompt;
    applyBtn.disabled = loadingModels || !!running || !compatible(entry);
    if (!compatible(entry)) setStatus("Draft belongs to a different source, duration, model or prompt. Generate again for the current context.", true);
    renderHistory();
  };
  const renderHistory = () => {
    const entries = forgeHistory(node);
    const clear = el("button", { type: "button", textContent: "Clear history", disabled: loadingModels || !!running || !entries.length, title: "Remove the three saved Forge prompts from this node" });
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
      const button = el("button", { type: "button", disabled: loadingModels || !!running, className: result === entry ? "selected" : "", title: "Show this prompt; Apply to node to use it" },
        el("span", { textContent: `${index + 1}. ${entry.mode}${entry.continuity ? " · Continuity" : ""} · ${entry.model || "model"} · ${date}` }),
        el("span", { className: "excerpt", textContent: entry.simple_prompt.replace(/\s+/g, " ").slice(0, 150) }));
      button.onclick = () => showResult(entry);
      historyBox.append(button);
    });
  };
  const savedShots = !shotTexts.has(node) && forgeHistory(node).find(entry => entry.mode === mode && !!entry.continuity === !!continuity && (!entry.contextKey || entry.contextKey === openedKey) && entry.brief === brief.value.trim());
  if (savedShots?.draftOptions?.shot_briefs) shotTexts.set(node, [...savedShots.draftOptions.shot_briefs]);
  renderHistory();
  document.body.append(overlay);
  brief.focus();

  const notes = el("div", { className: "muted" });
  box.insertBefore(notes, status.parentElement);
  let levels = {};
  setStatus("Loading Forge models…");
  try {
    const res = await api.fetchApi("/dasiwa/h3/forge/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ settings: forgeSettings() }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || res.statusText);
    if (closed) return;
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
    const counts = (data.shot_counts || ["Auto"]).map(String);
    for (const c of counts) shots.append(el("option", { value: c, textContent: c }));
    const preferredShots = savedShots?.draftOptions?.shots ?? prefs.shots;
    shots.value = preferredShots && counts.includes(String(preferredShots)) ? String(preferredShots) : String(data.default_shots || "Auto");
    renderShots();
    genBtn.disabled = false;
    setStatus("Ready. Generate a draft, then review and Apply.");
  } catch (err) {
    setStatus(err.message, true);
    genBtn.disabled = true;
  }
  if (closed) return;
  loadingModels = false;
  setControlsDisabled(false);
  brief.focus();
  const syncDetail = () => { detailLabel.textContent = `${detail.value} of 10 — ${levels[detail.value] || ""}`; };
  detail.oninput = syncDetail; syncDetail();
  const latest = forgeHistory(node).find(compatible);
  if (latest) showResult(latest); else renderHistory();
  const clearDraft = () => {
    if (running || closed) return;
    result = null; applyBtn.disabled = true; output.hidden = true; output.textContent = "";
    setStatus("Idea or options changed. Generate a new draft, or choose a saved draft.");
    renderHistory();
  };
  brief.addEventListener("input", clearDraft);
  modelSel.addEventListener("change", clearDraft);
  creativity.addEventListener("change", clearDraft);
  shots.addEventListener("change", () => { renderShots(); clearDraft(); });
  detail.addEventListener("input", clearDraft);
  structured.addEventListener("change", clearDraft);
  referenceControls.forEach(c => c.addEventListener(c.tagName === "SELECT" ? "change" : "input", clearDraft));
  genBtn.onclick = async () => {
    if (closed) return;
    if (running) { cancelRun(); genBtn.disabled = true; setStatus("Cancelling… the model stops at its next token, then unloads."); return; }
    const text = brief.value.trim();
    if (openedKey !== hook.contextKey?.()) { setStatus("Director context changed. Close and reopen Forge.", true); return; }
    const rows = shotRows();
    if (!text && !continuity && !rows.some(r => r.trim())) { setStatus("Write the idea first.", true); return; }
    const missing = refs.find(r => r.role === "custom" && !r.instructions.trim());
    if (missing) { setStatus("Custom reference: describe what this image should contribute, or choose a preset role.", true); return; }
    briefs.set(briefKey, text);
    remember({ model: modelSel.value, creativity: creativity.value, detail: Number(detail.value), shots: shots.value });
    result = null; output.hidden = true; output.textContent = ""; renderHistory();
    applyBtn.disabled = true;
    const draftOptions = { model: modelSel.value, detail: Number(detail.value), creativity: creativity.value, shots: shots.value, shot_briefs: rows };
    const requestKey = openedKey, forgeInputKey = inputKey();
    const requestId = `forge-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    running = requestId; setControlsDisabled(true); renderHistory();
    genBtn.textContent = "Cancel";
    const started = Date.now();
    statusTimer = setInterval(() => setStatus(`Writing with ${modelSel.selectedOptions[0]?.textContent || modelSel.value}… ${Math.round((Date.now() - started) / 1000)}s (Cancel stops it; the model unloads either way)`), 500);
    try {
      const res = await api.fetchApi("/dasiwa/h3/forge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: requestId, brief: text, mode, duration: hook.duration(), model: modelSel.value,
          output_canvas: hook.outputCanvas?.() ?? null,
          detail: Number(detail.value), creativity: creativity.value, shots: shots.value, shot_briefs: rows,
          references: refs.map(({ item, ...r }) => r), settings: forgeSettings(), continuity, easy: !continuity && labelled,
          structured: !!continuity && mode === "REF2VA" && structured.checked, existing_definitions: definitions.value.trim(),
        }),
      });
      const data = await res.json();
      if (!res.ok) { output.hidden = !data.raw; output.textContent = data.raw || ""; throw new Error(data.message || res.statusText); }
      if (closed || running !== requestId) throw new Error("Draft cancelled; no prompt was changed.");
      if (requestKey !== hook.contextKey?.() || forgeInputKey !== inputKey()) throw new Error("Source, duration, model or prompt changed during drafting. Reopen Forge and generate again.");
      if (!!data.continuity !== !!continuity || (continuity && data.source_id !== continuity.clip_id)) throw new Error("Draft does not match the selected continuity source.");
      data.contextKey = requestKey; data.draftOptions = draftOptions;
      data.forgeInputKey = forgeInputKey; data.existing_definitions = definitions.value; data.reference_snapshot = referenceSnapshot(refs);
      const saved = saveForgeResult(node, data, text);
      showResult(saved);
      const seen = data.easy ? " · picture labels used (no images sent to the writer)" : data.saw_images ? ` · looked at ${data.saw_images} picture${data.saw_images === 1 ? "" : "s"}` : continuity ? " · text context only (no tail images)" : refs.some(r => r.kind === "image") && data.vision === false ? " · this model cannot see images, so it wrote from your idea only" : "";
      const warned = [...(data.warnings || []), ...(data.unloaded ? [] : ["WARNING: model still loaded"])];
      setStatus(`Done in ${data.stats.seconds}s${data.stats.output_tokens ? ` · ${data.stats.output_tokens} tokens` : ""}${seen}${warned.length ? " · " + warned.join(" · ") : " · model unloaded"}`, warned.length > 0);
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      clearInterval(statusTimer); statusTimer = null;
      running = null;
      setControlsDisabled(false);
      applyBtn.disabled = !result || !compatible(result);
      genBtn.disabled = false;
      genBtn.textContent = "Regenerate";
      renderHistory();
    }
  };
  applyBtn.onclick = () => {
    if (!result) return;
    if (!compatible(result) || hook.apply(result) === false) { setStatus("Draft is stale. Generate again for the current source, duration and prompt.", true); return; }
    hook.setStatus(`Forge prompt applied (${result.model}).`);
    close();
  };
}

// At load, not on first open: the toolbar button's style lives here too.
installStyles();
window.DaSiWaH3Forge = { open, close: node => openDialogs.get(node)?.(), clearHistory: clearForgeHistory, settings: forgeSettings };
