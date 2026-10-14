// MiniMax H3 Forge: the overlay behind the Director's "Forge" button.
//
// Write an idea, pick a local model, get a prompt in the Director's own
// fields. Runs through /dasiwa/h3/forge (nodes/h3_forge.py), outside the
// ComfyUI queue: the LLM writes, unloads, and only then is the workflow run.
// The Director exposes node.__dasiwaH3Forge for reading the timeline and
// writing the result back.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { forgeReferences, labelRole, refPromptFields, referenceSnapshot, referenceTags, whoFromBoxes } from "./minimax_h3_forge_state.js";

// Server addresses live in ComfyUI Settings, never in the workflow, so a
// downloaded workflow cannot point this machine at a server of its choosing.
// Shared by the Director's Forge and the LLM nodes (nodes/llm_backends.py
// reads the same IDs from the settings file); the IDs predate the sharing.
const SETTING_OLLAMA = "DaSiWa.H3Forge.OllamaURL";
const SETTING_OPENAI = "DaSiWa.H3Forge.OpenAIURL";
const SETTING_OPENAI_KEY = "DaSiWa.H3Forge.OpenAIKey";
const SECTION = "LLM servers";
const SHARED = " Used by the H3 Director's Forge and the DaSiWa LLM nodes; press R after changing it to refresh their model lists.";
app.registerExtension({
  name: "DaSiWa.H3Forge",
  settings: [
    { id: SETTING_OLLAMA, category: ["DaSiWa", SECTION, "Ollama address"], name: "Ollama address", type: "text", defaultValue: "", tooltip: "Leave empty for Ollama on this computer (http://127.0.0.1:11434). Set it to use Ollama on another machine." + SHARED },
    { id: SETTING_OPENAI, category: ["DaSiWa", SECTION, "OpenAI-compatible server"], name: "OpenAI-compatible server address", type: "text", defaultValue: "", tooltip: "Optional: a llama.cpp server, llama-swap, LM Studio or koboldcpp, e.g. http://127.0.0.1:8080. Empty = off." + SHARED },
    { id: SETTING_OPENAI_KEY, category: ["DaSiWa", SECTION, "OpenAI-compatible API key"], name: "OpenAI-compatible API key", type: "text", defaultValue: "", tooltip: "Only if that server asks for one (llama-server --api-key, llama-swap apiKeys, LM Studio with authentication). Sent only to the address above. Stored in ComfyUI's settings file like every other setting." },
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
// A REF2VA picture's label: what it is, as buttons that combine (Character,
// Place, Style and the frames; Pose and Custom stand alone), and who is in it,
// numbers tapped in order. The saved forge_label is still one string
// ("character-2", "group-21", "place"), the first kind, which older readers
// understand; forge_kinds / forge_who / forge_who_axis carry the rest only
// when that one string cannot say it.
const KIND_BUTTONS = [["character", "Character"], ["place", "Place"], ["style", "Style"], ["first-frame", "First frame"], ["last-frame", "Last frame"], ["pose", "Pose"], ["custom", "Custom"]];
const KIND_ORDER = KIND_BUTTONS.map(([k]) => k);
const ALONE_KINDS = new Set(["pose", "custom"]);
const PICTURE_WHO = {
  character: Array.from({ length: 32 }, (_, i) => [`character-${i + 1}`, `Character ${i + 1}`]),
  group: [["group-12", "1 + 2 (1 on the left)"], ["group-21", "2 + 1 (2 on the left)"], ["group-13", "1 + 3 (1 on the left)"], ["group-31", "3 + 1 (3 on the left)"],
    ["group-23", "2 + 3 (2 on the left)"], ["group-32", "3 + 2 (3 on the left)"], ["group-123", "1 + 2 + 3 (left to right)"]],
};
// Keep aligned with nodes/h3_prompting.py EASY_ROLES.
const EASY_ROLES = new Set([...PICTURE_WHO.character.map(([role]) => role), ...PICTURE_WHO.group.map(([role]) => role), "place", "style", "first-frame", "last-frame", "pose", "custom"]);
const pictureKind = (label = "") => label.startsWith("character-") || label.startsWith("group-") ? "character" : label;
const GROUP_LABELS = new Set(PICTURE_WHO.group.map(([value]) => value));
// A reference's kinds and people: the buttons' own values, else its label.
function kindsOf(ref) {
  return Array.isArray(ref.picture_kinds) && ref.picture_kinds.length ? ref.picture_kinds : [pictureKind(ref.easy_role || "character-1")];
}
function whoOf(ref) {
  if (Array.isArray(ref.who)) return ref.who;
  const label = ref.easy_role || "";
  return label.startsWith("character-") ? [Number(label.slice(10))] : label.startsWith("group-") ? [...label.slice(6)].map(Number) : [];
}
// The single label for the first kind: a character's number, or a listed
// group when the order is left to right.
function primaryLabel(kinds, who, axis) {
  if (kinds[0] !== "character") return kinds[0];
  const group = `group-${who.join("")}`;
  return who.length > 1 && axis === "x" && GROUP_LABELS.has(group) ? group : `character-${who[0] || 1}`;
}
// True when that label alone says everything the buttons do.
function describesAlone(label, kinds, who, axis) {
  if (kinds.length !== 1) return false;
  if (kinds[0] === "character") return who.length === 1 ? label === `character-${who[0]}` : axis === "x" && label === `group-${who.join("")}`;
  return !who.length;
}
// The order of several people is optional (no order picked: the numbers say
// who is in it, not where). In the panel "x" / "y" / "z" is the order picked
// and "" is none. Stored, nothing is still left to right, so pictures saved
// before keep their meaning, and no order is "none".
const axisFrom = (stored, who) => stored === "none" ? "" : stored === "y" || stored === "z" ? stored : who.length > 1 ? "x" : "";
const axisTo = (axis, who) => who.length < 2 || axis === "x" ? "" : axis || "none";
const AXIS_BUTTONS = [["x", "left → right"], ["y", "top → bottom"], ["z", "front → back"]];
const AXIS_TITLE = "Optional: where each one stands, in the order tapped. Tap the picked order again to turn it off.";
const WHO_TITLE = "Who is in it. Pictures with the same Character number are one character. With two or more, the order below can say where each one stands.";
// Who buttons: six to start, one more each time the highest is in use, up to
// the 32 characters the node allows, so the row stays short until it is needed.
const WHO_SHOWN = 6, WHO_MAX = 32;
const whoShown = (pictures, used) => Math.max(Math.min(WHO_MAX, Math.max(WHO_SHOWN, pictures, Math.max(0, ...used) + 1)), ...used);
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
  .ds-forge{width:min(1520px,96vw);max-height:92vh;overflow:auto;background:#111820;color:#e5eef4;border:1px solid #40515e;border-radius:8px;padding:14px;font:13px system-ui,sans-serif;display:flex;flex-direction:column;gap:10px;box-shadow:0 10px 40px rgba(0,0,0,.6)}
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
  .ds-forge .ref{display:grid;grid-template-columns:48px 70px minmax(190px,250px) 1fr;gap:8px;align-items:center}
  .ds-forge .pick{display:flex;flex-direction:column;gap:5px}
  .ds-forge .kinds,.ds-forge .who,.ds-forge .who-axis{display:flex;flex-wrap:wrap;gap:3px;align-items:center}
  .ds-forge .kinds button,.ds-forge .who button,.ds-forge .who-axis button{padding:2px 8px;font-size:11px;border-radius:999px;position:relative}
  .ds-forge .kinds button[aria-pressed=true],.ds-forge .who button[aria-pressed=true],.ds-forge .who-axis button[aria-pressed=true]{background:rgba(151,91,255,.3);border-color:rgba(177,128,255,.8);color:#fff}
  .ds-forge .who button[data-order]:not([data-order=""])::after{content:attr(data-order);position:absolute;top:-6px;right:-5px;font-size:9px;line-height:12px;min-width:12px;border-radius:999px;background:#b180ff;color:#0d1217;text-align:center}
  .ds-forge .who-label{font-size:10px;color:#8fa3b2;margin-right:2px}
  .ds-forge .ref-notes textarea{min-height:48px;font-size:12px}
  .ds-forge .ref-notes details{font-size:11px;color:#9fb3c2}
  .ds-forge .ref-notes details input{margin-top:4px}
  .ds-forge .ref{align-items:start}
  .ds-forge .ref img{width:48px;height:36px;object-fit:cover;border-radius:3px;background:#090d11}
  .ds-forge pre{white-space:pre-wrap;background:#0b1015;border:1px solid #344452;border-radius:4px;padding:8px;margin:0;min-height:140px;height:240px;flex-shrink:0;resize:vertical;overflow:auto;font:12px/1.45 ui-monospace,monospace}
  .ds-forge .history{display:flex;flex-direction:column;gap:5px;border-top:1px solid #344452;padding-top:9px}
  .ds-forge .history-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
  .ds-forge .history-head button{padding:3px 8px;font-size:11px}
  .ds-forge .history button{text-align:left;display:flex;flex-direction:column;gap:3px;min-width:0}
  .ds-forge .history button.selected{border-color:#b180ff;background:rgba(151,91,255,.18)}
  .ds-forge .history .excerpt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#9fb3c2;font-size:11px}
  .ds-forge .people{grid-column:1/-1;font-size:12px}
  .ds-forge .people summary{cursor:pointer;color:#9fb3c2;font-size:11px}
  .ds-forge .people-tools{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
  .ds-forge .people-tools button{padding:3px 9px;font-size:11px}
  .ds-forge .people-tools .muted{flex:1 1 240px}
  .ds-forge .people-stage{position:relative;display:inline-block;line-height:0;user-select:none;touch-action:none;max-width:100%}
  .ds-forge .people-stage img{display:block;max-width:100%;max-height:360px;width:auto;height:auto;object-fit:contain;border-radius:3px}
  .ds-forge .people-box{position:absolute;box-sizing:border-box;border:2px solid #b180ff;background:rgba(151,91,255,.10);cursor:move;line-height:normal}
  .ds-forge .people-box.active{border-color:#f3c67a;z-index:2}
  .ds-forge .people-tag{position:absolute;top:-1px;left:-1px;max-width:max-content;background:#b180ff;color:#0d1217;font-size:10px;font-weight:600;padding:1px 4px;border-radius:0 0 3px 0;white-space:nowrap}
  .ds-forge .people-box.active .people-tag{background:#f3c67a}
  .ds-forge .people-tag select{position:absolute;inset:0;opacity:0;width:100%;height:100%;padding:0;cursor:pointer}
  .ds-forge .cast{display:flex;flex-direction:column;gap:6px}
  .ds-forge .cast-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:6px}
  .ds-forge .cast-card{display:grid;grid-template-columns:72px 1fr;gap:8px;align-items:start;border:1px solid #344452;border-radius:6px;padding:6px;transition:box-shadow .3s}
  .ds-forge .cast-card.flash{box-shadow:0 0 0 2px #f3c67a}
  .ds-forge .cast-card[data-state=error]{border-color:#ff8a8a}
  .ds-forge .cast-card[data-state=stale]{border-color:#f3c67a}
  .ds-forge .cast-thumb{width:72px;height:96px;display:flex;align-items:center;justify-content:center;background:#090d11;border-radius:4px;overflow:hidden}
  .ds-forge .cast-thumb img{max-width:100%;max-height:100%}
  .ds-forge .cast-body{display:flex;flex-direction:column;gap:4px;min-width:0}
  .ds-forge .cast-top{display:flex;gap:8px;align-items:baseline}
  .ds-forge .cast-name{font-weight:600}
  .ds-forge .cast-status{color:#8fa3b2;font-size:11px;flex:1}
  .ds-forge .cast-card[data-state=error] .cast-status{color:#ff8a8a}
  .ds-forge .cast-card[data-state=stale] .cast-status{color:#f3c67a}
  .ds-forge .cast-card textarea{min-height:64px;font-size:12px}
  .ds-forge .cast-count{color:#8fa3b2;font-size:10px;text-align:right}
  .ds-forge .people-resize{position:absolute;right:-5px;bottom:-5px;width:12px;height:12px;background:#b180ff;border-radius:2px;cursor:nwse-resize}
  `;
  document.head.append(style);
}

const el = (tag, props = {}, ...children) => { const n = Object.assign(document.createElement(tag), props); n.append(...children); return n; };
const viewUrl = path => api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`);
const BASE_ROLE = { I2VA: "first frame", FL2VA: "first / last frame", L2VA: "last frame" };

function referencesFor(hook, node) {
  return hook.references?.() || forgeReferences(hook.items(), hook.mode(), node.properties?.[GROUPS_KEY]);
}

function savedContextMatches(saved, current) {
  if (!saved || saved === current) return true;
  try {
    const withoutPrompt = key => {
      const context = JSON.parse(key);
      if (!Array.isArray(context) || context.length !== 12) return null;
      context[3] = null; context[11] = null;
      if (context[2]) { const { current_prompt, ...source } = context[2]; context[2] = source; }
      return JSON.stringify(context);
    };
    const previous = withoutPrompt(saved);
    return previous !== null && previous === withoutPrompt(current);
  } catch { return false; }
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
  const compatible = entry => entry.mode === hook.mode() && !!entry.continuity === !!hook.continuity?.() && (!entry.contextKey || entry.contextKey === hook.contextKey?.()) && (!entry.forgeInputKey || entry.forgeInputKey === inputKey()) && !!entry.draftOptions?.see_pictures === effectiveSeePictures();

  const overlay = el("div", { className: "ds-forge-overlay" });
  const box = el("div", { className: "ds-forge" });
  overlay.append(box);
  // A run in flight, so Cancel and closing the pop-out can stop it.
  let running = null, closed = false, loadingModels = true, statusTimer = null, draftWaiting = false;
  const cancelRun = () => { clearInterval(statusTimer); statusTimer = null; if (running) { const id = running; running = null; api.fetchApi("/dasiwa/h3/forge/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: id }) }).catch(() => {}); } };
  // Set once a Find people or Cast read has run: closing then unloads the model they kept warm.
  let warmed = false, castRequest = null, peopleRequest = null;
  const close = () => {
    if (closed) return; closed = true; cancelRun();
    if (castRequest) api.fetchApi("/dasiwa/h3/forge/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: castRequest }) }).catch(() => {});
    if (peopleRequest) api.fetchApi("/dasiwa/h3/forge/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: peopleRequest }) }).catch(() => {});
    if (warmed) api.fetchApi("/dasiwa/h3/forge/release", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }).catch(() => {});
    overlay.remove(); document.removeEventListener("keydown", onKey); openDialogs.delete(node);
  };
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
      for (const [key, value] of Object.entries(patch)) {
        if (["forge_kinds", "forge_who", "forge_who_axis", "forge_boxes"].includes(key) && value === "") delete ref.item[key];
        else ref.item[key] = value;
      }
      openedKey = hook.contextKey?.();
    }
  };
  // Set once the controls exist: a change to a picture's boxes drops the draft.
  let refsChanged = () => {};

  // ── People boxes (PromptForge's, 13 Oct 2026; nodes/h3_people.py) ──
  // A box round each character on the picture itself. Find people asks the
  // model picked below to box everyone; Add character drops one to drag over
  // someone. Either way every box moves, resizes from its corner and
  // renumbers, so a wrong guess is one drag to fix. With boxes on a picture,
  // they say who is in it and where each one stands, left to right, so the
  // Who buttons and the order step aside.
  const BOX_MIN = 0.04;
  const clamp01 = v => Math.min(1, Math.max(0, v));
  const usedNumbers = () => refs.flatMap(r => r.kind === "image" ? (Array.isArray(r.boxes) && r.boxes.length ? r.boxes.map(b => b.n) : whoOf(r)) : []);
  function peopleEditor(ref, name, onBoxes) {
    let boxes = Array.isArray(ref.boxes) ? ref.boxes.map(b => ({ n: b.n, box: [...b.box] })) : [];
    let active = null;
    const summary = el("summary");
    const wrap = el("details", { className: "people" }, summary);
    // Open from the start, as PromptForge shows the picture with its boxes:
    // folded away, Find people and the boxes were easy to miss.
    wrap.open = true;
    const note = el("span", { className: "muted" });
    const findBtn = el("button", { type: "button", textContent: "Find people", title: "The model picked below boxes everyone in this picture. It needs to see pictures and answer with boxes (Qwen-VL models do)." });
    const addBtn = el("button", { type: "button", textContent: "Add character", title: "Drop a box to drag over someone." });
    const clearBtn = el("button", { type: "button", textContent: "Clear boxes" });
    // Read everyone in this picture again from the boxes as they are; Find
    // people boxes the picture afresh (PromptForge, 14 Oct 2026).
    const scanBtn = el("button", { type: "button", textContent: "Scan people", title: "Read everyone in this picture again for their Cast cards, from their boxes as they are. Cards you edited are read again too." });
    const img = el("img", { src: viewUrl(ref.path), alt: "", draggable: false });
    const stage = el("div", { className: "people-stage" }, img);
    wrap.append(el("div", { className: "people-tools" }, findBtn, addBtn, clearBtn, scanBtn, note), stage);
    const busy = () => running || loadingModels || peopleRequest;
    const commit = next => {
      boxes = next.map(b => ({ n: b.n, box: b.box.map(v => Number(clamp01(v).toFixed(4))) }));
      if (boxes.length) ref.boxes = boxes; else delete ref.boxes;
      persistReference(ref, { forge_boxes: boxes.length ? JSON.stringify(boxes) : "" });
      onBoxes(boxes);
      paint();
      refsChanged();
    };
    const place = (node, [x1, y1, x2, y2]) => Object.assign(node.style, { left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` });
    function boxEl(b, offered, full) {
      const node = el("div", { className: `people-box${b.n === active ? " active" : ""}` });
      node.dataset.n = String(b.n);
      place(node, b.box);
      // A plain tag sized to its text with the picker laid over it unseen: a
      // <select> is as wide as its longest option, and ran past narrow boxes.
      const tag = el("span", { className: "people-tag", textContent: full ? `Character ${b.n}` : String(b.n) });
      const pick = el("select", { title: `Character ${b.n}: pick another number, or remove the box` });
      pick.setAttribute("aria-label", `${name}: which character this box is (Character ${b.n})`);
      for (let n = 1; n <= Math.max(offered, b.n); n += 1) pick.append(el("option", { value: String(n), textContent: `Character ${n}` }));
      pick.append(el("option", { value: "remove", textContent: "Remove box" }));
      pick.value = String(b.n);
      pick.addEventListener("pointerdown", e => e.stopPropagation());
      pick.addEventListener("change", () => {
        if (busy()) { pick.value = String(b.n); return; }
        if (pick.value === "remove") return commit(boxes.filter(x => x.n !== b.n));
        // Picking a number another box has swaps the two.
        const to = Number(pick.value);
        active = to;
        swapCards(ref, b.n, to);
        commit(boxes.map(x => x.n === b.n ? { ...x, n: to } : x.n === to ? { ...x, n: b.n } : x));
      });
      tag.append(pick);
      const handle = el("span", { className: "people-resize" });
      node.append(tag, handle);
      node.addEventListener("pointerdown", e => drag(e, node, b, e.target === handle ? "resize" : "move"));
      return node;
    }
    // Move a box, or resize it from its corner; saved when the pointer lets go.
    function drag(e, node, b, how) {
      if (e.button !== 0 || busy()) return;
      e.preventDefault();
      active = b.n;
      for (const x of stage.querySelectorAll(".people-box")) x.classList.toggle("active", Number(x.dataset.n) === active);
      const rect = stage.getBoundingClientRect();
      const [x1, y1, x2, y2] = b.box, sx = e.clientX, sy = e.clientY;
      let next = b.box;
      node.setPointerCapture(e.pointerId);
      const move = ev => {
        const dx = (ev.clientX - sx) / rect.width, dy = (ev.clientY - sy) / rect.height;
        if (how === "move") {
          const ox = Math.min(1 - x2, Math.max(-x1, dx)), oy = Math.min(1 - y2, Math.max(-y1, dy));
          next = [x1 + ox, y1 + oy, x2 + ox, y2 + oy];
        } else next = [x1, y1, Math.max(x1 + BOX_MIN, clamp01(x2 + dx)), Math.max(y1 + BOX_MIN, clamp01(y2 + dy))];
        place(node, next);
      };
      const up = () => {
        node.removeEventListener("pointermove", move); node.removeEventListener("pointerup", up); node.removeEventListener("pointercancel", up);
        if (next !== b.box) commit(boxes.map(x => x.n === b.n ? { ...x, box: next } : x));
        else focusCard(b.n);
      };
      node.addEventListener("pointermove", move); node.addEventListener("pointerup", up); node.addEventListener("pointercancel", up);
    }
    function paint() {
      summary.textContent = boxes.length ? `People boxes (${boxes.length}): who is in it, left to right` : "People boxes: box each character instead of tapping numbers";
      clearBtn.hidden = !boxes.length;
      scanBtn.hidden = !castOn || !peopleIn(ref).length;
      for (const x of stage.querySelectorAll(".people-box")) x.remove();
      if (!boxes.some(b => b.n === active)) active = null;
      const offered = whoShown(refs.filter(r => r.kind === "image").length, usedNumbers());
      const width = img.getBoundingClientRect().width || 0;
      // Under 84 px wide a box shows its number alone, so neighbours' labels do not collide.
      for (const b of boxes) stage.append(boxEl(b, offered, !width || width * (b.box[2] - b.box[0]) >= 84));
    }
    img.addEventListener("load", paint);
    wrap.addEventListener("toggle", paint);
    addBtn.onclick = () => {
      if (busy()) return;
      const taken = new Set(boxes.map(b => b.n));
      let n = 1; while (taken.has(n) && n < WHO_MAX) n += 1;
      if (taken.has(n)) return;
      const off = (boxes.length % 5) * 0.06;
      active = n;
      commit([...boxes, { n, box: [0.3 + off, 0.15, 0.6 + off, 0.95].map(clamp01) }]);
    };
    clearBtn.onclick = () => { if (!busy()) { active = null; commit([]); } };
    findBtn.onclick = async () => {
      if (busy()) return;
      const model = modelSel.value;
      if (!model) { note.textContent = "Pick a model first."; return; }
      const snapshot = () => JSON.stringify([ref.path, kindsOf(ref), boxes, ref.instructions, ref.keep, ref.drop, hook.contextKey?.()]);
      const before = snapshot();
      const requestId = `people-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
      peopleRequest = requestId;
      const cardBefore = JSON.stringify(savedCards());
      findBtn.disabled = true;
      // The server runs one model call at a time: wait for a card being read,
      // and hold the card queue until the boxes are back.
      if (castBusy) note.textContent = "Waiting for the Cast card being read…";
      await castQuiet(false);
      note.textContent = `Looking for people with ${modelSel.selectedOptions[0]?.textContent || model}…`;
      try {
        if (closed) return;
        warmed = true;
        const res = await api.fetchApi("/dasiwa/h3/forge/people", { timeoutMs: null, method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: requestId, model, settings: forgeSettings(), path: ref.path, keep_warm: castOn }) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || res.statusText);
        if (closed || snapshot() !== before || modelSel.value !== model) return;
        if (!data.people.length) { note.textContent = "Nobody was found: this model may not give boxes. Use Add character to box them yourself."; return; }
        // The numbers already on this picture carry over, left to right, when
        // the count matches; otherwise Character 1, 2, 3... left to right.
        const had = boxes.length ? whoFromBoxes(boxes) : whoOf(ref);
        const numbers = had.length === data.people.length ? had : data.people.map((_, i) => i + 1);
        active = null;
        warmed = true;
        // Find people again is how a card is read again: everyone it boxes is
        // read fresh, cards the person edited included (PromptForge, 14 Oct 2026).
        if (JSON.stringify(savedCards()) === cardBefore) for (const n of numbers) { const l = liveOf(n); l.force = true; l.failedSig = null; }
        commit(data.people.map((p, i) => ({ n: numbers[i], box: p.box })));
        note.textContent = `Found ${data.people.length} ${data.people.length === 1 ? "person" : "people"} in ${data.seconds}s${data.unloaded ? "" : " (model still loaded)"}. Drag, resize or renumber any box that is off.`;
      } catch (err) {
        note.textContent = err.message;
      } finally {
        if (peopleRequest === requestId) peopleRequest = null;
        findBtn.disabled = false;
        castResume();
      }
    };
    scanBtn.onclick = () => {
      if (busy()) return;
      for (const n of peopleIn(ref)) { const l = liveOf(n); l.force = true; l.failedSig = null; }
      renderCast();
      clearTimeout(castTimer);
      runCastQueue();
    };
    paint();
    return { element: wrap, boxes: () => boxes, clear: () => { if (boxes.length) commit([]); } };
  }

  let includeReferences = null;
  // ── The Cast list (PromptForge's, 13-14 Oct 2026; nodes/h3_cast.py) ──
  // One card per character in the pictures: a thumbnail from their box, and
  // what they look like, read from all their pictures and editable. The text
  // is used at Generate as written; an empty card is drafted as before. A card
  // the person never touched is read again when their pictures or box change;
  // one they edited is only marked out of date, never overwritten. Saved with
  // the node (properties.dasiwaH3ForgeCast), so a shared workflow carries them.
  const CAST_KEY = "dasiwaH3ForgeCast", CAST_MAX = 600;
  const castOn = !continuity && refs.length > 0 && (mode === "REF2VA" ? refs.some(r => r.kind === "image" && r.easy_role) : !!BASE_ROLE[mode]);
  const live = new Map();
  let castTimer = null, castBusy = false, castHeld = 0, castNoVision = false;
  // Find people and Generate wait for a card being read and hold the queue
  // while they run: the server does one model call at a time, and refused a
  // Find people pressed during a read ("Another Forge analysis is running").
  async function castQuiet(waitPeople = true) {
    castHeld += 1;
    clearTimeout(castTimer);
    while ((castBusy || (waitPeople && peopleRequest)) && !closed) await new Promise(r => setTimeout(r, 200));
  }
  function castResume() {
    castHeld = Math.max(0, castHeld - 1);
    if (!castHeld) renderCast();
  }
  const castEls = new Map();
  const castCards = el("div", { className: "cast-cards" });
  const autoCast = el("input", { type: "checkbox", checked: prefs.auto_cast !== false });
  const castNotice = el("span", { className: "muted" });
  autoCast.addEventListener("change", () => { remember({ auto_cast: autoCast.checked }); renderCast(); });
  const castWrap = el("div", { className: "field cast", hidden: true }, el("label", { textContent: "Cast and scene" }),
    el("span", { className: "muted", textContent: "One card per character, and one for the place, the style and each pose, read from their pictures (a character from their box, where they have one) with the model picked below. Edit a card and it is used as written. An empty card is drafted as before." }),
    el("label", { title: "Automatically read empty or unchanged cards with the selected vision model. This sends pictures independently of the draft's vision checkbox. Turn off to use manual scans or descriptions only." }, autoCast, " Read Cast automatically"), castNotice,
    castCards);
  // Card keys: a character's number, or "place", "style" and "pose:<item id>"
  // (a pose belongs to its picture, which keeps its id when pictures move).
  const SCENE_LABEL = { place: "The place", style: "The style" };
  const imagesOf = () => refs.filter(r => r.kind === "image");
  const pictureNumber = r => imagesOf().indexOf(r) + 1;
  const isScene = key => typeof key === "string";
  function savedCards() { const c = node.properties?.[CAST_KEY]; return c && typeof c === "object" && !Array.isArray(c) ? c : {}; }
  function cardOf(key) { const c = savedCards()[key]; return { text: typeof c?.text === "string" ? c.text.slice(0, CAST_MAX) : "", sig: typeof c?.sig === "string" ? c.sig : null, edited: !!c?.edited }; }
  function saveCard(key, patch) {
    const next = { ...cardOf(key), ...patch };
    const all = { ...savedCards() };
    if (!next.text && !next.edited) delete all[key]; else all[key] = next;
    node.properties ||= {};
    if (Object.keys(all).length) node.properties[CAST_KEY] = all; else delete node.properties[CAST_KEY];
    node.graph?.setDirtyCanvas(true, true);
  }
  function liveOf(key) { if (!live.has(key)) live.set(key, { reading: false, error: null, failedSig: null, force: false, typedDuring: false }); return live.get(key); }
  // Who is in a picture: its boxes, else its Who buttons (or its label).
  function peopleIn(r) { return Array.isArray(r.boxes) && r.boxes.length ? whoFromBoxes(r.boxes) : mode === "REF2VA" ? whoOf(r) : Array.isArray(r.who) ? r.who : []; }
  // Pictures that show people: characters and frames, never a Pose or Custom one.
  function castRefs(n = null) {
    return refs.filter(r => r.kind === "image" && r.path && !["pose", "custom"].includes(kindsOf(r)[0])
      && (mode !== "REF2VA" || kindsOf(r).some(k => k === "character" || k.endsWith("-frame")))
      && (n == null || peopleIn(r).includes(n)));
  }
  function castNumbers() { return castOn ? [...new Set(castRefs().flatMap(peopleIn))].sort((a, b) => a - b) : []; }
  // Every card in order: the characters, then the place, the style and each pose (labelled REF2VA only).
  function castKeys() {
    if (!castOn) return [];
    const keys = castNumbers();
    if (mode !== "REF2VA") return keys;
    const shown = imagesOf().filter(r => r.path);
    for (const kind of ["place", "style"]) if (shown.some(r => kindsOf(r)[0] !== "pose" && kindsOf(r).includes(kind))) keys.push(kind);
    for (const r of shown) if (kindsOf(r)[0] === "pose" && r.item?.id) keys.push(`pose:${r.item.id}`);
    return keys;
  }
  // The pictures a card is read from.
  function cardRefs(key) {
    if (!isScene(key)) return castRefs(key);
    if (key.startsWith("pose:")) return imagesOf().filter(r => r.path && r.item?.id === key.slice(5) && kindsOf(r)[0] === "pose");
    return imagesOf().filter(r => r.path && kindsOf(r)[0] !== "pose" && kindsOf(r).includes(key));
  }
  function cardLabel(key) {
    if (!isScene(key)) return `Character ${key}`;
    if (key.startsWith("pose:")) { const r = cardRefs(key)[0]; return `Pose (Picture ${r ? pictureNumber(r) : "?"})`; }
    return SCENE_LABEL[key];
  }
  // Everything the reader is told for a card, so a change shows as out of
  // date. Not the frame kinds: a mode switch is not a change to how they look.
  function castSig(key) {
    const idea = [brief.value.trim(), shots.value, shotRows()];
    if (isScene(key)) return JSON.stringify([idea, cardRefs(key).map(r => [r.path, kindsOf(r).join(","), key.startsWith("pose:") ? peopleIn(r).length : 0, r.instructions || "", r.keep || "", r.drop || ""])]);
    const n = key;
    return JSON.stringify([idea, castRefs(n).map(r => [r.path, kindsOf(r).filter(k => !k.endsWith("-frame")).join(","), (r.boxes || []).find(b => b.n === n)?.box ?? null,
      peopleIn(r).indexOf(n), peopleIn(r).length, r.who_axis ?? "", r.instructions || "", r.keep || "", r.drop || ""])]);
  }
  function needsRead(key) {
    const e = cardOf(key), l = liveOf(key), sig = castSig(key);
    if (l.reading || !cardRefs(key).length) return false;
    if (l.force) return true;
    if (!autoCast.checked || castNoVision) return false;
    if (e.edited || l.failedSig === sig) return false;
    return !e.text || e.sig !== sig;
  }
  function renderCast() {
    if (closed) return;
    const keys = castKeys();
    castWrap.hidden = !keys.length;
    for (const [key, card] of castEls) if (!keys.includes(key)) { card.remove(); castEls.delete(key); }
    keys.forEach((key, i) => {
      const card = castEls.get(key) ?? castCardEl(key);
      if (castCards.children[i] !== card) castCards.insertBefore(card, castCards.children[i] ?? null);
      paintCard(key);
    });
    clearTimeout(castTimer);
    if (keys.some(needsRead)) castTimer = setTimeout(runCastQueue, 1500);
  }
  async function runCastQueue() {
    if (castBusy || castHeld || running || loadingModels || closed || !modelSel.value) return;
    const key = castKeys().find(needsRead);
    if (key == null) return;
    castBusy = true;
    try { await readCard(key, liveOf(key).force); } finally { castBusy = false; }
    if (closed) return;
    renderCast();
    // The wait is for the person to stop changing things, not between reads.
    if (castKeys().some(needsRead)) { clearTimeout(castTimer); runCastQueue(); }
  }
  // Read a picture's cards again: everyone in it, or its place, style or pose.
  function scanCards(keys) {
    for (const key of keys) { const l = liveOf(key); l.force = true; l.failedSig = null; }
    renderCast();
    clearTimeout(castTimer);
    runCastQueue();
  }
  function keysOfPicture(r) {
    if (kindsOf(r)[0] === "pose") return r.item?.id ? [`pose:${r.item.id}`] : [];
    return [...peopleIn(r), ...["place", "style"].filter(k => kindsOf(r).includes(k))];
  }
  // What the draft and the reader call a card: "2", "place", "style", "pose@3".
  function requestKey(key) {
    if (!isScene(key)) return String(key);
    if (key.startsWith("pose:")) { const r = cardRefs(key)[0]; return r ? `pose@${pictureNumber(r)}` : null; }
    return key;
  }
  async function readCard(key, redo) {
    const l = liveOf(key), sig = castSig(key), readingModel = modelSel.value;
    Object.assign(l, { reading: true, typedDuring: false, error: null });
    paintCard(key);
    const requestId = `cast-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    castRequest = requestId;
    warmed = true;
    const which = isScene(key) ? { subject: requestKey(key) } : { character: key };
    try {
      const res = await api.fetchApi("/dasiwa/h3/forge/cast", { timeoutMs: null, method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId, model: modelSel.value, settings: forgeSettings(), mode, brief: brief.value, shots: shots.value,
          shot_briefs: shotRows(), references: refs.map(({ item, ...r }) => r), ...which, redo: !!redo, keep_warm: true }) });
      const data = await res.json();
      warmed = true;
      if (!res.ok && data.error === "no_vision") { castNoVision = true; castNotice.textContent = data.message; }
      if (!res.ok) throw new Error(data.message || res.statusText);
      // Typed into while it was being read: what they typed stays.
      if (!l.typedDuring && !closed && live.get(key) === l && castKeys().includes(key) && castSig(key) === sig && modelSel.value === readingModel) { saveCard(key, { text: String(data.text || "").slice(0, CAST_MAX), edited: false, sig }); refsChanged(); }
      l.failedSig = null;
    } catch (err) {
      l.error = err.message;
      l.failedSig = sig;
    } finally {
      if (castRequest === requestId) castRequest = null;
      Object.assign(l, { reading: false, force: false, typedDuring: false });
      paintCard(key);
    }
  }
  const PLACEHOLDER = {
    character: "What they look like: hair, eyes, build, clothes. Read from their pictures, or write it yourself.",
    place: "The setting: where it is, its landmarks, the light and the time of day.",
    style: "The medium (photo, semi-real, anime, oil painting…), then the linework, shading and colours.",
    pose: "The pose only: the stance, where the arms and legs are, the gesture, which way they face. Looks and clothes are left out.",
  };
  const kindOfKey = key => !isScene(key) ? "character" : key.startsWith("pose:") ? "pose" : key;
  function castCardEl(key) {
    const img = el("img", { alt: "" });
    const text = el("textarea", { rows: 3, maxLength: CAST_MAX, placeholder: PLACEHOLDER[kindOfKey(key)] });
    text.setAttribute("aria-label", `${cardLabel(key)}: description`);
    text.addEventListener("input", () => {
      const l = liveOf(key);
      // What they type wins over a read on its way, and over a Find people still waiting.
      l.force = false; l.error = null;
      if (l.reading) l.typedDuring = true;
      saveCard(key, { text: text.value, edited: true, sig: castSig(key) });
      paintCard(key);
      refsChanged();
    });
    const card = el("div", { className: "cast-card" }, el("div", { className: "cast-thumb" }, img),
      el("div", { className: "cast-body" }, el("div", { className: "cast-top" }, el("span", { className: "cast-name" }), el("span", { className: "cast-status" })),
        text, el("span", { className: "cast-count" })));
    card.dataset.n = String(key);
    castEls.set(key, card);
    return card;
  }
  function paintCard(key) {
    const card = castEls.get(key);
    if (!card) return;
    const e = cardOf(key), l = liveOf(key);
    card.querySelector(".cast-name").textContent = cardLabel(key);
    const text = card.querySelector("textarea");
    if (document.activeElement !== text && text.value !== e.text) text.value = e.text;
    text.disabled = !!running || draftWaiting;
    card.querySelector(".cast-count").textContent = `${e.text.length}/${CAST_MAX}`;
    const stale = !!e.text && e.sig !== castSig(key);
    card.dataset.state = l.reading ? "reading" : l.error ? "error" : stale && e.edited ? "stale" : "";
    const scene = isScene(key);
    card.querySelector(".cast-status").textContent = l.reading ? (scene ? "Reading the picture…" : "Reading their pictures…")
      : l.error ? l.error
        : stale && e.edited ? (scene ? "Out of date: its picture or notes changed. Your text is used as it is; Scan picture reads it again."
          : "Out of date: their pictures or box changed. Your text is used as it is; Scan people on their picture reads them again.")
          : !e.text ? (needsRead(key) ? "Waiting to read…" : "Empty: drafted from the pictures as before.")
            : e.edited ? "Edited by you" : scene ? "From the picture" : "From their pictures";
    paintThumb(key, card.querySelector("img"));
  }
  // The card's picture: a character's box in the first picture that has one, else the first picture whole.
  function paintThumb(key, img) {
    const shown = cardRefs(key);
    const n = isScene(key) ? null : key;
    const r = (n != null && shown.find(x => (x.boxes || []).some(b => b.n === n))) || shown[0];
    if (!r) { img.removeAttribute("src"); return; }
    const box = (n != null && (r.boxes || []).find(b => b.n === n)?.box) || [0, 0, 1, 1];
    const thumbKey = `${r.path}:${box.join(",")}`;
    if (img.dataset.key === thumbKey) return;
    img.dataset.key = thumbKey;
    const pic = new Image();
    pic.onload = () => {
      if (img.dataset.key !== thumbKey) return;
      const [x1, y1, x2, y2] = box;
      const sw = Math.max(1, (x2 - x1) * pic.naturalWidth), sh = Math.max(1, (y2 - y1) * pic.naturalHeight);
      const scale = Math.min(1, 192 / sh, 144 / sw);
      const c = document.createElement("canvas");
      c.width = Math.round(sw * scale); c.height = Math.round(sh * scale);
      c.getContext("2d").drawImage(pic, x1 * pic.naturalWidth, y1 * pic.naturalHeight, sw, sh, 0, 0, c.width, c.height);
      img.src = c.toDataURL("image/jpeg", 0.85);
    };
    pic.src = viewUrl(r.path);
  }
  // A click on a box (no drag): their card, scrolled to and lit for a moment.
  function focusCard(key) {
    const card = castEls.get(key);
    if (!card || castWrap.hidden) return;
    card.scrollIntoView({ block: "center", behavior: "smooth" });
    card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1200);
  }
  // Renumbering a box swaps its card with the other number's, when only this picture holds them.
  function swapCards(ref, a, b) {
    if (castRefs().some(r => r !== ref && (peopleIn(r).includes(a) || peopleIn(r).includes(b)))) return;
    const all = { ...savedCards() };
    const [ca, cb] = [all[a], all[b]];
    if (ca) all[b] = ca; else delete all[b];
    if (cb) all[a] = cb; else delete all[a];
    node.properties ||= {};
    node.properties[CAST_KEY] = all;
    for (const n of [a, b]) live.delete(n);
  }
  // The cards with text, for the draft: used as written.
  function castForDraft() {
    const out = {};
    for (const key of castKeys()) { const card = cardOf(key), t = card.text.trim(), k = requestKey(key); if (t && k && (card.edited || card.sig === castSig(key))) out[k] = t; }
    return out;
  }
  if (continuity) box.append(el("div", { className: "muted", textContent: "The source ending and Duration guide this draft. A vision model uses tail frames internally; audio is not analyzed. Review the result, then Apply to node." }));
  if (continuity && mode === "REF2VA" && hook.setUseReferences) {
    includeReferences = el("input", { type: "checkbox", checked: continuity.use_references, onchange: e => {
      if (hook.contextKey?.() !== openedKey) { hook.setStatus("Director changed. Reopen Forge.", true); return; }
      briefs.set(briefKey, brief.value); hook.setUseReferences(e.target.checked); void open(node);
    } });
    box.append(el("label", { title: "Apply the currently enabled timeline reference media and RefMods during the extension. References are taken from the Director, not restored from the checkpoint. REF2VA only; start/end images in other modes are not reused." }, includeReferences, " Use REF2VA references and RefMods"));
  }
  // REF2VA pictures: one label each - Character 1-4, several characters in
    // one picture, Place, Style, First or Last frame, Pose, Custom. Pictures
    // with the same Character number are one subject (what subject groups
    // did), and the idea names the labels. For a new draft the node writes
    // who is who and no picture goes to the model; H3 sees them itself.
  // Base-mode pictures are frames by definition and need no label.
  const labelled = mode === "REF2VA" && refs.some(r => r.kind === "image" && r.easy_role);
  // Off by default: labels alone are what small models write well from.
  const seePictures = el("input", { type: "checkbox", checked: !!prefs.see_pictures });
  // Label editing remains available for mixed references, but easy drafting
  // requires every reference to be an image with a backend-supported label.
  const easyEligible = () => mode === "REF2VA" && !continuity && refs.length > 0 && refs.every(r => r.kind === "image" && EASY_ROLES.has(r.easy_role));
  const effectiveSeePictures = () => easyEligible() && seePictures.checked;
  const visionChoice = el("label", {}, seePictures, " Let the model see the pictures");
  const visionHint = el("span", { className: "muted", textContent: "Off: the writer works from the labels alone, which any model can do. On: it also looks at the pictures to match their look and setting. Needs a vision model. Smaller models (9B and under) can get less accurate with pictures, mixing up who is who or describing looks the pictures already carry." });
  const normalReferenceHint = el("span", { className: "muted", textContent: "These references use the normal reference path. Pictures are automatically sent to a model capable of seeing them; the labels-only vision choice does not apply." });
  const syncVisionUI = () => {
    visionChoice.hidden = visionHint.hidden = !easyEligible();
    normalReferenceHint.hidden = !refs.length || !!continuity || easyEligible();
  };
  syncVisionUI();
  if (refs.length) {
    if (!continuity && labelled) brief.placeholder = 'Name the labels: "Character 1 sits on the bed in the place. Character 2 walks in and waves."';
    const list = el("div", { className: "refs" });
    const tags = referenceTags(refs);
    for (const [index, ref] of refs.entries()) {
      const name = tags[index].map(tag => tag.slice(1, -1)).join(" + ");
      const thumb = ref.kind === "image" && ref.path ? el("img", { src: viewUrl(ref.path) }) : el("span", { className: "muted", textContent: ref.saved_reference ? "saved" : ref.kind });
      let roleCell;
      let instructions = null;
      let people = null, scanPicture = null;
      if (ref.kind === "image" && mode === "REF2VA" && ref.item) {
        // What the picture is (kinds that combine: Characters 1 + 2 and the
        // place behind them) and who is in it, tapped in order. A picture the
        // old single label still describes is saved as that label alone, so
        // drafts made before the buttons keep matching.
        let kinds = kindsOf(ref), who = whoOf(ref), axis = axisFrom(ref.who_axis, who);
        // On a Pose the numbers say who takes it, not who is in the picture.
        const isPose = () => kinds[0] === "pose";
        const takesWho = () => isPose() || kinds.some(k => k === "character" || k.endsWith("-frame"));
        // Boxes go on a picture of characters; never on a Pose or Custom one.
        const canBox = () => kinds.includes("character");
        const kindButtons = el("span", { className: "kinds", role: "group", title: "What this picture is. Character, Place, Style and the frames combine; Pose and Custom stand alone." });
        kindButtons.setAttribute("aria-label", `${name} is`);
        const whoButtons = el("span", { className: "who", role: "group", title: WHO_TITLE });
        whoButtons.setAttribute("aria-label", `${name} who is in it`);
        const axisButtons = el("span", { className: "who-axis", role: "group", title: AXIS_TITLE });
        axisButtons.setAttribute("aria-label", `${name} order`);
        const paint = () => {
          for (const b of kindButtons.querySelectorAll("button")) b.setAttribute("aria-pressed", String(kinds.includes(b.dataset.kind)));
          const most = whoShown(refs.filter(r => r.kind === "image").length, [...refs.flatMap(r => whoOf(r)), ...who]);
          whoButtons.title = isPose() ? "Who takes this pose: the characters whose numbers you tap. Nobody in the pose picture becomes a character." : WHO_TITLE;
          whoButtons.replaceChildren(el("span", { className: "who-label", textContent: isPose() ? "Who takes it" : "Who" }), ...Array.from({ length: Math.min(most, 32) }, (_, i) => {
            const n = i + 1, at = who.indexOf(n);
            const b = el("button", { type: "button", textContent: String(n), onclick: () => tapWho(n) });
            b.setAttribute("aria-pressed", String(at >= 0));
            b.dataset.order = axis && who.length > 1 && at >= 0 ? String(at + 1) : "";
            return b;
          }));
          // People boxes on a character picture say who is in it, left to right.
          const boxed = !!people?.boxes().length;
          whoButtons.hidden = !takesWho() || boxed;
          for (const b of axisButtons.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.axis === axis));
          // Where takers stand comes from the shots, not from the pose picture.
          axisButtons.hidden = !takesWho() || isPose() || who.length < 2 || boxed;
          if (people) people.element.hidden = !canBox();
          if (scanPicture) scanPicture.hidden = !castOn || !ref.path || !(isPose() || kinds.includes("place") || kinds.includes("style"));
        };
        const save = () => {
          if (isPose()) axis = "";
          const label = primaryLabel(kinds, who, axis);
          const simple = describesAlone(label, kinds, who, axis);
          ref.easy_role = label;
          const stored = isPose() ? "" : axisTo(axis, who);
          if (simple) { delete ref.picture_kinds; delete ref.who; delete ref.who_axis; }
          else { ref.picture_kinds = [...kinds]; ref.who = [...who]; if (stored) ref.who_axis = stored; else delete ref.who_axis; }
          const role = labelRole(label);
          ref.role = role.forge_role; ref.subject_group = role.forge_subject_group;
          persistReference(ref, { forge_label: label, ...role, forge_kinds: simple ? "" : kinds.join(","), forge_who: simple ? "" : who.join(","), forge_who_axis: simple ? "" : stored });
          paint();
          syncVisionUI();
          if (instructions) instructions.placeholder = INSTRUCTIONS_HINT[label] || "What should this reference contribute? (optional)";
          node.graph?.setDirtyCanvas(true, true);
          // Called here, not from a click listener: the buttons are redrawn
          // above, so a listener higher up sees a detached button and misses it.
          refsChanged();
        };
        // Character picks a number no other picture uses.
        const freeCharacter = () => {
          const used = new Set(refs.filter(r => r !== ref && kindsOf(r).includes("character")).flatMap(r => whoOf(r)));
          let n = 1; while (used.has(n) && n < WHO_MAX) n += 1; return n;
        };
        const tapKind = k => {
          const wasPose = isPose();
          if (ALONE_KINDS.has(k)) kinds = [k];
          else if (kinds.includes(k)) { if (kinds.length === 1) return; kinds = kinds.filter(x => x !== k); }
          else kinds = KIND_ORDER.filter(x => x === k || (kinds.includes(x) && !ALONE_KINDS.has(x)));
          // In and out of Pose the numbers change meaning (who is in it /
          // who takes it), so the old ones go rather than carry across.
          if (wasPose !== isPose()) { who = []; axis = ""; }
          if (kinds.includes("character") && !who.length) who = [freeCharacter()];
          if (!takesWho()) { who = []; axis = ""; }
          // Not a character picture any more: its boxes would say who is in it.
          if (!canBox()) people?.clear();
          save();
        };
        // A tap adds that number at the end; a second tap takes it out. A
        // character picture always shows somebody, so its last one stays.
        const tapWho = n => {
          if (who.includes(n)) { if (kinds.includes("character") && who.length === 1) return; who = who.filter(x => x !== n); }
          else who = [...who, n];
          if (who.length < 2) axis = "";
          save();
        };
        for (const [k, text] of KIND_BUTTONS) kindButtons.append(el("button", { type: "button", textContent: text, onclick: () => tapKind(k) }));
        kindButtons.querySelectorAll("button").forEach((b, i) => { b.dataset.kind = KIND_BUTTONS[i][0]; });
        for (const [value, text] of AXIS_BUTTONS) {
          const b = el("button", { type: "button", textContent: text, onclick: () => { axis = axis === value ? "" : value; save(); } });
          b.dataset.axis = value;
          axisButtons.append(b);
        }
        people = ref.path ? peopleEditor(ref, name, boxes => {
          if (boxes.length) { who = whoFromBoxes(boxes); axis = who.length > 1 ? "x" : ""; }
          save();
        }) : null;
        // A place, style or pose picture has no people to box: this reads its card again.
        scanPicture = el("button", { type: "button", textContent: "Scan picture", title: "Read this picture's place, style or pose card again. A card you edited is read again too.",
          onclick: () => { if (!running && !loadingModels) scanCards(keysOfPicture(ref).filter(isScene)); } });
        paint();
        roleCell = el("span", { className: "pick" }, kindButtons, whoButtons, axisButtons, el("span", { className: "who-axis" }, scanPicture));
      } else if (ref.kind === "image" && BASE_ROLE[mode] && ref.item && !ref.saved_reference) {
        // A frame in I2VA / FL2VA / L2VA: who is in it, so the idea's
        // "Character 1" can say which person in the frame that is (a
        // piggyback: tap the one carrying, then the one carried, front to back).
        let who = Array.isArray(ref.who) ? [...ref.who] : [];
        let axis = axisFrom(ref.who_axis, who);
        const whoButtons = el("span", { className: "who", role: "group", title: "Who is in this frame. In the idea, write \"Character 1\". With two or more, the order below can say where each one stands." });
        whoButtons.setAttribute("aria-label", `${name} who is in it`);
        const axisButtons = el("span", { className: "who-axis", role: "group", title: AXIS_TITLE });
        axisButtons.setAttribute("aria-label", `${name} order`);
        const save = () => {
          if (who.length < 2) axis = "";
          const stored = axisTo(axis, who);
          if (who.length) ref.who = [...who]; else delete ref.who;
          if (stored) ref.who_axis = stored; else delete ref.who_axis;
          persistReference(ref, { forge_who: who.join(","), forge_who_axis: stored });
          paint();
          node.graph?.setDirtyCanvas(true, true);
          refsChanged();
        };
        const paint = () => {
          const most = whoShown(refs.filter(r => r.kind === "image").length, who);
          whoButtons.replaceChildren(el("span", { className: "who-label", textContent: "Who" }), ...Array.from({ length: Math.min(most, 32) }, (_, i) => {
            const n = i + 1, at = who.indexOf(n);
            const b = el("button", { type: "button", textContent: String(n), onclick: () => { who = at >= 0 ? who.filter(x => x !== n) : [...who, n]; save(); } });
            b.setAttribute("aria-pressed", String(at >= 0));
            b.dataset.order = axis && who.length > 1 && at >= 0 ? String(at + 1) : "";
            return b;
          }));
          for (const b of axisButtons.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.axis === axis));
          const boxed = !!people?.boxes().length;
          whoButtons.hidden = boxed;
          axisButtons.hidden = who.length < 2 || boxed;
        };
        for (const [value, text] of AXIS_BUTTONS) {
          const b = el("button", { type: "button", textContent: text, onclick: () => { axis = axis === value ? "" : value; save(); } });
          b.dataset.axis = value;
          axisButtons.append(b);
        }
        people = ref.path ? peopleEditor(ref, name, boxes => {
          if (boxes.length) { who = whoFromBoxes(boxes); axis = who.length > 1 ? "x" : ""; }
          save();
        }) : null;
        paint();
        // FL2VA's ends are its slots: the first picture is the first frame and
        // the second the last, as framed_brief reads them on the server.
        const end = mode === "FL2VA" ? (refs.filter(r => r.kind === "image").indexOf(ref) === 1 ? "last frame" : "first frame") : BASE_ROLE[mode];
        roleCell = el("span", { className: "pick" }, el("span", { className: "muted", textContent: end }), whoButtons, axisButtons);
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
      list.append(el("div", { className: "ref" }, thumb, el("span", { textContent: name }), roleCell, notes, ...(people ? [people.element] : [])));
    }
    box.append(el("div", { className: "field" }, el("label", { textContent: "References on the timeline" }), list));
    if (labelled) {
      box.append(el("span", { className: "muted", textContent: continuity
        ? "Pictures with the same Character number are one character."
        : 'Pictures with the same Character number are one character. A picture with several people: tap each number in order, left to right (or pick top to bottom, front to back). A picture can be more than one thing: Character and Place for people with the background behind them, or a character who is also the first frame. On a Pose, the numbers say who takes the pose. In the idea, write "Character 1", "Character 2" and "the place". Image-only labelled drafts need no writer vision; mixed media and saved references use the full REF2VA path.' }));
    }
    box.append(castWrap);
    box.append(visionChoice, visionHint, normalReferenceHint);
  } else if (mode !== "T2VA" && !continuity) {
    box.append(el("div", { className: "muted", textContent: `${mode} expects pictures on the timeline; none are loaded, so the model writes from the idea alone.` }));
  }

  const inherited = continuity && mode === "REF2VA" ? hook.existingDefinitions?.() || { text: "", warning: "" } : { text: "", warning: "" };
  // Continuity identities belong to the Director prompt, not a second Forge editor.
  const definitions = { value: inherited.text };
  const structured = el("input", { type: "checkbox", checked: !!continuity && mode === "REF2VA" && (continuity.use_references || !!refPromptFields(hook.currentPrompt?.())) });
  if (inherited.warning) box.append(el("span", { className: "muted", textContent: inherited.warning }));
  if (continuity && mode === "REF2VA") {
    box.append(el("label", {}, structured, " Structured REF2VA draft"));
    box.append(el("span", { className: "muted", textContent: continuity.use_references ? "Timeline reference media and RefMods are enabled for the extension. Existing identities are carried forward; new subjects are defined for review." : "Timeline reference media and RefMods are off. Enable Use REF2VA references and RefMods above to apply the Director's current references during the extension; they are not restored from the checkpoint." }));
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
      input.addEventListener("input", () => { typed[i] = input.value; clearDraft(); renderCast(); });
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
    const cards = castForDraft();
    // The cards join the key only when there are some, so drafts saved before them still match.
    return JSON.stringify([brief.value.trim(), structured.checked, definitions.value, refs.map(({ item, ...r }) => r), ...(rows.some(Boolean) ? [rows] : []), ...(Object.keys(cards).length ? [cards] : [])]);
  };
  const referenceControls = Array.from(box.querySelectorAll(".refs input, .refs select, .refs textarea, .refs .pick button"));
  const controls = [brief, modelSel, detail, creativity, shots, structured, seePictures, autoCast];
  const setControlsDisabled = disabled => {
    [...controls, ...box.querySelectorAll(".refs input, .refs select, .refs textarea, .refs .pick button, .refs .people button, .cast textarea")].forEach(c => { c.disabled = disabled; });
    if (!disabled) for (const n of castEls.keys()) paintCard(n);
    shotBox.querySelectorAll("textarea").forEach(c => { c.disabled = disabled; });
    if (includeReferences) includeReferences.disabled = !!running;
  };
  setControlsDisabled(true);
  let result = null, selectedEntry = null;
  const showResult = (entry, fromHistory = false) => {
    if (closed) return;
    brief.value = entry.brief || "";
    if (typeof entry.structured === "boolean") structured.checked = entry.structured;
    // Missing vision options in old history always mean labels-only, even
    // when the browser preference was saved as on by a later draft.
    seePictures.checked = !!entry.draftOptions?.see_pictures;
    // Saved drafts are previews, not a source of identities for the next request.
    if (entry.draftOptions) {
      const { model, detail: level, creativity: preset, shots: count } = entry.draftOptions;
      if (Array.from(modelSel.options).some(o => o.value === model)) modelSel.value = model;
      detail.value = level; creativity.value = preset;
      // Drafts saved before the Shots control have none: they were Auto.
      const shotsValue = String(count ?? "Auto");
      if (Array.from(shots.options).some(o => o.value === shotsValue)) shots.value = shotsValue;
      // Drafts saved before the vision choice were all written blind.
      // Vision choice was restored above, including entries without options.
      shotTexts.set(node, Array.isArray(entry.draftOptions.shot_briefs) ? [...entry.draftOptions.shot_briefs] : []);
      renderShots();
      if (detail.oninput) detail.oninput();
    }
    selectedEntry = entry;
    // Explicit history reuse replaces prompt text, but never its source or references.
    result = fromHistory && savedContextMatches(entry.contextKey, hook.contextKey?.())
      ? { ...entry, contextKey: hook.contextKey?.(), forgeInputKey: inputKey() } : entry;
    output.hidden = false;
    output.textContent = entry.simple_prompt;
    applyBtn.disabled = loadingModels || !!running || !compatible(result);
    if (!compatible(result)) setStatus("Draft belongs to a different source, duration, model or prompt. Generate again for the current context.", true);
    else if (fromHistory) setStatus("Saved draft selected. Apply to node to use it.");
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
      const button = el("button", { type: "button", disabled: loadingModels || !!running, className: result && selectedEntry === entry ? "selected" : "", title: "Show this prompt; Apply to node to use it" },
        el("span", { textContent: `${index + 1}. ${entry.mode}${entry.continuity ? " · Continuity" : ""} · ${entry.model || "model"} · ${date}` }),
        el("span", { className: "excerpt", textContent: entry.simple_prompt.replace(/\s+/g, " ").slice(0, 150) }));
      button.onclick = () => showResult(entry, true);
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
    renderCast();
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
  brief.addEventListener("input", () => { clearDraft(); renderCast(); });
  refsChanged = () => { clearDraft(); renderCast(); };
  modelSel.addEventListener("change", clearDraft);
  // Another model may be able to read what this one could not.
  modelSel.addEventListener("change", () => { castNoVision = false; castNotice.textContent = ""; for (const l of live.values()) l.failedSig = null; renderCast(); });
  creativity.addEventListener("change", clearDraft);
  shots.addEventListener("change", () => { renderShots(); clearDraft(); renderCast(); });
  detail.addEventListener("input", clearDraft);
  structured.addEventListener("change", clearDraft);
  seePictures.addEventListener("change", clearDraft);
  referenceControls.filter(c => c.tagName !== "BUTTON").forEach(c => c.addEventListener(c.tagName === "SELECT" ? "change" : "input", clearDraft));
  box.addEventListener("click", e => { if (e.target.closest?.(".refs .pick button")) { clearDraft(); renderCast(); } });
  // A picture's notes go to the reader too.
  referenceControls.filter(c => c.tagName !== "BUTTON").forEach(c => c.addEventListener("change", renderCast));
  genBtn.onclick = async () => {
    if (closed) return;
    if (draftWaiting) return;
    if (running) { cancelRun(); genBtn.disabled = true; setStatus("Cancelling… the model stops at its next token, then unloads."); return; }
    const text = brief.value.trim();
    if (openedKey !== hook.contextKey?.()) { setStatus("Director context changed. Close and reopen Forge.", true); return; }
    const rows = shotRows();
    if (!text && !continuity && !rows.some(r => r.trim())) { setStatus("Write the idea first.", true); return; }
    const missing = refs.find(r => r.role === "custom" && !r.instructions.trim());
    if (missing) { setStatus("Custom reference: describe what this image should contribute, or choose a preset role.", true); return; }
    genBtn.disabled = true;
    draftWaiting = true; setControlsDisabled(true);
    if (castBusy) setStatus("Waiting for the Cast card being read, then writing…");
    await castQuiet();
    draftWaiting = false;
    setControlsDisabled(false);
    genBtn.disabled = false;
    if (closed || openedKey !== hook.contextKey?.()) { castResume(); return; }
    briefs.set(briefKey, text);
    remember({ model: modelSel.value, creativity: creativity.value, detail: Number(detail.value), shots: shots.value, see_pictures: seePictures.checked });
    result = null; output.hidden = true; output.textContent = ""; renderHistory();
    applyBtn.disabled = true;
    const easy = easyEligible();
    const draftOptions = { model: modelSel.value, detail: Number(detail.value), creativity: creativity.value, shots: shots.value, shot_briefs: rows, see_pictures: easy && seePictures.checked };
    const requestKey = openedKey, forgeInputKey = inputKey();
    const requestId = `forge-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    running = requestId; setControlsDisabled(true); renderHistory();
    genBtn.textContent = "Cancel";
    const started = Date.now();
    statusTimer = setInterval(() => setStatus(`Writing with ${modelSel.selectedOptions[0]?.textContent || modelSel.value}… ${Math.round((Date.now() - started) / 1000)}s (Cancel stops it; the model unloads either way)`), 500);
    try {
      const res = await api.fetchApi("/dasiwa/h3/forge", {
        timeoutMs: null,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: requestId, brief: text, mode, duration: hook.duration(), model: modelSel.value,
          output_canvas: hook.outputCanvas?.() ?? null,
          detail: Number(detail.value), creativity: creativity.value, shots: shots.value, shot_briefs: rows,
          references: refs.map(({ item, ...r }) => r), settings: forgeSettings(), continuity, easy, see_pictures: easy && seePictures.checked,
          ...(castOn ? { cast: castForDraft() } : {}),
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
      const seen = data.easy ? (data.saw_images ? ` · picture labels used, and looked at ${data.saw_images} picture${data.saw_images === 1 ? "" : "s"}`
          : seePictures.checked && data.vision === false ? " · this model cannot see images, so it used the picture labels only"
          : " · picture labels used (no images sent to the writer)") : data.saw_images ? ` · looked at ${data.saw_images} picture${data.saw_images === 1 ? "" : "s"}` : continuity ? " · text context only (no tail images)" : refs.some(r => r.kind === "image") && data.vision === false ? " · this model cannot see images, so it wrote from your idea only" : "";
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
      castResume();
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
