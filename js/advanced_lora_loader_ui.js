import { app } from "../../scripts/app.js";

const THEMES = {
  a: {
    name: "JADE",
    rowOdd: "#1a1a1a", rowEven: "#202020",
    onColor: "#4CAF50", offColor: "#f44336",
    nameLoaded: "#dddddd", nameEmpty: "#444444",
    strColor: "#00FFCC", vColor: "#00FFCC", aColor: "#2196F3",
    ratioV: "#00FFCC99", ratioA: "#2196F399",
    arrowColor: "#555555",
    btnBg: "#00FFCC18", btnBorder: "#00FFCC55", btnText: "#00FFCC",
    divider: "#2a2a2a", wood: false,
  },
  b: {
    name: "NEON",
    rowOdd: "#0d0e1a", rowEven: "#111220",
    onColor: "#39ff7a", offColor: "#ff2d88",
    nameLoaded: "#9ba0cc", nameEmpty: "#2a2a3a",
    strColor: "#ff2d88", vColor: "#ff2d88", aColor: "#bf5fff",
    ratioV: "#ff2d8899", ratioA: "#bf5fff99",
    arrowColor: "#333355",
    btnBg: "#ff2d8818", btnBorder: "#ff2d8855", btnText: "#ff2d88",
    divider: "#1a1a2a", wood: false,
  },
  c: {
    name: "STUDIO",
    rowOdd: "#1e1a15", rowEven: "#231f19",
    onColor: "#7ecb6f", offColor: "#c05050",
    nameLoaded: "#c8b898", nameEmpty: "#4a4030",
    strColor: "#F5A623", vColor: "#F5A623", aColor: "#e07b39",
    ratioV: "#F5A62399", ratioA: "#e07b3999",
    arrowColor: "#4a4030",
    btnBg: "#F5A62318", btnBorder: "#F5A62355", btnText: "#F5A623",
    divider: "#3a3028", wood: false,
  },
  d: {
    name: "CHROME",
    rowOdd: "#eaeaea", rowEven: "#e0e0e0",
    onColor: "#1a7c38", offColor: "#b03030",
    nameLoaded: "#222222", nameEmpty: "#bbbbbb",
    strColor: "#111111", vColor: "#111111", aColor: "#1414cc",
    ratioV: "#11111199", ratioA: "#1414cc99",
    arrowColor: "#aaaaaa",
    btnBg: "#00000018", btnBorder: "#00000055", btnText: "#111111",
    divider: "#cccccc", wood: false,
  },
  e: {
    name: "OLED",
    rowOdd: "#000000", rowEven: "#080808",
    onColor: "#00ff41", offColor: "#ff0033",
    nameLoaded: "#ffffff", nameEmpty: "#333333",
    strColor: "#ffffff", vColor: "#00e5ff", aColor: "#ff6b35",
    ratioV: "#00e5ff99", ratioA: "#ff6b3599",
    arrowColor: "#333333",
    btnBg: "#ffffff08", btnBorder: "#ffffff33", btnText: "#ffffff",
    divider: "#111111", wood: false,
  },
  f: {
    name: "WOOD",
    rowOdd: "#2c1f0e", rowEven: "#321f0a",
    onColor: "#a8d878", offColor: "#d45a38",
    nameLoaded: "#f0d8a0", nameEmpty: "#6a5030",
    strColor: "#f0c060", vColor: "#f0c060", aColor: "#c87830",
    ratioV: "#f0c06099", ratioA: "#c8783099",
    arrowColor: "#6a5030",
    btnBg: "#f0c06018", btnBorder: "#f0c06055", btnText: "#f0c060",
    divider: "#4a3018", wood: true,
  },
};

const THEME_ORDER = ["a", "b", "c", "d", "e", "f"];
const MODEL_TYPES = ["Basic", "LTX-2.3"];
const hasSeparatedAudio = modelType => modelType === "LTX-2.3";
const keyCache = {};
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const bump = (v, d) => Math.round(clamp((v ?? 1.0) + d, 0.0, 2.0) * 100) / 100;
const bumpS = (v, d) => Math.round(clamp((v ?? 1.0) + d, -5.0, 5.0) * 100) / 100;

// ── Inline value editor (ComfyUI-style) ──────────────────────────────────────
// A DOM overlay anchored to the clicked pill in node space: a number input
// pre-filled with the current value plus OK / Cancel. Replaces window.prompt()
// on the STR / VIS / A pills. The box stays next to the pill, follows pan and
// node drags, and scales with the canvas zoom (re-tracked every frame), like
// the native ComfyUI widget value editors. Theme-colored, node-native look.
let activeValueEditor = null;

// Keep the editor pinned to its pill, mirroring the frontend's own widget
// DOM layer. The core's zoom helper gives the view law for this build:
//   graph = client/scale - ds.offset  =>  client = rect + scale*(graph + offset)
// (rect = canvas element's page position). So:
//   left/top    = rect + scale * (node.pos + anchor + gap + offset)   (client px)
//   transform   = scale(z), transform-origin 0 0
// The box corner stays glued to the pill through pan / zoom / node drags
// (re-tracked every frame) and grows/shrinks with the zoom level. The
// offset is always INSIDE the scale factor — adding it outside throws the
// box off by (scale - 1) * offset on any pan.
//
// View state is looked up defensively (property names differ between
// litegraph builds: canvas.ds / canvas.view / graph.view). The host element
// is the graph canvas's own HTMLCanvasElement (canvas.canvas) — NOT a bare
// document.querySelector("canvas"), which in a ComfyUI page grabs the first
// media-preview <canvas> and throws the whole position off. If we can't
// resolve a reliable host we bail to the click fallback rather than risk a
// wrong-canvas position.
function _viewState() {
  const lg = app?.canvas || app?.graph?.canvas || window.litegraph?.canvas;
  const view = lg?.ds || lg?.view || app?.graph?.view;
  if (!view) return null;
  const scale = view.scale;
  const off = view.offset;
  if (!(scale > 0) || typeof off?.[0] !== "number" || typeof off?.[1] !== "number") return null;
  const host =
    lg?.canvas ||
    document.getElementById("graph-canvas");
  if (!host?.getBoundingClientRect) return null;
  const rect = host.getBoundingClientRect();
  if (rect.width < 1) return null; // detached / hidden (view-mode switch)
  return { scale, offset: off, left: rect.left, top: rect.top };
}

function positionValueEditor(overlay) {
  const { node, anchor, clientX, clientY } = overlay._editorOpts;
  const surface = node?._loraSurface;
  if (surface?.isConnected) {
    const rect = surface.getBoundingClientRect();
    const scale = rect.width / node.size[0];
    overlay.style.left = (rect.left + scale * (anchor[0] + overlay._editorGap)) + "px";
    overlay.style.top = (rect.top + scale * anchor[1]) + "px";
    overlay.style.transform = `scale(${scale})`;
    return;
  }
  const v = _viewState();
  if (!node?.pos || !v) {
    // View state unavailable: static placement at the original click.
    overlay.style.transform = "";
    overlay.style.left = Math.max(4, Math.min(clientX + 10, window.innerWidth - overlay._editorWidth - 8)) + "px";
    overlay.style.top = Math.max(4, clientY + 10) + "px";
    return;
  }
  const z = v.scale;
  overlay.style.left =
    (v.left + z * (node.pos[0] + anchor[0] + overlay._editorGap + v.offset[0])) + "px";
  overlay.style.top = (v.top + z * (node.pos[1] + anchor[1] + v.offset[1])) + "px";
  overlay.style.transform = "scale(" + z + ")";
}

function openValueEditor(opts) {
  // opts: { node, anchor: [ax, ay] (local node px of the pill's top-right edge),
  //         clientX, clientY (fallback when the view API is unavailable),
  //         label, value, min, max, step, theme, onCommit }
  closeValueEditor();
  const t = opts.theme;
  const width = 150;
  const GAP = 6; // gap between the pill edge and the editor box, in node px

  const overlay = document.createElement("div");
  overlay.style.cssText = [
    "position:fixed",
    "transform-origin:0 0",
    "z-index:10003",
    "background:#0e1116",
    "border:1px solid " + t.btnBorder,
    "border-radius:4px",
    "padding:6px 8px",
    "display:flex",
    "flex-direction:column",
    "gap:5px",
    "box-shadow:0 6px 18px #000",
    "font:11px 'Courier New',monospace",
    "color:" + t.btnText,
  ].join(";");
  const title = document.createElement("div");
  title.textContent = opts.label;
  title.style.cssText = "font-size:10px;opacity:.75;";
  overlay.append(title);

  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:6px;align-items:center;";

  const inp = document.createElement("input");
  inp.type = "number";
  inp.step = String(opts.step ?? 0.01);
  inp.min = String(opts.min);
  inp.max = String(opts.max);
  inp.value = String(opts.value);
  inp.style.cssText =
    "width:" + (width - 90) + "px;background:#1a1f27;border:1px solid " + t.btnBorder +
    ";color:" + t.btnText + ";padding:3px 5px;font:11px 'Courier New',monospace;";
  row.append(inp);

  const okBtn = document.createElement("button");
  okBtn.textContent = "OK";
  okBtn.style.cssText =
    "background:" + t.btnBg + ";border:1px solid " + t.btnBorder + ";color:" + t.btnText +
    ";padding:3px 8px;border-radius:3px;cursor:pointer;font:11px 'Courier New',monospace;";

  const cancelBtn = document.createElement("button");
  cancelBtn.textContent = "✕";
  cancelBtn.style.cssText =
    "background:transparent;border:1px solid " + t.arrowColor + ";color:" + t.btnText +
    ";padding:3px 6px;border-radius:3px;cursor:pointer;font:11px 'Courier New',monospace;";
  row.append(okBtn, cancelBtn);
  overlay.append(row);
  document.body.append(overlay);

  const commit = raw => {
    const parsed = parseFloat(raw);
    if (Number.isNaN(parsed)) { close(); return; }
    const clamped = Math.min(opts.max, Math.max(opts.min, parsed));
    opts.onCommit(clamped);
    close();
  };
  const close = () => {
    document.removeEventListener("mousedown", onDocDown, { capture: true });
    overlay.remove();
    if (activeValueEditor === overlay) activeValueEditor = null;
  };
  const onDocDown = ev => {
    if (!overlay.contains(ev.target)) close();
  };
  document.addEventListener("mousedown", onDocDown, { capture: true });
  overlay._docDown = onDocDown;

  okBtn.onclick = () => commit(inp.value);
  cancelBtn.onclick = close;
  inp.onkeydown = ev => {
    if (ev.key === "Enter") { ev.preventDefault(); commit(inp.value); }
    else if (ev.key === "Escape") { ev.preventDefault(); close(); }
  };

  overlay._editorOpts = { node: opts.node, anchor: opts.anchor, clientX: opts.clientX, clientY: opts.clientY };
  overlay._editorWidth = width;
  overlay._editorGap = GAP;
  activeValueEditor = overlay;
  positionValueEditor(overlay);
  requestAnimationFrame(() => {
    inp.focus();
    inp.select?.();
    const track = () => {
      if (!activeValueEditor || activeValueEditor !== overlay) return;
      positionValueEditor(overlay);
      requestAnimationFrame(track);
    };
    requestAnimationFrame(track);
  });
}

function closeValueEditor() {
  if (!activeValueEditor) return;
  const el = activeValueEditor;
  activeValueEditor = null;
  if (el._docDown) document.removeEventListener("mousedown", el._docDown, { capture: true });
  el.remove();
}

// ── LoRA info panel (Civitai link, trigger words, images) ───────────────────
// DOM overlay, same language as the inline value editor: fixed position,
// theme-colored, Esc / outside-click closes. Data comes from
// /dasiwa/ltx2/lorainfo (sha256 + safetensors header metadata + cached
// Civitai by-hash lookup); sidecar images via /dasiwa/ltx2/loraimg.
const escHtml = v => String(v ?? "").replace(/[&<>"']/g, c => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

function buildLoraInfoPanelHtml(info, theme) {
  const t = theme || THEMES.a;
  const esc = escHtml;
  const words = (info.trainedWords || []).filter(w => w && w.word);
  const images = (info.images || []).filter(im => im && im.url);
  const btnStyle =
    "background:" + t.btnBg + ";border:1px solid " + t.btnBorder + ";color:" + t.btnText +
    ";padding:2px 8px;border-radius:3px;cursor:pointer;font:11px 'Courier New',monospace;";

  const out = [];
  // File name row (wraps independently) with the sha hint on its own row below.
  out.push(
    '<div style="font:11px \'Courier New\',monospace;opacity:.8;overflow-wrap:anywhere;">' + esc(info.file) +
    "</div>" +
    (info.sha256
      ? '<div style="font:9px \'Courier New\',monospace;opacity:.5;user-select:all;">sha256 ' + esc(info.sha256.slice(0, 16)) + "</div>"
      : "")
  );
  // Controls row: actions only.
  out.push(
    '<div style="display:flex;gap:6px;align-items:center;margin-top:4px;">' +
      '<button data-action="refresh" title="Re-fetch from Civitai" style="' + btnStyle + '">Refresh</button>' +
      (words.length ? '<button data-action="copy-words" style="' + btnStyle + '">Copy all</button>' : "") +
      (words.length ? '<button data-action="copy-selected" style="' + btnStyle + '">Copy selected</button>' : "") +
      '<button data-action="close" title="Close" style="' + btnStyle + ';margin-left:auto;">×</button>' +
    "</div>"
  );
  if (info.civitaiFound) {
    // The backend returns both mirror links (.com + .red, same modelId/versionId
    // on each). Both are always shown; each is preceded by a colored label
    // (BLUE: for .com, RED: for .red) and the anchor itself is tinted to match.
    const comLink = (info.links || []).find(l => typeof l === "string" && l.includes("civitai.com")) || "";
    const redLink = (info.links || []).find(l => typeof l === "string" && l.includes("civitai.red")) || "";
    const linkRow = (label, color, url) => url
      ? '<span style="display:block;overflow-wrap:anywhere;"><span style="color:' + color + ';font-weight:bold;">' + label + '</span> ' +
        '<a href="' + esc(url) + '" target="_blank" rel="noreferrer" style="color:' + color + ';">' + esc(url) + "</a></span>"
      : "";
    const linksHtml = linkRow("BLUE:", "#58a6ff", comLink) + linkRow("RED:", "#ff6b6b", redLink);
    out.push(
      '<div style="margin-top:6px;font:12px \'Courier New\',monospace;">' +
        '<div style="font-weight:bold;">' + esc(info.name || info.file) + "</div>" +
        (info.type || info.baseModel
          ? '<div style="opacity:.6;">' + esc([info.type, info.baseModel].filter(Boolean).join(" · ")) + "</div>"
          : "") +
        (linksHtml ? '<div style="margin-top:4px;">' + linksHtml + "</div>" : "") +
      "</div>"
    );
  } else {
    out.push('<div style="margin-top:6px;font:11px \'Courier New\',monospace;opacity:.8;">⚠ ' + esc(info.civitaiError || "no civitai data") + "</div>");
  }
  if (words.length) {
    out.push(
      '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">' +
        words.map(w =>
          '<button data-word="' + esc(w.word) + '" title="Toggle selection" style="' + btnStyle + '">' +
          esc(w.word) + (w.count ? ' <span style="opacity:.5;">' + w.count + "</span>" : "") +
          "</button>"
        ).join("") +
      "</div>"
    );
  }
  if (info.imageLocal) {
    out.push(
      '<div style="margin-top:8px;">' +
        '<img src="' + esc(info.imageLocal) + '" style="width:100%;max-height:160px;object-fit:cover;border-radius:3px;">' +
        '<div style="font:9px \'Courier New\',monospace;opacity:.5;">local sidecar</div>' +
      "</div>"
    );
  }
  if (images.length) {
    out.push(
      '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:8px;">' +
        images.slice(0, 6).map(im =>
          im.type === "video"
            ? '<video src="' + esc(im.url) + '" muted loop controls style="width:100%;max-height:140px;background:#000;"></video>'
            : '<figure style="margin:0;">' +
                '<img src="' + esc(im.url) + '" style="width:100%;max-height:140px;object-fit:cover;border-radius:2px;">' +
                '<figcaption style="font:8px \'Courier New\',monospace;opacity:.5;">' +
                  (im.seed != null ? "seed " + im.seed : "") + (im.model ? " · " + esc(im.model) : "") +
                "</figcaption></figure>"
        ).join("") +
      "</div>"
    );
  }
  return out.join("");
}

let activeInfoPanel = null;

function closeInfoPanel() {
  if (activeInfoPanel) {
    activeInfoPanel.remove();
    activeInfoPanel = null;
  }
}

function copyInfoWords(items) {
  const text = items.filter(Boolean).join(" ");
  if (text && navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
}

function openLoraInfo(lora, theme) {
  closeInfoPanel();
  const t = theme || THEMES.a;
  const panel = document.createElement("div");
  panel.style.cssText = [
    "position:fixed", "inset:0", "z-index:10004",
    "background:#00000080", "display:flex", "align-items:center", "justify-content:center",
  ].join(";");
  const box = document.createElement("div");
  box.style.cssText = [
    "max-width:420px", "max-height:70vh", "overflow:auto",
    "background:#0e1116", "border:1px solid " + t.btnBorder, "border-radius:6px",
    "padding:10px 12px", "box-shadow:0 6px 18px #000",
    "font:12px 'Courier New',monospace", "color:" + t.btnText,
  ].join(";");
  panel.appendChild(box);
  document.body.appendChild(panel);
  activeInfoPanel = panel;

  const close = () => {
    panel.remove();
    document.removeEventListener("keydown", onKey, true);
    if (activeInfoPanel === panel) activeInfoPanel = null;
  };
  const onKey = ev => { if (ev.key === "Escape") close(); };
  document.addEventListener("keydown", onKey, true);
  panel.addEventListener("mousedown", ev => { if (ev.target === panel) close(); });

  const paint = info => {
    box.innerHTML = buildLoraInfoPanelHtml(info, t);
    box.querySelector('[data-action="close"]')?.addEventListener("click", close);
    box.querySelector('[data-action="refresh"]')?.addEventListener("click", () => load(true));
    const wordBtns = [...box.querySelectorAll("[data-word]")];
    const copyAll = box.querySelector('[data-action="copy-words"]');
    const copySel = box.querySelector('[data-action="copy-selected"]');
    copyAll?.addEventListener("click", () => copyInfoWords(wordBtns.map(w => w.dataset.word)));
    copySel?.addEventListener("click", () =>
      copyInfoWords(wordBtns.filter(w => w.classList.contains("sel")).map(w => w.dataset.word))
    );
    wordBtns.forEach(w => w.addEventListener("click", () => {
      w.classList.toggle("sel");
      w.style.background = w.classList.contains("sel") ? "rgba(255,255,255,.15)" : "";
    }));
  };
  const load = refresh => {
    box.innerHTML = "<div style='opacity:.6;'>Loading info…</div>";
    fetch(`/dasiwa/ltx2/lorainfo?lora=${encodeURIComponent(lora)}${refresh ? "&refresh=1" : ""}`)
      .then(r => r.json())
      .then(paint)
      .catch(() => paint({
        file: lora,
        civitaiFound: false,
        civitaiError: "info fetch failed (server route unavailable?)",
        links: [], images: [], trainedWords: [],
      }));
  };
  load(false);
}

async function getCurrentLoraList(nodeData) {
  const fallback = nodeData?.input?.hidden?.available_loras?.[0] || ["None"];

  try {
    const response = await fetch("/object_info/DaSiWa_LTX2LoraLoader", { cache: "no-store" });
    if (!response.ok) return fallback;

    const info = await response.json();
    const refreshed = info?.DaSiWa_LTX2LoraLoader?.input?.hidden?.available_loras?.[0]
      || info?.input?.hidden?.available_loras?.[0];

    return Array.isArray(refreshed) && refreshed.length ? refreshed : fallback;
  } catch (error) {
    console.warn("[DaSiWa LTX2] Failed to refresh LoRA list from ComfyUI object_info:", error);
    return fallback;
  }
}

const CONTROL_DESCRIPTIONS = {
  mode: "Choose how this LoRA is applied. Only LTX-2.3 supports separate audio controls.",
  theme: "Cycle the visual theme of this advanced LoRA stacker.",
  toggleAll: "Enable every slot, or disable every slot when they are already enabled.",
  add: "Add one LoRA slot to the stack.",
  remove: "Remove the last LoRA slot from the stack.",
  clear: "Reset every current slot to None with STR/VIS/A at 1.00, without changing how many slots are showing.",
  enabled: "Enable or disable this LoRA slot.",
  lora: "Choose the LoRA file for this slot.",
  str: "Master LoRA strength. This is multiplied by VIS and, for LTX-2.3, A. Range: -5.0 to 5.0.",
  video: "Visual multiplier. In Basic mode it controls the full LoRA map, including image models. Effective visual strength = STR x VIS. Range: 0.0 to 2.0.",
  audio: "Audio branch multiplier. Effective audio strength = STR x A. Range: 0.0 to 2.0.",
  keys: "Detected LoRA key counts for video and audio branches.",
  cache: "Activate ComfyUI-side caching of the loaded LoRA file + metadata. Off by default; when on, each unique file is read once instead of once per slot.",
};

function setCanvasTooltip(text) {
  const canvas = app?.canvas?.canvas;
  if (canvas && canvas.title !== text) canvas.title = text || "";
}

function loraActionAt(node, [x, y]) {
  if (node.flags.collapsed) return null;
  const row = Math.floor((y - 59) / 28);
  const data = JSON.parse(node.properties.stack_data);
  if (!data[row] || y < 61 + row * 28 || y > 85 + row * 28) return null;
  const s = node.size[0] / 1000;
  if (x > 934 * s && x < 960 * s) return { row, action: "info", enabled: data[row].lora !== "None" };
  if (x > 966 * s && x < 992 * s) return { row, action: "trash", enabled: data[row].lora !== "None" };
  return null;
}

// Use DOM text and vector icons, like Director; never scale a text bitmap.
function mountLoraSurface(node, calcHeight, getTooltipAt) {
  if (!node.addDOMWidget || node._loraSurface) return;
  const root = document.createElement("div");
  root.style.cssText = "width:100%;min-width:0;flex:1;overflow:hidden;";
  const surface = document.createElement("div");
  surface.className = "dasiwa-lora-surface";
  surface.style.cssText = "position:relative;width:100%;font-family:'Courier New',monospace;";
  root.append(surface);
  node._loraSurface = surface;
  const height = () => calcHeight(JSON.parse(node.properties.stack_data).length);
  let frame = 0;
  let removedFromGraph = false;
  const paint = () => {
    frame = 0;
    root.style.height = surface.style.height = `${height()}px`;
    node.drawLoraStack();
  };
  const repaint = () => {
    if (!frame && !removedFromGraph) frame = requestAnimationFrame(paint);
  };
  const widget = node.addDOMWidget("lora_stack_ui", "custom", root, {
    serialize: false, hideOnZoom: false, margin: 0, hideInPanel: true, // #97
    getMinHeight: height, getMaxHeight: height, getHeight: height,
  });
  widget.serialize = false;
  widget.computeSize = () => [node.size[0], height()];
  const localPosition = event => {
    const rect = surface.getBoundingClientRect();
    return [(event.clientX - rect.left) * node.size[0] / rect.width,
      (event.clientY - rect.top) * height() / rect.height];
  };
  surface.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    event.stopPropagation();
    event.preventDefault();
    node.onMouseDown(event, localPosition(event));
    repaint();
  });
  surface.addEventListener("click", event => {
    if (event.detail !== 0 || event.target.tagName !== "BUTTON") return;
    const rect = event.target.getBoundingClientRect();
    node.onMouseDown(event, localPosition({clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2}));
    repaint();
  });
  surface.addEventListener("pointermove", event => {
    const position = localPosition(event);
    surface.title = getTooltipAt(node, position);
    node.onMouseMove(event, position);
    const action = loraActionAt(node, position);
    surface.style.cursor = action ? (action.enabled ? "pointer" : "not-allowed") : "";
  });
  surface.addEventListener("pointerleave", () => {
    surface.title = "";
    surface.style.cursor = "";
    node.onMouseLeave();
  });
  surface.addEventListener("contextmenu", event => {
    if (root.closest(".lg-node")) return;
    event.preventDefault();
    event.stopPropagation();
    app.canvas.processContextMenu(node, event);
  });
  const dirty = node.setDirtyCanvas;
  node.setDirtyCanvas = function (...args) {
    repaint();
    return dirty?.apply(this, args);
  };
  const configure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = configure?.apply(this, args);
    // Native serialized slots remain canonical, including array-only workflows.
    if (args[0]?.widgets_values) {
      for (const name of ["stack_data", "model_type", "use_cache"]) {
        const value = this.widgets.find(w => w.name === name)?.value;
        if (value !== undefined) this.properties[name] = value;
      }
    }
    repaint();
    return result;
  };
  const observer = new ResizeObserver(repaint);
  observer.observe(root);
  const added = node.onAdded;
  node.onAdded = function (...args) {
    const result = added?.apply(this, args);
    removedFromGraph = false;
    this._loraSurface = surface;
    observer.observe(root);
    repaint();
    return result;
  };
  const removed = node.onRemoved;
  node.onRemoved = function (...args) {
    removedFromGraph = true;
    observer.disconnect();
    cancelAnimationFrame(frame);
    frame = 0;
    if (activeValueEditor?._editorOpts?.node === this) closeValueEditor();
    this._loraSurface = null;
    return removed?.apply(this, args);
  };
  repaint();
}

app.registerExtension({
  name: "DaSiWa.LTX2LoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "DaSiWa_LTX2LoraLoader") return;

    // Compact layout constants
    const ROW_H = 28;
    const BTN_H = 16;
    const BTN_Y = 40;
    const ROW_START = BTN_Y + BTN_H + 3;
    const calcHeight = n => ROW_START + n * ROW_H + 2;

    nodeType.prototype.onNodeCreated = function () {
      this.properties = this.properties || {};
      if (!this.properties.stack_data) {
        this.properties.stack_data = JSON.stringify(
          [{ on: true, lora: "None", str: 1.0, vs: 1.0, as: 1.0 }]
        );
      }
      if (!this.properties.theme) this.properties.theme = "a";
      if (this.properties.use_cache === undefined) this.properties.use_cache = false;   // default off
      if (!MODEL_TYPES.includes(this.properties.model_type)) this.properties.model_type = "Basic";
      const rows = JSON.parse(this.properties.stack_data);
      this.size = [720, calcHeight(rows.length)];

      const hideWidget = widget => {
        widget.hidden = true;
        widget.options = { ...(widget.options || {}), hidden: true };
        widget.draw = () => {};
        widget.computeSize = () => [0, -4];
        return widget;
      };
      let stackWidget = this.widgets?.find(widget => widget.name === "stack_data");
      if (!stackWidget) stackWidget = this.addWidget("text", "stack_data", this.properties.stack_data, () => {});
      let modeWidget = this.widgets?.find(widget => widget.name === "model_type");
      if (!modeWidget) {
        modeWidget = this.addWidget("combo", "model_type", this.properties.model_type, value => {
          this.properties.model_type = value;
          this.setDirtyCanvas(true);
        }, { values: MODEL_TYPES });
      }
      let cacheWidget = this.widgets?.find(widget => widget.name === "use_cache");
      if (!cacheWidget) cacheWidget = this.addWidget("boolean", "use_cache", this.properties.use_cache, value => {
        this.properties.use_cache = value;
        this.setDirtyCanvas(true);
      });
      stackWidget.value = this.properties.stack_data;
      modeWidget.value = this.properties.model_type;
      cacheWidget.value = this.properties.use_cache;
      this.widgets = [hideWidget(stackWidget), hideWidget(modeWidget), hideWidget(cacheWidget)];
      mountLoraSurface(this, calcHeight, getTooltipAt);
    };

    nodeType.prototype.getExtraMenuOptions = function () {
      return [];
    };

    const sync = node => {
      const w = node.widgets.find(w => w.name === "stack_data");
      if (w) w.value = node.properties.stack_data;
    };

    const syncModeWidget = node => {
      const w = node.widgets.find(widget => widget.name === "model_type");
      if (w) w.value = node.properties.model_type;
    };

    const syncCacheWidget = node => {
      const w = node.widgets.find(widget => widget.name === "use_cache");
      if (w) w.value = node.properties.use_cache;
    };

    const getTooltipAt = (node, local_pos) => {
      if (node.flags.collapsed) return "";
      let data = [];
      try {
        data = JSON.parse(node.properties.stack_data);
      } catch {
        return "";
      }

      const action = loraActionAt(node, local_pos);
      if (action) {
        if (!action.enabled) return "Select a LoRA to enable this button";
        return action.action === "info"
          ? "Show LoRA info (Civitai, trigger words, images)"
          : "Clear this slot (does not delete the LoRA file)";
      }
      const [x, y] = local_pos;
      const W = node.size[0];
      const s = W / 1000;
      const BTN_H = 16;
      const BTN_Y = 40;
      const ROW_START = BTN_Y + BTN_H + 3;
      const ROW_H = 28;
      const btnW = 110;
      const btnX = (W - btnW) / 2;
      const plusX = btnX + btnW + 4;
      const minusX = plusX + BTN_H + 2;
      const allW = 40;
      const allX = btnX - allW - 4;
      const cacheW = 56, cacheX = allX - cacheW - 4;

      if (y > 20 && y < 36 && x > btnX && x < btnX + btnW) return CONTROL_DESCRIPTIONS.mode;
      if (y > BTN_Y && y < BTN_Y + BTN_H) {
        if (x > cacheX && x < cacheX + cacheW) return CONTROL_DESCRIPTIONS.cache;
        if (x > allX && x < allX + allW) return CONTROL_DESCRIPTIONS.toggleAll;
        if (x > btnX && x < btnX + btnW) return CONTROL_DESCRIPTIONS.theme;
        if (x > plusX && x < plusX + BTN_H) return CONTROL_DESCRIPTIONS.add;
        if (data.length > 1 && x > minusX && x < minusX + BTN_H) return CONTROL_DESCRIPTIONS.remove;
        const clearX = (data.length > 1 ? minusX + BTN_H : plusX + BTN_H) + 4;
        if (x > clearX && x < clearX + 36) return CONTROL_DESCRIPTIONS.clear;
      }

      const C = {
        onX: 8 * s, onW: 50 * s,
        nmX: 62 * s, nmW: 480 * s,
        stX: 548 * s, stW: 80 * s,
        vX: 635 * s, vW: 60 * s,
        aX: 702 * s, aW: 60 * s,
        rX: 770 * s, rW: 158 * s,
      };

      for (let i = 0; i < data.length; i++) {
        const ry = ROW_START + i * ROW_H;
        if (y < ry || y > ry + ROW_H) continue;
        if (x > C.onX && x < C.onX + C.onW) return CONTROL_DESCRIPTIONS.enabled;
        if (x > C.nmX && x < C.nmX + C.nmW) return CONTROL_DESCRIPTIONS.lora;
        if (x > C.stX && x < C.stX + C.stW) return CONTROL_DESCRIPTIONS.str;
        if (x > C.vX && x < C.vX + C.vW) return CONTROL_DESCRIPTIONS.video;
        if (x > C.aX && x < C.aX + C.aW) return CONTROL_DESCRIPTIONS.audio;
        if (x > C.rX && x < C.rX + C.rW) return CONTROL_DESCRIPTIONS.keys;
      }

      return "";
    };

    nodeType.prototype.onMouseMove = function (_e, local_pos) {
      setCanvasTooltip(getTooltipAt(this, local_pos));
      const action = loraActionAt(this, local_pos);
      const hovered = action?.enabled ? `${action.row}:${action.action}` : null;
      if (this._loraHoveredAction !== hovered) {
        this._loraHoveredAction = hovered;
        this.setDirtyCanvas(true);
      }
      return false;
    };

    nodeType.prototype.onMouseLeave = function () {
      setCanvasTooltip("");
      if (this._loraHoveredAction) {
        this._loraHoveredAction = null;
        this.setDirtyCanvas(true);
      }
      return false;
    };

    nodeType.prototype.drawLoraStack = function () {
      if (this.flags.collapsed || !this._loraSurface) return;
      const data = JSON.parse(this.properties.stack_data);
      const t = THEMES[this.properties.theme || "a"];
      const W = this.size[0], s = W / 1000;
      sync(this);
      syncModeWidget(this);
      syncCacheWidget(this);
      const modelType = this.properties.model_type || "Basic";
      const audioEnabled = hasSeparatedAudio(modelType);
      const surface = this._loraSurface;
      const focusedIndex = [...surface.querySelectorAll("button")].indexOf(document.activeElement);
      surface.replaceChildren();
      const layer = document.createElement("div");
      layer.style.cssText = `position:relative;width:${W}px;height:100%;transform-origin:0 0;transform:scaleX(${surface.clientWidth / W})`;
      surface.append(layer);
      const box = (tag, text, x, y, w, h, color, background, border, font = 8) => {
        const el = document.createElement(tag);
        el.textContent = text;
        el.style.cssText = `position:absolute;box-sizing:border-box;left:${x}px;top:${y}px;width:${w}px;height:${h}px;` +
          `margin:0;padding:0 2px;overflow:hidden;white-space:nowrap;font:${font}px 'Courier New',monospace;` +
          `line-height:${h - 2}px;color:${color};background:${background};border:1px solid ${border};border-radius:3px;`;
        if (tag === "button") {
          el.type = "button";
          el.style.cursor = "pointer";
          el.title = getTooltipAt(this, [x + w / 2, y + h / 2]);
        }
        layer.append(el);
        return el;
      };
      const button = (text, x, y, w, color = t.btnText, background = t.btnBg, border = t.btnBorder, h = 16) =>
        box("button", text, x, y, w, h, color, background, border);
      const btnX = (W - 110) / 2, allX = btnX - 44, plusX = btnX + 114;
      button(`MODEL: ${modelType} ▼`, (W - 150) / 2, 20, 150);
      button(data.every(row => row.on) ? "ALL✓" : "ALL", allX, 40, 40);
      button(`⬡ THEME: ${t.name} ▶`, btnX, 40, 110);
      button(this.properties.use_cache ? "⚡ CACHE ✓" : "⚡ CACHE", allX - 60, 40, 56,
        this.properties.use_cache ? t.onColor : t.arrowColor,
        this.properties.use_cache ? t.onColor + "22" : t.btnBg);
      button("+", plusX, 40, 16);
      if (data.length > 1) button("−", plusX + 18, 40, 16, "#f44336cc", t.btnBg, "#f4433655");
      button("Clear", plusX + (data.length > 1 ? 38 : 20), 40, 36, "#ffb3adcc", "#f4433622", "#f44336aa");
      box("div", "", 4, 58, W - 8, 1, t.divider, t.divider, "transparent");
      for (let i = 0; i < data.length; i++) {
        const row = data[i], y = 59 + i * 28, pY = y + 2;
        box("div", "", 4, y, W - 8, 27, t.btnText,
          t.wood ? (i % 2 ? "#321f0a" : "#2c1f0e") : (i % 2 ? t.rowEven : t.rowOdd), "transparent");
        button(row.on ? "✔" : "✖", 8 * s, pY, 50 * s, row.on ? t.onColor : t.offColor, "transparent", "transparent", 24);
        const name = row.lora === "None" ? "None" : row.lora.split(/[\\\/]/).pop().replace(/\.safetensors$/i, "");
        const picker = button(name, 62 * s, pY, 480 * s, row.lora === "None" ? t.nameEmpty : t.nameLoaded, "transparent", "transparent", 24);
        picker.style.textAlign = "left";
        for (const [label, value, x, w, color, zeroed] of [
          ["STR", row.str ?? 1, 548, 80, t.strColor, row.str === 0],
          ["VIS", row.vs ?? 1, 635, 60, t.vColor, row.vs === 0],
          ["A", row.as ?? 1, 702, 60, t.aColor, !audioEnabled || row.as === 0],
        ]) {
          const pill = button("", x * s, pY, w * s, zeroed ? color + "66" : color,
            color + (zeroed ? "12" : "1c"), color + (zeroed ? "25" : "60"), 24);
          const caption = document.createElement("span");
          caption.textContent = label;
          caption.style.cssText = "position:absolute;top:0;left:2px;font-size:5px;line-height:6px;pointer-events:none";
          const valueText = document.createElement("span");
          valueText.textContent = value.toFixed(2);
          valueText.style.cssText = "display:block;padding-top:6px;font-size:7px;font-weight:bold;pointer-events:none";
          pill.append(caption, valueText);
          for (const [arrow, side] of [["<", "left"], [">", "right"]]) {
            const span = document.createElement("span");
            span.textContent = arrow;
            span.style.cssText = `position:absolute;bottom:1px;${side}:1px;font-size:6px;line-height:7px;pointer-events:none`;
            pill.append(span);
          }
        }
        let counts = "Full LoRA map";
        if (audioEnabled) {
          counts = row.lora === "None" ? "" : "…";
          if (row.lora !== "None" && keyCache[row.lora]) {
            const {v, a} = keyCache[row.lora];
            counts = `V:${v}  A:${a}`;
          } else if (row.lora !== "None" && !keyCache[row.lora + "_p"]) {
            keyCache[row.lora + "_p"] = true;
            fetch(`/dasiwa/ltx2/keycounts?lora=${encodeURIComponent(row.lora)}&model_type=${encodeURIComponent(modelType)}`)
              .then(r => r.json())
              .then(d => { keyCache[row.lora] = {v: d.v, a: d.a}; this.setDirtyCanvas(true); })
              .catch(() => { keyCache[row.lora] = {v: "?", a: "?"}; this.setDirtyCanvas(true); });
          }
        }
        box("div", counts, 770 * s, pY, 158 * s, 24, t.nameEmpty, "transparent", "transparent", 6);
        for (const [action, x] of [["info", 934], ["trash", 966]]) {
          const enabled = row.lora !== "None", hovered = enabled && this._loraHoveredAction === `${i}:${action}`;
          const color = enabled ? (action === "info" ? t.strColor : t.offColor) : t.arrowColor;
          const control = button("", x * s, pY, 26 * s, color,
            enabled ? color + (hovered ? "40" : "18") : t.btnBg,
            enabled ? color + (hovered ? "FF" : "99") : t.arrowColor, 24);
          control.style.padding = "0";
          control.style.cursor = enabled ? "pointer" : "not-allowed";
          control.dataset.action = action;
          control.disabled = !enabled;
          control.setAttribute("aria-label", action === "info" ? "Show LoRA info" : "Clear LoRA slot");
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
          svg.setAttribute("viewBox", "-11 -12 22 24");
          svg.style.cssText = "width:100%;height:100%;pointer-events:none";
          // Fixed paths only; model filenames and other dynamic text use textContent.
          svg.innerHTML = action === "info"
            ? '<circle r="6" fill="none" stroke="currentColor"/><path d="M0 -1V3" stroke="currentColor"/><circle cy="-3.5" r=".8" fill="currentColor"/>'
            : '<path d="M-5 -4H5 M-2 -4V-6H2V-4 M-4 -1L-3 6H3L4 -1 M-1 0V3 M1 0V3" fill="none" stroke="currentColor"/>';
          svg.setAttribute("stroke-width", "1.4");
          control.append(svg);
        }
      }
      if (focusedIndex >= 0) surface.querySelectorAll("button")[focusedIndex]?.focus({preventScroll: true});
    };

    nodeType.prototype.onMouseDown = function (e, local_pos) {
      if (this.flags.collapsed) return;
      const data = JSON.parse(this.properties.stack_data);
      const [x, y] = local_pos;
      const W = this.size[0];
      const H = this.size[1];
      const s = W / 1000;
      const t = THEMES[this.properties.theme || "a"];

      const BTN_H = 16;
      const BTN_Y = 40;
      const ROW_START = BTN_Y + BTN_H + 3;
      const ROW_H = 28;

      const btnW = 110, btnX = (W - btnW) / 2;
      const plusX = btnX + btnW + 4;
      const allW = 40, allX = btnX - allW - 4;
      const cacheW = 56, cacheX = allX - cacheW - 4;

      const modeW = 150, modeX = (W - modeW) / 2;
      if (y > 20 && y < 36 && x > modeX && x < modeX + modeW) {
        new LiteGraph.ContextMenu(MODEL_TYPES, {
          event: e,
          callback: value => {
            this.properties.model_type = value;
            syncModeWidget(this);
            this.setDirtyCanvas(true);
          },
        });
        return true;
      }

      // Theme cycle
      if (y > BTN_Y && y < BTN_Y + BTN_H && x > allX && x < allX + allW) {
        const toggleAll = !data.every(row => row.on);
        data.forEach(row => { row.on = toggleAll; });
        this.properties.stack_data = JSON.stringify(data);
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }

      if (y > BTN_Y && y < BTN_Y + BTN_H && x > cacheX && x < cacheX + cacheW) {
        this.properties.use_cache = !this.properties.use_cache;
        syncCacheWidget(this);
        this.setDirtyCanvas(true);
        return true;
      }

      if (y > BTN_Y && y < BTN_Y + BTN_H && x > btnX && x < btnX + btnW) {
        const idx = THEME_ORDER.indexOf(this.properties.theme || "a");
        this.properties.theme = THEME_ORDER[(idx + 1) % THEME_ORDER.length];
        this.setDirtyCanvas(true);
        return true;
      }

      // Add row
      if (y > BTN_Y && y < BTN_Y + BTN_H && x > plusX && x < plusX + BTN_H) {
        data.push({ on: true, lora: "None", str: 1.0, vs: 1.0, as: 1.0 });
        this.properties.stack_data = JSON.stringify(data);
        this.size[1] = 40 + 16 + 3 + data.length * 28 + 2;
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }

      // Remove last row
      const minusX = plusX + BTN_H + 2;
      if (data.length > 1 && y > BTN_Y && y < BTN_Y + BTN_H && x > minusX && x < minusX + BTN_H) {
        data.pop();
        this.properties.stack_data = JSON.stringify(data);
        this.size[1] = 40 + 16 + 3 + data.length * 28 + 2;
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }

      // Clear all rows: keeps however many slots are currently showing, resets
      // each one back to None / 1.00 (the same shape a freshly-added row gets).
      const clearW = 36;
      const clearX = (data.length > 1 ? minusX + BTN_H : plusX + BTN_H) + 4;
      if (y > BTN_Y && y < BTN_Y + BTN_H && x > clearX && x < clearX + clearW) {
        const cleared = data.map(() => ({ on: true, lora: "None", str: 1.0, vs: 1.0, as: 1.0 }));
        this.properties.stack_data = JSON.stringify(cleared);
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }

      const C = {
        onX: 8 * s, onW: 50 * s,
          nmX: 62 * s, nmW: 480 * s,
          stX: 548 * s, stW: 80 * s,
          vX: 635 * s, vW: 60 * s,
          aX: 702 * s, aW: 60 * s,
      };

      for (let i = 0; i < data.length; i++) {
        const ry = ROW_START + i * ROW_H;
        if (y < ry || y > ry + ROW_H) continue;

        const action = loraActionAt(this, local_pos);
        if (action) {
          if (!action.enabled) return true;
          if (action.action === "info") {
            openLoraInfo(data[i].lora, t);
          } else {
            data[i].lora = "None";
            this.properties.stack_data = JSON.stringify(data);
            this._loraHoveredAction = null;
            sync(this);
            this.setDirtyCanvas(true);
          }
          return true;
        }

        if (x > C.onX && x < C.onX + C.onW) {
          data[i].on = !data[i].on;
        } else if (x > C.nmX && x < C.nmX + C.nmW) {
          getCurrentLoraList(nodeData).then(loraList => {
            const menu = new LiteGraph.ContextMenu(loraList, {
              event: e, scale: 1.2,
              callback: v => {
                data[i].lora = v;
                this.properties.stack_data = JSON.stringify(data);
                sync(this);
                this.setDirtyCanvas(true);
              }
            });
            const sw = document.createElement("div");
            sw.style = "padding:5px;background:#333;border-bottom:1px solid #555;";
            const inp = document.createElement("input");
            inp.style = "width:100%;background:#222;color:#00FFCC;border:1px solid #444;padding:4px;";
            inp.placeholder = "Search LoRAs...";
            sw.appendChild(inp);
            menu.root.prepend(sw);
            setTimeout(() => inp.focus(), 10);
            inp.addEventListener("input", ev => {
              const term = ev.target.value.toLowerCase();
              menu.root.querySelectorAll(".litemenu-entry").forEach(el => {
                el.style.display = el.textContent.toLowerCase().includes(term) ? "block" : "none";
              });
            });
          });
          return true;
        } else if (x > C.stX && x < C.stX + C.stW) {
          if (x < C.stX + 14 * s) data[i].str = bumpS(data[i].str, -0.05);
          else if (x > C.stX + C.stW - 14 * s) data[i].str = bumpS(data[i].str, 0.05);
          else {
            openValueEditor({
              node: this,
              anchor: [C.stX + C.stW, ry + 2],
              clientX: e.clientX,
              clientY: e.clientY,
              label: "LoRA Strength (-5.0 to 5.0)",
              value: data[i].str ?? 1,
              min: -5,
              max: 5,
              step: 0.01,
              theme: t,
              onCommit: v => {
                data[i].str = Math.round(v * 100) / 100;
                this.properties.stack_data = JSON.stringify(data);
                sync(this);
                this.setDirtyCanvas(true);
              },
            });
            return true;
          }
        } else if (x > C.vX && x < C.vX + C.vW) {
          if (x < C.vX + 10 * s) data[i].vs = bump(data[i].vs, -0.05);
          else if (x > C.vX + C.vW - 10 * s) data[i].vs = bump(data[i].vs, 0.05);
          else {
            openValueEditor({
              node: this,
              anchor: [C.vX + C.vW, ry + 2],
              clientX: e.clientX,
              clientY: e.clientY,
              label: "Visual Multiplier (0.0 to 2.0)",
              value: data[i].vs ?? 1,
              min: 0,
              max: 2,
              step: 0.01,
              theme: t,
              onCommit: v => {
                data[i].vs = Math.round(v * 100) / 100;
                this.properties.stack_data = JSON.stringify(data);
                sync(this);
                this.setDirtyCanvas(true);
              },
            });
            return true;
          }
        } else if (hasSeparatedAudio(this.properties.model_type) && x > C.aX && x < C.aX + C.aW) {
          if (x < C.aX + 10 * s) data[i].as = bump(data[i].as, -0.05);
          else if (x > C.aX + C.aW - 10 * s) data[i].as = bump(data[i].as, 0.05);
          else {
            openValueEditor({
              node: this,
              anchor: [C.aX + C.aW, ry + 2],
              clientX: e.clientX,
              clientY: e.clientY,
              label: "Audio Multiplier (0.0 to 2.0)",
              value: data[i].as ?? 1,
              min: 0,
              max: 2,
              step: 0.01,
              theme: t,
              onCommit: v => {
                data[i].as = Math.round(v * 100) / 100;
                this.properties.stack_data = JSON.stringify(data);
                sync(this);
                this.setDirtyCanvas(true);
              },
            });
            return true;
          }
        }

        this.properties.stack_data = JSON.stringify(data);
        sync(this);
        this.setDirtyCanvas(true);
        return true;
      }
    };
  }
});
