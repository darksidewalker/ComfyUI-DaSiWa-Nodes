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

function buildLoraInfoPanelHtml(info, theme, civitaiDomain) {
  const t = theme || THEMES.a;
  const esc = escHtml;
  const words = (info.trainedWords || []).filter(w => w && w.word);
  const images = (info.images || []).filter(im => im && im.url);
  const btnStyle =
    "background:" + t.btnBg + ";border:1px solid " + t.btnBorder + ";color:" + t.btnText +
    ";padding:2px 8px;border-radius:3px;cursor:pointer;font:11px 'Courier New',monospace;";

  const out = [];
  out.push(
    '<div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;">' +
      '<span style="font:11px \'Courier New\',monospace;opacity:.8;overflow-wrap:anywhere;">' + esc(info.file) + "</span>" +
      '<button data-action="refresh" title="Re-fetch from Civitai" style="' + btnStyle + '">Refresh</button>' +
      (words.length ? '<button data-action="copy-words" style="' + btnStyle + '">Copy all</button>' : "") +
      (words.length ? '<button data-action="copy-selected" style="' + btnStyle + '">Copy selected</button>' : "") +
      '<button data-action="close" title="Close" style="' + btnStyle + ';margin-left:auto;">×</button>' +
    "</div>"
  );
  if (info.sha256) {
    out.push('<div style="font:9px \'Courier New\',monospace;opacity:.5;">sha256 ' + esc(info.sha256.slice(0, 16)) + "</div>");
  }
  if (info.civitaiFound) {
    const rawLink = (info.links || []).find(l => typeof l === "string" && (l.includes("civitai.com") || l.includes("civitai.red"))) || "";
    const link = civitaiDomain === "red" ? rawLink.replace("civitai.com", "civitai.red") : rawLink.replace("civitai.red", "civitai.com");
    out.push(
      '<div style="margin-top:6px;font:12px \'Courier New\',monospace;">' +
        '<div style="font-weight:bold;">' + esc(info.name || info.file) + "</div>" +
        (info.type || info.baseModel
          ? '<div style="opacity:.6;">' + esc([info.type, info.baseModel].filter(Boolean).join(" · ")) + "</div>"
          : "") +
        (link ? '<a href="' + esc(link) + '" target="_blank" rel="noreferrer" style="color:' + t.btnText + ';">' + esc(link) + "</a>" : "") +
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

function openLoraInfo(lora, theme, civitaiDomain) {
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
    box.innerHTML = buildLoraInfoPanelHtml(info, t, civitaiDomain);
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

function drawWoodRow(ctx, x, y, w, h, seed) {
  ctx.fillStyle = seed % 2 === 0 ? "#2c1f0e" : "#321f0a";
  ctx.fillRect(x, y, w, h);
  ctx.save();
  ctx.globalAlpha = 0.07;
  for (let i = 0; i < 12; i++) {
    const lx = x + ((seed * 37 + i * 61) % w);
    const curve = ((seed * 13 + i * 7) % 6) - 3;
    ctx.strokeStyle = i % 3 === 0 ? "#ffcc66" : "#8B5E20";
    ctx.lineWidth = i % 4 === 0 ? 1.5 : 0.5;
    ctx.beginPath();
    ctx.moveTo(lx, y);
    ctx.bezierCurveTo(lx + curve, y + h * 0.3, lx - curve, y + h * 0.7, lx + curve * 0.5, y + h);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPill(ctx, t, x, y, w, h, label, val, color, zeroed) {
  ctx.fillStyle = color + (zeroed ? "12" : "1c");
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 2);
  ctx.fill();
  ctx.strokeStyle = color + (zeroed ? "25" : "60");
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // Left arrow
  ctx.fillStyle = zeroed ? t.arrowColor + "55" : t.arrowColor + "dd";
  ctx.font = "6px 'Courier New',monospace";
  ctx.textAlign = "left";
  ctx.fillText("<", x + 2, y + h - 2);

  // Right arrow
  ctx.textAlign = "right";
  ctx.fillText(">", x + w - 2, y + h - 2);

  // Label top-left
  ctx.fillStyle = color + (zeroed ? "55" : "88");
  ctx.font = "5px 'Courier New',monospace";
  ctx.textAlign = "left";
  ctx.fillText(label, x + 2, y + 4);

  // Value center
  ctx.fillStyle = zeroed ? color + "66" : color;
  ctx.font = "bold 7px 'Courier New',monospace";
  ctx.textAlign = "center";
  ctx.fillText(val.toFixed(2), x + w / 2, y + h / 2 + 2);
  ctx.textAlign = "left";
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
      if (!["com", "red"].includes(this.properties.civitai_domain)) this.properties.civitai_domain = "com";
      if (this.properties.use_cache === undefined) this.properties.use_cache = false;   // default off
      if (!MODEL_TYPES.includes(this.properties.model_type)) this.properties.model_type = "Basic";
      const rows = JSON.parse(this.properties.stack_data);
      this.size = [720, calcHeight(rows.length)];

      const hideWidget = widget => {
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
      this.widgets = [hideWidget(stackWidget), hideWidget(modeWidget), hideWidget(cacheWidget)];
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
        rX: 770 * s, rW: W - 770 * s - 8,
        iX: 952 * s, iW: 18 * s,
        tX: 974 * s, tW: 18 * s,
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
        if (x > C.iX && x < C.iX + C.iW) return "Show LoRA info (Civitai, trigger words, images)";
        if (x > C.tX && x < C.tX + C.tW) return "Trash this LoRA (set the slot back to None)";
      }

      return "";
    };

    nodeType.prototype.onMouseMove = function (_e, local_pos) {
      setCanvasTooltip(getTooltipAt(this, local_pos));
      return false;
    };

    nodeType.prototype.onMouseLeave = function () {
      setCanvasTooltip("");
      return false;
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      if (this.flags.collapsed) return;
      const data = JSON.parse(this.properties.stack_data);
      const t = THEMES[this.properties.theme || "a"];
      const W = this.size[0];
      const H = this.size[1];
      sync(this);
      syncModeWidget(this);
      syncCacheWidget(this);
      const modelType = this.properties.model_type || "Basic";
      const audioEnabled = hasSeparatedAudio(modelType);

      const s = W / 1000;

      const BTN_H = 16;
      const BTN_Y = 40;
      const ROW_START = BTN_Y + BTN_H + 3;
      const ROW_H = 28;

      const C = {
        onX: 8 * s, onW: 50 * s,
        nmX: 62 * s, nmW: 480 * s,
        stX: 548 * s, stW: 80 * s,
        vX: 635 * s, vW: 60 * s,
        aX: 702 * s, aW: 60 * s,
        rX: 770 * s, rW: W - 770 * s - 8,
        iX: 952 * s, iW: 18 * s,
        tX: 974 * s, tW: 18 * s,
      };

      const modeW = 150, modeX = (W - modeW) / 2;
      const civW = 80, civX = modeX - civW - 6;
      const civitaiDomain = this.properties.civitai_domain || "com";
      const civColors = civitaiDomain === "red"
        ? { bg: "rgba(220,50,50,.22)", border: "rgba(255,90,90,.9)", text: "#ffd6d6" }
        : { bg: "rgba(35,120,220,.22)", border: "rgba(70,150,255,.9)", text: "#d3e8ff" };
      ctx.fillStyle = civColors.bg;
      ctx.beginPath();
      ctx.roundRect(civX, 20, civW, 16, 3);
      ctx.fill();
      ctx.strokeStyle = civColors.border;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = civColors.text;
      ctx.font = "bold 7px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText(`Civitai: .${civitaiDomain}`, civX + civW / 2, 30);
      ctx.textAlign = "left";
      ctx.fillStyle = t.btnBg;
      ctx.beginPath();
      ctx.roundRect(modeX, 20, modeW, 16, 3);
      ctx.fill();
      ctx.strokeStyle = t.btnBorder;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = t.btnText;
      ctx.font = "bold 7px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText(`MODEL: ${modelType} ▼`, modeX + modeW / 2, 30);
      ctx.textAlign = "left";

      // Theme button
      const btnW = 110, btnX = (W - btnW) / 2;
      const allW = 40, allX = btnX - allW - 4;
      const allEnabled = data.every(row => row.on);
      ctx.fillStyle = t.btnBg;
      ctx.beginPath();
      ctx.roundRect(allX, BTN_Y, allW, BTN_H, 3);
      ctx.fill();
      ctx.strokeStyle = t.btnBorder;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = t.btnText;
      ctx.font = "bold 7px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText(allEnabled ? "ALL✓" : "ALL", allX + allW / 2, BTN_Y + 10);
      ctx.textAlign = "left";
      ctx.fillStyle = t.btnBg;
      ctx.beginPath();
      ctx.roundRect(btnX, BTN_Y, btnW, BTN_H, 3);
      ctx.fill();
      ctx.strokeStyle = t.btnBorder;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = t.btnText;
      ctx.font = "bold 7px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText(`⬡ THEME: ${t.name} ▶`, btnX + btnW / 2, BTN_Y + 10);
      ctx.textAlign = "left";

      // Cache toggle (opt-in, default off)
      const cacheW = 56, cacheX = allX - cacheW - 4;
      ctx.fillStyle = this.properties.use_cache ? t.onColor + "22" : t.btnBg;
      ctx.beginPath();
      ctx.roundRect(cacheX, BTN_Y, cacheW, BTN_H, 3);
      ctx.fill();
      ctx.strokeStyle = this.properties.use_cache ? t.onColor : t.arrowColor;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = this.properties.use_cache ? t.onColor : t.arrowColor;
      ctx.font = "bold 7px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText(this.properties.use_cache ? "⚡ CACHE ✓" : "⚡ CACHE", cacheX + cacheW / 2, BTN_Y + 10);
      ctx.textAlign = "left";

      // + button
      const plusX = btnX + btnW + 4;
      ctx.fillStyle = t.btnBg;
      ctx.beginPath();
      ctx.roundRect(plusX, BTN_Y, BTN_H, BTN_H, 3);
      ctx.fill();
      ctx.strokeStyle = t.btnBorder;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = t.btnText;
      ctx.font = "bold 11px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText("+", plusX + BTN_H / 2, BTN_Y + 11);
      ctx.textAlign = "left";

      // - button
      if (data.length > 1) {
        const minusX = plusX + BTN_H + 2;
        ctx.fillStyle = t.btnBg;
        ctx.beginPath();
        ctx.roundRect(minusX, BTN_Y, BTN_H, BTN_H, 3);
        ctx.fill();
        ctx.strokeStyle = "#f4433655";
        ctx.lineWidth = 0.5;
        ctx.stroke();
        ctx.fillStyle = "#f44336cc";
        ctx.font = "bold 11px 'Courier New',monospace";
        ctx.textAlign = "center";
        ctx.fillText("−", minusX + BTN_H / 2, BTN_Y + 11);
        ctx.textAlign = "left";
      }

      // Clear button -- resets every current row to None / 1.00, keeps row count
      const clearW = 36;
      const clearX = (data.length > 1 ? plusX + BTN_H + 2 + BTN_H : plusX + BTN_H) + 4;
      ctx.fillStyle = "#f4433622";
      ctx.beginPath();
      ctx.roundRect(clearX, BTN_Y, clearW, BTN_H, 3);
      ctx.fill();
      ctx.strokeStyle = "#f44336aa";
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.fillStyle = "#ffb3adcc";
      ctx.font = "bold 8px 'Courier New',monospace";
      ctx.textAlign = "center";
      ctx.fillText("Clear", clearX + clearW / 2, BTN_Y + 11);
      ctx.textAlign = "left";

      // Divider
      ctx.strokeStyle = t.divider;
      ctx.lineWidth = 0.3;
      ctx.beginPath();
      ctx.moveTo(4, ROW_START - 1);
      ctx.lineTo(W - 4, ROW_START - 1);
      ctx.stroke();

      // Rows
      for (let i = 0; i < data.length; i++) {
        const ry = ROW_START + i * ROW_H;
        const row = data[i];
        const str = row.str ?? 1.0;
        const vs = row.vs ?? 1.0;
        const as_ = row.as ?? 1.0;
        const pH = Math.max(ROW_H - 4, 14);
        const pY = ry + (ROW_H - pH) / 2;

        if (t.wood) {
          drawWoodRow(ctx, 4, ry, W - 8, ROW_H - 1, i);
        } else {
          ctx.fillStyle = i % 2 === 0 ? t.rowOdd : t.rowEven;
          ctx.fillRect(4, ry, W - 8, ROW_H - 1);
        }

        ctx.font = "bold 8px 'Courier New',monospace";
        ctx.fillStyle = row.on ? t.onColor : t.offColor;
        ctx.fillText(row.on ? "✔" : "✖", C.onX, ry + ROW_H / 2 + 2);

        ctx.font = "8px 'Courier New',monospace";
        ctx.fillStyle = row.lora === "None" ? t.nameEmpty : t.nameLoaded;
        const nm = row.lora === "None" ? "None"
          : row.lora.split(/[\\\/]/).pop().replace(/\.safetensors$/i, "").substring(0, 48);
        ctx.fillText(nm, C.nmX, ry + ROW_H / 2 + 2);

        drawPill(ctx, t, C.stX, pY, C.stW, pH, "STR", str, t.strColor, str === 0);
        drawPill(ctx, t, C.vX, pY, C.vW, pH, "VIS", vs, t.vColor, vs === 0);
        drawPill(ctx, t, C.aX, pY, C.aW, pH, "A", as_, t.aColor, !audioEnabled || as_ === 0);

        if (!audioEnabled) {
          ctx.font = "6px 'Courier New',monospace";
          ctx.fillStyle = t.nameEmpty;
          ctx.fillText(
            "Full LoRA map",
            C.rX, ry + ROW_H / 2 + 2,
          );
        } else if (row.lora !== "None" && keyCache[row.lora]) {
          const { v, a } = keyCache[row.lora];
          ctx.font = "6px 'Courier New',monospace";
          ctx.fillStyle = t.ratioV;
          ctx.fillText(`V:${v}`, C.rX, ry + ROW_H / 2 - 2);
          ctx.fillStyle = t.ratioA;
          ctx.fillText(`A:${a}`, C.rX, ry + ROW_H / 2 + 5);
        } else if (row.lora !== "None") {
          ctx.font = "6px 'Courier New',monospace";
          ctx.fillStyle = t.nameEmpty;
          ctx.textAlign = "center";
          ctx.fillText("…", C.rX + C.rW / 2, ry + ROW_H / 2 + 1);
          ctx.textAlign = "left";
          if (!keyCache[row.lora + "_p"]) {
            keyCache[row.lora + "_p"] = true;
            fetch(`/dasiwa/ltx2/keycounts?lora=${encodeURIComponent(row.lora)}&model_type=${encodeURIComponent(modelType)}`)
              .then(r => r.json())
              .then(d => { keyCache[row.lora] = { v: d.v, a: d.a }; app.graph.setDirtyCanvas(true); })
              .catch(() => { keyCache[row.lora] = { v: "?", a: "?" }; });
          }
        }

        // Info button (row right edge): drawn circle + "i"; opens the
        // Civitai/trigger-words panel.
        {
          const infoCol = row.lora !== "None" ? t.strColor + "AA" : t.nameEmpty;
          const cx = C.iX + C.iW / 2, cy = ry + ROW_H / 2;
          const r = 8.5 * s;
          ctx.strokeStyle = infoCol;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = infoCol;
          ctx.font = "bold " + Math.max(4, Math.round(9 * s)) + "px 'Courier New',monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("i", cx, cy + 0.5 * s);
          ctx.textBaseline = "alphabetic";
          ctx.textAlign = "left";
        }

        // Trash button (right of info): drawn trash-can shape; resets this
        // slot back to "None".
        {
          const trCol = row.lora !== "None" ? t.offColor + "CC" : t.nameEmpty;
          const cx = C.tX + C.tW / 2, cy = ry + ROW_H / 2;
          ctx.strokeStyle = trCol;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          // lid + handle
          ctx.moveTo(cx - 5.8 * s, cy - 5 * s);
          ctx.lineTo(cx + 5.8 * s, cy - 5 * s);
          ctx.moveTo(cx, cy - 6.4 * s);
          ctx.lineTo(cx, cy - 5 * s);
          // body
          ctx.moveTo(cx - 4.4 * s, cy - 2.9 * s);
          ctx.lineTo(cx + 4.4 * s, cy - 2.9 * s);
          ctx.lineTo(cx + 3.2 * s, cy + 5.8 * s);
          ctx.lineTo(cx - 3.2 * s, cy + 5.8 * s);
          ctx.closePath();
          // ribs
          ctx.moveTo(cx - 1.5 * s, cy - 1.6 * s);
          ctx.lineTo(cx - 1 * s, cy + 3.8 * s);
          ctx.moveTo(cx + 1.5 * s, cy - 1.6 * s);
          ctx.lineTo(cx + 1 * s, cy + 3.8 * s);
          ctx.stroke();
        }
      }
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
      const civW = 80, civX = modeX - civW - 6;
      if (y > 20 && y < 36 && x > civX && x < civX + civW) {
        this.properties.civitai_domain = (this.properties.civitai_domain || "com") === "com" ? "red" : "com";
        this.setDirtyCanvas(true);
        return true;
      }
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
          iX: 952 * s, iW: 18 * s,
          tX: 974 * s, tW: 18 * s,
      };

      for (let i = 0; i < data.length; i++) {
        const ry = ROW_START + i * ROW_H;
        if (y < ry || y > ry + ROW_H) continue;

        // Info glyph (row right edge)
        if (x > C.iX && x < C.iX + C.iW && data[i].lora !== "None") {
          openLoraInfo(data[i].lora, t, this.properties.civitai_domain || "com");
          return true;
        }

        // Trash button (right of info): reset this slot back to "None".
        if (x > C.tX && x < C.tX + C.tW && data[i].lora !== "None") {
          data[i].lora = "None";
          this.properties.stack_data = JSON.stringify(data);
          sync(this);
          this.setDirtyCanvas(true);
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
