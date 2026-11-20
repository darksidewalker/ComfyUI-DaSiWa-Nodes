import { app } from "../../scripts/app.js";

/**
 * Seed Control — the MiniMax H3 Director's seed panel as a standalone
 * node. Mirrors the Director UX: SEED field with an attached ▲/▼ spinner
 * (Pixaroma seed control style), a single Random|Fixed switch, New roll,
 * Use Last, Last-10 history with copy actions, "External seed connected"
 * note, and the before-queue auto-roll in Random mode.
 *
 * Layout (responsive panel):
 *   wide   — seed field, Random|Fixed + New, and Use Last + history share one line
 *   narrow — controls wrap from right to left below the seed as node width shrinks
 *
 * Backing widgets (hidden native widgets, created by the backend):
 *   seed_value        INT    — the effective local seed (0..2^64-1)
 *   seed_control_state STRING — JSON {mode, last_seed, recent[]}
 * External override:
 *   seed (INT, forceInput) — when linked, local controls are replaced by
 *   the external note and the connected value passes through.
 *
 * The native INT widget coerces its value to a JS number (lossy at 16
 * digits), so the panel keeps its own lossless display seed
 * (`lastSeedText`, a decimal string) as the source of truth for display
 * and stepping — the same architecture as Pixaroma's properties-based
 * seed. The widget is written as a mirror only, never read back.
 * Stepping, typing, and the switch all commit in place (no panel
 * rebuild), so a long spinner hold or a mid-commit mode click never
 * destroys the control under the pointer.
 *
 * Every top-level function and constant carries the dasiwaSeedControl /
 * DASIWASEED prefix so names stay unique across the pack's JS files.
 */

const DASIWASEED_MAX_SEED = 0xffffffffffffffffn;
const DASIWASEED_NODE_TYPES = new Set(["DaSiWa_SeedControl"]);
const DASIWASEED_STATE_WIDGET = "seed_control_state";
const DASIWASEED_VALUE_WIDGET = "seed_value";
// The panel starts compact, then its three control groups share one line
// once their minimum widths fit. While shrinking, the rightmost groups wrap
// one at a time below the seed field.
const DASIWASEED_MIN_PANEL_WIDTH = 240;
const DASIWASEED_TWO_LINE_PANEL_WIDTH = 384;
const DASIWASEED_WIDE_PANEL_WIDTH = 628;
const DASIWASEED_COMPUTE_SIZE_WIDTH = DASIWASEED_MIN_PANEL_WIDTH + 40;

// Field surfaces follow ComfyUI's own widget theme (--comfy-input-bg /
// --border-color / --input-text) with neutral dark fallbacks, so the
// controls read as standard ComfyUI fields regardless of the theme.
let dasiwaSeedControlCssInstalled = false;

function dasiwaSeedControlInstallStyles() {
  if (dasiwaSeedControlCssInstalled) return;
  dasiwaSeedControlCssInstalled = true;
  const style = document.createElement("style");
  style.textContent = `
    .ds-seed{box-sizing:border-box;width:100%;min-width:${DASIWASEED_MIN_PANEL_WIDTH}px;background:transparent;font:12px system-ui,sans-serif;display:flex;flex-direction:column;gap:4px;padding:2px;container-type:inline-size}
    .ds-seed-control{display:flex;flex-wrap:wrap;align-items:stretch;gap:4px;width:100%}
    .ds-seed-row{display:flex;align-items:center;gap:5px;min-width:0;flex:1 1 190px}
    .ds-seed-control > .ds-seed-row:first-child{flex:2 1 240px}
    .ds-seed-btn{width:100%;display:flex;align-items:center;gap:8px;box-sizing:border-box;height:42px;padding:0 9px;background:var(--comfy-input-bg,#222);border:1px solid var(--border-color,#4e4e4e);border-radius:5px;color:var(--input-text,#ddd);font:12px system-ui,sans-serif;cursor:pointer;text-align:left;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}
    .ds-seed-btn:hover:not(:disabled){background:var(--button-hover-surface,#262729);border-color:var(--border-color,#4e4e4e);box-shadow:0 0 9px rgba(0,0,0,.35)}
    .ds-seed-btn:focus-visible{outline:none;border-color:var(--border-color,#4e4e4e);box-shadow:0 0 0 2px rgba(255,255,255,.18)}
    .ds-seed-btn:disabled{opacity:.4;cursor:not-allowed}
    .ds-seed-num{width:100%;box-sizing:border-box;height:42px;padding:0 8px;background:var(--comfy-input-bg,#222);border:1px solid var(--border-color,#4e4e4e);border-radius:5px;color:var(--input-text,#ddd);font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:16px;text-align:center;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}
    .ds-seed-num:hover:not(:disabled){background:var(--button-hover-surface,#262729);border-color:var(--border-color,#4e4e4e)}
    .ds-seed-num:focus{outline:none;border-color:var(--border-color,#4e4e4e);box-shadow:0 0 0 2px rgba(255,255,255,.18)}
    .ds-seed-num:disabled{opacity:.4;cursor:not-allowed}
    /* Number field + stacked ▲/▼ spinner sit side-by-side, the spinner
       attached flush to the input's right edge (Pixaroma seed control). */
    .ds-seed-numwrap{display:flex;align-items:stretch;flex:1;min-width:0}
    .ds-seed-numwrap .ds-seed-num{flex:1;min-width:0;width:auto;border-top-right-radius:0;border-bottom-right-radius:0}
    .ds-seed-spin{flex:0 0 26px;box-sizing:border-box;display:flex;flex-direction:column;border:1px solid var(--border-color,#4e4e4e);border-left:none;border-radius:0 5px 5px 0;overflow:hidden}
    .ds-seed-spinbtn{flex:1;display:flex;align-items:center;justify-content:center;border:none;background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);font:10px system-ui,sans-serif;cursor:pointer;padding:0;user-select:none;appearance:none;-webkit-appearance:none;transition:background .08s,color .08s}
    .ds-seed-spinbtn+.ds-seed-spinbtn{border-top:1px solid var(--border-color,#4e4e4e)}
    .ds-seed-spinbtn:hover{background:var(--button-hover-surface,#262729);color:var(--input-text,#ddd)}
    /* Random|Fixed as ONE switch: a single segmented pill with two segments.
       Kippschalter look — no outer frame and no inner divider line, just a
       subtle track; the active segment is filled green. */
    .ds-seed-switch{flex:0 0 52%;box-sizing:border-box;display:flex;align-items:stretch;gap:0;background:rgba(255,255,255,.05);border:none;border-radius:6px;padding:2px;height:42px;overflow:hidden}
    .ds-seed-switch-seg{flex:1;display:flex;align-items:center;justify-content:center;box-sizing:border-box;border:none;background:transparent;color:var(--input-text,#ddd);font:12px system-ui,sans-serif;cursor:pointer;padding:0 4px;border-radius:4px;white-space:nowrap;user-select:none;appearance:none;-webkit-appearance:none;transition:background .16s ease,color .16s ease}
    .ds-seed-switch-seg:hover{background:rgba(255,255,255,.08);color:var(--input-text,#ddd)}
    .ds-seed-switch-seg.active{background:#0f2a1a;color:#7ee19d;font-weight:600}
  `;
  document.head.appendChild(style);
}

function dasiwaSeedControlParseState(raw) {
  let parsed = {};
  if (typeof raw === "string" && raw.trim()) {
    try { parsed = JSON.parse(raw); } catch { parsed = {}; }
  } else if (raw && typeof raw === "object") parsed = raw;
  const state = { mode: "random", last_seed: null, recent: [], ...(parsed && typeof parsed === "object" ? parsed : {}) };
  state.mode = state.mode === "fixed" ? "fixed" : "random";
  state.recent = Array.isArray(state.recent) ? state.recent.map(String).filter(value => /^\d+$/.test(value)).slice(0, 10) : [];
  if (state.last_seed != null && !/^\d+$/.test(String(state.last_seed))) state.last_seed = null;
  else if (state.last_seed != null) state.last_seed = String(state.last_seed);
  return state;
}

function dasiwaSeedControlInstall(node) {
  if (node.__dasiwaSeedInstalled) return;
  node.__dasiwaSeedInstalled = true;
  dasiwaSeedControlInstallStyles();

  const dasiwaSeedControlSeedWidget = () => node.widgets?.find(w => w.name === DASIWASEED_VALUE_WIDGET);
  const dasiwaSeedControlStateWidget = () => node.widgets?.find(w => w.name === DASIWASEED_STATE_WIDGET);
  if (!dasiwaSeedControlSeedWidget() || !dasiwaSeedControlStateWidget()) return;
  let controlState = dasiwaSeedControlParseState(dasiwaSeedControlStateWidget().value);
  // Lossless display seed (decimal string). The native INT widget can
  // coerce 16-digit values back to a lossy JS number, so the panel keeps
  // its own source of truth and only WRITES the widget as a mirror.
  let lastSeedText = controlState.last_seed || String(dasiwaSeedControlSeedWidget().value ?? 0);

  const dasiwaSeedControlHasExternalSeed = () => node.inputs?.find(i => i.name === "seed")?.link != null;
  let lastExternalSeedLinked = dasiwaSeedControlHasExternalSeed();
  // Persistent panel element references. The panel DOM is built once; a
  // queue-time roll refreshes it in place instead of rebuilding, because a
  // rebuild would collapse an expanded Last-10 list and flicker the
  // controls while the user is on the Run button.
  let panelInput = null;
  let panelList = null;
  let panelHistory = null;
  let panelLastBtn = null;
  let panelFitSeedFont = () => {};
  let panelHistoryRowBuilder = null;

  const dasiwaSeedControlEmit = () => {
    const widget = dasiwaSeedControlStateWidget();
    widget.value = JSON.stringify(controlState);
    widget.callback?.(widget.value);
    node.graph?.setDirtyCanvas(true, true);
  };

  const dasiwaSeedControlWriteSeedMirror = text => {
    const widget = dasiwaSeedControlSeedWidget();
    if (!widget) return;
    widget.value = text;
    widget.callback?.(text);
    node.setDirtyCanvas?.(true, true);
  };

  const dasiwaSeedControlRollSeed = () => {
    const words = crypto.getRandomValues(new Uint32Array(2));
    return ((BigInt(words[0]) << 32n) | BigInt(words[1])).toString();
  };

  // A Random-mode node with no local seed starts with a fresh roll, mirrored
  // into the hidden widget so the widget value and the panel never disagree.
  if (controlState.mode === "random" && (!lastSeedText || lastSeedText === "0")) { lastSeedText = dasiwaSeedControlRollSeed(); dasiwaSeedControlWriteSeedMirror(lastSeedText); }

  const dasiwaSeedControlBuildSeedPanel = () => {
    const seedControl = document.createElement("div"); seedControl.className = "ds-seed-control"; seedControl.style.cssText = "width:100%";
    if (dasiwaSeedControlHasExternalSeed()) {
      const external = document.createElement("span"); external.textContent = "External seed connected"; external.style.cssText = "font-size:11px;color:#9fb3c2;white-space:nowrap"; seedControl.append(external);
      return seedControl;
    }
    const maxSeed = DASIWASEED_MAX_SEED;
    const dasiwaSeedControlRemember = value => { controlState.last_seed = value; controlState.recent = [value, ...controlState.recent.filter(entry => entry !== value)].slice(0, 10); };
    // Re-fit the seed font after a value change on the PERSISTENT input
    // (typing / stepping don't rebuild the panel). Deferred so it runs
    // after layout.
    const dasiwaSeedControlRefitFont = () => requestAnimationFrame(() => dasiwaSeedControlFitSeedFont());

    // Row 1: seed field with an attached ▲/▼ spinner on the input's right
    // edge (Pixaroma seed control style): stacked ▲/▼ column, no heading —
    // the node title already says Seed Control. Hold-to-repeat included.
    // Stepping edits the lossless `lastSeedText` in place (no rebuild) so
    // the spinner is never destroyed mid-hold.
    const input = document.createElement("input"); input.type = "text"; input.inputMode = "numeric"; input.className = "ds-seed-num"; input.style.cssText = "flex:1;min-width:0;text-align:center"; input.value = lastSeedText; input.title = "Unsigned 64-bit seed";
    panelInput = input;
    input.onchange = event => { const digits = String(event.target.value ?? "").replace(/\D+/g, ""); const value = digits === "" ? "0" : BigInt(digits); if (value < 0n || value > maxSeed) { event.target.value = lastSeedText; return; } lastSeedText = value.toString(); dasiwaSeedControlWriteSeedMirror(lastSeedText); controlState.mode = "fixed"; dasiwaSeedControlRemember(lastSeedText); dasiwaSeedControlEmit(); dasiwaSeedControlSyncSwitch(); dasiwaSeedControlRefitFont(); };
    const dasiwaSeedControlStepSeed = delta => { let value = BigInt(lastSeedText || "0") + delta; if (value < 0n) value = maxSeed; else if (value > maxSeed) value = 0n; lastSeedText = value.toString(); input.value = lastSeedText; dasiwaSeedControlWriteSeedMirror(lastSeedText); controlState.mode = "fixed"; dasiwaSeedControlRemember(lastSeedText); dasiwaSeedControlEmit(); dasiwaSeedControlSyncSwitch(); dasiwaSeedControlRefitFont(); };
    const numwrap = document.createElement("div"); numwrap.className = "ds-seed-numwrap";
    const spin = document.createElement("div"); spin.className = "ds-seed-spin";
    // Shrink the seed font until the digits fit the field (a 16-digit seed
    // can overflow the input at the base size). Idempotent and cheap, like
    // Pixaroma's fitSeedFont; safe to call repeatedly.
    const dasiwaSeedControlFitSeedFont = () => { if (!input.isConnected) return; const MAX = 16, MIN = 11; input.style.fontSize = MAX + "px"; if (!input.clientWidth) return; let fs = MAX, guard = 0; while (fs > MIN && input.scrollWidth > input.clientWidth && guard++ < 24) { fs -= 1; input.style.fontSize = fs + "px"; } };
    panelFitSeedFont = dasiwaSeedControlFitSeedFont;
    // Press-and-hold auto-repeat: one step on press, then repeats every 80 ms
    // after a 400 ms hold; self-cleans on pointerup / cancel / leave.
    const dasiwaSeedControlBindHoldRepeat = (button, step) => { button.addEventListener("pointerdown", event => { if (event.button != null && event.button !== 0) return; event.preventDefault(); event.stopPropagation(); step(); let interval = null; const timeout = setTimeout(() => { interval = setInterval(step, 80); }, 400); const end = () => { clearTimeout(timeout); if (interval) clearInterval(interval); window.removeEventListener("pointerup", end, true); window.removeEventListener("pointercancel", end, true); button.removeEventListener("pointerleave", end); }; window.addEventListener("pointerup", end, true); window.addEventListener("pointercancel", end, true); button.addEventListener("pointerleave", end); }); };
    const dasiwaSeedControlSpinButton = (glyph, title, delta) => { const button = document.createElement("button"); button.type = "button"; button.className = "ds-seed-spinbtn"; button.textContent = glyph; button.title = title; button.tabIndex = -1; dasiwaSeedControlBindHoldRepeat(button, () => dasiwaSeedControlStepSeed(delta)); return button; };
    const upBtn = dasiwaSeedControlSpinButton("▲", "Seed +1 (wraps at 2^64-1, locks Fixed). Hold to repeat.", 1n);
    const downBtn = dasiwaSeedControlSpinButton("▼", "Seed -1 (wraps at 0, locks Fixed). Hold to repeat.", -1n);
    spin.append(upBtn, downBtn); numwrap.append(input, spin);
    const fieldRow = document.createElement("div"); fieldRow.className = "ds-seed-row"; fieldRow.append(numwrap);
    // Fit the seed font once the row is laid out (the input has 0 width
    // before it attaches to the canvas), covering the initial value.
    requestAnimationFrame(() => dasiwaSeedControlFitSeedFont());

    // Row 2: one Random|Fixed switch (Pixaroma style) + New roll. The
    // switch flips a single control: Random rolls on every queue, Fixed
    // keeps the current seed. Switching to Fixed locks the seed the field
    // currently shows. Updated in place (no rebuild) so a click on a
    // segment while the number field is committing never destroys it.
    const modeRow = document.createElement("div"); modeRow.className = "ds-seed-row";
    const switchEl = document.createElement("div"); switchEl.className = "ds-seed-switch"; switchEl.role = "group"; switchEl.title = "Seed mode: Random rolls a new seed on every queue, Fixed keeps the current seed.";
    const segButtons = [];
    for (const [modeName, label] of [["random", "Random"], ["fixed", "Fixed"]]) { const button = document.createElement("button"); button.type = "button"; button.className = "ds-seed-switch-seg" + (controlState.mode === modeName ? " active" : ""); button.textContent = label; button.dataset.mode = modeName; button.title = modeName === "random" ? "Roll a new seed on every queue" : "Keep the current seed on every queue (repeatable)"; button.onclick = () => { if (controlState.mode === modeName) return; controlState.mode = modeName; if (modeName === "fixed") { controlState.last_seed = lastSeedText; dasiwaSeedControlRemember(lastSeedText); } dasiwaSeedControlEmit(); dasiwaSeedControlSyncSwitch(); }; segButtons.push(button); switchEl.append(button); }
    // In-place switch sync (no panel rebuild): re-mark the active segment
    // and re-disable Use Last so a mode flip is instant.
    const dasiwaSeedControlSyncSwitch = () => { for (const button of segButtons) button.classList.toggle("active", button.dataset.mode === controlState.mode); if (lastBtn) lastBtn.disabled = !controlState.last_seed; };
    modeRow.append(switchEl);
    const roll = document.createElement("button"); roll.type = "button"; roll.className = "ds-seed-btn"; roll.textContent = "New"; roll.title = "Roll a new seed and retain the selected mode"; roll.style.cssText = "flex:1;padding:3px 0;justify-content:center;gap:0;text-align:center;white-space:nowrap"; roll.onclick = () => { lastSeedText = dasiwaSeedControlRollSeed(); dasiwaSeedControlRemember(lastSeedText); dasiwaSeedControlWriteSeedMirror(lastSeedText); input.value = lastSeedText; dasiwaSeedControlEmit(); dasiwaSeedControlSyncSwitch(); dasiwaSeedControlRefitFont(); };
    modeRow.append(roll);

    // Row 3: Use Last, Last 10 seeds.
    const last = document.createElement("button"); last.type = "button"; last.className = "ds-seed-btn"; last.textContent = "Use Last"; last.disabled = !controlState.last_seed; last.style.cssText = "width:68px;padding:3px 0;justify-content:center;gap:0;text-align:center;white-space:nowrap"; last.onclick = () => { if (!controlState.last_seed) return; lastSeedText = controlState.last_seed; dasiwaSeedControlWriteSeedMirror(lastSeedText); input.value = lastSeedText; controlState.mode = "fixed"; dasiwaSeedControlEmit(); dasiwaSeedControlSyncSwitch(); dasiwaSeedControlRefitFont(); };
    const lastBtn = last;
    panelLastBtn = last;
    // Keep the history trigger compact at every node width; only the row
    // itself participates in the responsive wrapping.
    const history = document.createElement("details"); history.style.cssText = "position:relative;flex:0 0 128px;min-width:0"; const summary = document.createElement("summary"); summary.className = "ds-seed-btn"; summary.textContent = "Last 10 seeds"; summary.style.cssText = "padding:3px 6px;justify-content:center;gap:0;text-align:center;white-space:nowrap;cursor:pointer"; history.append(summary); const list = document.createElement("div"); list.style.cssText = "position:absolute;z-index:10;right:0;top:24px;min-width:160px;max-height:180px;overflow:auto;padding:5px;background:var(--comfy-menu-bg,#353535);border:1px solid var(--border-color,#4e4e4e);border-radius:4px";
    // Populates (or repopulates) the Last-10 list in place; used both at
    // panel build and after a queue-time roll, so the list reflects the
    // freshly rolled seed without rebuilding the whole panel.
    panelHistoryRowBuilder = () => {
      list.textContent = "";
      for (const value of controlState.recent) { const row = document.createElement("div"); row.style.cssText = "display:flex;gap:4px;align-items:center"; const item = document.createElement("button"); item.type = "button"; item.textContent = value; item.style.cssText = "flex:1;padding:3px 5px;border:0;background:transparent;color:#d5e6f2;text-align:right;cursor:pointer"; item.onclick = () => { lastSeedText = value; dasiwaSeedControlWriteSeedMirror(lastSeedText); input.value = lastSeedText; controlState.mode = "fixed"; controlState.last_seed = value; dasiwaSeedControlRemember(value); dasiwaSeedControlEmit(); dasiwaSeedControlSyncSwitch(); dasiwaSeedControlRefitFont(); }; const copy = document.createElement("button"); copy.type = "button"; copy.textContent = "Copy"; copy.title = "Copy seed"; copy.style.cssText = "padding:2px 5px;font-size:10px"; copy.onclick = event => { event.stopPropagation(); navigator.clipboard.writeText(value).then(() => { copy.textContent = "✓"; setTimeout(() => { copy.textContent = "Copy"; }, 900); }); }; row.append(item, copy); list.append(row); }
      if (!controlState.recent.length) list.textContent = "No previous seeds";
    };
    panelHistoryRowBuilder();
    history.append(list);
    panelHistory = history;
    panelList = list;
    const historyRow = document.createElement("div"); historyRow.className = "ds-seed-row"; historyRow.append(last, history);

    seedControl.append(fieldRow, modeRow, historyRow);
    return seedControl;
  };

  const root = document.createElement("div"); root.className = "ds-seed"; root.style.cssText = "width:100%";
  const dasiwaSeedControlSyncWidth = width => {
    const contentWidth = Math.max(DASIWASEED_MIN_PANEL_WIDTH, Number(width || node.size?.[0] || DASIWASEED_COMPUTE_SIZE_WIDTH) - 40);
    root.style.width = `${contentWidth}px`;
  };
  const dasiwaSeedControlRender = () => { root.innerHTML = ""; root.append(dasiwaSeedControlBuildSeedPanel()); };

  // In-place panel refresh after a queue-time roll: update the seed input,
  // the Last-10 list, and the Use Last button without rebuilding the panel
  // (a rebuild would collapse an open Last-10 list and flicker the
  // controls while the user is on the Run button).
  const dasiwaSeedControlRefreshPanel = () => {
    if (panelInput?.isConnected) { panelInput.value = lastSeedText; panelFitSeedFont(); }
    panelHistoryRowBuilder?.();
    if (panelLastBtn) panelLastBtn.disabled = !controlState.last_seed;
  };

  // Each wrapped row costs 42 px plus the panel gap. Keep the minimum
  // height in lockstep with the responsive flex layout: three, two, or one row.
  const dasiwaSeedControlUiHeight = () => {
    const panelWidth = Number(node.size?.[0] || DASIWASEED_COMPUTE_SIZE_WIDTH) - 40;
    if (panelWidth >= DASIWASEED_WIDE_PANEL_WIDTH) return 50;
    if (panelWidth >= DASIWASEED_TWO_LINE_PANEL_WIDTH) return 96;
    return 140;
  };
  if (node.addDOMWidget) {
    const domWidget = node.addDOMWidget("dasiwa_seed_control_ui", "custom", root, { serialize: false, hideOnZoom: false, hideInPanel: true, getHeight: dasiwaSeedControlUiHeight, destroy: () => clearInterval(node.__dasiwaSeedExtPoll) });
    // This is a minimum size only. The root follows node resizing so the
    // container query can move rows 2/3 beside the seed field when space exists.
    domWidget.computeSize = () => [DASIWASEED_COMPUTE_SIZE_WIDTH, dasiwaSeedControlUiHeight()];
  }
  if (node.size?.[0] < DASIWASEED_COMPUTE_SIZE_WIDTH || !node.size?.[0]) node.setSize?.([DASIWASEED_COMPUTE_SIZE_WIDTH + 14, dasiwaSeedControlUiHeight()]);
  const dasiwaSeedControlOriginalResize = node.onResize;
  node.onResize = function (size) {
    dasiwaSeedControlOriginalResize?.apply(this, arguments);
    dasiwaSeedControlSyncWidth(size?.[0]);
  };
  dasiwaSeedControlSyncWidth(node.size?.[0]);
  requestAnimationFrame(() => dasiwaSeedControlSyncWidth(node.size?.[0]));

  node.__dasiwaSeedRestorePersistedState = () => { const parsed = dasiwaSeedControlParseState(dasiwaSeedControlStateWidget().value); controlState = parsed; lastSeedText = parsed.last_seed || String(dasiwaSeedControlSeedWidget().value ?? 0); dasiwaSeedControlRender(); };
  node.__dasiwaSeedPrepareSeed = () => { if (dasiwaSeedControlHasExternalSeed() || controlState?.mode !== "random") return; const value = dasiwaSeedControlRollSeed(); lastSeedText = value; const widget = dasiwaSeedControlSeedWidget(); if (widget) { widget.value = value; widget.callback?.(value); } controlState.last_seed = value; controlState.recent = [value, ...(controlState.recent || []).filter(entry => entry !== value)].slice(0, 10); dasiwaSeedControlEmit(); dasiwaSeedControlRefreshPanel(); };
  node.__dasiwaSeedExtPoll = window.setInterval(() => {
    const linked = dasiwaSeedControlHasExternalSeed();
    if (linked !== lastExternalSeedLinked) {
      lastExternalSeedLinked = linked;
      dasiwaSeedControlRender();
    }
  }, 300);
  dasiwaSeedControlRender();
}

// ── Queue-time seed preparation ────────────────────────────────────────
// The Vue-based ComfyUI frontend no longer dispatches the legacy extension
// queue hook, and its Run button queues through an internal store path
// that never calls the global app.queuePrompt — so the Random-mode roll is
// triggered by wrapping the app's prompt-build entry point (graphToPrompt)
// instead, the same approach Pixaroma's Seed node uses. That method is the
// single choke point every Run routes through: the Run button, batch items,
// and partial execution all serialize the graph via it. The pre-pass rolls
// the seed BEFORE the graph is serialized, so the fresh seed is what lands
// in the queued prompt body (and what H3 Cache keys on). One roll per
// prompt build: a batch-queued run of N outputs rolls once, matching the
// documented 'rolls a fresh seed on every queue' semantics. Fixed mode and
// linked external seeds are left alone — __dasiwaSeedPrepareSeed
// early-returns for both.

function dasiwaSeedControlNodesInGraphTree(graph, visited = new Set()) {
  if (!graph || visited.has(graph)) return [];
  visited.add(graph);
  const nodes = graph._nodes ?? graph.nodes ?? [];
  return nodes.flatMap(node => [node, ...dasiwaSeedControlNodesInGraphTree(node.subgraph, visited)]);
}

function dasiwaSeedControlPrepareAll() {
  for (const node of dasiwaSeedControlNodesInGraphTree(app.graph)) {
    if (DASIWASEED_NODE_TYPES.has(node.comfyClass)) node.__dasiwaSeedPrepareSeed?.();
  }
}

function dasiwaSeedControlPatchGraphToPrompt() {
  if (app.__dasiwaSeedGraphPromptPatched || typeof app?.graphToPrompt !== "function") return;
  const originalGraphToPrompt = app.graphToPrompt.bind(app);
  app.graphToPrompt = async function (...args) {
    dasiwaSeedControlPrepareAll();
    return originalGraphToPrompt(...args);
  };
  app.__dasiwaSeedGraphPromptPatched = true;
}

dasiwaSeedControlPatchGraphToPrompt();

app.registerExtension({
  name: "DaSiWa.SeedControl",
  nodeCreated(node) { if (DASIWASEED_NODE_TYPES.has(node.comfyClass)) dasiwaSeedControlInstall(node); },
  loadedGraphNode(node) { if (DASIWASEED_NODE_TYPES.has(node.comfyClass)) { dasiwaSeedControlInstall(node); node.__dasiwaSeedRestorePersistedState?.(); } },
});
