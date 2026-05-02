/**
 * DaSiWa Node Status Switch - ComfyUI frontend extension
 *
 * Minimal JS responsibilities:
 *   1. Manage dynamic target_XX inputs (grow on connect, up to 20)
 *   2. At queue time, read the effective "enabled" value (from the
 *      local widget OR from a connected upstream node) and set the
 *      mode of every target node accordingly.
 *
 * No live updates.  No widget callbacks.  No polling.
 * The mode change happens right before the prompt is sent.
 *
 * ComfyUI mode values:
 *   0 = ALWAYS  (active)
 *   2 = NEVER   (mute)
 *   4 = bypass
 */

import { app } from "../../scripts/app.js";

const MODE_ACTIVE   = 0;
const MODE_MUTE     = 2;
const MODE_BYPASS   = 4;
const NODE_TYPE     = "DaSiWa_NodeStatusSwitch";
const TARGET_PREFIX = "target_";
const MAX_TARGETS   = 99;

// ── helpers ─────────────────────────────────────────────────────────────

function getWidget(node, name) {
    return (node.widgets ?? []).find((w) => w.name === name);
}

function isTargetSlot(input) {
    return typeof input?.name === "string" &&
           input.name.startsWith(TARGET_PREFIX);
}

function targetSlotName(n) {
    return `${TARGET_PREFIX}${String(n).padStart(2, "0")}`;
}

function countTargetSlots(node) {
    return (node.inputs ?? []).filter(isTargetSlot).length;
}

function isBoolWidget(w) {
    if (!w) return false;
    if (w.type === "toggle") return true;
    if (typeof w.type === "string" && w.type.toUpperCase().includes("BOOLEAN")) return true;
    if (typeof w.value === "boolean") return true;
    return false;
}

function findBoolWidget(node) {
    return node?.widgets?.find(isBoolWidget) ?? null;
}

/**
 * Read a boolean value from a node, regardless of how the node
 * stores it.  Tries every plausible storage path.  Returns the
 * boolean or null if nothing usable is found.
 */
function readBoolFromNode(node) {
    if (!node) return null;

    // ── Path 1: conventional widget ─────────────────────────────────────
    const w = findBoolWidget(node);
    if (w != null && typeof w.value === "boolean") return !!w.value;

    // ── Path 2: any widget whose value is boolean (broader than Path 1) ──
    if (node.widgets) {
        for (const aw of node.widgets) {
            if (typeof aw?.value === "boolean") return !!aw.value;
        }
    }

    // ── Path 3: input slots may carry a widget object with the value ───
    // PrimitiveBoolean (and similar widget-input hybrids) put the widget
    // reference under inputs[i].widget; the value can live on the widget,
    // on the input slot itself, or in widgets_values keyed by widget name.
    if (node.inputs) {
        for (const inp of node.inputs) {
            if (typeof inp?.widget?.value === "boolean") return !!inp.widget.value;
            if (typeof inp?.value === "boolean") return !!inp.value;
        }
    }

    // ── Path 4: top-level widgets_values array ──────────────────────────
    if (Array.isArray(node.widgets_values)) {
        // First boolean entry
        for (const v of node.widgets_values) {
            if (typeof v === "boolean") return v;
        }
        // First entry coerced (some versions store "true"/"false" strings)
        const first = node.widgets_values[0];
        if (first === "true" || first === true) return true;
        if (first === "false" || first === false) return false;
    }

    // ── Path 5: properties may hold the value ───────────────────────────
    if (node.properties) {
        for (const v of Object.values(node.properties)) {
            if (typeof v === "boolean") return v;
        }
    }

    return null;
}

/**
 * Read the effective "enabled" value at queue time.
 * If the "enabled" input is connected, follow the link to the source
 * node and read its boolean value.  Otherwise fall back to the local
 * widget on the switch.
 */
function readEnabled(switchNode) {
    const graph = switchNode.graph ?? app.graph;

    const enabledInput = (switchNode.inputs ?? []).find(
        (inp) => inp?.name === "enabled"
    );

    if (enabledInput && enabledInput.link != null && graph) {
        const link =
            graph.links?.[enabledInput.link] ??
            graph._links?.get?.(enabledInput.link);
        if (link) {
            const allNodes = graph._nodes ?? graph.nodes ?? [];
            const src = allNodes.find((n) => n.id === link.origin_id);
            const v = readBoolFromNode(src);
            if (v != null) return v;
        }
    }

    // Fallback: local widget on the switch itself
    const localW = getWidget(switchNode, "enabled");
    return localW != null ? !!localW.value : true;
}

// ── target discovery ────────────────────────────────────────────────────

function getTargetNodeIds(switchNode) {
    const graph = switchNode.graph ?? app.graph;
    if (!graph) return [];

    const ids = [];
    for (const input of switchNode.inputs ?? []) {
        if (!isTargetSlot(input)) continue;
        if (input.link == null) continue;

        const link =
            graph.links?.[input.link] ?? graph._links?.get?.(input.link);
        if (link && link.origin_id != null) {
            ids.push(link.origin_id);
        }
    }
    return ids;
}

// ── dynamic input management ────────────────────────────────────────────

function syncTargetSlots(node) {
    const targetInputs = (node.inputs ?? []).filter(isTargetSlot);
    const connected = targetInputs.filter((inp) => inp.link != null).length;
    const desired = Math.min(connected + 1, MAX_TARGETS);

    while (countTargetSlots(node) < desired) {
        const nextNum = countTargetSlots(node) + 1;
        node.addInput(targetSlotName(nextNum), "*");
    }

    while (countTargetSlots(node) > desired) {
        const allInputs = node.inputs ?? [];
        for (let i = allInputs.length - 1; i >= 0; i--) {
            if (isTargetSlot(allInputs[i]) && allInputs[i].link == null) {
                node.removeInput(i);
                break;
            }
        }
    }

    node.setSize(node.computeSize());
}

// ── apply mode to targets ───────────────────────────────────────────────

/**
 *   "true -> active"  => targets ACTIVE when enabled=true
 *   "false -> active" => targets ACTIVE when enabled=false
 */
function applySwitch(switchNode) {
    const triggerW = getWidget(switchNode, "trigger_on");
    const actionW  = getWidget(switchNode, "action");
    if (!triggerW || !actionW) return;

    const enabled        = readEnabled(switchNode);
    const triggerOn      = triggerW.value ?? "";
    const action         = actionW.value ?? "bypass";
    const activeWhenTrue = triggerOn.startsWith("true");
    const targetsActive  = activeWhenTrue ? enabled : !enabled;
    const actionMode     = action === "mute" ? MODE_MUTE : MODE_BYPASS;
    const targetMode     = targetsActive ? MODE_ACTIVE : actionMode;

    const graph = switchNode.graph ?? app.graph;
    if (!graph) return;

    const targetIds = getTargetNodeIds(switchNode);
    const allNodes  = graph._nodes ?? graph.nodes ?? [];

    for (const id of targetIds) {
        const target = allNodes.find((n) => n.id === id);
        if (target) target.mode = targetMode;
    }
}

// ── live mirror: external boolean -> local widget ───────────────────────
//
// When the "enabled" input is connected to an external boolean source,
// poll that source and mirror its value into the switch's own
// "enabled" widget.  Writing to the local widget shows the toggle
// flipping on the switch's UI and triggers a live applySwitch.

function syncExternalToLocal(switchNode) {
    const enabledInput = (switchNode.inputs ?? []).find(
        (inp) => inp?.name === "enabled"
    );
    if (!enabledInput || enabledInput.link == null) return; // no external

    const localW = getWidget(switchNode, "enabled");
    if (!localW) return;

    const graph = switchNode.graph ?? app.graph;
    if (!graph) return;

    const link =
        graph.links?.[enabledInput.link] ??
        graph._links?.get?.(enabledInput.link);
    if (!link) return;

    const allNodes = graph._nodes ?? graph.nodes ?? [];
    const src = allNodes.find((n) => n.id === link.origin_id);

    const externalValue = readBoolFromNode(src);
    if (externalValue == null) return;

    if (localW.value === externalValue) return; // already in sync

    localW.value = externalValue;
    applySwitch(switchNode);
    graph.setDirtyCanvas?.(true, true);
}

let mirrorLoopRunning = false;

function mirrorFrame() {
    if (!mirrorLoopRunning) return;
    try {
        const allNodes = app.graph?._nodes ?? app.graph?.nodes ?? [];
        for (const n of allNodes) {
            if (n.type === NODE_TYPE) syncExternalToLocal(n);
        }
    } catch (_) {}
    requestAnimationFrame(mirrorFrame);
}

function startMirrorLoop() {
    if (mirrorLoopRunning) return;
    mirrorLoopRunning = true;
    requestAnimationFrame(mirrorFrame);
}

startMirrorLoop();

// ── extension registration ──────────────────────────────────────────────

app.registerExtension({
    name: "DaSiWa.NodeStatusSwitch",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            const self = this;

            while (self.outputs && self.outputs.length > 0) {
                self.removeOutput(0);
            }

            if (countTargetSlots(self) === 0) {
                self.addInput(targetSlotName(1), "*");
            }

            // Local widget callbacks: live updates when the user
            // toggles directly on the switch node.
            for (const w of self.widgets ?? []) {
                const origCb = w.callback;
                w.callback = function (...args) {
                    origCb?.apply(this, args);
                    requestAnimationFrame(() => applySwitch(self));
                };
            }

            self.setSize(self.computeSize());
        };

        const origConnChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (
            side,
            slotIndex,
            connected,
            linkInfo
        ) {
            origConnChange?.apply(this, arguments);
            const self = this;
            if (side === 1) {
                requestAnimationFrame(() => {
                    syncTargetSlots(self);
                    applySwitch(self);
                });
            }
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            origOnConfigure?.apply(this, arguments);
            const self = this;
            while (self.outputs && self.outputs.length > 0) {
                self.removeOutput(0);
            }
            requestAnimationFrame(() => syncTargetSlots(self));
        };
    },

    async loadedGraphNode(node) {
        if (node.type !== NODE_TYPE) return;
        requestAnimationFrame(() => {
            while (node.outputs && node.outputs.length > 0) {
                node.removeOutput(0);
            }
            syncTargetSlots(node);
        });
    },

    // The only place mode is applied: right before the prompt is sent.
    async beforeQueued() {
        const allNodes = app.graph?._nodes ?? app.graph?.nodes ?? [];
        for (const n of allNodes) {
            if (n.type === NODE_TYPE) applySwitch(n);
        }
        app.graph?.setDirtyCanvas?.(true, true);
    },
});
