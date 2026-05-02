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

// Toggle in browser console:  window.DASIWA_SWITCH_DEBUG = true
function dlog(...args) {
    if (typeof window !== "undefined" && window.DASIWA_SWITCH_DEBUG) {
        console.log("[DaSiWa Switch]", ...args);
    }
}

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
 *
 * Chaining: if the upstream node is another DaSiWa_NodeStatusSwitch,
 * recursively read its effective enabled value (which is what the
 * upstream switch outputs from its enabled_out pin).  A `visited`
 * set guards against accidental cycles in user graphs.
 */
function readEnabled(switchNode, visited) {
    visited = visited ?? new Set();
    if (visited.has(switchNode.id)) return null; // cycle guard
    visited.add(switchNode.id);

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
            if (src) {
                // If upstream is another switch, use its effective value,
                // not its raw widget — a chained switch's output is what
                // matters for downstream chaining.
                if (src.type === NODE_TYPE) {
                    const v = readEnabled(src, visited);
                    if (v != null) return v;
                } else {
                    const v = readBoolFromNode(src);
                    if (v != null) return v;
                }
            }
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

    dlog("applySwitch", {
        nodeId: switchNode.id,
        enabled,
        triggerOn,
        action,
        targetsActive,
        targetMode,
    });

    const graph = switchNode.graph ?? app.graph;
    if (!graph) return;

    const targetIds = getTargetNodeIds(switchNode);
    const allNodes  = graph._nodes ?? graph.nodes ?? [];

    for (const id of targetIds) {
        const target = allNodes.find((n) => n.id === id);
        if (target) target.mode = targetMode;
    }
}

// ── propagate to downstream chained switches ────────────────────────────
//
// When this switch's effective state changes, find every other switch
// whose `enabled` input is wired (directly or transitively) from this
// switch's `enabled_out` and re-apply them too.  This is more reliable
// than waiting for the polling mirror loop to notice.

function findDownstreamSwitches(switchNode, visited) {
    visited = visited ?? new Set();
    if (visited.has(switchNode.id)) return [];
    visited.add(switchNode.id);

    const graph = switchNode.graph ?? app.graph;
    if (!graph) return [];

    const allNodes = graph._nodes ?? graph.nodes ?? [];
    const linksMap = graph.links ?? graph._links;
    if (!linksMap) return [];

    // Find the output slot named "enabled_out" on this node.
    const outIdx = (switchNode.outputs ?? []).findIndex(
        (o) => o?.name === "enabled_out"
    );
    if (outIdx < 0) return [];

    const outSlot = switchNode.outputs[outIdx];
    const linkIds = outSlot?.links ?? [];

    const found = [];
    for (const linkId of linkIds) {
        const link = linksMap?.[linkId] ?? linksMap?.get?.(linkId);
        if (!link) continue;
        const target = allNodes.find((n) => n.id === link.target_id);
        if (!target || target.type !== NODE_TYPE) continue;
        // Only count it if the link lands on the target's "enabled" input.
        const tInput = target.inputs?.[link.target_slot];
        if (!tInput || tInput.name !== "enabled") continue;

        found.push(target);
        // Recurse to catch chains of length > 2.
        const further = findDownstreamSwitches(target, visited);
        for (const f of further) found.push(f);
    }
    return found;
}

function applySwitchAndDownstream(switchNode) {
    applySwitch(switchNode);
    const downstream = findDownstreamSwitches(switchNode);
    dlog("downstream of", switchNode.id, "=", downstream.map((d) => d.id));
    for (const d of downstream) {
        // Mirror upstream's effective enabled into the downstream
        // switch's local widget so the UI reflects the chain.
        const upstreamEnabled = readEnabled(switchNode);
        const localW = getWidget(d, "enabled");
        if (localW && localW.value !== upstreamEnabled) {
            localW.value = upstreamEnabled;
        }
        applySwitch(d);
    }
    const graph = switchNode.graph ?? app.graph;
    graph?.setDirtyCanvas?.(true, true);
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
    if (!src) return;

    // If the source is another switch, we want its *effective* enabled
    // (the value it would output), not its raw widget.  This makes the
    // local widget on a chained switch mirror the upstream switch
    // through any number of links.
    let externalValue;
    if (src.type === NODE_TYPE) {
        externalValue = readEnabled(src);
    } else {
        externalValue = readBoolFromNode(src);
    }
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

            // Ensure the chaining output pin exists.  ComfyUI normally
            // adds it from RETURN_TYPES, but be defensive in case the
            // node definition is reloaded without it.
            const hasOutput = (self.outputs ?? []).some(
                (o) => o?.name === "enabled_out"
            );
            if (!hasOutput) {
                self.addOutput("enabled_out", "BOOLEAN");
            }

            if (countTargetSlots(self) === 0) {
                self.addInput(targetSlotName(1), "*");
            }

            // Local widget callbacks: live updates when the user
            // toggles directly on the switch node.  Cascade through
            // any chained downstream switches.
            for (const w of self.widgets ?? []) {
                const origCb = w.callback;
                w.callback = function (...args) {
                    origCb?.apply(this, args);
                    requestAnimationFrame(() => applySwitchAndDownstream(self));
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
                    applySwitchAndDownstream(self);
                });
            }
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            origOnConfigure?.apply(this, arguments);
            const self = this;
            // Migration: older saved graphs had no outputs.  Add the
            // chaining output if missing.  Strip any *other* outputs
            // (defensive — should not occur).
            const outs = self.outputs ?? [];
            for (let i = outs.length - 1; i >= 0; i--) {
                if (outs[i]?.name !== "enabled_out") {
                    self.removeOutput(i);
                }
            }
            const hasOutput = (self.outputs ?? []).some(
                (o) => o?.name === "enabled_out"
            );
            if (!hasOutput) {
                self.addOutput("enabled_out", "BOOLEAN");
            }
            requestAnimationFrame(() => syncTargetSlots(self));
        };
    },

    async loadedGraphNode(node) {
        if (node.type !== NODE_TYPE) return;
        requestAnimationFrame(() => {
            const outs = node.outputs ?? [];
            for (let i = outs.length - 1; i >= 0; i--) {
                if (outs[i]?.name !== "enabled_out") {
                    node.removeOutput(i);
                }
            }
            const hasOutput = (node.outputs ?? []).some(
                (o) => o?.name === "enabled_out"
            );
            if (!hasOutput) {
                node.addOutput("enabled_out", "BOOLEAN");
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
