import { app } from "../../scripts/app.js";

const NODE_TYPE = "DaSiWaH3TiledUpscale";
const installed = new WeakSet();
const hiddenWidgets = new WeakMap();

function isTiledUpscale(node) {
    return node?.comfyClass === NODE_TYPE || node?.type === NODE_TYPE;
}

function widget(node, name) {
    return node.widgets?.find(entry => entry.name === name);
}

function setHidden(entry, hidden) {
    if (!entry) return;
    if (hidden && !hiddenWidgets.has(entry)) {
        hiddenWidgets.set(entry, { draw: entry.draw, computeSize: entry.computeSize });
        // Native Vue widgets consume hidden; classic canvas needs zero geometry.
        // Do NOT change type, serializeValue, options.serialize or array position.
        entry.hidden = true;
        entry.draw = () => {};
        entry.computeSize = () => [0, -4];
    } else if (!hidden && hiddenWidgets.has(entry)) {
        const native = hiddenWidgets.get(entry);
        entry.hidden = false;
        entry.draw = native.draw;
        entry.computeSize = native.computeSize;
        hiddenWidgets.delete(entry);
    }
}

function refresh(node) {
    setHidden(widget(node, "memory_budget_mb"), true);
    const planning = widget(node, "enforce_memory_budget");
    if (planning) planning.label = "Memory Planning";
    // Do not rewrite legacy booleans (including serialized string values).
    const enabled = !planning || planning.value === true || planning.value === "true";
    setHidden(widget(node, "memory_reserve_pct"), !enabled);
    node.setDirtyCanvas?.(true, true);
}

function install(node) {
    if (!isTiledUpscale(node)) return;
    if (!installed.has(node)) {
        installed.add(node);
        const planning = widget(node, "enforce_memory_budget");
        if (planning) {
            const originalCallback = planning.callback;
            planning.callback = function (...args) {
                const result = originalCallback?.apply(this, args);
                refresh(node);
                return result;
            };
        }
        const originalConfigure = node.onConfigure;
        node.onConfigure = function (info, ...args) {
            const result = originalConfigure?.call(this, info, ...args);
            const reserve = widget(this, "memory_reserve_pct");
            // Core restores the indexed slots. Only default the newly appended
            // input when loading an older array, including into an existing node.
            if (reserve && Array.isArray(info?.widgets_values) &&
                info.widgets_values.length <= this.widgets.indexOf(reserve)) {
                reserve.value = 5;
            }
            const sharpness = widget(this, "upscale_sharpness");
            if (sharpness && Array.isArray(info?.widgets_values) &&
                info.widgets_values.length <= this.widgets.indexOf(sharpness)) {
                sharpness.value = 1;
            }
            refresh(this);
            return result;
        };
    }
    refresh(node);
}

app.registerExtension({
    name: "DaSiWa.H3TiledUpscaleMemoryPlanning",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        const inputs = { ...nodeData.input?.required, ...nodeData.input?.optional };
        // Same native display mechanism as the LLM detail/creativity sliders;
        // ComfyUI owns float callbacks, step conversion and Nodes 2.0 rendering.
        if (inputs.memory_reserve_pct?.[1]) inputs.memory_reserve_pct[1].display = "slider";
        if (inputs.enforce_memory_budget?.[1]) {
            inputs.enforce_memory_budget[1].display_name = "Memory Planning";
        }
    },
    nodeCreated: install,
    loadedGraphNode: install,
});
