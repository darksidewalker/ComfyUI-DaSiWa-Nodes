import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
const source = fs.readFileSync(new URL("../js/dasiwa_ui.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/, "");
vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });
const oldOptional = ["empty_cache", "use_mmap", "auto_unload_models", "chunking", "lossless_fp16", "chunk_frames"];
const additions = ["image_encoding", "denoise_strength", "deblur_strength", "upscale_strength", "preserve_alpha",
    "channel_integrity_check", "hdr_enabled", "hdr_contrast", "hdr_saturation", "hdr_middle_gray", "hdr_luminance", "hdr_debanding"];
function data(modern) {
    const optional = Object.fromEntries(oldOptional.map(name => [name, ["INT", { default: 0 }]]));
    Object.assign(optional, { preserve_alpha: ["BOOLEAN", { default: false }], channel_integrity_check: ["BOOLEAN", { default: false }] });
    if (modern) {
        optional.image_encoding = [["8-bit RGB", "10-bit RGB (RGB10A2)"], { default: "8-bit RGB" }];
        for (const name of additions.filter(name => name.startsWith("hdr_") || name.endsWith("_strength"))) {
            optional[name] = ["INT", { default: 1, tooltip: `${name} SDK control help` }];
        }
    }
    if (modern === "unavailable") {
        for (const name of additions.filter(name => name.startsWith("hdr_") || name.endsWith("_strength") || name === "image_encoding")) {
            optional[name][1].dasiwa_unavailable = true;
        }
    }
    return { name: "DaSiWa_RTX_UpscalerRefiner", input: { required: {}, optional } };
}
async function create(modern) {
    class Node {
        onConfigure() { this.configured = true; }
        onSerialize() { this.serialized = true; }
        onNodeCreated() { this.created = true; }
    }
    const definition = data(modern);
    await extension.beforeRegisterNodeDef(Node, definition);
    const node = new Node();
    node.widgets = Object.entries(definition.input.optional).map(([name, config]) => ({ name, value: config[1].default }));
    node.onNodeCreated();
    assert.equal(node.created, true);
    assert.deepEqual(Object.keys(definition.input.optional), [...oldOptional, ...additions]);
    return node;
}
const newer = await create(true);
newer.widgets.find(w => w.name === "image_encoding").value = "10-bit RGB (RGB10A2)";
newer.widgets.find(w => w.name === "hdr_enabled").value = true;
newer.widgets.find(w => w.name === "hdr_luminance").value = 1000;
const saved = { properties: {} };
newer.onSerialize(saved);
const older = await create(false);
older.onConfigure(JSON.parse(JSON.stringify(saved)));
assert.equal(older.configured, true);
for (const name of additions) assert.equal(older.widgets.find(w => w.name === name).value, newer.widgets.find(w => w.name === name).value);
assert.equal(older.widgets.find(w => w.name === "hdr_enabled").disabled, true);
assert.match(older.widgets.find(w => w.name === "hdr_enabled").tooltip, /Unavailable/);
const legacy = await create(false);
legacy.onConfigure({ properties: {}, widgets_values: oldOptional.map(() => 0) });
assert.equal(legacy.widgets.find(w => w.name === "preserve_alpha").value, false);
const roundtrip = { properties: {} };
older.onSerialize(roundtrip);
const upgraded = await create(true);
upgraded.onConfigure(roundtrip);
assert.equal(upgraded.widgets.find(w => w.name === "hdr_luminance").value, 1000);
assert.equal(upgraded.widgets.find(w => w.name === "hdr_enabled").value, true);
assert.equal(upgraded.widgets.find(w => w.name === "hdr_enabled").disabled, undefined);
const unavailableSchema = await create("unavailable");
assert.equal(unavailableSchema.widgets.find(w => w.name === "hdr_enabled").disabled, true);
assert.match(unavailableSchema.widgets.find(w => w.name === "hdr_enabled").tooltip, /Unavailable/);
assert.match(unavailableSchema.widgets.find(w => w.name === "hdr_enabled").tooltip, /hdr_enabled SDK control help/);
assert.equal(unavailableSchema.widgets.find(w => w.name === "image_encoding").disabled, true);
// Hover help uses native tooltip metadata, prefers it over legacy descriptions,
// and reaches DOM-backed controls without changing serialized values/callbacks.
for (const name of ["DaSiWa_RTX_FrameGeneration", "DaSiWa_RTX_UpscalerRefiner"]) {
    class Node {}
    const definition = { name, input: { required: {
        mode: [["LOW", "MEDIUM", "HIGH"], { tooltip: "Native mode help", description: "Legacy mode help" }],
        legacy: ["BOOLEAN", { description: "Legacy-only help" }],
    }, optional: { use_mmap: ["BOOLEAN", { tooltip: "Disk fallback help" }] } } };
    await extension.beforeRegisterNodeDef(Node, definition);
    const node = new Node();
    node.widgets = ["mode", "legacy", "use_mmap"].map(name => ({ name, value: 42,
        callback() {}, element: { setAttribute(key, value) { this[key] = value; } },
        inputEl: { setAttribute(key, value) { this[key] = value; } } }));
    const callbacks = node.widgets.map(w => w.callback);
    node.onNodeCreated();
    for (const [index, help] of ["Native mode help", "Legacy-only help", "Disk fallback help"].entries()) {
        const widget = node.widgets[index];
        assert.equal(widget.tooltip, help);
        assert.equal(widget.options.tooltip, help);
        assert.equal(widget.element.title, help);
        assert.equal(widget.inputEl.title, help);
        assert.equal(widget.value, 42);
        assert.equal(widget.callback, callbacks[index]);
    }
}
console.log("RTX UI contract: stable slots, SDK roundtrip, unavailable placeholders, native/legacy/DOM hover help PASS");
