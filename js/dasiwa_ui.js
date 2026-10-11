import { app } from "../../scripts/app.js";

/**
 * DaSiWa UI Logic
 * Handles dynamic widget visibility for DaSiWa nodes to hide irrelevant options.
 */

const RTX_NODE = "DaSiWa_RTX_UpscalerRefiner";
const SCALER_NODE = "DaSiWa_ResolutionScaleCalculator";

app.registerExtension({
    name: "DaSiWa.UILogic",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData.name.startsWith("DaSiWa_")) return;

        // Keep the appended RTX widget slots stable across SDK upgrades/downgrades.
        // Backend INPUT_TYPES remains capability-aware; unavailable UI placeholders
        // retain saved values and execute rejects explicit unsupported requests.
        const unavailable = new Set();
        if (nodeData.name === RTX_NODE) {
            const fallback = {
                image_encoding: [["8-bit RGB"], { default: "8-bit RGB" }],
                denoise_strength: ["FLOAT", { default: 1, min: 0, max: 1 }],
                deblur_strength: ["FLOAT", { default: 1, min: 0, max: 1 }],
                upscale_strength: ["FLOAT", { default: 1, min: 0, max: 1 }],
                preserve_alpha: ["BOOLEAN", { default: false }],
                channel_integrity_check: ["BOOLEAN", { default: false }],
                hdr_enabled: ["BOOLEAN", { default: false }],
                hdr_contrast: ["INT", { default: 100, min: 0, max: 200 }],
                hdr_saturation: ["INT", { default: 100, min: 0, max: 200 }],
                hdr_middle_gray: ["INT", { default: 50, min: 10, max: 100 }],
                hdr_luminance: ["INT", { default: 650, min: 400, max: 2000 }],
                hdr_debanding: ["BOOLEAN", { default: true }],
            };
            const optional = nodeData.input.optional || {};
            const stable = Object.fromEntries(Object.entries(optional).filter(([name]) => !(name in fallback)));
            for (const [name, config] of Object.entries(fallback)) {
                if (!(name in optional) || optional[name][1]?.dasiwa_unavailable) unavailable.add(name);
                const configForSdk = optional[name] || config;
                const unavailableHelp = "Unavailable in this installed RTX SDK. Saved value is retained; requesting this feature fails explicitly. Restart ComfyUI after an SDK change.";
                const help = [configForSdk[1]?.tooltip || configForSdk[1]?.description, unavailableHelp].filter(Boolean).join("\n\n");
                stable[name] = unavailable.has(name) ? [configForSdk[0], { ...configForSdk[1],
                    description: help, tooltip: help }] : configForSdk;
            }
            nodeData.input.optional = stable;

            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const result = origConfigure?.apply(this, arguments);
                const saved = info.properties?.dasiwa_rtx_widgets;
                if (saved && typeof saved === "object") {
                    for (const widget of this.widgets || []) {
                        if (Object.hasOwn(saved, widget.name)) widget.value = saved[widget.name];
                    }
                }
                return result;
            };
            const origSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function (info) {
                const result = origSerialize?.apply(this, arguments);
                info.properties ||= {};
                info.properties.dasiwa_rtx_widgets = Object.fromEntries(
                    (this.widgets || []).filter(w => w.name in fallback).map(w => [w.name, w.value]));
                return result;
            };
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            // Preserve native tooltip metadata, with legacy description fallback.
            if (nodeData.input) {
                const allInputs = {
                    ...(nodeData.input.required || {}),
                    ...(nodeData.input.optional || {})
                };

                for (const [name, config] of Object.entries(allInputs)) {
                    const widget = this.widgets?.find(w => w.name === name);
                    const help = config[1]?.tooltip || config[1]?.description;
                    if (widget && help) {
                        widget.tooltip = help;
                        widget.options ||= {};
                        widget.options.tooltip = help;
                        // Native DOM-backed widgets, when exposed by the frontend.
                        for (const element of [widget.element, widget.inputEl]) {
                            if (element?.setAttribute) element.setAttribute("title", help);
                        }
                    }
                    if (widget && unavailable.has(name)) {
                        widget.disabled = true;
                        widget.options ||= {};
                        widget.options.disabled = true;
                        widget.options.readOnly = true;
                    }
                }
            }
            return r;
        };
    }
});
