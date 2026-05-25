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

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            // Auto-populate hover tooltips from Python descriptions
            if (nodeData.input) {
                const allInputs = {
                    ...(nodeData.input.required || {}),
                    ...(nodeData.input.optional || {})
                };

                for (const [name, config] of Object.entries(allInputs)) {
                    const widget = this.widgets?.find(w => w.name === name);
                    if (widget && config[1]?.description) {
                        // 'tooltip' is picked up by various ComfyUI UI enhancement extensions
                        widget.tooltip = config[1].description;
                    }
                }
            }
            return r;
        };
    }
});
