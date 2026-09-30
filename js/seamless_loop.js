import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DaSiWa.SeamlessLoop.Socketless",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DaSiWa_SeamlessLoop") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            for (const name of ["exact_endpoint"]) {
                const widget = this.widgets?.find((item) => item.name === name);
                if (widget) {
                    widget.options ??= {};
                    widget.options.socketless = true;
                }
                const index = this.inputs?.findIndex((input) => input.name === name);
                if (index >= 0) this.removeInput(index);
            }
            return result;
        };
    },
});
