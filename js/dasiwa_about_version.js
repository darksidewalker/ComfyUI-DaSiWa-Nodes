import { app } from "../../scripts/app.js";

const DASIWA_VERSION = "0.4.56";

app.registerExtension({
    name: "DaSiWa.AboutVersion",
    aboutPageBadges: [
        {
            label: `DaSiWa Custom Nodes v${DASIWA_VERSION}`,
            url: "https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes",
            icon: "pi pi-github",
        },
    ],
});
