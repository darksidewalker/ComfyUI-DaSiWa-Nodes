import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let label = "DaSiWa Custom Nodes";
try {
    const response = await api.fetchApi("/dasiwa/version", {
        cache: "no-store",
        signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error(`Version request failed: HTTP ${response.status}`);
    const { version } = await response.json();
    if (typeof version !== "string" || !version.trim()) {
        throw new Error("Version response is missing the package version");
    }
    label += ` v${version}`;
} catch (error) {
    console.warn("[DaSiWa] Could not load the package version:", error);
}

app.registerExtension({
    name: "DaSiWa.AboutVersion",
    aboutPageBadges: [
        {
            label,
            url: "https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes",
            icon: "pi pi-github",
        },
    ],
});
