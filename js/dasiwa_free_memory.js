import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SETTING = "DaSiWa.FreeMemory.Enabled";
const ID = "dasiwa-free-memory";
let observer;

function placeButton() {
    const root = document.getElementById(ID);
    if (!root) return false;
    const toolbar = document.querySelector('[data-testid="legacy-topbar-container"] > .flex');
    if (toolbar) {
        if (toolbar.firstElementChild !== root) toolbar.prepend(root);
        return true;
    }
    const extensions = document.querySelector('button[aria-label="Extensions"]');
    if (extensions?.parentElement) {
        extensions.before(root);
        return true;
    }
    return false;
}

function enable(enabled) {
    if (!enabled) {
        observer?.disconnect();
        observer = null;
        document.getElementById(ID)?.remove();
        document.getElementById(`${ID}-menu`)?.remove();
        document.getElementById(`${ID}-style`)?.remove();
        return;
    }
    if (document.getElementById(ID)) return;
    const style = document.createElement("style");
    style.id = `${ID}-style`;
    style.textContent = `#${ID} { position: relative; display: flex; align-items: center; height: 36px; margin-right: 6px; font: 600 11px var(--font-inter, sans-serif); } #${ID} button, #${ID}-menu button { height: 28px; padding: 0 8px; border: 1px solid var(--border-color); color: var(--input-text); background: var(--comfy-input-bg); cursor: pointer; } #${ID} > button { box-sizing: border-box; display: grid; place-items: center; width: 42px; height: 36px; padding: 0; } #${ID} button img { display: block; width: 32px; height: 32px; object-fit: contain; } #${ID} button:hover, #${ID} button:focus-visible, #${ID}-menu button:hover, #${ID}-menu button:focus-visible { border-color: #22d3ee; } #${ID}-menu { position: fixed; z-index: 10006; display: grid; min-width: 150px; padding: 4px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 8px 24px #0008; } #${ID}-menu[hidden] { display: none; } #${ID}-menu button { text-align: left; border: 0; } #${ID}-menu button:hover, #${ID}-menu button:focus-visible { background: color-mix(in srgb, var(--input-text) 18%, var(--comfy-menu-bg, #202020)); color: var(--input-text); box-shadow: inset 0 0 0 1px #22d3ee; outline: none; }`;
    document.head.appendChild(style);
    const root = document.createElement("div");
    root.id = ID;
    root.innerHTML = `<button type="button" aria-label="Free memory" title="Free ComfyUI memory" aria-expanded="false"><img src="${new URL("./assets/dasiwa-free-memory.png", import.meta.url).href}" alt="" /></button>`;
    document.body.appendChild(root);
    const toggle = root.querySelector("button");
    const menu = document.createElement("div");
    menu.id = `${ID}-menu`;
    menu.hidden = true;
    menu.innerHTML = `<button type="button" data-free="vram">Free VRAM</button><button type="button" data-free="ram">Free System RAM</button><button type="button" data-free="both">Free Both</button>`;
    document.body.appendChild(menu);
    const close = () => { menu.hidden = true; toggle.setAttribute("aria-expanded", "false"); };
    toggle.addEventListener("click", () => {
        menu.hidden = !menu.hidden;
        toggle.setAttribute("aria-expanded", String(!menu.hidden));
        if (!menu.hidden) {
            const rect = toggle.getBoundingClientRect();
            menu.style.top = `${Math.min(rect.bottom + 4, window.innerHeight - menu.offsetHeight - 8)}px`;
            menu.style.left = `${Math.max(8, Math.min(rect.right - menu.offsetWidth, window.innerWidth - menu.offsetWidth - 8))}px`;
        }
    });
    document.addEventListener("pointerdown", (event) => { if (!root.contains(event.target) && !menu.contains(event.target)) close(); }, { signal: controller.signal });
    menu.addEventListener("click", async (event) => {
        const choice = event.target.closest("button[data-free]");
        if (!choice) return;
        close();
        choice.disabled = true;
        try {
            const response = await api.fetchApi("/free", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ unload_models: true, free_memory: choice.dataset.free === "ram" || choice.dataset.free === "both" }),
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
        } catch (error) {
            console.error("DaSiWa Free Memory", error);
            window.alert(`Could not free memory: ${error.message}`);
        } finally {
            choice.disabled = false;
        }
    });
    placeButton();
    observer = new MutationObserver(() => { placeButton(); });
    observer.observe(document.body, { childList: true, subtree: true });
}

let controller = new AbortController();
app.registerExtension({
    name: "DaSiWa.FreeMemory",
    init() {
        app.ui.settings.addSetting({
            id: SETTING, name: "Show Free Memory Button",
            category: ["DaSiWa", "Free Memory", "Show Free Memory Button"],
            tooltip: "Show the independent VRAM and system RAM cleanup button in the toolbar.",
            type: "boolean", defaultValue: true,
            onChange: (value) => {
                if (value === false) controller.abort();
                else controller = new AbortController();
                enable(value !== false);
            },
        });
    },
    setup() { enable(app.ui.settings.getSettingValue(SETTING) !== false); },
});
