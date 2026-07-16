import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT_NAME = "dasiwa.system_monitor";
const ROOT_ID = "dasiwa-system-monitor";
const PANEL_ID = "dasiwa-system-monitor-panel";
const SETTINGS_KEY = "dasiwa.system_monitor.settings";
const HISTORY_LENGTH = 60;
const DEFAULT_SETTINGS = { enabled: true, mode: "lite" };

let settings = loadSettings();
let latestSnapshot = null;
let history = [];

function loadSettings() {
    try {
        const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY));
        return {
            enabled: saved?.enabled !== false,
            mode: saved?.mode === "full" ? "full" : "lite",
        };
    } catch {
        return { ...DEFAULT_SETTINGS };
    }
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function placePanel(root) {
    const legacyTopbar = document.querySelector('[data-testid="legacy-topbar-container"] > .flex');
    if (legacyTopbar) {
        legacyTopbar.prepend(root);
        return true;
    }
    const extensionsButton = document.querySelector('button[aria-label="Extensions"]');
    if (!extensionsButton?.parentElement) return false;
    extensionsButton.before(root);
    return true;
}

function bytes(value) {
    if (value === null || value === undefined) return "n/a";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let index = 0;
    while (value >= 1024 && index < units.length - 1) {
        value /= 1024;
        index++;
    }
    return `${value.toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function percent(value) {
    return value === null || value === undefined ? "n/a" : `${Math.round(value)}%`;
}

function meterFill(value) {
    return Math.max(0, Math.min(100, Number(value) || 0));
}

function metric(kind, label, rawValue, value, detail = "") {
    return `<div class="dasiwa-monitor-metric ${kind}" style="--fill:${meterFill(rawValue)}%" title="${detail}"><span>${label}</span><strong>${value}</strong></div>`;
}

function snapshotMetrics(snapshot) {
    return [
        { kind: "cpu", label: "CPU", value: snapshot.cpu_percent, text: percent(snapshot.cpu_percent), detail: `${snapshot.cpu_count ?? "?"} threads` },
        { kind: "ram", label: "RAM", value: snapshot.ram.percent, text: percent(snapshot.ram.percent), detail: `${bytes(snapshot.ram.used)} / ${bytes(snapshot.ram.total)}` },
        { kind: "swap", label: "SWAP", value: snapshot.swap.percent, text: percent(snapshot.swap.percent), detail: `${bytes(snapshot.swap.used)} / ${bytes(snapshot.swap.total)}` },
        { kind: "disk", label: "DISK", value: snapshot.disk.percent, text: percent(snapshot.disk.percent), detail: `${snapshot.disk.path}: ${bytes(snapshot.disk.used)} / ${bytes(snapshot.disk.total)}` },
        ...snapshot.gpus.flatMap((gpu) => [
            { kind: "gpu-util", label: `GPU${gpu.index} Util`, value: gpu.utilization, text: percent(gpu.utilization), detail: `${gpu.id} — ${gpu.name}; GPU utilization` },
            { kind: "gpu-vram", label: `GPU${gpu.index} VRAM`, value: gpu.memory_percent, text: percent(gpu.memory_percent), detail: `${gpu.id} — ${gpu.name}; VRAM ${bytes(gpu.memory_used)} / ${bytes(gpu.memory_total)}` },
            { kind: "gpu-temp", label: `GPU${gpu.index} Temp`, value: gpu.temperature, text: gpu.temperature === null ? "n/a" : `${Math.round(gpu.temperature)}°`, detail: `${gpu.id} — ${gpu.name}; GPU temperature` },
        ]),
    ];
}

function historyPoint(snapshot) {
    return Object.fromEntries(snapshotMetrics(snapshot).map(({ label, value }) => [label, meterFill(value)]));
}

function sparkline(label) {
    const values = history.map((point) => point[label]).filter((value) => Number.isFinite(value));
    if (!values.length) return "<div class=\"dasiwa-monitor-empty-graph\">Collecting history…</div>";
    const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1) * 100).toFixed(1)},${(100 - value).toFixed(1)}`).join(" ");
    return `<svg class="dasiwa-monitor-graph" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="${label} history"><polyline points="${points}" /></svg>`;
}

function renderLite(panel, snapshot) {
    panel.className = "dasiwa-monitor-display is-lite";
    panel.innerHTML = snapshotMetrics(snapshot).map(({ kind, label, value, text, detail }) => metric(kind, label, value, text, detail)).join("");
    requestAnimationFrame(() => fitPanel(panel));
}

function renderFull(panel, snapshot) {
    panel.className = "dasiwa-monitor-display is-full";
    panel.innerHTML = `
        <div class="dasiwa-monitor-full-header"><strong>System Monitor</strong><span>last ${history.length}s</span></div>
        <div class="dasiwa-monitor-full-grid">
            ${snapshotMetrics(snapshot).map(({ kind, label, value, text, detail }) => `
                <section class="dasiwa-monitor-full-metric ${kind}" title="${detail}">
                    <div><span>${label}</span><strong>${text}</strong></div>
                    <div class="dasiwa-monitor-graph-wrap">${sparkline(label)}</div>
                    <small>${detail}</small>
                    <i style="--fill:${meterFill(value)}%"></i>
                </section>`).join("")}
        </div>`;
    positionFullPanel(panel);
}

function positionFullPanel(panel) {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    const bounds = root.getBoundingClientRect();
    panel.style.top = `${Math.max(8, bounds.bottom + 8)}px`;
    panel.style.right = `${Math.max(8, window.innerWidth - bounds.right)}px`;
}

function render(snapshot = latestSnapshot) {
    const panel = document.getElementById(PANEL_ID);
    if (!panel || !snapshot) return;
    panel.hidden = !settings.enabled;
    if (!settings.enabled) return;
    if (settings.mode === "full") renderFull(panel, snapshot);
    else renderLite(panel, snapshot);
}

function fitPanel(panel) {
    if (settings.mode !== "lite") return;
    const extensionsButton = document.querySelector('button[aria-label="Extensions"]');
    const metrics = [...panel.children];
    metrics.forEach((metric) => metric.hidden = false);
    while (metrics.length && extensionsButton && panel.getBoundingClientRect().left < extensionsButton.getBoundingClientRect().right + 8) {
        metrics.pop().hidden = true;
    }
}

function closeMenu() {
    document.getElementById("dasiwa-monitor-settings-menu")?.remove();
}

function openMenu(button) {
    const existing = document.getElementById("dasiwa-monitor-settings-menu");
    if (existing) {
        existing.remove();
        return;
    }
    const menu = document.createElement("div");
    menu.id = "dasiwa-monitor-settings-menu";
    menu.setAttribute("role", "menu");
    menu.innerHTML = `
        <label><input type="checkbox" ${settings.enabled ? "checked" : ""}> Show system monitor</label>
        <div class="dasiwa-monitor-menu-label">Display mode</div>
        <label><input type="radio" name="dasiwa-monitor-mode" value="lite" ${settings.mode === "lite" ? "checked" : ""}> Lite <small>toolbar meters</small></label>
        <label><input type="radio" name="dasiwa-monitor-mode" value="full" ${settings.mode === "full" ? "checked" : ""}> Full <small>all metrics + 60s graphs</small></label>`;
    button.after(menu);
    menu.querySelector('input[type="checkbox"]').addEventListener("change", (event) => {
        settings.enabled = event.target.checked;
        saveSettings();
        render();
    });
    menu.querySelectorAll('input[type="radio"]').forEach((input) => input.addEventListener("change", (event) => {
        settings.mode = event.target.value;
        settings.enabled = true;
        saveSettings();
        render();
        closeMenu();
    }));
    requestAnimationFrame(() => document.addEventListener("pointerdown", (event) => {
        if (!menu.contains(event.target) && event.target !== button) closeMenu();
    }, { once: true }));
}

function addStyles() {
    const style = document.createElement("style");
    style.textContent = `
        #${ROOT_ID} { position: relative; display: flex; align-items: flex-start; gap: 3px; min-width: 0; margin-right: 6px; color: var(--input-text); font: 600 11px/1 var(--font-inter, sans-serif); }
        #${PANEL_ID}.is-lite { display: flex; align-items: center; gap: 3px; min-width: 0; }
        #${ROOT_ID} .dasiwa-monitor-metric { position: relative; box-sizing: border-box; display: grid; grid-template-columns: 52px 28px; align-items: center; width: 88px; height: 28px; overflow: hidden; padding: 0 3px; border: 1px solid color-mix(in srgb, var(--meter) 65%, var(--border-color)); background: var(--comfy-input-bg); white-space: nowrap; }
        #${ROOT_ID} .dasiwa-monitor-metric::before, #${ROOT_ID} .dasiwa-monitor-full-metric i { content: ""; position: absolute; inset: 0 auto 0 0; width: var(--fill); opacity: .38; background: var(--meter); transition: width .35s ease; }
        #${ROOT_ID} span, #${ROOT_ID} strong { position: relative; z-index: 1; font-variant-numeric: tabular-nums; }
        #${ROOT_ID} .dasiwa-monitor-metric span { color: var(--input-text); font-size: 9px; letter-spacing: .01em; }
        #${ROOT_ID} .dasiwa-monitor-metric strong { text-align: right; font-size: 11px; }
        #${ROOT_ID} .cpu { --meter: #38bdf8; } #${ROOT_ID} .ram { --meter: #a78bfa; } #${ROOT_ID} .swap { --meter: #f59e0b; } #${ROOT_ID} .disk { --meter: #fb7185; }
        #${ROOT_ID} .gpu-util { --meter: #4ade80; } #${ROOT_ID} .gpu-vram { --meter: #22d3ee; } #${ROOT_ID} .gpu-temp { --meter: #fb923c; }
        #${ROOT_ID} [hidden] { display: none !important; }
        #${ROOT_ID} .dasiwa-monitor-settings { position: relative; z-index: 1003; width: 28px; height: 28px; padding: 0; border: 1px solid var(--border-color); color: var(--input-text); background: var(--comfy-input-bg); cursor: pointer; font-size: 15px; } #${ROOT_ID} .dasiwa-monitor-settings:hover, #${ROOT_ID} .dasiwa-monitor-settings:focus-visible { border-color: #22d3ee; color: #22d3ee; outline: none; }
        #dasiwa-monitor-settings-menu { position: absolute; z-index: 1004; top: 32px; right: 0; display: grid; gap: 8px; width: 220px; padding: 10px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 8px 24px #0008; font: 500 12px/1.25 var(--font-inter, sans-serif); } #dasiwa-monitor-settings-menu label { display: grid; grid-template-columns: 16px auto; gap: 6px; align-items: start; cursor: pointer; } #dasiwa-monitor-settings-menu small { grid-column: 2; color: var(--descrip-text, #aaa); } #dasiwa-monitor-settings-menu .dasiwa-monitor-menu-label { padding-top: 4px; border-top: 1px solid var(--border-color); color: var(--descrip-text, #aaa); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
        #${PANEL_ID}.is-full { position: fixed; z-index: 1000; width: min(720px, calc(100vw - 24px)); max-height: calc(100vh - 64px); overflow: auto; padding: 14px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 12px 32px #0009; } #${ROOT_ID} .dasiwa-monitor-full-header { display: flex; justify-content: space-between; margin-bottom: 12px; } #${ROOT_ID} .dasiwa-monitor-full-header strong { font-size: 14px; } #${ROOT_ID} .dasiwa-monitor-full-header span, #${ROOT_ID} .dasiwa-monitor-full-metric small { color: var(--descrip-text, #aaa); }
        #${ROOT_ID} .dasiwa-monitor-full-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; } #${ROOT_ID} .dasiwa-monitor-full-metric { position: relative; min-height: 104px; overflow: hidden; padding: 9px; border: 1px solid color-mix(in srgb, var(--meter) 65%, var(--border-color)); background: var(--comfy-input-bg); } #${ROOT_ID} .dasiwa-monitor-full-metric > div:first-child { display: flex; justify-content: space-between; } #${ROOT_ID} .dasiwa-monitor-full-metric strong { font-size: 16px; } #${ROOT_ID} .dasiwa-monitor-full-metric small { position: relative; z-index: 1; display: block; overflow: hidden; margin-top: 5px; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; } #${ROOT_ID} .dasiwa-monitor-graph-wrap { position: relative; z-index: 1; height: 46px; margin-top: 7px; border-bottom: 1px solid color-mix(in srgb, var(--meter) 25%, transparent); } #${ROOT_ID} .dasiwa-monitor-graph { width: 100%; height: 100%; overflow: visible; fill: none; stroke: var(--meter); stroke-width: 3; vector-effect: non-scaling-stroke; } #${ROOT_ID} .dasiwa-monitor-empty-graph { color: var(--descrip-text, #aaa); font-size: 10px; padding-top: 18px; text-align: center; }
        @media (max-width: 640px) { #${PANEL_ID}.is-full { top: 42px; right: 6px; width: calc(100vw - 12px); padding: 9px; } #${ROOT_ID} .dasiwa-monitor-full-grid { grid-template-columns: 1fr; } }
    `;
    document.head.appendChild(style);
}

app.registerExtension({
    name: "DaSiWa.SystemMonitor",
    async setup() {
        addStyles();
        const root = document.createElement("div");
        root.id = ROOT_ID;
        root.innerHTML = `<div id="${PANEL_ID}" class="dasiwa-monitor-display is-lite">Loading…</div><button class="dasiwa-monitor-settings" type="button" title="DaSiWa node settings" aria-label="DaSiWa node settings">⚙</button>`;
        const settingsButton = root.querySelector("button");
        settingsButton.addEventListener("click", () => openMenu(settingsButton));
        if (!placePanel(root)) {
            document.body.appendChild(root);
            const observer = new MutationObserver(() => {
                if (placePanel(root)) observer.disconnect();
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
        api.addEventListener(EVENT_NAME, (event) => {
            latestSnapshot = event.detail;
            history = [...history, historyPoint(latestSnapshot)].slice(-HISTORY_LENGTH);
            render();
        });
        new ResizeObserver(() => {
            const panel = document.getElementById(PANEL_ID);
            if (!panel) return;
            if (settings.mode === "full") positionFullPanel(panel);
            else fitPanel(panel);
        }).observe(document.documentElement);
        try {
            latestSnapshot = await api.fetchApi("/dasiwa/system-monitor").then((response) => response.json());
            history = [historyPoint(latestSnapshot)];
            render();
        } catch (error) {
            document.getElementById(PANEL_ID).textContent = "DaSiWa System Monitor is waiting for backend telemetry.";
            console.warn("DaSiWa System Monitor", error);
        }
    },
});