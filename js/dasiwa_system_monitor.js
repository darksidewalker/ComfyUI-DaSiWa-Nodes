import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT_NAME = "dasiwa.system_monitor";
const ROOT_ID = "dasiwa-system-monitor";

function placePanel(panel) {
    const legacyTopbar = document.querySelector('[data-testid="legacy-topbar-container"] > .flex');
    if (legacyTopbar) {
        legacyTopbar.prepend(panel);
        return true;
    }
    const extensionsButton = document.querySelector('button[aria-label="Extensions"]');
    if (!extensionsButton?.parentElement) return false;
    extensionsButton.before(panel);
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

function render(snapshot) {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    const gpus = snapshot.gpus.flatMap((gpu) => [
        metric("gpu-util", `GPU${gpu.index} Util`, gpu.utilization, percent(gpu.utilization), `${gpu.id} — ${gpu.name}; GPU utilization`),
        metric("gpu-vram", `GPU${gpu.index} VRAM`, gpu.memory_percent, percent(gpu.memory_percent), `${gpu.id} — ${gpu.name}; VRAM ${bytes(gpu.memory_used)} / ${bytes(gpu.memory_total)}`),
        metric("gpu-temp", `GPU${gpu.index} Temp`, gpu.temperature, gpu.temperature === null ? "n/a" : `${Math.round(gpu.temperature)}°`, `${gpu.id} — ${gpu.name}; GPU temperature`),
    ]).join("");
    root.innerHTML = [
        metric("cpu", "CPU", snapshot.cpu_percent, percent(snapshot.cpu_percent), `${snapshot.cpu_count ?? "?"} threads`),
        metric("ram", "RAM", snapshot.ram.percent, percent(snapshot.ram.percent), `${bytes(snapshot.ram.used)} / ${bytes(snapshot.ram.total)}`),
        metric("swap", "SWAP", snapshot.swap.percent, percent(snapshot.swap.percent), `${bytes(snapshot.swap.used)} / ${bytes(snapshot.swap.total)}`),
        metric("disk", "DISK", snapshot.disk.percent, percent(snapshot.disk.percent), `${snapshot.disk.path}: ${bytes(snapshot.disk.used)} / ${bytes(snapshot.disk.total)}`),
        gpus,
    ].join("");
    requestAnimationFrame(() => fitPanel(root));
}

function fitPanel(panel) {
    const extensionsButton = document.querySelector('button[aria-label="Extensions"]');
    const metrics = [...panel.children];
    metrics.forEach((metric) => metric.hidden = false);
    while (metrics.length && extensionsButton && panel.getBoundingClientRect().left < extensionsButton.getBoundingClientRect().right + 8) {
        metrics.pop().hidden = true;
    }
}

function addStyles() {
    const style = document.createElement("style");
    style.textContent = `
        #${ROOT_ID} { display: flex; align-items: center; gap: 3px; min-width: 0; margin-right: 6px; color: var(--input-text); font: 600 11px/1 var(--font-inter, sans-serif); }
        #${ROOT_ID} .dasiwa-monitor-metric { position: relative; box-sizing: border-box; display: grid; grid-template-columns: 52px 28px; align-items: center; width: 88px; height: 28px; overflow: hidden; padding: 0 3px; border: 1px solid color-mix(in srgb, var(--meter) 65%, var(--border-color)); background: var(--comfy-input-bg); white-space: nowrap; }
        #${ROOT_ID} .dasiwa-monitor-metric::before { content: ""; position: absolute; inset: 0 auto 0 0; width: var(--fill); opacity: .38; background: var(--meter); transition: width .35s ease; }
        #${ROOT_ID} span, #${ROOT_ID} strong { position: relative; z-index: 1; font-variant-numeric: tabular-nums; }
        #${ROOT_ID} span { color: var(--input-text); font-size: 9px; letter-spacing: .01em; }
        #${ROOT_ID} strong { text-align: right; font-size: 11px; }
        #${ROOT_ID} .cpu { --meter: #38bdf8; } #${ROOT_ID} .ram { --meter: #a78bfa; }
        #${ROOT_ID} .swap { --meter: #f59e0b; } #${ROOT_ID} .disk { --meter: #fb7185; }
        #${ROOT_ID} .gpu-util { --meter: #4ade80; } #${ROOT_ID} .gpu-vram { --meter: #22d3ee; } #${ROOT_ID} .gpu-temp { --meter: #fb923c; }
        #${ROOT_ID} [hidden] { display: none; }
    `;
    document.head.appendChild(style);
}

app.registerExtension({
    name: "DaSiWa.SystemMonitor",
    async setup() {
        addStyles();
        const panel = document.createElement("div");
        panel.id = ROOT_ID;
        panel.innerHTML = "Loading…";
        if (!placePanel(panel)) {
            document.body.appendChild(panel);
            const observer = new MutationObserver(() => {
                if (placePanel(panel)) observer.disconnect();
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
        api.addEventListener(EVENT_NAME, (event) => render(event.detail));
        new ResizeObserver(() => fitPanel(panel)).observe(document.documentElement);
        try {
            render(await api.fetchApi("/dasiwa/system-monitor").then((response) => response.json()));
        } catch (error) {
            panel.textContent = "DaSiWa System Monitor is waiting for backend telemetry.";
            console.warn("DaSiWa System Monitor", error);
        }
    },
});
