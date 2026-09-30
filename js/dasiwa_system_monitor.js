import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT_NAME = "dasiwa.system_monitor";
const ROOT_ID = "dasiwa-system-monitor";
const PANEL_ID = "dasiwa-system-monitor-panel";
const SETTINGS_KEY = "dasiwa.system_monitor.settings";
const MONITOR_ENABLED_SETTING = "DaSiWa.SystemMonitor.Enabled";
const BACKGROUND_OPACITY_SETTING = "DaSiWa.SystemMonitor.BackgroundOpacity";
const CONTENT_OPACITY_SETTING = "DaSiWa.SystemMonitor.ContentOpacity";
const OPACITY_ATTRS = { min: 0, max: 100, step: 1 };
const MONITOR_ENABLED_ENDPOINT = "/dasiwa/system-monitor/enabled";
const HISTORY_LENGTH = 60;
const DEFAULT_SETTINGS = {
    mode: "lite", placement: "top", dockSide: "top", orientation: "horizontal",
    widgets: {}, x: 24, y: 60, width: null,
};

let settings = loadSettings();
let latestSnapshot = null;
let history = [];
let monitorRoot = null;
let fullOverlay = null;
let monitorEventListener = null;
let monitorResizeObserver = null;
let monitorToolbarObserver = null;
let monitorMenuOutsideListener = null;

function loadSettings() {
    try {
        const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY));
        return {
            mode: ["lite", "ultra", "full"].includes(saved?.mode) ? saved.mode : DEFAULT_SETTINGS.mode,
            placement: saved?.placement === "floating" ? "floating" : DEFAULT_SETTINGS.placement,
            dockSide: ["top", "left", "right"].includes(saved?.dockSide) ? saved.dockSide : DEFAULT_SETTINGS.dockSide,
            orientation: saved?.orientation === "horizontal" || saved?.orientation === "vertical" ? saved.orientation : DEFAULT_SETTINGS.orientation,
            widgets: saved?.widgets && typeof saved.widgets === "object" ? saved.widgets : {},
            x: Number.isFinite(saved?.x) ? saved.x : DEFAULT_SETTINGS.x,
            y: Number.isFinite(saved?.y) ? saved.y : DEFAULT_SETTINGS.y,
            width: Number.isFinite(saved?.width) && saved.width >= 120 ? saved.width : null,
        };
    } catch {
        return { ...DEFAULT_SETTINGS };
    }
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function applyOrientation(root) {
    root.classList.toggle("is-vertical", settings.orientation === "vertical");
    root.classList.toggle("is-horizontal", settings.orientation !== "vertical");
}

function hideSideDocks() {
    ["left", "right"].forEach((side) => {
        const dock = document.getElementById(`dasiwa-monitor-dock-${side}`);
        if (dock) dock.hidden = true;
    });
}

function applySize(root) {
    root.classList.toggle("is-ultra", settings.mode === "ultra");
    root.classList.toggle("is-resized", settings.mode !== "ultra" && settings.width !== null);
    root.style.width = settings.mode === "ultra" ? `${Math.min(240, window.innerWidth - 16)}px`
        : settings.width === null ? "" : `${Math.min(settings.width, window.innerWidth - 16)}px`;
    root.style.height = "";
}

function topDock() {
    const row = document.querySelector('[data-testid="top-menu-actionbars"]');
    if (!row?.parentElement) return null;
    let dock = document.getElementById("dasiwa-monitor-dock-top");
    if (!dock) {
        dock = document.createElement("div");
        dock.id = "dasiwa-monitor-dock-top";
    }
    if (dock.previousElementSibling !== row) row.after(dock);
    return dock;
}

function placePanel(root, side = settings.dockSide) {
    settings.dockSide = side;
    settings.orientation = side === "top" ? "horizontal" : "vertical";
    applyOrientation(root);
    root.classList.toggle("is-dock-top", side === "top");
    root.classList.toggle("is-dock-left", side === "left");
    root.classList.toggle("is-dock-right", side === "right");
    root.classList.remove("is-floating");
    root.style.left = "";
    root.style.top = "";
    root.style.transform = "";
    hideSideDocks();
    const currentTopDock = document.getElementById("dasiwa-monitor-dock-top");
    if (currentTopDock) currentTopDock.hidden = side !== "top";
    if (side === "left" || side === "right") {
        const dock = document.getElementById(`dasiwa-monitor-dock-${side}`);
        if (!dock) return false;
        dock.hidden = false;
        dock.appendChild(root);
        return true;
    }
    const dock = topDock();
    if (!dock) return false;
    dock.hidden = false;
    dock.appendChild(root);
    return true;
}

function dockPanel(root) {
    if (!placePanel(root, "top")) return false;
    settings.placement = "top";
    saveSettings();
    return true;
}

function floatPanel(root, x, y) {
    document.body.appendChild(root);
    root.classList.remove("is-dock-top", "is-dock-left", "is-dock-right");
    root.classList.add("is-floating");
    const dock = document.getElementById("dasiwa-monitor-dock-top");
    if (dock) dock.hidden = true;
    [...root.querySelector(`#${PANEL_ID}`)?.children ?? []].forEach((metric) => metric.hidden = false);
    settings.placement = "floating";
    settings.x = Math.max(8, Math.min(x, window.innerWidth - root.offsetWidth - 8));
    settings.y = Math.max(48, Math.min(y, window.innerHeight - root.offsetHeight - 8));
    root.style.transform = "";
    root.style.left = `${Math.round(settings.x)}px`;
    root.style.top = `${Math.round(settings.y)}px`;
}

function dockTarget(side) {
    return document.getElementById(`dasiwa-monitor-dock-target-${side}`);
}

function dockSideAtPoint(x, y) {
    if (y <= 48) return "top";
    if (x <= 64) return "left";
    if (x >= window.innerWidth - 64) return "right";
    return null;
}

function showDockTargets(activeSide = null) {
    ["top", "left", "right"].forEach((side) => {
        const target = dockTarget(side);
        target.hidden = false;
        target.classList.toggle("is-active", side === activeSide);
    });
}

function hideDockTargets() {
    ["top", "left", "right"].forEach((side) => {
        const target = dockTarget(side);
        target.hidden = true;
        target.classList.remove("is-active");
    });
}

function enablePanelDrag(root) {
    const handle = root.querySelector(".dasiwa-monitor-drag-handle");
    handle.title = "Drag to float; drop on Dock to top to return it to the toolbar";
    handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        const bounds = root.getBoundingClientRect();
        const offsetX = event.clientX - bounds.left;
        const offsetY = event.clientY - bounds.top;
        let dragging = false;
        const move = (moveEvent) => {
            if (!dragging && Math.hypot(moveEvent.clientX - event.clientX, moveEvent.clientY - event.clientY) < 4) return;
            if (!dragging) {
                dragging = true;
                floatPanel(root, moveEvent.clientX - offsetX, moveEvent.clientY - offsetY);
                showDockTargets(dockSideAtPoint(moveEvent.clientX, moveEvent.clientY));
            } else {
                floatPanel(root, moveEvent.clientX - offsetX, moveEvent.clientY - offsetY);
                showDockTargets(dockSideAtPoint(moveEvent.clientX, moveEvent.clientY));
            }
        };
        const end = (endEvent) => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
            if (dragging) {
                if (endEvent.type === "pointercancel") {
                    saveSettings();
                    hideDockTargets();
                    return;
                }
                const side = dockSideAtPoint(endEvent.clientX, endEvent.clientY);
                if (side === "top") dockPanel(root);
                else if (side) {
                    settings.dockSide = side;
                    settings.placement = "top";
                    placePanel(root, side);
                    saveSettings();
                }
                else saveSettings();
                hideDockTargets();
            }
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    });
}

function meterWidthLimits(root) {
    const panel = root.querySelector(`#${PANEL_ID}`);
    const widths = [...panel.children].map((metric) => metric.getBoundingClientRect().width);
    const gap = parseFloat(getComputedStyle(panel).columnGap) || 0;
    const chrome = root.querySelector(".dasiwa-monitor-drag-handle").offsetWidth
        + root.querySelector(".dasiwa-monitor-settings").offsetWidth
        + 15 // Corner grip keeps its own gutter, never covering the last meter.
        + 2 * (parseFloat(getComputedStyle(root).columnGap) || 0);
    return {
        min: Math.max(120, chrome + Math.max(0, ...widths)),
        max: Math.max(120, chrome + widths.reduce((total, width) => total + width, 0) + gap * Math.max(0, widths.length - 1)),
    };
}

function enablePanelResize(root) {
    const handle = root.querySelector(".dasiwa-monitor-resize-handle");
    handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || settings.mode === "ultra") return;
        event.preventDefault();
        event.stopPropagation();
        const startWidth = root.getBoundingClientRect().width;
        const startX = event.clientX;
        const move = (next) => {
            const rect = root.getBoundingClientRect();
            const { min, max } = meterWidthLimits(root);
            const containerRight = settings.placement === "floating" ? window.innerWidth : root.parentElement.getBoundingClientRect().right;
            const available = Math.min(window.innerWidth, containerRight) - rect.left - 8;
            settings.width = Math.max(min, Math.min(startWidth + next.clientX - startX, max, available));
            applySize(root);
            if (settings.placement === "floating") {
                settings.x = Math.max(8, Math.min(settings.x, window.innerWidth - root.offsetWidth - 8));
                settings.y = Math.max(8, Math.min(settings.y, window.innerHeight - root.offsetHeight - 8));
                root.style.left = `${Math.round(settings.x)}px`;
                root.style.top = `${Math.round(settings.y)}px`;
            }
        };
        const end = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
            saveSettings();
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    });
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

function mbPerSec(value) {
    if (value === null || value === undefined) return "n/a";
    const rounded = Math.round(value * 10) / 10;
    if (rounded >= 1000) return `${(rounded / 1000).toFixed(1)} GB/s`;
    return `${rounded.toFixed(1).padStart(4, "\u00A0")} MB/s`;
}

function percent(value) {
    return value === null || value === undefined ? "n/a" : `${String(Math.round(value)).padStart(3, "\u00A0")}%`;
}

function celsius(value) {
    return value === null || value === undefined ? "n/a" : `${String(Math.round(value)).padStart(3, "\u00A0")}°`;
}

function meterFill(value) {
    return Math.max(0, Math.min(100, Number(value) || 0));
}

function metric(kind, label, rawValue, value, detail = "") {
    return `<div class="dasiwa-monitor-metric ${kind}" style="--fill:${meterFill(rawValue)}%" title="${detail}"><span>${label}</span><strong>${value}</strong></div>`;
}

function isWidgetVisible(id) {
    return settings.widgets[id] !== false;
}

function allSnapshotMetrics(snapshot) {
    return [
        { id: "cpu", kind: "cpu", label: "CPU", value: snapshot.cpu_percent, text: percent(snapshot.cpu_percent), detail: `${snapshot.cpu_count ?? "?"} threads` },
        { id: "ram", kind: "ram", label: "RAM", value: snapshot.ram.percent, text: percent(snapshot.ram.percent), detail: `${bytes(snapshot.ram.used)} / ${bytes(snapshot.ram.total)}` },
        { id: "swap", kind: "swap", label: "SWAP", value: snapshot.swap.percent, text: percent(snapshot.swap.percent), detail: `${bytes(snapshot.swap.used)} / ${bytes(snapshot.swap.total)}` },
        ...(snapshot.disks ?? []).flatMap((disk) => [
            { id: `disk:${disk.path}`, kind: "disk", label: `DISK Space ${disk.path}`, value: disk.percent, text: percent(disk.percent), detail: `${disk.device}: ${bytes(disk.used)} / ${bytes(disk.total)}` },
            { id: `disk-rd:${disk.path}`, kind: "disk-rd", label: `DISK Read ${disk.path}`, value: Math.min(disk.read_mb_s, 100), text: mbPerSec(disk.read_mb_s), detail: `${disk.path}: disk read throughput` },
            { id: `disk-wr:${disk.path}`, kind: "disk-wr", label: `DISK Write ${disk.path}`, value: Math.min(disk.write_mb_s, 100), text: mbPerSec(disk.write_mb_s), detail: `${disk.path}: disk write throughput` },
        ]),
        ...snapshot.gpus.flatMap((gpu) => [
            { id: `gpu-util:${gpu.id}`, kind: "gpu-util", label: `GPU${gpu.index} Util`, value: gpu.utilization, text: percent(gpu.utilization), detail: `${gpu.id} — ${gpu.name}; GPU utilization` },
            { id: `gpu-vram:${gpu.id}`, kind: "gpu-vram", label: `GPU${gpu.index} VRAM`, value: gpu.memory_percent, text: percent(gpu.memory_percent), detail: `${gpu.id} — ${gpu.name}; VRAM ${bytes(gpu.memory_used)} / ${bytes(gpu.memory_total)}` },
            { id: `gpu-temp:${gpu.id}`, kind: "gpu-temp", label: `GPU${gpu.index} Temp`, value: gpu.temperature, text: celsius(gpu.temperature), detail: `${gpu.id} — ${gpu.name}; GPU temperature` },
        ]),
    ];
}

function snapshotMetrics(snapshot) {
    return allSnapshotMetrics(snapshot).filter(({ id }) => isWidgetVisible(id));
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
}

function renderUltra(panel, snapshot) {
    panel.className = "dasiwa-monitor-display is-ultra";
    panel.innerHTML = snapshotMetrics(snapshot).map(({ kind, label, value, text, detail }) =>
        `<div class="dasiwa-monitor-ultra-row ${kind}" style="--fill:${meterFill(value)}%" title="${detail}"><span>${label}</span><strong>${text}</strong></div>`
    ).join("");
}

function renderFull(panel, snapshot) {
    const root = monitorRoot;
    if (settings.dockSide === "top") {
        // Toolbar clips overflow — render full view as a body-level overlay anchored to the toolbar.
        // Keep inline panel compact so it doesn't break toolbar layout.
        panel.className = "dasiwa-monitor-display is-lite";
        panel.innerHTML = `<span class="dasiwa-monitor-metric cpu" style="--fill:0%"><span>FULL</span><strong>ON</strong></span>`;
        if (!fullOverlay) {
            fullOverlay = document.createElement("div");
            fullOverlay.id = "dasiwa-monitor-full-overlay";
            document.body.appendChild(fullOverlay);
            applyOpacity();
        }
        fullOverlay.hidden = false;
        const rect = root.getBoundingClientRect();
        fullOverlay.style.top = `${Math.round(rect.bottom + 8)}px`;
        fullOverlay.style.left = `${Math.round(rect.left)}px`;
        fullOverlay.innerHTML = `
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
    } else {
        // Side docks don't clip — render inline as before.
        if (fullOverlay) fullOverlay.hidden = true;
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
}

function positionFullPanel(panel) {
    panel.style.top = "calc(100% + 8px)";
    if (settings.dockSide === "left") {
        panel.style.left = "calc(100% + 8px)";
        panel.style.right = "auto";
    } else {
        panel.style.left = "auto";
        panel.style.right = "0";
    }
}

function render(snapshot = latestSnapshot) {
    const panel = document.getElementById(PANEL_ID);
    if (!panel || !snapshot) return;
    applySize(monitorRoot);
    if (settings.mode === "full") renderFull(panel, snapshot);
    else {
        if (fullOverlay) fullOverlay.hidden = true;
        if (settings.mode === "ultra") renderUltra(panel, snapshot);
        else renderLite(panel, snapshot);
    }
}

function opacityValue(id) {
    const saved = app.ui.settings.getSettingValue(id);
    const value = Number(saved);
    return saved != null && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 100;
}

function applyOpacity() {
    const background = `${opacityValue(BACKGROUND_OPACITY_SETTING)}%`;
    const content = opacityValue(CONTENT_OPACITY_SETTING) / 100;
    for (const element of [monitorRoot, fullOverlay]) {
        if (!element) continue;
        element.style.setProperty("--dasiwa-monitor-bg-opacity", background);
        element.style.setProperty("--dasiwa-monitor-content-opacity", content);
        element.style.setProperty("--dasiwa-monitor-content-alpha", `${Math.round(content * 100)}%`);
    }
    for (const [id, value] of [[BACKGROUND_OPACITY_SETTING, background], [CONTENT_OPACITY_SETTING, `${Math.round(content * 100)}%`]]) {
        const input = document.querySelector(`#dasiwa-monitor-settings-menu input[data-opacity-setting="${id}"]`);
        if (input) {
            input.value = opacityValue(id);
            input.nextElementSibling.textContent = value;
        }
    }
}

function positionMenu(button, menu) {
    const rect = button.getBoundingClientRect();
    const margin = 8;
    const width = menu.offsetWidth;
    const height = menu.offsetHeight;
    const below = window.innerHeight - rect.bottom - margin;
    const above = rect.top - margin;
    const preferredTop = below >= height || below >= above ? rect.bottom + 4 : rect.top - height - 4;
    menu.style.left = `${Math.round(Math.max(margin, Math.min(rect.left, window.innerWidth - width - margin)))}px`;
    menu.style.top = `${Math.round(Math.max(margin, Math.min(preferredTop, window.innerHeight - height - margin)))}px`;
}

function closeMenu() {
    if (monitorMenuOutsideListener) document.removeEventListener("pointerdown", monitorMenuOutsideListener);
    monitorMenuOutsideListener = null;
    window.removeEventListener("resize", repositionMenu);
    window.removeEventListener("scroll", repositionMenu, true);
    document.getElementById("dasiwa-monitor-settings-menu")?.remove();
}

function repositionMenu() {
    const button = monitorRoot?.querySelector(".dasiwa-monitor-settings");
    const menu = document.getElementById("dasiwa-monitor-settings-menu");
    if (button && menu) positionMenu(button, menu);
}

function isMonitorEnabled() {
    return app.ui.settings.getSettingValue(MONITOR_ENABLED_SETTING) !== false;
}

async function syncMonitorEnabled(enabled) {
    try {
        await api.fetchApi(MONITOR_ENABLED_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled }),
        });
    } catch (error) {
        console.warn("DaSiWa System Monitor", error);
    }
}

function unmountMonitor() {
    closeMenu();
    if (monitorEventListener) api.removeEventListener(EVENT_NAME, monitorEventListener);
    monitorEventListener = null;
    monitorResizeObserver?.disconnect();
    monitorResizeObserver = null;
    monitorToolbarObserver?.disconnect();
    monitorToolbarObserver = null;
    monitorRoot?.remove();
    monitorRoot = null;
    fullOverlay?.remove();
    fullOverlay = null;
    ["dasiwa-monitor-dock-top", "dasiwa-monitor-dock-left", "dasiwa-monitor-dock-right", "dasiwa-monitor-dock-target-top", "dasiwa-monitor-dock-target-left", "dasiwa-monitor-dock-target-right"].forEach((id) => document.getElementById(id)?.remove());
    latestSnapshot = null;
    history = [];
}

async function setMonitorEnabled(enabled) {
    if (enabled) {
        await syncMonitorEnabled(true);
        await mountMonitor();
    } else {
        unmountMonitor();
        await syncMonitorEnabled(false);
    }
}

function openMenu(button) {
    const existing = document.getElementById("dasiwa-monitor-settings-menu");
    if (existing) {
        closeMenu();
        return;
    }
    const root = button.closest(`#${ROOT_ID}`);
    const widgetControls = latestSnapshot ? allSnapshotMetrics(latestSnapshot).map(({ id, label }) =>
        `<label><input type="checkbox" data-widget-id="${id}" ${isWidgetVisible(id) ? "checked" : ""}> ${label}</label>`
    ).join("") : "<small>Waiting for telemetry…</small>";
    const menu = document.createElement("div");
    menu.id = "dasiwa-monitor-settings-menu";
    menu.setAttribute("role", "menu");
    menu.innerHTML = `
        <div class="dasiwa-monitor-menu-label">Display mode</div>
        <button type="button" class="dasiwa-monitor-reset">Reset to default Lite bar</button>
        <label><input type="radio" name="dasiwa-monitor-mode" value="lite" ${settings.mode === "lite" ? "checked" : ""}> Lite <small>toolbar meters</small></label>
        <label><input type="radio" name="dasiwa-monitor-mode" value="ultra" ${settings.mode === "ultra" ? "checked" : ""}> Ultra compact <small>all enabled metric lines</small></label>
        <label><input type="radio" name="dasiwa-monitor-mode" value="full" ${settings.mode === "full" ? "checked" : ""}> Full <small>all metrics + 60s graphs</small></label>
        <div class="dasiwa-monitor-menu-label">Dock</div>
        <label><input type="radio" name="dasiwa-monitor-dock" value="top" ${settings.dockSide === "top" ? "checked" : ""}> Top toolbar</label>
        <label><input type="radio" name="dasiwa-monitor-dock" value="left" ${settings.dockSide === "left" ? "checked" : ""}> Left side</label>
        <label><input type="radio" name="dasiwa-monitor-dock" value="right" ${settings.dockSide === "right" ? "checked" : ""}> Right side</label>
        <div class="dasiwa-monitor-menu-note">Layout is automatic: horizontal at top; vertical at either side.</div>
        <div class="dasiwa-monitor-menu-label">Transparency</div>
        <label class="dasiwa-monitor-opacity-row">Background <span><input type="range" min="0" max="100" step="1" data-opacity-setting="${BACKGROUND_OPACITY_SETTING}" value="${opacityValue(BACKGROUND_OPACITY_SETTING)}" aria-label="Monitor background opacity"><output>${opacityValue(BACKGROUND_OPACITY_SETTING)}%</output></span></label>
        <label class="dasiwa-monitor-opacity-row">Drawing / text / lines <span><input type="range" min="0" max="100" step="1" data-opacity-setting="${CONTENT_OPACITY_SETTING}" value="${opacityValue(CONTENT_OPACITY_SETTING)}" aria-label="Monitor content opacity"><output>${opacityValue(CONTENT_OPACITY_SETTING)}%</output></span></label>
        <div class="dasiwa-monitor-menu-label">Widgets</div>
        <div class="dasiwa-monitor-widgets-list">${widgetControls}</div>`;
    document.body.appendChild(menu);
    positionMenu(button, menu);
    window.addEventListener("resize", repositionMenu);
    window.addEventListener("scroll", repositionMenu, true);
    menu.querySelector(".dasiwa-monitor-reset").addEventListener("click", () => {
        settings.mode = DEFAULT_SETTINGS.mode;
        settings.width = null;
        settings.placement = DEFAULT_SETTINGS.placement;
        applySize(root);
        placePanel(root, DEFAULT_SETTINGS.dockSide);
        saveSettings();
        render();
        closeMenu();
    });
    menu.querySelectorAll("[data-opacity-setting]").forEach((input) => input.addEventListener("input", () => {
        input.nextElementSibling.textContent = `${input.value}%`;
        void app.ui.settings.setSettingValue(input.dataset.opacitySetting, Number(input.value));
    }));
    menu.querySelectorAll('input[name="dasiwa-monitor-mode"]').forEach((input) => input.addEventListener("change", (event) => {
        settings.mode = event.target.value;
        if (settings.mode === "ultra" || settings.mode === "lite") {
            settings.placement = "top";
            placePanel(root, settings.mode === "ultra" ? "right" : "top");
        }
        saveSettings();
        render();
        closeMenu();
    }));
    menu.querySelectorAll('input[name="dasiwa-monitor-dock"]').forEach((input) => input.addEventListener("change", (event) => {
        settings.dockSide = event.target.value;
        settings.placement = "top";
        placePanel(root, settings.dockSide);
        saveSettings();
        render();
        closeMenu();
    }));

    menu.querySelectorAll('[data-widget-id]').forEach((input) => input.addEventListener("change", (event) => {
        settings.widgets[event.target.dataset.widgetId] = event.target.checked;
        saveSettings();
        render();
    }));
    monitorMenuOutsideListener = (event) => {
        if (!menu.contains(event.target) && !button.contains(event.target)) closeMenu();
    };
    document.addEventListener("pointerdown", monitorMenuOutsideListener);
}

function addStyles() {
    if (document.getElementById("dasiwa-system-monitor-styles")) return;
    const style = document.createElement("style");
    style.id = "dasiwa-system-monitor-styles";
    style.textContent = `
        #${ROOT_ID} { position: relative; display: flex; align-items: flex-start; gap: 3px; min-width: 0; margin-right: 6px; color: var(--input-text); font: 600 11px/1 var(--font-inter, sans-serif); } #${ROOT_ID}.is-floating { position: fixed; z-index: 1002; margin: 0; } #${ROOT_ID} .dasiwa-monitor-drag-handle { width: 10px; min-height: 36px; border: 1px solid var(--border-color); background: var(--comfy-input-bg); cursor: grab; touch-action: none; } #${ROOT_ID} .dasiwa-monitor-drag-handle::after { content: "⠿"; display: block; padding-top: 7px; color: var(--descrip-text, #aaa); font-size: 13px; text-align: center; } #${ROOT_ID} .dasiwa-monitor-drag-handle:active { cursor: grabbing; }
        #dasiwa-monitor-dock-left, #dasiwa-monitor-dock-right { position: fixed; z-index: 1002; top: 120px; bottom: 8px; display: flex; align-items: flex-start; pointer-events: none; } #dasiwa-monitor-dock-left { left: 72px; } #dasiwa-monitor-dock-right { right: 72px; } #dasiwa-monitor-dock-left > #${ROOT_ID}, #dasiwa-monitor-dock-right > #${ROOT_ID} { pointer-events: auto; margin: 0; } #${ROOT_ID}.is-vertical .dasiwa-monitor-display.is-lite { flex-direction: column; align-items: stretch; } #${ROOT_ID}.is-horizontal .dasiwa-monitor-display.is-lite { flex-direction: row; }
        .dasiwa-monitor-dock-target { position: fixed; z-index: 1005; box-sizing: border-box; border: 2px dashed #22d3ee; background: #0891b533; color: #cffafe; font: 700 12px/38px var(--font-inter, sans-serif); letter-spacing: .04em; text-align: center; pointer-events: none; } .dasiwa-monitor-dock-target.is-active { background: #0891b588; border-style: solid; } #dasiwa-monitor-dock-target-top { top: 0; left: 0; width: 100vw; height: 42px; } #dasiwa-monitor-dock-target-left, #dasiwa-monitor-dock-target-right { top: 48px; bottom: 0; width: 56px; writing-mode: vertical-rl; padding-top: 12px; } #dasiwa-monitor-dock-target-left { left: 0; } #dasiwa-monitor-dock-target-right { right: 0; }
        #${PANEL_ID}.is-lite { display: flex; align-items: center; gap: 3px; min-width: 0; }
        #${ROOT_ID} .dasiwa-monitor-metric { position: relative; box-sizing: border-box; display: grid; grid-template-columns: max-content max-content; gap: 4px; align-items: center; flex: none; min-width: 56px; height: 36px; overflow: hidden; padding: 0 6px 0 4px; border: 1px solid color-mix(in srgb, var(--meter) 65%, var(--border-color)); background: var(--comfy-input-bg); white-space: nowrap; }
        #${ROOT_ID} .dasiwa-monitor-metric::before, #${ROOT_ID} .dasiwa-monitor-full-metric i { content: ""; position: absolute; inset: 0 auto 0 0; width: var(--fill); opacity: .38; background: var(--meter); transition: width .35s ease; }
        #${ROOT_ID} span, #${ROOT_ID} strong { position: relative; z-index: 1; font-variant-numeric: tabular-nums; }
        #${ROOT_ID} .dasiwa-monitor-metric span { color: var(--input-text); font-size: 9px; letter-spacing: .01em; }
        #${ROOT_ID} .dasiwa-monitor-metric strong { display: inline-block; width: 5ch; text-align: right; font-size: 11px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace; }
        #${ROOT_ID} .dasiwa-monitor-metric.disk-rd strong, #${ROOT_ID} .dasiwa-monitor-metric.disk-wr strong { width: 12ch; }
        #${ROOT_ID} .cpu { --meter: #38bdf8; } #${ROOT_ID} .ram { --meter: #a78bfa; } #${ROOT_ID} .swap { --meter: #f59e0b; } #${ROOT_ID} .disk { --meter: #fb7185; } #${ROOT_ID} .disk-rd { --meter: #34d399; } #${ROOT_ID} .disk-wr { --meter: #f472b6; }
        #${ROOT_ID} .gpu-util { --meter: #4ade80; } #${ROOT_ID} .gpu-vram { --meter: #22d3ee; } #${ROOT_ID} .gpu-temp { --meter: #fb923c; }
        #${ROOT_ID} [hidden] { display: none !important; }
        #${ROOT_ID} .dasiwa-monitor-settings { position: relative; z-index: 1003; width: 36px; height: 36px; padding: 0; border: 1px solid var(--border-color); color: var(--input-text); background: var(--comfy-input-bg); cursor: pointer; font-size: 15px; } #${ROOT_ID} .dasiwa-monitor-settings:hover, #${ROOT_ID} .dasiwa-monitor-settings:focus-visible { border-color: #22d3ee; color: #22d3ee; outline: none; }
        #dasiwa-monitor-settings-menu { position: fixed; z-index: 1004; display: grid; gap: 8px; width: 220px; padding: 10px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 8px 24px #0008; font: 500 12px/1.25 var(--font-inter, sans-serif); } #dasiwa-monitor-settings-menu label { display: grid; grid-template-columns: 16px auto; gap: 6px; align-items: start; cursor: pointer; } #dasiwa-monitor-settings-menu small { grid-column: 2; color: var(--descrip-text, #aaa); } #dasiwa-monitor-settings-menu .dasiwa-monitor-menu-label { padding-top: 4px; border-top: 1px solid var(--border-color); color: var(--descrip-text, #aaa); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; } #dasiwa-monitor-settings-menu .dasiwa-monitor-menu-note { color: var(--descrip-text, #aaa); font-size: 11px; line-height: 1.35; } #dasiwa-monitor-settings-menu .dasiwa-monitor-widgets-list { display: grid; gap: 3px; max-height: 180px; overflow-y: auto; }
        #dasiwa-monitor-full-overlay { position: fixed; z-index: 1003; width: min(720px, calc(100vw - 24px)); max-height: calc(100vh - 64px); overflow: auto; padding: 14px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 12px 32px #0009; color: var(--input-text); font: 600 11px/1 var(--font-inter, sans-serif); }
        #dasiwa-monitor-full-overlay .cpu { --meter: #38bdf8; } #dasiwa-monitor-full-overlay .ram { --meter: #a78bfa; } #dasiwa-monitor-full-overlay .swap { --meter: #f59e0b; } #dasiwa-monitor-full-overlay .disk { --meter: #fb7185; } #dasiwa-monitor-full-overlay .disk-rd { --meter: #34d399; } #dasiwa-monitor-full-overlay .disk-wr { --meter: #f472b6; }
        #dasiwa-monitor-full-overlay .gpu-util { --meter: #4ade80; } #dasiwa-monitor-full-overlay .gpu-vram { --meter: #22d3ee; } #dasiwa-monitor-full-overlay .gpu-temp { --meter: #fb923c; }
        #dasiwa-monitor-full-overlay span, #dasiwa-monitor-full-overlay strong { position: relative; z-index: 1; font-variant-numeric: tabular-nums; }
        #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric i { content: ""; position: absolute; inset: 0 auto 0 0; width: var(--fill); opacity: .38; background: var(--meter); transition: width .35s ease; }
        #${PANEL_ID}.is-full { position: absolute; z-index: 1000; width: min(720px, calc(100vw - 24px)); max-height: calc(100vh - 64px); overflow: auto; padding: 14px; border: 1px solid var(--border-color); background: var(--comfy-menu-bg, #202020); box-shadow: 0 12px 32px #0009; }
        #${ROOT_ID} .dasiwa-monitor-full-header, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-header { display: flex; justify-content: space-between; margin-bottom: 12px; } #${ROOT_ID} .dasiwa-monitor-full-header strong, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-header strong { font-size: 14px; } #${ROOT_ID} .dasiwa-monitor-full-header span, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-header span, #${ROOT_ID} .dasiwa-monitor-full-metric small, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric small { color: var(--descrip-text, #aaa); }
        #${ROOT_ID} .dasiwa-monitor-full-grid, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; }
        #${ROOT_ID} .dasiwa-monitor-full-metric, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric { position: relative; min-height: 104px; overflow: hidden; padding: 9px; border: 1px solid color-mix(in srgb, var(--meter) 65%, var(--border-color)); background: var(--comfy-input-bg); }
        #${ROOT_ID} .dasiwa-monitor-full-metric > div:first-child, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric > div:first-child { display: flex; justify-content: space-between; }
        #${ROOT_ID} .dasiwa-monitor-full-metric strong, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric strong { font-size: 16px; }
        #${ROOT_ID} .dasiwa-monitor-graph-wrap, #dasiwa-monitor-full-overlay .dasiwa-monitor-graph-wrap { position: relative; z-index: 1; height: 46px; margin-top: 7px; border-bottom: 1px solid color-mix(in srgb, var(--meter) 25%, transparent); }
        #${ROOT_ID} .dasiwa-monitor-graph, #dasiwa-monitor-full-overlay .dasiwa-monitor-graph { width: 100%; height: 100%; overflow: visible; fill: none; stroke: var(--meter); stroke-width: 3; vector-effect: non-scaling-stroke; }
        #${ROOT_ID} .dasiwa-monitor-empty-graph, #dasiwa-monitor-full-overlay .dasiwa-monitor-empty-graph { color: var(--descrip-text, #aaa); font-size: 10px; padding-top: 18px; text-align: center; }
        #dasiwa-monitor-dock-top { box-sizing: border-box; display: flex; align-self: stretch; min-width: 0; max-width: 100%; padding: 4px 0; pointer-events: none; } #dasiwa-monitor-dock-top[hidden] { display: none; } #dasiwa-monitor-dock-top > #${ROOT_ID} { pointer-events: auto; }
        #${ROOT_ID} { box-sizing: border-box; width: max-content; max-width: min(100%, calc(100vw - 16px)); min-height: 36px; align-items: flex-start; } #${ROOT_ID}.is-floating { max-width: calc(100vw - 16px); } #${ROOT_ID}.is-dock-left, #${ROOT_ID}.is-dock-right { max-height: calc(100vh - 128px); }
        #${ROOT_ID} .dasiwa-monitor-drag-handle { box-sizing: border-box; align-self: flex-start; height: 36px; min-height: 36px; display: flex; align-items: center; justify-content: center; flex: none; } #${ROOT_ID} .dasiwa-monitor-drag-handle::after { padding: 0; line-height: 1; }
        #${PANEL_ID}.is-lite { box-sizing: border-box; flex: 1 1 auto; width: max-content; max-width: 100%; min-width: 0; min-height: 36px; flex-wrap: wrap; align-content: flex-start; overflow: visible; } #${ROOT_ID} .dasiwa-monitor-settings { flex: none; } #${ROOT_ID} .dasiwa-monitor-resize-handle { position: absolute; z-index: 1004; right: 0; bottom: 0; width: 15px; height: 15px; cursor: ew-resize; touch-action: none; background: linear-gradient(135deg, transparent 55%, var(--border-color) 55% 62%, transparent 62% 74%, var(--border-color) 74% 81%, transparent 81%); }
        #${ROOT_ID}.is-resized { padding-right: 15px; }
        #${ROOT_ID}:not(.is-resized) { height: 36px; } #${ROOT_ID}:not(.is-resized) #${PANEL_ID}.is-lite { flex-wrap: nowrap; overflow-x: auto; overflow-y: hidden; }
        #${ROOT_ID}:not(.is-resized).is-vertical { height: auto; } #${ROOT_ID}:not(.is-resized).is-vertical #${PANEL_ID}.is-lite { overflow-x: hidden; overflow-y: auto; }
        #dasiwa-monitor-settings-menu .dasiwa-monitor-reset { padding: 5px; text-align: left; color: var(--input-text); border: 1px solid var(--border-color); background: var(--comfy-input-bg); cursor: pointer; }
        #dasiwa-monitor-settings-menu { box-sizing: border-box; max-width: calc(100vw - 16px); max-height: calc(100vh - 16px); overflow-y: auto; color: var(--input-text); } #dasiwa-monitor-settings-menu label.dasiwa-monitor-opacity-row { display: grid; grid-template-columns: 1fr; gap: 4px; } #dasiwa-monitor-settings-menu .dasiwa-monitor-opacity-row > span { display: flex; align-items: center; gap: 8px; } #dasiwa-monitor-settings-menu .dasiwa-monitor-opacity-row input { flex: 1; min-width: 0; } #dasiwa-monitor-settings-menu .dasiwa-monitor-opacity-row output { min-width: 4ch; text-align: right; }
        #${ROOT_ID} .dasiwa-monitor-drag-handle, #${ROOT_ID} .dasiwa-monitor-settings, #${ROOT_ID} .dasiwa-monitor-metric, #${ROOT_ID} .dasiwa-monitor-full-metric, #${PANEL_ID}.is-full, #dasiwa-monitor-full-overlay, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric { background: color-mix(in srgb, var(--comfy-input-bg) var(--dasiwa-monitor-bg-opacity, 100%), transparent); }
        #${PANEL_ID}.is-full, #dasiwa-monitor-full-overlay { background: color-mix(in srgb, var(--comfy-menu-bg, #202020) var(--dasiwa-monitor-bg-opacity, 100%), transparent); }
        #${ROOT_ID} .dasiwa-monitor-drag-handle, #${ROOT_ID} .dasiwa-monitor-settings, #${ROOT_ID} .dasiwa-monitor-metric, #${ROOT_ID} .dasiwa-monitor-full-metric, #${PANEL_ID}.is-full, #dasiwa-monitor-full-overlay, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric { border-color: color-mix(in srgb, var(--border-color) var(--dasiwa-monitor-content-alpha, 100%), transparent); }
        #${ROOT_ID} .dasiwa-monitor-settings { color: color-mix(in srgb, var(--input-text) var(--dasiwa-monitor-content-alpha, 100%), transparent); }
        #${ROOT_ID} .dasiwa-monitor-drag-handle::after, #${ROOT_ID} .dasiwa-monitor-metric > span, #${ROOT_ID} .dasiwa-monitor-metric > strong, #${ROOT_ID} .dasiwa-monitor-full-header, #${ROOT_ID} .dasiwa-monitor-full-metric > div, #${ROOT_ID} .dasiwa-monitor-full-metric > small, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-header, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric > div, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric > small { opacity: var(--dasiwa-monitor-content-opacity, 1); }
        #${ROOT_ID} .dasiwa-monitor-metric::before, #${ROOT_ID} .dasiwa-monitor-full-metric > i, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-metric > i { opacity: calc(.38 * var(--dasiwa-monitor-content-opacity, 1)); }
        #${ROOT_ID}.is-ultra { height: auto; max-height: none; gap: 4px; padding: 5px; border: 1px solid color-mix(in srgb, var(--border-color) var(--dasiwa-monitor-content-alpha, 100%), transparent); background: color-mix(in srgb, var(--comfy-input-bg) var(--dasiwa-monitor-bg-opacity, 100%), transparent); }
        #${ROOT_ID}.is-ultra .dasiwa-monitor-resize-handle { display: none; }
        #${ROOT_ID}.is-ultra .dasiwa-monitor-settings { width: 26px; height: 26px; }
        #${ROOT_ID}.is-ultra #${PANEL_ID}.is-ultra { position: static; display: flex; flex: 1 1 auto; flex-direction: column; gap: 0; width: auto; min-width: 0; height: auto; overflow: visible; padding: 0; border: 0; background: transparent; box-shadow: none; }
        #${ROOT_ID} .dasiwa-monitor-ultra-row { position: relative; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; flex: 0 0 29px; gap: 4px; min-width: 0; padding: 0 4px; border-bottom: 1px solid color-mix(in srgb, var(--border-color) 60%, transparent); white-space: nowrap; }
        #${ROOT_ID} .dasiwa-monitor-ultra-row::after { content: ""; position: absolute; left: 0; bottom: 0; width: var(--fill); height: 2px; background: var(--meter); opacity: var(--dasiwa-monitor-content-opacity, 1); }
        #${ROOT_ID} .dasiwa-monitor-ultra-row > span { overflow: hidden; text-overflow: ellipsis; font-size: 10px; color: var(--input-text); opacity: var(--dasiwa-monitor-content-opacity, 1); }
        #${ROOT_ID} .dasiwa-monitor-ultra-row > strong { color: var(--meter); font: 600 11px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; opacity: var(--dasiwa-monitor-content-opacity, 1); }
        @media (max-width: 640px) { #${PANEL_ID}.is-full, #dasiwa-monitor-full-overlay { width: calc(100vw - 12px); padding: 9px; } #${ROOT_ID} .dasiwa-monitor-full-grid, #dasiwa-monitor-full-overlay .dasiwa-monitor-full-grid { grid-template-columns: 1fr; } }
    `;
    document.head.appendChild(style);
}

async function mountMonitor() {
    if (monitorRoot) return;
    addStyles();
    const root = document.createElement("div");
    monitorRoot = root;
    root.id = ROOT_ID;
    root.innerHTML = `<span class="dasiwa-monitor-drag-handle" aria-label="Drag system monitor"></span><div id="${PANEL_ID}" class="dasiwa-monitor-display is-lite">Loading…</div><button class="dasiwa-monitor-settings" type="button" title="DaSiWa monitor settings" aria-label="DaSiWa monitor settings">⚙</button><span class="dasiwa-monitor-resize-handle" role="button" aria-label="Resize system monitor" title="Drag to resize monitor"></span>`;
    applySize(root);
    applyOpacity();
    const settingsButton = root.querySelector("button");
    settingsButton.addEventListener("click", () => openMenu(settingsButton));
    document.body.insertAdjacentHTML("beforeend", `
        <div id="dasiwa-monitor-dock-left" hidden></div>
        <div id="dasiwa-monitor-dock-right" hidden></div>
        <div id="dasiwa-monitor-dock-target-top" class="dasiwa-monitor-dock-target" hidden>Drop to dock to top</div>
        <div id="dasiwa-monitor-dock-target-left" class="dasiwa-monitor-dock-target" hidden>Dock left</div>
        <div id="dasiwa-monitor-dock-target-right" class="dasiwa-monitor-dock-target" hidden>Dock right</div>`);
    applyOrientation(root);
    enablePanelDrag(root);
    enablePanelResize(root);
    if (!placePanel(root)) document.body.appendChild(root);
    // ComfyUI can replace the action-bar parent when its side panels toggle.
    // Keep the same monitor element and its listeners; remount only its dock.
    monitorToolbarObserver = new MutationObserver(() => {
        if (settings.placement === "floating" || settings.dockSide !== "top") return;
        const row = document.querySelector('[data-testid="top-menu-actionbars"]');
        const dock = document.getElementById("dasiwa-monitor-dock-top");
        if (row?.parentElement && (!root.isConnected || dock?.previousElementSibling !== row || dock.parentElement !== row.parentElement)) {
            placePanel(root, "top");
        }
    });
    monitorToolbarObserver.observe(document.body, { childList: true, subtree: true });
    if (settings.placement === "floating") floatPanel(root, settings.x, settings.y);
    monitorEventListener = (event) => {
        latestSnapshot = event.detail;
        history = [...history, historyPoint(latestSnapshot)].slice(-HISTORY_LENGTH);
        render();
    };
    api.addEventListener(EVENT_NAME, monitorEventListener);
    monitorResizeObserver = new ResizeObserver(() => {
        applySize(root);
        if (settings.placement === "floating") {
            const x = Math.max(8, Math.min(settings.x, window.innerWidth - root.offsetWidth - 8));
            const y = Math.max(8, Math.min(settings.y, window.innerHeight - root.offsetHeight - 8));
            root.style.left = `${Math.round(x)}px`;
            root.style.top = `${Math.round(y)}px`;
        }
        const panel = document.getElementById(PANEL_ID);
        if (!panel) return;
        if (settings.mode === "full") {
            if (settings.dockSide === "top" && fullOverlay && monitorRoot) {
                const rect = monitorRoot.getBoundingClientRect();
                fullOverlay.style.top = `${Math.round(rect.bottom + 8)}px`;
                fullOverlay.style.left = `${Math.round(rect.left)}px`;
            } else {
                positionFullPanel(panel);
            }
        }
    });
    monitorResizeObserver.observe(document.documentElement);
    try {
        latestSnapshot = await api.fetchApi("/dasiwa/system-monitor").then((response) => response.json());
        history = [historyPoint(latestSnapshot)];
        render();
    } catch (error) {
        document.getElementById(PANEL_ID)?.replaceChildren("DaSiWa System Monitor is waiting for backend telemetry.");
        console.warn("DaSiWa System Monitor", error);
    }
}

app.registerExtension({
    name: "DaSiWa.SystemMonitor",
    init() {
        app.ui.settings.addSetting({
            id: MONITOR_ENABLED_SETTING,
            name: "Enable System Monitor",
            category: ["DaSiWa", "System Monitor", "Enable System Monitor"],
            tooltip: "Show the monitor and run its backend telemetry polling.",
            type: "boolean",
            defaultValue: true,
            onChange: (enabled) => { void setMonitorEnabled(enabled !== false); },
        });
        for (const [id, name, tooltip] of [
            [BACKGROUND_OPACITY_SETTING, "Background opacity", "Opacity of the monitor's background surfaces (0% is transparent)."],
            [CONTENT_OPACITY_SETTING, "Drawing / text / lines opacity", "Opacity of the monitor's meters, labels, graph lines and borders."],
        ]) {
            app.ui.settings.addSetting({
                id, name, category: ["DaSiWa", "System Monitor", name], tooltip,
                type: "slider", defaultValue: 100, attrs: OPACITY_ATTRS,
                onChange: applyOpacity,
            });
        }
    },
    async setup() {
        const enabled = isMonitorEnabled();
        await syncMonitorEnabled(enabled);
        if (enabled) await mountMonitor();
    },
});