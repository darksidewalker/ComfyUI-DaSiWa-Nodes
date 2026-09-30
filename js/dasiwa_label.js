import { app } from "../../scripts/app.js";

const TYPE = "Lable (DaSiWa)";
const FONTS = ["Arial", "Verdana", "Georgia", "Times New Roman", "Courier New", "sans-serif", "serif", "monospace"];
const MODES = ["background", "float left", "float right", "above text", "below text"];
const PALETTE = [
    "#ffffff", "#f5f5f5", "#d3d3d3", "#c0c0c0", "#808080", "#64748b", "#1e293b", "#000000",
    "#ff0000", "#ef4444", "#dc143c", "#800000", "#ff7f50", "#fa8072", "#ffa500", "#f97316",
    "#ffff00", "#facc15", "#ffd700", "#808000", "#00ff00", "#22c55e", "#008000", "#2e8b57",
    "#00ffff", "#06b6d4", "#40e0d0", "#008080", "#87ceeb", "#3b82f6", "#0000ff", "#000080",
    "#a855f7", "#800080", "#4b0082", "#ee82ee", "#ff00ff", "#ec4899", "#ff69b4", "#ffc0cb",
    "#ffe4e1", "#e6e6fa", "#f5deb3", "#f5f5dc", "#d2691e", "#a52a2a", "#8b4513", "#708090",
];
const DEFAULTS = {
    fontSize: 24, fontFamily: "Arial", fontColor: "#ffffff", textAlign: "left",
    backgroundColor: "#1e293b", backgroundOpacity: 0, fontOpacity: 1,
    padding: 12, borderRadius: 0, angle: 0, image: "", imageMode: "background",
    imageSize: 40, imageFit: "contain", imageOpacity: 1,
};
const IMAGE_PATTERN = /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

function normalize(properties) {
    const p = { ...DEFAULTS };
    for (const [key, value] of Object.entries(properties ?? {})) {
        if (!(key in p)) continue;
        if (typeof p[key] === "number" && Number.isFinite(value)) p[key] = value;
        else if (typeof p[key] === "string" && typeof value === "string") p[key] = value;
    }
    for (const key of ["fontColor", "backgroundColor"]) {
        if (!/^#[0-9a-f]{6}$/i.test(p[key])) p[key] = DEFAULTS[key];
    }
    if (!FONTS.includes(p.fontFamily)) p.fontFamily = DEFAULTS.fontFamily;
    if (!["left", "center", "right"].includes(p.textAlign)) p.textAlign = "left";
    if (!MODES.includes(p.imageMode)) p.imageMode = "background";
    if (!["contain", "cover"].includes(p.imageFit)) p.imageFit = "contain";
    for (const key of ["backgroundOpacity", "fontOpacity", "imageOpacity"]) p[key] = Math.max(0, Math.min(1, p[key]));
    p.fontSize = Math.max(1, Math.min(256, p.fontSize));
    p.padding = Math.max(0, Math.min(128, p.padding));
    p.borderRadius = Math.max(0, Math.min(256, p.borderRadius));
    p.angle = Math.max(-180, Math.min(180, p.angle));
    p.imageSize = Math.max(10, Math.min(90, p.imageSize));
    if (p.image.length > MAX_IMAGE_BYTES * 4 / 3 + 100 || (p.image && !IMAGE_PATTERN.test(p.image))) p.image = "";
    return p;
}

function installStyle() {
    if (document.getElementById("dasiwa-label-style")) return;
    const style = document.createElement("style");
    style.id = "dasiwa-label-style";
    // These scoped host selectors match frontend 1.53.6. No global renderer patches.
    style.textContent = `
        .lg-node:has(.dasiwa-label) { filter:none!important; }
        .lg-node:has(.dasiwa-label) [data-testid="node-inner-wrapper"],
        .lg-node:has(.dasiwa-label) [data-testid^="node-body-"] { background:transparent!important; }
        .lg-node:has(.dasiwa-label) [data-testid^="node-body-"] { padding:0!important; }
        .lg-node:has(.dasiwa-label) .lg-node-widgets { padding:0; display:flex; flex:1; }
        .lg-node:has(.dasiwa-label) .lg-node-widget { display:flex; flex:1; }
        .lg-node:has(.dasiwa-label) .lg-node-widget > div:first-child { display:none; }
        .lg-node:has(.dasiwa-label) .lg-node-widget > div:last-child { flex:1; min-width:0; }
        .dasiwa-label { width:100%; height:100%; min-height:40px; position:relative; overflow:hidden; }
        .dasiwa-label, .dasiwa-label * { cursor:crosshair!important; }
        .dasiwa-label-layer { position:absolute; inset:0; box-sizing:border-box; overflow:hidden; }
        .dasiwa-label-text { white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.2; }
        .dasiwa-label-image { display:none; }
        .dasiwa-label-editor { color:#eee; background:#20232a; border:1px solid #64748b; border-radius:10px; padding:16px; width:680px; max-width:90vw; max-height:calc(100vh - 32px); box-sizing:border-box; overflow:auto; }
        .dasiwa-label-editor::backdrop { background:#0006; }
        .dasiwa-label-editor h2 { margin:0 0 16px; font-size:18px; }
        .dasiwa-label-editor label { display:grid; grid-template-columns:130px minmax(0,1fr); align-items:center; gap:8px; margin:8px 0; }
        .dasiwa-label-editor textarea { width:100%; min-height:90px; box-sizing:border-box; }
        .dasiwa-label-editor button, .dasiwa-label-editor select, .dasiwa-label-editor textarea { color:#eee; background:#303641; border:1px solid #64748b; border-radius:4px; padding:5px; }
        @supports (appearance:base-select) {
            .dasiwa-label-editor select[data-setting="fontFamily"],
            .dasiwa-label-editor select[data-setting="fontFamily"]::picker(select) { appearance:base-select; }
            .dasiwa-label-editor select[data-setting="fontFamily"]::picker(select) { color:#eee; background:#303641; border:1px solid #64748b; border-radius:4px; }
            .dasiwa-label-editor select[data-setting="fontFamily"] option { padding:5px 10px; font-size:16px; }
        }
        .dasiwa-label-editor input[type=range] { width:100%; }
        .dasiwa-label-editor .color-controls { display:flex; gap:5px; flex-wrap:wrap; }
        .dasiwa-label-editor .hex-color { width:86px; box-sizing:border-box; flex-shrink:0; font-family:monospace; color:#eee; background:#303641; border:1px solid #64748b; border-radius:4px; padding:4px; }
        .dasiwa-label-editor .eyedropper { display:inline-flex; align-items:center; justify-content:center; }
        .dasiwa-label-editor .image-picker { display:flex; align-items:center; gap:8px; min-width:0; }
        .dasiwa-label-editor .image-picker button { flex-shrink:0; }
        .dasiwa-label-editor .image-picker span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .dasiwa-label-editor .swatch { width:20px; height:20px; padding:0; border:1px solid #aaa; }
        .dasiwa-label-editor .slider { display:flex; align-items:center; gap:8px; }
        .dasiwa-label-editor .slider input[type=range] { flex:1; min-width:0; }
        .dasiwa-label-editor input[type=number] { width:64px; flex-shrink:0; box-sizing:border-box; text-align:right; color:#eee; background:#303641; border:1px solid #64748b; border-radius:4px; padding:4px; }
        .dasiwa-label-editor footer { display:flex; gap:8px; margin-top:16px; }
        .dasiwa-label-editor .status { font-size:12px; color:#facc15; min-height:16px; }
    `;
    document.head.append(style);
}

class SimpleLabel extends LGraphNode {
    static title = TYPE;
    static title_mode = LiteGraph.NO_TITLE;
    static collapsable = false;

    constructor(title = "Label") {
        super(title);
        this.isVirtualNode = true;
        this.properties = { ...DEFAULTS };
        this.color = "transparent";
        this.bgcolor = "transparent";
        this.size = [320, 180];
        this.root = document.createElement("div");
        this.root.className = "dasiwa-label";
        this.root.title = "Double-click to edit label. Right-click for label settings.";
        this.layer = document.createElement("div");
        this.layer.className = "dasiwa-label-layer";
        this.picture = document.createElement("img");
        this.picture.className = "dasiwa-label-image";
        this.picture.alt = "";
        this.text = document.createElement("div");
        this.text.className = "dasiwa-label-text";
        this.layer.append(this.picture, this.text);
        this.root.append(this.layer);
        this.root.addEventListener("dblclick", (event) => {
            event.stopPropagation();
            this.edit();
        });
        for (const type of ["pointerdown", "pointermove", "pointerup"]) {
            this.root.addEventListener(type, event => {
                if (event.button === 2) return;
                const host = this.root.closest(".lg-node, .dom-widget") ?? this.root;
                const target = this.flags.pinned
                    ? document.elementsFromPoint(event.clientX, event.clientY)
                        .find(element => element !== host && !host.contains(element))
                    : host.matches(".lg-node") ? host : app.canvas.canvas;
                if (!target) return;
                event.stopPropagation();
                event.preventDefault();
                target.dispatchEvent(new PointerEvent(type, {
                    bubbles: true, cancelable: true, pointerId: event.pointerId,
                    pointerType: event.pointerType, isPrimary: event.isPrimary,
                    clientX: event.clientX, clientY: event.clientY,
                    button: event.button, buttons: event.buttons,
                    ctrlKey: event.ctrlKey, shiftKey: event.shiftKey,
                    altKey: event.altKey, metaKey: event.metaKey,
                }));
            });
        }
        this.root.addEventListener("contextmenu", event => {
            if (!this.flags.pinned && this.root.closest(".lg-node")) return;
            event.preventDefault();
            event.stopPropagation();
            if (this.flags.pinned) this.edit();
            else app.canvas.processContextMenu(this, event);
        });
        this.addDOMWidget("label", "dasiwa-label", this.root, {
            serialize: false, hideOnZoom: false, margin: 0,
            getMinHeight: () => 40,
        });
        this.render();
    }

    render() {
        this.color = "transparent";
        this.bgcolor = "transparent";
        const p = normalize(this.properties);
        Object.assign(this.layer.style, {
            padding: `${p.padding}px`, borderRadius: `${p.borderRadius}px`,
            backgroundColor: `rgba(${parseInt(p.backgroundColor.slice(1, 3), 16)},${parseInt(p.backgroundColor.slice(3, 5), 16)},${parseInt(p.backgroundColor.slice(5, 7), 16)},${p.backgroundOpacity})`,
            transform: `rotate(${p.angle}deg)`,
        });
        Object.assign(this.text.style, {
            fontFamily: p.fontFamily, fontSize: `${p.fontSize}px`, color: p.fontColor,
            opacity: p.fontOpacity, textAlign: p.textAlign, position: "relative",
        });
        this.text.textContent = String(this.title ?? "").replace(/\\n/g, "\n");
        this.picture.removeAttribute("style");
        this.picture.hidden = !p.image;
        if (p.image) {
            if (this.picture.getAttribute("src") !== p.image) this.picture.src = p.image;
            const background = p.imageMode === "background";
            Object.assign(this.picture.style, {
                display: "block", position: background ? "absolute" : "relative",
                width: background ? "100%" : `${p.imageSize}%`,
                height: background ? "100%" : "auto", maxHeight: background ? "100%" : `calc(${this.size[1]}px - ${p.padding * 2}px)`,
                objectFit: p.imageFit, opacity: p.imageOpacity,
                float: p.imageMode.startsWith("float") ? p.imageMode.split(" ")[1] : "none",
                margin: background ? "0" : p.imageMode.startsWith("float") ? "0 8px 8px" : "0 auto 8px",
                ...(background ? { inset: "0" } : {}),
            });
            if (p.imageMode === "below text") this.layer.append(this.picture);
            else this.layer.prepend(this.picture);
        } else this.picture.removeAttribute("src");
        this.setDirtyCanvas(true, true);
    }

    setColorOption(option) {
        if (option) {
            this.properties.backgroundColor = option.bgcolor.replace(/^#([0-9a-f])([0-9a-f])([0-9a-f])$/i, "#$1$1$2$2$3$3");
        } else this.properties.backgroundOpacity = 0;
        this.render();
    }
    computeSize() { return [120, 60]; }
    getWidgetOnPos() { return null; }
    isPointInside(x, y, ...args) {
        return !this.flags.pinned && super.isPointInside(x, y, ...args);
    }
    onConfigure() { this.render(); }
    onPropertyChanged() { this.render(); }
    onResize() { this.render(); }
    onDblClick() { this.edit(); }
    onRemoved() { this.dialog?.close(); }
    getExtraMenuOptions() {
        return [{ content: "Edit label…", callback: () => this.edit() }];
    }

    edit() {
        if (this.dialog?.open) { this.dialog.focus(); return; }
        this.properties = normalize(this.properties);
        const dialog = document.createElement("dialog");
        this.dialog = dialog;
        dialog.className = "dasiwa-label-editor";
        dialog.lang = "en";
        const heading = document.createElement("h2");
        heading.textContent = TYPE;
        dialog.append(heading);
        const status = document.createElement("p");
        status.className = "status";
        status.setAttribute("role", "status");
        const changed = () => { this.render(); this.graph?.change(); };
        const row = (name, control) => {
            const label = document.createElement("label");
            const span = document.createElement("span");
            span.textContent = name;
            label.append(span, control);
            dialog.append(label);
        };
        const button = (name, action) => {
            const control = document.createElement("button");
            control.type = "button";
            control.textContent = name;
            control.onclick = action;
            return control;
        };
        const select = (key, name, values) => {
            const control = document.createElement("select");
            control.dataset.setting = key;
            for (const value of values) {
                const option = new Option(value, value);
                if (key === "fontFamily") option.style.fontFamily = value;
                control.add(option);
            }
            control.value = this.properties[key];
            if (key === "fontFamily") control.style.fontFamily = control.value;
            control.onchange = () => {
                this.properties[key] = control.value;
                if (key === "fontFamily") control.style.fontFamily = control.value;
                changed();
            };
            row(name, control);
        };
        const slider = (key, name, min, max, step = 1, get = () => this.properties[key], set = v => this.properties[key] = v) => {
            const wrapper = document.createElement("div");
            wrapper.className = "slider";
            const control = document.createElement("input");
            control.type = "range";
            control.dataset.setting = key;
            Object.assign(control, { min, max, step, value: get() });
            control.setAttribute("aria-label", name);
            const number = document.createElement("input");
            number.type = "number";
            Object.assign(number, { min, max, step, value: get() });
            number.setAttribute("aria-label", `${name} value`);
            control.oninput = () => { set(Number(control.value)); number.value = get(); changed(); };
            number.onchange = () => {
                if (Number.isFinite(number.valueAsNumber)) {
                    control.value = Math.max(min, Math.min(max, number.valueAsNumber));
                    set(Number(control.value));
                }
                control.value = get();
                number.value = get();
                changed();
            };
            number.addEventListener("keydown", event => { if (event.key === "Enter") number.blur(); });
            wrapper.append(control, number);
            row(name, wrapper);
        };
        const color = (key, name) => {
            const wrapper = document.createElement("div");
            wrapper.className = "color-controls";
            const control = document.createElement("input");
            control.type = "color";
            control.dataset.setting = key;
            control.value = this.properties[key];
            control.setAttribute("aria-label", name);
            const hex = document.createElement("input");
            hex.type = "text";
            hex.className = "hex-color";
            hex.maxLength = 7;
            hex.spellcheck = false;
            hex.value = control.value;
            hex.setAttribute("aria-label", `${name} HEX`);
            hex.title = "HEX color: #RRGGBB or #RGB";
            const apply = value => { control.value = value; hex.value = control.value; this.properties[key] = control.value; changed(); };
            hex.onchange = () => {
                const match = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.value.trim());
                if (!match) {
                    hex.value = this.properties[key];
                    status.textContent = "Enter a HEX color such as #ff8800 or #f80.";
                    return;
                }
                status.textContent = "";
                const value = match[1].toLowerCase();
                apply("#" + (value.length === 3 ? value.replace(/./g, digit => digit + digit) : value));
            };
            hex.addEventListener("keydown", event => { if (event.key === "Enter") hex.blur(); });
            control.oninput = () => apply(control.value);
            wrapper.append(control, hex);
            const pick = button("", async () => {
                try { apply((await new EyeDropper().open()).sRGBHex); }
                catch (error) { if (error.name !== "AbortError") status.textContent = "Screen color picker unavailable. Try Chromium on localhost or HTTPS."; }
            });
            pick.className = "eyedropper";
            pick.setAttribute("aria-label", `Pick ${name.toLowerCase()} from screen`);
            pick.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 3 5 5-3 3-2-2-8 8-4 1 1-4 8-8-2-2z"/><path d="m12 5 7 7"/></svg>';
            pick.disabled = !window.EyeDropper || !window.isSecureContext;
            pick.title = pick.disabled ? "Screen color picker requires Chromium on localhost or HTTPS." : "Pick a pixel anywhere on screen";
            wrapper.append(pick);
            for (const value of PALETTE) {
                const swatch = button("", () => apply(value));
                swatch.className = "swatch";
                swatch.style.backgroundColor = value;
                swatch.setAttribute("aria-label", `${name} ${value}`);
                wrapper.append(swatch);
            }
            row(name, wrapper);
        };
        const text = document.createElement("textarea");
        text.value = this.title;
        text.setAttribute("aria-label", "Label text");
        text.oninput = () => { this.title = text.value; changed(); };
        row("Text", text);
        select("fontFamily", "Font", FONTS);
        slider("fontSize", "Font size", 1, 256);
        select("textAlign", "Alignment", ["left", "center", "right"]);
        color("fontColor", "Text color");
        slider("fontOpacity", "Text opacity", 0, 1, 0.01);
        color("backgroundColor", "Background");
        slider("backgroundOpacity", "Background opacity", 0, 1, 0.01);
        slider("padding", "Padding", 0, 128);
        slider("borderRadius", "Corner radius", 0, 256);
        slider("angle", "Rotation", -180, 180);
        slider("width", "Width", 120, 1600, 1, () => this.size[0], v => this.setSize([v, this.size[1]]));
        slider("height", "Height", 60, 1200, 1, () => this.size[1], v => this.setSize([this.size[0], v]));
        const upload = document.createElement("input");
        upload.type = "file";
        upload.hidden = true;
        upload.accept = "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp";
        const imagePicker = document.createElement("div");
        imagePicker.className = "image-picker";
        const imageName = document.createElement("span");
        imageName.textContent = this.properties.image ? "Embedded image" : "No image selected";
        const chooseImage = button("Choose image…", () => upload.click());
        chooseImage.setAttribute("aria-label", "Choose image…");
        imagePicker.append(chooseImage, imageName, upload);
        upload.onchange = async () => {
            const file = upload.files?.[0];
            if (!file) return;
            try {
                if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw new Error("Choose PNG, JPEG or WebP.");
                if (file.size > MAX_IMAGE_BYTES) throw new Error("Image limit is 10 MiB. Resize image first.");
                const image = await createImageBitmap(file);
                image.close();
                const reader = new FileReader();
                const data = await new Promise((resolve, reject) => {
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = () => reject(new Error("Cannot read image."));
                    reader.readAsDataURL(file);
                });
                this.properties.image = data;
                imageName.textContent = file.name;
                status.textContent = "Image embedded in workflow.";
                changed();
            } catch (error) { status.textContent = error.message; }
        };
        row("Image", imagePicker);
        select("imageMode", "Image position", MODES);
        select("imageFit", "Image fit", ["contain", "cover"]);
        slider("imageSize", "Image width %", 10, 90);
        slider("imageOpacity", "Image opacity", 0, 1, 0.01);
        const pinned = document.createElement("input");
        pinned.type = "checkbox";
        pinned.checked = Boolean(this.flags.pinned);
        pinned.onchange = () => { this.pin(pinned.checked); changed(); };
        row("Pin label", pinned);
        dialog.append(status);
        const footer = document.createElement("footer");
        footer.append(button("Fit to text", () => {
            const p = this.properties;
            const context = document.createElement("canvas").getContext("2d");
            context.font = `${p.fontSize}px ${p.fontFamily}`;
            const lines = this.text.textContent.split("\n");
            this.setSize([
                Math.max(120, ...lines.map(line => context.measureText(line).width + p.padding * 2)),
                Math.max(60, lines.length * p.fontSize * 1.2 + p.padding * 2),
            ]);
            changed();
            for (const [key, value] of [["width", this.size[0]], ["height", this.size[1]]]) {
                const control = dialog.querySelector(`[data-setting="${key}"]`);
                control.value = value;
                control.nextElementSibling.value = Math.round(value);
            }
        }), button("Remove image", () => { this.properties.image = ""; upload.value = ""; imageName.textContent = "No image selected"; changed(); }), button("Done", () => dialog.close()));
        dialog.append(footer);
        dialog.addEventListener("close", () => { dialog.remove(); if (this.dialog === dialog) this.dialog = null; }, { once: true });
        document.body.append(dialog);
        dialog.showModal();
        text.focus();
    }
}

app.registerExtension({
    name: "DaSiWa.Lable",
    registerCustomNodes() {
        if (LiteGraph.registered_node_types[TYPE]) return;
        installStyle();
        LiteGraph.registerNodeType(TYPE, SimpleLabel);
        SimpleLabel.category = "DaSiWa/utilities";
    },
});
