import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "DaSiWa_EnhancedVideoCombine";

function videoUrl(video) {
    const params = new URLSearchParams({
        filename: video.filename,
        subfolder: video.subfolder || "",
        type: video.type || "output",
    });
    return api.apiURL(`/view?${params}`);
}

function stopNodeInteraction(element) {
    for (const eventName of ["pointerdown", "mousedown", "touchstart"]) {
        element.addEventListener(eventName, (event) => event.stopPropagation());
    }
}

function syncBooleanWidget(node, name, checked) {
    const widget = node.widgets?.find((candidate) => candidate.name === name);
    if (!widget) return;
    widget.value = checked;
    widget.callback?.(checked);
}

function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00";
    const wholeSeconds = Math.max(0, Math.floor(seconds));
    return `${Math.floor(wholeSeconds / 60)}:${String(wholeSeconds % 60).padStart(2, "0")}`;
}

function fitPreviewHeight(node) {
    requestAnimationFrame(() => {
        const requiredSize = node.computeSize();
        node.setSize([node.size[0], Math.max(node.size[1], requiredSize[1])]);
        node.graph?.setDirtyCanvas(true);
    });
}

function showHelpDialog() {
    const dialog = document.createElement("dialog");
    dialog.style.cssText = "max-width:620px;color:#ddd;background:#202225;border:1px solid #555;border-radius:8px;padding:18px;font:13px sans-serif;line-height:1.45";
    dialog.innerHTML = `
        <form method="dialog" style="float:right"><button title="Close">×</button></form>
        <h3 style="margin:0 28px 10px 0">Enhanced Video Combine Help</h3>
        <p>Encodes a connected IMAGE batch into a video. It returns the frames optionally and shows an in-node preview after execution.</p>
        <dl style="margin:0;display:grid;grid-template-columns:max-content 1fr;gap:6px 12px">
            <dt><b>codec</b></dt><dd><b>Auto</b> tries AV1, H.265, VP9, then H.264 and keeps the first usable encoder. The other choices force that codec with hardware-to-software fallback.</dd>
            <dt><b>container</b></dt><dd>Auto selects compatible containers: WebM, MKV, then MP4 for AV1/VP9; MP4 then MKV for H.264/H.265.</dd>
            <dt><b>bit depth / quality</b></dt><dd>Auto bit depth detects 8- or 10-bit frame precision. Lower quality values retain more detail and create larger files.</dd>
            <dt><b>audio</b></dt><dd>Connect AUDIO to mux it. Auto audio uses Opus for WebM and AAC for MKV/MP4. Crop to audio ends video at the audio duration.</dd>
            <dt><b>log level</b></dt><dd>Standard logs compact selection and output details to the ComfyUI console. Verbose also logs missing and failed encoder attempts.</dd>
            <dt><b>preview</b></dt><dd>H.265 outputs receive an H.264 preview sidecar so browsers without HEVC playback can show the canvas preview.</dd>
            <dt><b>other</b></dt><dd>Ping-pong reverses interior frames for a loop. Save metadata embeds the ComfyUI workflow. Pass frames keeps frames available downstream.</dd>
        </dl>`;
    document.body.append(dialog);
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    dialog.showModal();
}

function isHelpIconHit(node, pos) {
    const titleHeight = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
    return pos[0] >= node.size[0] - 24 && pos[0] <= node.size[0] - 4 && pos[1] >= -titleHeight && pos[1] <= 0;
}

app.registerExtension({
    name: "DaSiWa.EnhancedVideoCombinePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        const originalOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
            const previewNode = this;
            const originalOnDrawForeground = this.onDrawForeground;
            const originalOnMouseDown = this.onMouseDown;
            this.onDrawForeground = function (ctx) {
                originalOnDrawForeground?.apply(this, arguments);
                const titleHeight = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
                const x = this.size[0] - 14;
                const y = -titleHeight / 2;
                ctx.save();
                ctx.fillStyle = "#d9e8ff";
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = "#1e293b";
                ctx.font = "bold 12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("?", x, y + 0.5);
                ctx.restore();
            };
            this.onMouseDown = function (event, pos) {
                if (isHelpIconHit(this, pos)) {
                    showHelpDialog();
                    return true;
                }
                return originalOnMouseDown?.apply(this, arguments);
            };
            const root = document.createElement("div");
            root.className = "dasiwa-enhanced-video-preview";
            root.style.width = "100%";
            let previewWidget;

            const preview = document.createElement("video");
            preview.loop = true;
            preview.muted = true;
            preview.playsInline = true;
            preview.style.cssText = "display:block;width:100%;background:#111;cursor:pointer";

            const info = document.createElement("div");
            info.style.cssText = "display:flex;gap:8px;margin-top:4px;color:var(--input-text,#ddd);font:11px sans-serif";
            const resolution = document.createElement("span");
            const duration = document.createElement("span");
            const fps = document.createElement("span");
            info.append(resolution, duration, fps);

            const controls = document.createElement("div");
            controls.style.cssText = "display:flex;align-items:center;gap:5px;margin-top:4px";
            const play = document.createElement("button");
            play.type = "button";
            play.textContent = "▶";
            const seek = document.createElement("input");
            seek.type = "range";
            seek.min = "0";
            seek.max = "1000";
            seek.value = "0";
            seek.style.cssText = "flex:1;min-width:0";
            const time = document.createElement("span");
            time.textContent = "0:00 / 0:00";
            time.style.cssText = "font:11px monospace;white-space:nowrap";
            const mute = document.createElement("button");
            mute.type = "button";
            mute.textContent = "🔇";
            controls.append(play, seek, time, mute);

            const actions = document.createElement("div");
            actions.style.cssText = "display:flex;gap:5px;margin-top:4px";
            const saveFirstFrame = document.createElement("input");
            saveFirstFrame.type = "checkbox";
            saveFirstFrame.checked = Boolean(this.widgets?.find((widget) => widget.name === "save_first_frame")?.value);
            saveFirstFrame.addEventListener("change", () => syncBooleanWidget(this, "save_first_frame", saveFirstFrame.checked));
            const saveFirstFrameLabel = document.createElement("label");
            saveFirstFrameLabel.append(saveFirstFrame, " Save first frame");
            const saveLastFrame = document.createElement("input");
            saveLastFrame.type = "checkbox";
            saveLastFrame.checked = Boolean(this.widgets?.find((widget) => widget.name === "save_last_frame")?.value);
            saveLastFrame.addEventListener("change", () => syncBooleanWidget(this, "save_last_frame", saveLastFrame.checked));
            const saveLastFrameLabel = document.createElement("label");
            saveLastFrameLabel.append(saveLastFrame, " Save last frame");
            const autoPlay = document.createElement("input");
            autoPlay.type = "checkbox";
            autoPlay.checked = true;
            const autoPlayLabel = document.createElement("label");
            autoPlayLabel.style.marginLeft = "auto";
            autoPlayLabel.append(autoPlay, " Autoplay");
            actions.append(saveFirstFrameLabel, saveLastFrameLabel, autoPlayLabel);

            preview.addEventListener("loadedmetadata", () => {
                previewWidget.aspectRatio = preview.videoWidth / preview.videoHeight;
                resolution.textContent = `${preview.videoWidth}×${preview.videoHeight}`;
                duration.textContent = formatTime(preview.duration);
                fps.textContent = preview.dataset.fps ? `${preview.dataset.fps} fps` : "";
                fitPreviewHeight(previewNode);
                if (autoPlay.checked) preview.play().catch(() => {});
            });
            preview.addEventListener("error", () => {
                resolution.textContent = "H.265/HEVC playback is not supported by this browser";
                duration.textContent = "";
                fps.textContent = "";
            });
            const togglePlayback = () => preview.paused ? preview.play() : preview.pause();
            play.addEventListener("click", togglePlayback);
            preview.addEventListener("click", togglePlayback);
            preview.addEventListener("play", () => { play.textContent = "❚❚"; });
            preview.addEventListener("pause", () => { play.textContent = "▶"; });
            preview.addEventListener("timeupdate", () => {
                seek.value = String(preview.duration ? Math.round(preview.currentTime / preview.duration * 1000) : 0);
                time.textContent = `${formatTime(preview.currentTime)} / ${formatTime(preview.duration)}`;
            });
            seek.addEventListener("input", () => {
                if (Number.isFinite(preview.duration)) preview.currentTime = preview.duration * Number(seek.value) / 1000;
            });
            mute.addEventListener("click", () => {
                preview.muted = !preview.muted;
                mute.textContent = preview.muted ? "🔇" : "🔊";
            });
            [preview, controls, play, seek, mute, actions, saveFirstFrame, saveFirstFrameLabel, saveLastFrame, saveLastFrameLabel, autoPlay, autoPlayLabel].forEach(stopNodeInteraction);

            root.append(preview, info, controls, actions);
            const previewHeight = () => (previewNode.size[0] - 20) / (previewWidget?.aspectRatio ?? 16 / 9) + 82;
            previewWidget = this.addDOMWidget("video_preview", "preview", root, {
                serialize: false,
                hideOnZoom: false,
                getHeight: () => previewHeight(),
            });
            previewWidget.aspectRatio = 16 / 9;
            previewWidget.computeSize = function (width) {
                if (root.hidden) return [width, -4];
                return [width, previewHeight()];
            };
            requestAnimationFrame(() => fitPreviewHeight(previewNode));
            this.dasiwaVideoPreview = preview;
            return result;
        };

        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted ? originalOnExecuted.apply(this, arguments) : undefined;
            const video = message?.gifs?.[0] ?? message?.videos?.[0];
            if (!video?.filename || !this.dasiwaVideoPreview) return result;

            const preview = this.dasiwaVideoPreview;
            preview.pause();
            preview.dataset.filename = video.filename;
            preview.dataset.fps = video.fps ?? "";
            preview.src = videoUrl(video);
            preview.load();
            return result;
        };
    },
});
