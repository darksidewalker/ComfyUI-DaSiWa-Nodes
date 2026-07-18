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

function saveFrame(video, lastFrame) {
    if (!video.videoWidth || !video.videoHeight || !Number.isFinite(video.duration)) return;
    const targetTime = lastFrame ? Math.max(0, video.duration - 0.001) : 0;
    const saveCurrentFrame = () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
            if (!blob) return;
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            const filename = (video.dataset.filename || "enhanced-video").replace(/\.[^/.]+$/, "");
            link.download = `${filename}-${lastFrame ? "last" : "first"}-frame.png`;
            link.click();
            URL.revokeObjectURL(link.href);
        }, "image/png");
    };
    if (Math.abs(video.currentTime - targetTime) < 0.0001) {
        saveCurrentFrame();
    } else {
        video.addEventListener("seeked", saveCurrentFrame, { once: true });
        video.currentTime = targetTime;
    }
}

function stopNodeInteraction(element) {
    for (const eventName of ["pointerdown", "mousedown", "touchstart"]) {
        element.addEventListener(eventName, (event) => event.stopPropagation());
    }
}

function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00";
    const wholeSeconds = Math.max(0, Math.floor(seconds));
    return `${Math.floor(wholeSeconds / 60)}:${String(wholeSeconds % 60).padStart(2, "0")}`;
}

function fitPreviewHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node.graph?.setDirtyCanvas(true);
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
            const saveFirstFrameLabel = document.createElement("label");
            saveFirstFrameLabel.append(saveFirstFrame, " Save first frame");
            const saveLastFrame = document.createElement("input");
            saveLastFrame.type = "checkbox";
            const saveLastFrameLabel = document.createElement("label");
            saveLastFrameLabel.append(saveLastFrame, " Save last frame");
            actions.append(saveFirstFrameLabel, saveLastFrameLabel);

            preview.addEventListener("loadedmetadata", () => {
                previewWidget.aspectRatio = preview.videoWidth / preview.videoHeight;
                resolution.textContent = `${preview.videoWidth}×${preview.videoHeight}`;
                duration.textContent = formatTime(preview.duration);
                fps.textContent = preview.dataset.fps ? `${preview.dataset.fps} fps` : "";
                fitPreviewHeight(previewNode);
                if (saveFirstFrame.checked) saveFrame(preview, false);
                if (saveLastFrame.checked) saveFrame(preview, true);
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
            [preview, controls, play, seek, mute, actions, saveFirstFrame, saveFirstFrameLabel, saveLastFrame, saveLastFrameLabel].forEach(stopNodeInteraction);

            root.append(preview, info, controls, actions);
            previewWidget = this.addDOMWidget("video_preview", "preview", root, {
                serialize: false,
                hideOnZoom: false,
            });
            previewWidget.computeSize = function (width) {
                if (!this.aspectRatio || root.hidden) return [width, -4];
                return [width, (previewNode.size[0] - 20) / this.aspectRatio + 82];
            };
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
