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

function transcodedVideoUrl(video) {
    const params = new URLSearchParams({
        filename: video.filename,
        subfolder: video.subfolder || "",
        type: video.type || "output",
    });
    return api.apiURL(`/dasiwa/enhanced-video-preview?${params}`);
}

const NATIVE_BROWSER_VIDEO = new Set(["H.264|MP4|8"]);

function shouldUseTranscodedPreview(video) {
    const key = `${video.codec}|${video.container}|${video.bit_depth ?? 8}`;
    return !NATIVE_BROWSER_VIDEO.has(key);
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

function hideLegacyLogLevelWidget(node) {
    const widget = node.widgets?.find((widget) => widget.name === "log_level");
    if (!widget) return;
    widget.draw = () => {};
    widget.computeSize = () => [0, -4];
}

function hideFrameExportWidgets(node) {
    for (const name of ["save_first_frame", "save_last_frame"]) {
        const widget = node.widgets?.find((widget) => widget.name === name);
        if (!widget) continue;
        widget.draw = () => {};
        widget.computeSize = () => [0, -4];
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

let videoCombineWidgetSchema = {};

function sanitizeWidgetValues(node) {
    for (const widget of node.widgets ?? []) {
        const spec = videoCombineWidgetSchema[widget.name];
        const def = Array.isArray(spec) ? spec[1] ?? {} : spec ?? {};
        if (widget.type === "COMBO") {
            const options = widget.options?.values ?? [];
            if (options.length && !options.includes(widget.value)) {
                widget.value = def.default ?? options[0];
                widget.callback?.(widget.value);
            }
        } else if (widget.type === "BOOLEAN") {
            if (typeof widget.value !== "boolean") {
                const d = def.default;
                widget.value = typeof d === "boolean" ? d : false;
                widget.callback?.(widget.value);
            }
        }
    }
}

function showHelpDialog() {
    const dialog = document.createElement("dialog");
    dialog.style.cssText = "max-width:620px;color:#ddd;background:#202225;border:1px solid #555;border-radius:8px;padding:18px;font:13px sans-serif;line-height:1.45";
    dialog.innerHTML = `
        <form method="dialog" style="float:right"><button title="Close">×</button></form>
        <h3 style="margin:0 28px 10px 0">Enhanced Video Combine Help</h3>
        <p>Encodes a connected IMAGE batch into a video. It returns the frames optionally and shows an in-node preview after execution.</p>
        <dl style="margin:0;display:grid;grid-template-columns:max-content 1fr;gap:6px 12px">
            <dt><b>codec</b></dt><dd><b>Auto</b> prefers browser-compatible 8-bit AV1/MP4 (faststart), then VP9 (WebM) and H.264 (MP4). H.265 remains an explicit choice with hardware-to-software fallback.</dd>
            <dt><b>container</b></dt><dd>Auto selects compatible containers: MP4 for AV1 and H.264, WebM for VP9. Explicit AV1 tries MP4, then WebM, then MKV; VP9 tries WebM, then MKV, then MP4; H.264/H.265 try MP4 then MKV.</dd>
            <dt><b>image animation</b></dt><dd>Animated WebP and Animated AVIF are manual image-animation outputs. They are excluded from Auto codec/container selection, ignore the codec choice, and omit connected audio.</dd>
            <dt><b>bit depth / quality</b></dt><dd>With an explicit codec, Auto bit depth detects 8- or 10-bit frame precision; codec Auto uses browser-compatible 8-bit output. Lower quality values retain more detail and create larger files.</dd>
            <dt><b>audio</b></dt><dd>Connect AUDIO to mux it. Choose Auto, AAC, Opus, or MP3 plus a target bitrate; Auto uses Opus for WebM and AAC for MKV/MP4. Crop to audio ends video at the audio duration. Preview sound is on only while the pointer is over the video. Check <b>Mute</b> to keep the preview permanently silent; the choice is saved with the node.</dd>
            <dt><b>preview</b></dt><dd>H.264/MP4 can use the native output. AV1, VP9, HEVC, 10-bit, and other outputs use a cached seekable H.264/AAC preview with one-second keyframes and HTTP range support. Autoplay and Mute are saved with the node. Use the seek bar below the video to click or drag to a position; Fullscreen or a video double-click opens fullscreen.</dd>
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
        videoCombineWidgetSchema = nodeData?.input?.required ?? {};

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
            sanitizeWidgetValues(this);
            hideLegacyLogLevelWidget(this);
            hideFrameExportWidgets(this);
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
            // Native controls hide themselves when the pointer leaves. Toggling the
            // controls property recreates Chromium's media layer inside a DOM widget.
            preview.controls = true;
            preview.controlsList = "nodownload noremoteplayback";
            preview.disablePictureInPicture = true;
            preview.style.cssText = "display:block;width:100%;background:#111;cursor:pointer";

            const info = document.createElement("div");
            info.style.cssText = "display:flex;gap:8px;margin-top:4px;color:var(--input-text,#ddd);font:11px sans-serif";
            const resolution = document.createElement("span");
            const duration = document.createElement("span");
            const fps = document.createElement("span");
            info.append(resolution, duration, fps);

            // A DOM range remains usable when the browser's shadow-DOM media
            // timeline is too small or hit-tested incorrectly in a scaled node.
            const transport = document.createElement("div");
            transport.style.cssText = "display:flex;align-items:center;gap:6px;margin-top:4px;font:11px sans-serif;color:var(--input-text,#ddd)";
            const timeline = document.createElement("input");
            timeline.type = "range";
            timeline.min = "0";
            timeline.max = "0";
            timeline.step = "0.01";
            timeline.value = "0";
            timeline.disabled = true;
            timeline.setAttribute("aria-label", "Seek video preview");
            timeline.title = "Seek video preview";
            timeline.style.cssText = "flex:1;min-width:0;cursor:pointer";
            const position = document.createElement("span");
            position.style.cssText = "white-space:nowrap;font-variant-numeric:tabular-nums";
            const updateTimeline = () => {
                const available = Number.isFinite(preview.duration) && preview.duration > 0 && preview.readyState >= 1;
                timeline.disabled = !available;
                timeline.max = String(available ? preview.duration : 0);
                timeline.value = String(available ? preview.currentTime : 0);
                position.textContent = `${formatTime(preview.currentTime)} / ${formatTime(preview.duration)}`;
            };
            timeline.addEventListener("input", () => {
                if (!timeline.disabled) preview.currentTime = Math.min(preview.duration, Math.max(0, Number(timeline.value)));
                updateTimeline();
            });
            for (const eventName of ["loadedmetadata", "durationchange", "timeupdate", "seeked", "emptied"]) {
                preview.addEventListener(eventName, updateTimeline);
            }
            updateTimeline();
            transport.append(timeline, position);

            const actions = document.createElement("div");
            actions.style.cssText = "display:flex;align-items:center;flex-wrap:wrap;gap:5px;margin-top:8px;padding:6px;box-sizing:border-box;background:rgba(0,0,0,.28);border:1px solid rgba(255,255,255,.08);border-radius:4px;font:11px sans-serif";
            const saveFirstFrame = document.createElement("input");
            saveFirstFrame.type = "checkbox";
            saveFirstFrame.checked = Boolean(this.widgets?.find((widget) => widget.name === "save_first_frame")?.value);
            saveFirstFrame.addEventListener("change", () => syncBooleanWidget(this, "save_first_frame", saveFirstFrame.checked));
            const saveFirstFrameLabel = document.createElement("label");
            saveFirstFrameLabel.style.cssText = "display:flex;align-items:center;gap:3px;white-space:nowrap;cursor:pointer";
            saveFirstFrameLabel.append(saveFirstFrame, " Save first frame");
            const saveLastFrame = document.createElement("input");
            saveLastFrame.type = "checkbox";
            saveLastFrame.checked = Boolean(this.widgets?.find((widget) => widget.name === "save_last_frame")?.value);
            saveLastFrame.addEventListener("change", () => syncBooleanWidget(this, "save_last_frame", saveLastFrame.checked));
            const saveLastFrameLabel = document.createElement("label");
            saveLastFrameLabel.style.cssText = "display:flex;align-items:center;gap:3px;white-space:nowrap;cursor:pointer";
            saveLastFrameLabel.append(saveLastFrame, " Save last frame");
            const autoPlay = document.createElement("input");
            autoPlay.type = "checkbox";
            this.properties ??= {};
            autoPlay.checked = this.properties.autoplay ?? true;
            autoPlay.addEventListener("change", () => {
                this.properties.autoplay = autoPlay.checked;
            });
            const autoPlayLabel = document.createElement("label");
            autoPlayLabel.style.cssText = "display:flex;align-items:center;gap:3px;margin-left:auto;white-space:nowrap;cursor:pointer";
            autoPlayLabel.append(autoPlay, " Autoplay");
            const mute = document.createElement("input");
            mute.type = "checkbox";
            mute.checked = this.properties.muted ?? false;
            mute.addEventListener("change", () => {
                this.properties.muted = mute.checked;
                if (mute.checked) preview.muted = true;
            });
            const muteLabel = document.createElement("label");
            muteLabel.style.cssText = "display:flex;align-items:center;gap:3px;white-space:nowrap;cursor:pointer";
            muteLabel.append(mute, " Mute");
            const download = document.createElement("a");
            download.textContent = "Download";
            download.href = "#";
            download.download = "video";
            download.style.cssText = "padding:2px 7px;color:inherit;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);border-radius:3px;text-decoration:none;cursor:pointer";
            download.addEventListener("click", (event) => {
                if (download.href.endsWith("#")) event.preventDefault();
            });
            const fullscreen = document.createElement("button");
            fullscreen.type = "button";
            fullscreen.textContent = "Fullscreen";
            fullscreen.title = "Open video preview in fullscreen";
            fullscreen.disabled = !(preview.requestFullscreen || preview.webkitEnterFullscreen);
            const enterFullscreen = () => {
                if (preview.requestFullscreen) {
                    preview.requestFullscreen().catch((error) => console.warn("[Enhanced Video Combine] Fullscreen unavailable", error));
                } else if (preview.webkitEnterFullscreen) {
                    preview.webkitEnterFullscreen();
                }
            };
            fullscreen.addEventListener("click", enterFullscreen);
            actions.append(saveFirstFrameLabel, saveLastFrameLabel, muteLabel, autoPlayLabel, fullscreen, download);

            preview.addEventListener("loadedmetadata", () => {
                previewWidget.aspectRatio = preview.videoWidth / preview.videoHeight;
                resolution.textContent = `${preview.videoWidth}×${preview.videoHeight}`;
                duration.textContent = formatTime(preview.duration);
                fps.textContent = preview.dataset.fps ? `${preview.dataset.fps} fps` : "";
                fitPreviewHeight(previewNode);
                if (autoPlay.checked) preview.play().catch(() => {});
            });
            preview.addEventListener("error", () => {
                if (preview.dataset.fallback !== "1" && preview.dataset.filename) {
                    preview.dataset.fallback = "1";
                    preview.src = transcodedVideoUrl({
                        filename: preview.dataset.filename,
                        subfolder: preview.dataset.subfolder,
                        type: preview.dataset.type,
                    });
                    preview.load();
                    return;
                }
                resolution.textContent = "Preview unavailable (PyAV transcode or browser decoder failed)";
                duration.textContent = "";
                fps.textContent = "";
            });
            let silentPlaybackTimer;
            const clearSilentPlaybackTimer = () => {
                if (silentPlaybackTimer) clearTimeout(silentPlaybackTimer);
                silentPlaybackTimer = undefined;
            };
            preview.addEventListener("timeupdate", clearSilentPlaybackTimer);
            preview.addEventListener("playing", () => {
                clearSilentPlaybackTimer();
                silentPlaybackTimer = setTimeout(() => {
                    if (preview.currentTime === 0 && preview.dataset.fallback !== "1") {
                        preview.dataset.fallback = "1";
                        preview.src = transcodedVideoUrl({ filename: preview.dataset.filename, subfolder: preview.dataset.subfolder, type: preview.dataset.type });
                        preview.load();
                    }
                }, 2500);
            });
            preview.addEventListener("mouseenter", () => {
                if (!mute.checked) preview.muted = false;
            });
            preview.addEventListener("mouseleave", () => {
                preview.muted = true;
            });
            preview.addEventListener("volumechange", () => {
                // A checked Mute checkbox permanently silences the preview, even
                // if the native player controls are used to un-mute it.
                if (this.properties?.muted && !preview.muted) preview.muted = true;
            });
            preview.addEventListener("dblclick", (event) => {
                event.preventDefault();
                event.stopPropagation();
                enterFullscreen();
            });
            [preview, info, transport, timeline, actions, saveFirstFrame, saveFirstFrameLabel, saveLastFrame, saveLastFrameLabel, autoPlay, autoPlayLabel, mute, muteLabel, fullscreen, download].forEach(stopNodeInteraction);

            root.append(preview, transport, info, actions);
            const previewHeight = () => (previewNode.size[0] - 20) / (previewWidget?.aspectRatio ?? 16 / 9) + 138;
            previewWidget = this.addDOMWidget("video_preview", "preview", root, {
                serialize: false,
                hideOnZoom: false,
                hideInPanel: true, // like core's video preview; panel sizing shrinks it (#97)
                getHeight: () => previewHeight(),
            });
            previewWidget.aspectRatio = 16 / 9;
            previewWidget.computeSize = function (width) {
                if (root.hidden) return [width, -4];
                return [width, previewHeight()];
            };
            this.dasiwaVideoPreview = preview;
            this.dasiwaVideoPreviewWidget = previewWidget;
            this.dasiwaFrameExportCheckboxes = { save_first_frame: saveFirstFrame, save_last_frame: saveLastFrame };
            this.dasiwaAutoPlayCheckbox = autoPlay;
            this.dasiwaMuteCheckbox = mute;
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
            // widgets_values / properties are applied after onNodeCreated built the
            // checkboxes, so their initial `checked` snapshot predates the restored state.
            for (const [name, checkbox] of Object.entries(this.dasiwaFrameExportCheckboxes ?? {})) {
                checkbox.checked = Boolean(this.widgets?.find((widget) => widget.name === name)?.value);
            }
            if (this.dasiwaAutoPlayCheckbox) {
                this.dasiwaAutoPlayCheckbox.checked = this.properties?.autoplay ?? true;
            }
            if (this.dasiwaMuteCheckbox) {
                this.dasiwaMuteCheckbox.checked = this.properties?.muted ?? false;
            }
            if (this.dasiwaMuteCheckbox?.checked && this.dasiwaVideoPreview) {
                this.dasiwaVideoPreview.muted = true;
            }
            return result;
        };

        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted ? originalOnExecuted.apply(this, arguments) : undefined;
            const video = message?.gifs?.[0] ?? message?.videos?.[0];
            if (!video?.filename || !this.dasiwaVideoPreview) return result;

            const preview = this.dasiwaVideoPreview;
            preview.pause();
            preview.dataset.filename = video.filename;
            preview.dataset.subfolder = video.subfolder || "";
            preview.dataset.type = video.type || "output";
            preview.dataset.fps = video.fps ?? "";
            preview.dataset.fallback = "0";
            preview.dataset.codec = video.codec || "";
            const originalUrl = videoUrl(video);
            const download = preview.parentElement.querySelector("a");
            if (download) {
                download.href = originalUrl;
                download.download = video.filename;
            }
            if (video.width && video.height) {
                this.dasiwaVideoPreviewWidget.aspectRatio = video.width / video.height;
                preview.parentElement.style.aspectRatio = String(this.dasiwaVideoPreviewWidget.aspectRatio);
            }
            // Prefer native playback (including browsers with HEVC support). If
            // decoding fails, the error handler switches once to FFmpeg streaming.
            preview.src = shouldUseTranscodedPreview(video) ? transcodedVideoUrl(video) : originalUrl;
            preview.load();
            return result;
        };
    },
});
