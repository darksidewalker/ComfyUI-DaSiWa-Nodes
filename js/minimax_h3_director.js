import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let h3VaeErrorPopupInstalled = false;

function installH3VaeErrorPopup() {
  if (h3VaeErrorPopupInstalled) return;
  h3VaeErrorPopupInstalled = true;

  api.addEventListener("execution_error", ({ detail }) => {
    if (detail?.node_type !== "MiniMaxH3DirectorGuide") return;

    const message = String(detail?.exception_message || "");

    if (
      message.includes("Audio VAE is connected to the 'vae'") ||
      message.includes("Video VAE is connected to the 'audio_vae'")
    ) {
      window.alert(
        "MiniMax H3 VAE MISMATCH\n\n" +
        message
      );
    }
  });
}

const DEFAULT_BUILDER_STATE = mode => {
  if (mode === "REF2VA") {
    return { version: 2, mode: "REF2VA", duration: 5, ref: { subject_definitions: "", summary: "", retention_analysis: "", detailed_description: "", soundscape: "", music: "" } };
  }
  return { version: 1, mode: mode || "FL2VA", imd: "", soundscape: "", music: "", duration: 5, ref: { subject_defs: [], summary_types: ["reference generation"], summary_text: "", retention: [], style_line: "", detail: "", soundscape: "", music: "" } };
};
const DEFAULT_STATE = { version: 1, items: [], prompt_blocks: [], builder_state: null, resolution: null };
const MAX = { image: 9, video: 3, audio: 3, total: 12 };
// H3's VAE emits 16px latent cells and the diffusion transformer patchifies
// them in 2×2 groups, so both canvas edges must be divisible by 32.
const MINIMAX_MULTIPLE = 32;
const ASPECT_OPTIONS = [["auto", "Auto"], ["1:1", "1:1"], ["16:9", "16:9"], ["9:16", "9:16"], ["2:1", "2:1"], ["1:2", "1:2"], ["3:2", "3:2"], ["2:3", "2:3"], ["4:3", "4:3"], ["3:4", "3:4"], ["4:5", "4:5"], ["5:4", "5:4"], ["custom", "CUSTOM"]];
const RESOLUTION_PRESETS = {
  "144p": 0.0352, "240p": 0.0977, "360p": 0.22, "480p": 0.391, "540p": 0.494, "576p": 0.396,
  "720p": 0.879, "900p": 1.373, "1024p": 1.00, "1080p": 1.978, "1152p": 2.25, "1440p": 3.516,
  "2160p": 7.91, "2K": 3.906, "4K": 7.91,
  "0.26 MP - Preview": 0.26, "0.36 MP - Small": 0.36, "0.52 MP - SD": 0.52, "0.65 MP - Balanced": 0.65,
  "0.83 MP - HD": 0.83, "1.00 MP - 1024p": 1.00, "1.05 MP - HD+": 1.05, "1.20 MP - HD++": 1.20,
  "1.35 MP - 2K lite": 1.35, "1.55 MP - 2K": 1.55, "1.65 MP - 2K+": 1.65, "1.75 MP - QHD": 1.75,
  "2.10 MP - FHD": 2.10, "3.30 MP - QHD+": 3.30, "4.75 MP - 2K Pro": 4.75, "6.50 MP - Production": 6.50, "8.30 MP - UHD": 8.30,
};
const REPOSITORY_URL = "https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/main/docs/minimax_h3_director.md";
const IMAGE_EXTENSIONS = new Set(["avif", "bmp", "gif", "heic", "heif", "jpeg", "jpg", "jxl", "png", "tif", "tiff", "webp"]);
const VIDEO_EXTENSIONS = new Set(["3gp", "avi", "flv", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts", "ts", "webm", "wmv"]);
const AUDIO_EXTENSIONS = new Set(["aac", "aif", "aiff", "alac", "amr", "ape", "caf", "flac", "m4a", "mka", "mp3", "oga", "ogg", "opus", "wav", "wave", "weba", "wma"]);
let cssInstalled = false;

function installStyles() {
  if (cssInstalled) return;
  cssInstalled = true;
  const style = document.createElement("style");
  style.textContent = `
    .ds-h3{box-sizing:border-box;width:100%;min-width:0;min-height:0;align-self:stretch;background:transparent;border:0;border-radius:0;padding:0 0 25px 0;font:12px system-ui,sans-serif;display:flex;flex-direction:column;gap:6px;overflow:visible}
    .ds-h3 button{background:#202b35;color:#dbe7f0;border:1px solid #40515e;border-radius:4px;padding:4px 7px;cursor:pointer}.ds-h3 button:hover{background:#2c3c49}.ds-h3-lane-add{position:absolute;right:6px;z-index:3;width:22px;height:22px;padding:0!important;border-radius:50%!important;font-size:17px;line-height:18px;background:rgba(70,150,105,.3)!important;border-color:rgba(126,210,157,.75)!important;color:#bff3d0!important}
    .ds-h3-actions{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.ds-h3-modebar{display:flex;gap:4px;padding:4px;background:#0d1217;border:1px solid #344452;border-radius:6px}.ds-h3-modebar button{padding:4px 9px!important;border-radius:999px!important;background:transparent!important;color:#9fb3c2!important}.ds-h3-modebar button:hover{background:rgba(126,235,167,.10)!important;box-shadow:0 0 10px rgba(126,235,167,.5)}.ds-h3-modebar button.active{color:#efe6ff!important;border-color:rgba(177,128,255,.8)!important;box-shadow:0 0 10px rgba(151,91,255,.6);font-weight:700}.ds-h3-clear-btn,.ds-h3-remove-btn{padding:3px 7px!important;font-size:11px;border-radius:999px!important}.ds-h3-modebar .ds-h3-clear-btn{background:rgba(255,100,100,.08)!important;color:#ffb0b0!important;border:1px solid rgba(255,100,100,.35)!important}.ds-h3-modebar .ds-h3-remove-btn{background:rgba(255,150,60,.08)!important;color:#ffcfab!important;border:1px solid rgba(255,150,60,.3)!important}.ds-h3-modebar .ds-h3-clear-btn:hover{background:rgba(255,100,100,.35)!important;color:#ffe2e2!important;border-color:rgba(255,100,100,.95)!important;box-shadow:0 0 14px rgba(255,100,100,.75)}.ds-h3-modebar .ds-h3-remove-btn:hover{background:rgba(255,150,60,.35)!important;color:#ffeadb!important;border-color:rgba(255,150,60,.95)!important;box-shadow:0 0 14px rgba(255,150,60,.75)}.ds-h3-modebar .ds-h3-clear-btn-empty{opacity:.45}.ds-h3-modebar .ds-h3-clear-btn-empty:hover{opacity:1}.ds-h3-modebar .ds-h3-io-dropdown{min-width:0!important;display:inline-flex!important}.ds-h3-modebar .ds-h3-io-dropdown .ds-h3-res-btn{min-height:0!important;padding:3px 8px!important;font-size:11px!important;border-radius:999px!important;gap:4px!important;font-weight:400}.ds-h3-modebar .ds-h3-io-dropdown.load .ds-h3-res-btn{background:rgba(90,160,255,.08)!important;border:1px solid rgba(90,160,255,.35)!important;color:#bcd9ff!important}.ds-h3-modebar .ds-h3-io-dropdown.load .ds-h3-res-btn:hover{background:rgba(90,160,255,.35)!important;border-color:rgba(90,160,255,.95)!important;box-shadow:0 0 14px rgba(90,160,255,.75)!important;color:#e5f1ff!important}.ds-h3-modebar .ds-h3-io-dropdown.save .ds-h3-res-btn{background:rgba(90,220,140,.08)!important;border:1px solid rgba(90,220,140,.35)!important;color:#bdf5d3!important}.ds-h3-modebar .ds-h3-io-dropdown.save .ds-h3-res-btn:hover{background:rgba(90,220,140,.35)!important;border-color:rgba(90,220,140,.95)!important;box-shadow:0 0 14px rgba(90,220,140,.75)!important;color:#e3fff0!important}.ds-h3-modebar .ds-h3-io-dropdown .ds-h3-res-caret{color:inherit!important;font-size:8px!important}.ds-h3-modebar .ds-h3-io-dropdown .ds-h3-res-menu.cols.open{gap:4px!important}.ds-h3-modebar .ds-h3-io-dropdown .ds-h3-res-col{min-width:110px!important}.ds-h3-prompt{width:100%;min-height:88px;box-sizing:border-box;background:#0d1217;color:#e5eef4;border:1px solid #40515e;border-radius:4px;padding:7px;resize:vertical}.ds-h3-prompt-panel{width:100%;box-sizing:border-box;border:0;border-radius:0;padding:0;display:flex;flex-direction:column;gap:6px;background:transparent;flex-shrink:0}.ds-h3-status{min-height:16px;color:#f3c67a;flex-shrink:0}.ds-h3-info-field{box-sizing:border-box;min-height:28px;border:1px solid #40515e;border-radius:4px;padding:6px 7px;background:#0d1217}.ds-h3-status.error{color:#ff6f6f;font-weight:700}.ds-h3-small{font-size:11px;color:#9fb3c2}.ds-h3-ruler{position:relative;height:19px;color:#8fa3b2;font-size:10px;white-space:nowrap;overflow:hidden}.ds-h3-ruler span{position:absolute;top:1px;border-left:1px solid #587084;padding-left:2px;height:16px}.ds-h3-track{position:relative;min-height:0;max-width:100%;overflow-x:auto;overflow-y:auto;background:#0b1015;border:1px solid #344452;border-radius:5px;padding:7px 6px 6px;flex-shrink:0}.ds-h3-track::before{content:none}.ds-h3-track-inner{position:relative;min-width:100%;height:360px;overflow:visible;background:repeating-linear-gradient(90deg,#111a21 0,#111a21 49px,#1b2933 50px)}.ds-h3-track-inner::after{content:'';position:absolute;left:var(--insert-x,-8px);top:0;height:100%;border-left:2px solid #f3c67a;pointer-events:none}.ds-h3-track-inner.over{outline:2px solid #8dd7ff;outline-offset:-2px}.ds-h3-timeline-lane{position:absolute;left:0;right:0;height:120px;box-sizing:border-box;border-bottom:1px solid #344452;cursor:pointer}.ds-h3-empty-slot{position:absolute;top:21px;height:88px;box-sizing:border-box;border:1px dashed #587084;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#7890a0;font-size:10px;pointer-events:none}.ds-h3-timeline-lane.disabled .ds-h3-empty-slot{display:none}.ds-h3-timeline-lane.visual{top:0;background:rgba(17,30,39,.72)}.ds-h3-timeline-lane.audio{top:120px;background:rgba(22,49,36,.55)}.ds-h3-timeline-lane.selected{box-shadow:inset 0 0 0 2px #8dd7ff}.ds-h3-timeline-lane.disabled{background:rgba(51,55,60,.72);filter:grayscale(1);cursor:not-allowed}.ds-h3-timeline-lane.disabled::after{content:"Not supported by the selected mode";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#a1a8ad;font-size:11px;font-weight:600;background:rgba(0,0,0,.35);pointer-events:none}.ds-h3-lane-label{position:absolute;left:5px;top:2px;color:#8fa3b2;font-size:10px;text-transform:uppercase;pointer-events:none;z-index:1}.ds-h3-grip{position:absolute;top:0;width:11px;height:100%;cursor:ew-resize;background:rgba(255,255,255,.22);z-index:4}.ds-h3-grip.left{left:0;border-right:1px solid rgba(255,255,255,.65)}.ds-h3-grip.right{right:0;border-left:1px solid rgba(255,255,255,.65)}.ds-h3-clip{position:absolute;top:18px;height:48px;min-width:64px;box-sizing:border-box;border:1px solid #73c7ef;border-radius:4px;background:#1b4558;color:#e5eef4;padding:6px 14px 19px;cursor:grab;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.ds-h3-clip.image,.ds-h3-clip.video{min-width:112px!important;width:112px!important;height:112px;top:7px}.ds-h3-clip.audio{border-color:#7ecf9d;background:#254b38;top:7px;height:112px}.ds-h3-waveform{position:absolute;inset:22px 12px 55px;width:calc(100% - 24px);height:calc(100% - 77px);pointer-events:none;opacity:.9}.ds-h3-crop-marker,.ds-h3-audio-crop-marker{position:absolute;top:20px;bottom:18px;width:4px;background:#fff;box-shadow:0 0 4px #000;cursor:ew-resize;z-index:6}.ds-h3-crop-marker.start,.ds-h3-audio-crop-marker.start{background:#f3c67a}.ds-h3-crop-marker.end,.ds-h3-audio-crop-marker.end{background:#8dd7ff;transform:translateX(-4px)}.ds-h3-crop-readout{position:absolute;left:14px;right:14px;bottom:3px;font-size:10px;line-height:12px;color:#d9f5e2;background:rgba(0,0,0,.36);pointer-events:none;text-align:center;overflow:hidden;white-space:nowrap}.ds-h3-clip-close{position:absolute!important;right:2px;top:2px;width:18px;height:18px;padding:0!important;line-height:15px!important;font-size:16px;color:#fff!important;background:rgba(105,28,28,.9)!important;border-color:#f08080!important;z-index:5}.ds-h3-clip.video{border-color:#b887d8;background:#432e52}.ds-h3-clip.text{border-color:#83c98a;background:#27442d}
  `;
  style.textContent += `.ds-h3-preview-overlay{position:fixed;inset:0;z-index:10001;display:flex;align-items:center;justify-content:center;background:rgba(8,10,14,.6)}.ds-h3-preview-panel{width:min(600px,90vw);max-height:85vh;display:flex;flex-direction:column;background:#111820;border:1px solid #40515e;border-radius:10px;overflow:hidden;box-shadow:0 8px 32px #000}.ds-h3-preview-header,.ds-h3-preview-meta{padding:8px 12px;background:#0d1217;color:#dbe7f0}.ds-h3-preview-header{display:flex;justify-content:space-between;border-bottom:1px solid #344452}.ds-h3-preview-body{padding:12px;background:#090d11;display:flex;justify-content:center}.ds-h3-preview-media{max-width:100%;max-height:40vh;object-fit:contain}.ds-h3-preview-controls{padding:8px 12px;background:#0d1217;display:flex;flex-direction:column;gap:6px}.ds-h3-preview-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:11px;color:#9fb3c2}.ds-h3-preview-row input[type="number"]{width:60px;padding:2px 4px;background:#111a21;color:#dbe7f0;border:1px solid #40515e;border-radius:3px}.ds-h3-preview-row input[type="text"],.ds-h3-preview-row textarea{flex:1;min-width:150px;padding:3px 5px;background:#111a21;color:#dbe7f0;border:1px solid #40515e;border-radius:3px;font-size:11px}.ds-h3-preview-row textarea{resize:vertical;min-height:30px}.ds-h3-preview-meta{font-size:11px;color:#9fb3c2;border-top:1px solid #344452}`;
  style.textContent += `.ds-h3-refmod-overlay{position:fixed;inset:0;z-index:10002;display:flex;align-items:center;justify-content:center;padding:24px;background:rgba(5,9,13,.76);backdrop-filter:blur(3px);box-sizing:border-box}.ds-h3-refmod-panel{width:min(920px,96vw);max-height:min(860px,92vh);display:flex;flex-direction:column;overflow:hidden;background:#101820;color:#dbe7f0;border:1px solid #416079;border-radius:10px;box-shadow:0 18px 60px rgba(0,0,0,.72);font:12px system-ui,sans-serif}.ds-h3-refmod-header{display:flex;justify-content:space-between;gap:24px;padding:18px 20px;background:#0b1218;border-bottom:1px solid #2d4558}.ds-h3-refmod-header h2{margin:0 0 5px;font-size:18px;color:#f0f7fb}.ds-h3-refmod-header p{margin:0;max-width:650px;color:#91a9ba;line-height:1.45}.ds-h3-refmod-close{width:32px;height:32px;padding:0;border:1px solid #405b6e;border-radius:5px;background:#16232d;color:#b9ccd9;font-size:22px;cursor:pointer}.ds-h3-refmod-content{display:flex;flex:1 1 auto;min-height:0;flex-direction:column;gap:10px;padding:16px 20px;overflow:auto;scrollbar-gutter:stable}.ds-h3-refmod-help{display:flex;flex-direction:column;gap:3px;padding:10px 12px;background:#0d2027;border-left:3px solid #56a7c7;color:#93acbb}.ds-h3-refmod-help strong{color:#ccebf5}.ds-h3-refmod-card{flex-shrink:0;border:1px solid #314b5d;border-radius:7px;background:#111c24;overflow:hidden}.ds-h3-refmod-cardhead{display:flex;align-items:center;gap:10px;padding:9px 12px;background:#0c151c;border-bottom:1px solid #2a404f}.ds-h3-refmod-cardhead h3{margin:0;font-size:13px}.ds-h3-refmod-cardhead code{color:#7ee19d}.ds-h3-refmod-cardhead .ds-h3-refmod-insert{margin-left:0;min-height:0;padding:4px 8px}.ds-h3-refmod-cardhead .ds-h3-refmod-remove{margin-left:auto}.ds-h3-refmod-grid{display:flex;flex-direction:column}.ds-h3-refmod-field{display:grid;grid-template-columns:140px minmax(180px,1fr) minmax(220px,1.25fr);gap:14px;align-items:center;min-width:0;padding:10px 12px;border-top:1px solid rgba(49,75,93,.7)}.ds-h3-refmod-field:first-child{border-top:0}.ds-h3-refmod-field strong{font-size:11px;color:#dce9ef}.ds-h3-refmod-field>span:not(.ds-h3-refmod-toggle){min-height:0;color:#829ba9;font-size:10px;line-height:1.4}.ds-h3-refmod-field select,.ds-h3-refmod-field input[type="number"],.ds-h3-refmod-field textarea{box-sizing:border-box;width:100%;min-width:0;padding:7px 8px;background:#091117;color:#e4edf2;border:1px solid #38556a;border-radius:4px}.ds-h3-refmod-field textarea{resize:vertical;min-height:72px}.ds-h3-refmod-field:has(textarea){min-height:92px}.ds-h3-refmod-toggle{display:flex;align-items:center;gap:7px;min-height:32px}.ds-h3-refmod-button{justify-content:space-between}.ds-h3-refmod-badge{margin-left:auto;padding:2px 6px;border-radius:999px;background:#173b29;color:#83e7a7;font-size:10px;font-weight:700;text-transform:none;letter-spacing:0}.ds-h3-refmod-panel button{padding:6px 10px;background:#172833;color:#dceaf1;border:1px solid #3d6075;border-radius:4px;cursor:pointer}.ds-h3-refmod-panel button:hover:not(:disabled){background:#203a49;border-color:#61a1c2}.ds-h3-refmod-panel button:disabled{opacity:.4;cursor:not-allowed}.ds-h3-refmod-add{align-self:flex-start}.ds-h3-refmod-insert{min-height:32px}.ds-h3-refmod-footer{display:flex;align-items:center;justify-content:space-between;padding:11px 20px;background:#0b1218;border-top:1px solid #2d4558;color:#91a9ba}@media(max-width:720px){.ds-h3-refmod-overlay{padding:8px}.ds-h3-refmod-field{grid-template-columns:1fr;gap:5px;align-items:start}.ds-h3-refmod-field>span:not(.ds-h3-refmod-toggle){margin-bottom:3px}}`;
  style.textContent += `
    .ds-h3-docs{font-weight:700;min-width:26px;padding:4px 8px!important}.ds-h3-ruler{display:none}.ds-h3-clip.refmod{border:2px solid #7ee19d;box-shadow:0 0 8px rgba(126,225,157,.4)}.ds-h3-refmod-badge{position:absolute;left:5px;bottom:22px;z-index:5;padding:1px 5px;border-radius:3px;background:#7ee19d;color:#0d1217;font-size:10px;font-weight:800;pointer-events:none}.ds-h3-refmod-badge-inline{position:absolute;right:8px;top:50%;transform:translateY(-50%);padding:1px 6px;border-radius:999px;background:#7ee19d;color:#0d1217;font-size:10px;font-weight:800;pointer-events:none}.ds-h3-refmod-strength{position:absolute;left:5px;top:22px;z-index:5;padding:1px 4px;border-radius:3px;background:rgba(0,0,0,.6);color:#7ee19d;font-size:9px;pointer-events:none}.ds-h3-clip.video{width:var(--clip-width)!important;min-width:180px!important}.ds-h3-clip-identity{position:absolute;left:5px;top:4px;z-index:5;padding:1px 4px;border-radius:3px;background:rgba(0,0,0,.65);color:#fff;font-size:10px;font-weight:700;pointer-events:none}.ds-h3-video-scale{position:absolute;left:12px;right:12px;top:27px;height:44px;pointer-events:none;opacity:.8;background:repeating-linear-gradient(90deg,rgba(216,174,245,.82) 0,rgba(216,174,245,.82) 1px,transparent 1px,transparent 12px),linear-gradient(transparent 48%,rgba(216,174,245,.7) 49%,rgba(216,174,245,.7) 52%,transparent 53%)}.ds-h3-prompt-panel{min-height:120px;overflow:visible}.ds-h3-prompt-field{position:relative;flex:none;min-height:90px}.ds-h3-prompt-panel .ds-h3-prompt-field>.ds-h3-prompt{height:100%;min-height:0;padding-bottom:16px;resize:none}.ds-h3-prompt-field-resizer{position:absolute;bottom:0;left:0;width:100%;height:12px;cursor:ns-resize;display:flex;justify-content:center;align-items:flex-end;padding-bottom:4px;box-sizing:border-box;z-index:2;touch-action:none}.ds-h3-prompt-field-resizer::after{content:"";width:40px;height:4px;background:rgba(255,255,255,.16);border-radius:2px}.ds-h3-prompt-field-resizer:hover::after,.ds-h3-prompt-field-resizer.active::after{background:rgba(141,215,255,.8)}
    .ds-h3-video-stream-controls{position:absolute;right:5px;top:4px;z-index:7;display:flex;gap:3px}.ds-h3-clip.selected .ds-h3-video-stream-controls{right:24px}.ds-h3-video-stream-controls button{min-width:24px;padding:3px 6px!important;font-size:11px;line-height:14px;background:rgba(10,17,23,.85)!important}.ds-h3-video-stream-controls button.active{background:rgba(125,82,188,.9)!important;border-color:#d4b3ff;color:#fff}.ds-h3-lock-icon{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:8;font-size:39px;color:#f3c67a;text-shadow:0 0 4px rgba(0,0,0,.9);pointer-events:none}.ds-h3-clip.audio-echo{cursor:default}.ds-h3-edit-btn{position:absolute;right:5px;bottom:5px;z-index:7;font-size:14px;padding:3px 6px!important;border-radius:3px!important;background:rgba(20,35,45,.8)!important;border-color:rgba(100,150,180,.5)!important}.ds-h3-empty-slot{display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#7eeba7;border:1.5px dashed #7eeba7;border-radius:6px;cursor:pointer;box-sizing:border-box;background:rgba(126,235,167,.06);transition:all .15s ease;pointer-events:auto}.ds-h3-empty-slot:hover{background:rgba(126,235,167,.18);border-color:#bff3d0;color:#d4ffe1;box-shadow:0 0 14px rgba(126,235,167,.45);transform:scale(1.02)}.ds-h3-empty-slot-dead{font-size:11px;font-weight:600;color:#6b7a85;border-color:#4a5762;background:rgba(90,100,110,.06);cursor:not-allowed}.ds-h3-empty-slot-dead:hover{background:rgba(90,100,110,.1);border-color:#6b7a85;color:#8b99a3;box-shadow:none;transform:none}.ds-h3-clip.locked{opacity:.55;filter:grayscale(.4);cursor:not-allowed}.ds-h3-timeline-lane.disabled .ds-h3-empty-slot{opacity:.15;cursor:not-allowed;pointer-events:none}.ds-h3-timeline-lane.audio .ds-h3-empty-slot{color:#5b8dd9;border-color:#5b8dd9;background:rgba(91,141,217,.06)}.ds-h3-timeline-lane.audio .ds-h3-empty-slot:hover{background:rgba(91,141,217,.18);border-color:#8bb4f0;color:#b3d4fc;box-shadow:0 0 14px rgba(91,141,217,.45)}.ds-h3-prompt-panel.disabled{opacity:.5;filter:grayscale(.55)}.ds-h3-prompt-panel.disabled textarea,.ds-h3-prompt-panel.disabled input,.ds-h3-prompt-panel.disabled button{pointer-events:none}.ds-h3-ext-note{font-weight:600;color:#7ec8f0}.ds-h3-prompt-mode-btn{white-space:nowrap}.ds-h3-modebar .ds-h3-prompt-mode-btn.active{background:rgba(126,235,167,.14)!important;color:#d7ffe3!important;border-color:rgba(126,235,167,.9)!important;box-shadow:0 0 8px rgba(126,235,167,.45);font-weight:700}.ds-h3-global-prompt{min-height:140px}
  `;
  style.textContent += `.ds-h3-res-field{display:flex;flex-direction:column;gap:3px;min-width:150px;font-size:10px;font-weight:600;letter-spacing:.4px;color:#8fb3d6;text-transform:uppercase}.ds-h3-res-control{display:flex;position:relative}.ds-h3-res-select{position:absolute;inset:0;width:100%;opacity:0;pointer-events:none}.ds-h3-res-btn{position:relative;width:100%;display:flex;align-items:center;gap:8px;box-sizing:border-box;min-height:30px;padding:0 9px;background:#16283a;border:1px solid #2f5478;border-radius:5px;color:#d6ebff;font:12px system-ui,sans-serif;cursor:pointer;text-align:left;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}.ds-h3-res-btn:hover:not(:disabled){background:#1d3550;border-color:#3f79b4;box-shadow:0 0 9px rgba(74,144,217,.28)}.ds-h3-res-btn:focus-visible{outline:none;border-color:#4f97d6;box-shadow:0 0 0 2px rgba(74,144,217,.35)}.ds-h3-res-btn:disabled{opacity:.4;cursor:not-allowed}.ds-h3-res-label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.ds-h3-res-caret{flex:none;color:#6fa8e0;font-size:9px;transition:transform .18s ease}.ds-h3-res-btn[aria-expanded="true"] .ds-h3-res-caret{transform:rotate(180deg)}.ds-h3-res-swatch{flex:none;display:flex;align-items:center;justify-content:center;width:20px;height:20px}.ds-h3-res-swatch-box{background:#3f79b4;border:1px solid #9fd0ff;border-radius:1px;box-shadow:inset 0 0 4px rgba(0,0,0,.4);transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}.ds-h3-res-btn:hover:not(:disabled) .ds-h3-res-swatch-box,.ds-h3-res-btn[aria-expanded="true"] .ds-h3-res-swatch-box{background:#5b9be0;border-color:#c4e4ff}.ds-h3-res-menu{position:absolute;top:calc(100% + 4px);left:0;z-index:2500;min-width:100%;max-height:290px;overflow-y:auto;padding:4px;background:#101c28;border:1px solid #35618f;border-radius:6px;box-shadow:0 10px 30px rgba(0,0,0,.65);display:none}.ds-h3-res-menu.open{display:block}.ds-h3-res-menu.grid.open{display:grid;gap:2px}.ds-h3-res-menu.grid .ds-h3-res-item-label{white-space:normal;overflow-wrap:anywhere}.ds-h3-res-menu[data-place="up"]{top:auto;bottom:calc(100% + 4px)}.ds-h3-res-item{display:flex;align-items:center;gap:8px;width:100%;box-sizing:border-box;padding:6px 9px;background:transparent;border:0;border-radius:4px;color:#cfe3f7;font:12px system-ui,sans-serif;cursor:pointer;text-align:left;transition:background .12s ease,color .12s ease,box-shadow .12s ease}.ds-h3-res-item:hover{background:rgba(74,144,217,.24);color:#fff;box-shadow:inset 0 0 0 1px rgba(96,168,232,.35)}.ds-h3-res-item.active{background:rgba(74,144,217,.34);color:#fff;font-weight:600;box-shadow:inset 0 0 0 1px rgba(120,190,255,.55)}.ds-h3-res-item.active:hover{background:rgba(74,144,217,.44)}.ds-h3-res-item-label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.ds-h3-res-num{width:100%;box-sizing:border-box;height:30px;padding:0 8px;background:#16283a;border:1px solid #2f5478;border-radius:5px;color:#d6ebff;font:12px system-ui,sans-serif;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease}.ds-h3-res-num:hover:not(:disabled){background:#1d3550;border-color:#3f79b4}.ds-h3-res-num:focus{outline:none;border-color:#4f97d6;box-shadow:0 0 0 2px rgba(74,144,217,.3)}.ds-h3-res-num:disabled{opacity:.4;cursor:not-allowed}.ds-h3-res-menu.cols.open{display:flex;align-items:flex-start;gap:7px}.ds-h3-res-col{display:flex;flex-direction:column;gap:2px;min-width:84px}.ds-h3-res-col-title{font-size:9px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:#6fa8e0;padding:1px 9px 3px;border-bottom:1px solid #2f5478;margin-bottom:2px}.ds-h3-res-menu.cols .ds-h3-res-item{white-space:nowrap}`;
  document.head.appendChild(style);
}

function parseState(value) { try { const s = JSON.parse(value || "{}"); return { ...DEFAULT_STATE, ...s, items: Array.isArray(s.items) ? s.items : [], prompt_blocks: Array.isArray(s.prompt_blocks) ? s.prompt_blocks : [] }; } catch { return structuredClone(DEFAULT_STATE); } }
function textValue(value) { return typeof value === "string" ? value.trim() : ""; }
function builderHasContent(builder) {
  if (!builder || typeof builder !== "object") return false;
  if (textValue(builder.simple_prompt) || textValue(builder.imd) || textValue(builder.soundscape)) return true;
  const ref = builder.ref;
  return !!(ref && typeof ref === "object" && (
    ["subject_definitions", "summary", "retention_analysis", "detailed_description", "soundscape"].some(key => textValue(ref[key])) ||
    (Array.isArray(ref.subject_defs) && ref.subject_defs.length) || (Array.isArray(ref.retention) && ref.retention.length)
  ));
}
function legacyPrompt(state, widgetPrompt) {
  for (const candidate of [state?.resolved_prompt, state?.full_prompt]) if (textValue(candidate)) return textValue(candidate);
  const global = textValue(widgetPrompt) || textValue(state?.prompt);
  const blocks = Array.isArray(state?.prompt_blocks) ? [...state.prompt_blocks].sort((a, b) => (Number(a?.start) || 0) - (Number(b?.start) || 0) || (Number(a?.order) || 0) - (Number(b?.order) || 0)) : [];
  return [global, ...blocks.filter(block => block?.enabled !== false).map(block => textValue(block?.text)).filter(Boolean)].filter(Boolean).join("\n");
}
function viewUrl(path) { return api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`); }
function count(state, type) { return state.items.filter(i => i.enabled !== false && i.type === type).length; }
function idFor(type, n) { return `${type}-${Date.now()}-${n}`; }
function mediaTypeFor(file) {
  const mimeType = String(file.type || "").toLowerCase();
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType.startsWith("video/")) return "video";
  if (mimeType.startsWith("audio/")) return "audio";
  const extension = String(file.name || "").split(".").pop().toLowerCase();
  if (IMAGE_EXTENSIONS.has(extension)) return "image";
  if (VIDEO_EXTENSIONS.has(extension)) return "video";
  return AUDIO_EXTENSIONS.has(extension) ? "audio" : null;
}
function wavDurationFromBuffer(buffer) {
  const data = new DataView(buffer);
  if (data.byteLength < 12 || data.getUint32(0, false) !== 0x52494646 || data.getUint32(8, false) !== 0x57415645) return null;
  let byteRate = null;
  let dataSize = null;
  for (let offset = 12; offset + 8 <= data.byteLength;) {
    const chunk = data.getUint32(offset, false);
    const size = data.getUint32(offset + 4, true);
    const contentOffset = offset + 8;
    if (contentOffset + size > data.byteLength) return null;
    if (chunk === 0x666d7420 && size >= 16) byteRate = data.getUint32(contentOffset + 8, true);
    if (chunk === 0x64617461) dataSize = size;
    offset = contentOffset + size + (size & 1);
  }
  return byteRate && dataSize !== null ? dataSize / byteRate : null;
}
function mediaLabel(item) {
  const name = String(item.value || "media").split("/").pop();
  const prompt = String(item.prompt || "").trim().replace(/\s+/g, " ");
  const promptPreview = prompt ? ` · “${prompt.slice(0, 42)}${prompt.length > 42 ? "…" : ""}”` : "";
  if (item.type === "image") return `${name}${promptPreview}`;
  const duration = Number(item.duration);
  const durationText = Number.isFinite(duration) ? ` · ${duration.toFixed(2)}s` : "";
  const trim = (item.type === "video" || item.type === "audio") && (Number(item.trim_start) > 0 || item.trim_end != null)
    ? ` · crop L ${Number(item.trim_start || 0).toFixed(2)}s / R ${item.trim_end == null ? "end" : `${Number(item.trim_end).toFixed(2)}s`}` : "";
  return `${name}${durationText}${trim}${promptPreview}`;
}

const mediaReferenceName = type => type === "image" ? "Picture" : type === "video" ? "Video" : type === "audio" ? "Audio" : "Media";

async function uploadFile(file, status) {
  const form = new FormData();
  form.append("image", file, file.name);
  form.append("type", "input");
  form.append("overwrite", "false");
  status.textContent = `Uploading ${file.name}…`;
  const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
  if (!response.ok) throw new Error(`upload failed (${response.status})`);
  const result = await response.json();
  return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
}

function insertAtCursor(textarea, text) {
  const start = textarea.selectionStart || 0;
  const end = textarea.selectionEnd || 0;
  textarea.value = textarea.value.slice(0, start) + text + textarea.value.slice(end);
  textarea.selectionStart = textarea.selectionEnd = start + text.length;
  textarea.focus();
  textarea.dispatchEvent(new Event("change"));
}

function createBuilderField(label, value, opts = {}, fieldHeights = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "ds-h3-prompt-field";
  wrapper.style.display = "flex"; wrapper.style.flexDirection = "column"; wrapper.style.gap = "2px"; wrapper.style.position = "relative";
  const key = opts.fieldKey || "";
  if (opts.dataAttr) wrapper.dataset.ref2vaTarget = opts.dataAttr;
  if (label) { const lbl = document.createElement("div"); lbl.className = "ds-h3-small"; lbl.textContent = label; wrapper.appendChild(lbl); }
  const area = document.createElement(opts.tag || "textarea");
  area.className = opts.class || "ds-h3-prompt";
  area.rows = opts.rows || 3;
  area.placeholder = opts.placeholder || "";
  area.value = value || "";
  if (opts.onChange) area.onchange = e => opts.onChange(e.target.value);
  wrapper.appendChild(area);
  const resizer = document.createElement("div");
  resizer.className = "ds-h3-prompt-field-resizer";
  let dragging = false, startY = 0, startH = 0;
  resizer.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; startY = ev.clientY; startH = area.clientHeight; resizer.setPointerCapture(ev.pointerId); });
  window.addEventListener("pointermove", ev => { if (!dragging) return; const height = Math.max(60, startH + (ev.clientY - startY)); area.style.height = `${height}px`; if (key) fieldHeights[key] = height; });
  window.addEventListener("pointerup", () => { if (dragging) { dragging = false; window.dispatchEvent(new CustomEvent("dasiwa-h3-field-resized")); } });
  wrapper.appendChild(resizer);
  if (key) {
    const saved = Number(fieldHeights[key]);
    if (Number.isFinite(saved) && saved >= 60) area.style.height = `${saved}px`;
  }
  return wrapper;
}

function install(node) {
  if (node.__dasiwaH3Installed) return;
  node.__dasiwaH3Installed = true;
  installStyles();
  if (!window.__dasiwaH3MenuCloseInstalled) { window.__dasiwaH3MenuCloseInstalled = true; window.addEventListener("pointerdown", event => { const open = document.querySelector(".ds-h3-res-menu.open"); if (open && !open.parentElement.contains(event.target)) open.classList.remove("open"); }, true); }
  // Resizing a prompt field's textarea never triggered a render()/emit() pass
  // on its own (it's a local DOM mutation only) so the node's own bounding
  // box -- what LiteGraph uses for viewport culling -- was never refreshed,
  // and scrolling far enough away made the whole node vanish. Resync once,
  // on release, rather than continuously while dragging.
  window.addEventListener("dasiwa-h3-field-resized", () => requestAnimationFrame(() => syncNodeBounds("fit")));
  const dataWidget = node.widgets?.find(w => w.name === "timeline_data");
  if (!dataWidget) return;
  dataWidget.hidden = true; dataWidget.options = { ...(dataWidget.options || {}), hidden: true };
  dataWidget.draw = () => {}; dataWidget.computeSize = () => [0, -4];
  let state = parseState(dataWidget.value);
  // Resized prompt text-box heights, kept per node so a re-render (media add,
  // mode/resolution change, workflow reload) restores the user's layout.
  const fieldHeights = {};
  for (const [key, value] of Object.entries(state.field_heights || {})) {
    const height = Number(value);
    if (Number.isFinite(height) && height >= 60) fieldHeights[key] = height;
  }
  const mode = () => node.widgets?.find(w => w.name === "mode")?.value || "FL2VA";
  let builderState = state.builder_state && typeof state.builder_state === "object" ? { ...DEFAULT_BUILDER_STATE(mode()), ...state.builder_state, ref: { ...DEFAULT_BUILDER_STATE(mode()).ref, ...(state.builder_state.ref || {}) } } : DEFAULT_BUILDER_STATE(mode());
  builderState.mode = mode();
  const isNaSentinel = v => typeof v === "string" && v.trim() === "N/A";
  if (isNaSentinel(builderState.music)) builderState.music = "";
  if (builderState.ref && isNaSentinel(builderState.ref.music)) builderState.ref.music = "";
  const builderWidget = node.widgets?.find(w => w.name === "builder_state");
  if (builderWidget) { builderWidget.hidden = true; builderWidget.draw = () => {}; builderWidget.computeSize = () => [0, -4]; }
  const modeWidget = node.widgets?.find(w => w.name === "mode");
  if (modeWidget) { modeWidget.hidden = true; modeWidget.draw = () => {}; modeWidget.computeSize = () => [0, -4]; }
  const widthWidget = node.widgets?.find(w => w.name === "width");
  const heightWidget = node.widgets?.find(w => w.name === "height");
  for (const widget of [widthWidget, heightWidget]) { if (widget) { widget.hidden = true; widget.draw = () => {}; widget.computeSize = () => [0, -4]; } }
  const promptWidget = node.widgets?.find(w => w.name === "prompt"); if (promptWidget) { promptWidget.hidden = true; promptWidget.draw = () => {}; promptWidget.computeSize = () => [0, -4]; }
  function migrateLegacyExternalPromptInput() {
    const legacyIndex = node.inputs?.findIndex(input => input.name === "external_prompt") ?? -1;
    if (legacyIndex < 0) return;
    const overwriteIndex = node.inputs?.findIndex(input => input.name === "external_prompt_overwrite") ?? -1;
    const legacy = node.inputs[legacyIndex];
    if (overwriteIndex >= 0) {
      const overwrite = node.inputs[overwriteIndex];
      if (legacy.link != null && overwrite.link == null) {
        node.removeInput(overwriteIndex);
        legacy.name = "external_prompt_overwrite";
        return;
      }
      node.removeInput(legacyIndex);
      return;
    }
    legacy.name = "external_prompt_overwrite";
  }
  migrateLegacyExternalPromptInput();
  const externalPromptWidget = () => node.widgets?.find(w => w.name === "external_prompt_overwrite");
  const externalPromptInput = () => node.inputs?.find(i => i.name === "external_prompt_overwrite");
  const externalCanvasInput = name => node.inputs?.find(i => i.name === name);
  const hasExternalCanvas = () => externalCanvasInput("external_width_overwrite")?.link != null && externalCanvasInput("external_height_overwrite")?.link != null;
  const hasExternalPrompt = () => {
    if (externalPromptInput()?.link != null) return true;
    const w = externalPromptWidget();
    return Boolean(w && String(w.value || "").trim().length > 0);
  };
  let lastExternalPromptLinked = hasExternalPrompt();
  let lastExternalCanvasLinked = hasExternalCanvas();
  const status = document.createElement("div"); status.className = "ds-h3-status ds-h3-info-field"; status.textContent = ""; status.style.display = "none";
  const timeline = document.createElement("div"); timeline.className = "ds-h3 ds-h3-root"; timeline.tabIndex = 0;
  timeline.addEventListener("wheel", event => {
    const openMenu = event.target instanceof Element ? event.target.closest(".ds-h3-res-menu.open") : null;
    if (openMenu && openMenu.scrollHeight > openMenu.clientHeight) return;
    const canvas = app.canvas?.canvas;
    if (!canvas) return;
    event.preventDefault();
    canvas.dispatchEvent(new WheelEvent("wheel", {
      bubbles: true, cancelable: true, clientX: event.clientX, clientY: event.clientY,
      deltaX: event.deltaX, deltaY: event.deltaY, deltaZ: event.deltaZ,
      deltaMode: event.deltaMode, ctrlKey: event.ctrlKey, shiftKey: event.shiftKey,
      altKey: event.altKey, metaKey: event.metaKey,
    }));
  }, { passive: false });
  const setStatus = (message, isError = false) => { status.textContent = message; status.classList.toggle("error", isError); };
  const emit = () => { builderState.mode = mode(); const duration = Number(node.widgets?.find(w => w.name === "duration")?.value); if (Number.isFinite(duration)) builderState.duration = duration; const resolved = builderPromptForWidget(builderState, mode()); if (promptWidget && promptWidget.value !== resolved) { promptWidget.value = resolved; promptWidget.callback?.(resolved); } state.field_heights = { ...fieldHeights }; state.builder_state = builderState; state.resolved_prompt = resolved; dataWidget.value = JSON.stringify(state); dataWidget.callback?.(dataWidget.value); if (builderWidget) { builderWidget.value = JSON.stringify(builderState); builderWidget.callback?.(builderWidget.value); } node.graph?.setDirtyCanvas(true, true); };
  const refModLibrary = { loaded: false, loading: false, entries: [], error: "" };
  let refModPromptField = null;
  let refModActiveBadge = null;
  const updateRefModActiveBadge = () => { const active = (state.refmods || []).filter(row => row.enabled !== false && row.name && Number(row.strength) !== 0).length; if (refModActiveBadge) refModActiveBadge.textContent = `${active} active`; };
  let activeRefModOverlayCleanup = null;
  let activeRefModLibraryRefresh = null;
  const onRefModReloadKey = event => {
    if (event.key.toLowerCase() !== "r" || event.ctrlKey || event.metaKey || event.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(event.target?.tagName || "")) return;
    refModLibrary.loaded = false; refModLibrary.error = "";
    activeRefModLibraryRefresh?.();
  };
  document.addEventListener("keydown", onRefModReloadKey);
  function openRefModOverlay(trigger) {
    if (document.querySelector(".ds-h3-refmod-overlay")) return;
    const overlay = document.createElement("div"); overlay.className = "ds-h3-refmod-overlay"; overlay.setAttribute("role", "dialog"); overlay.setAttribute("aria-modal", "true"); overlay.setAttribute("aria-labelledby", "ds-h3-refmod-title");
    const panel = document.createElement("section"); panel.className = "ds-h3-refmod-panel";
    const header = document.createElement("header"); header.className = "ds-h3-refmod-header";
    const heading = document.createElement("div");
    const title = document.createElement("h2"); title.id = "ds-h3-refmod-title"; title.textContent = "Saved RefMod references";
    const intro = document.createElement("p"); intro.textContent = "Attach reusable H3 reference latents to stable prompt tags. Selections and descriptions are saved with this workflow.";
    heading.append(title, intro);
    const close = document.createElement("button"); close.type = "button"; close.className = "ds-h3-refmod-close"; close.textContent = "×"; close.title = "Close RefMod settings"; close.setAttribute("aria-label", "Close RefMod settings");
    header.append(heading, close);
    const content = document.createElement("div"); content.className = "ds-h3-refmod-content";
    const footer = document.createElement("footer"); footer.className = "ds-h3-refmod-footer";
    const count = document.createElement("span");
    const done = document.createElement("button"); done.type = "button"; done.textContent = "Done";
    footer.append(count, done); panel.append(header, content, footer); overlay.append(panel); document.body.append(overlay);
    const closeOverlay = () => { document.removeEventListener("keydown", onKey); overlay.remove(); activeRefModLibraryRefresh = null; activeRefModOverlayCleanup = null; trigger?.focus(); };
    const onKey = event => {
      if (event.key === "Escape") { closeOverlay(); return; }
      if (event.key === "Tab") {
        const focusable = [...panel.querySelectorAll('button:not(:disabled),select:not(:disabled),input:not(:disabled),textarea:not(:disabled),[tabindex]:not([tabindex="-1"])')];
        if (!focusable.length) { event.preventDefault(); return; }
        const first = focusable[0]; const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    };
    activeRefModOverlayCleanup = closeOverlay;
    close.onclick = closeOverlay; done.onclick = closeOverlay; overlay.onpointerdown = event => { if (event.target === overlay) closeOverlay(); }; document.addEventListener("keydown", onKey);
    const help = (headline, text) => { const box = document.createElement("div"); box.className = "ds-h3-refmod-help"; const strong = document.createElement("strong"); strong.textContent = headline; const copy = document.createElement("span"); copy.textContent = text; box.append(strong, copy); return box; };
    const redraw = () => {
      content.replaceChildren();
      const rows = state.refmods || (state.refmods = []);
      count.textContent = `${rows.filter(row => row.enabled !== false && row.name).length} of 8 references enabled`; updateRefModActiveBadge();
      if (refModLibrary.loading) { content.append(help("Loading reference library", "Reading RefMod metadata from models/refmods and its subfolders…")); return; }
      if (refModLibrary.error) { content.append(help("Reference library unavailable", refModLibrary.error)); return; }
      content.append(help("How this works", "1. Choose a saved reference. 2. Enable it and set its influence. 3. Click INSERT IN PROMPT to insert its full expanded reference text—e.g. <Video 1>: digital animation, ...—at your cursor."));
      const add = document.createElement("button"); add.type = "button"; add.className = "ds-h3-refmod-add"; add.textContent = "+ Add reference"; add.disabled = rows.length >= 8;
      add.onclick = () => { const used = new Set(rows.map(row => row.slot)); const slot = Array.from({ length: 8 }, (_, index) => index + 1).find(value => !used.has(value)); if (slot) { rows.push({ slot, name: "", description: "", strength: 1, enabled: false }); emit(); redraw(); } };
      content.append(add);
      if (!rows.length) content.append(help("No references selected", "Add a reference to bind a file from models/refmods to a prompt tag."));
      for (const row of [...rows].sort((a, b) => a.slot - b.slot)) {
        const card = document.createElement("section"); card.className = "ds-h3-refmod-card";
        const cardHead = document.createElement("div"); cardHead.className = "ds-h3-refmod-cardhead";
        const cardTitle = document.createElement("h3"); cardTitle.textContent = `Reference ${row.slot}`;
        const alias = document.createElement("code"); alias.textContent = `<RefMod ${row.slot}>`;
        const insert = document.createElement("button"); insert.type = "button"; insert.className = "ds-h3-refmod-insert"; insert.textContent = "INSERT IN PROMPT"; insert.disabled = !row.name || row.enabled === false || Number(row.strength) === 0; insert.onpointerdown = event => event.preventDefault(); let insertTarget = null; insert.addEventListener("mousedown", () => { const active = document.activeElement; insertTarget = (active && active.tagName === "TEXTAREA" ? active : null); }); insert.onclick = () => { const promptField = insertTarget || refModPromptField || timeline.querySelector(".ds-h3-prompt-panel textarea:not(:disabled)"); insertExpandedRefMod(promptField, row.slot); };
        const remove = document.createElement("button"); remove.type = "button"; remove.className = "ds-h3-refmod-remove"; remove.textContent = "Remove"; remove.onclick = () => { state.refmods = rows.filter(item => item !== row); emit(); redraw(); render(); };
        cardHead.append(cardTitle, alias, insert, remove); card.append(cardHead);
        const grid = document.createElement("div"); grid.className = "ds-h3-refmod-grid";
        const field = (headline, description, control) => { const label = document.createElement("label"); label.className = "ds-h3-refmod-field"; const name = document.createElement("strong"); name.textContent = headline; const hint = document.createElement("span"); hint.textContent = description; label.append(name, control, hint); return label; };
        const select = document.createElement("select"); select.append(new Option("Select a saved RefMod…", "")); for (const entry of refModLibrary.entries) select.append(new Option(entry.name, entry.name)); select.value = row.name || "";
        select.onchange = () => { row.name = select.value; const entry = refModLibrary.entries.find(item => item.name === row.name); if (entry) { row.description = entry.description || ""; row.media_type = entry.kind; } else delete row.media_type; row.enabled = Boolean(row.name); emit(); redraw(); render(); };
        const strength = document.createElement("input"); strength.type = "number"; strength.min = "0"; strength.max = "1"; strength.step = "0.05"; strength.value = row.strength ?? 1; strength.onchange = () => { const parsed = Number(strength.value); row.strength = Number.isFinite(parsed) ? Math.max(0, Math.min(1, parsed)) : 1; strength.value = row.strength; emit(); updateRefModActiveBadge(); render(); };
        const enabledWrap = document.createElement("span"); enabledWrap.className = "ds-h3-refmod-toggle"; const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = row.enabled !== false; enabled.onchange = () => { row.enabled = enabled.checked; emit(); redraw(); render(); }; const enabledText = document.createElement("span"); enabledText.textContent = enabled.checked ? "Enabled" : "Disabled"; enabledWrap.append(enabled, enabledText);
        const description = document.createElement("textarea"); description.value = row.description || ""; description.placeholder = "Describe the person, character, motion, voice, or concept represented by this reference."; description.rows = 3; description.oninput = () => { row.description = description.value; emit(); };
        grid.append(
          field("Reference library", "Select a standalone .safetensors RefMod from models/refmods. Subfolders appear as paths.", select),
          field("Reference strength", "Controls how strongly this saved latent influences generation. 1 is full strength; 0 disables it.", strength),
          field("Use reference", "Include this reference in the next generation without removing its saved settings.", enabledWrap),
          field("Prompt description", "Workflow-owned context for what the reference represents. This overrides the file description.", description),
        );
        card.append(grid); content.append(card);
      }
    };
    const refreshLibrary = () => {
      if (refModLibrary.loading) return;
      refModLibrary.loading = true; refModLibrary.error = ""; redraw();
      void api.fetchApi("/dasiwa/refmods").then(response => { if (!response.ok) throw new Error("Could not read models/refmods."); return response.json(); }).then(entries => { refModLibrary.entries = entries; refModLibrary.loaded = true; }).catch(error => { refModLibrary.error = error.message; }).finally(() => { refModLibrary.loading = false; if (overlay.isConnected) redraw(); });
    };
    activeRefModLibraryRefresh = refreshLibrary;
    redraw(); close.focus();
    if (!refModLibrary.loaded) refreshLibrary();
  }
  const promptStyle = () => { const v = builderState?.prompt_mode; return v === "simple" || v === "structured" ? v : "structured"; };
  function legacySimplePromptFor(m) {
    if (m === "REF2VA") {
      const r = builderState.ref || {};
      return [
        r.subject_definitions ? `subject_definitions: ${r.subject_definitions}` : "",
        r.summary ? `summary: ${r.summary}` : "",
        r.retention_analysis ? `retention_analysis: ${r.retention_analysis}` : "",
        r.detailed_description ? `detailed_description: ${r.detailed_description}` : "",
        r.soundscape ? `overall_soundscape: ${r.soundscape}` : "",
        `non_diegetic_music: ${r.music || "N/A"}`,
      ].filter(Boolean).join("\n");
    }
    return [
      builderState.imd ? `integrated_multimodal_description: ${builderState.imd}` : "",
      builderState.soundscape ? `overall_soundscape: ${builderState.soundscape}` : "",
      `non_diegetic_music: ${builderState.music || "N/A"}`,
    ].filter(Boolean).join("\n");
  }
  function parseSimplePromptToStructured(text, m) {
    const fields = {};
    if (m === "REF2VA") {
      const labels = ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"];
      let rest = text;
      for (const label of labels) {
        const idx = rest.indexOf(`${label}:`);
        if (idx === -1) { fields[label] = ""; continue; }
        rest = rest.slice(idx + label.length + 1).trimStart();
        const nextLabelIdx = labels.slice(labels.indexOf(label) + 1).reduce((min, l) => { const i = rest.indexOf(`${l}:`); return i !== -1 && (min === -1 || i < min) ? i : min; }, -1);
        fields[label] = nextLabelIdx === -1 ? rest.trim() : rest.slice(0, nextLabelIdx).trim();
        if (nextLabelIdx !== -1) rest = rest.slice(nextLabelIdx);
      }
    } else {
      const labels = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"];
      let rest = text;
      for (const label of labels) {
        const idx = rest.indexOf(`${label}:`);
        if (idx === -1) { fields[label] = ""; continue; }
        rest = rest.slice(idx + label.length + 1).trimStart();
        const nextLabelIdx = labels.slice(labels.indexOf(label) + 1).reduce((min, l) => { const i = rest.indexOf(`${l}:`); return i !== -1 && (min === -1 || i < min) ? i : min; }, -1);
        fields[label] = nextLabelIdx === -1 ? rest.trim() : rest.slice(0, nextLabelIdx).trim();
        if (nextLabelIdx !== -1) rest = rest.slice(nextLabelIdx);
      }
    }
    const anyFilled = Object.values(fields).some(v => v && v.trim());
    if (!anyFilled && text && text.trim()) {
      if (m === "REF2VA") fields.detailed_description = text.trim();
      else fields.integrated_multimodal_description = text.trim();
    }
    return fields;
  }

  function builderPromptForWidget(builder, m) {
    if (builder?.prompt_mode === "simple" && typeof builder.simple_prompt === "string") return builder.simple_prompt;
    const r = builder?.ref || {};
    if (m === "REF2VA") return `subject_definitions:\n${r.subject_definitions || ""}\n\nsummary:\n${r.summary || ""}\n\nretention_analysis:\n${r.retention_analysis || ""}\n\ndetailed_description:\n${r.detailed_description || ""}\n\noverall_soundscape:\n${r.soundscape || ""}\n\nnon_diegetic_music:\n${r.music || "N/A"}`;
    const music = builder?.music || "N/A";
    let head = "";
    if (m === "I2VA") head = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
    else if (m === "FL2VA") head = "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the end mark of the target video.";
    else if (m === "L2VA") head = "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the end mark of the target video.";
    const body = `integrated_multimodal_description: ${builder?.imd || ""}\n\noverall_soundscape: ${builder?.soundscape || ""}\n\nnon_diegetic_music: ${music}`;
    return head ? `${head}\n\n${body}` : body;
  }
  function previewTextFor(m, external) {
    if (external) {
      const t = String(externalPromptWidget()?.value || "").trim();
      return t || "(External prompt detected — the prompt is supplied by the upstream node at execution time.)";
    }
    if (promptStyle() === "simple") return typeof builderState.simple_prompt === "string" ? builderState.simple_prompt : legacySimplePromptFor(m);
    const areas = timeline.querySelectorAll(".ds-h3-prompt-panel .ds-h3-prompt");
    if (m === "REF2VA") {
      const vals = []; areas.forEach(a => vals.push(String(a.value || "").trim()));
      const subj = vals[0] || ""; const summ = vals[1] || ""; const ret = vals[2] || ""; const detail = vals[3] || ""; const sound = vals[4] || ""; const music = vals[5] || "N/A";
      if (promptStyle() === "simple") {
        const parts = [
          subj ? `subject_definitions: ${subj}` : "",
          summ ? `summary: ${summ}` : "",
          ret ? `retention_analysis: ${ret}` : "",
          detail ? `detailed_description: ${detail}` : "",
          sound ? `overall_soundscape: ${sound}` : "",
          `non_diegetic_music: ${music || "N/A"}`,
        ].filter(Boolean);
        return parts.join("\n");
      }
      return `subject_definitions:\n${subj}\n\nsummary:\n${summ}\n\nretention_analysis:\n${ret}\n\ndetailed_description:\n${detail}\n\noverall_soundscape:\n${sound}\n\nnon_diegetic_music:\n${music}`;
    }
    const imd = String(areas[0]?.value || "").trim(); const sound = String(areas[1]?.value || "").trim(); const music = String(areas[2]?.value || "N/A").trim();
    let head = "";
    if (m === "I2VA") head = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
    else if (m === "FL2VA") head = "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the end mark of the target video.";
    else if (m === "L2VA") head = "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the end mark of the target video.";
    if (promptStyle() === "simple") {
      const parts = [
        imd ? `integrated_multimodal_description: ${imd}` : "",
        sound ? `overall_soundscape: ${sound}` : "",
        `non_diegetic_music: ${music || "N/A"}`,
      ].filter(Boolean);
      return parts.join("\n");
    }
    const body = `integrated_multimodal_description: ${imd}\n\noverall_soundscape: ${sound}\n\nnon_diegetic_music: ${music}`;
    return head ? `${head}\n\n${body}` : body;
  }
  const allowNativeTextEditing = element => { ["pointerdown","mousedown","keydown","keypress","keyup","copy","cut","paste"].forEach(type => element.addEventListener(type, event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") return; event.stopPropagation(); })); };

  function addCharCounter(panel) {
    const helpers = panel.querySelector(".ds-h3-actions");
    if (!helpers) return null;
    const counter = document.createElement("span");
    counter.className = "ds-h3-char-counter";
    counter.style.cssText = "margin-left:auto;color:#8fa3b2;font-size:10px;font-variant-numeric:tabular-nums;white-space:nowrap;padding-top:4px";
    helpers.appendChild(counter);
    const updateCount = () => {
      try {
        const text = previewTextFor(mode(), externalPromptWidget()?.value);
        counter.textContent = `${text.length} chars`;
      } catch (e) {
        let total = 0;
        panel.querySelectorAll("textarea").forEach(ta => { total += (ta.value || "").length; });
        counter.textContent = `${total} chars`;
      }
    };
    panel.addEventListener("input", updateCount);
    updateCount();
    return counter;
  }

  function buildBaseForm(panel) {
    panel.replaceChildren();
    const label = document.createElement("div"); label.className = "ds-h3-small"; label.textContent = `${mode()} prompt builder`; panel.appendChild(label);
    const imdArea = createBuilderField("integrated_multimodal_description", builderState.imd, { rows: 6, placeholder: "[Shot 1] Start your scene description here...", onChange: val => { builderState.imd = val; emit(); }, fieldKey: "imd" }, fieldHeights);
    allowNativeTextEditing(imdArea.querySelector("textarea"));
    const helpers = document.createElement("div"); helpers.className = "ds-h3-actions";
    const shotBtn = document.createElement("button"); shotBtn.textContent = "Insert [Shot N]"; shotBtn.onclick = () => { const n = window.prompt("Shot number:", "1"); if (n) insertAtCursor(imdArea.querySelector("textarea"), `[Shot ${n}] `); }; helpers.appendChild(shotBtn);
    const refmodBtn = document.createElement("button"); refmodBtn.textContent = "Insert RefMod #"; let baseRefTarget = null; refmodBtn.addEventListener("mousedown", () => { const active = document.activeElement; baseRefTarget = (active && active.tagName === "TEXTAREA" ? active : null); }); refmodBtn.onclick = () => { const n = window.prompt("RefMod number:", "1"); if (n) insertExpandedRefMod(baseRefTarget || imdArea.querySelector("textarea"), n); }; helpers.appendChild(refmodBtn);
    panel.appendChild(helpers); panel.appendChild(imdArea);
    const soundscape = createBuilderField("overall_soundscape", builderState.soundscape, { rows: 3, placeholder: "Describe ambient sounds, dialogue, effects...", onChange: val => { builderState.soundscape = val; emit(); }, fieldKey: "soundscape" }, fieldHeights); allowNativeTextEditing(soundscape.querySelector("textarea")); panel.appendChild(soundscape);
    const music = createBuilderField("non_diegetic_music", builderState.music, { rows: 2, placeholder: 'N/A or describe background score...', onChange: val => { builderState.music = val; emit(); }, fieldKey: "music" }, fieldHeights); allowNativeTextEditing(music.querySelector("textarea")); panel.appendChild(music);
    addCharCounter(panel);
  }

  function buildSimpleForm(panel) {
    panel.replaceChildren();
    const label = document.createElement("div"); label.className = "ds-h3-small"; label.textContent = `${mode()} simple prompt`; panel.appendChild(label);
    const helpers = document.createElement("div"); helpers.className = "ds-h3-actions";
    const shotBtn = document.createElement("button"); shotBtn.textContent = "Insert [Shot N]"; helpers.appendChild(shotBtn);
    const refmodBtn = document.createElement("button"); refmodBtn.textContent = "Insert RefMod #"; helpers.appendChild(refmodBtn);
    panel.appendChild(helpers);
    const simplePrompt = createBuilderField("Prompt", builderState.simple_prompt, { rows: 10, placeholder: "Write the complete MiniMax H3 prompt...", onChange: val => { builderState.simple_prompt = val; emit(); }, fieldKey: "simple_prompt" }, fieldHeights);
    allowNativeTextEditing(simplePrompt.querySelector("textarea"));
    shotBtn.onclick = () => { const n = window.prompt("Shot number:", "1"); if (n) insertAtCursor(simplePrompt.querySelector("textarea"), `[Shot ${n}] `); };
    refmodBtn.textContent = "Insert RefMod #"; let simpleRefTarget = null; refmodBtn.addEventListener("mousedown", () => { const active = document.activeElement; simpleRefTarget = (active && active.tagName === "TEXTAREA" ? active : null); }); refmodBtn.onclick = () => { const n = window.prompt("RefMod number:", "1"); if (n) insertExpandedRefMod(simpleRefTarget || simplePrompt.querySelector("textarea"), n); };
    panel.appendChild(simplePrompt);
    addCharCounter(panel);
  }

  function buildRefForm(panel) {
    panel.replaceChildren();
    const r = builderState.ref || {};
    const label = document.createElement("div"); label.className = "ds-h3-small"; label.textContent = "REF2VA prompt builder — write freely; headers are added automatically"; panel.appendChild(label);
    const helpers = document.createElement("div"); helpers.className = "ds-h3-actions";
    const shotBtn = document.createElement("button"); shotBtn.textContent = "Insert [Shot N]"; shotBtn.onclick = () => { const wrapper = panel.querySelector("[data-ref2va-target='detail']"); const ta = wrapper?.querySelector("textarea") ?? wrapper; const n = window.prompt("Shot number:", "1"); if (n && ta) insertAtCursor(ta, `[Shot ${n}] `); }; helpers.appendChild(shotBtn);
    const refmodInsertBtn = document.createElement("button"); refmodInsertBtn.textContent = "Insert RefMod #"; refmodInsertBtn.title = "Insert a selected RefMod's expanded native label and description at cursor position"; let refmodBuilderTarget = null; refmodInsertBtn.addEventListener("mousedown", () => { const active = document.activeElement; refmodBuilderTarget = (active && active.tagName === "TEXTAREA" ? active : null); }); refmodInsertBtn.onclick = () => { const target = refmodBuilderTarget || panel.querySelector("[data-ref2va-target='detail'] textarea") || panel.querySelector("[data-ref2va-target='detail']"); if (!target) return; const n = window.prompt("RefMod number:", "1"); if (n) insertExpandedRefMod(target, n); }; helpers.appendChild(refmodInsertBtn);
    const prefillBtn = document.createElement("button"); prefillBtn.textContent = "Prefill Labels & Summary"; prefillBtn.title = "Generate Subject/Picture/Video/Audio labels from inserted media and fill summary template"; prefillBtn.onclick = () => { generateRefLabelsAndSummary(panel); }; helpers.appendChild(prefillBtn);
    const previewBtn = document.createElement("button"); previewBtn.textContent = "Preview Prompt"; previewBtn.title = "Show the full prompt that will be sent upstream"; previewBtn.onclick = () => { showPromptPreview(); }; helpers.appendChild(previewBtn);
    panel.appendChild(helpers);
    const subjField = createBuilderField("subject_definitions:", r.subject_definitions, { rows: 4, placeholder: "<Subject 1> is the woman in <Picture 1>, with short dark hair and a red coat.\n<Picture 1> is the opening-frame anchor for [Shot 1].\n<Subject 2> is the walking motion taken from <Video 1>.\n<Video 1> provides the camera path and pacing structure.\n<Audio 1> is the voice-timbre reference for <Subject 1>.", onChange: val => { r.subject_definitions = val; emit(); }, fieldKey: "ref_subject_definitions" }, fieldHeights); allowNativeTextEditing(subjField.querySelector("textarea")); panel.appendChild(subjField);
    const summaryField = createBuilderField("summary:", r.summary, { rows: 3, placeholder: "[reference generation + audio reference] Use <Subject 1> from <Picture 1>, the motion and pacing of <Video 1>, and the voice character of <Audio 1>.", onChange: val => { r.summary = val; emit(); }, fieldKey: "ref_summary" }, fieldHeights); allowNativeTextEditing(summaryField.querySelector("textarea")); panel.appendChild(summaryField);
    const retField = createBuilderField("retention_analysis:", r.retention_analysis, { rows: 4, placeholder: "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity and clothing remain consistent.\n<Picture 1> ([Shot 1] first frame): fully_preserved - opening composition anchor.\n<Subject 2> (motion transferred to <Subject 1>): attribute_transfer - walk rhythm is applied to <Subject 1>.\n<Video 1> (pacing structure): weak_reference - general timing and camera rhythm are retained.\n<Audio 1>: reference - timbre and delivery are followed without copying the signal.", onChange: val => { r.retention_analysis = val; emit(); }, fieldKey: "ref_retention_analysis" }, fieldHeights); allowNativeTextEditing(retField.querySelector("textarea")); panel.appendChild(retField);
    const detailField = createBuilderField("detailed_description:", r.detailed_description, { rows: 5, placeholder: "[Shot 1] ... [Shot 2] At 00:04.500, ...", dataAttr: "detail", onChange: val => { r.detailed_description = val; emit(); }, fieldKey: "ref_detailed_description" }, fieldHeights); allowNativeTextEditing(detailField.querySelector("textarea")); panel.appendChild(detailField);
    const rsoundField = createBuilderField("overall_soundscape:", r.soundscape, { rows: 2, placeholder: "Audio environment description...", onChange: val => { r.soundscape = val; emit(); }, fieldKey: "ref_soundscape" }, fieldHeights); allowNativeTextEditing(rsoundField.querySelector("textarea")); panel.appendChild(rsoundField);
    const rmusicField = createBuilderField("non_diegetic_music:", r.music, { rows: 2, placeholder: "N/A or background score...", onChange: val => { r.music = val; emit(); }, fieldKey: "ref_music" }, fieldHeights); allowNativeTextEditing(rmusicField.querySelector("textarea")); panel.appendChild(rmusicField);
    addCharCounter(panel);
  }

  function generateRefLabelsAndSummary(panel) {
    const items = activeItems().sort((a, b) => {
      const typeOrder = { image: 0, video: 1, audio: 2 };
      return (typeOrder[a.type] ?? 3) - (typeOrder[b.type] ?? 3) || a.slot - b.slot;
    });
    const pictures = []; const videos = []; const audios = [];
    items.forEach(item => {
      if (item.type === "image") pictures.push(item);
      else if (item.type === "video") videos.push(item);
      else if (item.type === "audio") audios.push(item);
    });
    const subjLines = [];
    let pictureIdx = 1; let videoIdx = 1; let audioIdx = 1;
    pictures.forEach(item => {
      subjLines.push(`<Picture ${pictureIdx}> is the opening-frame anchor.`);
      pictureIdx++;
    });
    videos.forEach(item => {
      subjLines.push(`<Video ${videoIdx}> provides the camera path and pacing structure.`);
      videoIdx++;
    });
    audios.forEach(item => {
      subjLines.push(`<Audio ${audioIdx}> is the voice-timbre and audio reference.`);
      audioIdx++;
    });
    const subjArea = panel.querySelector("[data-ref2va-target='subj']")?.querySelector("textarea") || panel.querySelectorAll(".ds-h3-prompt")[0]?.querySelector("textarea") || panel.querySelectorAll(".ds-h3-prompt")[0];
    if (subjArea && subjLines.length > 0) {
      subjArea.value = subjLines.join("\n");
      builderState.ref.subject_definitions = subjArea.value;
    }
    const summaryArea = panel.querySelectorAll(".ds-h3-prompt")[1]?.querySelector("textarea") || panel.querySelectorAll(".ds-h3-prompt")[1];
    if (summaryArea) {
      const refs = [];
      pictures.forEach((_, i) => refs.push(`<Picture ${i + 1}>`));
      videos.forEach((_, i) => refs.push(`<Video ${i + 1}>`));
      audios.forEach((_, i) => refs.push(`<Audio ${i + 1}>`));
      const taskPrefixes = [];
      if (pictures.length > 0) taskPrefixes.push("reference generation");
      if (videos.length > 0) taskPrefixes.push("video editing");
      if (audios.length > 0) taskPrefixes.push("audio reference");
      const prefix = taskPrefixes.length ? `[${taskPrefixes.join(" + ")}] ` : "";
      const summaryLine = prefix + "Use " + refs.map(r => r).join(", ") + ".";
      summaryArea.value = summaryLine;
      builderState.ref.summary = summaryArea.value;
    }
    emit();
  }

  function refModTagMap() {
    const counts = { image: 0, video: 0, audio: 0 };
    for (const item of activeItems()) {
      if (item.type === "image") counts.image++;
      else if (item.type === "video" && item.media_mode !== "audio") counts.video++;
      else if (item.type === "audio" || (item.type === "video" && item.media_mode === "audio")) counts.audio++;
    }
    const tags = {}, expansions = {}, descriptions = [];
    const rows = (state.refmods || []).filter(r => r.name && r.enabled !== false && Number(r.strength ?? 1) > 0).sort((a, b) => a.slot - b.slot);
    for (const row of rows) {
      const entry = refModLibrary.entries?.find(e => e.name === row.name);
      if (!entry) continue;
      const labels = (Array.isArray(entry.kinds) && entry.kinds.length ? entry.kinds : [entry.kind]).map(kind => {
        counts[kind]++;
        return `<${{ image: "Picture", video: "Video", audio: "Audio" }[kind]} ${counts[kind]}>`;
      });
      const tag = labels.join(" ");
      const description = (row.description || "").trim();
      tags[row.slot] = tag;
      expansions[row.slot] = description ? `${tag}: ${description}` : tag;
      if (description) descriptions.push(expansions[row.slot]);
    }
    return { tags, expansions, descriptions };
  }

  function insertExpandedRefMod(textarea, slot) {
    const expansion = refModTagMap().expansions[Number(slot)];
    if (!textarea || !expansion) { setStatus("Choose and enable that saved reference before inserting it.", true); return; }
    insertAtCursor(textarea, expansion + " ");
  }

  function refModTranslatePreview(text) {
    const { tags, descriptions } = refModTagMap();
    let result = text.replace(/<\s*refmod\s*_?\s*(\d+)(?:\s*:[^>]+)?\s*>/gi, (match, slot) => {
      if (!tags[slot]) return match; // leave unknown tags as-is instead of throwing
      return tags[slot];
    });
    if (descriptions.length) result += "\n\nReference descriptions:\n" + descriptions.join("\n");
    return result;
  }

  function showPromptPreview() {
    const m = mode();
    let promptText = previewTextFor(m, hasExternalPrompt());
    if (m === "REF2VA" && !hasExternalPrompt()) {
      try { promptText = refModTranslatePreview(promptText); } catch (error) { setStatus(error.message, true); return; }
    }
    let overlay = document.createElement("div"); overlay.style.cssText = "position:fixed;inset:0;z-index:10002;display:flex;align-items:center;justify-content:center;background:rgba(8,10,14,.7);";
    overlay.onclick = event => { if (event.target === overlay) overlay.remove(); };
    const panel = document.createElement("div"); panel.style.cssText = "width:min(720px,90vw);max-height:85vh;display:flex;flex-direction:column;background:#111820;border:1px solid #40515e;border-radius:10px;overflow:hidden;box-shadow:0 8px 32px #000;";
    const header = document.createElement("div"); header.style.cssText = "padding:8px 12px;background:#0d1217;color:#dbe7f0;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #344452;";
    header.innerHTML = `<span>Prompt Preview (${m} · ${promptStyle() === "simple" ? "simple" : "structured"}${hasExternalPrompt() ? " · external" : ""})</span><button title="Copy">📋</button>`;
    const copyBtn = header.querySelector("button");
    copyBtn.onclick = () => { navigator.clipboard.writeText(promptText).then(() => { copyBtn.textContent = "✓"; setTimeout(() => copyBtn.textContent = "📋", 1200); }).catch(() => { copyBtn.textContent = "×"; }); };
    const closeBtn = document.createElement("button"); closeBtn.textContent = "×"; closeBtn.style.cssText = "background:none;border:none;color:#fff;font-size:18px;cursor:pointer;padding:0 4px;"; closeBtn.onclick = () => overlay.remove(); header.appendChild(closeBtn);
    const body = document.createElement("div"); body.style.cssText = "padding:10px 12px;overflow:auto;";
    const textarea = document.createElement("textarea"); textarea.readOnly = true; textarea.value = promptText; textarea.style.cssText = "width:100%;min-height:300px;max-height:60vh;resize:vertical;background:#090d11;color:#e5eef4;border:1px solid #40515e;border-radius:4px;padding:7px;font-family:monospace;font-size:11px;line-height:1.4;box-sizing:border-box;"; body.appendChild(textarea);
    const footer = document.createElement("div"); footer.style.cssText = "padding:8px 12px;background:#0d1217;display:flex;justify-content:flex-end;gap:6px;border-top:1px solid #344452;";
    const doneBtn = document.createElement("button"); doneBtn.textContent = "Done"; doneBtn.onclick = () => overlay.remove(); footer.appendChild(doneBtn);
    panel.append(header, body, footer); overlay.appendChild(panel); document.body.appendChild(overlay);
    window.addEventListener("keydown", event => { if (event.key === "Escape") overlay.remove(); }, { once: true });
  }

  let previewOverlay = null;
  let previewCloseFn = null;
  let previewItemRef = null;
  function closePreview(discard = false) {
    if (!discard && previewOverlay && previewItemRef) {
      applyPreviewChanges(previewOverlay, previewItemRef);
    }
    previewOverlay?.remove();
    previewOverlay = null;
    previewItemRef = null;
    if (previewCloseFn) window.removeEventListener("keydown", previewCloseFn);
    previewCloseFn = null;
  }
  function applyPreviewChanges(overlay, item) {
    const tsInput = overlay.querySelector(".ds-h3-ts-input");
    const teInput = overlay.querySelector(".ds-h3-te-input");
    const sourceDuration = Number(item.source_duration) || Number(item.duration) || 0;
    if ((item.type === "video" || item.type === "audio") && sourceDuration > 0) {
      const ts = parseFloat(tsInput?.value) ?? 0;
      const te = parseFloat(teInput?.value) ?? sourceDuration;
      mutate(s => {
        const x = s.items.find(i => i.id === item.id);
        if (x) {
          x.trim_start = Math.min(sourceDuration, Math.max(0, ts));
          x.trim_end = Math.min(sourceDuration, Math.max(x.trim_start + 2, te));
          x.duration = x.trim_end - x.trim_start;
        }
      });
    }
  }
  function openPreview(item) {
    if (isLockedSlot(item)) { setStatus("This slot isn't used in L2VA and can't be edited.", true); return; }
    closePreview();
    previewItemRef = item;
    const sourceDuration = Number(item.source_duration) || Number(item.duration) || 0;
    const trimStart = Number(item.trim_start || 0);
    const trimEnd = Number(item.trim_end ?? sourceDuration);
    previewOverlay = document.createElement("div"); previewOverlay.className = "ds-h3-preview-overlay";
    previewOverlay.onclick = event => { if (event.target === previewOverlay) closePreview(); };
    const panel = document.createElement("div"); panel.className = "ds-h3-preview-panel";
    const header = document.createElement("div"); header.className = "ds-h3-preview-header";
    const title = document.createElement("span"); title.textContent = `${item.type}: ${String(item.value || "").split("/").pop()}`;
    const close = document.createElement("button"); close.textContent = "×"; close.title = "Close and save (Escape or click outside)"; close.onclick = () => closePreview(false); header.append(title, close);
    const body = document.createElement("div"); body.className = "ds-h3-preview-body";
    const media = document.createElement(item.type === "image" ? "img" : item.type === "audio" ? "audio" : "video"); media.className = "ds-h3-preview-media"; media.src = viewUrl(item.value); if (item.type !== "image") media.controls = true; body.append(media);
    const controls = document.createElement("div"); controls.className = "ds-h3-preview-controls";
    if ((item.type === "video" || item.type === "audio") && sourceDuration > 0) {
      const row1 = document.createElement("div"); row1.className = "ds-h3-preview-row";
      const lblTs = document.createElement("span"); lblTs.textContent = "Start:"; row1.appendChild(lblTs);
      const tsInput = document.createElement("input"); tsInput.type = "number"; tsInput.step = "0.25"; tsInput.min = "0"; tsInput.max = String(sourceDuration.toFixed(2)); tsInput.value = String(trimStart.toFixed(2)); tsInput.className = "ds-h3-ts-input"; row1.appendChild(tsInput);
      const lblTe = document.createElement("span"); lblTe.textContent = "End:"; row1.appendChild(lblTe);
      const teInput = document.createElement("input"); teInput.type = "number"; teInput.step = "0.25"; teInput.min = "0"; teInput.max = String(sourceDuration.toFixed(2)); teInput.value = String(trimEnd.toFixed(2)); teInput.className = "ds-h3-te-input"; row1.appendChild(teInput);
      const lblDur = document.createElement("span"); lblDur.textContent = `/ ${sourceDuration.toFixed(2)}s`; row1.appendChild(lblDur);
      const playCropBtn = document.createElement("button"); playCropBtn.textContent = "▶ Play crop"; playCropBtn.title = "Play only the current crop range"; playCropBtn.className = "ds-h3-play-crop-btn"; row1.appendChild(playCropBtn);
      controls.appendChild(row1);
      const rowSlider = document.createElement("div"); rowSlider.className = "ds-h3-preview-row"; rowSlider.style.flexDirection = "column"; rowSlider.style.alignItems = "stretch";
      const minRef = Math.min(2, sourceDuration);
      const updateFromInputs = () => {
        let ts = parseFloat(tsInput.value) ?? 0;
        let te = parseFloat(teInput.value) ?? sourceDuration;
        ts = Math.min(Math.max(0, ts), sourceDuration - minRef);
        te = Math.max(ts + minRef, Math.min(sourceDuration, te));
        tsInput.value = ts.toFixed(2);
        teInput.value = te.toFixed(2);
        const sPct = (ts / sourceDuration) * 100;
        const ePct = (te / sourceDuration) * 100;
        rangeTs.value = sPct; rangeTe.value = ePct;
        syncMarkers();
      };
      const trackBg = document.createElement("div"); trackBg.title = "Drag the highlighted crop to move it; drag either end marker to resize it."; trackBg.style.cssText = "position:relative;height:18px;background:#111a21;border-radius:3px;margin-top:4px;margin-bottom:4px;cursor:grab;";
      const filledBar = document.createElement("div"); filledBar.style.cssText = "position:absolute;top:0;left:0;right:0;bottom:0;border-radius:3px;background:rgba(126,210,157,.25);pointer-events:none;";
      const markerS = document.createElement("div"); markerS.style.cssText = "position:absolute;top:-2px;width:8px;height:22px;background:#f3c67a;border-radius:2px;z-index:3;cursor:pointer;transform:translateX(-50%);";
      const markerE = document.createElement("div"); markerE.style.cssText = "position:absolute;top:-2px;width:8px;height:22px;background:#8dd7ff;border-radius:2px;z-index:3;cursor:pointer;transform:translateX(-50%);";
      const rangeTs = document.createElement("input"); rangeTs.type = "range"; rangeTs.min = "0"; rangeTs.max = "100"; rangeTs.step = "0.1"; rangeTs.value = ((trimStart / sourceDuration) * 100).toFixed(1); rangeTs.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;pointer-events:none;z-index:0;";
      const rangeTe = document.createElement("input"); rangeTe.type = "range"; rangeTe.min = "0"; rangeTe.max = "100"; rangeTe.step = "0.1"; rangeTe.value = ((trimEnd / sourceDuration) * 100).toFixed(1); rangeTe.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;pointer-events:none;z-index:0;";
      const MARKER_GRAB_RADIUS_PX = 12;
      const pointerPct = pointerX => {
        const rect = trackBg.getBoundingClientRect();
        return ((pointerX - rect.left) / rect.width) * 100;
      };
      const syncMarkers = () => {
        const sPct = parseFloat(rangeTs.value);
        const ePct = parseFloat(rangeTe.value);
        markerS.style.left = sPct + "%";
        markerE.style.left = ePct + "%";
        filledBar.style.left = sPct + "%";
        filledBar.style.right = (100 - ePct) + "%";
      };
      const clampSliders = () => {
        let s = parseFloat(rangeTs.value);
        let e = parseFloat(rangeTe.value);
        const gap = (minRef / sourceDuration) * 100;
        if (e - s < gap) {
          if (dragging === "left") e = s + gap;
          else s = e - gap;
        }
        rangeTs.value = Math.max(0, Math.min(s, 100 - gap));
        rangeTe.value = Math.max(gap, Math.min(e, 100));
        syncMarkers();
        const tsSec = (parseFloat(rangeTs.value) / 100) * sourceDuration;
        const teSec = (parseFloat(rangeTe.value) / 100) * sourceDuration;
        tsInput.value = tsSec.toFixed(2);
        teInput.value = teSec.toFixed(2);
      };
      let dragging = null;
      let cropDragOffset = 0;
      const onStart = pointerX => {
        const pct = pointerPct(pointerX);
        const sPct = parseFloat(rangeTs.value);
        const ePct = parseFloat(rangeTe.value);
        const markerGrabRadius = (MARKER_GRAB_RADIUS_PX / trackBg.getBoundingClientRect().width) * 100;
        if (Math.abs(pct - sPct) <= markerGrabRadius && Math.abs(pct - sPct) <= Math.abs(pct - ePct)) {
          dragging = "left";
        } else if (Math.abs(pct - ePct) <= markerGrabRadius) {
          dragging = "right";
        } else if (pct > sPct && pct < ePct) {
          dragging = "range";
          cropDragOffset = pct - sPct;
        } else {
          dragging = null;
        }
      };
      const onMove = pointerX => {
        if (!dragging) return;
        const pct = pointerPct(pointerX);
        if (dragging === "left") {
          rangeTs.value = Math.max(0, Math.min(100, pct));
        } else if (dragging === "right") {
          rangeTe.value = Math.max(0, Math.min(100, pct));
        } else {
          const width = parseFloat(rangeTe.value) - parseFloat(rangeTs.value);
          const start = Math.max(0, Math.min(100 - width, pct - cropDragOffset));
          rangeTs.value = start;
          rangeTe.value = start + width;
        }
        clampSliders();
      };
      const onEnd = () => { dragging = null; trackBg.style.cursor = "grab"; };
      const onPointerStart = pointerX => { onStart(pointerX); if (dragging) trackBg.style.cursor = "grabbing"; };
      trackBg.addEventListener("mousedown", event => { onPointerStart(event.clientX); });
      window.addEventListener("mousemove", event => { onMove(event.clientX); });
      window.addEventListener("mouseup", () => { onEnd(); });
      trackBg.addEventListener("touchstart", event => { onPointerStart(event.touches[0].clientX); }, { passive: false });
      trackBg.addEventListener("touchmove", event => { event.preventDefault(); onMove(event.touches[0].clientX); }, { passive: false });
      trackBg.addEventListener("touchend", () => { onEnd(); });
      tsInput.onchange = updateFromInputs;
      teInput.onchange = updateFromInputs;
      let cropPlayback = false;
      const stopCropPlayback = () => { cropPlayback = false; playCropBtn.textContent = "▶ Play crop"; };
      playCropBtn.onclick = async () => {
        const start = Number(tsInput.value);
        const end = Number(teInput.value);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
        media.currentTime = start;
        cropPlayback = true;
        playCropBtn.textContent = "Playing crop…";
        try { await media.play(); }
        catch (error) { console.warn("[MiniMax H3 Director] Crop playback failed", error); stopCropPlayback(); }
      };
      media.addEventListener("timeupdate", () => {
        if (cropPlayback && media.currentTime >= Number(teInput.value)) {
          media.pause();
          media.currentTime = Number(teInput.value);
          stopCropPlayback();
        }
      });
      media.addEventListener("pause", () => {
        if (cropPlayback && media.currentTime < Number(teInput.value)) stopCropPlayback();
      });
      media.addEventListener("ended", stopCropPlayback);
      updateFromInputs();
      trackBg.append(filledBar, markerS, markerE, rangeTs, rangeTe);
      rowSlider.appendChild(trackBg);
      controls.appendChild(rowSlider);
    }
    const row2 = document.createElement("div"); row2.className = "ds-h3-preview-row";
    const saveBtn = document.createElement("button"); saveBtn.textContent = "Save"; saveBtn.className = "ds-h3-save-btn"; saveBtn.onclick = () => { applyPreviewChanges(previewOverlay, item); closePreview(false); }; row2.appendChild(saveBtn);
    const cancelBtn = document.createElement("button"); cancelBtn.textContent = "Cancel"; cancelBtn.onclick = () => { closePreview(true); }; cancelBtn.style.background = "#2c2c2c"; row2.appendChild(cancelBtn);
    controls.appendChild(row2);
    const meta = document.createElement("div"); meta.className = "ds-h3-preview-meta"; meta.textContent = sourceDuration && (item.type === "video" || item.type === "audio") ? `Source: ${sourceDuration.toFixed(2)}s · Current crop: ${trimStart.toFixed(2)}s – ${trimEnd.toFixed(2)}s` : "Full media preview";
    panel.append(header, body, controls, meta); previewOverlay.append(panel); document.body.append(previewOverlay);
    previewCloseFn = event => { if (event.key === "Escape") closePreview(); };
    window.addEventListener("keydown", previewCloseFn);
  }
  const syncAttachedPrompts = () => { const attached = state.items.filter(item => String(item.prompt || "").trim()).map((item, index) => ({ id: `attached-${item.id}`, text: String(item.prompt).trim(), enabled: item.enabled !== false, start: Number(item.start) || 0, duration: Number(item.duration) || 1, order: index })); if (attached.length || state.items.every(item => !item.prompt)) state.prompt_blocks = attached; };
  const mutate = fn => { fn(state); syncAttachedPrompts(); state.items.forEach((x, i) => { x.order = i; }); state.prompt_blocks.forEach((x, i) => { x.order = i; }); applyResolution?.(); emit(); render(); };
  const activeItems = () => state.items.filter(x => x.enabled !== false);
  const isReferenceMode = () => mode() === "REF2VA";
  const allowsType = type => isReferenceMode() || (mode() === "Image Inpaint" ? type === "image" : ((mode() === "I2VA" || mode() === "FL2VA" || mode() === "L2VA") && type === "image"));
  const frameSlots = () => mode() === "T2VA" ? [] : mode() === "I2VA" ? [0] : mode() === "FL2VA" ? [0, 1] : mode() === "L2VA" ? [0, 1] : mode() === "Image Inpaint" ? [0] : [];
  const imageCapacity = () => frameSlots().length;
  const isLockedSlot = (item) => mode() === "L2VA" && laneForItem(item) === "image" && item.slot === 0;
  const displayedItems = () => isReferenceMode() ? activeItems() : activeItems().filter(x => x.type === "image" && frameSlots().includes(x.slot)).sort((a, b) => a.slot - b.slot);
  // Synthesize pseudo-timeline items from active RefMods so they appear in the lanes.
  const refmodTimelineItems = () => (state.refmods || []).filter(row => row.enabled !== false && row.name && Number(row.strength) !== 0).sort((a, b) => a.slot - b.slot).map((row, index) => {
    let mediaType = row.media_type;
    if (!mediaType && refModLibrary.loaded) mediaType = refModLibrary.entries.find(entry => entry.name === row.name)?.media_type;
    mediaType ||= "image";
    const type = mediaType === "video" || mediaType === "audio" ? mediaType : "image";
    return { id: `refmod_${row.slot}`, type, value: row.name, slot: index, start: index, duration: type === "image" ? 1 : 2, _isRefMod: true, refmodSlot: row.slot, strength: Number(row.strength) || 1, media_type: mediaType };
  });
  const ensureLayout = () => { let cursor = 0; state.items.forEach(item => { if (!Number.isFinite(item.start)) item.start = cursor; if (!Number.isFinite(item.duration)) item.duration = item.type === "image" ? 1 : 2; cursor = Math.max(cursor, item.start + item.duration + 0.25); }); };
  let insertAt = 0;
  let selectedId = null;
  let selectedLane = "image";
  let lastTimelineLength = null;
  // Content-fit height is measured from the DOM. It grows the node when
  // content needs more room, and an explicit fit-to-content pass can shrink it.
  // LiteGraph's own node resize is handled separately and never gets overwritten.
  let trackedContentHeight = 850;
  const hiddenLanes = new Set();
  const laneForItem = item => { if (item._audioEcho) return "audio"; if (item.type === "audio") return "audio"; if (item.type === "video") return item.media_mode === "audio" ? "audio" : "video"; return "image"; };
  const hasAudioEcho = item => item.type === "video" && item.media_mode === "video_audio";
  // L2VA displays 2 slots (frameSlots() below) so the same pair of references
  // can be reused when swapping over from FL2VA, but slot 0 is a display-only
  // holdover never read by generation -- new insertions always target slot 1.
  const availableSlots = lane => lane === "audio" ? [...Array(isReferenceMode() ? MAX.audio : 0).keys()] : lane === "video" ? [...Array(isReferenceMode() ? MAX.video : 0).keys()] : isReferenceMode() ? [...Array(MAX.image).keys()] : mode() === "L2VA" ? [1] : frameSlots();
  // A video's own .slot always means its primary lane position (Video lane
  // for V/V+A, Audio lane for A-only). V+A additionally reserves a *second*,
  // independent slot in the Audio lane specifically, tracked via .audioSlot
  // -- this is the one place that needs to know about that second field.
  const occupiedSlotsForLane = (items, lane) => {
    const occupied = new Set(items.filter(x => laneForItem(x) === lane).map(x => x.slot).filter(Number.isInteger));
    if (lane === "audio") items.filter(hasAudioEcho).forEach(x => { if (Number.isInteger(x.audioSlot)) occupied.add(x.audioSlot); });
    return occupied;
  };
  // Handles switching a video between V / A / V+A. A-only moves the item's
  // primary slot into the Audio lane outright; V+A keeps the primary slot in
  // Video and additionally claims a second, independent slot in Audio. Where
  // an item's Audio-lane presence continues across the switch (A -> V+A or
  // V+A -> A), it keeps the same slot number rather than jumping elsewhere.
  // Returns false (leaving target untouched) if the destination lane has no
  // free slot, so the caller can surface that instead of corrupting state.
  const retargetVideoSlots = (items, target, newMode) => {
    const oldMode = ["video", "audio", "video_audio"].includes(target.media_mode) ? target.media_mode : "video";
    if (oldMode === newMode) return true;
    const others = items.filter(x => x.id !== target.id);
    const findFreeSlot = lane => { const occupied = occupiedSlotsForLane(others, lane); return availableSlots(lane).find(index => !occupied.has(index)); };
    const wasInVideoLane = oldMode === "video" || oldMode === "video_audio";
    const willBeInVideoLane = newMode === "video" || newMode === "video_audio";
    if (wasInVideoLane && newMode === "audio") {
      const reused = Number.isInteger(target.audioSlot) ? target.audioSlot : findFreeSlot("audio");
      if (reused == null) return false;
      target.slot = reused; target.start = reused; delete target.audioSlot;
    } else if (oldMode === "audio" && willBeInVideoLane) {
      const newSlot = findFreeSlot("video");
      if (newSlot == null) return false;
      if (newMode === "video_audio") target.audioSlot = target.slot;
      target.slot = newSlot; target.start = newSlot;
    } else if (wasInVideoLane && willBeInVideoLane) {
      if (newMode === "video_audio" && !Number.isInteger(target.audioSlot)) {
        const echoSlot = findFreeSlot("audio");
        if (echoSlot == null) return false;
        target.audioSlot = echoSlot;
      } else if (newMode === "video") {
        delete target.audioSlot;
      }
    }
    target.media_mode = newMode;
    return true;
  };
  const addItem = item => mutate(s => { const lane = laneForItem(item); const occupied = occupiedSlotsForLane(s.items, lane); const slot = availableSlots(lane).find(index => !occupied.has(index)); if (slot == null) throw new Error(`No free ${lane} slot is available.`); s.items.push({ id: idFor(item.type, s.items.length), enabled: true, order: s.items.length, slot, start: slot, duration: item.type === "image" ? 1 : 2, ...item }); });
  const remove = id => mutate(s => { s.items = s.items.filter(x => x.id !== id); if (selectedId === id) selectedId = null; });
  const resetBuilderState = () => { builderState = DEFAULT_BUILDER_STATE(mode()); builderState.mode = mode(); };
  const hasBuilderContent = () => [builderState.imd, builderState.soundscape, builderState.simple_prompt, ...Object.values(builderState.ref || {})].some(value => typeof value === "string" && value !== "N/A" && value.trim());
  const clearAll = () => { selectedId = null; resetBuilderState(); if (promptWidget) { promptWidget.value = ""; promptWidget.callback?.(promptWidget.value); } mutate(s => { s.items = []; s.prompt_blocks = []; (s.refmods || []).forEach(row => { row.enabled = false; }); }); updateRefModActiveBadge(); setStatus("All media and prompts cleared."); };

  // --- Reference-pack save/load ---
  const REFERENCE_PACK_MARKER = "dasiwa_minimax_h3_reference_pack";
  const VALID_MODES = ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA", "Image Inpaint"];
  const PORTABLE_ITEM_KEYS = ["type", "value", "media_mode", "audioSlot", "trim_start", "trim_end", "duration", "source_duration", "source_width", "source_height"];
  const toPortableItem = item => { const out = {}; for (const key of PORTABLE_ITEM_KEYS) if (item[key] !== undefined) out[key] = item[key]; return out; };
  const countOf = (items, type) => items.filter(i => i.type === type).length;
  // Counts by which lane an item actually occupies (accounting for A-mode and
  // V+A videos redirecting to/also occupying Audio), not raw item.type. A V+A
  // video counts once toward video and once toward audio here, matching how
  // the Python side's validate_reference_limits double-counts it too.
  const countInLane = (items, lane) => items.filter(i => laneForItem(i) === lane).length + (lane === "audio" ? items.filter(hasAudioEcho).length : 0);
  // HEAD request against the same /view endpoint thumbnails/waveforms already
  // fetch from -- cheap way to tell "file's gone" apart from "just no preview
  // yet" before we commit to adding an item nobody can actually use.
  async function fileExistsOnServer(value) {
    try {
      const response = await fetch(viewUrl(value), { method: "HEAD" });
      return response.ok;
    } catch {
      return false;
    }
  }
  // Real insertion capacity per Model Mode, independent of the live mode() --
  // needed because a failsafe check has to evaluate the file's *target* mode
  // before we've actually switched into it. L2VA's slot 0 is a display-only
  // holdover (see availableSlots/isLockedSlot) so its real capacity is 1, not
  // the 2 slots frameSlots() reports for rendering purposes.
  const insertableSlotCount = (type, targetMode) => {
    if (targetMode === "REF2VA") return MAX[type];
    if (type !== "image") return 0;
    if (targetMode === "T2VA") return 0;
    if (targetMode === "L2VA" || targetMode === "I2VA" || targetMode === "Image Inpaint") return 1;
    if (targetMode === "FL2VA") return 2;
    return 0;
  };
  // Saves each item's rank (0-based position among same-type items, ordered
  // by its current slot) instead of the raw slot number itself, so Load can
  // preserve "which one was first" when remapping onto a different mode's
  // slot space rather than depending on raw slot numbers that may not even
  // exist in the target mode (e.g. REF2VA's slot 7 has no equivalent in FL2VA).
  const buildPortableItems = () => {
    const byType = { image: [], video: [], audio: [] };
    activeItems().filter(i => byType[i.type]).slice().sort((a, b) => (Number(a.slot) || 0) - (Number(b.slot) || 0)).forEach(i => byType[i.type].push(i));
    const out = [];
    for (const type of ["image", "video", "audio"]) byType[type].forEach((item, rank) => out.push({ ...toPortableItem(item), _rank: rank }));
    return out;
  };
  function buildPortablePrompt() {
    if (promptStyle() === "simple") return { prompt_mode: "simple", simple_prompt: String(builderState.simple_prompt || "") };
    if (mode() === "REF2VA") {
      const r = builderState.ref || {};
      return { prompt_mode: "structured", fields: { subject_definitions: r.subject_definitions || "", summary: r.summary || "", retention_analysis: r.retention_analysis || "", detailed_description: r.detailed_description || "", soundscape: r.soundscape || "", music: r.music || "" } };
    }
    return { prompt_mode: "structured", fields: { imd: builderState.imd || "", soundscape: builderState.soundscape || "", music: builderState.music || "" } };
  }
  const joinText = (a, b) => { a = String(a || "").trim(); b = String(b || "").trim(); return a && b ? `${a}\n\n${b}` : (a || b); };
  // Appends into whatever's already there. By the time this runs, Model Mode
  // has already been switched to match the file (see performLoad), so the
  // saved field shape (ref.* vs imd/soundscape/music) should always match --
  // the flatten-to-Simple branch below is a defensive fallback only.
  function appendPortablePrompt(saved) {
    if (!saved || typeof saved !== "object") return;
    const savedMode = saved.prompt_mode === "simple" ? "simple" : "structured";
    builderState.prompt_mode = savedMode;
    if (savedMode === "simple") { builderState.simple_prompt = joinText(builderState.simple_prompt, saved.simple_prompt); return; }
    const savedFields = saved.fields || {};
    const savedIsRef = Object.prototype.hasOwnProperty.call(savedFields, "subject_definitions");
    const currentIsRef = mode() === "REF2VA";
    if (savedIsRef && currentIsRef) { builderState.ref = builderState.ref || {}; for (const key of ["subject_definitions", "summary", "retention_analysis", "detailed_description", "soundscape", "music"]) builderState.ref[key] = joinText(builderState.ref[key], savedFields[key]); return; }
    if (!savedIsRef && !currentIsRef) { for (const key of ["imd", "soundscape", "music"]) builderState[key] = joinText(builderState[key], savedFields[key]); return; }
    const flat = Object.entries(savedFields).filter(([, v]) => String(v || "").trim()).map(([k, v]) => `${k}: ${v}`).join("\n");
    builderState.prompt_mode = "simple"; builderState.simple_prompt = joinText(builderState.simple_prompt, flat);
  }
  // Replaces outright, same spirit as Clear but scoped to prompt data only.
  function overwritePortablePrompt(saved) {
    if (!saved || typeof saved !== "object") { const fresh = DEFAULT_BUILDER_STATE(mode()); builderState.prompt_mode = fresh.prompt_mode; builderState.simple_prompt = fresh.simple_prompt; builderState.ref = fresh.ref; builderState.imd = fresh.imd; builderState.soundscape = fresh.soundscape; builderState.music = fresh.music; return; }
    const savedMode = saved.prompt_mode === "simple" ? "simple" : "structured";
    builderState.prompt_mode = savedMode;
    if (savedMode === "simple") { builderState.simple_prompt = String(saved.simple_prompt || ""); return; }
    const savedFields = saved.fields || {};
    const savedIsRef = Object.prototype.hasOwnProperty.call(savedFields, "subject_definitions");
    if (savedIsRef) { builderState.ref = { ...DEFAULT_BUILDER_STATE("REF2VA").ref, ...savedFields }; }
    else { builderState.imd = savedFields.imd || ""; builderState.soundscape = savedFields.soundscape || ""; builderState.music = savedFields.music || ""; }
  }
  // Places incoming items rank-first (lowest rank = the slot the target mode
  // actually reads, e.g. Picture 1 / L2VA's working slot), skipping anything
  // beyond what fits -- the failsafe in performLoad should already have
  // blocked that case, this is a defensive backstop only.
  function placeIncomingItems(incoming) {
    const byType = { image: [], video: [], audio: [] };
    incoming.forEach(i => { if (byType[i.type]) byType[i.type].push(i); });
    for (const type of ["image", "video", "audio"]) byType[type].sort((a, b) => (Number(a._rank) || 0) - (Number(b._rank) || 0));
    const added = [];
    mutate(s => {
      for (const type of ["image", "video", "audio"]) {
        for (const raw of byType[type]) {
          const lane = laneForItem({ type, media_mode: raw.media_mode });
          const occupied = occupiedSlotsForLane(s.items, lane);
          const slot = availableSlots(lane).find(index => !occupied.has(index));
          if (slot == null) continue;
          const { _rank, audioSlot, ...clean } = raw;
          const newItem = { id: idFor(type, s.items.length), enabled: true, order: s.items.length, slot, start: slot, duration: type === "image" ? 1 : 2, ...clean };
          if (type === "video" && newItem.media_mode === "video_audio") {
            const echoOccupied = occupiedSlotsForLane(s.items, "audio");
            const echoSlot = availableSlots("audio").find(index => !echoOccupied.has(index));
            if (echoSlot != null) newItem.audioSlot = echoSlot;
            else newItem.media_mode = "video"; // no room for the audio echo -- fall back to video-only rather than a broken half-linked state
          }
          s.items.push(newItem); added.push(newItem);
        }
      }
    });
    return added;
  }
  async function saveReferencePack(dataType) {
    const pack = { [REFERENCE_PACK_MARKER]: true, schema_version: 1, saved_at: new Date().toISOString(), model_mode: mode() };
    if (dataType === "files" || dataType === "all") { pack.items = buildPortableItems(); pack.refmods = state.refmods || []; }
    if (dataType === "prompt" || dataType === "all") pack.prompt = buildPortablePrompt();
    const text = JSON.stringify(pack, null, 2);
    const suggestedName = `minimax-h3-${dataType}-pack-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    try {
      if (window.showSaveFilePicker) {
        const handle = await window.showSaveFilePicker({ suggestedName, types: [{ description: "MiniMax H3 reference pack", accept: { "application/json": [".json"] } }] });
        const writable = await handle.createWritable();
        await writable.write(text); await writable.close();
        setStatus("Saved."); return;
      }
    } catch (error) {
      if (error?.name === "AbortError") return;
      console.warn("[MiniMax H3 Director] showSaveFilePicker failed, falling back to download", error);
    }
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = suggestedName; document.body.append(link); link.click(); link.remove();
    URL.revokeObjectURL(url);
    setStatus("Downloaded to your browser's default downloads folder.");
  }
  // loadMode: "append" (never overwrites, errors and stops on overflow) or
  // "overwrite" (replaces whichever data types were selected, same spirit as
  // Clear). dataType: "files" | "prompt" | "all".
  async function performLoad(pack, loadMode, dataType) {
    if (!pack || pack[REFERENCE_PACK_MARKER] !== true) { setStatus("That file isn't a MiniMax H3 reference pack.", true); return; }
    const targetMode = VALID_MODES.includes(pack.model_mode) ? pack.model_mode : mode();
    const wantsFiles = (dataType === "files" || dataType === "all") && Array.isArray(pack.items);
    const wantsPrompt = (dataType === "prompt" || dataType === "all") && pack.prompt;
    const rawIncomingItems = wantsFiles ? pack.items.filter(i => i && ["image", "video", "audio"].includes(i.type)) : [];
    const missingCounts = { image: 0, video: 0, audio: 0 };
    let incomingItems = rawIncomingItems;
    if (rawIncomingItems.length) {
      const checked = await Promise.all(rawIncomingItems.map(async item => ({ item, exists: await fileExistsOnServer(item.value) })));
      incomingItems = checked.filter(c => c.exists).map(c => c.item);
      for (const { item, exists } of checked) if (!exists) missingCounts[item.type] += 1;
    }

    if (wantsFiles) {
      const problems = [];
      for (const lane of ["image", "video", "audio"]) {
        const cap = insertableSlotCount(lane, targetMode);
        const currentCount = loadMode === "append" ? countInLane(activeItems(), lane) : 0;
        const projected = currentCount + countInLane(incomingItems, lane);
        if (projected > cap) problems.push({ label: mediaReferenceName(lane).toLowerCase(), exceedBy: projected - cap });
      }
      if (targetMode === "REF2VA") {
        const laneSum = items => ["image", "video", "audio"].reduce((sum, lane) => sum + countInLane(items, lane), 0);
        const currentTotal = loadMode === "append" ? laneSum(activeItems()) : 0;
        const projectedTotal = currentTotal + laneSum(incomingItems);
        if (projectedTotal > MAX.total) problems.push({ label: "combined total", exceedBy: projectedTotal - MAX.total });
      }
      if (problems.length) {
        window.alert(problems.map(p => `Json reference ${p.label} exceeds max number of files allowed for ${targetMode}. Remove ${p.exceedBy} and try again`).join("\n"));
        return;
      }
    }

    if (targetMode !== mode() && modeWidget) {
      modeWidget.value = targetMode; modeWidget.callback?.(targetMode);
      if ((selectedLane === "audio" || selectedLane === "video") && targetMode !== "REF2VA") selectedLane = "image";
    }

    if (wantsPrompt) { if (loadMode === "overwrite") overwritePortablePrompt(pack.prompt); else appendPortablePrompt(pack.prompt); }
    let added = [];
    if (wantsFiles) {
      if (loadMode === "overwrite") mutate(s => { s.items = s.items.filter(i => !["image", "video", "audio"].includes(i.type)); s.refmods = []; });
      else if (Array.isArray(pack.refmods)) mutate(s => { s.refmods = [...(s.refmods || []), ...pack.refmods]; });
      added = placeIncomingItems(incomingItems);
    }
    emit(); render();

    for (const item of added) {
      if (item.type === "video") { try { const thumb = await captureFirstFrame(viewUrl(item.value)); if (thumb) mutate(s => { const target = s.items.find(x => x.id === item.id); if (target) target.thumbnail = thumb; }); } catch { /* source file may not exist locally yet */ } }
      else if (item.type === "audio") { void extractWaveform(item.value, item.id); }
    }
    const missingParts = ["image", "video", "audio"].filter(type => missingCounts[type] > 0).map(type => { const label = mediaReferenceName(type).toLowerCase(); const count = missingCounts[type]; return `${count} ${label}${count === 1 ? "" : "s"} ${count === 1 ? "was" : "were"} missing and not loaded.`; });
    if (missingParts.length) window.alert(missingParts.join("\n"));
    setStatus(`${loadMode === "overwrite" ? "Overwrote" : "Appended"} ${dataType === "all" ? "reference files and prompt" : dataType === "files" ? "reference files" : "prompt"} from pack (${targetMode}).`);
  }
  function loadReferencePack(loadMode, dataType) {
    const input = document.createElement("input"); input.type = "file"; input.accept = "application/json,.json";
    input.onchange = async () => {
      const file = input.files?.[0]; if (!file) return;
      let pack;
      try { pack = JSON.parse(await file.text()); } catch { setStatus("That file isn't valid JSON.", true); return; }
      performLoad(pack, loadMode, dataType);
    };
    input.click();
  }
  timeline.addEventListener("keydown", event => { if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return; if ((event.key === "Delete" || event.key === "Backspace") && selectedId) { const sel = state.items.find(x => x.id === selectedId); if (sel && isLockedSlot(sel)) { setStatus(`${mediaReferenceName(sel.type)} ${sel.slot + 1} is locked in L2VA mode`, true); return; } event.preventDefault(); event.stopPropagation(); remove(selectedId); } });
  const move = (id, delta) => mutate(s => { const i = s.items.findIndex(x => x.id === id); const j = i + delta; if (i >= 0 && j >= 0 && j < s.items.length) [s.items[i], s.items[j]] = [s.items[j], s.items[i]]; });
  const replace = (id, value) => mutate(s => { const x = s.items.find(i => i.id === id); if (x) x.value = value; });
  const resolutionState = () => ({ aspect: "auto", resolution: "auto", input_scaling: "Auto", custom_aspect_w: 16, custom_aspect_h: 9, custom_mp: 1, custom_width: 1344, custom_height: 768, ...(state.resolution || {}) });
  const snap16 = value => Math.max(MINIMAX_MULTIPLE, Math.round(Number(value) / MINIMAX_MULTIPLE) * MINIMAX_MULTIPLE);
  const sourceDimensions = () => activeItems().filter(item => (item.type === "image" || item.type === "video") && Number(item.source_width) > 0 && Number(item.source_height) > 0).sort((a, b) => (Number(a.slot) || 0) - (Number(b.slot) || 0) || a.order - b.order)[0];
  const selectedAspect = settings => {
    if (settings.aspect === "auto") { const source = sourceDimensions(); return source ? Number(source.source_width) / Number(source.source_height) : 4 / 3; }
    if (settings.aspect === "custom") return Math.max(1, Number(settings.custom_aspect_w) || 16) / Math.max(1, Number(settings.custom_aspect_h) || 9);
    const [w, h] = String(settings.aspect).split(":").map(Number); return w > 0 && h > 0 ? w / h : null;
  };
  const setCanvasWidgets = (width, height) => { for (const [widget, value] of [[widthWidget, width], [heightWidget, height]]) { if (widget) { widget.value = value; widget.callback?.(value); } } node.setDirtyCanvas?.(true, true); };
  const resolveCanvas = settings => {
    if (settings.resolution === "custom" && settings.custom_mode === "fixed") return [snap16(settings.custom_width), snap16(settings.custom_height)];
    const aspect = selectedAspect(settings); if (!aspect) return null;
    if (settings.resolution === "auto") { const shortSide = 768; return aspect >= 1 ? [snap16(shortSide * aspect), shortSide] : [shortSide, snap16(shortSide / aspect)]; }
    if (settings.resolution === "custom") { const pixels = Math.max(.01, Number(settings.custom_mp) || 1) * 1024 * 1024; const h = Math.sqrt(pixels / aspect); return [snap16(h * aspect), snap16(h)]; }
    const pixels = Number(RESOLUTION_PRESETS[settings.resolution]) * 1024 * 1024; const h = Math.sqrt(pixels / aspect); return [snap16(h * aspect), snap16(h)];
  };
  const applyResolution = () => { if (hasExternalCanvas()) return; const canvas = resolveCanvas(resolutionState()); if (canvas) setCanvasWidgets(...canvas); };

  const fileInput = document.createElement("input"); fileInput.type = "file"; fileInput.multiple = true; fileInput.accept = "image/*,video/*,audio/*"; fileInput.hidden = true;
  fileInput.onchange = async () => { for (const file of fileInput.files || []) await acceptFile(file); fileInput.value = ""; };
  timeline.addEventListener("paste", async event => {
    if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return;
    const files = [...(event.clipboardData?.files || [])];
    if (!files.length) files.push(...[...(event.clipboardData?.items || [])].filter(item => item.kind === "file").map(item => item.getAsFile()).filter(Boolean));
    if (!files.length) return;
    event.preventDefault(); event.stopPropagation();
    const selected = state.items.find(item => item.id === selectedId);
    if (selected) {
      const replacementFile = files.find(file => { const type = mediaTypeFor(file); return type && allowsType(type) && laneForItem(selected) === (type === "audio" ? "audio" : type === "video" ? "video" : "image"); });
      if (replacementFile) {
        if (isLockedSlot(selected)) { setStatus(`${mediaReferenceName(selected.type)} ${selected.slot + 1} is locked in L2VA mode`, true); return; }
        await replaceSelectedFile(replacementFile, selected);
        return;
      }
    }
    if (mode() === "FL2VA" && selectedLane === "audio") { setStatus("FL2VA supports image references only; select the Image lane.", true); return; }
    setStatus(`Pasting ${files.length} file${files.length === 1 ? "" : "s"} into the ${selectedLane === "audio" ? "Audio" : selectedLane === "video" ? "Video" : "Image"} lane.`);
    for (const file of files) await acceptFile(file, selectedLane);
  });
  async function probeWavDuration(value) {
    try { return wavDurationFromBuffer(await (await fetch(viewUrl(value))).arrayBuffer()); }
    catch (error) { console.warn("[MiniMax H3 Director] WAV metadata probe failed", error); return null; }
  }
  async function probeDuration(value, type) {
    if (type === "image") return null;
    const media = document.createElement(type === "video" ? "video" : "audio");
    media.preload = "metadata";
    media.src = viewUrl(value);
    const duration = await new Promise(resolve => {
      const done = result => { media.remove(); resolve(Number.isFinite(result) ? result : null); };
      media.onloadedmetadata = () => done(media.duration);
      media.onerror = () => done(null);
    });
    return duration ?? (type === "audio" ? await probeWavDuration(value) : null);
  }
  async function probeDimensions(value, type) {
    if (type !== "image" && type !== "video") return null;
    const media = document.createElement(type === "video" ? "video" : "img");
    media.preload = "metadata";
    media.src = viewUrl(value);
    return await new Promise(resolve => {
      const done = () => { const width = type === "video" ? media.videoWidth : media.naturalWidth; const height = type === "video" ? media.videoHeight : media.naturalHeight; media.remove(); resolve(width > 0 && height > 0 ? { source_width: width, source_height: height } : null); };
      if (type === "video") media.onloadedmetadata = done; else media.onload = done;
      media.onerror = () => { media.remove(); resolve(null); };
    });
  }
  async function captureFirstFrame(videoUrl) {
    return await new Promise(resolve => {
      const vid = document.createElement("video");
      vid.preload = "auto"; vid.muted = true; vid.crossOrigin = "";
      const fail = () => { vid.remove(); resolve(null); };
      const draw = () => {
        try {
          const c = document.createElement("canvas");
          c.width = vid.videoWidth || 160; c.height = vid.videoHeight || 96;
          const ctx = c.getContext("2d");
          ctx.drawImage(vid, 0, 0, c.width, c.height);
          vid.remove();
          resolve(c.toDataURL("image/jpeg", 0.7));
        } catch { fail(); }
      };
      vid.onloadeddata = () => { vid.currentTime = 0; };
      vid.onseeked = draw;
      vid.onerror = fail;
      vid.src = videoUrl;
    });
  }
  async function extractWaveform(value, id) {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const response = await fetch(viewUrl(value));
      const buffer = await audioContext.decodeAudioData(await response.arrayBuffer());
      const channel = buffer.getChannelData(0); const peakCount = 240; const step = Math.max(1, Math.floor(channel.length / peakCount)); const peaks = [];
      for (let index = 0; index < peakCount; index += 1) { let peak = 0; for (let sample = index * step; sample < Math.min(channel.length, (index + 1) * step); sample += 1) peak = Math.max(peak, Math.abs(channel[sample])); peaks.push(peak); }
      audioContext.close?.();
      mutate(s => { const item = s.items.find(x => x.id === id); if (item) item.waveform_peaks = peaks; });
    } catch (error) { console.warn("[MiniMax H3 Director] Audio waveform decode failed", error); }
  }
  async function replaceSelectedFile(file, selected) {
    const type = mediaTypeFor(file);
    const lane = type === "audio" ? "audio" : type === "video" ? "video" : "image";
    if (!type || !allowsType(type) || laneForItem(selected) !== lane) { setStatus(`Paste a compatible ${laneForItem(selected) === "audio" ? "audio" : laneForItem(selected) === "video" ? "video" : "image"} file to replace the selected media.`, true); return; }
    if (count(state, type) - (selected.type === type ? 1 : 0) >= MAX[type]) { setStatus(`Limit reached: ${MAX[type]} ${type}s.`, true); return; }
    try {
      const value = await uploadFile(file, status);
      const [sourceDuration, dimensions] = await Promise.all([probeDuration(value, type), probeDimensions(value, type)]);
      if (sourceDuration !== null && sourceDuration < 2) { setStatus(`${file.name}: MiniMax references must be at least 2 seconds.`, true); return; }
      const duration = sourceDuration === null ? null : Math.min(sourceDuration, 15);
      const thumbnail = type === "video" ? await captureFirstFrame(viewUrl(value)) : null;
      const { thumbnail: oldThumbnail, source_width, source_height, duration: oldDuration, source_duration, trim_start, trim_end, waveform_peaks, ...stable } = selected;
      const item = { ...stable, type, value, thumbnail, ...dimensions, ...(duration !== null ? { duration, source_duration: sourceDuration } : {}), ...((type === "video" || type === "audio") ? { trim_start: 0, trim_end: duration } : {}) };
      mutate(s => { const index = s.items.findIndex(existing => existing.id === selected.id); if (index >= 0) s.items[index] = item; });
      setStatus(`Replaced selected ${mediaReferenceName(type)}.`);
      if (type === "audio") void extractWaveform(value, selected.id);
    } catch (error) { console.error(error); setStatus(`Upload failed: ${error.message || error}`, true); }
  }
  async function acceptFile(file, targetLane = null) { const type = mediaTypeFor(file); const validLane = targetLane === "image" ? type === "image" : targetLane === "video" ? type === "video" : targetLane === "audio" ? (type === "audio" || type === "video") : true; const modeSupportsType = !!type && allowsType(type); const lane = (type === "video" && targetLane === "audio") ? "audio" : type === "audio" ? "audio" : type === "video" ? "video" : "image"; if (!type || !validLane || !modeSupportsType) { const requirement = mode() === "FL2VA" ? "FL2VA supports image references only; video and audio are unavailable." : targetLane ? `Drop ${targetLane === "audio" ? "audio or video" : targetLane === "video" ? "video" : "image"} files on this lane.` : "This media type is not available in the selected MiniMax mode."; setStatus(requirement, true); return; } if (occupiedSlotsForLane(activeItems(), lane).size >= MAX[lane] || activeItems().length >= MAX.total) { setStatus(`Limit reached: ${MAX[lane]} ${lane}s / ${MAX.total} files.`, true); return; } const laneAvail = availableSlots(lane); const laneOccupied = occupiedSlotsForLane(activeItems(), lane); const laneFree = laneAvail.some(slot => !laneOccupied.has(slot)); if (!laneFree) { setStatus(`No free ${lane} slot is available.`, true); return; } try { const value = await uploadFile(file, status); const [sourceDuration, dimensions] = await Promise.all([probeDuration(value, type), probeDimensions(value, type)]); if (sourceDuration !== null && sourceDuration < 2) { setStatus(`${file.name}: MiniMax references must be at least 2 seconds.`, true); return; } const duration = sourceDuration === null ? null : Math.min(sourceDuration, 15); let thumbnail = null; if (type === "video") { thumbnail = await captureFirstFrame(viewUrl(value)); } const item = { type, value, thumbnail, ...dimensions, ...(duration !== null ? { duration, source_duration: sourceDuration } : {}), ...((type === "video" || type === "audio") ? { trim_start: 0, trim_end: duration } : {}), ...(type === "video" && targetLane === "audio" ? { media_mode: "audio" } : {}) }; addItem(item); applyResolution(); const added = state.items[state.items.length - 1]; if (type === "audio") void extractWaveform(value, added.id); setStatus(sourceDuration > 15 ? `${file.name} added; cropped to the first 15 seconds.` : `${file.name} added.`); } catch (error) { setStatus(error.message || "Upload failed", true); } }
  const render = () => {
    timeline.replaceChildren();
    const selected = state.items.find(item => item.id === selectedId);
    const modeGroup = document.createElement("div"); modeGroup.className = "ds-h3-mode-group"; modeGroup.style.display = "flex"; modeGroup.style.flexDirection = "column"; modeGroup.style.alignItems = "flex-start"; modeGroup.style.gap = "4px"; modeGroup.style.padding = "6px"; modeGroup.style.background = "#0d1217"; modeGroup.style.border = "1px solid #344452"; modeGroup.style.borderRadius = "6px"; modeGroup.style.flexShrink = "0"; modeGroup.style.boxSizing = "border-box"; modeGroup.style.width = "100%";
    const topRow = document.createElement("div"); topRow.className = "ds-h3-modebar"; topRow.style.flexWrap = "wrap"; topRow.style.gap = "8px"; topRow.style.padding = "0"; topRow.style.border = "0"; topRow.style.background = "transparent"; topRow.style.width = "100%"; topRow.style.maxWidth = "100%"; topRow.style.boxSizing = "border-box";
    const controlGroup = () => { const group = document.createElement("span"); group.className = "ds-h3-actions"; group.style.cssText = "gap:4px;flex-wrap:wrap;white-space:nowrap;max-width:100%"; return group; };
    const modesSide = controlGroup(); const modeLabel = document.createElement("span"); modeLabel.textContent = "Model Mode:"; modeLabel.style.cssText = "color:#9fb3c2;font-weight:600"; modesSide.append(modeLabel); ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA", "Image Inpaint"].forEach(value => { const button = document.createElement("button"); button.textContent = value; button.classList.toggle("active", mode() === value); button.title = value === "Image Inpaint" ? "One image reference; output exactly one frame through Get Image from Batch." : value; button.onclick = () => { if (modeWidget) { modeWidget.value = value; modeWidget.callback?.(value); } if ((selectedLane === "audio" || selectedLane === "video") && value !== "REF2VA") selectedLane = "image"; render(); }; modesSide.append(button); });
    const promptSide = controlGroup(); promptSide.style.cssText += ";padding-left:8px;border-left:1px solid #344452"; const promptLabel = document.createElement("span"); promptLabel.textContent = "Prompt Mode:"; promptLabel.style.cssText = "color:#9fb3c2;font-weight:600"; promptSide.append(promptLabel); const styleLabel = promptStyle(); [["simple", "Simple"], ["structured", "Structured"]].forEach(([value, label]) => { const promptButton = document.createElement("button"); promptButton.className = "ds-h3-prompt-mode-btn"; promptButton.textContent = label; promptButton.classList.toggle("active", styleLabel === value); promptButton.title = `Use the ${label.toLowerCase()} prompt editor`; promptButton.onclick = () => { if (styleLabel === value) return; if (value === "simple") builderState.simple_prompt = previewTextFor(mode(), false); else { const parsed = parseSimplePromptToStructured(builderState.simple_prompt || "", mode()); if (mode() === "REF2VA") { if (!builderState.ref) builderState.ref = {}; Object.assign(builderState.ref, parsed); } else { builderState.imd = parsed.integrated_multimodal_description || ""; builderState.soundscape = parsed.overall_soundscape || ""; builderState.music = parsed.non_diegetic_music || "N/A"; } } builderState.prompt_mode = value; emit(); render(); }; promptSide.append(promptButton); });
    const ioSide = controlGroup(); ioSide.style.cssText += ";padding-left:8px;border-left:1px solid #344452"; if (window.DaSiWaH3Forge && mode() !== "Image Inpaint") { const forgeButton = document.createElement("button"); forgeButton.className = "ds-h3-forge-btn"; forgeButton.textContent = "✨ Forge"; forgeButton.title = "Write this prompt with a local LLM (runs before the workflow, then unloads)"; forgeButton.onclick = () => window.DaSiWaH3Forge.open(node); ioSide.append(forgeButton); }
    const actionsSide = controlGroup(); actionsSide.style.cssText += ";padding-left:8px;border-left:1px solid #344452"; const hasContent = state.items.length || state.prompt_blocks?.length || hasBuilderContent() || String(promptWidget?.value || "").trim(); if (selected) { if (!isLockedSlot(selected)) { const removeButton = document.createElement("button"); removeButton.className = "ds-h3-remove-btn"; removeButton.textContent = "Remove"; removeButton.title = `Remove selected ${selected.type}`; removeButton.onclick = () => remove(selected.id); actionsSide.append(removeButton); } else { setStatus(`${mediaReferenceName(selected.type)} ${selected.slot + 1} is locked in L2VA mode`, true); } } if (hasContent) { const clearButton = document.createElement("button"); clearButton.className = "ds-h3-clear-btn"; clearButton.textContent = "Clear"; clearButton.title = "Remove all media and prompts"; clearButton.onclick = clearAll; actionsSide.append(clearButton); } else { const clearButton = document.createElement("button"); clearButton.className = "ds-h3-clear-btn ds-h3-clear-btn-empty"; clearButton.textContent = "Clear"; clearButton.title = "Nothing to clear yet"; clearButton.onclick = () => setStatus("Nothing to clear."); actionsSide.append(clearButton); }
    const spacer = document.createElement("span"); spacer.style.flex = "1";
    const docsButton = document.createElement("button"); docsButton.className = "ds-h3-docs"; docsButton.textContent = "?"; docsButton.title = "Open MiniMax H3 Director documentation on GitHub"; docsButton.onclick = () => window.open(REPOSITORY_URL, "_blank", "noopener,noreferrer");
    topRow.append(modesSide, promptSide, spacer, ioSide, actionsSide, docsButton); modeGroup.append(topRow);
    const modeHint = { T2VA: "T2VA · no input frame", I2VA: "I2VA · one opening-frame slot", FL2VA: "FL2VA · opening and closing-frame slots", L2VA: "L2VA · one closing-frame slot", "Image Inpaint": "Image Inpaint · exactly one image · outputs one frame through Get Image from Batch · no video/audio" }; const hint = document.createElement("div"); hint.className = "ds-h3-mode-hint"; hint.style.fontSize = "11px"; hint.style.color = "#9fb3c2"; hint.style.margin = "0"; hint.textContent = modeHint[mode()] || `REF2VA · up to ${MAX.image} image, ${MAX.video} video, and ${MAX.audio} audio slots · ${MAX.total} combined files maximum`; modeGroup.append(hint);
    timeline.append(modeGroup);

    const resolutionPanel = document.createElement("div"); resolutionPanel.className = "ds-h3-resolution-panel"; resolutionPanel.style.cssText = "width:100%;box-sizing:border-box;display:flex;flex-wrap:wrap;align-items:end;gap:6px;padding:7px;margin-top:6px;background:#0d1217;border:1px solid #344452;border-radius:6px;flex-shrink:0";
    const settings = resolutionState(); const externalCanvas = hasExternalCanvas(); const currentCanvas = externalCanvas ? ["external", "external"] : (resolveCanvas(settings) || [Number(widthWidget?.value) || 1344, Number(heightWidget?.value) || 768]);
    const updateResolution = patch => mutate(s => { s.resolution = { ...resolutionState(), ...patch }; });
    const closeAllMenus = () => resolutionPanel.querySelectorAll(".ds-h3-res-menu.open").forEach(menu => menu.classList.remove("open"));
    const aspectSwatch = id => { if (!String(id).includes(":")) return null; const [w, h] = String(id).split(":").map(Number); if (!w || !h) return null; const boxSize = 15; const box = document.createElement("span"); box.className = "ds-h3-res-swatch-box"; box.style.width = `${Math.round(w >= h ? boxSize : boxSize * w / h)}px`; box.style.height = `${Math.round(w >= h ? boxSize * h / w : boxSize)}px`; const wrap = document.createElement("span"); wrap.className = "ds-h3-res-swatch"; wrap.title = `Aspect ${id}`; wrap.append(box); return wrap; };
    const inputScalingHint = id => ({ Off: "Skip input scaling — references are sent to MiniMax at their original size.", Auto: "Scale references down only when their short edge exceeds 2048 px; smaller inputs pass through untouched.", Target: "Resize references to the selected Aspect & Resolution box (stretch, aspect not preserved).", Fit: "Scale references to fit inside the target box while preserving their aspect ratio.", "Fill and crop": "Scale references to cover the target box, preserving aspect, then crop the overflow.", "Fit and pad": "Scale references to fit inside the target box, preserving aspect, then pad the empty edges.", "Long side with divisible crop": "Scale so the long side matches the target, then crop to an exact, divisible target size." }[id] || "");
    const addDropdown = (label, value, options, onChange, withAspectSwatches = false, withHints = false, groups = null, target = resolutionPanel, actionMenu = false, fixedLabel = null, extraClass = "") => { const field = document.createElement("div"); field.className = extraClass ? `ds-h3-res-field ${extraClass}` : "ds-h3-res-field"; field.textContent = label; const control = document.createElement("span"); control.className = "ds-h3-res-control"; const select = document.createElement("select"); select.className = "ds-h3-res-select"; select.setAttribute("aria-hidden", "true"); options.forEach(([id, text]) => { const option = document.createElement("option"); option.value = id; option.textContent = text; select.append(option); }); select.value = value; select.oninput = event => { const picked = event.target.value; onChange(picked); if (actionMenu) select.value = value; syncButton(); syncMenu(); }; const button = document.createElement("button"); button.type = "button"; button.className = "ds-h3-res-btn"; button.setAttribute("aria-haspopup", "listbox"); button.setAttribute("aria-expanded", "false"); const labelSpan = document.createElement("span"); labelSpan.className = "ds-h3-res-label"; const caret = document.createElement("span"); caret.className = "ds-h3-res-caret"; caret.textContent = "▾"; button.append(labelSpan, caret); const menu = document.createElement("div"); const hasGroups = Array.isArray(groups) && groups.length > 0; menu.className = hasGroups ? "ds-h3-res-menu cols" : (options.length >= 6 ? "ds-h3-res-menu grid" : "ds-h3-res-menu"); if (menu.classList.contains("grid")) { const cols = Math.max(3, Math.min(6, Math.round(Math.sqrt(options.length)))); menu.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`; menu.style.width = `${cols * 148}px`; menu.style.maxHeight = "420px"; } if (hasGroups) menu.style.maxHeight = "420px"; menu.setAttribute("role", "listbox"); const syncButton = () => { if (fixedLabel !== null) { labelSpan.textContent = fixedLabel; return; } const current = options.find(([id]) => id === select.value); labelSpan.textContent = current ? current[1] : select.value; const swatch = withAspectSwatches ? aspectSwatch(select.value) : null; const existing = button.querySelector(".ds-h3-res-swatch"); if (existing) existing.remove(); if (swatch) button.insertBefore(swatch, labelSpan); button.title = withHints ? inputScalingHint(select.value) : ""; }; const syncMenu = () => menu.querySelectorAll(".ds-h3-res-item").forEach(item => item.classList.toggle("active", !actionMenu && item.dataset.id === select.value)); const makeItem = ([id, text]) => { const item = document.createElement("button"); item.type = "button"; item.className = "ds-h3-res-item"; item.setAttribute("role", "option"); item.dataset.id = id; if (withAspectSwatches) { const swatch = aspectSwatch(id); if (swatch) item.append(swatch); } const itemLabel = document.createElement("span"); itemLabel.className = "ds-h3-res-item-label"; itemLabel.textContent = text; item.append(itemLabel); if (withHints) item.title = inputScalingHint(id); item.onclick = () => { onChange(id); select.value = actionMenu ? value : id; syncButton(); syncMenu(); closeAllMenus(); button.setAttribute("aria-expanded", "false"); }; return item; }; if (hasGroups) { groups.forEach(group => { const col = document.createElement("div"); col.className = "ds-h3-res-col"; if (group.title) { const title = document.createElement("span"); title.className = "ds-h3-res-col-title"; title.textContent = group.title; col.append(title); } (group.items || []).forEach(pair => col.append(makeItem(pair))); menu.append(col); }); } else { options.forEach(pair => menu.append(makeItem(pair))); } button.onclick = () => { const willOpen = !menu.classList.contains("open"); closeAllMenus(); menu.classList.toggle("open", willOpen); button.setAttribute("aria-expanded", String(willOpen)); if (willOpen) { const margin = 8; const rect = control.getBoundingClientRect(); const menuRect = menu.getBoundingClientRect(); const spaceBelow = window.innerHeight - rect.bottom - margin; const spaceAbove = rect.top - margin; menu.dataset.place = menuRect.height > spaceBelow && spaceAbove > spaceBelow ? "up" : "down"; let left = rect.left; if (left + menuRect.width > window.innerWidth - margin) left = Math.max(margin, window.innerWidth - margin - menuRect.width); menu.style.left = `${left - rect.left}px`; } }; syncButton(); syncMenu(); control.append(select, button, menu); field.append(control); target.append(field); };
    const dataTypeLabels = [["files", "Reference Files"], ["prompt", "Prompts"], ["all", "All"]];
    addDropdown("", "load-picker", [], id => { const [loadMode, dataType] = id.split(":"); loadReferencePack(loadMode, dataType); }, false, false, [
      { title: "Append", items: dataTypeLabels.map(([id, text]) => [`append:${id}`, text]) },
      { title: "Overwrite", items: dataTypeLabels.map(([id, text]) => [`overwrite:${id}`, text]) },
    ], ioSide, true, "Load", "ds-h3-io-dropdown load");
    addDropdown("", "save-picker", dataTypeLabels, id => { void saveReferencePack(id); }, false, false, null, ioSide, true, "Save", "ds-h3-io-dropdown save");
    const disableWhenExternal = () => { if (externalCanvas) resolutionPanel.querySelectorAll("select,input,button").forEach(control => { control.disabled = true; }); };
    const ratioOf = id => { const [w, h] = String(id).split(":").map(Number); return w && h ? w / h : 0; };
    const aspectItems = { general: [], horizontal: [], square: [], vertical: [] };
    ASPECT_OPTIONS.forEach(pair => { const [id] = pair; if (id === "auto" || id === "custom" || !id.includes(":")) aspectItems.general.push(pair); else { const [w, h] = id.split(":").map(Number); aspectItems[w > h ? "horizontal" : w < h ? "vertical" : "square"].push(pair); } });
    aspectItems.horizontal.sort((a, b) => ratioOf(a[0]) - ratioOf(b[0]));
    aspectItems.vertical.sort((a, b) => ratioOf(a[0]) - ratioOf(b[0]));
    const resNames = Object.keys(RESOLUTION_PRESETS);
    const pVal = n => Number(String(n).replace(/K$/i, "000").replace(/p$/i, ""));
    const mpVal = n => Number((String(n).match(/^([\d.]+)\s*MP/) || [])[1] || 0);
    addDropdown("ASPECT", settings.aspect, ASPECT_OPTIONS, aspect => updateResolution({ aspect }), true, false, [
      { title: "General", items: aspectItems.general },
      { title: "Horizontal", items: aspectItems.horizontal },
      { title: "Square", items: aspectItems.square },
      { title: "Vertical", items: aspectItems.vertical },
    ]);
    addDropdown("RESOLUTION", settings.resolution, [["auto", "Native (ShortEdge 768px)"], ...resNames.map(name => [name, name]), ["custom", "CUSTOM"]], resolution => updateResolution({ resolution }), false, false, [
      { title: "General", items: [["auto", "Native (ShortEdge 768px)"], ["custom", "CUSTOM"]] },
      { title: "###p", items: resNames.filter(n => !/MP/.test(n)).sort((a, b) => pVal(a) - pVal(b)).map(n => [n, n]) },
      { title: "MP", items: resNames.filter(n => /MP/.test(n)).sort((a, b) => mpVal(a) - mpVal(b)).map(n => [n, n]) },
    ]);
    addDropdown("INPUT SCALING", settings.input_scaling, [["Off", "Off"], ["Auto", "Native (ShortEdge 2048px)"], ["Target", "Target · Selected Aspect & Resolution"], ["Fit", "Fit"], ["Fill and crop", "Fill and crop"], ["Fit and pad", "Fit and pad"], ["Long side with divisible crop", "Long side with divisible crop"]], input_scaling => updateResolution({ input_scaling }), false, true);
    if (mode() === "REF2VA") {
      const refModField = document.createElement("div"); refModField.className = "ds-h3-res-field"; refModField.textContent = "SAVED REFERENCES";
      const refModButton = document.createElement("button"); refModButton.type = "button"; refModButton.className = "ds-h3-res-btn ds-h3-refmod-button"; refModButton.title = "Open saved RefMod reference settings"; refModButton.onclick = () => openRefModOverlay(refModButton);
      const refModLabel = document.createElement("span"); refModLabel.textContent = "REFMOD";
      refModActiveBadge = document.createElement("span"); refModActiveBadge.className = "ds-h3-refmod-badge-inline"; refModButton.append(refModLabel, refModActiveBadge); updateRefModActiveBadge();
      refModField.append(refModButton); resolutionPanel.append(refModField);
    }
    if (settings.aspect === "custom") { for (const [name, key] of [["W", "custom_aspect_w"], ["H", "custom_aspect_h"]]) { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#9fb3c2;width:56px"; field.textContent = `RATIO ${name}`; const input = document.createElement("input"); input.type = "number"; input.min = "1"; input.step = "1"; input.className = "ds-h3-res-num"; input.value = settings[key]; input.onchange = event => updateResolution({ [key]: Math.max(1, Number(event.target.value) || 1) }); field.append(input); resolutionPanel.append(field); } }
    if (settings.resolution === "custom") { addDropdown("CUSTOM", settings.custom_mode || "mp", [["mp", "Megapixels"], ["fixed", "Fixed pixels"]], custom_mode => updateResolution({ custom_mode })); const customKey = (settings.custom_mode || "mp") === "mp" ? "custom_mp" : "custom_width"; if (customKey === "custom_mp") { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#9fb3c2;width:76px"; field.textContent = "MP"; const input = document.createElement("input"); input.type = "number"; input.min = ".01"; input.step = ".01"; input.className = "ds-h3-res-num"; input.value = settings.custom_mp; input.onchange = event => updateResolution({ custom_mp: Number(event.target.value) || settings.custom_mp }); field.append(input); resolutionPanel.append(field); } else { const dimensions = document.createElement("div"); dimensions.className = "ds-h3-fixed-dimensions"; dimensions.style.cssText = "display:flex;gap:6px;align-items:end"; const dimensionField = (label, value, onChange) => { const field = document.createElement("label"); field.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#9fb3c2;width:76px"; field.textContent = label; const input = document.createElement("input"); input.type = "number"; input.min = "16"; input.step = "16"; input.className = "ds-h3-res-num"; input.value = value; input.onchange = event => onChange(Number(event.target.value) || value); field.append(input); return field; }; const widthField = dimensionField("WIDTH", settings.custom_width, custom_width => updateResolution({ custom_width })); const heightField = dimensionField("HEIGHT", settings.custom_height, custom_height => updateResolution({ custom_height })); dimensions.append(widthField, heightField); resolutionPanel.append(dimensions); } }
    disableWhenExternal();
    const readout = document.createElement("span"); readout.style.cssText = "margin-left:auto;font-size:11px;color:#7ee19d;white-space:nowrap"; readout.textContent = externalCanvas ? "External width/height overwrite · Director sizing and input scaling disabled" : `${settings.aspect === "auto" ? "Auto 768px" : settings.aspect} · ${settings.resolution === "auto" ? "Auto 768px" : settings.resolution} · ${currentCanvas[0]} × ${currentCanvas[1]} · 32px H3 grid`; resolutionPanel.append(readout); timeline.append(resolutionPanel);

    const lengthWidget = node.widgets?.find(w => w.name === "duration"); const timelineSeconds = Math.max(1, Number(lengthWidget?.value) || 5); lastTimelineLength = timelineSeconds;

    ensureLayout();
    const track = document.createElement("div"); track.className = "ds-h3-track"; track.addEventListener("pointerdown", () => timeline.focus());
    const trackInner = document.createElement("div"); trackInner.className = "ds-h3-track-inner";
    trackInner.onclick = event => { if (event.target !== trackInner) return; const rect = trackInner.getBoundingClientRect(); insertAt = Math.max(0, Math.round(((event.clientX - rect.left) / scale) * 4) / 4); status.textContent = `Insert cursor: ${insertAt.toFixed(2)}s · choose + image, + video, + audio, or + text.`; trackInner.style.setProperty("--insert-x", `${insertAt * scale}px`); };
    const laneTypes = mode() === "T2VA" ? [] : !isReferenceMode() ? ["Image"] : ["Image", "Video", "audio"]; const laneHeight = 120; const trackEntries = (() => { const base = displayedItems(); const echoes = base.filter(hasAudioEcho).filter(item => Number.isInteger(item.audioSlot)).map(item => ({ ...item, _audioEcho: true, slot: item.audioSlot })); const refmods = isReferenceMode() ? refmodTimelineItems() : []; return [...base, ...echoes, ...refmods]; })(); const scale = 48; const audioSlotWidth = sourceDuration => Math.round(Math.max(180, Math.min(420, 96 * Math.log2(Math.max(2, sourceDuration) + 1)))); const slotWidthFor = item => item && (item.type === "audio" || item.type === "video") ? audioSlotWidth(Number(item.source_duration) || item.duration) : 112; const laneNameFor = item => laneForItem(item) === "audio" ? "audio" : laneForItem(item) === "video" ? "Video" : "Image"; const slotItem = (lane, slot) => trackEntries.find(item => laneNameFor(item) === lane && item.slot === slot); const slotLeft = (lane, slot) => { let left = 6; for (let index = 0; index < slot; index += 1) left += slotWidthFor(slotItem(lane, index)) + 8; return left; }; const acceptLaneDrop = async (event, targetLane) => { event.preventDefault(); event.stopPropagation(); trackInner.classList.remove("over"); const rect = trackInner.getBoundingClientRect(); insertAt = Math.max(0, Math.round(((event.clientX - rect.left) / scale) * 4) / 4); trackInner.style.setProperty("--insert-x", `${insertAt * scale}px`); for (const file of event.dataTransfer.files || []) await acceptFile(file, targetLane); }; const imageSlotCount = () => mode() === "T2VA" ? 0 : mode() === "I2VA" || mode() === "Image Inpaint" ? 1 : mode() === "L2VA" ? 2 : mode() === "FL2VA" ? 2 : MAX.image; const videoSlotCount = () => isReferenceMode() ? MAX.video : 0; const slotExtent = lane => Array.from({ length: lane === "audio" ? MAX.audio : lane === "Video" ? videoSlotCount() : imageSlotCount() }, (_, index) => slotWidthFor(slotItem(lane, index)) + 8).reduce((sum, width) => sum + width, 6); trackInner.style.width = `${Math.max(240, slotExtent("Image"), slotExtent("Video"), slotExtent("audio"))}px`; trackInner.style.height = `${laneTypes.length * laneHeight}px`; track.style.display = laneTypes.length ? "" : "none"; trackInner.style.background = `repeating-linear-gradient(0deg, transparent 0, transparent ${laneHeight - 1}px, #344452 ${laneHeight - 1}px, #344452 ${laneHeight}px), repeating-linear-gradient(90deg,#111a21 0,#111a21 49px,#1b2933 50px)`; const ruler = document.createElement("div"); ruler.className = "ds-h3-ruler"; for (let second = 0; second <= timelineSeconds; second++) { if (second % 2 !== 0) continue; const mark = document.createElement("div"); mark.style.position = "absolute"; mark.style.left = `${second * scale}px`; mark.style.bottom = "0"; mark.style.fontSize = "9px"; mark.style.color = "#8fa3b2"; mark.textContent = `${second}s`; ruler.appendChild(mark); } track.append(ruler);
    const lanes = new Map(); laneTypes.forEach((type, index) => { const lane = document.createElement("div"); const targetLane = type === "audio" ? "audio" : type === "Video" ? "video" : "image"; const m = mode(); const audioDisabledByMode = m === "T2VA" || m === "I2VA" || m === "FL2VA" || m === "L2VA" || m === "Image Inpaint"; const videoDisabledByMode = m !== "REF2VA"; const supported = !(targetLane === "audio" && audioDisabledByMode) && !(targetLane === "video" && videoDisabledByMode); const slotCount = targetLane === "audio" ? MAX.audio : targetLane === "video" ? videoSlotCount() : imageSlotCount(); lane.className = `ds-h3-timeline-lane ${targetLane}${selectedLane === targetLane ? " selected" : ""}${supported ? "" : " disabled"}`; lane.style.top = `${index * laneHeight}px`; lane.onclick = () => { if (!supported) { setStatus(m === "T2VA" ? "T2VA has no input lanes." : targetLane === "video" ? "This mode does not support video references." : audioDisabledByMode ? "This mode does not support audio references." : "", true); return; } selectedLane = targetLane; setStatus(`${type === "audio" ? "Audio" : type} lane selected. Paste files here with Ctrl+V.`); render(); }; lane.ondragover = event => { if ([...event.dataTransfer.items].some(item => item.kind === "file")) { event.preventDefault(); if (supported) trackInner.classList.add("over"); } }; lane.ondragleave = () => trackInner.classList.remove("over"); lane.ondrop = event => { if (!supported) { event.preventDefault(); event.stopPropagation(); setStatus(targetLane === "video" ? "This mode does not support video references." : audioDisabledByMode ? "This mode does not support audio references." : "", true); return; } selectedLane = targetLane; acceptLaneDrop(event, targetLane); }; const label = document.createElement("span"); label.className = "ds-h3-lane-label"; label.textContent = `${type}${selectedLane === targetLane ? " · selected" : ""}`; label.style.display = hiddenLanes.has(type) ? "none" : "block"; lane.append(label); for (let slotIndex = 0; slotIndex < slotCount; slotIndex += 1) { if (slotItem(type, slotIndex)) continue; const isL2VADeadSlot = m === "L2VA" && targetLane === "image" && slotIndex === 0; const slot = document.createElement("span"); slot.className = `ds-h3-empty-slot${isL2VADeadSlot ? " ds-h3-empty-slot-dead" : ""}`; slot.textContent = isL2VADeadSlot ? "Not Used" : "+"; slot.style.left = `${slotLeft(type, slotIndex)}px`; slot.style.width = `${slotWidthFor(null)}px`; if (isL2VADeadSlot) { const lockIcon = document.createElement("span"); lockIcon.className = "ds-h3-lock-icon"; lockIcon.textContent = "🔒"; slot.style.position = "relative"; slot.appendChild(lockIcon); } if (!m.includes("T2VA") && supported && !isL2VADeadSlot) { slot.onclick = event => { event.stopPropagation(); selectedLane = targetLane; const accepts = targetLane === "audio" ? "audio/*,video/*" : targetLane === "video" ? "video/*" : "image/*"; fileInput.accept = accepts; fileInput.click(); }; } else if (isL2VADeadSlot) { slot.onclick = event => { event.stopPropagation(); setStatus("This slot isn't used in L2VA — new images always go to the working slot.", true); }; } lane.append(slot); } lanes.set(type, lane); trackInner.append(lane); });
    trackEntries.forEach(item => { const clip = document.createElement("div"); clip.className = `ds-h3-clip ${item.type} ${selectedId === item.id ? "selected" : ""}${isLockedSlot(item) ? " locked" : ""}${item._audioEcho ? " audio-echo" : ""}${item._isRefMod ? " refmod" : ""}`; clip.textContent = mediaLabel(item); clip.title = item.type === "text" ? String(item.value || "") : String(item.value || ""); clip.style.display = "block"; if (item._isRefMod) { const badge = document.createElement("span"); badge.className = "ds-h3-refmod-badge"; badge.textContent = `REFMOD ${item.refmodSlot}`; clip.append(badge); const strengthLabel = document.createElement("span"); strengthLabel.className = "ds-h3-refmod-strength"; strengthLabel.textContent = `strength ${(Number(item.strength) * 100).toFixed(0)}%`; clip.append(strengthLabel); } const bgSrc = item.type === "image" ? viewUrl(item.value) : item.thumbnail || null; if (bgSrc) { clip.style.backgroundImage = `linear-gradient(90deg, rgba(20,35,45,.78), rgba(20,35,45,.5)), url("${bgSrc}")`; clip.style.backgroundSize = "cover"; clip.style.backgroundPosition = "center"; } item.slot = Number.isInteger(item.slot) ? item.slot : trackEntries.filter(x => laneNameFor(x) === laneNameFor(item)).indexOf(item); item.start = item.slot; item.duration = Number.isFinite(item.duration) ? item.duration : 1; const visualStart = item.start; const sourceDuration = Number(item.source_duration) || item.duration; const clipWidth = slotWidthFor(item); clip.style.left = `${slotLeft(laneNameFor(item), item.slot)}px`; clip.style.top = "7px"; clip.style.width = `${clipWidth}px`; clip.style.setProperty("--clip-width", `${clipWidth}px`); clip.dataset.slot = String(item.slot); const identity = document.createElement("span"); identity.className = "ds-h3-clip-identity"; const displayLane = laneNameFor(item); const isL2VAUnusedSlot = mode() === "L2VA" && displayLane === "Image" && item.slot === 0; const displayTypeName = laneNameFor(item) === "audio" ? "Audio" : mediaReferenceName(item.type); const typePosition = trackEntries.filter(entry => laneNameFor(entry) === displayLane && !(mode() === "L2VA" && displayLane === "Image" && entry.slot === 0)).sort((a, b) => a.slot - b.slot).indexOf(item) + 1; const isChainLinked = hasAudioEcho(item) || item._audioEcho; identity.textContent = isL2VAUnusedSlot ? "Not Used" : `${isChainLinked ? "🔗 " : ""}${displayTypeName} ${typePosition}`; if (isChainLinked) identity.title = item._audioEcho ? "This is the embedded audio from a Video+Audio reference — change or remove it from that video's V/A/V+A controls" : "This video's audio is also linked in the Audio lane"; clip.append(identity); if (isLockedSlot(item)) { const lockIcon = document.createElement("span"); lockIcon.className = "ds-h3-lock-icon"; lockIcon.textContent = "🔒"; clip.append(lockIcon); } if (item.type === "video" && !item._audioEcho) { const streamControls = document.createElement("span"); streamControls.className = "ds-h3-video-stream-controls"; const currentMediaMode = ["video", "audio", "video_audio"].includes(item.media_mode) ? item.media_mode : "video"; [["video", "V", "Video only"], ["audio", "A", "Audio only"], ["video_audio", "V+A", "Video + embedded audio"]].forEach(([value, label, title]) => { const button = document.createElement("button"); button.textContent = label; button.title = title; button.classList.toggle("active", value === currentMediaMode); button.onpointerdown = event => event.stopPropagation(); button.onclick = event => { event.stopPropagation(); let failed = false; mutate(s => { const target = s.items.find(x => x.id === item.id); if (!target) return; if (!retargetVideoSlots(s.items, target, value)) failed = true; }); if (failed) setStatus("No free slot is available for that mode — free up a slot first.", true); }; streamControls.append(button); }); clip.append(streamControls); const videoScale = document.createElement("span"); videoScale.className = "ds-h3-video-scale"; videoScale.title = "Full source-duration scale"; clip.append(videoScale); } if (selectedId === item.id && !item._audioEcho) { const close = document.createElement("button"); close.className = "ds-h3-clip-close"; close.textContent = "×"; close.title = item._isRefMod ? `Disable RefMod ${item.refmodSlot}` : `Remove selected ${item.type}`; close.onclick = event => { event.stopPropagation(); if (item._isRefMod) { mutate(s => { const row = (s.refmods || []).find(r => r.slot === item.refmodSlot); if (row) row.enabled = false; }); updateRefModActiveBadge(); render(); } else { remove(item.id); } }; clip.append(close); }
      const cropReadout = document.createElement("span"); cropReadout.className = "ds-h3-crop-readout"; if (item.type === "video" || item.type === "audio") { cropReadout.textContent = `crop ${Number(item.trim_start || 0).toFixed(2)}s–${item.trim_end == null ? "end" : Number(item.trim_end).toFixed(2) + "s"}`; clip.append(cropReadout); } if (item.type === "video") { for (const edge of ["start", "end"]) { const marker = document.createElement("span"); marker.className = `ds-h3-crop-marker ${edge}`; marker.style.left = `${(Number(edge === "start" ? item.trim_start || 0 : item.trim_end ?? sourceDuration) / sourceDuration) * 100}%`; clip.append(marker); } } if (item.type === "audio") { const peaks = Array.isArray(item.waveform_peaks) ? item.waveform_peaks : []; const waveform = document.createElement("canvas"); waveform.className = "ds-h3-waveform"; waveform.width = 720; waveform.height = 160; waveform.title = peaks.length ? "Full audio waveform; crop markers show the selected reference window" : "Decoding audio waveform…"; const context = waveform.getContext("2d"); if (context && peaks.length) { const center = waveform.height / 2; context.fillStyle = "rgba(126, 225, 157, .8)"; for (let x = 0; x < waveform.width; x += 1) { const peakIndex = Math.min(peaks.length - 1, Math.floor((x / waveform.width) * peaks.length)); const amplitude = (Math.log1p(9 * peaks[peakIndex]) / Math.log(10)) * (waveform.height - 16) * .45; context.fillRect(x, center - amplitude, 1, amplitude * 2); } } const cropStartMarker = document.createElement("span"); cropStartMarker.className = "ds-h3-audio-crop-marker start"; const cropEndMarker = document.createElement("span"); cropEndMarker.className = "ds-h3-audio-crop-marker end"; const updateAudioCropMarkers = () => { const startPercent = Math.max(0, Math.min(100, (Number(item.trim_start || 0) / sourceDuration) * 100)); const endPercent = Math.max(startPercent, Math.min(100, (Number(item.trim_end ?? sourceDuration) / sourceDuration) * 100)); cropStartMarker.style.left = `${startPercent}%`; cropEndMarker.style.left = `${endPercent}%`; }; updateAudioCropMarkers(); clip.append(waveform, cropStartMarker, cropEndMarker); }
      const resize = (edge, event) => { event.stopPropagation(); clip.setPointerCapture?.(event.pointerId); const origin = event.clientX; const start = item.start; const duration = item.duration; const sourceDuration = Number(item.source_duration) || duration; const minimumReferenceDuration = Math.min(2, sourceDuration); const cropStartAtDrag = Number(item.trim_start) || 0; const cropEndAtDrag = Number(item.trim_end) || sourceDuration; const onMove = moveEvent => { const delta = (item.type === "audio" || item.type === "video") ? ((moveEvent.clientX - origin) / clipWidth) * sourceDuration : (moveEvent.clientX - origin) / scale; if (edge === "left") { if (item.type === "video" || item.type === "audio") { item.trim_start = Math.min(cropEndAtDrag - minimumReferenceDuration, Math.max(0, Math.round((cropStartAtDrag + delta) * 4) / 4)); item.duration = Math.min(15, Math.max(minimumReferenceDuration, Math.round((cropEndAtDrag - item.trim_start) * 4) / 4)); item.trim_end = cropEndAtDrag; } else { const nextStart = Math.max(0, Math.round((start + delta) * 4) / 4); const end = start + duration; item.start = Math.min(nextStart, end - 0.25); item.duration = Math.max(0.25, Math.round((end - item.start) * 4) / 4); item.trim_start = item.start; } } else if (item.type === "video" || item.type === "audio") { item.trim_end = Math.min(sourceDuration, Math.max(cropStartAtDrag + minimumReferenceDuration, Math.round((cropEndAtDrag + delta) * 4) / 4)); item.duration = Math.min(15, Math.max(minimumReferenceDuration, Math.round((item.trim_end - cropStartAtDrag) * 4) / 4)); item.trim_end = cropStartAtDrag + item.duration; } else { item.duration = Math.max(0.25, Math.round((duration + delta) * 4) / 4); item.trim_end = item.start + item.duration; } if (item.type === "video" || item.type === "audio") { cropReadout.textContent = `crop ${Number(item.trim_start || 0).toFixed(2)}s–${Number(item.trim_end).toFixed(2)}s / ${sourceDuration.toFixed(2)}s`; setStatus(`${edge === "left" ? "Crop start" : "Crop end"}: ${cropReadout.textContent}`); } if (item.type === "video" || item.type === "audio") { const startPercent = Math.max(0, Math.min(100, (Number(item.trim_start || 0) / sourceDuration) * 100)); const endPercent = Math.max(startPercent, Math.min(100, (Number(item.trim_end ?? sourceDuration) / sourceDuration) * 100)); const [cropStartMarker, cropEndMarker] = clip.querySelectorAll(".ds-h3-crop-marker, .ds-h3-audio-crop-marker"); if (cropStartMarker) cropStartMarker.style.left = `${startPercent}%`; if (cropEndMarker) cropEndMarker.style.left = `${endPercent}%`; } clip.style.left = `${slotLeft(laneNameFor(item), item.slot)}px`; clip.style.width = `${clipWidth}px`; }; const onUp = () => { clip.removeEventListener("pointermove", onMove); clip.removeEventListener("pointerup", onUp); if (item._block) { item._block.start = item.start; item._block.duration = item.duration; } mutate(() => {}); }; clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); };
      const leftGrip = document.createElement("span"); leftGrip.className = "ds-h3-grip left"; leftGrip.onpointerdown = event => { if (isLockedSlot(item)) return; resize("left", event); }; const rightGrip = document.createElement("span"); rightGrip.className = "ds-h3-grip right"; rightGrip.onpointerdown = event => { if (isLockedSlot(item)) return; resize("right", event); }; clip.append(leftGrip, rightGrip); if (item.type === "video" || item.type === "audio") { leftGrip.style.display = "none"; rightGrip.style.display = "none"; clip.querySelectorAll(".ds-h3-crop-marker, .ds-h3-audio-crop-marker").forEach(marker => { marker.onpointerdown = event => { if (isLockedSlot(item)) return; resize(marker.classList.contains("start") ? "left" : "right", event); }; }); }
      if (!isLockedSlot(item) && !item._isRefMod) { const editBtn = document.createElement("button"); editBtn.textContent = "☰"; editBtn.className = "ds-h3-edit-btn"; editBtn.title = "Open preview and details"; editBtn.onclick = event => { event.stopPropagation(); openPreview(item); }; clip.append(editBtn); }
      clip.onclick = event => { if (event.target !== clip || item._isRefMod) return; selectedId = item.id; render(); }; clip.onpointerdown = event => { selectedId = item.id; if (event.target !== clip) return; if (isLockedSlot(item) || item._audioEcho || item._isRefMod) return; event.stopPropagation(); clip.setPointerCapture?.(event.pointerId); const origin = event.clientX; const originalLeft = slotLeft(laneNameFor(item), item.slot); const lane = laneNameFor(item); const slotCount = lane === "audio" ? MAX.audio : lane === "Video" ? videoSlotCount() : imageSlotCount(); let dragged = false; const onMove = moveEvent => { dragged ||= Math.abs(moveEvent.clientX - origin) >= 4; if (dragged) clip.style.left = `${originalLeft + moveEvent.clientX - origin}px`; }; const onUp = moveEvent => { clip.removeEventListener("pointermove", onMove); clip.removeEventListener("pointerup", onUp); if (!dragged) return; const rect = trackInner.getBoundingClientRect(); const x = moveEvent.clientX - rect.left; const slotOptions = (mode() === "L2VA" && lane === "Image") ? [1] : Array.from({ length: slotCount }, (_, slot) => slot); const targetSlot = slotOptions.reduce((nearest, slot) => Math.abs((slotLeft(lane, slot) + slotWidthFor(slotItem(lane, slot)) / 2) - x) < Math.abs((slotLeft(lane, nearest) + slotWidthFor(slotItem(lane, nearest)) / 2) - x) ? slot : nearest, slotOptions[0]); mutate(s => { const moved = s.items.find(x => x.id === item.id); if (!moved) return; const occupant = s.items.find(x => x.id !== moved.id && laneForItem(x) === laneForItem(moved) && x.slot === targetSlot); if (occupant && isLockedSlot(occupant)) return; const previousSlot = moved.slot; moved.slot = targetSlot; moved.start = targetSlot; if (occupant) { occupant.slot = previousSlot; occupant.start = previousSlot; } }); }; clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); }; lanes.get(laneNameFor(item)).append(clip); }); track.append(trackInner); timeline.append(track);
    // Unified prompt-builder form replacing legacy per-item/global prompts
    const promptPanel = document.createElement("div"); promptPanel.className = "ds-h3-prompt-panel";
    if (promptStyle() === "simple") {
      buildSimpleForm(promptPanel);
    } else if (mode() === "REF2VA") {
      buildRefForm(promptPanel);
    } else {
      buildBaseForm(promptPanel);
    }
    if (hasExternalPrompt()) {
      promptPanel.classList.add("disabled");
      promptPanel.querySelectorAll("textarea, input, button").forEach(el => { el.disabled = true; });
      const note = document.createElement("div"); note.className = "ds-h3-small ds-h3-ext-note";
      note.textContent = "External prompt detected — builder fields are disabled.";
      promptPanel.prepend(note);
    }
    promptPanel.addEventListener("focusin", event => { if (event.target.tagName === "TEXTAREA") refModPromptField = event.target; });
    timeline.append(promptPanel);
    // Prompt text is edited on the corresponding media row and synchronized to prompt_blocks.
    timeline.append(status); if (domWidget) domWidget.computeSize = () => [Math.max(420, node.size?.[0] || 520), uiHeight()];
    syncNodeBounds?.("grow");
  };
  let domWidget;
  // Keep DOM-widget sizing independent from node.size. Using node.size as the
  // widget's preferred height creates a feedback loop: LiteGraph asks the DOM
  // widget how tall it wants to be, the widget reports the current node height,
  // and that result can become the next node height.
  const NODE_CHROME_HEIGHT = 30;
  const minimumNodeSize = [440, 520];
  const minimumUiHeight = minimumNodeSize[1] - NODE_CHROME_HEIGHT;
  let syncingNodeBounds = false;

  // Measure the content while the root has no height tied to node.size. This is
  // the stable value used for automatic grow/fit operations.
  const measureContentHeight = () => {
    const measured = Math.ceil(Number(timeline.scrollHeight) || 0);
    trackedContentHeight = Math.max(minimumUiHeight, measured);
    return trackedContentHeight;
  };

  // IMPORTANT: this is content height, not node.size[1]. It must remain stable
  // while LiteGraph is manually resizing the node.
  const uiHeight = () => Math.max(minimumUiHeight, trackedContentHeight);

  // "grow" = content changed and the node needs more room.
  // "fit"  = a prompt field was resized and the node may shrink/grow to fit.
  // "manual" = LiteGraph owns node.size; never write another size during drag.
  const syncNodeBounds = (strategy = "grow") => {
    if (syncingNodeBounds) return;

    const contentHeight = measureContentHeight();
    const [currentWidth = minimumNodeSize[0], currentHeight = minimumNodeSize[1]] = node.size || [];
    const width = Math.max(minimumNodeSize[0], currentWidth);
    const requiredHeight = contentHeight + NODE_CHROME_HEIGHT;

    let height = currentHeight;
    if (strategy === "fit") {
      height = Math.max(minimumNodeSize[1], requiredHeight);
    } else if (strategy === "grow" && currentHeight < requiredHeight) {
      height = Math.max(minimumNodeSize[1], requiredHeight);
    }

    if (width !== currentWidth || height !== currentHeight) {
      syncingNodeBounds = true;
      node.setSize?.([width, height]);
      syncingNodeBounds = false;
      node.graph?.setDirtyCanvas(true, true);
    }

    // Width follows the node. Do NOT set timeline.height from node.size; the
    // DOM widget itself is positioned/sized by ComfyUI's addDOMWidget layer.
    const actualWidth = Number(node.size?.[0]) || width;
    timeline.style.width = `${Math.max(1, actualWidth - 20)}px`;
    timeline.style.height = "auto";
  };

  if (node.addDOMWidget) {
    domWidget = node.addDOMWidget("minimax_h3_director_ui", "custom", timeline, {
      serialize: false,
      hideOnZoom: false,
      // Stable floor; do not derive this from the current node height.
      getMinHeight: () => minimumUiHeight,
      // Preferred height comes from measured content, not node.size.
      getHeight: () => uiHeight(),
    });
  }

  const oldResize = node.onResize;
  node.onResize = function (...args) {
    oldResize?.apply(this, args);
    // LiteGraph owns node.size during a corner resize. Only update the DOM
    // width here; never call setSize again from inside the resize callback.
    requestAnimationFrame(() => {
      const width = Number(node.size?.[0]) || minimumNodeSize[0];
      timeline.style.width = `${Math.max(1, width - 20)}px`;
      timeline.style.height = "auto";
      node.graph?.setDirtyCanvas(true, true);
    });
  };
  const restorePersistedState = () => {
    state = parseState(dataWidget.value);
    for (const key of Object.keys(fieldHeights)) delete fieldHeights[key];
    for (const [key, value] of Object.entries(state.field_heights || {})) {
      const height = Number(value);
      if (Number.isFinite(height) && height >= 60) fieldHeights[key] = height;
    }
    selectedId = null;
    const m = mode();
    const baseDefaults = DEFAULT_BUILDER_STATE(m);
    const rawBuilder = builderWidget?.value || state.builder_state;
    if (rawBuilder) {
      try {
        const parsed = typeof rawBuilder === "string" ? JSON.parse(rawBuilder) : rawBuilder;
        if (parsed && typeof parsed === "object") {
          builderState = { ...baseDefaults, ...parsed, ref: { ...baseDefaults.ref, ...(parsed.ref || {}) } };
          builderState.mode = m;
        }
      } catch { /* fall back to defaults */ }
    } else {
      builderState = baseDefaults;
      builderState.mode = m;
    }
    if (!builderHasContent(builderState)) {
      const preserved = legacyPrompt(state, promptWidget?.value);
      if (preserved) {
        builderState.prompt_mode = "simple";
        builderState.simple_prompt = preserved;
      }
    }
    emit();
    syncNodeBounds();
    render();
    void Promise.all(state.items.filter(item => item.type === "video" && !item.thumbnail).map(async item => {
      const thumb = await captureFirstFrame(viewUrl(item.value));
      if (thumb) mutate(s => { const x = s.items.find(i => i.id === item.id); if (x) x.thumbnail = thumb; });
    })).then(render);
  };
  const oldConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    oldConfigure?.apply(this, args);
    requestAnimationFrame(restorePersistedState);
  };
  const oldConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    oldConnectionsChange?.apply(this, args);
    const linked = hasExternalPrompt();
    if (linked !== lastExternalPromptLinked) {
      lastExternalPromptLinked = linked;
      requestAnimationFrame(render);
    }
    const externalCanvasLinked = hasExternalCanvas();
    if (externalCanvasLinked !== lastExternalCanvasLinked) {
      lastExternalCanvasLinked = externalCanvasLinked;
      requestAnimationFrame(render);
    }
  };
  node.__dasiwaH3RestorePersistedState = () => requestAnimationFrame(restorePersistedState);
  node.__dasiwaH3State = () => state; node.__dasiwaH3Render = render;
  // H3 Forge (js/minimax_h3_forge.js) reads the timeline from here and writes
  // its result back through apply(), so the builder fields, the Simple box and
  // the hidden widgets all update the same way a typed edit does.
  node.__dasiwaH3Forge = {
    mode, promptStyle, setStatus,
    duration: () => Number(node.widgets?.find(w => w.name === "duration")?.value) || null,
    items: () => activeItems().filter(item => !item._audioEcho && !isLockedSlot(item)).map(item => ({ ...item, lane: laneForItem(item) })),
    apply: (result) => {
      if (result.mode === "REF2VA") builderState.ref = { ...(builderState.ref || {}), ...result.fields.ref };
      else Object.assign(builderState, result.fields);
      if (promptStyle() === "simple") builderState.simple_prompt = result.simple_prompt;
      emit(); render();
    },
  };
  if (modeWidget) { const old = modeWidget.callback; modeWidget.callback = value => { old?.(value); render(); }; }
  const extWidget = externalPromptWidget();
  if (extWidget) { const old = extWidget.callback; extWidget.callback = value => { old?.(value); render(); }; }
  const lengthWidget = node.widgets?.find(w => w.name === "duration");
  if (lengthWidget) { const old = lengthWidget.callback; lengthWidget.callback = value => { old?.(value); render(); }; }
  const oldDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function (...args) { oldDrawForeground?.apply(this, args); const seconds = Math.max(1, Number(lengthWidget?.value) || 5); if (seconds !== lastTimelineLength) render(); };
  node.__dasiwaH3LengthPoll = window.setInterval(() => { const seconds = Math.max(1, Number(lengthWidget?.value) || 5); if (seconds !== lastTimelineLength) render(); }, 200);
  node.__dasiwaH3ExtPoll = window.setInterval(() => {
    const linked = hasExternalPrompt();
    if (linked !== lastExternalPromptLinked) {
      lastExternalPromptLinked = linked;
      render();
    }
    const externalCanvasLinked = hasExternalCanvas();
    if (externalCanvasLinked !== lastExternalCanvasLinked) {
      lastExternalCanvasLinked = externalCanvasLinked;
      render();
    }
  }, 300);
  const oldRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    activeRefModOverlayCleanup?.();
    document.removeEventListener("keydown", onRefModReloadKey);
    window.clearInterval(node.__dasiwaH3LengthPoll);
    window.clearInterval(node.__dasiwaH3ExtPoll);
    oldRemoved?.apply(this, args);
  };
  syncNodeBounds();
  requestAnimationFrame(() => syncNodeBounds("grow"));
  render();
}

installH3VaeErrorPopup();

app.registerExtension({
  name: "DaSiWa.MiniMaxH3Director",
  nodeCreated(node) {
    if (node.comfyClass === "MiniMaxH3Director") install(node);
  },
  loadedGraphNode(node) {
    if (node.comfyClass === "MiniMaxH3Director") {
      install(node);
      node.__dasiwaH3RestorePersistedState?.();
    }
  }
});
