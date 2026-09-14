import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const sourcePath = new URL("../js/minimax_h3_director.js", import.meta.url);
let source = (await readFile(sourcePath, "utf8")).replace(/\r\n/g, "\n");
source = source.replace(
    'import { app } from "../../scripts/app.js";\nimport { api } from "../../scripts/api.js";',
    "const app = { registerExtension() {} }; const api = { addEventListener() {} };"
);
source = source.replace('import { renderRefMods, refModPreview } from "./director_refmods.js";', 'const renderRefMods = () => {}; const refModPreview = text => text;');
source += "\nexport { mediaTypeFor, wavDurationFromBuffer, REPOSITORY_URL, MINIMAX_MULTIPLE, ASPECT_OPTIONS, RESOLUTION_PRESETS };";

assert.match(source, /lane\.ondrop = event => \{ if \(!supported\)/, "each timeline lane must own its direct drop handler");
assert.match(source, /acceptLaneDrop\(event, targetLane\)/, "supported lane drops must use the direct lane upload path");
assert.match(source, /resizer\.className = "ds-h3-prompt-field-resizer"/, "each prompt field must expose a bottom resize grabber");
assert.match(source, /wrapper\.appendChild\(resizer\)/, "the resize grabber must live inside the prompt field");
assert.match(source, /\.ds-h3\{box-sizing:border-box[^`]*background:transparent;border:0;border-radius:0;padding:0/, "the Director UI must render directly on the node without an outer panel");
assert.doesNotMatch(source, /promptPanelHeight/, "restoring a workflow must not reference the removed prompt-panel divider state");
assert.match(source, /FL2VA supports image references only; video and audio are unavailable\./, "unsupported drops must show an explicit FL2VA error");
assert.match(source, /ds-h3-timeline-lane \$\{targetLane\}[^`]*\$\{supported \? "" : " disabled"\}/, "unsupported lanes must be visibly blocked");
assert.match(source, /const ioSide = controlGroup\(\); ioSide\.style\.cssText \+= ";padding-left:8px;border-left:1px solid #344452";/, "Load and Save must have their own separated toolbar group");
assert.match(source, /topRow\.append\(modesSide, promptSide, spacer, ioSide, actionsSide, docsButton\)/, "the toolbar must place Load/Save before the right-aligned Remove/Clear group");
assert.match(source, /\], ioSide, true, "Load", "ds-h3-io-dropdown load"\);[\s\S]*?null, ioSide, true, "Save", "ds-h3-io-dropdown save"\);/, "Load and Save must render in the separated toolbar group");
assert.match(source, /removeButton\.textContent = "Remove"/, "selected media must have a toolbar remove button");
assert.match(source, /close\.className = "ds-h3-clip-close"/, "selected media must have a clip-corner remove button");
assert.doesNotMatch(source, /Remove selected media item/, "the old prompt-panel remove button must be absent");
assert.match(source, /\.ds-h3-prompt-panel\{width:100%;box-sizing:border-box;border:0;border-radius:0;padding:0/, "prompt fields must not sit inside another framed panel");
assert.match(source, /ds-h3-status ds-h3-info-field/, "status messages must use their own info field");
assert.match(source, /Math\.min\(sourceDuration, 15\)/, "long uploaded media must default to a 15-second crop");
assert.match(source, /extractWaveform\(value, added\.id\)/, "audio uploads must decode a waveform");
assert.match(source, /waveform\.className = "ds-h3-waveform"/, "audio clips must render their waveform canvas");
assert.match(source, /let selectedLane = "image"/, "the timeline must keep an active destination lane");
assert.match(source, /ds-h3-timeline-lane\.selected/, "the active lane must be visibly highlighted");
assert.match(source, /timeline\.addEventListener\("paste", async event =>/, "the timeline must accept clipboard paste events");
assert.match(source, /await acceptFile\(file, selectedLane\)/, "clipboard files must be routed to the selected lane");
assert.match(source, /lane selected\. Paste files here with Ctrl\+V\./, "lane selection must explain the clipboard interaction");
assert.doesNotMatch(source, /FL2VA: \$\{timelineSeconds\.toFixed\(2\)\}s/, "the obsolete time-scale hint must be absent");
assert.match(source, /audioSlotWidth = sourceDuration =>.*Math\.log2/, "audio slots must fit long source files logarithmically");
assert.match(source, /ds-h3-audio-crop-marker start/, "audio clips must draw a crop-start position marker");
assert.match(source, /ds-h3-audio-crop-marker end/, "audio clips must draw a crop-end position marker");
assert.match(source, /dragging = "range"/, "the preview crop range must be draggable as a whole");
assert.match(source, /const width = parseFloat\(rangeTe\.value\) - parseFloat\(rangeTs\.value\);[\s\S]*?rangeTe\.value = start \+ width;/, "moving the crop range must preserve its duration");
assert.match(source, /playCropBtn\.textContent = "▶ Play crop"/, "audio and video previews must expose a crop-playback button");
assert.match(source, /media\.currentTime = start;[\s\S]*?await media\.play\(\)/, "crop playback must seek to the current crop start before playing");
assert.doesNotMatch(source, /media\.pause\(\);\s*media\.currentTime = start;/, "crop playback must not clear its end guard with an asynchronous pause event before starting");
assert.match(source, /if \(cropPlayback && media\.currentTime >= Number\(teInput\.value\)\)[\s\S]*?media\.pause\(\);/, "crop playback must stop at the current crop end");
assert.match(source, /Math\.log1p\(9 \* peaks\[peakIndex\]\)/, "audio waveform amplitudes must use logarithmic scaling");
assert.doesNotMatch(source, /if \(item\.type === "audio"\) return; if \(event\.target !== clip\) return;/, "audio slots must be horizontally movable");
assert.match(source, /window\.open\(REPOSITORY_URL, "_blank", "noopener,noreferrer"\)/, "the help button must open the GitHub documentation safely");
assert.match(source, /const promptStyle = \(\) => \{ const v = builderState\?\.prompt_mode;/, "the builder must expose a validated prompt-style accessor");
assert.match(source, /modeLabel\.textContent = "Model Mode:"/, "the mode bar must label the mode selector");
assert.match(source, /promptLabel\.textContent = "Prompt Mode:"/, "the mode bar must label the prompt-style selector");
assert.match(source, /\[\["simple", "Simple"\], \["structured", "Structured"\]\]/, "the prompt-style selector must expose separate Simple and Structured buttons");
assert.match(source, /promptButton\.classList\.toggle\("active", styleLabel === value\)/, "the selected prompt style must be highlighted");
assert.match(source, /spacer\.style\.flex = "1"/, "the mode bar must use an automatic spacer before the help button");
assert.match(source, /function buildSimpleForm\(panel\)/, "simple mode must render its own one-field prompt form");
assert.match(source, /createBuilderField\("Prompt", builderState\.simple_prompt/, "simple mode must bind the visible field to serialized simple_prompt state");
assert.match(source, /if \(value === "simple"\) builderState\.simple_prompt = previewTextFor\(mode\(\), false\);/, "switching into simple mode must seed the single field from the current builder prompt");
assert.match(source, /promptButton\.className = "ds-h3-prompt-mode-btn"/, "prompt-style choices must use the prompt-mode button styling");
assert.match(source, /function showPromptPreview\(\) \{[\s\S]*?let promptText = previewTextFor\(m, hasExternalPrompt\(\)\);/, "the prompt preview must render the selected style");
assert.match(source, /builderState\.prompt_mode = value/, "prompt-style buttons must set the explicitly selected mode");
assert.match(source, /const resetBuilderState = \(\) => \{ builderState = DEFAULT_BUILDER_STATE\(mode\(\)\); builderState\.mode = mode\(\); \};/, "Clear must reset all serialized builder fields to the current mode defaults");
assert.match(source, /const clearAll = \(\) => \{ selectedId = null; resetBuilderState\(\);/, "Clear must reset prompt/text fields as well as selected media");
assert.match(source, /const hasContent = state\.items\.length \|\| state\.refmods\?\.length \|\| state\.prompt_blocks\?\.length \|\| hasBuilderContent\(\) \|\| String\(promptWidget\?\.value \|\| ""\)\.trim\(\);/, "Clear must remain available when only builder text is filled");

assert.match(source, /function probeDimensions\(value, type\)/, "visual uploads must probe image and video source dimensions");
assert.match(source, /source_width: width, source_height: height/, "source dimensions must persist with the media item");
assert.match(source, /ds-h3-resolution-panel/, "the resolution controls must render in their own panel");
assert.match(source, /Native \(ShortEdge 768px\)/, "Auto resolution must describe the MiniMax 768px short-side default");
assert.match(source, /MINIMAX_MULTIPLE = 32/, "MiniMax output dimensions must align to the 32px H3 grid");
assert.match(source, /source \? Number\(source\.source_width\) \/ Number\(source\.source_height\) : 4 \/ 3/, "Auto aspect must use the MiniMax T2VA 4:3 fallback without visual media");
assert.match(source, /settings\.resolution === "custom" && settings\.custom_mode === "fixed"\) return \[snap16\(settings\.custom_width\), snap16\(settings\.custom_height\)\]/, "fixed custom pixels must resolve without a media-derived aspect");
assert.match(source, /options\.forEach\([\s\S]*?select\.value = value; select\.oninput/, "resolution selects must set their value after options exist and react on input");
assert.match(source, /\$\{settings\.aspect === "auto" \? "Auto 768px" : settings\.aspect\} · \$\{settings\.resolution === "auto" \? "Auto 768px" : settings\.resolution\}/, "the readout must identify the actual selected aspect and resolution");
assert.match(source, /INPUT SCALING/, "the resolution panel must expose the third input-scaling dropdown");
assert.match(source, /input_scaling: "Auto"/, "input scaling must default to Auto alongside Aspect and Resolution");
assert.match(source, /\[\["Off", "Off"], \["Auto", "Native \(ShortEdge 2048px\)"], \["Target", "Target · Selected Aspect & Resolution"], \["Fit", "Fit"\]/, "input scaling must expose the reused Torch Resize behaviours");
assert.match(source, /external_width_overwrite/, "the Director frontend must recognize the external width overwrite input");
assert.match(source, /external_height_overwrite/, "the Director frontend must recognize the external height overwrite input");
assert.match(source, /external_prompt_overwrite/, "the external prompt input must be named as an overwrite");
assert.match(source, /function migrateLegacyExternalPromptInput\(\)/, "restored Director nodes must migrate the old external prompt socket");
assert.match(source, /legacy\.name = "external_prompt_overwrite"/, "a linked legacy prompt socket must be renamed instead of duplicated");
assert.match(source, /hasExternalCanvas\(\).*return;/, "external dimensions must prevent Director canvas recalculation");
assert.match(source, /Director sizing and input scaling disabled/, "the UI must disclose external canvas overwrite mode");
assert.match(source, /ds-h3-fixed-dimensions/, "fixed custom resolution must group its dimensions in a dedicated row");
assert.match(source, /display:flex;gap:6px/, "fixed custom resolution dimensions must use a horizontal flex layout");
assert.match(source, /dimensionField\("WIDTH", settings\.custom_width/, "fixed custom resolution must label the width field");
assert.match(source, /dimensionField\("HEIGHT", settings\.custom_height/, "fixed custom resolution must label the height field");
assert.match(source, /state\.field_heights = \{ \.\.\.fieldHeights \};/, "resized prompt fields must persist their height in the serialized timeline state");
assert.match(source, /const key = opts\.fieldKey \|\| "";/, "prompt fields must record resized heights under stable per-field keys");

const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { mediaTypeFor, wavDurationFromBuffer, REPOSITORY_URL, MINIMAX_MULTIPLE, ASPECT_OPTIONS, RESOLUTION_PRESETS } = await import(moduleUrl);

assert.equal(mediaTypeFor({ name: "REFERENCE.WAVE", type: "" }), "audio", "the .wave WAV alias must be accepted even without a browser MIME type");
assert.equal(mediaTypeFor({ name: "reference.wav", type: undefined }), "audio", "WAV detection must tolerate missing MIME types");
const wav = new ArrayBuffer(64);
const wavView = new DataView(wav);
for (const [offset, text] of [[0, "RIFF"], [8, "WAVE"], [12, "fmt "], [36, "JUNK"], [48, "data"]]) for (let index = 0; index < text.length; index++) wavView.setUint8(offset + index, text.charCodeAt(index));
wavView.setUint32(4, 56, true); wavView.setUint32(16, 16, true); wavView.setUint32(28, 192000, true); wavView.setUint32(40, 4, true); wavView.setUint32(52, 8, true);
assert.equal(wavDurationFromBuffer(wav), 8 / 192000, "WAV duration fallback must accept metadata chunks between fmt and data");

for (const [name, type] of [
    ["reference.webp", "image"],
    ["reference.heic", "image"],
    ["reference.mkv", "video"],
    ["reference.m2ts", "video"],
    ["reference.flac", "audio"],
    ["reference.opus", "audio"],
]) {
    assert.equal(mediaTypeFor({ name, type: "" }), type, `${name} must classify by extension when the browser omits MIME type`);
}
assert.equal(mediaTypeFor({ name: "reference.bin", type: "" }), null, "unknown files must not be uploaded as media");
assert.equal(mediaTypeFor({ name: "renamed.bin", type: "audio/ogg" }), "audio", "known MIME types take precedence over extensions");
assert.equal(REPOSITORY_URL, "https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/main/docs/minimax_h3_director.md");
assert.equal(MINIMAX_MULTIPLE, 32);
assert.equal(ASPECT_OPTIONS[0][0], "auto");
assert.equal(ASPECT_OPTIONS.at(-1)[0], "custom");
assert.equal(RESOLUTION_PRESETS["8.30 MP - UHD"], 8.30);
