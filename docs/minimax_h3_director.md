# MiniMax H3 Director

A timeline-based authoring node for ComfyUI's native MiniMax H3 models. It centralizes media management, ordering, trimming, per-reference prompts, and the global prompt into one workflow node, validates H3 constraints before execution, and routes everything to the installed native MiniMax H3 implementation — no duplicate backend logic.

[News & Changelog — collection-wide news and change history →](news_and_changelog.md)

## Changelog

The Director's full change history now lives in the collection-wide [News & Changelog](news_and_changelog.md#minimax-h3-director-v1).

## Quick overview

- One node holds all your references, trims, ordering, endpoint frames, and prompts.
- Five endpoint modes — T2VA, I2VA, L2VA, FL2VA (text/image endpoints, up to 2 frame slots) and REF2VA (multi-image/video/audio references) — plus **Image Inpaint** (exactly one image, single-frame output).
- Separate Image, Video, and Audio lanes. Click a lane to select it; paste / drop compatible media there.
- Per-video stream switch: choose Video only, Audio only, or Video+embedded-audio with identical trim ranges.
- Standalone audio clips can be trimmed with left/right handles just like video.
- REF2VA shows a **REFMOD** button directly after **INPUT SCALING**. It opens a separate overlay without changing the Director node's size. The overlay loads saved standalone and v5 bundle references only when opened and explains every setting.

## RefMods in REF2VA

Place `.safetensors` RefMod files in `ComfyUI/models/refmods/` or any subfolder, for example `models/refmods/people/alice.safetensors`. The Director reads standalone image, video, and audio files plus upstream v5 bundle files directly, with no runtime dependency on another custom-node pack. You may optionally install [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) to create RefMod files.

**Credit:** The saved person RefMod concept, `.safetensors` latent file format, and strength scaling design are based on the upstream work in [Luisacaotica/ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod). This standalone DaSiWa implementation reads and writes the same file format, so RefMods created with either pack are interchangeable. Both packs can be installed side-by-side without conflict — they register different node names and categories, and both use ComfyUI's shared `models/refmods/` folder type via `folder_paths`.

Each selected row stores its editable description in the workflow. That workflow description takes precedence over the description embedded in the file. The overlay and prompt-builder **Insert RefMod #** buttons write the full expanded native text at the cursor, such as `<Video 1>: digital animation, slime girl`; `<RefMod N>` aliases remain supported in saved workflows and are translated when queued.

Upstream v5 bundles are expanded into their image, video, and audio members. A single `<RefMod N>` tag resolves to every native reference label contained by that bundle, in member order. Strength uses direct latent scaling (`latent * strength`), not the upstream pack's blur-mix behavior; use full strength if low-strength scaling does not suit a particular file.
- Video thumbnails: each uploaded video shows its first frame as a background preview behind the clip tile.
- One free-text prompt editor for all modes. The optional structure button inserts the corresponding H3 template; the reference prefill uses enabled media and saved reference descriptions.
- Frame rate: `frame_rate` input (0.1–240, default 24) sets the output FPS and is re-emitted as an output.
- Crop preview: ▶ Play crop plays only the current crop range; the preview range is draggable.
- Paste-replace: pasting over a selected tile replaces it in place, keeping its slot.
- Mode-specific prompt structure templates: T2VA/I2VA/FL2VA/L2VA use a multimodal description and audio headers, with frame alignment where applicable; REF2VA uses the six official full-reference sections. Both are optional insertions into the same prompt field.
- Only the selected model is loaded: `ref2va_model` for REF2VA, `fl2va_model` for all image-to-video modes (FL2VA family + Image Inpaint); the Guide node calls ComfyUI's built-in H3 nodes. REF2VA passes native inputs by name, preserving compatibility if Core reorders them (v0.4.36).

## Installation and graph setup

Install dependencies and restart ComfyUI:

```bash
pip install -r requirements.txt
```

Ensure your ComfyUI version includes native MiniMax H3 support. Add these two nodes from `DaSiWa/MiniMax H3`:

1. **MiniMax H3 Director** — your timeline, references, and prompt editor.
2. **MiniMax H3 Director Guide** — validation and routing to native H3 nodes.

Wire them like this:

```text
┌──────────────────────────────────────┐
│ UNET Loader                          │     diffusion_models/*.safetensors
│ CLIP Loader                          │     text_encoders/qwen3vl_32b_minimax_h3_*.safetensors
│ VAE Loader (visual)                  │     vae/minimax_h3_video_vae_fp16.safetensors
│ VAE Loader (audio)                   │     vae/minimax_h3_audio_vae_fp32.safetensors
│                                      │     (only for REF2VA mode)
└───┬────────────┬──────────┬──────────┘
    │            │          │
    ▼            ▼          ▼
┌──────────────────────────────────────────────┐
│ DaSiWa MiniMax H3 Director                   │
│  - add/edit references                       │
│  - set trims, ordering, prompts              │
│  - emits structured "guide" dict             │
└────┬─────────────────────────────────────────┘
     │ guide
     ▼
┌──────────────────────────────────────────────┐
│ DaSiwa MiniMax H3 Director Guide             │
│  - validates director output                 │
│  - assembles final prompt                    │
│  - CALLS the native ComfyUI H3 nodes:        │
│      • MiniMaxH3ImageToVideo   (FL2VA)       │
│      • MiniMaxH3ReferenceToVideo (REF2VA)    │
│  - you NEVER wire those native nodes yourself│
└────┬─────────────────────────────────────────┘
     │ positive, latent
     ▼
┌──────────────────────────────────────────────┐
│ Standard ComfyUI sampling/decoding chain     │
│  - KSampler                                  │
│  - VAE Decode                                │
│  - Enhanced Video Combine / Image Save etc.  │
└──────────────────────────────────────────────┘
```

Connections detail:

- Director `guide` → Guide `guide`
- Selected MiniMax H3 model → Director `fl2va_model` or `ref2va_model` → Director `model` output → model chain; the Guide itself takes `clip`, `vae` and `guide`, not a `model` input.
- `CLIP` → Guide `clip`
- Visual `VAE` → Guide `vae`
- In REF2VA: audio VAE → Guide `audio_vae`
- Guide outputs `positive` and `latent` → standard MiniMax H3 sampler/decoder chain

Important: the Guide node replaces and wraps ComfyUI's native `MiniMaxH3ImageToVideo` and `MiniMaxH3ReferenceToVideo` nodes. You do not add or wire those native nodes yourself — the Guide calls them internally based on the chosen mode.

The Director has optional model sockets (`fl2va_model`, `ref2va_model`) for lazy loading: connect whichever model matches your active mode. The Guide refuses REF2VA without an audio VAE and detects a swapped MiniMax H3 video/audio VAE before native execution (v0.4.37).

### Continuity wiring and pictogram legend

**Continue is a Director view, not an additional MiniMax model mode.** Choose the underlying mode (for example FL2VA or REF2VA) as usual, then use **Continue** to extend a completed take. Older workflows without the technical continuation path still run normally, but cannot capture resumable checkpoints. The optional continuity workflow adds two processing nodes after the ordinary Guide and sampler; neither is a second Director control.

```text
[01  SOURCE]  completed video + immutable AV latent checkpoint
      │  22 hidden context frames (default; video AND audio)
      ▼
[02  GUIDE]   Director Guide + native synchronized tail conditioning
      │  fresh 141-frame sample window = 22 context + 119 new frames
      ▼
[03  SAMPLE]  standard H3 sampler
      │  sampled joint video/audio latent
      ▼
[04  APPEND]  DaSiWa H3 Continuity Append: discard hidden overlap,
      │      append only 119 new visible AV frames; stage checkpoint
      ├── cumulative latent ──► existing upscale / decode / video export
      │                                              │ actual filename
      └── checkpoint ticket ─────────────────────────┴──► [05 PUBLISH]
                                                        verify export, mark ready
```

For the first **New** take, there is no source or hidden overlap: the ordinary Guide and sampler generate the clip, `[04]` stages its complete AV latent, and `[05]` publishes it after export. On subsequent Continue runs, `[01]` means a **completed, pinned source** rather than any imported MP4. `[02]` provides invisible audiovisual motion/sound context, not repeated output footage. `[03]` runs a bounded fresh window. `[04]` keeps the previous latent prefix and adds only the newly generated suffix. `[05]` makes the new clip selectable **only if** the exporter wrote a real file. The two arrows into `[05]` are necessary: sampler success alone is not export success. The sampled-window cap does not cap total length, but each cumulative checkpoint and full decode becomes larger as you continue. Previously exported pixels/audio may change under cumulative decode or postprocessing even though the prior latent prefix is unchanged.

**Manual long-video loop:** Enable **Keep take for Continue** for the first New take and queue it; choose its completed clip ID under Continue; edit the separate next-action prompt and queue; when export finishes, explicitly use that result as the next source (or leave the original selected to reroll a branch). This never silently advances to the latest clip. Continue requires native 24 fps, a compatible model family/canvas and the original checkpoint; downstream frame interpolation may still use another display frame rate. Context length and session live under the Director's advanced Continuity controls. Audio is conditioned and extended together with video; no audio waveform is sent to Prompt Forge. The native AV tail/window/append helpers are derived from [ttulttul/ComfyUI-Minimax-H3-Continuation](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation), with its MIT license retained in `nodes/h3_continuity/vendor/LICENSE`.

**Validation boundary:** CPU latent/graph tests and isolated ComfyUI registration do not establish visual or acoustic seam quality. A real H3 GPU run with its model, video VAE and audio VAE is still required before relying on this for production long-form output; each cumulative decode and upscale becomes more expensive as the clip grows.

## Modes at a glance

### FL2VA (First/Last Frame to Video)

Use for text-only generation, single-image starting frames, or first+last frame interpolation.

- Zero images → pure T2VA
- One image → first-frame conditioning (I2VA-style)
- Two images → first and last frame interpolation (true FL2VA)
- Up to 2 image slots; video/audio files are blocked in this mode.
- Endpoint images require an alignment instruction line as the very first line of the global prompt.

Alignment instruction patterns:

For one endpoint (first frame only):
```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: ...
```

For two endpoints (first and last frame):
```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {duration}.00-second mark of the target video.

integrated_multimodal_description: ...
```

Replace `{duration}` with your Director duration setting (e.g. `8.00`). Images map left-to-right by slot order: first image = Picture 1, second = Picture 2. FL2VA always uses `integrated_multimodal_description`, not `detailed_description`.

### REF2VA (Reference to Video)

Use when you want the generated video to borrow identity, appearance, motion, composition, or sound from existing media.

- Up to 9 images, 3 videos, 3 audio clips, 12 files total.
- Each video clip offers three modes via small buttons on its clip tile:
  - **V** — Video only (frames as visual reference)
  - **A** — Audio only (extract embedded audio as reference; no video frames)
  - **V+A** — Video + embedded audio (both streams decoded with the same trim range)
- Standalone audio clips show a waveform preview and support left/right crop markers.
- Attach an external soundtrack file to any video; it shares the same trim window.
- Keep at least one image or video when using audio.

When a video provides audio (via A/V+A or attached soundtrack), that audio becomes `<Audio N>` in the reference numbering. Embedded audio and separate soundtracks share the same time crop as the host video frames.

### Image Inpaint (single image)

Use to produce a single refined/edited frame from exactly one image reference — a still, not a video.

- Exactly **one** image reference; video and audio files are rejected (a video/audio in the timeline is a hard error).
- The single image is scaled by the Input Scaling setting, then run through the native `MiniMaxH3ImageToVideo` node as a **5-frame** image-to-video pass with the image as keyframe and no last frame.
- The Guide node routes the mode to that native call (`first_frame` set, `last_frame = None`), emitting `positive` (conditioning) + `latent`.
- The output is a **one-frame batch**: feed the latent into a sampler/decoder, then use a **Get Image from Batch** node to extract the single result image.
- The Director emits an `inpaint_requested` output (true in this mode, false in all others) so a graph can branch sampling/decode paths by mode. The `fl2va_requested` output is also true for every image-to-video mode (FL2VA family + Image Inpaint).
- The resolution panel still applies (canvas drives width/height); frame rate does not (output is a single frame).

## UI walkthrough: controls and buttons

Open the node and read top-to-bottom.

### Toolbar row

- **Title label:** "MiniMax H3 Director"
- **Model Mode buttons (left):** T2VA, I2VA, FL2VA, L2VA, REF2VA, Image Inpaint shown as small pills. Active mode has purple highlight; click to switch modes. When switching modes:
  - Going to FL2VA hides non-image references but keeps them in memory so they reappear when you switch back.
  - Going to REF2VA restores all previously added media.
  - Going to Image Inpaint requires exactly one image; video/audio are blocked and the audio lane is disabled.
- **Prompt Mode toggle:** a **Simple** / **Structured** pair next to the mode buttons switches how builder fields assemble into the final prompt (Structured keeps the labelled sections, Simple renders one flat block). The selection is persisted and restored on load.
- **Load / Save:** save Reference Files, Prompts, or All as a pack; load the same scope by append or overwrite. Loading validates target-mode limits and missing files before changing the timeline, and restores the saved model mode.
- **Clear button:** always visible; removes all media and prompts from the timeline. With no content it is dimmed and reports "Nothing to clear." instead of clearing.
- **Remove button:** appears when a clip is selected; deletes that item.
- **? button:** opens the online documentation on GitHub.

### Timeline area

The main workspace has separate Image, Video, and Audio lanes stacked vertically.

#### Lane selection

- Click anywhere on a lane to select it. The selected lane gets a highlight and displays "· selected".
- Selection determines where pasted media goes:
  - Select **Image**, **Video**, or **Audio**, then Ctrl+V or drop a compatible file into that lane.
  - In REF2VA, a video set to **A** occupies the Audio lane; **V+A** presents linked video and audio references while retaining their shared trim.
- FL2VA disables the Video and Audio lanes entirely.

#### Adding media

Three ways:

1. **+ buttons:** each empty slot shows a "+"; click to open a file picker filtered for that lane type.
2. **Drag-and-drop:** drag files from your OS directly onto the desired lane.
3. **Paste:** select a lane, focus the node, press Ctrl+V with images/videos/audio on your clipboard.

Each uploaded file is stored in ComfyUI's `input/` directory and linked by relative path.

#### Clip tiles

Each media item renders as a labeled tile inside its lane.

Common elements:

- **Background preview:** images show themselves; videos display their first-frame thumbnail extracted at upload time. Makes it easy to identify references visually without opening previews.
- **Identity badge (top-left):** shows the MiniMax reference label used in prompts: `Picture 1`, `Video 2`, `Audio 1`, etc., assigned by type in timeline order.
- **Label text:** filename, duration, current crop range (for video/audio), and a truncated preview of the media prompt if present.
- **Selection outline:** click a tile to select it; enables the Remove button and populates the Media Prompt editor below.
- **Drag-to-reorder:** grab a tile and move it horizontally; dropping near another slot swaps positions.

Video-specific elements:

- **Stream selector (top-right):** three tiny buttons:
  - `V` — treat as video-only reference
  - `A` — extract and use only the embedded audio track
  - `V+A` — use both video frames and embedded audio
- **Crop markers:** vertical lines overlaid on the clip showing the selected start/end within the source. Drag these markers left/right to adjust which portion of the source is sent to MiniMax.
- **Crop readout:** displays current crop range vs source duration.

Audio-specific elements:

- **Waveform preview:** rendered canvas showing amplitude peaks across the entire source.
- **Crop markers:** same concept as video; drag to select the 2–15 second segment to use.

Images:

- No trimming; treated as single-frame anchors.

#### Removing/disabling items

- Click a tile then hit the toolbar **Remove** button, or press Delete/Backspace while a tile is selected.
- Items can also be disabled via the legacy list view (hidden by default).

### Prompt editors

Below the timeline, every model mode has one free-text prompt field, initially empty. **Insert Prompt Structure** inserts the former structured-mode template at the cursor; for REF2VA it contains `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music`, while base modes include their applicable frame-alignment instruction and description/sound/music headers. Edit or omit any part of the template. The text area is resizable and its height persists in the workflow JSON.

The dark prompt toolbar also has **Insert [Shot N]** and, for REF2VA, **Insert RefMod #** (expands a selected saved reference into native label and description). Their number controls open small popovers beside the clicked button; Enter inserts and Escape dismisses. **Prefill Labels & Summary** fills the six H3 reference sections from enabled image/video/audio references and RefMods, including audio-only/video+audio tracks, their lane positions, and any authored media or RefMod descriptions. It does not assume that an image is an opening keyframe, that a video is being edited, or that audio is copied. Existing filled sections stay intact; plain free text is moved into `detailed_description`. Complete the shot-by-shot description and soundscape yourself and check preservation markers against the intended use—no visual or audio content is inferred from pixels or waveforms. There is no separate Director prompt preview. **Prompt Forge** opens the optional LLM prompt writer (local ComfyUI model, Ollama, or configured OpenAI-compatible server): Generate shows its draft in the Forge dialog; **Apply to node** then replaces the prompt field. The last three successful generations are listed under the draft. Click one to preview it and apply it later, even if you previously closed Forge without applying; they are saved in this Director node's workflow properties and survive a saved-workflow reload. A draft generated for another mode can be viewed but must be applied while that mode is selected. Storing the workflow also stores these drafts, so use **Clear history** in Forge to remove saved drafts without touching the Director prompt, or the Director's **Clear** button to remove media, prompt and Forge history together. Reference packs still serialize only the editable prompt; the three Forge drafts live in the workflow's node properties. Old structured workflows, embedded video metadata, and structured reference packs are assembled into the same prompt field when loaded.

#### Prompt Forge: connect an LLM and apply a draft

Prompt Forge is optional and runs **before** the H3 video queue, not as a second sampler or an automatic prompt replacement. Open it from the Director's prompt toolbar, enter an **Idea**, choose **Model**, **Creativity** and **Detail** (1–10), then click **Generate**. The resulting draft changes nothing until **Apply to node** is clicked; review and edit the Director prompt before queueing. **Regenerate** writes a different draft; **Cancel** or closing the dialog stops an in-progress generation. The three most recent successful drafts are saved with the Director node and can be previewed later; a draft for another H3 mode can only be applied after switching back. **Clear history** removes drafts without changing the active prompt; the Director's **Clear** removes both. Reference packs save the applied prompt, not Forge history.

**Connect a model using one of these sources:**

| Source | Setup | How it appears in Forge |
| --- | --- | --- |
| Local ComfyUI model | Put a supported model directory (with its model configuration and weights) or a GGUF file under `ComfyUI/models/llm/`; reopen Forge to refresh the dynamically listed models. GGUF additionally needs `llama-cpp-python`. Bare `.safetensors` files without a model directory are not chat models. | `local:<name>`; runs in the ComfyUI process and unloads after generation. |
| Ollama | Install and start Ollama, then download a chat or vision model using Ollama's own model management. The default address is `http://127.0.0.1:11434`; for an Ollama instance on another machine, set **Settings → DaSiWa → H3 Forge → Ollama address**. | `ollama:<name>` from Ollama's `/api/tags`; embedding models are filtered out. |
| OpenAI-compatible server | Start a server exposing `/v1/models` and `/v1/chat/completions`, then set **Settings → DaSiWa → H3 Forge → OpenAI-compatible server address** to its base URL (for example `http://127.0.0.1:8080`). Leave it empty to disable this source. | `openai:<model-id>` from `/v1/models`. The setting accepts the server root or a URL ending in `/v1`; do not add `/chat/completions` yourself. |

These server addresses are **ComfyUI Settings, not workflow fields**; you must set them on the ComfyUI instance actually running Forge. Reopen Forge after changing a URL to refresh its picker; a missing configured server appears as a source-specific error, not as an available model. The server must support streaming chat responses. Ollama normally unloads the requested model on completion; other OpenAI-compatible servers may keep it in VRAM, and Forge reports that warning. Forge refuses a new generation while a workflow is sampling so it does not evict the video model. If you use a server on another GPU, its memory is managed by that server.

Forge uses the Director's current H3 mode and duration. Timeline references are sent in lane order: REF2VA images can be tagged **subject**, **style** or **keyframe**; image/video references may carry a **keep** instruction, and video rows state whether their video/audio streams are used. A vision-capable model can receive attached reference pictures; a text-only model receives the idea and reference descriptions, **not** visual contents. Neither path analyzes soundtrack audio. If using **Continue**, its separate Analyze → Draft action uses a selected completed clip's chronological tail thumbnails where available; **Apply draft** updates only the Continue prompt and does not queue a video. This action uses the same Forge model sources/Settings, not a second `llm_config.json`.

## Limits and validation

MiniMax H3 enforces hard caps; the Director checks these before sending data downstream:

- Reference clip length: minimum 2 seconds, maximum 15 seconds each.
- Combined visual total: ≤ 15 seconds.
- Combined audio total: ≤ 15 seconds.
- Slot counts (REF2VA): max 9 images, 3 videos, 3 audio clips, 12 total files.
- FL2VA: max 2 images; no video/audio allowed.
- Path safety: all input paths resolve strictly under ComfyUI's input directory.

Violations appear as red status messages inside the node. Fix them before queuing.

## How processing flows: upstream and downstream

Understanding the data path makes wiring and debugging easier.

### Upstream inputs (what feeds into the Director)

- **Resolution panel / duration / frame rate:** the Resolution panel under the mode controls drives the hidden width/height widgets. **Aspect: Auto** reads the first image or video reference and preserves its aspect; **Resolution: Auto** sets the resulting short side to 768 px. Common horizontal/vertical aspect choices plus DaSiWa MP and fixed-resolution presets are rounded to MiniMax's 16-pixel grid. Both selectors offer **CUSTOM** values for manual aspect, MP, or exact pixels. All three selectors default to **Auto**. The third **Input scaling** selector preprocesses visual references through the included DaSiWa Torch Resize implementation before they reach H3: **Off** preserves the original tensor, **Auto** preserves its aspect with a 2048-px short edge only when that would downscale the source (smaller inputs pass through unchanged), **Target - Selected Aspect & Resolution** stretches it to the selected Director canvas, and **Fit**, **Fill and crop**, **Fit and pad**, and **Long side with divisible crop** use the corresponding Torch Resize aspect modes against that canvas. Audio is never resized. When both external dimension overwrite inputs are connected, these Director calculations and preprocessing are disabled. The `frame_rate` FLOAT input (0.1–240, default 24) sets the output frame rate and is re-emitted as a `frame_rate` output for downstream nodes.
- **Optional model sockets** (`fl2va_model`, `ref2va_model`): connect only the model matching your current mode; the Guide uses them lazily.
- All media is managed inside the node UI (upload/paste/drop), but paths ultimately live in ComfyUI's `input/` folder.

### Inside the Director

On queue, the Director executes this sequence:

1. Reads current mode (FL2VA-family, REF2VA, or Image Inpaint).
2. Iterates over all enabled timeline items in slot order.
   - For FL2VA: keeps only image items (max 2); discards others temporarily.
   - For REF2VA: processes images, videos, and audio respecting slot limits.
   - For Image Inpaint: accepts exactly one image reference; video/audio items are a hard error.
3. Loads each asset:
   - Images → resized tensors.
   - Videos → decoded to `frame_rate` fps frame batches (default 24), cropped according to trim_start/trim_end.
   - Audio → decoded waveforms, cropped identically.
   - For videos in A or V+A mode → embedded audio is extracted using the same crop window.
   - Attached soundtracks → loaded and cropped using the host video's trim range.
4. Builds a structured `guide` dictionary containing:
   - Mode flag, dimensions, duration, and frame rate.
   - Ordered lists of images, videos, audios with metadata.
   - Endpoint frames (FL2VA).
   - Reference maps keyed as `ref_image_N`, `ref_video_N`, `ref_audio_N`, `ref_video_audio_N`.
5. Reads the mode-specific prompt-builder state:
   - FL2VA/I2VA/L2VA/T2VA: uses integrated_multimodal_description, overall_soundscape, non_diegetic_music fields.
   - REF2VA: uses the six free-text sections (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music).
   - Alignment lines (I2VA/FL2VA/L2VA) are injected automatically based on mode and duration.

This `guide` object and `builder_state` are passed out to the Guide node.

### The Guide node

The Guide is a thin adapter between your authored timeline and ComfyUI's native H3 nodes:

1. Validates the incoming `guide`:
   - Confirms mode consistency.
   - Checks that required models/CLIP/VAEs are connected.
   - For REF2VA, ensures an audio VAE exists.
2. Assembles the final prompt via the prompt-builder helper:
   - Reads `builder_state` from the Director.
   - For FL2VA/I2VA/L2VA/T2VA: injects alignment lines (when applicable), combines integrated_multimodal_description + overall_soundscape + non_diegetic_music into the canonical format.
   - For REF2VA: wraps the six user-written sections with their standard headers (`subject_definitions:`, `summary:`, etc.). Legacy v1 structured-builder data is merged automatically if present.
   - Writes the result as `resolved_prompt`.
3. Routes to the appropriate native node:
   - FL2VA / I2VA / L2VA / T2VA → calls `MiniMaxH3ImageToVideo` with endpoint frames and prompt.
   - Image Inpaint → calls `MiniMaxH3ImageToVideo` with the single image as first frame, `last_frame = None`, and a fixed 5-frame length.
   - REF2VA → calls `MiniMaxH3ReferenceToVideo` with all reference maps and prompt.
4. Emits standard ComfyUI outputs plus an optional continuity context:
   - `positive` (conditioning)
   - `latent` (image batch — a one-frame batch for Image Inpaint; extract the result with **Get Image from Batch**)
   - `continuity_context` (disabled for older workflows and uncaptured New takes; wire only to **H3 Continuity • Append & Stage** in the continuity workflow).
   - These feed downstream samplers and decoders exactly like any other H3 workflow.

You never call the native MiniMax H3 nodes directly when using Director+Guide; the Guide abstracts that away.

## Model chain: patching & preview

The Director's `fl2va_model` and `ref2va_model` inputs are plain `MODEL` sockets, so any node that outputs `MODEL` can sit upstream of the Director — a LoRA loader, the **MiniMax H3 Cache** patcher, **Patch Comfy Kitchen Attention**, or a KJ `ModelPreviewOverrideKJ`. This is the same forward chain you already use:

```
Checkpoint.MODEL → LoRALoader.MODEL → [KJ.model → KJ.MODEL] → Director.fl2va_model
Checkpoint.CLIP  → LoRALoader.clip  → Director.clip
```

Three rules keep the chain valid:

1. **Forward chain only — never a loop.** A patcher's output feeds *into* the Director's model input; it must never come back out of the Director. The Director is a terminal media node (it emits `frame_rate`, `duration`, `images`, never `MODEL`), and a wire from the Director back into its own model input would be a graph `dependency_cycle`, which ComfyUI's validation rejects.
2. **One loader per model, in mode order.** The Director picks `ref2va_model` for REF2VA and `fl2va_model` otherwise; the active input must be connected (the unconnected twin may stay empty).
3. **Type-safe wires.** ComfyUI only lets you connect type-compatible sockets, so `LoRA.MODEL → Director.fl2va_model` is legal but `LoRA.MODEL → Director.clip` is not. No name or type resolution happens at runtime — the socket you plugged in arrives as the keyword-argument named for that socket.

## Dense prompting guide

MiniMax H3 expects structured natural language rather than keyword piles. Use the official field names and shot/timestamp conventions.

### FL2VA prompt structure

Text-only (no endpoint images):
```text
integrated_multimodal_description: [Shot 1] ... [Shot 2] At 00:04.500, ...

overall_soundscape: ...

non_diegetic_music: ...
```

With endpoint images: prepend the alignment instruction line shown earlier, blank line, then the three fields above.

### REF2VA prompt structure

Use the six-section format. Define assets once and reuse labels consistently.

Label rules:
- `<Subject N>`: visible content abstracted from references (people, objects, scenes, clothing, actions). This is what actually appears.
- `<Picture N>`: ONLY for concrete frame anchors (opening/key/last frame, storyboard). If an image defines style/appearance only, cite it inside a Subject and don't create a Picture entry.
- `<Video N>`: structural roles only (editing source, continuation, pacing/cuts/rhythm). Specific subjects/actions/styles from a video belong under Subjects.
- `<Audio N>`: copied or referenced audio signals (dialogue, music, ambience, voice timbre).

Template:
```text
subject_definitions:
<Subject 1> is the woman in <Picture 1>, with short dark hair and a red coat.
<Picture 1> is the opening-frame anchor for [Shot 1].
<Subject 2> is the walking motion taken from <Video 1>.
<Video 1> provides the camera path and pacing structure.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] Use <Subject 1> from <Picture 1>, the motion and pacing of <Video 1>, and the voice character of <Audio 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity and clothing remain consistent.
<Picture 1> ([Shot 1] first frame): fully_preserved - opening composition anchor.
<Subject 2> (motion transferred to <Subject 1>): attribute_transfer - walk rhythm is applied to <Subject 1>.
<Video 1> (pacing structure): weak_reference - general timing and camera rhythm are retained.
<Audio 1>: reference - timbre and delivery are followed without copying the signal.

detailed_description: [Shot 1] ... [Shot 2] At 00:04.500, ...
overall_soundscape: ...
non_diegetic_music: ...
```

Summary task-type prefixes (combine with ` + `):
- `[keyframe completion]`, `[reference generation]`, `[video editing]`, `[video continuation]`, `[audio reuse]`, `[audio reference]`

Retention markers:
- Visual: `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`
- Audio: `fully_copy`, `partially_copy`, `reference`, `weak_reference`

Number labels by timeline order: images first (Pictures), then videos, then audio. Keep meanings stable everywhere.

### Shots and timestamps

- `[Shot 1]` starts without a timestamp. Later shots begin at cut times in `MM:SS.mmm`.
- Use timestamps for actual cuts or important transitions, not every sentence.
- Timestamps are relative to the generated output timeline; keep them within your duration and align with media timeline when relevant.
- Single continuous shot: one `[Shot 1]` block, describe temporal changes inline.
- Distinguish source vs target timing for references: `At 00:05.000 in the target video, reproduce the hand gesture seen near 00:02.400 in <Video 1>`.

### Camera motion vocabulary

Use as natural English inside shots:
- Types: Zoom/Push/Pan/Truck/Tilt/Pedestal/Arc/Tracking/Shake/POV/Roll
- Amplitude: `with small/large amplitude`
- Speed: `at slow/fast speed`

Example: `The camera pushes in with small amplitude at slow speed toward her hands.`

### Dialogue and special tokens

- Speaker IDs: `(S1)`, `(S2)` by vocal appearance order; reuse consistently.
- Exact dialogue: `<d>[Language] ...</d>` preserving original language/punctuation.
- Voiceover: say `in an off-screen voiceover` and specify lips remain closed after the tag.
- Cross-cut speech: `<scenetrans>` at both sides; `<cutoff>` when truncated.
- On-screen text: double-quote in English, preserve original characters exactly.

### Practical workflow

1. Choose FL2VA for endpoint/text work; REF2VA for multi-reference transfer.
2. Add only references that contribute specific identity, motion, layout, or sound.
3. Trim videos/audio to the strongest 2–15s segments; respect totals.
4. Write directly in the prompt field, or click **Insert Prompt Structure** for the mode's H3 format.
   - FL2VA/I2VA/L2VA/T2VA: fill the description, soundscape and music; add the correct frame-alignment timing.
   - REF2VA: click **Prefill Labels & Summary** to populate labels for enabled references, then check their roles and retention markers and write the actual detailed shots and soundscape. Use **Insert [Shot N]** for clean shot markers.
5. Verify the text in the prompt field, duration, aspect ratio, and motion against your references; queue through the Guide.

Official MiniMax H3 guides (canonical conventions):
- [Video Prompt Writing Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [Full-Reference Rewrite Format Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
