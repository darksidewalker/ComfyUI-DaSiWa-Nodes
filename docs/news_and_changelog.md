# News & Changelog

Collection-wide news and change history for the DaSiWa Custom Nodes — one place to see what changed across every node. Per-node deep dives (UI guides, wiring, options) stay in their own docs, linked from the README.

This changelog covers **2026-07-05 → 2026-09-24**. Older history lives in the git log. Entries within each section are listed **newest first**.

## News

- **Prompt Forge guide:** The [Director guide](minimax_h3_director.md#prompt-forge-writing-and-applying-a-draft) now walks through idea, model, creativity and detail choices, reference roles, draft review, applying, cancellation, and saved-history behavior. The README lists Prompt Forge as a separate major Director feature.

- **H3 Forge saved drafts (09-24, 0.4.57):** Each Director keeps its last three successful Forge generations in its saved workflow. Reopen Forge to preview and apply one; **Clear history** removes drafts only, while the Director's **Clear** also clears its Forge history. [Director guide →](minimax_h3_director.md)

- **H3 Prompt Forge and single prompt editor (09-24, 0.4.56):** PR #51 adds on-demand prompt writing with a local ComfyUI LLM, Ollama, or a configured OpenAI-compatible server. Director prompts start empty in one editor; Insert Prompt Structure restores the former section template on demand. Prompt helpers and Prompt Forge share a dark rounded toolbar. Old structured workflows, embedded prompts, and reference packs migrate into the single field. [Director guide →](minimax_h3_director.md)

- **Unreleased — Registry automation:** The failing scheduled Registry-status workflow is removed. On a future `pyproject.toml` version bump, the publish workflow will retain the newly active release, deprecate its four immediate predecessors, and unpublish older versions. No Registry versions are changed by the workflow edit itself.

- **Free Memory toolbar control (09-23, 0.4.55):** A separate DaSiWa-logo button beside the top-docked monitor offers **Free VRAM** (unload ComfyUI-managed models) and **Free System RAM** (unload models and reset the execution cache). It stays available when monitoring is off, has its own DaSiWa setting, and closes its menu on outside clicks. The monitor chips, settings button, and memory button now align at 36px height. [Usage and limits →](system_monitor.md#free-memory-toolbar-button)
- **Director prompt count and monitor overlays (09-22, 0.4.53–0.4.54):** The prompt character counter now counts the assembled prompt, including structured headers. Full monitor mode and its settings menu render above toolbar content instead of being clipped; the Full view retains meter colors and fill bars.

- **MiniMax H3 Director: RefMod lane & prompt-mode fixes (09-22):** Fixed stale/duplicate RefMod numbering on fresh ComfyUI load (falls back to saved `media_type` when library data isn't loaded yet). Remove button in REFMOD overlay now updates the timeline immediately. Clear button disables all refmods in the overlay and removes them from lanes. Mode-aware lane visibility: T2VA hides the reference grid entirely, non-reference modes show a single Image row only (no clutter). Track height adapts to fit shown lanes instead of fixed 280px minimum. Simple/Structured prompt mode switching now preserves content bidirectionally — switching Structured → Simple flattens fields, Simple → Structured parses them back; unlabeled text dumps into detailed_description. Added right-aligned character counter to all prompt builder forms (Base, Simple, REF2VA) that updates live as you type. Refmod X button in slotline now disables the refmod in the overlay instead of trying to remove it from items. Strength changes reflect immediately in the timeline without requiring a lane re-click. Version bump to 0.4.52.

- **Recent stability fixes (09-19, 0.4.48–0.4.49):** RefMod image dimensions are read correctly from 5D latents; Enhanced Video Combine no longer exposes videos as image assets; stale drive mountpoints no longer break monitor polling. The experimental MiniMax H3 Latent Upscaler was removed after output-quality issues—it is not a shipped node.

- **System Monitor: global DaSiWa settings switch (09-21):** **Settings → Other → DaSiWa → System Monitor** now controls the monitor completely. Off removes its toolbar/floating UI, dock targets, frontend listeners, and backend telemetry polling; on mounts and starts them again. Version bump to 0.4.51.

- **Settings About: installed nodepack version (09-21):** ComfyUI → Settings → About now shows a linked `DaSiWa Custom Nodes v…` badge, using the packaged release version. Version bump to 0.4.50.
- **Recent stability fixes (09-19, 0.4.48–0.4.49):** RefMod image dimensions are read correctly from 5D latents; Enhanced Video Combine no longer exposes videos as image assets; stale drive mountpoints no longer break monitor polling. The experimental MiniMax H3 Latent Upscaler was removed after output-quality issues—it is not a shipped node.

- **MiniMax H3 Director: upstream RefMod v5 bundles (09-19):** REF2VA now loads current standalone and bundled RefMods created by ComfyUI-MiniMaxH3Mod. Bundle members are expanded into their image, video, and audio references; one `<RefMod N>` alias resolves to every contained native label in member order. Version bump to 0.4.47.

- **Enhanced Video Combine: `%seed%` output naming (09-19):** The optional `seed` input now expands `%seed%` in `filename_prefix`, so videos and selected frame exports can include the exact generation seed. Existing workflows remain unchanged when no seed is connected. Version bump to 0.4.45.

- **MiniMax H3 Director: RefMod timeline visualization + upstream credit (09-18):** Enabled RefMods now render as read-only clips in their appropriate reference lane (Image/Video/audio) with a green REFMOD badge, slot number, and strength indicator — users can see total reference count at a glance alongside uploaded media. Missing RefMod files now warn-and-skip instead of hard-erroring, preventing stale workflow saves from crashing generation. **Upstream credit:** the saved person RefMod concept, `.safetensors` latent file format, and strength scaling design are based on [Luisacaotica/ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod); both packs can be installed side-by-side and share the same `models/refmods/` folder. Version bump to 0.4.43.

- **MiniMax H3 Director: standalone RefMod references (09-18):** REF2VA gains a **REFMOD** button after **INPUT SCALING** that opens a separate explanatory overlay without expanding the node. The overlay lazily loads standalone image, video, and audio RefMod files under `models/refmods/`. Reading is self-contained with no upstream runtime dependency; workflow descriptions override file metadata, file changes invalidate ComfyUI caches, and unsafe paths or bundle files are rejected. Version bump to 0.4.41.

- **Enhanced Video Combine: PyAV-native encoding and seekable previews (09-15):** Video encoding, audio muxing, metadata, animated WebP/AVIF, and compatibility preview transcoding now run in-process through PyAV 18 without launching an FFmpeg executable. Hardware candidates retain the NVENC → QSV → AMF → VAAPI order and fall back to software after a real encode attempt fails. Every generated video receives one-second keyframes; MP4 uses fast-start metadata, while AV1, VP9, HEVC, 10-bit, and other compatibility previews use cached H.264/AAC files served with HTTP byte-range support for reliable pause and timeline scrubbing. Original downloads keep their selected codec and container. Version bump to 0.4.40.

- **Registry security remediation (09-15):** LLM/VLM nodes now use already-installed local model folders or GGUF files only; runtime Hugging Face downloads, Hugging Face token reads, and custom remote model code are removed. The Ollama backend is fixed to the loopback API, so workflows cannot send the ComfyUI server to an arbitrary URL. Enhanced Video Combine preview transcodes only files from the output root, never a caller-selected ComfyUI asset root. The Registry audit utility and security-review document record the release status and remaining manually reviewed capabilities. Version bump to 0.4.39.

- **MiniMax H3 Director: reference-pack workflow and layout fixes (09-11):** Save/Load packs can append or overwrite reference files, prompts, or both after validating the saved mode, real target limits, and referenced-file availability. REF2VA now separates Image, Video, and Audio lanes, while V / A / V+A references retain their correct limits and linked audio placement. L2VA's closing-frame slot is locked, and the Guide reports swapped H3 VAEs before native execution. Prompt-mode toolbar controls wrap inside their node instead of overflowing. Version bump to 0.4.37.

- **MiniMax H3 Director Guide: REF2VA native-call compatibility (09-10):** REF2VA now passes every native `MiniMaxH3ReferenceToVideo` input by name. The current Core order remains correct, and the Guide stays compatible if Core reorders those inputs. Version bump to 0.4.36.

- **Advanced LoRA Loader: Civitai mirrors + (i) panel rework (09-08):** the info panel now always shows both Civitai mirror links — `.com` and `.red` — with color-coded labels (`BLUE:` in blue, `RED:` in red, labels plain text outside the clickable links). The backend looks the file up on `.com` first and falls back to the `.red` mirror, so a working mirror still yields a link when one side is down. Lookups are memoized, misses included — a LoRA with no Civitai page no longer re-hits the API on every panel open; the panel's **Refresh** button forces a re-lookup. The file name and sha256 now sit on their own rows at the top of the panel, so long folder paths can't stretch the controls. Version bump to 0.4.35.

- **Advanced LoRA Loader trash button (08-29):** every slot row gets a trash button (plain canvas paths, ASCII — a drawn trash-can, no emoji) directly right of the info button — it resets that slot's LoRA back to "None" (strengths and multipliers kept, exactly like selecting None in the picker), so you can unstack a LoRA without reopening the picker. The info button shifted slightly left to make room. No backend change; version bump to 0.4.29.
- **Seed Control queue roll (08-29):** Random mode rolls a fresh seed on every prompt build under the new Vue frontend — the legacy extension-level queue hook stopped dispatching there, and the Run button's internal queue path doesn't call the global queue entry point either, so the roll now runs by wrapping the app's prompt build (`app.graphToPrompt`), the choke point every Run routes through (Pixaroma-seed-node style; Fixed mode and external-seed links unchanged).
- **LoRA info button (08-29):** the Advanced LoRA Loader rows now have an ⓘ glyph at the right edge. It opens a panel with the LoRA's Civitai link (looked up by the file's SHA-256, cached in `lorainfo/`), trigger/trained words from the safetensors header and Civitai (click-select, copy), and preview images (Civitai + a local sidecar `*.png` next to the LoRA if present).
- **Advanced LoRA Loader: universal rename + PDD/ACC support + opt-in cache (08-29):** the LTX-2-only loader is renamed to a universal **Advanced LoRA Loader** (serialized node ID unchanged), forwards PDD/ACC LoRA metadata to Core so PDD/ACC head banks activate, and gains an **opt-in** `use_cache` button (default off) that caches each unique LoRA file across slots.
- **H3 Cache compatibility & quality parity (08-29):** PDD LoRA head bank support (ComfyUI 0.34+) and per-token denoise-mask parity with Core.
- **Image Inpaint mode for the Director (08-28):** a 5-frame image-to-video pass through the native `MiniMaxH3ImageToVideo` node; the `inpaint_requested` output lets downstream sampling branch on mode.
- **Director 2.0 frozen (08-28):** the experimental v2 fork is removed from the nodepack and preserved under `frozen/`; v1 plus Image Inpaint is the supported path.
- **Seed Control node (08-26):** the Director's seed panel extracted into a standalone node with full 64-bit seeds, a Random|Fixed switch, and a NOISE-compatible output.
- **Issue forms (08-17):** structured bug-report and feature-request forms; the DaSiWa node list syncs automatically from the node registry.
- **MiniMax H3 family complete (08-16):** the collection now ships the full H3 stack — **MiniMax H3 Director** (timeline authoring, integrated 08-03), **MiniMax H3 Cache** (approximate block-stack residual cache), and **Patch Comfy Kitchen Attention** (INT8 attention model patch).
- **License: Apache 2.0 → GPL v3 (08-04).**
- **New authoring nodes (July):** Wildcard & Preset Prompt Builder, LLM / VLM Analyze (GGUF and Ollama backends), DaSiWa System Monitor, DaSiWa Torch Resize, and Enhanced Video Combine.

## Releases

Quick reference for the version bumps inside this window, newest first:

| Version | Date | Headline |
|---|---|---|
| 0.4.57 | 09-24 | Forge saves three drafts per Director; Forge history and Director Clear controls |
| 0.4.56 | 09-24 | H3 Prompt Forge and single free-text Director prompt with optional structure insertion and legacy migration |
| 0.4.55 | 09-23 | Independent Free Memory toolbar button; VRAM/model unload and RAM/cache reset actions |
| 0.4.54 | 09-22 | Director prompt counter includes assembled structured headers |
| 0.4.53 | 09-22 | Monitor Full overlay and settings-menu clipping, colors, and fill bars fixed |
| 0.4.52 | 09-22 | MiniMax H3 Director: RefMod lane fixes, prompt-mode content preservation, char counter, immediate overlay sync |
| 0.4.51 | 09-21 | System Monitor: global DaSiWa settings switch that fully mounts/stops telemetry |
| 0.4.50 | 09-21 | Settings About: visible installed DaSiWa Custom Nodes version badge |
| 0.4.49 | 09-19 | Experimental H3 Latent Upscaler (subsequently removed before 0.4.50 due to output-quality issues) |
| 0.4.47 | 09-19 | MiniMax H3 Director: upstream RefMod v5 bundle loading |
| 0.4.45 | 09-19 | Enhanced Video Combine: `%seed%` filename token via an optional seed input |
| 0.4.43 | 09-18 | MiniMax H3 Director: RefMod timeline visualization, missing-file resilience, upstream credit to ComfyUI-MiniMaxH3Mod |
| 0.4.42 | 09-18 | MiniMax H3 Director: RefMod preview translation, reference pack persistence, Insert RefMod # buttons |
| 0.4.41 | 09-18 | MiniMax H3 Director: lazy standalone RefMod references with self-contained loading and cache invalidation |
| 0.4.40 | 09-15 | Enhanced Video Combine: PyAV 18 migration, hardware-to-software fallback, one-second keyframes, and seekable cached browser previews |
| 0.4.39 | 09-15 | Registry security remediation: local-only LLM models, output-only FFmpeg preview, loopback-only Ollama, audit tooling |
| 0.4.38 | 09-15 | Registry security remediation: local-only LLM models, no remote model code, loopback-only Ollama, audit tooling | 
| 0.4.37 | 09-11 | MiniMax H3 Director: save/load packs, split media lanes, L2VA lock, VAE validation, and responsive toolbar |
| 0.4.36 | 09-10 | MiniMax H3 Director Guide: named REF2VA native-call inputs for Core-order compatibility |
| 0.4.35 | 09-08 | Advanced LoRA Loader: dual Civitai `.com`/`.red` links, `.red` mirror fallback, negative-lookup memoization, (i) panel layout rework |
| 0.4.33 | 09-04 | System Monitor chips: stable fixed-width formatting; fast disks switch to GB/s (#36) |
| 0.4.30 | 08-29 | Seed Control: Random-mode roll via the graphToPrompt choke point; panel DOM syncs after a run |
| 0.4.29 | 08-29 | Advanced LoRA Loader trash button (per-row, resets the slot to None) |
| 0.4.28 | 08-29 | LoRA info button in the Advanced LoRA Loader (Civitai link, trigger words, images) |
| 0.4.27 | 08-29 | Advanced LoRA Loader universal rename; PDD/ACC metadata passthrough; opt-in cache button |
| 0.4.26 | 08-29 | H3 Cache PDD head-bank + per-token mask support |
| 0.4.25 | 08-28 | Director v1 Image Inpaint mode; `inpaint_requested` output switch |
| 0.4.24 | 08-28 | Director 2.0 freeze (v2 archived to `frozen/`) |
| 0.4.23 | 08-26 | Seed Control node + Director seed panel revamp |
| 0.4.22 | 08-25 | LTX-2.3 loader inline value editor |
| 0.4.21 | 08-23 | RTX lazy output allocation; Director v2 preview combo widget (pre-freeze) |
| 0.4.19 | 08-17 | RTX Upscaler `use_mmap` disk fallback + auto model unload; wildcard library additions |
| 0.4.17 | 08-16 | H3 Director prompt serialization + legacy preservation; H3 patch-grid alignment; script import fix (#28) |
| 0.4.14 | 08-16 | H3 Director resolution panel, grouped dropdowns, crop preview; H3 Cache + Kitchen Attention nodes added |
| 0.4.13 | 08-15 | System Monitor chip sizing + container-safe probes; Video Combine audio counter fix; LoRA loader nullish fallback |
| 0.4.10 | 08-10 | REF2VA `imd` KeyError and packed stereo audio duration fixes (PR #15, #17); external prompt input (PR #21, 08-12) |
| 0.4.6 | 08-05 | H3 Director prompt-builder suite (builders, thumbnails, helper buttons) |
| 0.4.5 | 08-05 | MythicAlchemy v12 H3 workflow |
| 0.4.1 | 08-04 | H3 Director docs + changelog; license switch to GPL v3 |
| 0.3.8 | 08-03 | MiniMax H3 Director integrated; hardened video preview fallback |
| 0.2.17 | 07-21 | Animated image outputs (AVIF / WebP) |
| 0.2.12 | 07-17 | DaSiWa Torch Resize node |
| 0.2.9 | 07-16 | DaSiWa System Monitor node |

## Changelog

### MiniMax H3 Director (v1)

- **09-22 (0.4.54):** The character counter measures the final assembled prompt, including structured field headers, rather than only raw textarea text.

- **09-22:** **RefMod lane & prompt-mode fixes (0.4.52):** Fixed stale/duplicate RefMod numbering on fresh ComfyUI load — falls back to the refmod's saved `media_type` when library data isn't loaded yet, so tags are unique until REFMOD panel opens. Remove button in REFMOD overlay now calls `render()` after removal so the timeline updates immediately instead of requiring a lane re-click. Clear button (modebar) now disables all refmods in the overlay (`enabled = false`) and removes them from lanes. Mode-aware lane visibility: T2VA hides the reference grid entirely, non-reference modes (I2VA, L2VA, FL2VA, Image Inpaint) show a single-row Image lane only — no clutter from unused Video/Audio rows. Track height adapts to fit shown lanes (min-height 0 instead of fixed 280px). Simple/Structured prompt mode switching preserves content bidirectionally: Structured → Simple flattens fields into `field: value` lines; Simple → Structured parses those labels back into separate fields, and if no recognized labels are found the full text dumps into `detailed_description` (REF2VA) or `integrated_multimodal_description` (other modes). Added right-aligned character counter to all prompt builder forms (Base, Simple, REF2VA) that sums characters across all textareas and updates live on every keystroke. Refmod X button in the slotline now sets the refmod's `enabled = false` in the overlay (matching overlay behavior) instead of calling `remove()` which only works on regular media items. Strength changes in the REFMOD overlay reflect immediately in the timeline clip without requiring a lane re-click.

- **09-19:** RefMod image latent width and height now use the correct dimensions of 5D tensors.
- **09-19:** **Upstream RefMod v5 bundles (0.4.47):** The REF2VA overlay recognizes RefMod containers from ComfyUI-MiniMaxH3Mod as well as standalone image/video/audio files. It expands each `ref_N` member into its native reference; `<RefMod N>` resolves to all labels from that selected bundle in member order.

- **09-18:** **RefMod timeline visualization + upstream credit (0.4.43):** Enabled RefMods now render as read-only clips in their appropriate reference lane with a green REFMOD badge, slot number, and strength indicator — total reference count visible at a glance alongside uploaded media. Missing RefMod files warn-and-skip instead of hard-erroring, so stale workflow saves don't crash generation. **Upstream credit:** the saved person RefMod concept, `.safetensors` latent file format, and strength scaling design are based on [Luisacaotica/ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod); both packs can be installed side-by-side and share the same `models/refmods/` folder.

- **09-18:** **Standalone RefMods (0.4.41):** REF2VA provides a **REFMOD** button after **INPUT SCALING**. Its separate overlay does not resize the node, loads metadata only when opened, and explains the reference selector, strength, enable switch, workflow description, and prompt tag. It selects standalone image/video/audio RefMods recursively from `models/refmods/`. Workflow row names/descriptions remain authoritative, native reference numbering includes video soundtracks, changed files invalidate cached execution, disabled empty rows are ignored, and path resolution rejects traversal and symlink escapes. Bundle files remain intentionally out of scope.
- **09-18:** **RefMod preview translation & pack persistence (0.4.42):** Prompt Preview now resolves `<RefMod N>` tags to their native reference labels (`<Video 2>`, etc.) and appends a Reference descriptions block mapping each resolved tag to its description. Save/Load reference packs persist RefMod selections alongside media items — overwrite clears, append merges. The Clear button resets RefMod state. Insert RefMod # buttons added to all prompt builders (base, REF2VA, simple) for single-number insertion at cursor position.
- **09-11:** **Reference packs, lanes, and compatibility (0.4.37):** Save/Load packs preserve reference-file and prompt data independently, support append or overwrite, validate the saved target mode and missing files before applying, and preserve relative placement on mode remaps. REF2VA now separates Image, Video, and Audio lanes; V / A / V+A sources reserve and label their correct reference slots. L2VA locks the decorative slot 0 and uses slot 1 for its closing frame; legacy saved L2VA closing-frame layouts remain accepted. The Guide detects swapped H3 video/audio VAEs before native execution, while the toolbar wraps inside the node at narrow widths.
- **09-10:** **REF2VA native-call compatibility (0.4.36):** the Guide now binds every `MiniMaxH3ReferenceToVideo` input by name. The current Core prompt-before-VAE order was already correct; named binding preserves it and remains safe if Core reorders inputs later.
- **08-28:** **Director 2.0 frozen:** the v2 fork is removed from the nodepack and preserved under `frozen/`; Image Inpaint is documented as a v1 feature.
- **08-28:** **Image Inpaint mode** in v1: normalizer, 5-frame single-image guide, Guide-node conditioning + latent emission; `ref2va_requested` output renamed to `inpaint_requested`; request outputs now follow the mode widget; auto aspect resolves by timeline slot instead of insertion order.
- **08-26:** seed panel revamp: spinner column slimmed and aligned with the input.
- **08-19 / 08-21:** Director forked into independent v1 + v2 on 08-19; v2 gained a seed control panel, pill modals, external sampling/shift override sockets, and a socketless preview tiny-VAE combo widget (08-23); resized prompt-field heights now persist across re-renders and workflow reloads in both versions (08-21); verified dead code removed from both.
- **08-16:** Resolution/Aspect/Input-Scaling panel with external overwrite inputs; `frame_rate` FLOAT input + output (legacy `external_prompt` input dropped); paste-replace onto a selected tile; WAV `.wave` alias + RIFF duration fallback; ▶ Play crop button with a draggable preview range; Ctrl+Enter run shortcut preserved and timeline wheel forwarded to the canvas; resolution dropdowns grouped by orientation / ###p / MP with **Native (ShortEdge 768px / 2048px)** labels; `non_diegetic_music` starts empty (N/A only at assembly time); nodes re-registered under the **DaSiWa/MiniMax H3** category; canvases aligned to the H3 32-px patch grid; legacy pre-builder prompts migrated into a Simple-prompt builder state instead of being dropped; `emit()` now serializes the resolved prompt back to the `prompt` widget; fixed the double `../` in scripts/app.js + api.js imports (#28). Simple/Structured prompt-mode toggle (persisted, honored by Preview Prompt).
- **08-10:** external prompt input. (PR #21)
- **08-08:** packed stereo audio de-interleaved so stereo WAV references keep their real duration; integer PCM scaled by magnitude (a full-scale sample no longer trips the ~90 dB attenuation guard). (PR #17)
- **08-06:** fixed `KeyError 'imd'` on every REF2VA run; ref schema normalized for mixed v1/v2 workflows.
- **08-05:** prompt-builder suite: unified builder UI with mode-aware forms, REF2VA simplified to six free-text fields with v1→v2 backward-compat merge, **Insert [Shot N]** / **Prefill Labels & Summary** / **Preview Prompt** helper buttons, video thumbnail previews, single-line toolbar with Clear/Remove/?, dark-blue audio lane, and prompt-builder fields persisted across workflow reloads. Fixes: trim sliders only drag when grabbed, images drag out of locked L2VA slots, slot-capacity checks corrected, KeyError guards on `p2_shot`/`last_shot`, non-string builder values, textarea onChange receives the value instead of the Event.
- **08-04:** full refresh; helper/JS/test alignment; native generation-length alignment; VAE/CLIP naming corrected to the Comfy-Org repo; wiring diagram added and "Guide replaces native nodes" clarified in docs.
- **08-03:** initial integration of the timeline-based H3 Director and Guide nodes.
- **Earlier additions (pre-window):** embedded video audio extraction (a video can supply its own audio reference; V / A / V+A stream switch per video clip), standalone-audio trim support with waveform preview + draggable crop markers, attached soundtracks that share a video's trim window, FL2VA ↔ REF2VA mode-switch safety (incompatible references are preserved, not deleted), and hardened video duration detection with a container-level fallback.

### MiniMax H3 Cache & Patch Comfy Kitchen Attention

- **08-29:** **PDD compatibility:** the node detects the live `FinalLayer.forward` signature at patch time and passes the ComfyUI 0.34+ PDD sigma-schedule arguments, so the PDD LoRA head bank works with cache enabled. **Per-token masks:** honors `denoise_mask` / `audio_denoise_mask` exactly like Core — mixed masks run masked rows at their own strength via per-row `rows_to_mod_index` modulation; absent or uniform masks collapse to the scalar path, byte-identical to the previous behaviour. **Spectrum patch artifact:** `patches/comfyui-spectrum-minimax-h3-pdd.patch` for xmarre's ComfyUI-Spectrum-MiniMax-H3 v0.2.20, which silently degrades on the PDD signature.
- **08-16:** both nodes added: an approximate, model-scoped whole-block-stack residual cache (relative-L1 threshold sampling, 15–90% sampling window, bounded cache hits, auto/CUDA/CPU storage) and a one-input INT8-attention model patch; both are model-clone patches and chain in either order.

### Seed Control

- **08-26:** new standalone Seed Control node extracted from the Director seed panel: full 64-bit unsigned seeds, Random|Fixed segmented switch, spinner with hold-to-repeat, Last 10 seeds history, external override socket, and INT + NOISE outputs; lossless 64-bit display via decimal-string mirroring; mode, last seed, and history persist with the workflow.

### Advanced LoRA Loader (formerly LTX-2.3)

- **09-08:** **Civitai mirror links + (i) panel rework (0.4.35):** the (i) panel now always shows **both** Civitai mirror links — the `.com` link labeled `BLUE:` (blue) and the `.red` link labeled `RED:` (red); the labels are plain text, only the URL is clickable. Backend (`nodes/lora_info.py`): the by-hash lookup runs on `.com` first and **falls back to the `.red` mirror** on a 404 (verified byte-identical mirror, same `modelId`/`versionId`), and results are memoized **including misses** — a negative lookup no longer re-hits the Civitai API on every panel open (the stored error text is preserved in the cache entry; the panel's **Refresh** button forces a re-lookup via `refresh=true`). Panel layout: the file name moved to its own full-width row above the controls (long folder paths wrap independently and can no longer stretch the button row), and the sha256 hint sits on its own row directly under the name (9 px, `user-select:all`). The earlier `.com`/`.red` domain selector buttons and the `civitai_domain` localStorage/property machinery were removed with the rework — both links are always shown.
- **08-29:** **Trash button (0.4.29):** each slot row gained a trash button (plain canvas paths, ASCII-drawn trash-can, no emoji) placed directly right of the info button. Clicking it resets that slot's LoRA back to "None" (the STR / VIS / A multipliers are kept — exactly like picking "None" in the slot picker), so a LoRA can be unstacked without reopening the picker. The info button shifted slightly left (x 976 → 962) to make room; the trash button occupies the previous info x. UI-only, no backend change.
- **08-29:** **LoRA info button (0.4.28):** each slot row gained an ⓘ glyph at its right edge. It opens an info panel with the LoRA's Civitai link (SHA-256 looked up on Civitai's `model-versions/by-hash` API, results cached in `lorainfo/`), trigger/trained words from the safetensors header and Civitai (click-select, copy), and preview images (Civitai plus a local sidecar image next to the LoRA if present). Two new GET-only routes (`/dasiwa/ltx2/lorainfo`, `/dasiwa/ltx2/loraimg`) in `nodes/lora_info.py`; no new dependencies.
- **08-29:** **PDD/Acc LoRA head-bank guard (warn-only):** when a PDD LoRA (`pdd_num_steps`) is applied to a single-head H3 model, the loader prints a console warning — the incompatible `final_layer` head-bank `set`/`bias` keys are **kept**, so the protective core shape crash at `comfy/lora.py` still aborts the run (deliberate: no circumvention until an upstream patch lands). The guard warns only on positive evidence (bank width and model width both readable and different); a genuine PDD model whose width matches is untouched. **Recommended fix:** pair PDD LoRAs with a PDD model (`final_layer.video_out [3072,5376]`). Bugfix — no version bump.
- **08-29:** **universal rename + PDD/ACC support + opt-in cache:** the LTX-2-only `DaSiWa LTX-2 Master Loader` is renamed to the **Advanced LoRA Loader** — module/class/JS/docs/display name changed internally, but the *serialized* node ID `DaSiWa_LTX2LoraLoader` and display name stay stable for saved workflows (display now drops the "DaSiWa" prefix). The file is read with `return_metadata=True` and the PDD/ACC metadata is forwarded to Core's `load_lora_for_models`, so PDD / ACC LoRA head banks activate (older builds fall back via `TypeError`). A new **⚡ CACHE** button in the control strip enables an opt-in per-path LRU cache (max 4 entries, **off by default**), so a LoRA reused across slots is read once.
- **08-25:** **inline value editor:** the STR, VIS, and AUDIO pills open an in-canvas editor instead of a `prompt()` dialog; the editor is pinned to its pill and tracks pan and zoom; documented.
- **08-15:** STR / VIS / A value editors no longer bounce off 0 (nullish fallback).
- **07-11:** LoRA list refreshes dynamically.

### RTX Upscaler & Refiner

- **08-23:** output batch now allocated lazily (kernel-decided, no up-front memory pressure); `use_mmap` **off by default**.
- **08-17:** **`use_mmap` disk-backed fallback** as a permission switch (VRAM → RAM → disk chain) plus default-on `auto_unload_models`.
- **08-14:** `empty_cache` made optional to restore API workflow compatibility (fixes #24, PR #26).
- **08-06:** VRAM-aware output allocation; optional `empty_cache` switch; fixed `use_mmap` swap bloat.
- **08-01:** adaptive CPU RAM reserve shared with the Watermark compositor.

### Enhanced Video Combine

- **09-19:** Video outputs are no longer advertised as `ui["images"]` assets; image-only consumers no longer receive video entries.

- **09-19:** **`%seed%` output naming (0.4.45):** An optional `seed` input expands `%seed%` in `filename_prefix`, including the corresponding first/last-frame export names. Unconnected inputs preserve the literal token for existing workflows.
- **09-15:** **PyAV-native encoding and seekable previews (0.4.40):** all video/audio encoding, metadata, animated WebP/AVIF, and compatibility preview transcoding moved from external FFmpeg processes to PyAV 18. Hardware encoder candidates are runtime-tested and fall back through NVENC → QSV → AMF → VAAPI → software. Outputs receive explicit one-second keyframes; MP4 uses fast-start metadata. AV1, VP9, HEVC, 10-bit, and other compatibility previews are cached as H.264/AAC MP4 and served with byte-range support for reliable browser scrubbing, while downloads remain the untouched original codec/container.
- **08-22 / 08-25:** preview checkboxes (Autoplay, Mute) persist across reloads (PR #30); permanent Mute checkbox persisted with node properties.
- **08-21:** drifted combo/boolean widget values self-heal on load; audio_codec positional drift repair; MythicAlchemy v16 workflow with clean video-combine widgets.
- **08-15:** fixed audio outputs overwriting each other (counter always reset to 1).
- **08-03:** browser-compatible AV1 auto-encoding; Auto codec precedence documented (AV1 → VP9 → H.264; H.265 excluded from Auto).
- **07-22:** video frames streamed to FFmpeg instead of temp files.
- **07-21:** animated image outputs: Animated AVIF (GPU AV1 or software) and Animated WebP, with accelerated AVIF encoding.
- **07-19:** node added (IMAGE batch → video with optional AUDIO muxing and in-node preview); frame exports published to ComfyUI Assets; re-encode on queued runs; audio settings preserved in saved workflows.

### Wildcard & Preset Prompt Builder

- **08-17:** library additions: Background, Intimate Backgrounds, Weapons & Items; duplicate entries merged.
- **07-31:** segment picker interface; picker layout restored after workflow reload.
- **07-30:** node added: dual Booru / Natural-Language wildcard library, weighted bounded prompts, reproducible rerolls.

### LLM / VLM Analyze

- **09-15: Registry security remediation (0.4.38–0.4.39):** removed workflow-controlled Hugging Face repository downloads, Hugging Face token reads, and `trust_remote_code`; local model folders must be installed before a workflow runs. Ollama requests are fixed to `http://127.0.0.1:11434/api/chat`; the former arbitrary URL input is removed. Enhanced Video Combine preview accepts only a real file under ComfyUI's output directory, rather than selecting a root from a request parameter. `tools/audit_comfy_registry_status.py` redacts Registry status payloads and verifies the exact post-publication version becomes active.
- **07-30:** LLM cache and GGUF backends added (local GGUF via llama.cpp alongside Ollama and Hugging Face download).

### DaSiWa System Monitor

- **09-23 (0.4.55):** Separate Free Memory toolbar button with a DaSiWa logo, independent visibility setting, and VRAM versus system-RAM/cache actions via ComfyUI's `/free` route. The controls and meter chips align at 36px height; both menus close on outside clicks. [Details →](system_monitor.md#free-memory-toolbar-button)
- **09-22 (0.4.53):** Full panel and settings menu escape toolbar clipping; Full mode restores its meter colors and proportional fill bars.
- **09-19:** Disk polling skips stale mountpoints that raise `OSError`.

- **09-04:** stable chip width — values render in a fixed-width monospace cell (percent/°C pad to 3 digits, disk throughput to one decimal), so the chip width no longer shifts as digits tick; values ≥ 1000 MB/s switch to `x.x GB/s` so fast (M2-class) disks don't overflow the chip (issue #36).
- **08-28:** Windows CIM probe now cached — no more per-second PowerShell spawns.
- **08-15:** content-sized chips (no more LITE text clipping at 4K); real disable flag + container-safe probes (`DASWA_SYSTEM_MONITOR=0`).
- **08-01:** disk telemetry + docking.
- **07-16:** node added; configurable Lite/Full display; dockable placement.

### Torch Resize

- **07-17:** node added: Lanczos batch-aware resizing on the PyTorch build ComfyUI already uses; README section rewritten as user-facing benefits.

### Resolution Scale Calculator

- **07-16:** matches the ComfyUI-native MP convention (1024²) (#4); new 1 MP resolution preset.

### Node Status Switch

- **07-31:** promoted boolean inputs now sync live.

### Metadata Image Saver

- **07-31:** skips cleanly when no image is received.

### Repository & tooling

- **08-23 / 08-26:** DaSiWa node list in the bug-report issue form synced.
- **08-19:** `.projectatlas/` build artifacts gitignored.
- **08-17:** node dropdown auto-synced from `NODE_DISPLAY_NAME_MAPPINGS`.
- **08-16:** bug-report + feature-request issue forms added; unregistered System Monitor entry dropped from the node dropdown.
- **08-04:** **license switched from Apache 2.0 to GPL v3.**
- **08-01:** node output logging centralized; modules renamed; tests folder moved behind dot-prefix.
- **07-19:** local Hermes state ignored; watermark assets + implementation plans documented.
- **07-05:** supports the new ComfyUI `validate_inputs` signature.
