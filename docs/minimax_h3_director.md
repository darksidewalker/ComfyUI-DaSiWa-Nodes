# MiniMaxH3 Director

Build an H3 video in one place: arrange references on a timeline, trim them, write a prompt, and choose a canvas. **Director** prepares the inputs; **Director Guide** checks them and calls ComfyUI's native H3 nodes. You do not wire the native image-to-video or reference-to-video nodes separately.

[Collection changelog](news_and_changelog.md#minimax-h3-director-v1) · [Continuity guide](h3_continuity.md)

## Start here

1. Install the pack's dependencies (`pip install -r requirements.txt` in ComfyUI's Python environment), restart ComfyUI, and use a ComfyUI build with native MiniMax H3 support.
2. Add **MiniMaxH3 Director** and **MiniMaxH3 Director Guide** from `DaSiWa/MiniMax H3`.
3. Pick a mode, add media to the appropriate lane, set Duration and canvas, and write a prompt. Connect the matching model to Director, and CLIP and visual VAE to Guide. REF2VA also needs an audio VAE.
4. Connect Guide's conditioning and latent to your normal sampler/decode/export path. The Director forwards the selected model to that sampler path.

```mermaid
flowchart LR
    M["H3 model loader<br>optional LoRA or model patchers"] -->|MODEL| D["Director<br>timeline + prompt"]
    D -->|guide| G["Director Guide<br>native H3 routing"]
    C["H3 CLIP loader"] -->|clip| G
    V["H3 visual VAE loader"] -->|vae| G
    A["H3 audio VAE loader<br>REF2VA + video import"] -.->|audio_vae| G
    D -->|model| S["Sampler"]
    G -->|positive + latent| S
    S --> X["Decode → export"]
```

| Requirement | Where it goes | When needed |
| --- | --- | --- |
| H3 diffusion `MODEL` | Director `fl2va_model` | T2VA, I2VA, L2VA, FL2VA, Image Inpaint |
| H3 diffusion `MODEL` | Director `ref2va_model` | REF2VA |
| H3 text encoder / `CLIP` | Guide `clip` | Every mode |
| H3 visual `VAE` | Guide `vae` | Every mode |
| H3 audio `VAE` | Guide `audio_vae` | REF2VA; also importing an ordinary video for continuity, even if silent |
| Optional RefMod files | `ComfyUI/models/refmods/` | REF2VA saved references; no other node pack required to read them |
| Optional Prompt Forge model | `ComfyUI/models/llm/`, Ollama, or OpenAI-compatible server | Only for AI-assisted prompt drafts |
| Optional continuity nodes | After sampling and after export | Only to save checkpoints or extend a source; see [wiring](h3_continuity.md#wiring) |

Only the active Director model socket is requested (lazy loading). A LoRA loader, MiniMax H3 Cache, attention patcher, or model preview override may feed that socket **before** Director. Do not loop Director's model output into its own upstream loader. External width and height overwrite sockets must both be connected; they override canvas calculations and input scaling. `external_prompt_overwrite` accepts a nonempty external prompt for ordinary generations. `frame_rate` defaults to 24, accepts 0.1–240, and is also an output; continuity uses its own native timing constraints.

## Choose a mode

| Mode | Use it for | References |
| --- | --- | --- |
| T2VA | Text to video | None |
| I2VA | Start from an image | First-frame image |
| L2VA | Finish at an image | Last-frame image |
| FL2VA | Start and/or finish at an image | Up to two endpoint images; zero works as text-only |
| REF2VA | Borrow identity, style, motion, composition or sound | Up to 9 images, 3 videos, 3 audio clips; 12 files total |
| Image Inpaint | Edit/refine a still | Exactly one image, no video/audio references; native five-frame pass yields one output frame (use Get Image from Batch). Optional Enhanced Upscale goes before decode/extraction. |

Endpoint modes accept images only. Switching modes hides incompatible timeline items rather than deleting them; switching back restores them. T2VA does not use those hidden references. REF2VA audio needs at least one visual reference. Each reference video/audio crop must be 2–15 seconds; combined visual video length and combined audio length must each stay within 15 seconds. The Director reports violations in its node status before native execution.

## Work in the timeline

Choose the Image, Video or Audio lane, then use **+**, drag/drop or paste (Ctrl+V). Media is saved under ComfyUI's `input/` folder. Click a tile to edit its description, trim or remove it; drag to reorder. Pasting onto a selected tile replaces it without moving its slot. Video tiles show a first-frame thumbnail, and audio tiles show a waveform. Video and audio crops have draggable endpoints and a crop-play preview.

Audio and video previews show a source-time crop track, live playback line and white draggable cursor grip. Audio also shows the full-file amplitude waveform, with the selected crop highlighted and the rest dimmed. Drag either endpoint to resize the crop, or drag inside it to move the whole selection. A live playhead and source-time readout follow both **Play crop** and the native media player. Drag the white grip at the top of the playhead to choose a playback position without changing the crop; the native player’s Play button starts there. Arrow keys move the focused grip by 1 second, Shift+Arrow by 0.1 seconds. Scrubbing leaves crop-only playback and returns to the full-file player. **Play crop** always starts at the crop start and stops at its end, waiting for media readiness and the seek to complete. Audio/video sources that refuse HTTP seeking use a temporary local Blob instead; this needs the compressed file in browser memory until the preview closes. Closing the preview stops playback. Existing timeline waveform data is reused; otherwise the browser decodes the file once for that preview, retaining only compact amplitude peaks afterward. Long files need additional temporary browser memory. If waveform decoding fails, audio playback and crop controls remain available.

For a REF2VA video tile, choose **V** (frames), **A** (embedded sound), or **V+A** (both). An attached soundtrack shares its video's trim. Audio gets its own `<Audio N>` label; image, video and audio labels are numbered within their type in timeline order. Use those labels consistently in your prompt.

**Load / Save** handles reference files, prompts, or both as a pack, with append/overwrite choices. Loading checks mode limits and missing files first. **Clear** removes the timeline, prompt and Forge draft history. The toolbar **Remove** acts on the selected tile.

### Canvas and input scaling

The Resolution panel offers Auto or custom aspect ratio and resolution. Auto aspect follows the first image/video; Auto resolution uses a 768-pixel short side. Presets snap to H3's 16-pixel grid. Input scaling affects visual references, not audio:

| Setting | Effect |
| --- | --- |
| Off | Keep the input tensor as-is |
| Auto | Downscale only if needed to a 2048-pixel short side; never enlarge smaller inputs |
| Target - Selected Aspect & Resolution | Stretch to the selected canvas |
| Fit | Preserve source aspect inside the canvas box, with dimensions divisible by 16; output may be smaller than the canvas, without padding or cropping |
| Fill and crop | Preserve source aspect, scale to cover the canvas and center-crop excess edges; output fills the canvas |
| Fit and pad | Preserve source aspect, fit inside the canvas and center with black padding; output fills the canvas without removing source edges |
| Long side with divisible crop | Match the source long side to the corresponding canvas dimension, then center-crop the other side to fit the box and the 16-pixel grid; output may be smaller than the canvas |

## Write a prompt

There is one editable prompt field. Write freely, or click **Insert Prompt Structure** for mode-appropriate H3 headings. **Simple / Structured** controls how older builder content is assembled and is saved with the workflow. **Insert [Shot N]** adds a shot marker. In REF2VA, **Prefill Labels & Summary** fills reference labels and descriptions without inventing visual/audio details; check its proposed roles and finish the actual action yourself.

```mermaid
flowchart LR
    R["Timeline references<br>labels + descriptions"] --> P["Your prompt<br>or inserted structure"]
    F["Prompt Forge<br>optional draft"] -->|"Apply to node only"| P
    P --> G["Guide → native H3 conditioning"]
```

Use `[Shot 1]` for the start and `[Shot 2] At 00:04.500` for a later cut (target-video time, not source time). For endpoint images, state how `<Picture 1>` at 0 seconds and, if present, `<Picture 2>` at the end align with the target video. The base structure uses `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music`.

REF2VA's optional structure has six headings: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`. Use `<Subject N>` for who/what appears, `<Picture N>` for concrete frame anchors, `<Video N>` for structural pacing/editing, and `<Audio N>` for sound references. Describe transferred appearances or motion under Subjects. Retention terms include `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference` (visual) and `fully_copy`, `partially_copy`, `reference`, `weak_reference` (audio). For example, distinguish “At 00:05.000 in the target video” from “near 00:02.400 in `<Video 1>`.”

For more prompt syntax, see MiniMax's [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) and [full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

### Prompt Forge (optional)

Click **Prompt Forge**, enter an **Idea** (or **Next action** during continuity), choose a model, **Creativity** and **Detail** (1–10), then **Generate**. Review the draft and explicitly click **Apply to node**; generating alone does not change your prompt or queue a video. **Regenerate** makes another draft; Cancel/close stops an active request. The last three successful drafts are saved in this node's workflow properties, not reference packs. Choose one from history to preview and explicitly reapply it, even after the node prompt has been edited or another draft applied. This replaces prompt text only; a changed source, references, mode, Duration, canvas or model connections still makes the draft incompatible. Changes after selecting it are checked again on Apply. A draft for another mode must be applied in that mode. **Clear history** leaves the applied prompt intact; Director **Clear** removes both.

**Canvas:** Forge receives the Director's current output width and height and derives the exact aspect ratio from those dimensions, not from reference images. Director dimensions take priority over conflicting format requests in the Idea. When external width/height overrides are connected, their runtime values are unknown to Forge, so it is instructed to omit aspect ratio and resolution rather than guess. Canvas changes invalidate existing drafts.

**Detail and Creativity:** Detail increases descriptive precision, not shot count or additional events. Low Creativity uses one continuous shot unless the Idea explicitly requests multiple shots. Medium Creativity permits small plausible surprises and only clearly motivated cuts; high Creativity permits unexpected events and optional creative cuts, not a cutting quota. Explicit single-shot/no-cut or multi-shot requests take priority. FL2VA still requires an explicit request for multiple shots; continuity remains uninterrupted at every Creativity level. These are model instructions, not a guarantee that every model follows them.

**Shots:** choose **Auto** or exactly **1–5** shots for a new clip. A selected count overrides shot counts in the Idea, Detail and guide examples. Each numbered shot has an optional description box; filled boxes join the Idea as `Shot N: ...`, and the boxes alone are sufficient to generate. Auto ignores these boxes; reducing the count excludes later boxes without erasing their session text. Successful drafts save their count and descriptions in history. Shots is hidden during continuity, which remains uninterrupted. A mismatched generated count produces a warning. Invalid cut timestamps at/past the clip end or before the previous cut are redistributed within the available gap; shot text is unchanged.

**Picture labels and People boxes:** REF2VA pictures can combine Character, Place, Style and frame roles; Pose and Custom remain exclusive. Character numbers identify the same person across pictures. Who buttons start at six and grow up to 32. For several people, select an optional left/right, top/bottom or front/back order, or leave the order unset. In REF2VA character pictures and endpoint-mode frames, **Find people** asks the selected vision model for boxes; **Add character** creates a manual box. Drag or resize boxes, change their numbers (an occupied number swaps), remove one or clear them all. Boxes determine identity and left-to-right placement on that picture, replacing its Who/order controls. Pose's **Who takes it** selects target characters, never the reference person's identity. FL2VA's slots remain first/last-frame anchors. Labels, notes and boxes travel with workflows and reference-file packs.

**Cast and scene:** editable cards describe each character; labelled REF2VA also provides Place, Style and one card per Pose picture. **Read Cast automatically** is on by default, with a browser-saved opt-out. Opening Forge reads empty cards with the selected model; unedited cards refresh after relevant picture, box, note, Idea or Shot changes. **Scan people / Scan picture** explicitly rereads cards even with automatic reading off. Edited cards are preserved and marked out of date instead of overwritten. Character crops are used when boxes isolate a person; heavily overlapping boxes use the whole picture and position instead. Cards are saved in node properties, including embedded PNG workflows, not reference-file packs. Director Clear removes them.

**Card priority:** cards supply reference appearance. Explicit changes in Idea or Shot instructions override conflicting card traits, retaining everything else. Carded labelled REF2VA subjects use their card in definitions/retention; endpoint modes send cards alongside spatial placement. Cards do not introduce new native conditioning or replace reference pixels. Review the result before Apply; model compliance is not guaranteed.

Automatic cards whose inputs changed are omitted from Generate until reread; edited cards remain available. Pending reads cannot overwrite renumbered identities or newer box/role edits, and Generate waits for Find people. The reader inherits the existing adult-only prompt policy; generated or edited cards containing explicit minor/age-reducing terms are rejected with an instruction to describe an adult. Warm-model cleanup remains active when draft preparation fails and replaces old request ownership instead of accumulating backend objects.

**No vision:** a known non-vision reader is refused before image loading. An OpenAI-compatible server has no standard capability endpoint, so its first image call may be the capability test. A no-vision response stops further automatic cards until another model is selected; empty cards fall back to the existing draft path. Manual cards and boxes remain usable, and Forge does not switch models automatically. Automatic Cast reading is independent of **Let the model see the pictures**, which controls only the subsequent draft's images.

For new image-only labelled REF2VA drafts, the prompt model receives the cast and instructions, **not image pixels by default**. Enable **Let the model see the pictures** to also send them in label order. Vision descriptions join each subject’s retention line or the pose definition; missing descriptions produce a warning. Small models may mix up identities with pictures, so labels-only remains the default. A non-vision model falls back to labels. Toggling the choice invalidates the current draft; saved drafts retain it and older drafts remain blind. Mixed audio/video references or saved unlabelled references use the full REF2VA path, retaining their definitions and retention; vision-capable writers can then receive timeline pictures. Code writes subject definitions and retention from the labels and generated shots; the video model still consumes the references. Review identities, acting and retention before Apply. Pose transfers posture only. A labelled Custom picture needs no note (the idea says what it is for) and gets its own Cast card describing only that part, object, action or detail; an unlabelled custom reference still requires instructions. Forge preserves reference appearance instead of inventing a visual medium. An English keyword heuristic warns about possibly unsolicited music but never deletes a score requested in another language or with an unlisted instrument. Repetitive output may be rejected as runaway text; the raw draft remains available for review.

| Forge source | Setup | Notes |
| --- | --- | --- |
| Local ComfyUI model | Chat-model directory or GGUF under `models/llm/` | `local:`; GGUF needs `llama-cpp-python`. Vision GGUF also needs its matching `mmproj` in the same model folder. Bare `.safetensors` is not a chat model. |
| Ollama | Running Ollama server with a downloaded chat/vision model | `ollama:`; defaults to `http://127.0.0.1:11434` |
| OpenAI-compatible | Server with `/v1/models` and streaming `/v1/chat/completions` | `openai:`; configure its base URL to enable it |

Set remote URLs in **ComfyUI Settings → DaSiWa → LLM servers** (Ollama address or OpenAI-compatible server address); set the OpenAI-compatible API key there if required. Reopen Forge after changing a server or local model. The Director does not download models or launch servers. In-process local models unload after every call. Cast/People reads may keep server models warm across calls, then request unload after 30 seconds idle, when Forge closes, or before the Director runs; actual unload depends on the server. Drafting still unloads after use. Forge refuses model work while a workflow is running. For GGUF installation/build alternatives, see [LLM nodes](llm_nodes.md).

For local GGUF and Ollama drafts, Forge adds 1,820 context tokens per attached picture, rounding up to 256: the default 16,384 becomes 18,432 for one picture or 32,768 for nine. Continuity uses the same rule, counting both references and source-tail frames. Larger contexts need more memory. Transformers and OpenAI-compatible servers retain their own context limits. Token estimates include the requested reply budget but only warn; they are not exact tokenizer counts. A recognized backend context overflow is reported without retrying blindly, and local out-of-memory errors remain distinct.

Forge uses the selected mode, Duration and reference instructions. REF2VA image labels also retain the compatible subject/style/keyframe/pose/custom role metadata used by continuity. Choose **pose** to transfer posture without automatically copying the reference person's identity, clothing or background. Use the optional **Reference instructions** box to specify exactly what to transfer; **custom** requires a description. **Keep / ignore** is an optional collapsed detail. These choices persist with the workflow and travel with saved reference-file packs. An **all** pack also carries verified media mappings; overwrite-loading it rebuilds them against the new timeline IDs and remaps surviving links automatically. Missing media links are removed with a notice. Legacy/manual and prompt-only packs retain their authored text, but their links are not silently treated as verified. Appending prompt text or replacing files alone invalidates ambiguous old links and reports this in the status. RefMod selections and settings also survive overwrite-loading; native links to RefMod members still require destination-library review, while `<RefMod N>` aliases remain intact. Only subject references join subject groups. First/last-frame roles in endpoint modes remain actual frame anchors.

For example, select **pose** and write “Use only the raised arms and stance; keep the target character's face, clothes and proportions.” Or select **custom** and write “Use the room layout and light direction, not the people.” In endpoint modes and continuity, vision models can see attached timeline pictures; new labelled REF2VA drafts use labels and instructions unless picture vision is explicitly enabled. Text-only models never see pixels. Saved RefMods contribute their descriptions, not decoded pictures; video and audio reference content is not analyzed. Prompt guidance is not dedicated skeleton/pose conditioning and does not guarantee a matching pose. During continuity, Forge also receives source-tail context and the next action; a changed source, Duration, mode or prompt invalidates an old draft. Applying only edits the continuation prompt.

### RefMods (REF2VA)

Put standalone or upstream v5 bundle `.safetensors` files in `ComfyUI/models/refmods/` (subfolders work). Open **REFMOD** beside Input Scaling, select a file, enable its row, adjust strength (0–1) and edit its description. Up to eight RefMod slots are supported, subject to REF2VA's final reference limits. The overlay is separate from the node; files load when opened. Workflow descriptions override embedded ones. An optional [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) installation can create compatible files, but is not needed for playback. This pack scales latents directly by strength rather than using upstream blur-mix behavior.

**Insert RefMod #** writes its expanded native `<Picture N>`, `<Video N>`, or `<Audio N>` label and description into the prompt. Saved `<RefMod N>` aliases still resolve at queue time. A v5 bundle expands to all members in order. Disabled/zero-strength rows contribute no reference; an unresolved alias is an error. Save/Load reference packs also preserve RefMod selection.

### Create a RefMod

To edit a saved archive, select it in **Saved RefMod**; it loads automatically and the action changes to **Save Changes** (saves a new copy, never overwrites). Choose **Select a saved RefMod…** again to restore the normal **Create RefMod** view with the Director media lines. Loading replaces the dialog’s export selection, not the Director timeline. Supported older standalone RefMods and v5 bundles expose their individual members: edit descriptions, uncheck unwanted members, and change the overall description or concept type. Save under a new name (or leave it empty for an automatic name). Original files and retained latent values remain unchanged; no VAE is required. Saved members have no original-media preview here. Pixel content and encoded crops cannot be edited in this metadata-only path.

Click **Create RefMod** beside **Save**. The separate export overlay lists inserted images, video/audio crops and populated saved RefMod slots; disabled and mode-hidden references are initially included too. Uncheck anything you do not want to save. Preview media, edit each reference description, and enter an optional filename, overall description and concept type. These texts are metadata, not learned trigger words. Concept type is descriptive; it does not select a different learning algorithm.

**Create RefMod** queues only reference encoding and saving, not the Director generation, CLIP, diffusion model or sampler. Raw images/videos need the H3 video VAE; raw audio needs the H3 audio VAE. The button uses the native VAE loaders connected directly to one Director Guide. For a different loader topology, use the separate **MiniMaxH3 Create RefMod** output node. Packing existing saved RefMods requires no VAE and preserves their unweighted tensors.

The exporter keeps independent Full Reference VAE latents in a single upstream-compatible v5 `.safetensors` bundle in the first configured RefMod root. It preserves selected crops and video V/A/V+A choices; video plus audio become separate members. Reference pixels keep their source aspect rather than baking the generation canvas into the file. Long/invalid crops and unsafe decoder estimates fail clearly; native H3 frame-grid adjustments are reported. Existing filenames are never overwritten.

The result appears in the RefMod library but is not automatically enabled beside its raw sources. Select it explicitly for later REF2VA use and avoid duplicate conditioning. A bundle’s storage capacity does not increase generation reference limits. RefMods do not guarantee identity isolation, voice cloning or audiovisual synchronization. Closing the overlay stops previews and monitoring, not an already submitted server job; use ComfyUI queue controls to manage that job.

The exporter uses the unique `DaSiWaH3RefModExport` node ID and namespaced frontend controls. It does not replace or patch another RefMod pack.

## Continue an existing video (optional)

Capture of new takes is **off by default**; enable **∞ Save new takes** before the initial generation if you want a checkpoint. Older active Continuity workflows migrate their saved added-frame count to Duration once, retaining next-action text and structured prompt definitions without moving action text into music. Later Duration edits remain authoritative. Save the workflow to retain migrated settings. [Older-workflow migration →](h3_continuity.md#loading-older-workflows)

Choose a completed checkpoint or ordinary video using **Choose start video…**. **Continuity Active** appears automatically; Duration now means *new* seconds, not total length. The source stays pinned until you change or clear it. **∞ Save new takes** saves fresh generations as checkpoints; continuations are saved regardless. Add **H3 Continuity • Append & Stage** after sampling and **H3 Continuity • Publish Export** after your actual exporter to publish only successful exports.

To save VRAM on long chains, choose **Advanced → Output → New segment only** instead: Segment Prepare/Decode then export only the newly added section, and **Join clips…** later assembles the chain without loading a model. The same dialog also joins ordinary videos, optionally with H3-generated transitions between them. [Options overview →](h3_continuity.md#which-option-to-use)

```mermaid
flowchart LR
    G["Director Guide"] -->|positive + latent| S["Sampler"]
    G -->|continuity_context| A["Append & Stage"]
    S -->|sampled| A
    A -->|cumulative_latent| E["Decode → exporter"]
    A -->|ticket| P["Publish Export"]
    E -->|filename| P
```

**Advanced → Keep REF2VA timeline references** is **off by default**. Continuation normally skips both endpoint-frame anchors and REF2VA timeline media so the pinned source tail drives the next segment. Turn this on if you intentionally want REF2VA image/video/audio references (and enabled RefMods) included during the continuation; it does not erase or change the saved timeline. The same switch is available directly in Forge as **Include timeline references**. When enabled, Forge sees the current reference instructions and a vision model also sees timeline pictures, separately from the chronological source-tail frames. **Structured REF2VA draft** is then selected automatically: existing definitions provide draft context and all six sections survive runtime continuation assembly. Template creation is in the Director's **Insert Prompt Structure** button during Continuity, not Forge: it retains the next-action text and carries existing definitions into the same editable prompt. New reference media still need their subject definitions authored or reviewed; Keep references alone does not describe a new character or scene. Video/audio reference content and RefMod pixels are not analyzed. For other modes, endpoint anchors remain skipped even when the option is on.

Advanced also provides preferred context, session resume/New session, refresh, and **Match source settings** for checkpoint mismatches. The selected source and settings live in the saved workflow; checkpoint files live in ComfyUI output. Ordinary-video import requires both H3 VAEs, including for silent video. See [H3 Continuity](h3_continuity.md) for exact wiring, duration rounding, limits, session behavior and resource costs.
