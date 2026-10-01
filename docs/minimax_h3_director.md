# MiniMax H3 Director

Build an H3 video in one place: arrange references on a timeline, trim them, write a prompt, and choose a canvas. **Director** prepares the inputs; **Director Guide** checks them and calls ComfyUI's native H3 nodes. You do not wire the native image-to-video or reference-to-video nodes separately.

[Collection changelog](news_and_changelog.md#minimax-h3-director-v1) · [Continuity guide](h3_continuity.md)

## Start here

1. Install the pack's dependencies (`pip install -r requirements.txt` in ComfyUI's Python environment), restart ComfyUI, and use a ComfyUI build with native MiniMax H3 support.
2. Add **MiniMax H3 Director** and **MiniMax H3 Director Guide** from `DaSiWa/MiniMax H3`.
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
| Image Inpaint | Edit/refine a still | Exactly one image, no video/audio; native five-frame pass yields one output frame (use Get Image from Batch) |

Endpoint modes accept images only. Switching modes hides incompatible timeline items rather than deleting them; switching back restores them. T2VA does not use those hidden references. REF2VA audio needs at least one visual reference. Each reference video/audio crop must be 2–15 seconds; combined visual video length and combined audio length must each stay within 15 seconds. The Director reports violations in its node status before native execution.

## Work in the timeline

Choose the Image, Video or Audio lane, then use **+**, drag/drop or paste (Ctrl+V). Media is saved under ComfyUI's `input/` folder. Click a tile to edit its description, trim or remove it; drag to reorder. Pasting onto a selected tile replaces it without moving its slot. Video tiles show a first-frame thumbnail, and audio tiles show a waveform. Video and audio crops have draggable endpoints and a crop-play preview.

For a REF2VA video tile, choose **V** (frames), **A** (embedded sound), or **V+A** (both). An attached soundtrack shares its video's trim. Audio gets its own `<Audio N>` label; image, video and audio labels are numbered within their type in timeline order. Use those labels consistently in your prompt.

**Load / Save** handles reference files, prompts, or both as a pack, with append/overwrite choices. Loading checks mode limits and missing files first. **Clear** removes the timeline, prompt and Forge draft history. The toolbar **Remove** acts on the selected tile.

### Canvas and input scaling

The Resolution panel offers Auto or custom aspect ratio and resolution. Auto aspect follows the first image/video; Auto resolution uses a 768-pixel short side. Presets snap to H3's 16-pixel grid. Input scaling affects visual references, not audio:

| Setting | Effect |
| --- | --- |
| Off | Keep the input tensor as-is |
| Auto | Downscale only if needed to a 2048-pixel short side; never enlarge smaller inputs |
| Target - Selected Aspect & Resolution | Stretch to the selected canvas |
| Fit / Fill and crop / Fit and pad / Long side with divisible crop | Use the corresponding Torch Resize behavior against the canvas |

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

Click **Prompt Forge**, enter an **Idea** (or **Next action** during continuity), choose a model, **Creativity** and **Detail** (1–10), then **Generate**. Review the draft and explicitly click **Apply to node**; generating alone does not change your prompt or queue a video. **Regenerate** makes another draft; Cancel/close stops an active request. The last three successful drafts are saved in this node's workflow properties, not reference packs. Choose one from history to preview it. A draft for another mode must be applied in that mode. **Clear history** leaves the applied prompt intact; Director **Clear** removes both.

**Canvas:** Forge receives the Director's current output width and height and derives the exact aspect ratio from those dimensions, not from reference images. Director dimensions take priority over conflicting format requests in the Idea. When external width/height overrides are connected, their runtime values are unknown to Forge, so it is instructed to omit aspect ratio and resolution rather than guess. Canvas changes invalidate existing drafts.

**Detail and Creativity:** Detail increases descriptive precision, not shot count or additional events. Low Creativity uses one continuous shot unless the Idea explicitly requests multiple shots. Medium Creativity permits small plausible surprises and only clearly motivated cuts; high Creativity permits unexpected events and optional creative cuts, not a cutting quota. Explicit single-shot/no-cut or multi-shot requests take priority. FL2VA still requires an explicit request for multiple shots; continuity remains uninterrupted at every Creativity level. These are model instructions, not a guarantee that every model follows them.

**Subject-aware grouping** (optional, REF2VA): assign the same **Group A/B/…** in the reference rows to two or more pictures of the *same* person or object, then Generate. Forge sends those pictures as one subject reference line, including each picture's keep/drop notes. Leave **Separate** for unrelated pictures. Style and keyframe references cannot join a subject group; a letter assigned to just one picture has no effect. The selected groups are shown below the reference list and saved with the node. This is especially useful with smaller prompt models, but does not depend on model size or guess intent from the Idea. After generation, Forge warns if distinct `<Subject N>` definitions explicitly cite different pictures from one group; it cannot reliably judge identity from free-form text, so review the draft before applying.

| Forge source | Setup | Notes |
| --- | --- | --- |
| Local ComfyUI model | Chat-model directory or GGUF under `models/llm/` | `local:`; GGUF needs `llama-cpp-python`. Vision GGUF also needs its matching `mmproj` in the same model folder. Bare `.safetensors` is not a chat model. |
| Ollama | Running Ollama server with a downloaded chat/vision model | `ollama:`; defaults to `http://127.0.0.1:11434` |
| OpenAI-compatible | Server with `/v1/models` and streaming `/v1/chat/completions` | `openai:`; configure its base URL to enable it |

Set remote URLs in **ComfyUI Settings → DaSiWa → H3 Forge** (Ollama address or OpenAI-compatible server address); set the OpenAI-compatible API key there if required. Reopen Forge after changing a server or local model. The Director does not download models or launch servers. Local models unload after use; Ollama normally unloads too, while other servers may retain VRAM. Forge refuses a new generation while the workflow samples. For GGUF installation/build alternatives, see [LLM nodes](llm_nodes.md).

Forge uses the selected mode, Duration and reference instructions. REF2VA images can be marked **subject**, **style**, **keyframe**, **pose**, or **custom**. Choose **pose** to transfer posture without automatically copying the reference person's identity, clothing or background. Use the optional **Reference instructions** box to specify exactly what to transfer; **custom** requires a description. **Keep / ignore** is an optional collapsed detail. These choices persist with the workflow and travel with saved reference-file packs. An **all** pack also carries verified media mappings; overwrite-loading it rebuilds them against the new timeline IDs and remaps surviving links automatically. Missing media links are removed with a notice. Legacy/manual and prompt-only packs retain their authored text, but their links are not silently treated as verified. Appending prompt text or replacing files alone invalidates ambiguous old links and reports this in the status. RefMod selections and settings also survive overwrite-loading; native links to RefMod members still require destination-library review, while `<RefMod N>` aliases remain intact. Only subject references join subject groups. First/last-frame roles in endpoint modes remain actual frame anchors.

For example, select **pose** and write “Use only the raised arms and stance; keep the target character's face, clothes and proportions.” Or select **custom** and write “Use the room layout and light direction, not the people.” Vision models can see attached timeline pictures; text-only models use instructions, not pixels. Saved RefMods contribute their descriptions, not decoded pictures; video and audio reference content is not analyzed. Prompt guidance is not dedicated skeleton/pose conditioning and does not guarantee a matching pose. During continuity, Forge also receives source-tail context and the next action; a changed source, Duration, mode or prompt invalidates an old draft. Applying only edits the continuation prompt.

### RefMods (REF2VA)

Put standalone or upstream v5 bundle `.safetensors` files in `ComfyUI/models/refmods/` (subfolders work). Open **REFMOD** beside Input Scaling, select a file, enable its row, adjust strength (0–1) and edit its description. Up to eight RefMod slots are supported, subject to REF2VA's final reference limits. The overlay is separate from the node; files load when opened. Workflow descriptions override embedded ones. An optional [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) installation can create compatible files, but is not needed for playback. This pack scales latents directly by strength rather than using upstream blur-mix behavior.

**Insert RefMod #** writes its expanded native `<Picture N>`, `<Video N>`, or `<Audio N>` label and description into the prompt. Saved `<RefMod N>` aliases still resolve at queue time. A v5 bundle expands to all members in order. Disabled/zero-strength rows contribute no reference; an unresolved alias is an error. Save/Load reference packs also preserve RefMod selection.

## Continue an existing video (optional)

Choose a completed checkpoint or ordinary video using **Choose start video…**. **Continuity Active** appears automatically; Duration now means *new* seconds, not total length. The source stays pinned until you change or clear it. **∞ Save new takes** saves fresh generations as checkpoints; continuations are saved regardless. Add **H3 Continuity • Append & Stage** after sampling and **H3 Continuity • Publish Export** after your actual exporter to publish only successful exports.

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
