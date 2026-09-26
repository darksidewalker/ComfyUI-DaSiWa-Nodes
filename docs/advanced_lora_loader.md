# Advanced LoRA Loader

**Category:** `loaders/lora`  
**Class name:** `DaSiWa_AdvancedLoRALoader` (serialized node ID: `DaSiWa_LTX2LoraLoader`, unchanged for workflow compatibility)  
**File:** `nodes/nodes_advanced_lora_loader.py` · `js/advanced_lora_loader_ui.js` · `nodes/lora_info.py` (info panel backend)

---

## Overview

The **Advanced LoRA Loader** is a 10-slot LoRA stacker designed specifically for LTX-2.3 workflows. LTX-2.3 is unique because it generates **both video and audio** from a single model using completely separate transformer branches. This node exploits that architecture to give you independent control over how LoRAs affect video and audio generation.

Each slot lets you:
- **STR** — Master LoRA strength (works like any normal LoRA loader, −2.0 to +2.0)
- **V×** — Video branch multiplier (0.0–2.0, default 1.0)
- **A×** — Audio branch multiplier (0.0–2.0, default 1.0)

**Effective video strength** = STR × V×  
**Effective audio strength** = STR × A×

> **Fix (v0.4.13):** the STR / VIS / A steppers no longer bounce off `0`. The old `|| 1.0` fallback treated a real `0` as falsy, so STR snapped back to a positive default and could not go negative, and VIS/A would not hold at `0`. The UI now uses a nullish fallback (`?? 1.0`), so a genuine `0` is preserved and STR spans its full negative-to-positive range.

---

## Why This Matters

Imagine you have a celebrity LoRA trained on video of them speaking. That LoRA learned:
- Their face in the **video branch**
- Their voice in the **audio branch**

With the Lora Loader, you can now:

✓ Load it with **V:0.0 A:1.0** — get their voice only, applied to your own character  
✓ Stack a different celebrity LoRA with **V:1.0 A:0.0** — their face, someone else's voice  
✓ Fix crackling audio by setting **A:0.7** while keeping visuals at full strength  
✓ Mix up to 10 LoRAs at once without competing audio artifacts

---

## Inputs

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | The LTX-2.3 model to apply LoRAs to. |
| `clip` | CLIP | The CLIP model (passed through unchanged). |
| `stack_data` | STRING | JSON-encoded LoRA stack configuration (auto-managed by UI). |
| `model_type` | COMBO (`Basic` / `LTX-2.3`) | `Basic` loads every tensor once; `LTX-2.3` separates video/audio branches. |
| `use_cache` | BOOLEAN | Opt-in cache for the loaded LoRA file + metadata. **Off by default**; when on, each unique file is read once instead of once per slot. |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `model` | MODEL | The model with all active LoRAs applied. |
| `clip` | CLIP | The CLIP model (unchanged). |

---

## UI Controls

### Rows (LoRA slots)

Each row represents one LoRA slot. Columns are:

| Column | Control |
|---|---|
| **✔ ON / ✖ OFF** | Toggle this slot on/off. When off, the LoRA is ignored. |
| **LoRA Name** | Click to select a LoRA from your library. Search box appears for quick filtering. |
| **STR** | Master strength. Click `<` / `>` to adjust by ±0.05, or click the middle to open an inline editor (type a value, OK to apply / Esc to cancel). Range: −5.0 to +5.0. |
| **V×** | Video multiplier. Left/right arrows adjust by ±0.05, middle click opens an inline editor. Range: 0.0 to 2.0. |
| **A×** | Audio multiplier. Same controls as V×. Range: 0.0 to 2.0. |
| **V:N A:N** | Key count indicator (right side). Shows how many video and audio keys this LoRA contains. Updates automatically. |
| **ⓘ** | Info button (row right edge, dimmed for empty slots). Drawn as a circle with an "i" (ASCII, no emoji). Opens the LoRA info panel: both Civitai mirror links (`.com` + `.red`, looked up by the file's SHA-256), trigger/trained words, and preview images. |
| **Trash** | Trash button (directly right of the info button, dimmed for empty slots). Drawn as an ASCII trash-can (no emoji). Clicking it sets this slot's LoRA back to **None** — the STR / V× / A× values are kept, exactly like picking "None" in the slot picker. Use it to unstack a LoRA without reopening the picker. (v0.4.29) |

### Buttons

- **⚡ CACHE / ⚡ CACHE ✓** — Toggle the opt-in LoRA file cache. **Off by default** (grey, reads the file per slot, current behaviour). When on (theme-colored, shows `✓`), each unique LoRA file is read once and reused across slots. Persists with the workflow.
- **⬡ THEME: [NAME] ▶** — Cycle through 6 color themes (Jade, Neon, Studio, Chrome, OLED, Wood). Persists with workflow.
- **+** — Add a new LoRA slot at the bottom.
- **−** — Remove the last LoRA slot (only visible if more than 1 slot exists).

---

## Key Count Indicator

When you load a LoRA, the node scans the file and shows:
- **V:N** — Number of video-branch keys in the LoRA
- **A:N** — Number of audio-branch keys in the LoRA

If **A:0**, the LoRA was trained on silent data and audio mode won't have any effect. This helps you identify which LoRAs are worth using in audio-multiplier mode before wasting a generation.

---

## LoRA Info Panel (ⓘ)

Click the ⓘ glyph at the right edge of a slot's row (v0.4.28) to open an info panel for that LoRA. It shows:

- **Civitai mirror links** — the file's SHA-256 is looked up on Civitai's `model-versions/by-hash` API and the panel shows **both** mirrors, `.com` (labeled `BLUE:` in blue) and `.red` (labeled `RED:` in red); the labels are plain text, only the URL is clickable (v0.4.35). The lookup runs on `.com` first and **falls back to the `.red` mirror** on a 404, so a working mirror still yields a link when the other is down. Results are cached in `lorainfo/<sha256>.json` next to the nodepack — **misses included**, so a LoRA with no Civitai page no longer re-hits the API on every open (the **Refresh** button forces a re-lookup via `refresh=true`).
- **Trigger / trained words** — collected from the LoRA's safetensors `ss_tag_frequency` metadata and from Civitai when available. Click words to select them, then **Copy all** / **Copy selected** to put them on the clipboard for your prompt.
- **Images** — Civitai preview images (first six), plus a local sidecar image (`.png` / `.jpg` / `.jpeg` / `.webp` with the same basename as the LoRA) if one sits next to the file.

The panel's **file name** and **sha256** sit on their own rows at the top (above the control buttons), so a long folder path can't stretch the button row (v0.4.35).

If the LoRA is not on either Civitai mirror, the panel says so and still shows the metadata-based words and any local image. No internet access is required for the metadata/local-image parts; only the Civitai lookup touches the network.

---

## Trash Button (per slot, v0.4.29)

Every slot row shows a small ASCII-drawn trash button directly to the right of the info button (dimmed when the slot is already **None**). Clicking it sets that slot's LoRA back to **None** while keeping the slot's STR / V× / A× values — the same result as picking "None" from the slot picker, but without reopening it. The info button was shifted slightly left to make room for the trash button.

---

## Example Workflows

### Scenario 1: Single voice LoRA with full control
```text
STR: 1.0, V×: 1.0, A×: 1.0    (Normal, both video and audio)
STR: 1.0, V×: 0.0, A×: 1.0    (Audio only)
STR: 1.0, V×: 1.0, A×: 0.0    (Video only)
```

### Scenario 2: Stacking two character LoRAs
```text
Slot 1: Celebrity A    STR: 1.0, V×: 1.0, A×: 0.0  (their face)
Slot 2: Celebrity B    STR: 1.0, V×: 0.0, A×: 1.0  (their voice)
Result: A's face + B's voice
```

### Scenario 3: Blending with negative strength
```text
STR: −0.5, V×: 1.0, A×: 0.0   (Reduce specific video features)
```

### Typical wiring in a workflow

```mermaid
flowchart LR
    CL[Checkpoint Loader] -->|MODEL| ALR[Advanced LoRA Loader<br>Slot 1: face V×1.0 A×0.0<br>Slot 2: voice V×0.0 A×1.0]
    CL -->|CLIP| CTE[CLIP Text Encode]
    ALR -->|MODEL patched| KS[KSampler]
    CTE -->|CONDITIONING| KS
    KS -->|latent| VD[VAE Decode]
    VD -->|IMAGE batch| out((output))
```

---

## Tips & Tricks

- **Explore the themes** — Each theme optimizes for different lighting conditions. "OLED" is great for dark environments; "Chrome" for bright.
- **Use prompt-based strength tuning** — Load the same LoRA in multiple slots with different STR values to fine-tune blend amounts.
- **Disabled slots are free** — Toggling a slot off costs nothing; it won't process at all.
- **No generation overhead** — The node separates video/audio keys before loading, so unused branches are skipped.
- **JSON is editable** — The `stack_data` is plain JSON; if you need to script or batch-edit LoRA stacks, you can write Python to generate the JSON string directly.

---

## Technical Details

- **LTX-2.3 key format:**
  - Video keys: `diffusion_model.transformer_blocks.N.attn*`, `diffusion_model.transformer_blocks.N.ff*`
  - Audio keys: `diffusion_model.transformer_blocks.N.audio_*`, plus cross-modal attention keys
- **Branch separation:** The node scans each LoRA's weights, filters by key name, and applies them separately.
- **Strength multiplication:** Effective strengths are computed as `STR × multiplier`, allowing negative STR to invert effects.
- **Safe fallback:** If a LoRA file is missing or corrupted, the node logs a warning and continues with the remaining LoRAs.
- **Civitai mirror links + panel rework (v0.4.35):** the (i) panel now always shows both Civitai mirrors (`.com` labeled `BLUE:` in blue, `.red` labeled `RED:` in red, labels plain text outside the links); the backend by-hash lookup runs on `.com` first and falls back to the `.red` mirror on a 404, and results are memoized including misses (a missing LoRA no longer re-hits the Civitai API on every panel open — **Refresh** forces a re-lookup). The file name and sha256 moved to their own rows above the controls so long folder paths can't stretch the button row; the earlier `.com`/`.red` domain selector buttons were removed.
- **Renamed (v0.4.27):** the loader was renamed from the LTX-2-only `DaSiWa LTX-2 Master Loader` to the universal **Advanced LoRA Loader**. The *serialized* node ID `DaSiWa_LTX2LoraLoader` is unchanged, so saved workflows load untouched; only the module, class, JS, docs, and display name changed internally.
- **LoRA info button (v0.4.28):** the per-row ⓘ glyph opens a panel served by two new GET routes, `/dasiwa/ltx2/lorainfo` (sha256 of the file + safetensors header metadata + cached Civitai by-hash lookup) and `/dasiwa/ltx2/loraimg` (a sidecar image next to the LoRA file). Both live in `nodes/lora_info.py`; lookup results are cached as `lorainfo/<sha256>.json` next to the nodepack.
- **PDD/ACC metadata (v0.4.27):** the LoRA file is now read with `return_metadata=True` and the metadata is forwarded to Core's `load_lora_for_models`, matching the native `LoraLoader`. This is what activates PDD / ACC LoRA head banks. Older ComfyUI builds that don't accept the metadata are still supported via a `TypeError` fallback.
- **Opt-in cache (v0.4.27):** when **use_cache** is on, each absolute LoRA path is kept in a bounded LRU cache (max 4 entries), so a LoRA reused across slots is read once. **Off by default** — when off, the file is read per slot (current behaviour, no change).
- **PDD head-bank guard (warn-only):** if a loaded LoRA carries PDD metadata (`pdd_num_steps`/`pdd_block_size`) **and** its `final_layer` head-bank width differs from the target model's live `final_layer.video_out` width, the loader **prints a console warning** but leaves every key in place. The core shape crash at `comfy/lora.py` (`The size of tensor a … must match …`) is intentional protection against applying a PDD Acc LoRA to a single-head H3 model — the guard does **not** strip or circumvent it. The warning tells you the head bank won't apply and how to fix it (use a PDD model). Only on positive evidence (both widths readable and different) does the guard warn; a genuine PDD model whose width matches is never touched.

---

## Common Issues

### "LoRA not found"
The LoRA file is missing from your `loras/` folder. Check the filename and spelling, or re-download.

### A slot shows "?" for key counts
The node is still scanning the LoRA file in the background. Wait a moment and the counts will appear.

### Audio seems unchanged even with A× at 1.0
The LoRA might have **A:0** keys (trained on silent data). Check the key count indicator. If A:0, train or find a different LoRA with audio content.

### Performance is slow
Disable unused slots or reduce the number of active LoRAs. Fewer slots = faster execution.

### PDD / Acc LoRA on a single-head model — `The size of tensor a (96) must match … (3072)`
This error is raised by ComfyUI core (`comfy/lora.py`) when a **PDD Acc LoRA** (its `__metadata__` has `pdd_num_steps`, and it carries a wide `final_layer` head bank, e.g. `video_out.set_weight [3072,5376]`) is applied to a **single-head** H3 model (`final_layer.video_out [96,5376]`). The LoRA's 32-head bank cannot be copied onto a 1-head weight.

- **Recommended fix:** pair the PDD LoRA with a **PDD model** (`final_layer.video_out [3072,5376]`), or use a non-PDD Acc LoRA. (PDD LoRAs from `Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc` / `aptech0081/…` require a PDD model — none of the single-head H3 models in a typical library are PDD.)
- **Console warning:** the loader prints a warning to the ComfyUI console before the run so you know the PDD head bank won't apply to this single-head model — and the run still aborts with the core shape crash on purpose. The crash is protection, not a bug: it stops an incompatible PDD-LoRA/single-head-model pairing. The loader deliberately does **not** strip the head-bank keys to work around it; that stays that way until an upstream ComfyUI patch lands.

---

## Keyboard / Mouse Shortcuts

| Action | How |
|---|---|
| **Change slot on/off** | Click the ON/OFF column |
| **Change LoRA** | Click the LoRA name, search to filter |
| **Fine-tune STR/V×/A×** | Click `<` or `>` arrows (±0.05 per click) |
| **Manual input** | Click the value pill (middle) to open an inline editor; type the value and press OK / Enter (Esc cancels). The box stays pinned next to the pill and follows pan / zoom, like native ComfyUI value editors |
| **Cycle theme** | Click the theme button |
| **Add slot** | Click the `+` button |
| **Remove slot** | Click the `−` button |

---
