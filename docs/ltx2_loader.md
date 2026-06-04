# DaSiWa LTX-2 Master Loader

**Category:** `loaders/lora`  
**Class name:** `DaSiWa_LTX2LoraLoader`  
**File:** `nodes/ltx2_loader.py` · `js/ltx2_dynamic_ui.js`

---

## Overview

The **DaSiWa LTX-2 Lora Loader** is a 10-slot LoRA stacker designed specifically for LTX-2.3 workflows. LTX-2.3 is unique because it generates **both video and audio** from a single model using completely separate transformer branches. This node exploits that architecture to give you independent control over how LoRAs affect video and audio generation.

Each slot lets you:
- **STR** — Master LoRA strength (works like any normal LoRA loader, −2.0 to +2.0)
- **V×** — Video branch multiplier (0.0–2.0, default 1.0)
- **A×** — Audio branch multiplier (0.0–2.0, default 1.0)

**Effective video strength** = STR × V×  
**Effective audio strength** = STR × A×

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
| **STR** | Master strength. Click `<` / `>` to adjust by ±0.05, or click the middle to type a custom value. Range: −2.0 to +2.0. |
| **V×** | Video multiplier. Left/right arrows adjust by ±0.05, middle click for custom input. Range: 0.0 to 2.0. |
| **A×** | Audio multiplier. Same controls as V×. Range: 0.0 to 2.0. |
| **V:N A:N** | Key count indicator (right side). Shows how many video and audio keys this LoRA contains. Updates automatically. |

### Buttons

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

## Example Workflows

### Scenario 1: Single voice LoRA with full control
```
STR: 1.0, V×: 1.0, A×: 1.0    (Normal, both video and audio)
STR: 1.0, V×: 0.0, A×: 1.0    (Audio only)
STR: 1.0, V×: 1.0, A×: 0.0    (Video only)
```

### Scenario 2: Stacking two character LoRAs
```
Slot 1: Celebrity A    STR: 1.0, V×: 1.0, A×: 0.0  (their face)
Slot 2: Celebrity B    STR: 1.0, V×: 0.0, A×: 1.0  (their voice)
Result: A's face + B's voice
```

### Scenario 3: Blending with negative strength
```
STR: −0.5, V×: 1.0, A×: 0.0   (Reduce specific video features)
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

---

## Keyboard / Mouse Shortcuts

| Action | How |
|---|---|
| **Change slot on/off** | Click the ON/OFF column |
| **Change LoRA** | Click the LoRA name, search to filter |
| **Fine-tune STR/V×/A×** | Click `<` or `>` arrows (±0.05 per click) |
| **Manual input** | Click the value pill directly to type |
| **Cycle theme** | Click the theme button |
| **Add slot** | Click the `+` button |
| **Remove slot** | Click the `−` button |

---
