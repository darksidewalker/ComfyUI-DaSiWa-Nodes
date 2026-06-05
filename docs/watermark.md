# DaSiWa Watermark Overlay

**Category:** `DaSiWa/Video`  
**Class name:** `DaSiWa_Watermark`  
**File:** `nodes/watermark_nodes.py`

---

## Overview

The **DaSiWa Watermark Overlay** is a professional-grade node designed for adding watermarks to images and video batches. It features high-quality bicubic resampling, precise rotation, and a stable CPU compositor for consistent frame-to-frame output.

## Key Features

- **Robust Batch Processing:** Stores output batches in system RAM and processes watermark regions frame by frame.
- **Stable CPU Compositing:** Initializes every output frame from the source image before blending the watermark region, avoiding uninitialized frames, CUDA fallback differences, and black-frame artifacts.
- **Dynamic Random Positioning:** Toggleable corner cycling that starts from the selected position, with a seed to keep chained nodes from matching unintentionally.
- **Dynamic Fade (Splash Mode):** Toggleable fade that brings the watermark in at the start and end of the sequence while keeping the middle clear.
- **Optical Padding:** Automatically adjusts the watermark's position based on its visual center of mass. This ensures that asymmetric or rotated watermarks look perfectly aligned relative to the edges.
- **Premultiplied Alpha Blending:** Premultiplies before scaling/rotation and clamps the result to avoid dark halos or black transparent backgrounds.
- **Bicubic Scaling:** Maintains sharpness and detail when resizing the watermark image.

## Inputs

| Input | Type | Description |
|---|---|---|
| `images` | IMAGE | The image or video batch to apply the watermark to. |
| `watermark_path` | Combo | Select or upload a watermark file (PNG with transparency is highly recommended). |
| `position` | Combo | Static placement, or the starting placement when `randomize_position` is enabled. |
| `scale` | FLOAT | Size relative to the input image width (e.g., 0.2 = 20% width). |
| `resampling` | Combo | The algorithm used to resize the watermark. **Bicubic** is the recommended default for quality. |
| `transparency` | FLOAT | Opacity of the watermark (1.0 = Opaque, 0.0 = Invisible). |
| `rotation` | INT | Rotation in degrees (0–359). Canvas expands automatically to prevent clipping. |
| `padding_x / y` | INT | Pixel offset from the chosen edge. |
| `optical_padding`| BOOLEAN | Enable centroid-based alignment for visual balance. |
| `random_switches` | INT | How many times the position should jump when `randomize_position` is enabled. |
| `fade` | BOOLEAN | Enable the splash branding effect. |
| `fade_margin` | FLOAT | Duration for the start and end fades as a percentage of total frames (0.1 = 10%). |
| `randomize_position` | BOOLEAN | Enable seeded corner cycling. |
| `random_seed` | INT | Seed for corner order. `0` derives a stable seed from the watermark file name. |

## Tips for Best Results

### Visual Centering
If your watermark has a significant shadow on one side or is a long string of text, standard corner alignment often looks "off." Enable **Optical Padding** to let the node calculate the visual center of the logo and nudge it into the perfect position.

### Anti-Cropping with Random Mode
Enable **randomize_position** to alternate between the four corners. The first interval uses the selected **position**, then the node jumps exactly the number of times specified in `random_switches` over the length of the batch. Use different start positions or different `random_seed` values when chaining multiple watermark nodes.

### Splash Branding with Dynamic Fade
Enable **fade** and set a **fade_margin** (e.g., 0.1 for 10%) to create a professional "splash" effect. The watermark will fade in over the first 10% of frames and reappear during the last 10%, remaining hidden in between.

### Stability and Memory
When working with 2K/4K video batches or complex LTX workflows, the node keeps the output batch in system RAM and uses one consistent CPU blend path. This is intentionally more conservative than opportunistic GPU blending, but it avoids frame-to-frame differences, fallback flicker, and black-border artifacts from device-path changes.

### High Quality Blending
The node automatically premultiplies the alpha channel before resizing and clamps the rotated result so transparent pixels stay transparent. This means you don't need to pre-process your logos in Photoshop; just provide a high-quality transparent PNG, and the node handles the math for a seamless overlay.

## Technical Details

- **Rotation:** Uses bicubic interpolation with an axis-aligned bounding box expansion.
- **Memory Strategy:** 
    1. Watermark is loaded, rescaled, and rotated on CPU.
    2. Output batches are initialized from the source frames in CPU/RAM.
    3. Frames are processed one-by-one using the same premultiplied-alpha blend path for every frame.
