# DaSiWa: RTX Upscaler & Refiner

This node leverages NVIDIA RTX Video SDK features to provide professional-grade image/video enhancement directly within ComfyUI. It executes up to three sequential passes in a single node, processing frame-by-frame to keep VRAM usage predictable and low.

## ⚡ Processing Pipeline
The node executes effects in this order:
1.  **Denoise (Pass 1):** Cleans up grain and compression artifacts at the source resolution.
2.  **Deblur (Pass 2):** Sharpens focus and reduces motion blur at the source resolution.
3.  **Upscale (Pass 3):** Scales the image to the target resolution using AI (VSR) or High Bitrate processing.

## 📐 Resize Types Explained

Choosing the right `resize_type` is key to getting the correct output resolution:

*   **Same Size:** The node processes the input at its original resolution. Use this if you only want to Denoise or Deblur without changing the size.
*   **Scale:** Simply multiplies the current width and height by the `scale` value. For example, a 1080p image at 2.0x scale becomes 4K.
*   **Keep Ratio:** Calculates a new resolution that hits your `megapixels` budget while preserving the exact aspect ratio of the input image. Great for maintaining the "look" while increasing quality.
*   **Preset Ratio:** Targets the `megapixels` budget but forces the image into a specific shape (like `16:9` or `9:16`) regardless of the input. Use `resize_method` to decide if the image should be cropped or letterboxed to fit.
*   **Manual:** Allows you to set the exact `width` and `height` in pixels.

## 💎 Quality Levels (Low to Ultra)

RTX Quality settings control the complexity of the AI models used:
*   **Low / Medium:** Balanced for speed. Best for real-time previews or mid-range RTX cards.
*   **High:** Standard high-quality reconstruction.
*   **Ultra (Default):** Maximum parameter count models. Provides the cleanest edges and highest detail.

## ⚠️ When Results Can Look Worse

RTX VSR is an upscaler, not a detail recovery tool for material that has already been enlarged. If the input video was previously upscaled, heavily sharpened, denoised, or re-encoded, RTX VSR may enhance those artifacts instead of creating real new detail. This can look pixelated, smeared, or over-processed.

Check these settings first when users report degraded quality:

*   **Target resolution below input:** `Keep Ratio` uses the `megapixels` value. For example, a 4K input is about 8.3 MP, so `Keep Ratio` at `2.0` MP is a downscale and will look softer.
*   **Manual size below input:** Manual `1920x1080` on a 4K source is also a downscale.
*   **Divisibility too high:** `divisible_by=8` matches RTX VSR behavior and common video sizes like 3840x2160. Use `32` only for downstream nodes that require it, because standard 2160p is not divisible by 32.
*   **Aspect changes:** `Preset Ratio` and mismatched `Manual` dimensions intentionally crop or letterbox before upscaling. Use `Scale` or `Keep Ratio` when you want to preserve the source framing.
*   **Too many cleanup passes:** Denoise and Deblur can help poor sources, but on already clean video they may remove texture before the upscale pass.

## ⚙️ Key Parameters

| Parameter | Description |
| :--- | :--- |
| **upscale** | `VSR` is best for clean AI upscaling. `High Bitrate` is specifically tuned for sources that are noisy or heavily compressed. |
| **divisible_by** | Use **8** for RTX-style image/video upscaling. Use **32** only when the next model in the workflow requires it. |
| **resize_method** | When the source and target aspect ratios don't match: `Center Crop` fills the whole target frame by cutting edges; `Letterbox` fits the whole image inside and adds black bars. |
| **device_id** | If you have multiple GPUs, set this to the index of your RTX card (usually `0`). |

## 💡 Pro-Tips

*   **Refinement First:** If your source is very low quality, enable `denoise` or `deblur` alongside `upscale`. Cleaning the image before upscaling usually yields much sharper results.
*   **VRAM Management:** This node handles the output tensor pre-allocation efficiently. However, upscaling a long video batch to 4K still requires significant system RAM. If you hit OOM, try processing smaller batches or using `Keep Ratio` with a lower `megapixels` target.
*   **Div32 Snapping:** Even in `Manual` or `Scale` modes, `divisible_by=32` can slightly adjust your dimensions. That is useful for some video VAEs, but it can avoid standard output sizes such as exact 4K.

---
*Note: This node requires the NVIDIA RTX Video SDK / Broadcast SDK to be installed on your system.*
