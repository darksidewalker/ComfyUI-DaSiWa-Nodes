# DaSiWa: Resolution Scale Calculator

The **DaSiWa Scale Calculator** provides mathematically precise resolution management for high-performance video models. It uses a **Constant-Area Square-Root method** to ensure that your GPU VRAM usage remains stable regardless of the aspect ratio.

## 📐 The Core Logic
Standard scalers often change the total pixel count when you change aspect ratios, leading to unexpected "Out of Memory" (OOM) errors. DaSiWa treats the resolution as a **Pixel Budget (Megapixels)**. 

**MP Convention:** Like ComfyUI's native `Scale Image to Total Pixels` node, 1 MP = 1024 × 1024 = 1,048,576 pixels. This ensures DaSiWa produces the same dimensions as the built-in node for the same megapixel value.

Whether you are generating in **1:1 Square**, **9:16 Vertical**, or **21:9 Ultra-Wide**, the node calculates dimensions so the total surface area remains consistent.

## 🚀 Key Features
- **WAN/LTX (Div32) Mode:** Snaps calculations to the nearest **32-pixel boundary**. Mandatory for WAN 2.1 and LTX-Video.
- **Unified Resolution Preset Selector:** Standard labels from 144p to 2160p/4K and optimized MP tiers from 0.26 MP to 8.30 MP live in one dropdown.
- **Clear Aspect Modes:** **IMAGE ASPECT** uses the connected image shape. **USE ASPECT BELOW** uses the always-visible aspect controls.
- **No Scale Toggle:** A dedicated "Bypass" switch to use source image dimensions exactly.
- **Batch-Safe Scaling:** Intelligent first-frame parsing for video and image batches.

# 🛠 Quick Start Guide

### 1. Define your "Size" (Resolution Preset)
The node uses a **Constant-Area** formula. Instead of setting raw width/height, you set a **Pixel Budget**.
* **resolution_preset:** Choose a standard resolution label such as **540p**, **720p**, or **1080p**, or an optimized megapixel tier such as **0.52 MP - SD**. The node converts the choice into a pixel budget and adapts it to your aspect ratio.

### 2. Define your "Shape" (Aspect Ratio)
* **IMAGE ASPECT:** The node "peeks" at the first frame of your connected image. It calculates the aspect ratio and applies your chosen MP budget to that shape. The aspect controls remain visible, but they are ignored.
* **USE ASPECT BELOW:** The node uses the visible **aspect_preset_when_not_image** selector (e.g., 9:16) instead of the image shape.
* **Manual Aspect Note:** **custom_aspect_width** and **custom_aspect_height** are only used when **USE ASPECT BELOW** is active and the aspect preset is **CUSTOM**. They define a ratio such as 21 x 9, not final pixel dimensions.

### 3. Choose your "Engine" (Mode)
* **WAN/LTX (Div32):** **Mandatory for Video.** Forces the math to snap to the nearest 32-pixel block. This prevents VAE artifacts and edge flickering.
* **Standard:** Use for Flux, SDXL, or SD1.5. It rounds to the nearest single pixel for maximum mathematical accuracy.

### 4. The "Bypass" (No Scale)
* **Toggle ON:** Disables all calculations. The node outputs the exact width/height of the source image/manual input. Use this for native-resolution upscaling or final renders.

---

## 💡 Quick Reference Table

| Goal | Resolution Preset | Mode | Aspect Mode |
| :--- | :--- | :--- | :--- |
| **WAN/LTX Video** | 0.52 MP - SD | WAN/LTX (Div32) | IMAGE ASPECT |
| **Flux Image** | 1080p | Standard | USE ASPECT BELOW |
| **Social Media Clip** | 0.52 MP - SD | WAN/LTX (Div32) | USE ASPECT BELOW with 9:16 |
| **Original Dims** | *Any* | *Any* | **No Scale: ON** |
