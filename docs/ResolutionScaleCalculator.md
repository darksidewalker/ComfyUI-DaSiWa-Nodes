# DaSiWa: Resolution Scale Calculator

The **DaSiWa Scale Calculator** provides mathematically precise resolution management for high-performance video models. It uses a **Constant-Area Square-Root method** to ensure that your GPU VRAM usage remains stable regardless of the aspect ratio.

## 📐 The Core Logic
Standard scalers often change the total pixel count when you change aspect ratios, leading to unexpected "Out of Memory" (OOM) errors. DaSiWa treats the resolution as a **Pixel Budget (Megapixels)**. 

Whether you are generating in **1:1 Square**, **9:16 Vertical**, or **21:9 Ultra-Wide**, the node calculates dimensions so the total surface area remains consistent.

## 🚀 Key Features
- **WAN/LTX (Div32) Mode:** Snaps calculations to the nearest **32-pixel boundary**. Mandatory for WAN 2.1 and LTX-Video.
- **Precision Presets:** Multiple tiers from 0.26 MP (Preview) to 8.30 MP (4K Production).
- **No Scale Toggle:** A dedicated "Bypass" switch to use source image dimensions exactly.
- **Batch-Safe Scaling:** Intelligent first-frame parsing for video and image batches.

# 🛠 Quick Start Guide

### 1. Define your "Size" (Method)
The node uses a **Constant-Area** formula. Instead of setting raw width/height, you set a **Pixel Budget**.
* **Precision Presets:** Optimized Megapixel tiers. **0.52 MP (SD)** is the baseline for most video models.
* **Resolution Presets:** Standard targets (e.g., **1080p**). The node adjusts the actual dimensions to fit your custom aspect ratio while keeping the "1080p density."

### 2. Define your "Shape" (Aspect Ratio)
* **Scale From Image (YES):** The node "peeks" at the first frame of your input. It calculates the aspect ratio and applies your chosen MP budget to that specific shape.
* **Scale From Image (NO):** Uses the **Aspect Preset** (e.g., 9:16) or **Manual Aspect** sliders. 
    * *Note: Manual sliders define the ratio (e.g., 21 x 9), not the final pixels.*

### 3. Choose your "Engine" (Mode)
* **WAN/LTX (Div32):** **Mandatory for Video.** Forces the math to snap to the nearest 32-pixel block. This prevents VAE artifacts and edge flickering.
* **Standard:** Use for Flux, SDXL, or SD1.5. It rounds to the nearest single pixel for maximum mathematical accuracy.

### 4. The "Bypass" (No Scale)
* **Toggle ON:** Disables all calculations. The node outputs the exact width/height of the source image/manual input. Use this for native-resolution upscaling or final renders.

---

## 💡 Quick Reference Table

| Goal | Method | Mode | Scale From Image? |
| :--- | :--- | :--- | :--- |
| **WAN/LTX Video** | Precision Presets | WAN/LTX (Div32) | YES |
| **Flux Image** | Resolution Presets | Standard | NO (Manual) |
| **Social Media Clip** | Precision Presets | WAN/LTX (Div32) | NO (Use 9:16) |
| **Original Dims** | *Any* | *Any* | **No Scale: ON** |