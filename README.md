# DaSiWa Resolution Scaler

A mathematically precise resolution scaling node for ComfyUI.

## 🚀 Why?
Standard scalers often change the total pixel count when you change aspect ratios, leading to unexpected VRAM spikes or OOM (Out of Memory) errors. This scaler uses a constant-area formula:

$$Width = \sqrt{TargetPixels \times AspectRatio}$$
$$Height = \sqrt{TargetPixels / AspectRatio}$$

This ensures that whether you are generating 1:1, 9:16, or 21:9, the **total stress on your GPU remains identical.**

## ✨ Features
- **Pixel Precision Tiers**: Choose from optimized Megapixel levels.
- **Resolution Presets**: Standard targets like 720p, 1080p, and 4K areas.
- **Smart Aspect Engine**: 
  - **Scale From Image**: Automatically detects the shape of your input image.
  - **Manual Aspect**: Define custom ratios (e.g., 9:21) without changing the "detail" level.
- **WAN/LTX Optimization**: Optional **Div32** mode that snaps all dimensions to 32-pixel boundaries (mandatory for WAN/LTX VAEs to prevent artifacts).
- **Standard Mode**: Clean rounding for use with SD1.5, SDXL, or Flux.
- **Error Protection**: Built-in validation to prevent crashes if inputs are missing.

## 🛠 Installation

### Manual install

1. Activate your venv inside your comfyui folder

2. Clone this repo into your `custom_nodes` folder:
   ```
   git clone https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes
   ```

3. Install alld ependencies    
e.g.
    ```
    uv pip install -r requirements.txt
    ```
### Use ComfyUI-Manager

Search for DaSiWa-Nodes and install