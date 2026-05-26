# DaSiWa Custom Nodes Collection

A high-performance collection of custom nodes for ComfyUI, optimized for video workflows, resolution management, and logic control.

## Included Nodes

---

### 💎 RTX Upscaler & Refiner

State-of-the-art image and video enhancement using NVIDIA RTX Video SDK. It executes up to three sequential passes (Denoise, Deblur, and Upscale) in a single node, processing frame-by-frame to keep VRAM usage predictable and low.

- **Refine:** Independent Denoise and Deblur passes.
- **Upscale:** AI-powered VSR and High Bitrate upscaling.
- **Smart Sizing:** Multiple resize modes including Constant Megapixel targets.
- **Efficiency:** Frame-by-frame processing for minimal VRAM usage.

![RTX_UpscalerRefiner.png](assets/RTX_UpscalerRefiner.png)

[Full documentation →](docs/rtx_upscaler_refiner.md)

---

### 📐 Resolution Scale Calculator

The **DaSiWa Scale Calculator** provides mathematically precise resolution management for high-performance video models. It uses a **Constant-Area Square-Root method** to ensure that your GPU VRAM usage remains stable regardless of the aspect ratio.

![ResolutionScaleCalculator.png](assets/ResolutionScaleCalculator.png)

[Full documentation →](docs/ResolutionScaleCalculator.md)

---

### 🎛️ Node Status Switch

The **DaSiWa Node Status Switch** lets you mute or bypass any node in your workflow using a single toggle. Targets are registered by wiring their outputs into the switch's input slots, which grow dynamically as you connect more nodes (up to 99).

![NodeStatusSwitch.png](assets/NodeStatusSwitch.png)

[Full documentation →](docs/node_status_switch.md)

**Quick start:**

1. Add a **DaSiWa Node Status Switch** to your workflow
2. Drag any **output** from the node(s) you want to control into the switch's `target_01` input — new slots appear as you connect more
3. Set `action` to `mute` or `bypass` and configure `trigger_on` to taste
4. Toggle `enabled` directly on the switch

---

### 💾 Metadata Image Saver (Civitai Ready)

The **DaSiWa Metadata Image Saver** ensures your images are fully compatible with Civitai, Hugging Face, and other galleries by embedding A1111-style metadata. It automatically detects LoRAs used in the workflow and supports dynamic filenames.

- **Civitai Compatibility:** Writes the standard `parameters` block for auto-parsing of prompts and resources.
- **LoRA Detection:** Scans your workflow and appends `<lora:name:weight>` triggers automatically.
- **WebP Support:** Full "Drag-and-Drop" workflow reconstruction support for both PNG and WebP formats.
- **Dynamic Filenames:** Use placeholders like `%seed%`, `%date%`, `%model%`, `%width%`, and `%height%`.
- **Privacy:** Toggle workflow JSON embedding to share images without exposing your full graph.

![DaSiWa-MetadataImageSaver.png](assets/DaSiWa-MetadataImageSaver.png)

[Full documentation →](docs/metadata_image_saver.md)

---

## 🛠️ Installation

### Manual install

1. Activate your venv inside your ComfyUI folder
2. Clone this repo into your `custom_nodes` folder:
   ```bash
   git clone https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes
   ```
3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Requirement:** NVIDIA RTX GPU with drivers 530+. (Windows users may need the NVIDIA Broadcast SDK; Linux usually works out-of-the-box with the pip package).
5. Restart ComfyUI.

### Use ComfyUI-Manager

Search for **DaSiWa-Nodes** and install.

---

## Credits

The RTX implementation in this collection is based on the excellent work by Deno2026/comfyui-deno-custom-nodes.