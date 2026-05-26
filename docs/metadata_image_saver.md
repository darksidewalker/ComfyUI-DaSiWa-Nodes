# DaSiWa: Metadata Image Saver

The **DaSiWa Metadata Image Saver** is designed to bridge the gap between ComfyUI's complex graph-based metadata and the simplified text-based format expected by sites like Civitai and Hugging Face.

!DaSiWa-MetadataImageSaver.png

## 🚀 Key Features

### 1. Civitai / A1111 Compatibility
Unlike the standard ComfyUI Save Image node, this node generates a `parameters` field in the image metadata. This field contains:
*   Positive and Negative prompts.
*   Sampler, Scheduler, Seed, CFG, and Steps.
*   Model name and Model Hash.
*   Automatically detected LoRAs.
*   **Full WebP Support:** Unlike native nodes, DaSiWa embeds the workflow into WebP EXIF tags, allowing for drag-and-drop reconstruction in ComfyUI.

### 2. Automatic LoRA Detection
The node "spies" on your workflow at execution time. It performs an aggressive scan for LoRA loaders (standard, rgthree, Impact, EasyUse, etc.) and automatically appends the correct `<lora:name:weight>` syntax to your positive prompt in the metadata.

### 3. Dynamic Filenaming
You can use the following placeholders in the `filename_prefix` field:
*   `%seed%`: The seed used for the generation.
*   `%model%`: The base name of the checkpoint.
*   `%width%` / `%height%`: Output dimensions.
*   `%date:format%`: Current timestamp. Example: `%date:yyyy-MM-dd_HHmm%`.

### 4. Extra Metadata Injection
Pair this node with the **DaSiWa Create Extra Metadata** node to add custom fields to the settings line, such as "Author", "License", or custom triggers.

## ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| **file_format** | Choose between `webp` (modern, small) or `png` (lossless, high compatibility). |
| **compression** | 0 to 100. For **PNG**, 0 is fastest and 100 is max compression (level 9). For **WebP**, 0 is best quality (lossless) and 100 is max compression. |
| **save_output** | If **True**, images are saved to the `output` folder. If **False**, they are saved to `temp` for preview in the node canvas. |
| **save_workflow** | If enabled, the full ComfyUI JSON graph is embedded. Disable this for "privacy mode". |
| **node_positive** | **Connect the `CONDITIONING` output from your positive prompt node.** |
| **node_negative** | **Connect the `CONDITIONING` output from your negative prompt node.** |
| **node_model** | **Connect the `MODEL` output from your model loader node.** |
| **node_latent** | **Connect the `LATENT` output from your KSampler node.** |
| **node_noise** | **Connect a `NOISE` output (e.g., RandomNoise) to detect seeds.** |
| **node_sigmas** | **Connect a `SIGMAS` output (e.g., BasicScheduler) to detect steps/scheduler.** |
| **node_sampler** | **Connect a `SAMPLER` output (e.g., KSamplerSelect) to detect sampler names.** |
| **model_hash** | Optional. Providing this helps Civitai link your image to the correct model page. |
| **text_positive** | Optional. Override for positive prompt. |
| **text_negative** | Optional. Override for negative prompt. |
| **text_steps** | Optional. Override for steps. |
| **text_cfg** | Optional. Override for CFG scale. |
| **text_sampler** | Optional. Override for sampler name. |
| **text_scheduler** | Optional. Override for scheduler. |
| **text_seed** | Optional. Override for seed. |
| **text_model** | Optional. Override for model name. |

## 🛠️ Usage Example

1.  Connect your final **IMAGE** to the `images` input.
2.  Convert your **CLIP Text Encode** (Prompt) and **Sampler** widgets to inputs.
3.  Pass the same strings/values used for generation into the Metadata Image Saver.
4.  Set your filename prefix to something like `MyProject_%date:yyyyMMdd%_%seed%`.
5.  Run the queue. The saved image will now be "Drag-and-Drop" ready for Civitai.

---
*Note: This node does not require any additional Python packages beyond standard ComfyUI dependencies.*