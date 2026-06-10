# DaSiWa LLM / VLM Nodes

Run local transformers text or vision-language models inside a ComfyUI workflow.

## Nodes

### DaSiWa LLM Model Selector

Creates a lightweight `DASIWA_LLM_CONFIG` bundle. It does not output a live model object, which helps the analyze node unload memory reliably after generation.

Place full Hugging Face-style model folders in:

```text
ComfyUI/models/llm/
```

The folder should include files such as `config.json`, tokenizer files, processor files for vision models, and `.safetensors` weights. A single `.safetensors` file is usually not enough for LLM chat inference.

Important controls:

- `task`: use `auto` for most workflows. Connect images to use a vision-language model.
- `device`: `auto`, `cuda`, or `cpu`.
- `dtype`: `auto`, `float16`, `bfloat16`, or `float32`.
- `quantization`: optional `8bit` or `4bit`, requiring `bitsandbytes`.
- `cache_mode`: `cached` keeps the model loaded; `unload_after_run` frees references and clears CUDA cache after each response.
- `trust_remote_code`: enable only for models that require trusted custom model code.

### DaSiWa LLM Analyze

Runs the selected model and returns:

- `response`: generated `STRING`
- `info`: model path, cache mode, image count, and resize setting

Inputs:

- `llm_config`: from DaSiWa LLM Model Selector
- `system_prompt_preset`: preset instruction selector. `custom` uses the `system_prompt` widget.
- `system_prompt`: visible custom system instruction widget
- `prompt`: visible task prompt widget
- `images`: native ComfyUI `IMAGE` input, compatible with Load Image and VHS/image-sequence frame batches
- `text_input`: connected text to analyze

System prompt presets:

- `custom`: use the system prompt widget exactly as written.
- `enhance_video_ltx23`: turn input text plus optional image into one flowing LTX-2.3 video prompt with shot, scene, action, character cues, camera movement, atmosphere, and audio.
- `enhance_video_wan22`: turn input text plus optional image into a detailed Wan2.2 video prompt, preserving image identity for I2V/TI2V and enriching motion, setting, lighting, and camera language.
- `caption_image_*`: caption a single image.
- `caption_video_*`: caption sampled video frames as one coherent clip.

Caption preset suffixes:

- Detail: `simple`, `detailed`, `very_detailed`.
- Style: `mixed`, `tag`, `natural`.
- `mixed`: booru-style tags followed by one natural-language sentence.
- `tag`: comma-separated WD14/Pony/Illustrious-style tags only.
- `natural`: descriptive natural language for FLUX, Wan, LTX, SD3, and similar prompt-following models.

Video/image-sequence handling:

- ComfyUI and VHS expose videos as an `IMAGE` batch.
- `max_frames` limits how many frames are sent to the VLM.
- `frame_strategy` chooses first, middle, last, every nth, or evenly spaced frames.
- `resize_max_px` downscales frames before inference to save VRAM.
- `resize_algorithm` selects the downscale filter: `lanczos`, `bicubic`, `bilinear`, `hamming`, `box`, or `nearest`.
- `max_input_tokens` optionally truncates long text/context input before generation. This can reduce attention memory for long prompts.
- `use_kv_cache` keeps generation key/value cache when enabled. Turning it off may reduce peak memory for some models, but generation is slower.
- `memory_cleanup` can clear cached DaSiWa LLM models before and/or after this node runs. Use `after_run` when you want a cached model for this response but want VRAM freed before later workflow steps.

## Notes

Text-only LLMs can analyze text and prompts. Image or video-frame analysis requires a vision-language model with a compatible `AutoProcessor`, such as Qwen-VL/LLaVA-style transformers model folders.

Image compression is intentionally not exposed as a memory option. Lossless compression can preserve file quality, but after the VLM processor decodes the image it does not reduce vision token count or runtime VRAM. Use `max_frames` and `resize_max_px` for image/video memory control.

GGUF/llama.cpp loading is not implemented in this version.
