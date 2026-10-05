# DaSiWa LLM / VLM Nodes

Run local Transformers, llama.cpp GGUF, loopback Ollama, or operator-configured external LLM servers inside a ComfyUI workflow. Model execution and server transports are shared with the Director's Prompt Forge; the Director keeps its existing settings, routes, cancellation and reference workflow.

## Nodes

### DaSiWa LLM Prompt Writer

The simple way in. Type an idea, pick what the prompt is for, get a prompt back.

- `model`: one list of everything this machine can write with:
  - GGUF files and Hugging Face model folders in `ComfyUI/models/llm` (a `.gguf` runs through llama.cpp, a folder through Transformers);
  - `Ollama: <name>` for every model Ollama has installed, on this computer unless Settings say otherwise;
  - `Server: <id>` for every model on the OpenAI-compatible server (llama-swap, LM Studio, llama.cpp server), once its address is set.

  Both addresses are set in **Settings > DaSiWa > LLM servers**, the same ones the Director's Forge uses (see External LLM servers below).

  The list is read when ComfyUI loads; press R after adding a model. A server that is not running is simply left out of the list.
- `write_for`: Anima, Illustrious, Krea2, Wan 2.2 or LTX 2.3.
- `idea`: what you want, in your own words or as tags.
- `seed`: change it, or let it randomize, for a different take on the same idea.
- `keep_loaded`: off frees the memory after every prompt so the image model has it; on is faster when writing several in a row.
- `picture` (optional): a reference picture for a vision model. With an empty idea, the prompt is written from the picture.

Everything else uses the advanced nodes' defaults, with a 2048-token budget so long prompts are not cut short. For any other setting, use the two advanced nodes below. Both run the same code.

### DaSiWa LLM Model Selector (Advanced)

Creates a lightweight `DASIWA_LLM_CONFIG` bundle. It does not output a live model object, which helps the analyze node unload memory reliably after generation.

Place full Hugging Face-style model folders in:

```text
ComfyUI/models/llm/
```

For the `transformers` backend, the folder must include `config.json`, tokenizer files, processor files for vision models, and `.safetensors` weights. A single `.safetensors` file is not enough for LLM chat inference.

For the `llama_cpp` backend, select a local `.gguf` file. `llama-cpp-python>=0.3.26` is declared as a dependency, but a generic package install does not guarantee a CUDA-enabled build. On a ComfyUI environment using CUDA 13.0, install its prebuilt GPU wheel explicitly with `/path/to/ComfyUI/venv/bin/python -m pip install --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130 'llama-cpp-python>=0.3.26'`. On other CUDA versions, use a matching GPU wheel index or build it with `CMAKE_ARGS="-DGGML_CUDA=on" /path/to/ComfyUI/venv/bin/python -m pip install --force-reinstall --no-cache-dir llama-cpp-python`. The `ollama` backend sends requests only to the local Ollama API at `http://127.0.0.1:11434/api/chat` and uses `ollama_model` as its model name.

Models must already be present on disk before the node runs. Download Hugging Face models with the official tooling, review their provenance, and place the complete folder in `ComfyUI/models/llm`, or set `custom_path` to an existing local folder. Runtime downloads, Hugging Face token reads, and custom remote model code are deliberately unsupported: values supplied through ComfyUI's `/prompt` API must never choose code for the server to download or execute.

Important controls:

- `task`: use `auto` for most workflows. Connect images to use a vision-language model.
- `device`: `auto`, `cuda`, or `cpu`.
- `dtype`: `auto`, `float16`, `bfloat16`, or `float32`.
- `quantization`: optional Transformers model-weight `8bit` or `4bit`, requiring `bitsandbytes`.
- `kv_cache_implementation`: Transformers generation cache strategy. `quantized` reduces long-generation VRAM use; use only with a compatible Transformers cache backend.
- `kv_cache_quant_backend`, `kv_cache_nbits`, and `kv_cache_residual_length`: controls used only for a quantized Transformers KV cache.
- `cache_mode`: `cached` keeps the DaSiWa backend loaded. `unload_after_run` unloads DaSiWa models, requests ComfyUI to unload its managed models, garbage-collects Python objects, and clears the device allocator after each response. For Ollama it additionally sends `keep_alive: 0` so the separate Ollama server releases its model.
- `llama_n_ctx`, `llama_n_gpu_layers`, `llama_n_threads`, and `llama_chat_format`: llama.cpp GGUF controls. `-1` GPU layers requests full offload; `0` uses CPU only.
- `ollama_model` and `ollama_timeout`: local Ollama API controls. The endpoint is fixed to loopback.

### DaSiWa LLM Analyze (Advanced)

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
- `use_kv_cache` is a per-generation toggle for Transformers. Turning it off may reduce peak memory for some models, but generation is slower. The implementation and quantization strategy belong to Model Selector because they are backend configuration.
- `memory_cleanup` uses the same full cleanup path before and/or after the node: DaSiWa model cache, ComfyUI managed models, Python garbage, and the device allocator are cleared. This intentionally makes later image/video models reload rather than retain VRAM/RAM.

## External LLM servers

Choose `openai` or `ollama_server` in Model Selector and enter the server's model ID in `server_model`. Local `model`/`custom_path` are not used for these modes. The original `ollama` mode remains fixed to loopback and uses `ollama_model`.

Addresses are configured outside the workflow, so a downloaded workflow cannot point this machine at a server of its choosing. Set them in **Settings > DaSiWa > LLM servers**: Ollama address, OpenAI-compatible server address and API key. These are the same settings the Director's Forge uses; ComfyUI keeps them in its settings file on this machine (`user/default/comfy.settings.json`), and the nodes read them when they run. Press R after changing them to refresh the Prompt Writer's model list.

An operator can override them in the environment used to start ComfyUI, which wins over Settings:

```fish
set -gx DASIWA_LLM_OPENAI_URL http://127.0.0.1:8041/v1
set -gx DASIWA_LLM_OLLAMA_URL http://127.0.0.1:11434
```

`DASIWA_LLM_OPENAI_API_KEY` supplies authentication when needed. Never put keys in workflow JSON. OpenAI-compatible addresses may include `/v1`; the transport adds it when absent. With neither a setting nor an environment variable, Ollama is looked for on this computer. With ComfyUI's `--multi-user`, the default user's settings are the ones read.

For these server modes, `max_new_tokens`, temperature, top-p, seed and `ollama_timeout` are passed to the shared transport. `llama_n_ctx` also supplies Ollama's context size. Transformers dtype, quantization, KV-cache controls and `max_input_tokens` are local controls; they do not configure remote servers. OpenAI-compatible APIs have no standard repetition-penalty equivalent; a nondefault requested penalty is reported as unsupported rather than mapped to presence penalty.

Cached Ollama-server calls use a five-minute keep-alive; unload-after-run requests zero keep-alive. OpenAI-compatible unload is best-effort using the server's per-model unload endpoint, and `info` reports whether it succeeded. It cannot unload a server that offers no unload API. Remote default unload does not clear ComfyUI's local model cache; explicitly selecting `memory_cleanup` still clears local memory as before.

## PromptForge presets

These new choices request model-specific output and return only the generated prompt, without segment delimiters or explanation panels:

| Preset | Output dialect |
|---|---|
| `promptforge_h3` | H3 description, soundscape and music fields |
| `promptforge_wan22` | Wan 2.2 subject → motion → camera → scene prose |
| `promptforge_ltx` | LTX enhanced paragraph |
| `promptforge_krea2` | Krea2 continuous prose, no negative prompt or weight syntax |
| `promptforge_anima` | Anima positive prompt |
| `promptforge_illustrious` | Illustrious positive prompt |

The legacy Wan/LTX and caption preset IDs and instructions are unchanged. `custom` still uses the system widget; selecting a preset does not append custom instructions. Use the ordinary multiline `prompt` field for the idea, or convert that widget to an input to connect a STRING node. Optional `text_input` is still appended after it, separated by a blank line. Leave either text source empty if only the other should contribute.

For H3, choose `h3_mode` and `h3_duration`. T2VA requires no sampled pictures, I2VA/L2VA one, and FL2VA two in endpoint order. Set `max_frames` accordingly. These are reference endpoints, not arbitrary video analysis frames; do not infer output aspect ratio from them. REF2VA, labelled casts and continuation drafting remain in the Director. The existing 256-token output default is preserved for older workflows; select a larger budget such as 3500 for a full H3 result and sufficient context for its long system instructions.

The Wan, LTX, Krea2, Anima and Illustrious presets are short prompting guides in `data/llm_skills/`, one markdown file per model. Each one says what the job is, how that model's prompts are written, and gives one worked example. The frontmatter names the output segment and, for the tag models, `tag_style: space`, which respells any `best_quality` the model writes out of habit as `best quality` (score tags such as `score_7` and emoticons such as `o_o` keep their underscores). Edit a guide to change how that model is written for; no code change is needed. They are written for small local models, so they stay short and avoid example phrasing a small model would copy.

H3 uses the existing `data/h3_forge.json`, shared with the Director. PromptForge is not needed at runtime, and none of these presets use tools such as tag search.

## Notes

Text-only LLMs can analyze text and prompts. Images require an appropriate vision-language model: a compatible Transformers processor, a GGUF with its matching mmproj projector, or a vision-capable external server. GGUF Analyze discovers a matching `mmproj*.gguf` beside the model using the same pairing logic as Director; ambiguous or missing projectors fail with a clear error. GGUF vision requires llama-cpp-python 0.3.26 or newer with MTMD support. The legacy loopback `ollama` mode remains text-only.

Image compression is intentionally not exposed as a memory option. Lossless compression can preserve file quality, but after the VLM processor decodes the image it does not reduce vision token count or runtime VRAM. Use `max_frames` and `resize_max_px` for image/video memory control.

GGUF loading is provided through llama.cpp, not through ComfyUI diffusion checkpoint loaders. ComfyUI's native `MODEL` / `CLIP` / `VAE` objects represent diffusion models and cannot be used as LLM/VLM Transformers objects.
