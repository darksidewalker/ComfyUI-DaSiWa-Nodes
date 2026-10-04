"""DaSiWa LLM Prompt Writer: the few choices most people need, on top of the advanced nodes.

Builds the same config the Model Selector would, with its defaults, and runs the
Analyze node's code path, so every backend, unload rule and picture handling
stays in one place.
"""

import os

from .llm_runtime import _list_llm_models, _resolve_model_path
from .nodes_llm import DaSiWa_LLMAnalyze

# What the person picks, and the preset that writes it.
WRITE_FOR = {
    "Anima": "promptforge_anima",
    "Illustrious": "promptforge_illustrious",
    "Krea2": "promptforge_krea2",
    "Wan 2.2": "promptforge_wan22",
    "LTX 2.3": "promptforge_ltx",
}

# Room for a long prompt plus a thinking model's reasoning; the advanced
# node's 256 cuts a full Anima or Krea2 prompt short.
MAX_NEW_TOKENS = 2048
PICTURE_ONLY = "Write the prompt from the attached picture."


def simple_config(model, server_model, keep_loaded):
    """The Model Selector's output for this choice, with its defaults."""
    server_model = str(server_model or "").strip()
    if server_model:
        # An operator-set OpenAI-compatible address wins; otherwise Ollama, on loopback by default.
        backend = "openai" if os.environ.get("DASIWA_LLM_OPENAI_URL", "").strip() else "ollama_server"
        model_path = server_model
    else:
        if not model or model == "None":
            raise ValueError(
                "No model chosen. Put a GGUF file or a Hugging Face model folder in ComfyUI/models/llm, "
                "restart ComfyUI and pick it here, or type a model name in server_model to use Ollama."
            )
        backend = "llama_cpp" if str(model).lower().endswith(".gguf") else "transformers"
        model_path = _resolve_model_path(model, "", allow_gguf=backend == "llama_cpp")
    return {
        "model_path": model_path,
        "backend": backend,
        "task": "auto",
        "device": "auto",
        "dtype": "auto",
        "quantization": "none",
        "cache_mode": "cached" if keep_loaded else "unload_after_run",
        "attention_implementation": "auto",
        "kv_cache_implementation": "default",
        "kv_cache_quant_backend": "quanto",
        "kv_cache_nbits": 4,
        "kv_cache_residual_length": 128,
        "llama_n_ctx": 8192,
        "llama_n_gpu_layers": -1,
        "llama_n_threads": 0,
        "llama_chat_format": "",
        "ollama_timeout": 300,
    }


class DaSiWa_LLMPromptWriter:
    DESCRIPTION = (
        "DaSiWa LLM Prompt Writer: type an idea, pick the image or video model it is for, and get "
        "a prompt written in that model's style. For every setting, use the Advanced LLM nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_llm_models(), {"description": "A GGUF or model folder from ComfyUI/models/llm. A vision model can also read a connected picture."}),
                "write_for": (list(WRITE_FOR), {"default": "Anima", "description": "The image or video model the prompt is for."}),
                "idea": ("STRING", {"default": "", "multiline": True, "description": "What you want in the picture, in your own words or as tags."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True, "description": "Change it for a different take on the same idea."}),
                "keep_loaded": ("BOOLEAN", {"default": False, "description": "Off frees the memory after every prompt so the image model has it. On is faster for repeated prompts."}),
            },
            "optional": {
                "picture": ("IMAGE", {"description": "Optional reference picture. Needs a vision model."}),
                "server_model": ("STRING", {"default": "", "description": "Optional: a model name on your Ollama (or operator-configured) server, used instead of a local model."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "write"
    CATEGORY = "DaSiWa/LLM"

    def write(self, model, write_for, idea, seed, keep_loaded, picture=None, server_model=""):
        idea = str(idea or "").strip()
        if not idea and picture is None:
            raise ValueError("Type an idea, or connect a picture to write the prompt from.")
        prompt, _ = DaSiWa_LLMAnalyze().analyze(
            llm_config=simple_config(model, server_model, keep_loaded),
            system_prompt_preset=WRITE_FOR[write_for],
            system_prompt="",
            prompt=idea or PICTURE_ONLY,
            max_new_tokens=MAX_NEW_TOKENS,
            max_input_tokens=0,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.0,
            use_kv_cache=True,
            seed=seed,
            max_frames=1,
            frame_stride=1,
            frame_strategy="first",
            resize_max_px=768,
            resize_algorithm="lanczos",
            memory_cleanup="off",
            images=picture,
            text_input="",
        )
        return (prompt,)
