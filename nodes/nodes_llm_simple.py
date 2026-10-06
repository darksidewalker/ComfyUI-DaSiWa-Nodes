"""LLM Prompt Writer: the few choices most people need, on top of the advanced nodes.

Builds the same config the Model Selector would, with its defaults, and runs the
Analyze node's code path, so every backend, unload rule and picture handling
stays in one place.
"""

import json
from pathlib import Path

from .llm_backends import ForgeError, parse_server_choice, server_model_choices
from .llm_prompt_presets import preset_spec
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
DEFAULT_MAX_TOKENS = 2048
PICTURE_ONLY = "Write the prompt from the attached pictures."
# A batch goes in whole up to this many, Analyze's own default; a longer one
# (a loaded video) is sampled evenly, always starting with the first image.
MAX_IMAGES = 8

# Detail and creativity, 1-10 each; 5 is standard and adds no line, so the
# guide decides. Detail is how much gets written, creativity how far past the
# idea the writer may go - two different things, so two sliders.
STANDARD = 5

# Kept inside the guide's own length, so Wan's and LTX's word bands still
# hold. A first wording, not yet measured.
DETAIL_RULES = {
    1: "as short as it can be. Only what the idea names, at the very bottom of your length.",
    2: "very short. The idea plus one or two essentials.",
    3: "short. The idea plus the few details it needs.",
    4: "a little shorter than usual.",
    6: "a little more than usual: one more detail of light, material or setting.",
    7: "detailed. Add more of the light, materials, setting and camera.",
    8: "very detailed. Cover the light, materials, setting, camera and the small touches.",
    9: "richly detailed, toward the top of your length.",
    10: "as detailed as the format allows: every visible part of the picture, at the top of your length.",
}

# PromptForge's ten Creativity presets, Literal to Wild: its temperatures and
# its rules, as used on these same models there.
CREATIVITY = {
    1: (0.30, "Add nothing. Put exactly what the person wrote into this model's dialect and stop. If they left something out it stays out - no setting, no light, no mood, no wardrobe that is not already in the idea."),
    2: (0.45, "Stay close to what the person wrote. Put what they named into this model's dialect and add only what the image cannot be built without. Do not invent a setting, a mood or a wardrobe they did not ask for."),
    3: (0.55, "Follow the idea closely, and fill only the gaps that would otherwise leave the image undefined - where this happens, what the light is doing. Take the most ordinary answer available and move on."),
    4: (0.62, "Complete the scene they described using conventional, expected choices. Every addition should be one they would have written themselves had they thought of it."),
    5: (0.70, None),
    6: (0.78, "Fill the gaps with choices that have some character to them rather than the safest one. You may add a supporting element they never mentioned, as long as the scene they described is still the subject."),
    7: (0.85, "Bring your own ideas to the setting, the light and the framing. Add elements they never asked for where they make the image stronger, and let the atmosphere be a deliberate choice rather than a default."),
    8: (0.92, "Expand freely. Add atmosphere, framing and supporting detail they never asked for, as far as it makes the image better. Nothing you add may contradict what they did write."),
    9: (1.00, "Push well past the idea. Reach for a striking setting, a strong light and an unusual angle rather than a plausible one. The subject and the action they named are fixed; treat everything else as an invitation."),
    10: (1.10, "Treat the idea as a starting point and build well past it. Unexpected detail, strong atmosphere and bold framing are wanted here. Only the subject and the action they named are fixed; everything around them is yours."),
}

# Styles: one label list for every model. Each label carries tags for the tag
# models and descriptive words for the prose ones (PromptForge's Anima and
# Krea2 lists); a label is a payload, never just its name.
NO_STYLE = "None"


def load_styles():
    path = Path(__file__).resolve().parents[1] / "data" / "llm_styles.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["styles"]


def style_line(style, spec):
    """PromptForge's style instruction, worded for the model's dialect."""
    if not style or style == NO_STYLE:
        return ""
    entry = load_styles().get(style)
    if not entry:
        return ""
    if spec.get("tag_style"):
        return f"Style: {style} - include these tags: {', '.join(entry['tags'] or entry['words'])}"
    return f"Style: {style} - carry this look through the description: {', '.join(entry['words'] or entry['tags'])}"


# Server models sit in the one model list as "Ollama: name" and "Server: id".
# Addresses are operator-configured; Settings require explicit opt-in, never the graph.
NO_MODEL = "None"


def model_choices():
    local = [m for m in _list_llm_models() if m != NO_MODEL]
    return (local + server_model_choices()) or [NO_MODEL]


def request_text(idea, detail, creativity, style_text=""):
    """The idea, then the style, creativity and detail lines that apply."""
    lines = [idea or PICTURE_ONLY]
    if style_text:
        lines.append(style_text)
    rule = CREATIVITY[int(creativity)][1]
    if rule:
        lines.append(f"Creativity {int(creativity)} of 10: {rule}")
    if DETAIL_RULES.get(int(detail)):
        lines.append(f"Detail {int(detail)} of 10: {DETAIL_RULES[int(detail)]}")
    return "\n\n".join(lines)


def assemble(prompt, spec, start_with):
    """The person's own text exactly as typed, then the written prompt. On a
    tag model, a tag already up front is dropped from the written part so
    nothing appears twice."""
    pinned = str(start_with or "")
    if not pinned:
        return prompt
    if spec.get("tag_style"):
        # Only normalize for comparison; the user's prefix is never rewritten.
        same = lambda t: t.strip().lower().replace("_", " ")
        taken = {same(t) for t in pinned.split(",") if t.strip()}
        prompt = ", ".join(p.strip() for p in prompt.split(",") if p.strip() and same(p) not in taken)
    if not prompt:
        return pinned
    # Reuse a trailing comma or line break rather than adding a second one.
    if pinned.endswith(("\n", "\r")) or not pinned.strip():
        separator = ""
    elif pinned.rstrip().endswith(","):
        separator = "" if pinned[-1].isspace() else " "
    else:
        separator = ", "
    return pinned + separator + prompt


def simple_config(model, keep_loaded):
    """The Model Selector's output for this choice, with its defaults."""
    model = str(model or "")
    server = parse_server_choice(model)
    if server:
        backend, model_path = server
    elif not model or model == NO_MODEL:
        raise ValueError(
            "No model found. Put a GGUF file or a Hugging Face model folder in ComfyUI/models/llm, "
            "or start Ollama and pull a model, then press R in ComfyUI to refresh the list."
        )
    else:
        backend = "llama_cpp" if model.lower().endswith(".gguf") else "transformers"
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
        "LLM Prompt Writer: type an idea, pick the image or video model it is for, and get "
        "a prompt written in that model's style. For every setting, use the Advanced LLM nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_choices(), {"description": "Models in ComfyUI/models/llm plus cached Ollama and OpenAI-compatible server models. Workflow servers use the service environment; Settings require DASIWA_LLM_ALLOW_SETTINGS=1 on a trusted single-user installation. Discovery runs in the background; press R afterwards. A vision model can read connected images."}),
                "write_for": (list(WRITE_FOR), {"default": "Anima", "description": "The image or video model the prompt is for."}),
                "start_with": ("STRING", {"default": "", "multiline": True, "placeholder": "Quality tags and LoRA trigger words go here. Always first, exactly as typed.", "description": "Quality tags, LoRA trigger words or anything else that must lead the prompt. Put at the very top exactly as typed; the model never sees or changes it."}),
                "idea": ("STRING", {"default": "", "multiline": True, "placeholder": "Write your idea here", "description": "What you want in the picture, in your own words or as tags."}),
                "style": ([NO_STYLE, *load_styles()], {"default": NO_STYLE, "description": "A look to carry through the prompt. Tag models get matching tags, prose models a description of the look. None adds nothing."}),
                "detail": ("INT", {"default": STANDARD, "min": 1, "max": 10, "step": 1, "display": "slider", "description": "How much gets written. 5 is standard; lower is shorter, higher covers more."}),
                "creativity": ("INT", {"default": STANDARD, "min": 1, "max": 10, "step": 1, "display": "slider", "description": "How far past your idea the writer may go. 1 adds nothing, 5 is balanced, 10 builds well past it."}),
                "max_tokens": ("INT", {"default": DEFAULT_MAX_TOKENS, "min": 128, "max": 8192, "step": 64, "description": "The most the model may write. Lower it to stop a model that runs on; set it too low and the prompt stops mid-sentence."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True, "description": "Change it for a different take on the same idea."}),
                "keep_loaded": ("BOOLEAN", {"default": False, "description": "Off frees the memory after every prompt so the image model has it. On is faster for repeated prompts."}),
            },
            "optional": {
                "images": ("IMAGE", {"description": "Optional reference pictures, one image or a batch, as on the Analyze node. Needs a vision model. Up to 8 are sent; a longer batch is sampled evenly from the first image to the last. For Wan and LTX the first image is the first frame."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "write"
    CATEGORY = "DaSiWa/LLM"

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        # A server model saved in a workflow is checked when the node runs,
        # where a stopped Ollama gets a clear message instead of "not in list".
        return True

    def write(self, model, write_for, idea, seed, keep_loaded, start_with="", style=NO_STYLE, detail=STANDARD,
              creativity=STANDARD, max_tokens=DEFAULT_MAX_TOKENS, images=None):
        idea = str(idea or "").strip()
        if not idea and images is None:
            raise ValueError("Type an idea, or connect a picture to write the prompt from.")
        config = simple_config(model, keep_loaded)
        preset = WRITE_FOR[write_for]
        spec = preset_spec(preset)
        try:
            prompt = self._analyze(config, preset, request_text(idea, detail, creativity, style_line(style, spec)),
                                   CREATIVITY[int(creativity)][0], max_tokens, seed, images)
        except ForgeError as exc:
            if exc.code != "connection":
                raise
            where = "Ollama" if config["backend"] == "ollama_server" else "the model server"
            raise ValueError(f"Could not reach {where}. Start it, check {config['model_path']} "
                             "is still installed, and press R to refresh the model list.") from None
        return (assemble(prompt, spec, start_with),)

    @staticmethod
    def _analyze(config, preset, text, temperature, max_tokens, seed, images):
        prompt, _ = DaSiWa_LLMAnalyze().analyze(
            llm_config=config,
            system_prompt_preset=preset,
            system_prompt="",
            prompt=text,
            max_new_tokens=max_tokens,
            max_input_tokens=0,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.0,
            use_kv_cache=True,
            seed=seed,
            max_frames=MAX_IMAGES,
            frame_stride=1,
            frame_strategy="evenly_spaced",
            resize_max_px=768,
            resize_algorithm="lanczos",
            memory_cleanup="off",
            images=images,
            text_input="",
        )
        return prompt
