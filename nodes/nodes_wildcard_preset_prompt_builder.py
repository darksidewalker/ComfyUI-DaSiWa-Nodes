"""Wildcard & Preset Prompt Builder backend and JSON metadata endpoint."""

import json
import re
import secrets
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    PromptServer = None
    web = None


DATA_PATH = Path(__file__).parents[1] / "data" / "wildcards_and_presets_dual.json"
_BRACE_GROUP_RE = re.compile(r"\{([^{}]*)\}")
_TOKEN_RE = re.compile(r"\w+(?:['’-]\w+)?|[^\s\w]", re.UNICODE)


def _load_library() -> dict[str, list[dict[str, Any]]]:
    parsed = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    categories = parsed.get("wildcard_library", {}).get("categories")
    if not isinstance(categories, dict):
        raise ValueError("Wildcard library has no valid wildcard_library.categories object.")
    return categories


def is_negative_category(category: str) -> bool:
    normalized = category.casefold()
    return "negative prompt" in normalized


def _subject_key(category: str, subject: dict[str, Any], entry_type: str) -> str:
    return f"{category}/{subject.get('subject', '')}/{entry_type}"


def _hash_seed(*parts: object) -> int:
    """FNV-1a over UTF-8; mirrored exactly by the frontend preview."""
    value = 0x811C9DC5
    for byte in "\x1f".join(str(part) for part in parts).encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def _choice_index(option_count: int, seed: int, subject_key: str, reroll: int, group_index: int) -> int:
    return _hash_seed(seed, subject_key, reroll, group_index) % option_count


def resolve_pattern(pattern: str, seed: int, subject_key: str, reroll: int = 0) -> str:
    group_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal group_index
        options = match.group(1).split("|")
        if not options:
            return ""
        result = options[_choice_index(len(options), seed, subject_key, reroll, group_index)]
        group_index += 1
        return result

    return _BRACE_GROUP_RE.sub(replace, str(pattern))


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


_VAR_TOKEN_RE = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_VAR_LIBRARY_MAX_CHARS = 1_000_000
_VAR_SELECTIONS_MAX_CHARS = 256_000


def parse_var_library(raw: object) -> dict[str, list[str]]:
    if isinstance(raw, str):
        if len(raw) > _VAR_LIBRARY_MAX_CHARS:
            raise ValueError(f"Variable library exceeds {_VAR_LIBRARY_MAX_CHARS} characters.")
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(f"Variable library is not valid JSON: {error}") from error
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ValueError('The variable library must be a JSON object: {"color": ["red", "blue"]}.')
    for name, options in parsed.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
            raise ValueError(f"Invalid variable name: {name}")
        if not isinstance(options, list) or not options or any(not isinstance(x, str) or not x.strip() for x in options):
            raise ValueError(f"${name} must contain a non-empty list of strings.")
    return {str(name): list(options) for name, options in parsed.items()}


def _var_selections(raw: object) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, str)}
    if isinstance(raw, str):
        if len(raw) > _VAR_SELECTIONS_MAX_CHARS:
            raise ValueError(f"Variable selections exceed {_VAR_SELECTIONS_MAX_CHARS} characters.")
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Variable selections are not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("Variable selections must be a JSON object: {\"color:0\": \"red\"}.")
    return {str(k): v for k, v in parsed.items() if isinstance(v, str)}


def resolve_variables(template: str, library: dict[str, list[str]], selections: dict[str, str], seed: int) -> str:
    """Resolve $name / ${name} tokens; $$ is a literal dollar. Each occurrence resolves
    independently (key name:index), pinning wins over the seed-hashed pick, and unknown
    tokens stay in the text. Mirrored exactly by DEAS's golden vectors."""
    counts: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        if match.group() == "$$":
            return "$"
        name = match.group(1) or match.group(2)
        if name not in library:
            return match.group()
        index = counts.get(name, 0)
        counts[name] = index + 1
        key = f"{name}:{index}"
        options = library[name]
        selected = selections.get(key)
        if selected in options:
            return selected
        return options[_hash_seed(seed, key) % len(options)]

    return _VAR_TOKEN_RE.sub(replace, str(template))


def _normalized_weight(value: object) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 1.0


def _format_weighted(text: str, weight: float) -> str:
    if weight == 1.0:
        return text
    return f"({text}:{weight:g})"


def _prepend_prompt(prefix: object, generated: str) -> str:
    parts = [str(value).strip() for value in (prefix, generated) if value and str(value).strip()]
    return ", ".join(parts)


def _selection_state(raw: object) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_prompts(
    library: dict[str, list[dict[str, Any]]],
    style: str,
    seed: int,
    reroll: int,
    token_budget: int,
    selection_state: object,
) -> tuple[str, str, int, int]:
    state = _selection_state(selection_state)
    key_prefix = "nl" if style == "Natural Language" else "booru"
    groups: dict[bool, list[dict[str, Any]]] = {False: [], True: []}

    for category, subjects in library.items():
        for subject in subjects:
            for entry_type in ("wildcards", "presets"):
                key = _subject_key(category, subject, entry_type)
                config = state.get(key, {})
                if not isinstance(config, dict) or not config.get("enabled", False):
                    continue
                entries = subject.get(f"{key_prefix}_{entry_type}", [])
                resolved = [resolve_pattern(entry, seed, key, reroll) for entry in entries if str(entry).strip()]
                text = ", ".join(part for part in resolved if part)
                if not text:
                    continue
                weight = _normalized_weight(config.get("weight", 1.0))
                groups[is_negative_category(category)].append({
                    "text": _format_weighted(text, weight),
                    "weight": weight,
                    "index": len(groups[is_negative_category(category)]),
                })

    def limit(items: list[dict[str, Any]]) -> tuple[str, int]:
        kept = list(items)
        while kept and count_tokens(", ".join(item["text"] for item in kept)) > token_budget:
            lowest = min(kept, key=lambda item: (item["weight"], -item["index"]))
            kept.remove(lowest)
        prompt = ", ".join(item["text"] for item in kept)
        return prompt, count_tokens(prompt)

    positive, positive_tokens = limit(groups[False])
    negative, negative_tokens = limit(groups[True])
    return positive, negative, positive_tokens, negative_tokens


class DaSiWa_WildcardPresetPromptBuilder:
    DESCRIPTION = (
        "Builds positive and negative prompts from the bundled dual-style wildcard library. "
        "Selections and weights are stored in the workflow."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (["Booru", "Natural Language"], {"default": "Booru"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "token_budget": ("INT", {"default": 200, "min": 1, "max": 10000}),
                "selection_state": ("STRING", {"default": "{}", "multiline": True}),
                "reroll": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "auto_reroll": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
            },
            "optional": {
                "positive_input": ("STRING", {"forceInput": True}),
                "negative_input": ("STRING", {"forceInput": True}),
                "variables_json": ("STRING", {"default": "{}", "multiline": True}),
                "variable_selections": ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "build"
    CATEGORY = "DaSiWa/Text"

    @classmethod
    def IS_CHANGED(cls, style, seed, token_budget, selection_state, reroll, auto_reroll=False,
                   positive_input=None, negative_input=None, variables_json="{}", variable_selections="{}"):
        if auto_reroll:
            # A newly allocated NaN is essential: ComfyUI cache keys are tuples and
            # Python treats tuples containing the same ``math.nan`` singleton as
            # equal by identity.  Fresh NaNs prevent Preview-as-Text selected-output
            # executions from reusing the prior builder output.
            return float("nan")
        return (style, seed, token_budget, selection_state, reroll, positive_input, negative_input,
                variables_json, variable_selections)

    def build(self, style, seed, token_budget, selection_state, reroll, auto_reroll=False,
              positive_input=None, negative_input=None, variables_json="{}", variable_selections="{}"):
        effective_reroll = secrets.randbits(31) if auto_reroll else reroll
        # Prompt variables resolve on the fixed seed: auto_reroll re-rolls only the
        # generated preset side, never the $var picks.
        library = parse_var_library(variables_json)
        selections = _var_selections(variable_selections)
        positive, negative, _, _ = build_prompts(
            _load_library(), style, seed, effective_reroll, token_budget, selection_state
        )
        return (
            _prepend_prompt(resolve_variables(positive_input or "", library, selections, seed), positive),
            _prepend_prompt(resolve_variables(negative_input or "", library, selections, seed), negative),
        )


if PromptServer is not None:
    @PromptServer.instance.routes.get("/dasiwa/wildcard-preset-prompt-builder/library")
    async def wildcard_preset_library(request):
        try:
            return web.json_response({
                "categories": _load_library(),
            })
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=500)

    @PromptServer.instance.routes.post("/dasiwa/wildcard-preset-prompt-builder/build")
    async def wildcard_preset_build(request):
        """Resolve the node's inputs outside the graph, e.g. to seed an LLM prompt writer.
        With positive_input/negative_input templates plus variables_json/variable_selections
        it also returns the fully resolved final prompts (*_base = templates with $vars only,
        positive/negative = templates + generated presets, identical to a node run)."""
        try:
            rules = await request.json()
            seed = int(rules.get("seed", 0))
            library = parse_var_library(rules.get("variables_json", "{}"))
            selections = _var_selections(rules.get("variable_selections", "{}"))
            positive_base = resolve_variables(str(rules.get("positive_input", "") or ""), library, selections, seed)
            negative_base = resolve_variables(str(rules.get("negative_input", "") or ""), library, selections, seed)
            positive, negative, _, _ = build_prompts(
                _load_library(), str(rules.get("style", "Booru")), seed,
                int(rules.get("reroll", 0)), int(rules.get("token_budget", 200)), rules.get("selection_state"),
            )
            return web.json_response({
                "positive": _prepend_prompt(positive_base, positive),
                "negative": _prepend_prompt(negative_base, negative),
                "positive_base": positive_base,
                "negative_base": negative_base,
            })
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)
