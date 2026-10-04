"""Single-owner contracts: compatibility names must be identity aliases."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_local_loader_has_node_independent_owner():
    from nodes import nodes_llm
    runtime = importlib.import_module("nodes.llm_runtime")
    assert nodes_llm._load_transformers_model is runtime._load_transformers_model
    assert nodes_llm._LLM_CACHE is runtime._LLM_CACHE
    assert runtime._load_transformers_model.__module__ == "nodes.llm_runtime"


def test_forge_uses_shared_backend_classes():
    from nodes import h3_forge
    backends = importlib.import_module("nodes.llm_backends")
    assert h3_forge.Ollama is backends.Ollama
    assert h3_forge.OpenAICompatible is backends.OpenAICompatible
    assert h3_forge.Local is backends.Local
    assert h3_forge.ForgeError is backends.ForgeError
    assert backends.Local()._llm().__name__ == "nodes.llm_runtime"


def test_h3_prompt_builder_is_not_owned_by_director():
    from nodes import h3_forge
    prompts = importlib.import_module("nodes.h3_prompting")
    assert h3_forge.build_user_message is prompts.build_user_message
    assert h3_forge.parse_segments is prompts.parse_segments
    assert h3_forge.simple_prompt is prompts.simple_prompt
    assert h3_forge.load_bundle is prompts.load_bundle
    assert h3_forge._REF_FIELDS is prompts._REF_FIELDS
    assert h3_forge.EASY_MODE == prompts.EASY_MODE
    assert h3_forge.BASE_MODES == prompts.BASE_MODES


def test_legacy_presets_have_one_owner():
    from nodes import nodes_llm
    prompts = importlib.import_module("nodes.llm_prompt_presets")
    assert nodes_llm._SYSTEM_PROMPT_PRESETS is prompts._SYSTEM_PROMPT_PRESETS
    assert nodes_llm._compose_user_text is prompts._compose_user_text
    assert prompts._compose_user_text.__module__ == "nodes.llm_prompt_presets"
