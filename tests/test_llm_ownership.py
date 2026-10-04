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
