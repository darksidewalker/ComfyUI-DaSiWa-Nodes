"""Prompt variables: $name/${name}/$$ resolve seed-deterministically, in the node and the route."""
import asyncio
import importlib
import sys
import types

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def _module(monkeypatch):
    routes = web.RouteTableDef()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(
        PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))))
    return routes, importlib.reload(importlib.import_module("nodes.nodes_wildcard_preset_prompt_builder"))


def test_resolver_golden_vectors(monkeypatch):
    _, module = _module(monkeypatch)
    library = {"color": ["red", "blue", "green"], "hairstyle": ["bob", "ponytail"]}
    # Determinism: same seed -> same text, token order stable.
    a = module.resolve_variables("$color and $color", library, {}, 42)
    b = module.resolve_variables("$color and $color", library, {}, 42)
    assert a == b
    picks = a.split(" and ")
    assert picks[0] in library["color"] and picks[1] in library["color"]
    # Occurrence independence: the same name twice resolves to two independent
    # picks (all three values reachable over many seeds, not one per template).
    seen = {module.resolve_variables("$color/$color", library, {}, seed) for seed in range(50)}
    assert len({tuple(text.split("/")) for text in seen}) > len(library["color"])
    # $$ is a literal dollar; ${name} delimits; unknown names stay verbatim.
    assert module.resolve_variables("$$100", library, {}, 1) == "$100"
    assert module.resolve_variables("${color}ball", library, {}, 1).startswith(("red", "blue", "green"))
    assert module.resolve_variables("$wether $unknown", library, {}, 1) == "$wether $unknown"
    # Pinned selections beat the seed pick and survive any seed.
    pinned = module.resolve_variables("$color/$color", library, {"color:0": "green"}, 999)
    assert pinned.startswith("green/")
    assert pinned[6:] in library["color"]
    # A pinned value outside the library falls back to the seed pick.
    assert module.resolve_variables("$color", library, {"color:0": "magenta"}, 5) in library["color"]


def test_library_validation(monkeypatch):
    _, module = _module(monkeypatch)
    assert module.parse_var_library('{"color": ["red"]}') == {"color": ["red"]}
    for raw in ('{"9bad": ["x"]}', '{"a": []}', '{"a": ["", " "]}', '"text"', '{"a": "x"}'):
        try:
            module.parse_var_library(raw)
        except ValueError:
            continue
        raise AssertionError(f"parse_var_library accepted {raw}")


def test_node_and_route_resolve_variables(monkeypatch):
    routes, module = _module(monkeypatch)
    library = {"Art": [{"subject": "Light", "booru_wildcards": ["{red|blue} light"]}]}
    monkeypatch.setattr(module, "_load_library", lambda: library)
    variables = {"color": ["crimson", "azure"], "location": ["beach"]}
    rules = {"style": "Booru", "seed": 11, "reroll": 0, "token_budget": 200,
             "selection_state": {"Art/Light/wildcards": {"enabled": True, "weight": 1}},
             "positive_input": "a $color cat at the $location",
             "negative_input": "$unknowntoken",
             "variables_json": '{"color": ["crimson", "azure"], "location": ["beach"]}',
             "variable_selections": '{"color:0": "crimson"}'}

    def node_result():
        node = module.DaSiWa_WildcardPresetPromptBuilder()
        return node.build(rules["style"], rules["seed"], rules["token_budget"], rules["selection_state"],
                          rules["reroll"], False, rules["positive_input"], rules["negative_input"],
                          rules["variables_json"], rules["variable_selections"])

    positive, negative = node_result()
    assert positive.startswith("a crimson cat at the beach, ")
    assert positive.endswith(" light")
    assert negative == "$unknowntoken"

    async def exercise():
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/dasiwa/wildcard-preset-prompt-builder/build", json=rules)
            assert response.status == 200
            payload = await response.json()
            # Route equals the node run; *_base carries the variables-only text.
            assert (payload["positive"], payload["negative"]) == (positive, negative)
            assert payload["positive_base"] == "a crimson cat at the beach"
            assert payload["negative_base"] == "$unknowntoken"
            # Without the optional fields the legacy shape keeps working.
            legacy = await client.post("/dasiwa/wildcard-preset-prompt-builder/build",
                                       json={k: v for k, v in rules.items() if k not in
                                             ("positive_input", "negative_input", "variables_json", "variable_selections")})
            assert legacy.status == 200
            payload = await legacy.json()
            assert payload["positive_base"] == ""
            assert payload["positive"] in ("red light", "blue light")
            broken = await client.post("/dasiwa/wildcard-preset-prompt-builder/build",
                                       json=dict(rules, variables_json="{oops"))
            assert broken.status == 400

    asyncio.run(exercise())
