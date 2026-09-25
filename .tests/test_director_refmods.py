import json

import pytest
import torch

from nodes import nodes_minimax_h3_director as director
from nodes import nodes_minimax_h3_director_guide as guide_module


def _loaded(name):
    kind = {"picture": "image", "voice": "audio"}.get(name, "video")
    return [(torch.ones(1, 24, 2, 4, 4), {"kind": kind, "name": name, "latent_t": 2, "latent_h": 4, "latent_w": 4})]


def test_refmods_stable_slots_translate_all_prompt_fields(monkeypatch):
    monkeypatch.setattr(director, "load_refmods", _loaded)
    monkeypatch.setattr(director, "refmod_fingerprint", lambda _name: (123, 456))
    state = {"items": [{"type": "video", "value": "native", "slot": 0, "duration": 2}], "refmods": [
        {"slot": 3, "name": "person", "description": "desc <RefMod 3>", "strength": .5, "enabled": True},
        {"slot": 1, "name": "picture", "description": "", "strength": 0, "enabled": True},
    ]}
    fields = {"prompt_mode": "simple", "simple_prompt": "<RefMod 3>", "imd": "<RefMod 3>",
              "soundscape": "<RefMod 3>", "music": "<RefMod 3>", "ref": {
              "subject_definitions": "<RefMod 3>", "summary": "<RefMod 3>",
              "retention_analysis": "<RefMod 3>", "detailed_description": "<RefMod 3>"}}
    result = director.MiniMaxH3Director().build_guide("REF2VA", "<RefMod 3>", 64, 64, 1, "match",
                                                       json.dumps(state), json.dumps(fields),
                                                       external_prompt_overwrite="<RefMod 3>")
    guide = result[0]
    assert guide["resolved_prompt"].startswith("<Video 2>")
    assert "<Video 2>: desc <Video 2>" in guide["resolved_prompt"]
    assert guide["minimax_ref_items"][0]["kind"] == "video"
    assert torch.all(guide["minimax_ref_items"][0]["latent"] == .5)
    assert guide["selection_stamp"] == 123
    assert "<RefMod 3>" not in json.dumps(guide["builder_state"])


def test_refmod_bundle_expands_members_and_inserts_all_resolved_tags(monkeypatch):
    monkeypatch.setattr(director, "load_refmods", lambda _name: [
        (torch.ones(1, 24, 4, 4, 4), {"kind": "video", "name": "slime_visual", "latent_t": 4, "latent_h": 4, "latent_w": 4}),
        (torch.ones(1, 32, 2, 5), {"kind": "audio", "name": "slime_audio", "latent_t": 5}),
    ])
    monkeypatch.setattr(director, "refmod_fingerprint", lambda _name: (7, 11))

    guide = director.MiniMaxH3Director().build_guide(
        "REF2VA", "<RefMod 1>", 64, 64, 1, "match",
        json.dumps({"refmods": [{"slot": 1, "name": "slime_cat_full", "enabled": True}]}),
    )[0]

    assert [item["kind"] for item in guide["minimax_ref_items"]] == ["video", "audio"]
    assert guide["resolved_prompt"].startswith("<Video 1> <Audio 1>")


def test_row_name_and_description_override_file_metadata_for_stamp(monkeypatch):
    monkeypatch.setattr(director, "load_refmods", lambda _name: [(
        torch.ones(1), {"kind": "video", "name": "file-name", "description": "file description"})])
    stamped = []
    monkeypatch.setattr(director, "refmod_fingerprint", lambda name: stamped.append(name) or (7, 11))
    row = {"slot": 1, "name": "folder/selected", "description": "workflow description", "enabled": True}
    built = director.MiniMaxH3Director().build_guide(
        "REF2VA", "<RefMod 1>", 64, 64, 1, "match", json.dumps({"refmods": [row]}))[0]
    assert built["minimax_ref_items"][0]["name"] == "folder/selected"
    assert built["minimax_ref_items"][0]["description"] == "workflow description"
    assert "workflow description" in built["resolved_prompt"]
    assert stamped == ["folder/selected"]


def test_audio_alias_counts_video_soundtracks_and_standalone_audio(monkeypatch):
    monkeypatch.setattr(director, "load_refmods", lambda _name: [(torch.ones(1), {"kind": "audio"})])
    monkeypatch.setattr(director, "refmod_fingerprint", lambda _name: (1, 1))
    state = {"items": [
        {"type": "video", "value": "video", "audio": "soundtrack", "media_mode": "video", "slot": 0, "duration": 2},
        {"type": "audio", "value": "standalone", "slot": 0, "duration": 2},
    ], "refmods": [{"slot": 1, "name": "voice", "enabled": True}]}
    built = director.MiniMaxH3Director().build_guide(
        "REF2VA", "<RefMod 1>", 64, 64, 1, "match", json.dumps(state))[0]
    assert built["resolved_prompt"].startswith("<Audio 3>")


def test_disabled_empty_refmod_row_is_ignored_but_enabled_empty_fails():
    node = director.MiniMaxH3Director()
    disabled = json.dumps({"refmods": [{"slot": 1, "name": "", "enabled": False}]})
    assert "minimax_ref_items" not in node.build_guide("REF2VA", "", 64, 64, 1, "match", disabled)[0]
    enabled = json.dumps({"refmods": [{"slot": 1, "name": "", "enabled": True}]})
    with pytest.raises(ValueError, match="name is required"):
        node.build_guide("REF2VA", "", 64, 64, 1, "match", enabled)


def test_refmod_validation_rejects_duplicate_bad_strength_missing_and_unknown(monkeypatch):
    monkeypatch.setattr(director, "load_refmods", _loaded)
    cases = [
        ([{"slot": 1, "name": "a"}, {"slot": 1, "name": "b"}], "unique"),
        ([{"slot": 9, "name": "a"}], "1 to 8"),
        ([{"slot": 1, "name": "a", "strength": float("nan")}], "between 0 and 1"),
        ([{"slot": 1, "name": ""}], "name is required"),
    ]
    for rows, message in cases:
        with pytest.raises(ValueError, match=message):
            director.MiniMaxH3Director().build_guide("REF2VA", "", 64, 64, 1, "match", json.dumps({"refmods": rows}))
    # Missing files are now warned-and-skipped (not hard errors) — validates the skip path works
    monkeypatch.setattr(director, "load_refmods", lambda name: (_ for _ in ()).throw(ValueError("not found")))
    result = director.MiniMaxH3Director().build_guide("REF2VA", "", 64, 64, 1, "match",
                                                     json.dumps({"refmods": [{"slot": 1, "name": "missing"}]}))
    # No minimax_ref_items since the only refmod was skipped
    assert "minimax_ref_items" not in result[0]


def test_zero_refmods_preserves_legacy_guide_and_native_kwargs(monkeypatch):
    calls = []
    class Native:
        @staticmethod
        def execute(**kwargs):
            calls.append(kwargs)
            return [["embedding", {}]], {"samples": "legacy"}
    monkeypatch.setattr(guide_module, "_native_node", lambda _name: Native)
    guide = director.MiniMaxH3Director().build_guide("REF2VA", "plain", 64, 64, 1, "match", "{}")[0]
    assert "minimax_ref_items" not in guide and "selection_stamp" not in guide
    guide_module.MiniMaxH3DirectorGuide().apply(object(), object(), guide, object())
    assert calls[0]["clip"].__class__ is object


def test_guide_passes_audio_refmod_without_decoding(monkeypatch):
    seen = {}
    class Clip:
        def tokenize(self, text, **kwargs): seen.update(kwargs); return text
        def encode_from_tokens_scheduled(self, tokens): return [["e", {"minimax_token_tags": "t"}]]
    class Vae:
        def decode(self, latent): raise AssertionError("audio must not use video VAE")
    class Native:
        @staticmethod
        def execute(**kwargs):
            token = kwargs["clip"].tokenize(kwargs["prompt"])
            return kwargs["clip"].encode_from_tokens_scheduled(token), {}
    monkeypatch.setattr(guide_module, "_native_node", lambda _: Native)
    guide = {"mode": "REF2VA", "resolved_prompt": "<Audio 1>", "width": 64, "height": 64, "length": 5,
             "minimax_ref_items": [{"kind": "audio", "latent": torch.ones(1, 32, 2, 4), "latent_t": 4}]}
    positive, _, context = guide_module.MiniMaxH3DirectorGuide().apply(Clip(), Vae(), guide, None)
    assert context == {"disabled": True}
    assert seen["minimax_ref_items"] == [{"type": "audio"}]
    assert positive[0][1]["minimax_refs"][0]["audio_latent"].shape == (1, 32, 2, 4)


def test_guide_passes_ref_items_to_tokenize_without_new_socket(monkeypatch):
    seen = {}
    class Clip:
        def tokenize(self, text, **kwargs):
            seen.update(text=text, kwargs=kwargs)
            return text
        def encode_from_tokens_scheduled(self, tokens):
            return [["embedding", {"minimax_token_tags": "tags", "minimax_refs": []}]]
    class Vae:
        def decode(self, latent):
            return torch.ones(2, 8, 8, 3)
    class Native:
        @staticmethod
        def execute(**kwargs):
            tokens = kwargs["clip"].tokenize(kwargs["prompt"], minimax_ref_items=[])
            return kwargs["clip"].encode_from_tokens_scheduled(tokens), {"samples": "ok"}
    monkeypatch.setattr(guide_module, "_native_node", lambda _name: Native)
    guide = {"mode": "REF2VA", "resolved_prompt": "<Video 1>", "width": 64, "height": 64,
             "length": 5, "minimax_ref_items": [{"kind": "video", "latent": torch.ones(1, 24, 2, 4, 4),
             "latent_t": 2, "latent_h": 4, "latent_w": 4}], "selection_stamp": 10}
    positive, _, context = guide_module.MiniMaxH3DirectorGuide().apply(Clip(), Vae(), guide, None)
    assert context == {"disabled": True}
    assert seen["kwargs"]["minimax_ref_items"][0]["type"] == "video"
    assert len(positive[0][1]["minimax_refs"]) == 1
    assert "ref_mods" not in guide_module.MiniMaxH3DirectorGuide.INPUT_TYPES()["optional"]


def test_audio_vae_stays_required_when_native_audio_exists_with_refmod(monkeypatch):
    state = {"mode": "REF2VA", "resolved_prompt": "", "width": 64, "height": 64, "length": 5,
             "ref_audios": {"ref_audio_1": object()},
             "minimax_ref_items": [{"kind": "video", "latent": torch.ones(1, 24, 1, 2, 2)}]}
    with pytest.raises(ValueError, match="audio_vae is required"):
        guide_module.MiniMaxH3DirectorGuide().apply(object(), object(), state, None)


def test_director_and_guide_cache_keys_use_current_file_fingerprint(monkeypatch):
    fingerprints = {"person": (100, 20)}
    monkeypatch.setattr(director, "refmod_fingerprint", lambda name: fingerprints[name])
    monkeypatch.setattr(guide_module, "refmod_fingerprint", lambda name: fingerprints[name])
    timeline = json.dumps({"refmods": [{"slot": 1, "name": "person", "enabled": True}]})
    before = director.MiniMaxH3Director.IS_CHANGED("REF2VA", "", 64, 64, 1, "match", timeline)
    guide = {"minimax_ref_items": [{"name": "person"}]}
    guide_before = guide_module.MiniMaxH3DirectorGuide.IS_CHANGED(None, None, guide)
    fingerprints["person"] = (101, 21)
    assert director.MiniMaxH3Director.IS_CHANGED("REF2VA", "", 64, 64, 1, "match", timeline) != before
    assert guide_module.MiniMaxH3DirectorGuide.IS_CHANGED(None, None, guide) != guide_before
