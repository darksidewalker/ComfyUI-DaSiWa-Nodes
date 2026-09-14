import copy
import json
from types import SimpleNamespace

import pytest
import torch

from nodes import helper_director_refmods as refs
from nodes.nodes_minimax_h3_director import MiniMaxH3Director
from nodes import nodes_minimax_h3_director_guide as guide_module


def mod(name, slot, kind="video"):
    value = SimpleNamespace(name=name, director_slot=slot, kind=kind, description=name,
                            latent=torch.ones(1, 24, 2 if kind == "video" else 1, 4, 4))
    def block(strength):
        return {"kind": kind, "latent": value.latent * strength, "latent_h": 4, "latent_w": 4,
                **({"latent_t": 2, "ref_audio_t": 0, "audio_latent": None} if kind == "video" else {})}
    value.ref_block=block
    return value


def test_two_people_and_video_keep_distinct_slots():
    mods=[(mod("Bilge",1),1), (mod("Ahmet",3),1)]
    tags,_,_,_=refs.mappings(mods, {}, {"video": object()})
    assert refs.translate("<RefMod 1> greets <RefMod 3> in <Video 1>.",tags) == "<Video 2> greets <Video 3> in <Video 1>."
    with pytest.raises(ValueError,match="no active reference"):
        refs.translate("<RefMod 2>", tags)


def test_zero_strength_does_not_shift_other_person():
    tags,*_=refs.mappings([(mod("A",1),0),(mod("B",2),1)],{}, {})
    assert tags=={2:"<Video 1>"}


def test_same_file_slots_do_not_mutate_cached_object(monkeypatch):
    cached=mod("stored",7)
    monkeypatch.setattr(refs,"refmod_nodes",lambda:SimpleNamespace(_load_mod=lambda name:cached))
    loaded=refs.load_selections([{"slot":1,"name":"same","description":"A"},{"slot":2,"name":"same","description":"B"}])
    assert [m.custom_desc for m,s in loaded]==["A","B"]
    assert cached.director_slot==7 and not hasattr(cached,"custom_desc")
    assert loaded[0][0] is not loaded[1][0]


def test_director_persists_descriptions_and_translates_simple_prompt(monkeypatch):
    import nodes.nodes_minimax_h3_director as director_module
    selected=[(mod("Bilge",1),1),(mod("Ahmet",2),1)]
    monkeypatch.setattr(director_module,"load_selections",lambda rows:selected)
    builder={"prompt_mode":"simple","simple_prompt":"<RefMod 1> greets <RefMod 2>.","ref":{}}
    result=MiniMaxH3Director().build_guide("REF2VA","",64,64,1,"match",json.dumps({"refmods":[{"slot":1}]}),json.dumps(builder))
    assert "<Video 1> greets <Video 2>." in result[2]
    assert "<Video 1>: Bilge" in result[2] and "<Video 2>: Ahmet" in result[2]
    assert result[0]["ref_mods"]==selected
    assert "<RefMod 1>" in builder["simple_prompt"]


def test_legacy_refmod_sockets_and_nodes_are_not_registered():
    from nodes.nodes_minimax_h3_director_guide import MiniMaxH3DirectorGuide, NODE_CLASS_MAPPINGS
    assert "ref_mods" not in MiniMaxH3Director.INPUT_TYPES()["optional"]
    assert "ref_mods" not in MiniMaxH3DirectorGuide.INPUT_TYPES()["optional"]
    assert "MiniMaxH3DirectorRefModGuide" not in NODE_CLASS_MAPPINGS


def test_guide_uses_matching_qwen_and_native_reference_order(monkeypatch):
    seen={}
    native_block={"kind":"video","latent_t":2,"latent_h":4,"latent_w":4,"latent":torch.zeros(1,24,2,4,4)}
    class Native:
        @staticmethod
        def execute(**kwargs):
            tokens=kwargs["clip"].tokenize(kwargs["prompt"],minimax_ref_items=[{"type":"video","data":"native"}])
            cond=kwargs["clip"].encode_from_tokens_scheduled(tokens)
            cond[0][1]["minimax_refs"]=[native_block]
            return cond,{"samples":"native output"}
    class Clip:
        def tokenize(self,text,**kwargs): seen.update(text=text,items=kwargs["minimax_ref_items"]); return text
        def encode_from_tokens_scheduled(self,tokens): return [["embedding",{"minimax_token_tags":"tags","sentinel":12}]]
    class Vae:
        def decode(self,z): return torch.ones(1,5,8,8,3)*float(z.mean())
    monkeypatch.setattr(guide_module,"_native_node",lambda name:Native)
    data={"mode":"REF2VA","resolved_prompt":"<RefMod 1> and <RefMod 2>","ref_videos":{"ref_video_1":"native"},"ref_mods":[(mod("A",1),1),(mod("B",2),.5)]}
    positive,latent,mapping=guide_module.execute_director_guide(Clip(),Vae(),data)
    assert seen["text"]=="<Video 2> and <Video 3>"
    assert [i["type"] for i in seen["items"]]==["video"]*3
    assert seen["items"][0]["data"]=="native"
    blocks=positive[0][1]["minimax_refs"]
    assert blocks[0] is native_block and len(blocks)==3
    assert blocks[1]["latent"].mean()==1 and blocks[2]["latent"].mean()==.5
    assert positive[0][1]["sentinel"]==12 and latent=={"samples":"native output"}
    with pytest.raises(ValueError,match="Combined reference"):
        guide_module.execute_director_guide(Clip(),Vae(),data,max_total_tokens=16)


def test_visual_mod_allows_raw_audio_validation():
    from nodes.helper_minimax_h3_director import validate_reference_limits
    validate_reference_limits(audios=[{"duration":2}],external_visual=True)
    with pytest.raises(ValueError,match="accompanied"):
        validate_reference_limits(audios=[{"duration":2}])


def test_bad_slots_and_strengths_fail(monkeypatch):
    monkeypatch.setattr(refs,"refmod_nodes",lambda:SimpleNamespace(_load_mod=lambda name:mod(name,1)))
    for rows in ([{"slot":1,"name":"x"},{"slot":1,"name":"y"}], [{"slot":1,"name":"x","strength":float("nan")}], [{"slot":9,"name":"x"}]):
        with pytest.raises(ValueError): refs.load_selections(rows)
