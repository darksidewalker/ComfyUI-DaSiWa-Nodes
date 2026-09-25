"""Serialized C23 graph invariants, including subgraph boundaries and active AV path."""
import json
from collections import defaultdict, deque
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/DaSiWa MiniMaxH3 MythicAlchemy C-MMH3-23 Continuity.json"
FORBIDDEN = {"DFH3ContinuityControl", "DFH3ContinuityGuide", "DFH3ContinuityCommit", "DFH3ContinuityPublish"}


def graph():
    root = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    return root, root["definitions"]["subgraphs"][0]


def rows(graph):
    for link in graph["links"]:
        yield [link[k] for k in ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")] if isinstance(link, dict) else link


def node(graph, ident):
    return next(n for n in graph["nodes"] if n["id"] == ident)


def test_only_director_controls_continuity_and_two_new_technical_types():
    root, sub = graph()
    all_nodes = root["nodes"] + sub["nodes"]
    assert not FORBIDDEN.intersection(n["type"] for n in all_nodes)
    assert [n["type"] for n in all_nodes if "Continuity" in n["type"]] == [
        "DaSiWaH3ContinuityPublish", "DaSiWaH3ContinuityAppend"
    ]
    director = node(root, 2730)
    assert len(director["outputs"]) == 9
    assert [o["name"] for o in director["outputs"]] == [
        "guide", "duration", "positive_prompt", "width", "height", "model",
        "fl2va_requested", "inpaint_requested", "frame_rate",
    ]
    assert not any("continuity" in i["name"].lower() for i in director["inputs"])
    state = json.loads(director["widgets_values"][6])
    continuity = state["continuity"]
    assert continuity["version"] == 1
    assert continuity["operation"] == "new"
    assert continuity["capture"] is True
    assert continuity["session"]
    assert continuity["overlap_frames"] == 22
    assert continuity["extension_frames"] == 119
    assert "continuity" not in state.get("builder_state", {})
    assert director["widgets_values"][8] == 24
    exporter = node(root, 2568)
    assert exporter["widgets_values"][0] == 24
    assert exporter["widgets_values"][9] is True  # save_output
    assert exporter["widgets_values"][6] is False  # pingpong
    assert exporter["widgets_values"][11] is False  # crop_to_audio


def test_every_endpoint_and_backlink_is_reciprocal():
    for graph_part in graph():
        nodes = {n["id"]: n for n in graph_part["nodes"]}
        links = {r[0]: r for r in rows(graph_part)}
        assert len(nodes) == len(graph_part["nodes"])
        assert len(links) == len(graph_part["links"])
        for ident, origin, oslot, target, tslot, kind in links.values():
            if origin >= 0:
                socket = nodes[origin]["outputs"][oslot]
                assert ident in (socket.get("links") or [])
            else:
                assert origin == -10
                socket = graph_part["inputs"][oslot]
                assert ident in socket["linkIds"]
            assert kind == "*" or socket["type"] == "*" or set(kind.split(",")) <= set(socket["type"].split(","))
            if target >= 0:
                socket = nodes[target]["inputs"][tslot]
                assert socket["link"] == ident
            else:
                assert target == -20
                socket = graph_part["outputs"][tslot]
                assert ident in socket["linkIds"]
            assert kind == "*" or socket["type"] == "*" or set(kind.split(",")) <= set(socket["type"].split(","))
        for n in nodes.values():
            for index, socket in enumerate(n.get("inputs", [])):
                if socket.get("link") is not None:
                    assert links[socket["link"]][3:5] == [n["id"], index]
            for index, socket in enumerate(n.get("outputs", [])):
                for ident in socket.get("links") or []:
                    assert links[ident][1:3] == [n["id"], index]
        for boundary, endpoint, ix in (("inputs", 1, 2), ("outputs", 3, 4)):
            for slot, port in enumerate(graph_part.get(boundary, [])):
                for ident in port["linkIds"]:
                    assert links[ident][endpoint] == (-10 if boundary == "inputs" else -20)
                    assert links[ident][ix] == slot


def test_subgraph_ports_match_root_instance_and_preserve_original_exposures():
    root, sub = graph()
    instance = node(root, 1512)
    assert instance["type"] == sub["id"]
    assert [p["name"] for p in instance["outputs"]] == [p["name"] for p in sub["outputs"]]
    assert [p["type"] for p in instance["outputs"]] == [p["type"] for p in sub["outputs"]]
    for slot, port in enumerate(instance["inputs"]):
        assert any(i["name"] == port["name"] and i["type"] == port["type"] for i in sub["inputs"])
        if port.get("link") is not None:
            matching = [i for i in sub["inputs"] if i["name"] == port["name"]]
            assert len(matching) == 1
            assert matching[0]["linkIds"], (slot, port["name"])
    assert len(instance["inputs"]) == 24
    assert len(sub["inputs"]) == 31
    assert not any(i["name"] == "continuity" for i in sub["inputs"] + instance["inputs"])
    assert len(instance["outputs"]) == 8
    assert instance["outputs"][7]["name"] == "continuity_ticket"
    assert [o["name"] for o in instance["outputs"][:7]] == [
        "CLIP", "IMAGE", "FLOAT", "AUDIO", "MODEL", "MODEL_1", "output",
    ]


def test_sampler_to_export_and_publish_are_on_one_active_path():
    root, sub = graph()
    inner = {tuple(r[1:5]) for r in rows(sub)}
    outer = {tuple(r[1:5]) for r in rows(root)}
    guide = node(sub, 2701)
    assert guide["type"] == "MiniMaxH3DirectorGuide"
    assert [o["name"] for o in guide["outputs"]] == ["positive", "latent", "continuity_context"]
    assert [i["name"] for i in guide["inputs"]] == ["clip", "vae", "guide", "audio_vae"]
    assert (2701, 0, 2592, 1) in inner
    assert (2701, 1, 2591, 4) in inner
    assert (2701, 2, 2795, 1) in inner
    assert (2591, 0, 2795, 0) in inner
    assert (2795, 0, 2769, 2) in inner
    assert (2795, 1, -20, 7) in inner
    assert (2769, 0, 2671, 0) in inner  # cumulative video decode
    assert (2769, 0, 2782, 0) in inner  # cumulative audio decode
    assert (2782, 0, -20, 3) in inner
    assert (1512, 1, 2568, 0) in outer  # video to exporter
    assert (1512, 3, 2568, 1) in outer  # audio to exporter
    assert (1512, 7, 2796, 1) in outer
    assert (2568, 1, 2796, 0) in outer
    assert node(sub, 2795)["type"] == "DaSiWaH3ContinuityAppend"
    assert node(root, 2796)["type"] == "DaSiWaH3ContinuityPublish"
    assert node(root, 2796)["mode"] == node(root, 2568)["mode"] == 0


def test_expanded_graph_is_acyclic():
    root, sub = graph()
    outer = {r[0]: r for r in rows(root)}
    inner = {r[0]: r for r in rows(sub)}
    instance = node(root, 1512)

    def root_source(ident, slot):
        if ident != 1512:
            return ("root", ident)
        return inner_source(*inner[sub["outputs"][slot]["linkIds"][0]][1:3])

    def inner_source(ident, slot):
        if ident != -10:
            return ("sub", ident)
        name = sub["inputs"][slot]["name"]
        port = next((i for i in instance["inputs"] if i["name"] == name), None)
        if port is None or port.get("link") is None:
            return None
        return root_source(*outer[port["link"]][1:3])

    edges = []
    for _, origin, oslot, target, _, _ in outer.values():
        if target != 1512:
            source = root_source(origin, oslot)
            if source:
                edges.append((source, ("root", target)))
    for _, origin, oslot, target, _, _ in inner.values():
        if target != -20:
            source = inner_source(origin, oslot)
            if source:
                edges.append((source, ("sub", target)))
    degrees = defaultdict(int)
    next_nodes = defaultdict(set)
    for source, target in edges:
        degrees[source] += 0
        if target not in next_nodes[source]:
            next_nodes[source].add(target)
            degrees[target] += 1
    ready = deque(n for n in degrees if degrees[n] == 0)
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for target in next_nodes[current]:
            degrees[target] -= 1
            if degrees[target] == 0:
                ready.append(target)
    assert visited == len(degrees), [n for n, degree in degrees.items() if degree]
