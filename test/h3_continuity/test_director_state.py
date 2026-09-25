import json
import pytest
from dasiwa_nodes_test.nodes_minimax_h3_director import MiniMaxH3Director


def build(state=None, **kwargs):
    return MiniMaxH3Director().build_guide("T2VA", "original", 48, 32, 5, "match", json.dumps(state or {}), "", **kwargs)


def test_absent_continuity_preserves_nine_outputs_and_new_prompt():
    result = build()
    assert len(result) == 9
    assert result[0]["resolved_prompt"] == result[2] == "original"
    assert result[0]["mode"] == "T2VA"
    assert "continuity" not in result[0]


def test_explicit_continuity_and_frame_rate_travel_in_guide():
    result = build({"continuity": {"session": "take1", "capture": True}}, frame_rate=24.0)
    assert result[0]["continuity"]["capture"] is True
    assert result[0]["continuity"]["session"] == "take1"
    assert result[0]["frame_rate"] == result[8]


@pytest.mark.parametrize("setting", [{"capture": "false"}, {"operation": "redo"}, {"session": "../bad"}, {"extension_frames": 118}])
def test_invalid_continuity_rejected_in_director(setting):
    with pytest.raises(ValueError):
        build({"continuity": setting})
