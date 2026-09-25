import ast
from pathlib import Path


def test_root_exposes_exactly_two_continuity_nodes_and_routes():
    root = Path(__file__).resolve().parents[2] / "__init__.py"
    tree = ast.parse(root.read_text())
    text = root.read_text()
    assert '"DaSiWaH3ContinuityAppend": DaSiWaH3ContinuityAppend' in text
    assert '"DaSiWaH3ContinuityPublish": DaSiWaH3ContinuityPublish' in text
    assert 'continuity_routes.register_routes' in text
    assert '"DFH3ContinuityControl"' not in text
    assert '"DFH3ContinuityGuide"' not in text


def test_continuity_nodes_share_director_minimax_category():
    from dasiwa_continuity_test.nodes import (
        DaSiWaH3ContinuityAppend,
        DaSiWaH3ContinuityPublish,
    )
    expected = "DaSiWa/MiniMax H3"
    assert DaSiWaH3ContinuityAppend.CATEGORY == expected
    assert DaSiWaH3ContinuityPublish.CATEGORY == expected
