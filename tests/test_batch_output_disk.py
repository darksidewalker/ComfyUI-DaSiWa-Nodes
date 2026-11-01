import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
import helper_batch_output as hbo


def test_disk_budget_is_a_share_of_free_space_not_a_fixed_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(hbo.shutil, "disk_usage", lambda _p: SimpleNamespace(free=100 * 1024 ** 3))
    assert hbo.disk_backed_output_budget_bytes(str(tmp_path)) == 90 * 1024 ** 3
    assert hbo._has_free_disk_space(str(tmp_path), 90 * 1024 ** 3)
    assert not hbo._has_free_disk_space(str(tmp_path), 90 * 1024 ** 3 + 1)
    # 76 GiB on a 100 GiB-free drive used to fail the old 64 GiB cap; now it only depends on the drive.
    monkeypatch.setattr(hbo, "can_allocate_in_ram", lambda _b: False)
    with pytest.raises(RuntimeError, match="at most 90% of the free space"):
        hbo.allocate_cpu_output((1, 1), torch.float32, str(tmp_path), has_free_disk_space=lambda _d, _b: False)
    assert not hasattr(hbo, "MAX_DISK_BACKED_OUTPUT_BYTES")
