"""rocm-smi VRAM parsing in _amd_gpus (issue #95): --showmeminfo vram reports bytes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nodes"))

import nodes_system_monitor as mon

# Shape of `rocm-smi --showproductname --showuse --showmemuse --showmeminfo vram --showtemp --json`
ROCM_JSON = json.dumps({
    "card0": {
        "GPU": "0",
        "Product Name": "AMD Radeon RX 9070 XT",
        "GPU use (%)": "6.0",
        "GPU Memory Allocated (VRAM%)": "13.6",
        "VRAM Total Memory (B)": "17095983104",
        "VRAM Total Used Memory (B)": "2329735168",
        "Temperature (Sensor edge) (C)": "51.0",
    }
})


def test_amd_gpus_reads_vram_bytes_from_showmeminfo():
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return ROCM_JSON

    gpus = mon._amd_gpus(run=fake_run)
    assert "--showmeminfo" in calls[0] and "vram" in calls[0]
    (gpu,) = gpus
    assert gpu["memory_used"] == 2329735168.0
    assert gpu["memory_total"] == 17095983104.0
    assert gpu["memory_percent"] == 13.6
    assert gpu["utilization"] == 6.0
    assert gpu["temperature"] == 51.0
