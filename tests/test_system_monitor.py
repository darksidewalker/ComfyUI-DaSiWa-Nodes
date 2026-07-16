import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault("psutil", types.ModuleType("psutil"))

MODULE_PATH = Path(__file__).parents[1] / "nodes" / "system_monitor.py"
spec = importlib.util.spec_from_file_location("system_monitor", MODULE_PATH)
assert spec is not None and spec.loader is not None
system_monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(system_monitor)


def test_nvidia_parser_keeps_each_gpu_id_and_metrics():
    output = "0, GPU-a, RTX A, 55, 1024, 8192, 62\n1, GPU-b, RTX B, 21, 2048, 16384, 51\n"

    gpus = system_monitor._nvidia_gpus(lambda _: output)

    assert [gpu["id"] for gpu in gpus] == ["NVIDIA:0", "NVIDIA:1"]
    assert gpus[1]["vendor"] == "NVIDIA"
    assert gpus[0]["memory_used"] == 1024 * 1024 * 1024
    assert gpus[1]["memory_percent"] == 12.5


def test_amd_parser_keeps_vendor_and_device_identifier():
    output = '{"card0": {"Card series": "AMD Radeon", "GPU use (%)": "42", "Used Memory (MiB)": "1024", "Total Memory (MiB)": "8192", "Temperature (Sensor edge) (C)": "61"}}'

    gpu = system_monitor._amd_gpus(lambda _: output)[0]

    assert gpu["id"] == "AMD:card0"
    assert gpu["vendor"] == "AMD"
    assert gpu["utilization"] == 42
    assert gpu["memory_percent"] == 12.5


def test_intel_discovery_uses_drm_card_id(tmp_path):
    card = tmp_path / "card3" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    (card / "uevent").write_text("DRIVER=i915\n")

    gpu = system_monitor._intel_gpus(tmp_path)[0]

    assert gpu["id"] == "Intel:3"
    assert gpu["name"] == "Intel i915"


def test_windows_parser_detects_intel_and_amd_adapters_without_vendor_tools():
    output = '[{"Name":"Intel(R) Arc","PNPDeviceID":"PCI\\\\VEN_8086&DEV_7D55","VideoProcessor":"Intel Arc","AdapterRAM":8589934592},{"Name":"AMD Radeon","PNPDeviceID":"PCI\\\\VEN_1002&DEV_744C","VideoProcessor":"AMD Radeon","AdapterRAM":17179869184}]'

    gpus = system_monitor._windows_gpus(lambda _: output)

    assert [gpu["vendor"] for gpu in gpus] == ["Intel", "AMD"]
    assert gpus[0]["id"].startswith("Intel:PCI\\VEN_8086")
    assert gpus[1]["memory_total"] == 17179869184
