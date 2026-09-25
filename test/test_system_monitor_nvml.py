"""NVIDIA telemetry through one persistent NVML session (nodes_system_monitor._NvmlSession)."""
import sys
import types

import pytest

from nodes import nodes_system_monitor as monitor


class _Memory:
    used = 4 * 1024**3
    total = 16 * 1024**3


class _Util:
    gpu = 37


def _fake_pynvml(fail_enumeration=False):
    calls = {"init": 0, "shutdown": 0}
    fake = types.ModuleType("pynvml")
    fake.NVML_TEMPERATURE_GPU = 0

    class NVMLError(Exception):
        pass

    fake.NVMLError = NVMLError

    def init():
        calls["init"] += 1

    def count():
        if fail_enumeration:
            raise NVMLError("enumeration failed")
        return 1

    fake.nvmlInit = init
    fake.nvmlShutdown = lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1)
    fake.nvmlDeviceGetCount = count
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"handle-{i}"
    fake.nvmlDeviceGetName = lambda h: b"Test GPU"
    fake.nvmlDeviceGetUUID = lambda h: "GPU-test"
    fake.nvmlDeviceGetMemoryInfo = lambda h: _Memory()
    fake.nvmlDeviceGetUtilizationRates = lambda h: _Util()
    fake.nvmlDeviceGetTemperature = lambda h, sensor: 55
    return fake, calls


@pytest.fixture
def fake_nvml(monkeypatch):
    def install(**kwargs):
        fake, calls = _fake_pynvml(**kwargs)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        return calls
    return install


def test_session_is_opened_once_and_reused(fake_nvml):
    calls = fake_nvml()
    session = monitor._NvmlSession()
    first = session.gpus()
    for _ in range(5):
        session.gpus()
    assert calls == {"init": 1, "shutdown": 0}
    assert first == [{
        "id": "NVIDIA:0", "index": 0, "vendor": "NVIDIA", "name": "Test GPU", "uuid": "GPU-test",
        "utilization": 37.0, "memory_used": 4 * 1024**3, "memory_total": 16 * 1024**3,
        "memory_percent": 25.0, "temperature": 55.0,
    }]


def test_partial_init_is_shut_down_and_retry_is_throttled(fake_nvml):
    calls = fake_nvml(fail_enumeration=True)
    session = monitor._NvmlSession()
    assert session.gpus() is None
    assert session.gpus() is None  # inside the retry window: no second session
    assert calls == {"init": 1, "shutdown": 1}


def test_falls_back_to_nvidia_smi_without_nvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import pynvml -> ImportError
    monkeypatch.setattr(monitor, "_NVML_SESSION", monitor._NvmlSession())
    monkeypatch.setattr(monitor, "_nvidia_gpus", lambda: [{"id": "NVIDIA:0", "source": "nvidia-smi"}])
    assert monitor._nvidia_telemetry() == [{"id": "NVIDIA:0", "source": "nvidia-smi"}]
