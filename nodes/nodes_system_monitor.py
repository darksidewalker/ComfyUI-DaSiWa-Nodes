import json
import os
import re
import shutil
import subprocess
import threading
import time
import warnings
from pathlib import Path

import psutil

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    PromptServer = None
    web = None


UNKNOWN = None
NVIDIA_QUERY = "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu"


def _probe(callable_, default):
    """Run a hardware/psutil probe, swallowing warnings and errors.

    Containers and some sandboxes expose incomplete /proc, which makes psutil
    raise or emit RuntimeWarning (e.g. missing /proc/vmstat for swap). A probe
    must never crash the monitor or spam the log, so we silence warnings and
    fall back to a default.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return callable_()
    except Exception:
        return default


def _memory_fields(sample):
    if sample is None:
        return {"used": UNKNOWN, "total": UNKNOWN, "percent": UNKNOWN}
    return {"used": sample.used, "total": sample.total, "percent": sample.percent}


def _monitor_enabled():
    """Continuous polling is on by default; any explicit falsy value disables it."""
    return os.environ.get("DASWA_SYSTEM_MONITOR", "").strip().lower() not in (
        "0", "false", "no", "off", "disable", "disabled",
    )


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return UNKNOWN


def _percent(used, total):
    if used is None or total in (None, 0):
        return UNKNOWN
    return round(used / total * 100, 1)


def _run(command):
    if not shutil.which(command[0]):
        return ""
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia_gpus(run=_run):
    output = run(["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"])
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        index, identifier, name, utilization, used, total, temperature = fields
        used_bytes = _number(used)
        total_bytes = _number(total)
        if used_bytes is not None:
            used_bytes *= 1024 * 1024
        if total_bytes is not None:
            total_bytes *= 1024 * 1024
        gpus.append({
            "id": f"NVIDIA:{index}", "index": int(index), "vendor": "NVIDIA", "name": name,
            "uuid": identifier, "utilization": _number(utilization), "memory_used": used_bytes,
            "memory_total": total_bytes, "memory_percent": _percent(used_bytes, total_bytes),
            "temperature": _number(temperature),
        })
    return gpus


class _NvmlSession:
    """One NVML session for the life of the process, reused for every sample.

    Spawning ``nvidia-smi`` every second opens a new NVML session each time. Under Docker
    Desktop / WSL2 on Windows each session opened in the guest leaks NVIDIA driver memory
    on the host that is only reclaimed by a reboot (~23 allocations per ``nvidia-smi
    --query-gpu`` call, ~14 per nvmlInit/nvmlShutdown pair, none for queries on an open
    session). Polling through one persistent session avoids that and the process spawns.
    """

    RETRY_SECONDS = 600  # a failed init is retried rarely: every attempt opens a session

    def __init__(self):
        self._lock = threading.Lock()
        self._nvml = None
        self._handles = []
        self._static = []
        self._failed_at = None

    def _ensure(self):
        with self._lock:
            if self._nvml is not None:
                return self._nvml
            if self._failed_at is not None and time.monotonic() - self._failed_at < self.RETRY_SECONDS:
                return None
            pynvml = None
            initialized = False
            try:
                import pynvml  # provided by nvidia-ml-py

                pynvml.nvmlInit()
                initialized = True
                handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
                self._static = [
                    (_nvml_text(pynvml.nvmlDeviceGetName(h)), _nvml_text(pynvml.nvmlDeviceGetUUID(h)))
                    for h in handles
                ]
                self._handles = handles
                self._nvml = pynvml
                return pynvml
            except Exception:  # no nvidia-ml-py, no NVIDIA driver, or NVML error
                if initialized:
                    # Close a session whose device enumeration failed, so a later retry does
                    # not leave a second session open.
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass
                self._handles, self._static = [], []
                self._failed_at = time.monotonic()
                return None

    def gpus(self):
        """Returns NVIDIA GPU samples, or None when NVML is unavailable (caller falls back)."""
        nvml = self._ensure()
        if nvml is None:
            return None
        gpus = []
        for index, (handle, (name, identifier)) in enumerate(zip(self._handles, self._static)):
            memory = _probe(lambda: nvml.nvmlDeviceGetMemoryInfo(handle), None)
            used_bytes = memory.used if memory is not None else UNKNOWN
            total_bytes = memory.total if memory is not None else UNKNOWN
            gpus.append({
                "id": f"NVIDIA:{index}", "index": index, "vendor": "NVIDIA", "name": name,
                "uuid": identifier,
                "utilization": _number(_probe(lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu, UNKNOWN)),
                "memory_used": used_bytes, "memory_total": total_bytes,
                "memory_percent": _percent(used_bytes, total_bytes),
                "temperature": _number(_probe(
                    lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU), UNKNOWN)),
            })
        return gpus


def _nvml_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


_NVML_SESSION = _NvmlSession()


def _nvidia_telemetry():
    gpus = _NVML_SESSION.gpus()
    return _nvidia_gpus() if gpus is None else gpus


def _rocm_value(data, field):
    if isinstance(data, dict):
        for key, value in data.items():
            if field.lower() in key.lower() and isinstance(value, (str, int, float)):
                match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
                if match:
                    return float(match.group())
            result = _rocm_value(value, field)
            if result is not None:
                return result
    if isinstance(data, list):
        for value in data:
            result = _rocm_value(value, field)
            if result is not None:
                return result
    return UNKNOWN


def _amd_gpus(run=_run):
    output = run(["rocm-smi", "--showproductname", "--showuse", "--showmemuse", "--showtemp", "--json"])
    try:
        devices = json.loads(output)
    except json.JSONDecodeError:
        return []
    gpus = []
    for index, (device_id, data) in enumerate(devices.items()):
        name = next((str(value) for key, value in data.items() if "product name" in key.lower()), "AMD GPU")
        used = _rocm_value(data, "used memory")
        total = _rocm_value(data, "total memory")
        if used is not None:
            used *= 1024 * 1024
        if total is not None:
            total *= 1024 * 1024
        gpus.append({
            "id": f"AMD:{device_id}", "index": index, "vendor": "AMD", "name": name,
            "uuid": device_id, "utilization": _rocm_value(data, "gpu use"),
            "memory_used": used, "memory_total": total, "memory_percent": _percent(used, total),
            "temperature": _rocm_value(data, "temperature"),
        })
    return gpus


def _intel_gpus(drm_root=Path("/sys/class/drm")):
    gpus = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        vendor_file = card / "device/vendor"
        try:
            if vendor_file.read_text().strip().lower() != "0x8086":
                continue
        except OSError:
            continue
        device_id = card.name.replace("card", "")
        name = "Intel GPU"
        uevent = card / "device/uevent"
        try:
            for line in uevent.read_text().splitlines():
                if line.startswith("DRIVER="):
                    name = f"Intel {line.split('=', 1)[1]}"
                    break
        except OSError:
            pass
        gpus.append({
            "id": f"Intel:{device_id}", "index": len(gpus), "vendor": "Intel", "name": name,
            "uuid": card.resolve().name, "utilization": UNKNOWN, "memory_used": UNKNOWN,
            "memory_total": UNKNOWN, "memory_percent": UNKNOWN, "temperature": UNKNOWN,
        })
    return gpus


def _windows_vendor(adapter):
    description = " ".join(str(adapter.get(key, "")) for key in ("Name", "VideoProcessor", "PNPDeviceID")).lower()
    if "nvidia" in description:
        return "NVIDIA"
    if "amd" in description or "radeon" in description:
        return "AMD"
    if "intel" in description:
        return "Intel"
    return "Other"


def _windows_gpus(run=_run):
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,VideoProcessor,AdapterRAM | ConvertTo-Json -Compress",
    ]
    try:
        adapters = json.loads(run(command))
    except json.JSONDecodeError:
        return []
    if isinstance(adapters, dict):
        adapters = [adapters]
    if not isinstance(adapters, list):
        return []
    gpus = []
    for index, adapter in enumerate(adapters):
        vendor = _windows_vendor(adapter)
        identifier = adapter.get("PNPDeviceID") or str(index)
        name = adapter.get("Name") or f"{vendor} GPU"
        total = _number(adapter.get("AdapterRAM"))
        gpus.append({
            "id": f"{vendor}:{identifier}", "index": index, "vendor": vendor, "name": name,
            "uuid": identifier, "utilization": UNKNOWN, "memory_used": UNKNOWN,
            "memory_total": total, "memory_percent": UNKNOWN, "temperature": UNKNOWN,
        })
    return gpus


class DaSiWaSystemMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self._thread = None
        self._stop = threading.Event()
        self._io_lock = threading.Lock()
        self._previous_disk_io = None
        self._previous_disk_io_time = None
        # The Windows CIM query (Win32_VideoController) is expensive (a fresh
        # powershell.exe per call) and its data (adapter name, PNPDeviceID,
        # AdapterRAM) is static. Cache it after the first successful probe so
        # we do not spawn a powershell process every second.
        self._windows_gpus_cache = None
        self._windows_gpus_lock = threading.Lock()

    def _windows_gpus_cached(self):
        if self._windows_gpus_cache:
            return self._windows_gpus_cache
        with self._windows_gpus_lock:
            if self._windows_gpus_cache:
                return self._windows_gpus_cache
            gpus = _probe(_windows_gpus, [])
            if gpus:
                self._windows_gpus_cache = gpus
            # An empty result is treated as "probe not available yet" and is
            # retried on the next call.
            return gpus

    def gpu_info(self):
        gpus = _probe(_nvidia_telemetry, []) + _probe(_amd_gpus, [])
        if os.name == "nt":
            vendor_telemetry = {gpu["vendor"] for gpu in gpus}
            gpus.extend(gpu for gpu in self._windows_gpus_cached() if gpu["vendor"] not in vendor_telemetry)
        else:
            gpus.extend(_probe(_intel_gpus, []))
        return gpus

    @staticmethod
    def _disk_device_name(device):
        name = os.path.basename(device)
        if name.startswith(("nvme", "mmcblk")):
            return re.sub(r"p\d+$", "", name)
        return re.sub(r"\d+$", "", name)

    def _disk_io_rates(self):
        try:
            counters = psutil.disk_io_counters(perdisk=True)
        except (AttributeError, OSError):
            return {}
        now = time.monotonic()
        with self._io_lock:
            previous = self._previous_disk_io
            previous_time = self._previous_disk_io_time
            self._previous_disk_io = counters
            self._previous_disk_io_time = now
        if previous is None or previous_time is None:
            return {name: (0.0, 0.0) for name in counters}
        elapsed = max(now - previous_time, 0.001)
        rates = {}
        for name, current in counters.items():
            prior = previous.get(name)
            if prior is None:
                rates[name] = (0.0, 0.0)
                continue
            rates[name] = (
                max(current.read_bytes - prior.read_bytes, 0) / elapsed / (1024 * 1024),
                max(current.write_bytes - prior.write_bytes, 0) / elapsed / (1024 * 1024),
            )
        return rates

    @staticmethod
    def _partition_for_path(partitions, path):
        target = os.path.normcase(os.path.realpath(path))
        matches = []
        for partition in partitions:
            # A stale drive letter (e.g. a phone/USB drive unplugged since boot)
            # can still show up here on Windows; realpath() on it raises OSError
            # (WinError 1005) instead of a plain "not found", so it needs the
            # same _probe() treatment as the other hardware probes in this file.
            mountpoint = _probe(lambda p=partition: os.path.normcase(os.path.realpath(p.mountpoint)), None)
            if mountpoint is None:
                continue
            try:
                if os.path.commonpath((target, mountpoint)) == mountpoint:
                    matches.append(partition)
            except ValueError:
                continue
        return max(matches, key=lambda partition: len(partition.mountpoint), default=None)

    @staticmethod
    def _comfyui_path():
        try:
            import folder_paths

            base_path = getattr(folder_paths, "base_path", None)
            if base_path:
                return base_path
        except ImportError:
            pass
        return str(Path(__file__).resolve())

    def _monitored_partitions(self):
        partitions = psutil.disk_partitions(all=False)
        selected = []
        for path in (os.path.abspath(os.sep), self._comfyui_path()):
            partition = self._partition_for_path(partitions, path)
            if partition and partition.device not in {item.device for item in selected}:
                selected.append(partition)
        return selected

    def snapshot(self):
        memory = _probe(psutil.virtual_memory, None)
        swap = _probe(psutil.swap_memory, None)
        disk_io_rates = self._disk_io_rates()
        disks = []
        for partition in self._monitored_partitions():
            usage = _probe(lambda p=partition: psutil.disk_usage(p.mountpoint), None)
            if usage is None:
                continue
            read_mb_s, write_mb_s = disk_io_rates.get(self._disk_device_name(partition.device), (0.0, 0.0))
            disks.append({
                "path": partition.mountpoint,
                "device": partition.device,
                "fstype": partition.fstype,
                "used": usage.used,
                "total": usage.total,
                "percent": usage.percent,
                "read_mb_s": round(read_mb_s, 1),
                "write_mb_s": round(write_mb_s, 1),
            })
        return {
            "cpu_percent": _probe(lambda: psutil.cpu_percent(interval=None), UNKNOWN),
            "cpu_count": _probe(lambda: psutil.cpu_count(logical=True), UNKNOWN),
            "ram": _memory_fields(memory),
            "swap": _memory_fields(swap),
            "disks": disks,
            "gpus": self.gpu_info(),
        }

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DaSiWaSystemMonitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.interval + 1)

    def _run(self):
        while not self._stop.is_set():
            if PromptServer is not None:
                PromptServer.instance.send_sync("dasiwa.system_monitor", self.snapshot())
            self._stop.wait(self.interval)


monitor = DaSiWaSystemMonitor()
if PromptServer is not None:
    if _monitor_enabled():
        monitor.start()

    @PromptServer.instance.routes.get("/dasiwa/system-monitor")
    async def system_monitor_snapshot(request):
        if not monitor.running:
            return web.json_response({"error": "System Monitor is disabled."}, status=503)
        return web.json_response(monitor.snapshot())

    @PromptServer.instance.routes.get("/dasiwa/system-monitor/gpus")
    async def system_monitor_gpus(request):
        if not monitor.running:
            return web.json_response({"error": "System Monitor is disabled."}, status=503)
        return web.json_response(monitor.gpu_info())

    @PromptServer.instance.routes.post("/dasiwa/system-monitor/enabled")
    async def system_monitor_enabled(request):
        try:
            enabled = (await request.json()).get("enabled")
        except (AttributeError, ValueError, json.JSONDecodeError):
            return web.json_response({"error": "Expected a JSON enabled boolean."}, status=400)
        if not isinstance(enabled, bool):
            return web.json_response({"error": "Expected a JSON enabled boolean."}, status=400)
        if enabled:
            monitor.start()
        else:
            monitor.stop()
        return web.json_response({"enabled": monitor.running})
