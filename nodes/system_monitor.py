import json
import os
import re
import shutil
import subprocess
import threading
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

    def gpu_info(self):
        gpus = _nvidia_gpus() + _amd_gpus()
        if os.name == "nt":
            vendor_telemetry = {gpu["vendor"] for gpu in gpus}
            gpus.extend(gpu for gpu in _windows_gpus() if gpu["vendor"] not in vendor_telemetry)
        else:
            gpus.extend(_intel_gpus())
        return gpus

    def snapshot(self):
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram": {"used": memory.used, "total": memory.total, "percent": memory.percent},
            "swap": {"used": swap.used, "total": swap.total, "percent": swap.percent},
            "disk": {"path": os.path.abspath(os.sep), "used": disk.used, "total": disk.total, "percent": disk.percent},
            "gpus": self.gpu_info(),
        }

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DaSiWaSystemMonitor", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if PromptServer is not None:
                PromptServer.instance.send_sync("dasiwa.system_monitor", self.snapshot())
            self._stop.wait(self.interval)


monitor = DaSiWaSystemMonitor()
if PromptServer is not None:
    monitor.start()

    @PromptServer.instance.routes.get("/dasiwa/system-monitor")
    async def system_monitor_snapshot(request):
        return web.json_response(monitor.snapshot())

    @PromptServer.instance.routes.get("/dasiwa/system-monitor/gpus")
    async def system_monitor_gpus(request):
        return web.json_response(monitor.gpu_info())
