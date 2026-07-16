# DaSiWa System Monitor

A compact, non-intrusive system telemetry bar integrated directly into the ComfyUI top toolbar.

## Overview

The System Monitor displays real-time resource utilization in the ComfyUI header area. A DaSiWa settings button sits directly beside the monitor and is the home for settings shared by DaSiWa nodes as they are added.

The current settings are stored in the browser, so they remain active after a ComfyUI page reload:

- **Show system monitor:** hides or shows the monitor while keeping the settings button available.
- **Lite:** the default compact fixed-width, color-coded toolbar meters. Each meter shows a label, a numeric value, and a proportional background fill representing 0–100% usage.
- **Full:** a spacious monitor panel with every available metric, its current value and detail, plus a live graph covering the most recent 60 telemetry samples (normally about one minute).

## Metrics

| Metric | Description | Color |
|--------|-------------|-------|
| CPU | Overall CPU utilization across all threads | Blue (`#38bdf8`) |
| RAM | Physical memory usage | Purple (`#a78bfa`) |
| SWAP | Swap space (Linux) or Pagefile (Windows) | Amber (`#f59e0b`) |
| DISK | Root disk usage | Pink (`#fb7185`) |
| GPU0 Util | GPU 0 compute utilization | Green (`#4ade80`) |
| GPU0 VRAM | GPU 0 video memory usage | Cyan (`#22d3ee`) |
| GPU0 Temp | GPU 0 temperature in °C | Orange (`#fb923c`) |

Additional GPUs appear as GPU1, GPU2, etc., each with Util, VRAM, and Temp chips.

## Tooltips

Hover over any metric chip to see detailed information:

- **CPU:** Thread count
- **RAM/SWAP:** Used / Total in human-readable units (MiB/GiB)
- **DISK:** Mount path, used / total
- **GPU:** Device ID, name, and exact VRAM used / total

## GPU Support

| Platform | Vendor | Detection Method |
|----------|--------|------------------|
| Linux | NVIDIA | `nvidia-smi` query |
| Linux | AMD | `rocm-smi` JSON output |
| Linux | Intel | DRM/sysfs device tree |
| Windows | NVIDIA | `nvidia-smi` if available, otherwise CIM `Win32_VideoController` |
| Windows | AMD | CIM `Win32_VideoController` fallback |
| Windows | Intel | CIM `Win32_VideoController` fallback |

When multiple GPUs of the same vendor exist, each receives a sequential index starting at 0. If a specific GPU tool is unavailable, the system gracefully degrades to generic device enumeration.

## Responsive Behavior

When toolbar width is insufficient to display all metrics, lower-priority chips are hidden first. The priority order (highest to lowest):

1. CPU
2. RAM
3. GPU metrics (Util, VRAM, Temp per GPU)
4. SWAP
5. DISK

A ResizeObserver monitors window changes and adjusts visibility dynamically without user interaction. Full mode uses a scrollable panel and collapses to one metric column on narrow screens.

## Backend Requirements

- **psutil** — Cross-platform system metrics (CPU, RAM, swap, disk). Included in project dependencies.
- **nvidia-smi** — Optional, bundled with NVIDIA drivers.
- **rocm-smi** — Optional, part of ROCm toolkit for AMD GPUs.
- No additional GPU tools required on Windows beyond standard drivers.

## API Endpoints

The backend exposes two REST endpoints for external consumption:

- `/dasiwa/system-monitor` — Full system snapshot (JSON)
- `/dasiwa/system-monitor/gpus` — GPU-specific data only (JSON)

Additionally, updates are broadcast via WebSocket event `dasiwa.system_monitor` approximately once per second.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Monitor shows "Loading..." | Backend route not registered | Ensure `nodes/system_monitor.py` is imported in `__init__.py` |
| No GPU metrics shown | Missing GPU query tool | Verify `nvidia-smi --query-gpu=index,name --format=csv` runs successfully |
| Swap shows "n/a" | No swap configured | Normal behavior; indicates swap/pagefile is disabled |
| Panel overlaps other toolbar items | Insufficient toolbar width | Lower-priority metrics auto-hide; check browser developer console for errors |

## Disabling

Use the settings button next to the monitor and disable **Show system monitor**. The setting is browser-local and does not stop the lightweight backend telemetry endpoint or WebSocket event; this keeps re-enabling instant without requiring a ComfyUI restart.
