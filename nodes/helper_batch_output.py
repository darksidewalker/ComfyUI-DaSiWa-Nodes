import gc
import math
import os
import shutil
import tempfile
import weakref
from typing import Callable, Optional, Tuple

import psutil
import torch


MAX_RAM_SAFETY_RESERVE_BYTES = 8 * 1024 * 1024 * 1024
MIN_RAM_SAFETY_RESERVE_BYTES = 1 * 1024 * 1024 * 1024
LOW_RAM_SAFETY_RESERVE_FRACTION = 0.25
# A disk-backed output may take at most this share of the temp drive's free space; the rest
# stays for ComfyUI's own temp files, the OS and the encoder's output. No fixed GiB cap.
MAX_DISK_BACKED_FREE_FRACTION = 0.9


def tensor_nbytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def available_ram_bytes() -> int:
    return int(psutil.virtual_memory().available)


def total_ram_bytes() -> int:
    return int(psutil.virtual_memory().total)


def ram_safety_reserve_bytes() -> int:
    proportional_reserve = int(total_ram_bytes() * LOW_RAM_SAFETY_RESERVE_FRACTION)
    return min(MAX_RAM_SAFETY_RESERVE_BYTES, max(MIN_RAM_SAFETY_RESERVE_BYTES, proportional_reserve))


def can_allocate_in_ram(required_bytes: int) -> bool:
    return required_bytes <= max(0, available_ram_bytes() - ram_safety_reserve_bytes())


def disk_backed_output_budget_bytes(directory: str) -> int:
    return int(shutil.disk_usage(directory).free * MAX_DISK_BACKED_FREE_FRACTION)


def _has_free_disk_space(directory: str, required_bytes: int) -> bool:
    return required_bytes <= disk_backed_output_budget_bytes(directory)


def _remove_temporary_output(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def unload_all_comfy_models() -> bool:
    """Ask ComfyUI to unload all managed models and empty the allocator.

    Returns True when the unload path was executed, False when
    model_management is unavailable (plain pytest, ComfyUI rebuild).
    """
    try:
        import model_management
    except Exception:
        return False
    try:
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        return True
    except Exception:
        # A failed unload must never abort the node run; the caller's
        # re-check will simply decide whether to fall back to disk.
        return False


def force_gc_and_cleanup(directory: Optional[str] = None) -> None:
    """Force GC immediately to release mmap files."""
    gc.collect()
    if directory is None:
        try:
            import folder_paths
            directory = folder_paths.get_temp_directory()
        except Exception:
            return
    if directory is None:
        return


def allocate_cpu_output(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    directory: str,
    has_free_disk_space: Optional[Callable[[str, int], bool]] = None,
    force_mmap: bool = False,
    before_mmap: Optional[Callable[[], None]] = None,
) -> Tuple[torch.Tensor, Optional[str]]:
    required_bytes = tensor_nbytes(shape, dtype)
    if not force_mmap and can_allocate_in_ram(required_bytes):
        return torch.zeros(shape, dtype=dtype), None
    os.makedirs(directory, exist_ok=True)
    has_free_disk_space = has_free_disk_space or _has_free_disk_space
    if not has_free_disk_space(directory, required_bytes):
        raise RuntimeError(
            f"Not enough free disk space for a {required_bytes / 1024 ** 3:.2f} GiB "
            f"disk-backed output: at most {MAX_DISK_BACKED_FREE_FRACTION:.0%} of the free "
            f"space in '{directory}' may be used. Free up space, use a larger/other temp "
            "drive, disable 'Use disk-backed (mmap) output', or reduce the batch "
            "size / target resolution."
        )
    if before_mmap is not None:
        before_mmap()

    descriptor, path = tempfile.mkstemp(prefix="dasiwa_output_", suffix=".mmap", dir=directory)
    os.close(descriptor)
    try:
        output = torch.from_file(path, shared=True, size=math.prod(shape), dtype=dtype).reshape(shape)
    except Exception:
        _remove_temporary_output(path)
        raise
    weakref.finalize(output, _remove_temporary_output, path)
    return output, path
