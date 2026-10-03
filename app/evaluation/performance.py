"""Helpers for performance measurements: latency summaries, throughput and a description of
the machine and software they were taken on.

Percentiles are nearest-rank (``app.observability.metrics.percentile``), the same method
as the live metrics, so offline benchmarks and ``GET /api/metrics`` compare directly. With
few samples the high percentiles are the slowest samples: every summary carries its count.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.observability.metrics import percentile

T = TypeVar("T")

PACKAGES = (
    "torch",
    "sentence-transformers",
    "transformers",
    "numpy",
    "sqlalchemy",
    "psycopg",
    "fastapi",
    "uvicorn",
    "httpx",
)


def latency_stats(values_ms: Sequence[float]) -> dict[str, float | int | None]:
    """count, mean, min, p50, p95, p99 and max of a list of milliseconds."""
    values = [float(v) for v in values_ms]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def r(value: float | None) -> float | None:
        return None if value is None else round(value, 2)

    return {
        "count": len(values),
        "mean": r(sum(values) / len(values)),
        "min": r(min(values)),
        "p50": r(percentile(values, 50)),
        "p95": r(percentile(values, 95)),
        "p99": r(percentile(values, 99)),
        "max": r(max(values)),
    }


def rate(count: float, seconds: float) -> float | None:
    """Items per second, or None when nothing was timed."""
    return round(count / seconds, 2) if seconds > 0 else None


def timed(call: Callable[[], T]) -> tuple[T, float]:
    """``call()`` and its wall time in milliseconds."""
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1000


def keep_awake() -> bool:
    """Ask the OS not to sleep while this process measures (Windows; elsewhere a no-op).

    A laptop that suspends mid-measurement turns one sample into minutes or hours. The
    request ends with the process; nothing is changed in the power settings."""
    if sys.platform != "win32":
        return False
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    return bool(kernel32.SetThreadExecutionState(es_continuous | es_system_required))


def cpu_name() -> str:
    """The processor's marketing name where the OS exposes it."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def total_memory_gb() -> float | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / 1024**3, 1)
        return None
    try:
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def power_state() -> dict[str, Any]:
    """Mains or battery, where the OS says. A laptop on battery (or in a power-saving mode)
    can run several times slower, so every measurement records it."""
    if sys.platform == "win32":

        class SystemPowerStatus(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = SystemPowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
            ac = {0: False, 1: True}.get(status.ACLineStatus)
            battery = status.BatteryLifePercent if status.BatteryLifePercent <= 100 else None
            return {
                "ac_power": ac,
                "battery_percent": battery,
                "battery_saver": bool(status.SystemStatusFlag & 1),
            }
        return {}
    supplies = Path("/sys/class/power_supply")
    for online in supplies.glob("*/online") if supplies.exists() else []:
        try:
            return {"ac_power": online.read_text().strip() == "1"}
        except OSError:
            continue
    return {}


def environment(engine: Engine | None = None) -> dict[str, Any]:
    """The hardware and software a measurement ran on."""
    info: dict[str, Any] = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "memory_gb": total_memory_gb(),
        "python": platform.python_version(),
        "packages": {p: v for p in PACKAGES if (v := _version(p))},
        "power": power_state(),
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
        info["cuda"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass
    if engine is not None:
        database: dict[str, Any] = {"dialect": engine.dialect.name}
        with engine.connect() as connection:
            if engine.dialect.name == "postgresql":
                database["server"] = connection.execute(text("SHOW server_version")).scalar()
                database["pgvector"] = connection.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar()
            elif engine.dialect.name == "sqlite":
                database["server"] = connection.execute(text("SELECT sqlite_version()")).scalar()
        info["database"] = database
    return info


__all__ = ["environment", "keep_awake", "latency_stats", "power_state", "rate", "timed"]
