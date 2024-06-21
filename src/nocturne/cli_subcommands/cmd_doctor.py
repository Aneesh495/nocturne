"""System environment and hardware diagnostic probe."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch


@dataclass(frozen=True)
class DoctorReport:
    """Consolidated hardware and dependency diagnostics."""

    python_version: str
    os_system: str
    os_release: str
    architecture: str
    torch_version: str
    numpy_version: str
    device_available: str
    cuda_available: bool
    mps_available: bool
    healthy: bool


def probe_environment() -> DoctorReport:
    """Inspect local hardware acceleration, Python interpreter, and library versions."""
    py_ver = sys.version.split()[0]
    os_sys = platform.system()
    os_rel = platform.release()
    arch = platform.machine()
    torch_ver = torch.__version__
    np_ver = np.__version__

    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_avail:
        device_str = f"cuda ({torch.cuda.get_device_name(0)})"
    elif mps_avail:
        device_str = "mps (Apple Silicon Metal Performance Shaders)"
    else:
        device_str = "cpu"

    healthy = True
    return DoctorReport(
        python_version=py_ver,
        os_system=os_sys,
        os_release=os_rel,
        architecture=arch,
        torch_version=torch_ver,
        numpy_version=np_ver,
        device_available=device_str,
        cuda_available=cuda_avail,
        mps_available=mps_avail,
        healthy=healthy,
    )


def run_doctor_command(verbose: bool = True) -> int:
    """Execute doctor CLI command, printing diagnostics to stdout."""
    report = probe_environment()
    if verbose:
        print("=== Nocturne System Diagnostics ===")
        print(f"Python:       {report.python_version}")
        print(f"Platform:     {report.os_system} {report.os_release} ({report.architecture})")
        print(f"PyTorch:      {report.torch_version}")
        print(f"NumPy:        {report.numpy_version}")
        print(f"Acceleration: {report.device_available}")
        print(f"Status:       {'HEALTHY' if report.healthy else 'DEGRADED'}")
    return 0 if report.healthy else 1
