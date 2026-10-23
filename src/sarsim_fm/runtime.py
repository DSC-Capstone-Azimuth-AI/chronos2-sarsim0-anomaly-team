"""Device, precision, and environment detection so the same config runs on CPU or GPU."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    bf16: bool
    tf32: bool

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def resolve_device(spec: str = "auto") -> torch.device:
    """`auto` picks CUDA when available, else CPU. `mps` must be requested explicitly."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    return torch.device(spec)


def resolve_runtime(device: str = "auto", precision: str = "auto") -> Runtime:
    """`precision`: auto (bf16 on supporting CUDA GPUs, else fp32) | bf16 | fp32."""
    dev = resolve_device(device)
    cuda_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    if precision == "auto":
        bf16 = cuda_bf16
    elif precision == "bf16":
        if dev.type == "cuda" and not cuda_bf16:
            raise RuntimeError(f"{torch.cuda.get_device_name(dev)} does not support bf16")
        bf16 = True
    elif precision == "fp32":
        bf16 = False
    else:
        raise ValueError(f"Unknown precision {precision!r}; use auto, bf16, or fp32")

    # TF32 matmuls exist on Ampere (sm_80) and newer.
    tf32 = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8
    return Runtime(device=dev, bf16=bf16, tf32=tf32)


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sarsim0_commit() -> str | None:
    try:
        direct_url = metadata.distribution("sarsim0").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def environment_info(runtime: Runtime | None = None) -> dict:
    info = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "chronos_forecasting": metadata.version("chronos-forecasting"),
        "transformers": metadata.version("transformers"),
        "sarsim0_commit": _sarsim0_commit(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    if runtime is not None:
        info.update(
            device=str(runtime.device),
            device_name=(
                torch.cuda.get_device_name(runtime.device) if runtime.is_cuda else platform.processor()
            ),
            bf16=runtime.bf16,
            tf32=runtime.tf32,
        )
    return info
