"""GPU configuration and detection utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GPUPrecision(Enum):
    FP32 = "fp32"
    FP64 = "fp64"


@dataclass
class GPUConfig:
    """Configuration for GPU-accelerated simulation."""

    enabled: bool = False
    device_ids: Optional[List[int]] = None
    multi_gpu: bool = False
    precision: GPUPrecision = GPUPrecision.FP32
    memory_limit_gb: Optional[float] = None
    custatevec: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_gpus(self) -> int:
        if not self.enabled:
            return 0
        return len(self.device_ids) if self.device_ids is not None else 1

    @property
    def cuda_visible_devices(self) -> Optional[str]:
        return ",".join(str(d) for d in self.device_ids) if self.device_ids else None

    def apply_env(self) -> None:
        cvd = self.cuda_visible_devices
        if cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
            logger.info("CUDA_VISIBLE_DEVICES set to %s", cvd)

    def __repr__(self) -> str:
        if not self.enabled:
            return "GPUConfig(enabled=False)"
        parts = ["enabled=True"]
        if self.device_ids:
            parts.append(f"device_ids={self.device_ids}")
        if self.multi_gpu:
            parts.append("multi_gpu=True")
        parts.append(f"precision={self.precision.value}")
        if self.custatevec:
            parts.append("custatevec=True")
        return f"GPUConfig({', '.join(parts)})"


def detect_gpus() -> List[Dict[str, Any]]:
    """Detect available NVIDIA GPUs."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "id": i,
                "name": name,
                "memory_total_gb": round(mem.total / (1024**3), 2),
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []

