"""Tests for hlquantum.gpu and GPU configuration across backends."""

import os

import pytest

from hlquantum.gpu import GPUConfig, GPUPrecision, detect_gpus


class TestGPUConfig:
    def test_defaults(self):
        cfg = GPUConfig()
        assert cfg.enabled is False
        assert cfg.multi_gpu is False
        assert cfg.precision == GPUPrecision.FP32
        assert cfg.custatevec is False
        assert cfg.num_gpus == 0
        assert cfg.device_ids is None

    def test_enabled_single_gpu(self):
        cfg = GPUConfig(enabled=True)
        assert cfg.num_gpus == 1

    def test_device_ids(self):
        cfg = GPUConfig(enabled=True, device_ids=[0, 1, 2])
        assert cfg.num_gpus == 3
        assert cfg.cuda_visible_devices == "0,1,2"

    def test_cuda_visible_devices_none(self):
        cfg = GPUConfig(enabled=True)
        assert cfg.cuda_visible_devices is None

    def test_precision_fp64(self):
        cfg = GPUConfig(enabled=True, precision=GPUPrecision.FP64)
        assert cfg.precision == GPUPrecision.FP64
        assert cfg.precision.value == "fp64"

    def test_apply_env(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        cfg = GPUConfig(enabled=True, device_ids=[2, 3])
        cfg.apply_env()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_repr_disabled(self):
        cfg = GPUConfig()
        assert "enabled=False" in repr(cfg)

    def test_repr_enabled(self):
        cfg = GPUConfig(enabled=True, multi_gpu=True, custatevec=True)
        r = repr(cfg)
        assert "enabled=True" in r
        assert "multi_gpu=True" in r
        assert "custatevec=True" in r

    def test_extra_kwargs(self):
        cfg = GPUConfig(enabled=True, extra={"max_fused_gate_size": 4})
        assert cfg.extra["max_fused_gate_size"] == 4


class TestDetectGPUs:
    def test_returns_list(self):
        result = detect_gpus()
        assert isinstance(result, list)


class TestCudaQGPUTargetResolution:
    def test_cpu_default(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        backend = CudaQBackend()
        assert backend._target == "default"
        assert backend.supports_gpu is True

    def test_gpu_enabled_auto_nvidia(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        backend = CudaQBackend(gpu_config=GPUConfig(enabled=True))
        assert backend._target == "nvidia"

    def test_gpu_fp64(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        backend = CudaQBackend(
            gpu_config=GPUConfig(enabled=True, precision=GPUPrecision.FP64)
        )
        assert backend._target == "nvidia-fp64"

    def test_gpu_multi(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        backend = CudaQBackend(
            gpu_config=GPUConfig(enabled=True, multi_gpu=True)
        )
        assert backend._target == "nvidia-mqpu"

    def test_explicit_target_overrides_config(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        backend = CudaQBackend(
            target="nvidia-mgpu",
            gpu_config=GPUConfig(enabled=True),
        )
        assert backend._target == "nvidia-mgpu"


class TestQiskitGPUConfig:
    def test_name_reflects_gpu(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        backend = QiskitBackend(gpu_config=GPUConfig(enabled=True))
        assert "GPU" in backend.name
        assert backend.supports_gpu is True

    def test_name_no_gpu(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        backend = QiskitBackend()
        assert "GPU" not in backend.name


class TestCirqGPUConfig:
    def test_name_reflects_gpu(self):
        from hlquantum.backends.cirq_backend import CirqBackend
        backend = CirqBackend(gpu_config=GPUConfig(enabled=True))
        assert "GPU" in backend.name
        assert backend.supports_gpu is True

    def test_name_no_gpu(self):
        from hlquantum.backends.cirq_backend import CirqBackend
        backend = CirqBackend()
        assert "Simulator" in backend.name


class TestPennyLaneGPUConfig:
    def test_auto_selects_lightning_gpu(self):
        from hlquantum.backends.pennylane_backend import PennyLaneBackend
        backend = PennyLaneBackend(gpu_config=GPUConfig(enabled=True))
        assert backend._device_name == "lightning.gpu"
        assert backend.supports_gpu is True

    def test_default_device_no_gpu(self):
        from hlquantum.backends.pennylane_backend import PennyLaneBackend
        backend = PennyLaneBackend()
        assert backend._device_name == "default.qubit"

    def test_explicit_device_overrides(self):
        from hlquantum.backends.pennylane_backend import PennyLaneBackend
        backend = PennyLaneBackend(
            device_name="lightning.qubit",
            gpu_config=GPUConfig(enabled=True),
        )
        assert backend._device_name == "lightning.qubit"


class TestBraketNoGPU:
    def test_no_gpu_support(self):
        from hlquantum.backends.braket_backend import BraketBackend
        backend = BraketBackend()
        assert backend.supports_gpu is False


class TestBackendRepr:
    def test_gpu_tag_in_repr(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        backend = CudaQBackend(gpu_config=GPUConfig(enabled=True))
        assert "[GPU]" in repr(backend)

    def test_no_gpu_tag(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        backend = CudaQBackend()
        assert "[GPU]" not in repr(backend)
