# GPU Acceleration

HLQuantum provides a unified `GPUConfig` that works across all GPU-capable backends.
The same configuration object drives target/device selection on every framework.

## Configuration

```python
from hlquantum import GPUConfig, GPUPrecision

# Single GPU (default device)
gpu = GPUConfig(enabled=True)

# Multi-GPU
gpu = GPUConfig(enabled=True, multi_gpu=True, device_ids=[0, 1])

# FP64 precision
gpu = GPUConfig(enabled=True, precision=GPUPrecision.FP64)

# Enable cuStateVec (Qiskit Aer)
gpu = GPUConfig(enabled=True, custatevec=True)
```

## GPU Support by Backend

| Backend            | GPU Library              | Auto-selected target / device                |
| ------------------ | ------------------------ | -------------------------------------------- |
| `CudaQBackend`     | CUDA-Q (native)          | `"nvidia"`, `"nvidia-fp64"`, `"nvidia-mqpu"` |
| `QiskitBackend`    | qiskit-aer-gpu           | `AerSimulator(device='GPU')`                 |
| `CirqBackend`      | qsimcirq                 | `QSimSimulator(use_gpu=True)`                |
| `PennyLaneBackend` | pennylane-lightning[gpu] | `"lightning.gpu"`                            |
| `BraketBackend`    | _(not available)_        | _(cloud-managed hardware)_                   |
| `IonQBackend`      | _(not available)_        | _(cloud-managed trapped-ion hardware)_       |

## Per-Backend Examples

```python
from hlquantum import GPUConfig, GPUPrecision
from hlquantum.backends import CudaQBackend, QiskitBackend, CirqBackend, PennyLaneBackend

gpu = GPUConfig(enabled=True)

# CUDA-Q — auto-selects "nvidia" target
cudaq = CudaQBackend(gpu_config=gpu)

# CUDA-Q — multi-GPU with FP64
cudaq_multi = CudaQBackend(
    gpu_config=GPUConfig(enabled=True, multi_gpu=True, precision=GPUPrecision.FP64)
)

# Qiskit Aer — GPU + cuStateVec
qiskit = QiskitBackend(gpu_config=GPUConfig(enabled=True, custatevec=True))

# Cirq — qsim GPU simulator
cirq = CirqBackend(gpu_config=gpu)

# PennyLane — auto-selects lightning.gpu
pl = PennyLaneBackend(gpu_config=gpu)
```

## Detecting GPUs

```python
from hlquantum import detect_gpus

for gpu in detect_gpus():
    print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
```

---

## Reference

::: hlquantum.gpu
