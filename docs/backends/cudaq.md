# CUDA-Q Backend Configuration

The `CudaQBackend` is highly optimized to run directly on the NVIDIA CUDA-Q platform, supporting multi-node, multi-GPU frameworks efficiently.

## Installation

Ensure you have the required extra installed:
```bash
pip install hlquantum[cudaq]
```
This installs the `cuda-quantum` Python Wheel locally.

## Target Selection

By default, NVIDIA CUDA-Q will use the fast `qpp` CPU simulator. When you provide an NVIDIA GPU, you can leverage native CUDA targets capable of distributing quantum state arrays across distributed memory architectures.

You configure your CUDA-Q setup by passing explicit `"nvidia"`, `"nvidia-fp64"`, or `"nvidia-mgpu"` text markers to the CUDA-Q target backend wrapper.

```python
from hlquantum.backends import CudaQBackend
import hlquantum as hlq

# Instantiate an NVIDIA multi-GPU node backend
backend = CudaQBackend(target="nvidia-mgpu")
result = hlq.run(circuit, backend=backend, shots=1000)
```

No external API keys are required for NVIDIA simulators. When working with NVIDIA systems via cloud virtual environments or Docker nodes containing multiple devices, ensure variables like `CUDA_VISIBLE_DEVICES` remain exported natively into the terminal context as standard. Or, utilize HLQuantum `GPUConfig(enabled=True)` across your instance.
