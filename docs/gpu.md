# GPU Acceleration

HLQuantum provides a unified `GPUConfig` that works across all GPU-capable backends.

## Usage

```python
from hlquantum import GPUConfig, GPUPrecision

# Simple — single GPU
gpu = GPUConfig(enabled=True)

# Run with GPU
result = hlquantum.run(circuit, backend=backend, gpu_config=gpu)
```

## Reference

::: hlquantum.gpu
::: hlquantum.gpu.GPUConfig
::: hlquantum.gpu.GPUPrecision
::: hlquantum.gpu.detect_gpus
