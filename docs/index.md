# HLQuantum

Welcome to the official documentation for **HLQuantum** (High Level Quantum).

HLQuantum is a high-level Python package designed to simplify working with quantum hardware. Write your quantum logic once and run it on any supported backend — CUDA-Q, Qiskit, Cirq, Braket, PennyLane, or IonQ.

## Quick Start

```python
import hlquantum as hlq
from hlquantum import kernel

# Option 1: Build a circuit directly
qc = hlq.Circuit(2)
qc.h(0).cx(0, 1).measure_all()

result = hlq.run(qc, shots=1000)
print(result.counts)   # {'00': ~500, '11': ~500}

# Option 2: Use the @kernel decorator
@kernel(num_qubits=2)
def bell(qc):
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

result = hlq.run(bell, shots=1000)
```

## Features

- **Backend-Agnostic Circuits** — A single `Circuit` IR that translates to any supported framework.
- **Quantum Pipelines** — Build modular architectures using ML-inspired `Layer` and `Sequential` models.
- **Resilient Workflows** — Orchestrate complex executions with loops, branching, parallelism, and state persistence (save/resume).
- **Asynchronous Execution** — Multi-backend concurrency with `async/await` support and throttling.
- **Unitary-Agnostic `@kernel`** — Write quantum logic as plain Python functions.
- **GPU Acceleration** — Unified `GPUConfig` across all backends.
- **Built-in Algorithms** — QFT, Grover, Bernstein-Vazirani, Deutsch-Jozsa, VQE, QAOA, GQE, arithmetic circuits, and parameter-shift gradients.
- **Transpilation** — Built-in optimisation passes (redundant-gate removal, rotation merging).
- **Error Mitigation** — Pluggable post-processing for noisy results.

## Supported Backends

| Backend            | Framework                                              | Install extra                      |
| ------------------ | ------------------------------------------------------ | ---------------------------------- |
| `CudaQBackend`     | [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum) | `pip install hlquantum[cudaq]`     |
| `QiskitBackend`    | [IBM Qiskit](https://qiskit.org)                       | `pip install hlquantum[qiskit]`    |
| `CirqBackend`      | [Google Cirq](https://quantumai.google/cirq)           | `pip install hlquantum[cirq]`      |
| `BraketBackend`    | [Amazon Braket](https://aws.amazon.com/braket/)        | `pip install hlquantum[braket]`    |
| `PennyLaneBackend` | [Xanadu PennyLane](https://pennylane.ai)               | `pip install hlquantum[pennylane]` |
| `IonQBackend`      | [IonQ](https://ionq.com) (via qiskit-ionq)             | `pip install hlquantum[ionq]`      |

## What's Next?

- [**Core API**](api/core.md) — Circuits, kernels, parameters, and results.
- [**Backends**](api/backends.md) — Per-backend configuration and examples.
- [**Algorithms**](api/algorithms.md) — Built-in quantum algorithms and friendly aliases.
- [**Layers & Pipelines**](api/layers.md) — ML-style circuit composition.
- [**Workflows**](api/workflows.md) — Async orchestration, branching, and checkpoints.
- [**Transpiler**](api/transpiler.md) — Optimisation passes.
- [**Mitigation**](api/mitigation.md) — Error mitigation techniques.
- [**GPU Acceleration**](gpu.md) — Multi-GPU and precision configuration.
- [**Examples**](examples.md) — End-to-end demos and tutorials.
