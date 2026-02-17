# HLQuantum

**HLQuantum** (High Level Quantum) is a high-level Python package designed to simplify working with quantum hardware. Write your quantum logic once and run it on any supported backend.

## Supported Backends

| Backend | Framework | Install extra |
|---------|-----------|---------------|
| `CudaQBackend` | [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum) | `pip install hlquantum[cudaq]` |
| `QiskitBackend` | [IBM Qiskit](https://qiskit.org) | `pip install hlquantum[qiskit]` |
| `CirqBackend` | [Google Cirq](https://quantumai.google/cirq) | `pip install hlquantum[cirq]` |
| `BraketBackend` | [Amazon Braket](https://aws.amazon.com/braket/) | `pip install hlquantum[braket]` |
| `PennyLaneBackend` | [Xanadu PennyLane](https://pennylane.ai) | `pip install hlquantum[pennylane]` |

## Installation

```bash
# Core only (no backend dependencies)
pip install .

# With a specific backend
pip install ".[qiskit]"

# With all backends
pip install ".[all]"

# Development
pip install ".[dev]"
```

## Features

- **Backend-Agnostic Circuits** — A single `QuantumCircuit` IR that translates to any supported framework.
- **`@kernel` Decorator** — Write quantum logic as plain Python functions.
- **One-Liner Execution** — `hlquantum.run(circuit, shots=1000)` with automatic backend resolution.
- **Unified Results** — `ExecutionResult` with `.counts`, `.probabilities`, `.most_probable`, and `.expectation_value()`.
- **GPU Acceleration** — Unified `GPUConfig` for GPU execution across CUDA-Q, Qiskit Aer, Cirq (qsim), and PennyLane (lightning.gpu).
- **Lazy Imports** — Backends only import their dependencies when actually used.

## GPU Acceleration

HLQuantum provides a unified `GPUConfig` that works across all GPU-capable backends:

```python
from hlquantum import GPUConfig, GPUPrecision

# Simple — single GPU
gpu = GPUConfig(enabled=True)

# Multi-GPU
gpu = GPUConfig(enabled=True, multi_gpu=True, device_ids=[0, 1])

# FP64 precision
gpu = GPUConfig(enabled=True, precision=GPUPrecision.FP64)

# Enable cuStateVec (Qiskit Aer)
gpu = GPUConfig(enabled=True, custatevec=True)
```

### GPU Support by Backend

| Backend | GPU Library | Auto-selected target / device |
|---------|-------------|-------------------------------|
| `CudaQBackend` | CUDA-Q (native) | `"nvidia"`, `"nvidia-fp64"`, `"nvidia-mqpu"` |
| `QiskitBackend` | qiskit-aer-gpu | `AerSimulator(device='GPU')` |
| `CirqBackend` | qsimcirq | `QSimSimulator(use_gpu=True)` |
| `PennyLaneBackend` | pennylane-lightning[gpu] | `"lightning.gpu"` |
| `BraketBackend` | *(not available)* | *(cloud-managed hardware)* |

### Per-Backend GPU Examples

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

### Detecting Available GPUs

```python
from hlquantum import detect_gpus

for gpu in detect_gpus():
    print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
```

## Quick Start

```python
import hlquantum
from hlquantum import kernel

@kernel(num_qubits=2)
def bell(qc):
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

print(bell.circuit)
# QuantumCircuit(num_qubits=2, gates=4)
```

## Backend Examples

### CUDA-Q

```python
from hlquantum.backends import CudaQBackend

backend = CudaQBackend(target="default")
result = hlquantum.run(bell, shots=1000, backend=backend)
print(result.counts)  # {'00': ~500, '11': ~500}
```

### Qiskit (IBM)

```python
from hlquantum.backends import QiskitBackend

# Local AerSimulator (default)
backend = QiskitBackend()
result = hlquantum.run(bell, shots=1000, backend=backend)

# Real IBM hardware
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
ibm_backend = service.least_busy(min_num_qubits=2)
backend = QiskitBackend(backend=ibm_backend)
```

### Cirq (Google)

```python
from hlquantum.backends import CirqBackend

backend = CirqBackend()
result = hlquantum.run(bell, shots=1000, backend=backend)

# With noise
import cirq
noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.01))
noisy_backend = CirqBackend(noise_model=noise)
```

### Amazon Braket

```python
from hlquantum.backends import BraketBackend

# Local simulator
backend = BraketBackend()
result = hlquantum.run(bell, shots=1000, backend=backend)

# IonQ on AWS
from braket.aws import AwsDevice
ionq = AwsDevice("arn:aws:braket:::device/qpu/ionq/Harmony")
backend = BraketBackend(device=ionq, s3_destination=("my-bucket", "results"))
```

### PennyLane (Xanadu)

```python
from hlquantum.backends import PennyLaneBackend

# default.qubit simulator
backend = PennyLaneBackend()
result = hlquantum.run(bell, shots=1000, backend=backend)

# Lightning fast simulator
backend = PennyLaneBackend(device_name="lightning.qubit")
```

## Working with Results

```python
result = hlquantum.run(bell, shots=1000)

result.counts           # {'00': 512, '11': 488}
result.probabilities    # {'00': 0.512, '11': 0.488}
result.most_probable    # '00'
result.expectation_value()  # 1.0 (parity-based)
result.shots            # 1000
result.backend_name     # 'qiskit (aer_simulator)'

# State Vector (Simulators only)
result = hlquantum.run(bell, include_statevector=True)
sv = result.get_state_vector()
print(sv)  # [0.707+0j, 0, 0, 0.707+0j]

# Transpilation & Error Mitigation
from hlquantum.mitigation import ThresholdMitigation

result = hlquantum.run(
    bell, 
    transpile=True, 
    mitigation=ThresholdMitigation(threshold=0.01)
)

# Built-in Algorithms
from hlquantum import algorithms

# Foundational
qft_circuit = algorithms.qft(num_qubits=4)
bv_circuit = algorithms.bernstein_vazirani("1011")
dj_circuit = algorithms.deutsch_jozsa(num_qubits=2, oracle=my_oracle)
grover_circuit = algorithms.grover(num_qubits=3, target_states=["101"])

# Classical Logic (Quantum Arithmetic)
adder = algorithms.full_adder()

# Hybrid / Optimization (VQE)
from hlquantum.algorithms.vqe import vqe_solve, hardware_efficient_ansatz
ansatz = lambda p: hardware_efficient_ansatz(2, p)
results = vqe_solve(ansatz, initial_params=[0.1, 0.2])
```

## Adding a Custom Backend

```python
from hlquantum.backends import Backend
from hlquantum.circuit import QuantumCircuit
from hlquantum.result import ExecutionResult

class MyBackend(Backend):
    @property
    def name(self) -> str:
        return "my_backend"

    def run(self, circuit: QuantumCircuit, shots: int = 1000, **kwargs) -> ExecutionResult:
        # Translate circuit.gates → your framework
        # Execute and collect counts
        return ExecutionResult(counts={"00": shots}, shots=shots, backend_name=self.name)
```

## Running Tests

```bash
pip install ".[dev]"
pytest tests/ -v
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
