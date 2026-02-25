# HLQuantum

**HLQuantum** (High Level Quantum) is a high-level Python package designed to simplify working with quantum hardware. Write your quantum logic once and run it on any supported backend.

## Supported Backends

| Backend            | Framework                                              | Install extra                      |
| ------------------ | ------------------------------------------------------ | ---------------------------------- |
| `CudaQBackend`     | [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum) | `pip install hlquantum[cudaq]`     |
| `QiskitBackend`    | [IBM Qiskit](https://qiskit.org)                       | `pip install hlquantum[qiskit]`    |
| `CirqBackend`      | [Google Cirq](https://quantumai.google/cirq)           | `pip install hlquantum[cirq]`      |
| `BraketBackend`    | [Amazon Braket](https://aws.amazon.com/braket/)        | `pip install hlquantum[braket]`    |
| `PennyLaneBackend` | [Xanadu PennyLane](https://pennylane.ai)               | `pip install hlquantum[pennylane]` |
| `IonQBackend`      | [IonQ](https://ionq.com) (via qiskit-ionq)             | `pip install hlquantum[ionq]`      |

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
- **Quantum Pipelines** — Build modular architectures using ML-inspired `Layer` and `Sequential` models.
- **Resilient Workflows** — Orchestrate complex executions with loops, branching, and state persistence (save/resume).
- **Asynchronous Execution** — Multi-backend concurrency with `async/await` support.
- **Unitary-Agnostic @kernel** — Write quantum logic as plain Python functions.
- **GPU Acceleration** — Unified `GPUConfig` across all backends.

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

| Backend            | GPU Library              | Auto-selected target / device                |
| ------------------ | ------------------------ | -------------------------------------------- |
| `CudaQBackend`     | CUDA-Q (native)          | `"nvidia"`, `"nvidia-fp64"`, `"nvidia-mqpu"` |
| `QiskitBackend`    | qiskit-aer-gpu           | `AerSimulator(device='GPU')`                 |
| `CirqBackend`      | qsimcirq                 | `QSimSimulator(use_gpu=True)`                |
| `PennyLaneBackend` | pennylane-lightning[gpu] | `"lightning.gpu"`                            |
| `BraketBackend`    | _(not available)_        | _(cloud-managed hardware)_                   |
| `IonQBackend`      | _(not available)_        | _(cloud-managed trapped-ion hardware)_       |

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

```python
from hlquantum import detect_gpus

for gpu in detect_gpus():
    print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
```

## Quantum Pipelines (ML-Style)

Build complex circuits modularly by stacking layers:

```python
from hlquantum.layers import Sequential, GroverLayer, QFTLayer, RealAmplitudes

# Stack algorithms and variational layers
model = Sequential([
    QFTLayer(num_qubits=4),
    GroverLayer(num_qubits=4, target_states=["1010"]),
    RealAmplitudes(num_qubits=4, reps=2)
])

# Compile to a single circuit
circuit = model.build()
```

## Resilient Workflows

Orchestrate complex execution flows with automatic state persistence and parallel execution.

```python
from hlquantum.workflows import Workflow, Parallel, Loop, Branch

wf = Workflow(state_file="checkpoint.json", name="Discovery")

# Add parallel paths
wf.add(Parallel(circuit1, circuit2))

# Add a loop
wf.add(Loop(base_circuit, iterations=10))

# Execute asynchronously (with optional throttling for rate-limits)
import asyncio
results = asyncio.run(wf.run(resume=True))

# Export to Mermaid for visualization
print(wf.to_mermaid())
```

### Hybrid Quantum–Classical Workflows

Classical post-processing functions can run in the same workflow as quantum circuits.
Each node receives a context dict with results from all prior steps:

```python
from hlquantum.workflows import Workflow, Branch, WorkflowRunner

wf = Workflow(name="HybridPipeline")

# Quantum step
wf.add(Circuit(2).h(0).cx(0, 1).measure_all(), name="bell")

# Classical step — extract and analyse
wf.add(lambda ctx: ctx["previous_result"].counts, name="extract_counts")
wf.add(lambda ctx: sum(ctx["previous_result"].get(k, 0) for k in ("00", "11")) / 1000, name="correlation")

# Branch on the result
wf.add(Branch(
    lambda ctx: ctx["previous_result"] > 0.9,
    lambda ctx: "entangled",
    lambda ctx: "not entangled",
), name="classify")

results = asyncio.run(wf.run())
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

backend = CudaQBackend(target="nvidia")
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

### IonQ

```python
from hlquantum.backends import IonQBackend

# IonQ cloud simulator (default)
backend = IonQBackend(api_key="your-ionq-api-key")
result = hlquantum.run(bell, shots=1000, backend=backend)

# IonQ trapped-ion QPU
backend = IonQBackend(backend_name="ionq_qpu", api_key="your-ionq-api-key")
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
qft_circuit = algorithms.frequency_transform(num_qubits=4)
bv_circuit = algorithms.find_hidden_pattern("1011")
search_circuit = algorithms.quantum_search(num_qubits=3, target_states=["101"])

# Classical Logic (Quantum Arithmetic)
adder = algorithms.add_two_bits()

# Variational & Optimization
from hlquantum.algorithms import find_minimum_energy, optimize_combinatorial, learn_distribution

# VQE with parameterized circuits
res = find_minimum_energy(my_ansatz, initial_params=[0.1, 0.2])

# QAOA for combinatorial optimization
res = optimize_combinatorial(cost_hamiltonian, p=2)

# GQE for generative modeling
res = learn_distribution(ansatz, my_loss_fn)

# Differentiable Programming
from hlquantum.algorithms import compute_gradient
grads = compute_gradient(circuit, {"theta": 0.5})
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

## Documentation

Full documentation — including API reference for all modules (circuits, backends,
algorithms, layers, workflows, transpiler, mitigation, GPU) and runnable examples —
is available via MkDocs:

```bash
pip install ".[dev]"
mkdocs serve
```

## Sponsors

HLQuantum is made possible with the support of our sponsors. If you'd like to support this project, please reach out.

|     | Sponsor                                        | Description                                     |
| --- | ---------------------------------------------- | ----------------------------------------------- |
|     | [**Venture Chain**](https://venture-chain.com) | Supporting initial development effort & release |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
