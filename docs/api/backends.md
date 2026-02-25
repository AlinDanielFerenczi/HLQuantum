# Backends API Reference

HLQuantum translates its `Circuit` IR into the native representation of each backend.
All backends implement the `Backend` abstract base class and return a unified `ExecutionResult`.

## Supported Backends

| Backend            | Framework          | GPU support | Install extra                      |
| ------------------ | ------------------ | ----------- | ---------------------------------- |
| `CudaQBackend`     | NVIDIA CUDA-Q      | Yes         | `pip install hlquantum[cudaq]`     |
| `QiskitBackend`    | IBM Qiskit         | Yes         | `pip install hlquantum[qiskit]`    |
| `CirqBackend`      | Google Cirq        | Yes         | `pip install hlquantum[cirq]`      |
| `BraketBackend`    | Amazon Braket      | No (cloud)  | `pip install hlquantum[braket]`    |
| `PennyLaneBackend` | Xanadu PennyLane   | Yes         | `pip install hlquantum[pennylane]` |
| `IonQBackend`      | IonQ (qiskit-ionq) | No (cloud)  | `pip install hlquantum[ionq]`      |

## Quick Examples

```python
from hlquantum.backends import (
    CudaQBackend, QiskitBackend, CirqBackend,
    BraketBackend, PennyLaneBackend, IonQBackend,
)
import hlquantum as hlq

# CUDA-Q
result = hlq.run(circuit, backend=CudaQBackend())

# Qiskit (local Aer simulator)
result = hlq.run(circuit, backend=QiskitBackend())

# Cirq
result = hlq.run(circuit, backend=CirqBackend())

# Amazon Braket (local)
result = hlq.run(circuit, backend=BraketBackend())

# PennyLane
result = hlq.run(circuit, backend=PennyLaneBackend())

# IonQ (cloud simulator)
result = hlq.run(circuit, backend=IonQBackend(api_key="..."))
```

## Adding a Custom Backend

```python
from hlquantum.backends import Backend
from hlquantum.result import ExecutionResult

class MyBackend(Backend):
    @property
    def name(self) -> str:
        return "my_backend"

    def run(self, circuit, shots=1000, **kwargs):
        # Translate circuit.gates → your framework, execute, collect counts
        return ExecutionResult(counts={"00": shots}, shots=shots, backend_name=self.name)
```

---

## Base Class

::: hlquantum.backends.base

## CUDA-Q

::: hlquantum.backends.cudaq_backend

## Qiskit

::: hlquantum.backends.qiskit_backend

## Cirq

::: hlquantum.backends.cirq_backend

## Amazon Braket

::: hlquantum.backends.braket_backend

## PennyLane

::: hlquantum.backends.pennylane_backend

## IonQ

::: hlquantum.backends.ionq_backend
