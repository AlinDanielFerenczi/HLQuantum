# Core API Reference

The core modules provide the fundamental building blocks: circuits, gates, parameters, kernels, the execution runner, and the unified result type.

## Circuits

The `Circuit` class (aliased as `QuantumCircuit`) is the backend-agnostic intermediate representation.
It supports single- and multi-qubit gates, parameterised rotations, measurements, composition, and parameter binding.

```python
from hlquantum.circuit import Circuit, Parameter

# Build a parameterised circuit
qc = Circuit(2)
qc.rx(0, "theta").cx(0, 1).measure_all()

# Inspect
print(qc.depth)        # critical-path length
print(qc.gate_count)   # total gates
print(qc.parameters)   # [Parameter('theta')]

# Bind parameters
bound = qc.bind_parameters({"theta": 1.57})

# Compose circuits with |
c_total = qc | Circuit(2).h(1)
```

::: hlquantum.circuit

---

## Kernels

The `@kernel` decorator lets you write quantum logic as plain Python functions.

```python
from hlquantum import kernel

@kernel(num_qubits=2)
def bell(qc):
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

print(bell.circuit)  # QuantumCircuit(num_qubits=2, gates=4)
```

::: hlquantum.kernel

---

## Runner

The `run()` function is the high-level entry point. It accepts a circuit or kernel, optionally transpiles it, executes on a backend, and applies error mitigation.

```python
import hlquantum as hlq
from hlquantum.mitigation import ThresholdMitigation

result = hlq.run(
    bell,
    shots=1000,
    transpile=True,
    mitigation=ThresholdMitigation(threshold=0.01),
)
```

::: hlquantum.runner

---

## Results

Every backend returns an `ExecutionResult` dataclass with counts, probabilities, expectation values, and optional state-vector access.

```python
result.counts             # {'00': 512, '11': 488}
result.probabilities      # {'00': 0.512, '11': 0.488}
result.most_probable      # '00'
result.expectation_value()  # 1.0 (parity-based)
result.get_state_vector() # numpy array (simulators only)
```

::: hlquantum.result

---

## Exceptions

All HLQuantum-specific errors inherit from `HLQuantumError`.

| Exception                  | Purpose                    |
| -------------------------- | -------------------------- |
| `HLQuantumError`           | Base class                 |
| `BackendError`             | Backend execution failure  |
| `CircuitValidationError`   | Invalid circuit structure  |
| `BackendNotAvailableError` | Required SDK not installed |

::: hlquantum.exceptions
