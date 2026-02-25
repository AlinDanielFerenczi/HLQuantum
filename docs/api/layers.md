# Layers & Pipelines API Reference

HLQuantum borrows the _layer / sequential_ pattern from ML frameworks to let you compose complex quantum circuits from reusable building blocks.

## Key Concepts

| Class                       | Purpose                                                     |
| --------------------------- | ----------------------------------------------------------- |
| `Layer`                     | Abstract base — every layer implements `build() → Circuit`. |
| `Sequential`                | Container that chains layers and composes their circuits.   |
| `CircuitLayer`              | Wraps an existing `Circuit` as a layer.                     |
| `QFTLayer`                  | Layer wrapping the QFT algorithm.                           |
| `GroverLayer`               | Layer wrapping Grover's search.                             |
| `RealAmplitudes`            | RY + CX variational ansatz (full / linear entanglement).    |
| `HardwareEfficientAnsatz`   | RX + RY + CX hardware-efficient ansatz.                     |
| `QuantumMultiHeadAttention` | Quantum multi-head attention mechanism.                     |
| `QuantumTransformerBlock`   | Attention + variational feed-forward block.                 |

## Quick Example

```python
from hlquantum.layers import (
    Sequential, CircuitLayer, QFTLayer, GroverLayer, RealAmplitudes,
)
from hlquantum.circuit import Circuit

# Wrap a hand-crafted circuit
init = CircuitLayer(Circuit(4).h(0).cx(0, 1))

# Stack layers into a pipeline
model = Sequential([
    init,
    QFTLayer(num_qubits=4),
    GroverLayer(num_qubits=4, target_states=["1010"]),
    RealAmplitudes(num_qubits=4, reps=2),
])

# Compile to a single circuit
circuit = model.build()
print(circuit)
```

Layers can also be composed with the `|` (pipe) operator:

```python
from hlquantum.layers.core import CircuitLayer

a = CircuitLayer(Circuit(2).h(0))
b = CircuitLayer(Circuit(2).cx(0, 1))
combined = a | b          # returns a Sequential
circuit = combined.build()
```

---

## Base & Container

::: hlquantum.layers.base

## Core Layers

::: hlquantum.layers.core

## Functional Layers

::: hlquantum.layers.functional

## Variational Templates

::: hlquantum.layers.templates

## Attention & Transformer Layers

::: hlquantum.layers.attention
