# HLQuantum

Welcome to the official documentation for **HLQuantum** (High Level Quantum).

HLQuantum is a high-level Python package designed to simplify working with quantum hardware. Write your quantum logic once and run it on any supported backend (CUDA-Q, Qiskit, Cirq, Braket, PennyLane).

## Quick Start

```python
import hlquantum as hlq

# Create a circuit
qc = hlq.Circuit(2)
qc.h(0).cx(0, 1).measure_all()

# Run it
result = hlq.run(qc, shots=1000)
print(result.counts)
```

## Features

- **Backend-Agnostic**: One circuit, many backends.
- **GPU Accelerated**: Unified interface for GPU execution.
- **Transpilation**: built-in optimizations for your circuits.
- **Error Mitigation**: Hooks for post-processing noisy results.
