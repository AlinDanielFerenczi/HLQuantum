# Examples

HLQuantum ships with a collection of runnable example scripts in the `examples/` directory.
Each demonstrates a different aspect of the library.

## Example Index

| Script                                                                                                            | What it demonstrates                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [`hybrid_workflow_demo.py`](https://github.com/user/hlquantum/blob/main/examples/hybrid_workflow_demo.py)         | Quantum → classical hybrid workflow with `ClassicalNode`, `Branch`, and Mermaid export.                                  |
| [`complex_system_demo.py`](https://github.com/user/hlquantum/blob/main/examples/complex_system_demo.py)           | `Sequential` pipeline with `CircuitLayer`, `QFTLayer`, `GroverLayer`; full `Workflow` with `Parallel`, `Loop`, `Branch`. |
| [`advanced_features_demo.py`](https://github.com/user/hlquantum/blob/main/examples/advanced_features_demo.py)     | `RealAmplitudes` ansatz, parameter binding, async workflows with `Parallel`.                                             |
| [`advanced_algorithms_demo.py`](https://github.com/user/hlquantum/blob/main/examples/advanced_algorithms_demo.py) | QAOA (Max-Cut), GQE, `QuantumTransformerBlock`.                                                                          |
| [`parallel_algorithms_demo.py`](https://github.com/user/hlquantum/blob/main/examples/parallel_algorithms_demo.py) | Friendly algorithm aliases combined with a `Parallel` workflow for concurrent execution.                                 |
| [`parameter_shift_demo.py`](https://github.com/user/hlquantum/blob/main/examples/parameter_shift_demo.py)         | `compute_gradient` via the parameter-shift rule with a mock backend.                                                     |
| [`resilient_workflow_demo.py`](https://github.com/user/hlquantum/blob/main/examples/resilient_workflow_demo.py)   | `WorkflowRunner` throttling and checkpoint save/resume.                                                                  |
| [`cudaq_gpu_demo.py`](https://github.com/user/hlquantum/blob/main/examples/cudaq_gpu_demo.py)                     | Bernstein-Vazirani on CUDA-Q with single/multi-GPU and FP64 `GPUConfig`.                                                 |
| [`ionq_demo.py`](https://github.com/user/hlquantum/blob/main/examples/ionq_demo.py)                               | `@kernel` decorator with `IonQBackend` (simulator and QPU).                                                              |
| [`state_machine_loop_demo.py`](https://github.com/user/hlquantum/blob/main/examples/state_machine_loop_demo.py)   | State-machine workflow with an async `Loop`, classical evaluation, and convergence checking.                             |

## Running an Example

```bash
# Make sure you're in the project root with the package installed
pip install -e ".[dev]"

# Run any example
python examples/hybrid_workflow_demo.py
python examples/complex_system_demo.py
```

Most examples use a lightweight mock backend so they work without any real quantum SDK installed.
Examples that target a specific backend (CUDA-Q, IonQ) will print a clear error message if that SDK is missing.
