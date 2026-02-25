# Algorithms API Reference

HLQuantum ships with a library of ready-to-use quantum algorithms.
Each module exposes a **canonical** function name as well as a **friendly alias** for readability.

## Friendly Aliases

| Alias (friendly)         | Canonical function                  | Module                          |
| ------------------------ | ----------------------------------- | ------------------------------- |
| `frequency_transform`    | `qft(num_qubits, inverse)`          | `algorithms.qft`                |
| `quantum_search`         | `grover(num_qubits, target, iters)` | `algorithms.grover`             |
| `find_hidden_pattern`    | `bernstein_vazirani(secret)`        | `algorithms.bernstein_vazirani` |
| `check_balance`          | `deutsch_jozsa(num_qubits, oracle)` | `algorithms.deutsch_jozsa`      |
| `add_two_bits`           | `half_adder()`                      | `algorithms.arithmetic`         |
| `add_three_bits`         | `full_adder()`                      | `algorithms.arithmetic`         |
| `add_numbers`            | `ripple_carry_adder(num_bits)`      | `algorithms.arithmetic`         |
| `find_minimum_energy`    | `vqe_solve(...)`                    | `algorithms.vqe`                |
| `variational_circuit`    | `hardware_efficient_ansatz(...)`    | `algorithms.vqe`                |
| `optimize_combinatorial` | `qaoa_solve(...)`                   | `algorithms.qaoa`               |
| `learn_distribution`     | `gqe_solve(...)`                    | `algorithms.gqe`                |
| `compute_gradient`       | `parameter_shift_gradient(...)`     | `algorithms.grad`               |

## Quick Example

```python
from hlquantum import algorithms

# Use friendly aliases
qft_circuit = algorithms.frequency_transform(num_qubits=4)
bv_circuit  = algorithms.find_hidden_pattern("1011")
search      = algorithms.quantum_search(num_qubits=3, target_states=["101"])

# Variational / optimisation
from hlquantum.algorithms import find_minimum_energy, optimize_combinatorial, compute_gradient

# Parameter-shift gradients
grads = compute_gradient(circuit, {"theta": 0.5})
```

---

## Foundational Algorithms

### Quantum Fourier Transform

::: hlquantum.algorithms.qft

### Grover's Search

::: hlquantum.algorithms.grover

### Bernstein-Vazirani

::: hlquantum.algorithms.bernstein_vazirani

### Deutsch-Jozsa

::: hlquantum.algorithms.deutsch_jozsa

### Quantum Arithmetic

::: hlquantum.algorithms.arithmetic

---

## Variational & Optimisation

### VQE (Variational Quantum Eigensolver)

::: hlquantum.algorithms.vqe

### QAOA (Quantum Approximate Optimisation)

::: hlquantum.algorithms.qaoa

### GQE (Generative Quantum Eigensolver)

::: hlquantum.algorithms.gqe

---

## Differentiable Programming

### Parameter-Shift Gradients

::: hlquantum.algorithms.grad
