"""
hlquantum.algorithms
~~~~~~~~~~~~~~~~~~~~

Pre-implemented quantum algorithms.

Each algorithm is available under both its canonical name and a
user-friendly alias that describes the use case:

================================================  ========================
Friendly name                                     Canonical name
================================================  ========================
``frequency_transform``                           ``qft``
``quantum_search``                                ``grover``
``find_hidden_pattern``                           ``bernstein_vazirani``
``check_balance``                                 ``deutsch_jozsa``
``add_two_bits``                                  ``half_adder``
``add_three_bits``                                ``full_adder``
``add_numbers``                                   ``ripple_carry_adder``
``find_minimum_energy``                           ``vqe_solve``
``variational_circuit``                           ``hardware_efficient_ansatz``
``optimize_combinatorial``                        ``qaoa_solve``
``learn_distribution``                            ``gqe_solve``
``compute_gradient``                              ``parameter_shift_gradient``
================================================  ========================
"""

from hlquantum.algorithms.qft import qft, frequency_transform
from hlquantum.algorithms.grover import grover, quantum_search
from hlquantum.algorithms.bernstein_vazirani import bernstein_vazirani, find_hidden_pattern
from hlquantum.algorithms.deutsch_jozsa import deutsch_jozsa, check_balance
from hlquantum.algorithms.arithmetic import (
    half_adder, full_adder, ripple_carry_adder,
    add_two_bits, add_three_bits, add_numbers,
)
from hlquantum.algorithms.vqe import (
    vqe_solve, hardware_efficient_ansatz,
    find_minimum_energy, variational_circuit,
)
from hlquantum.algorithms.qaoa import qaoa_solve, optimize_combinatorial
from hlquantum.algorithms.gqe import gqe_solve, learn_distribution
from hlquantum.algorithms.grad import parameter_shift_gradient, compute_gradient

__all__ = [
    # Canonical names (backward-compatible)
    "qft",
    "grover",
    "bernstein_vazirani",
    "deutsch_jozsa",
    "half_adder",
    "full_adder",
    "ripple_carry_adder",
    "vqe_solve",
    "hardware_efficient_ansatz",
    "qaoa_solve",
    "gqe_solve",
    "parameter_shift_gradient",
    # User-friendly aliases
    "frequency_transform",
    "quantum_search",
    "find_hidden_pattern",
    "check_balance",
    "add_two_bits",
    "add_three_bits",
    "add_numbers",
    "find_minimum_energy",
    "variational_circuit",
    "optimize_combinatorial",
    "learn_distribution",
    "compute_gradient",
]
