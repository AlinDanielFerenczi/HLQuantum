"""
hlquantum.algorithms
~~~~~~~~~~~~~~~~~~~~

Pre-implemented quantum algorithms.
"""

from hlquantum.algorithms.qft import qft
from hlquantum.algorithms.grover import grover
from hlquantum.algorithms.bernstein_vazirani import bernstein_vazirani
from hlquantum.algorithms.deutsch_jozsa import deutsch_jozsa
from hlquantum.algorithms.arithmetic import half_adder, full_adder, ripple_carry_adder
from hlquantum.algorithms.vqe import vqe_solve, hardware_efficient_ansatz
from hlquantum.algorithms.qaoa import qaoa_solve
from hlquantum.algorithms.gqe import gqe_solve
from hlquantum.algorithms.grad import parameter_shift_gradient

__all__ = [
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
]
