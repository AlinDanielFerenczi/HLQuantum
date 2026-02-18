"""
hlquantum.layers.functional
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functional layers wrapping standard algorithms.
"""

from __future__ import annotations
from typing import List, Optional
from hlquantum.layers.base import Layer
from hlquantum.circuit import Circuit
from hlquantum.algorithms.grover import grover
from hlquantum.algorithms.qft import qft


class GroverLayer(Layer):
    """A layer implementing Grover's search."""

    def __init__(self, num_qubits: int, target_states: List[str], iterations: Optional[int] = None) -> None:
        self.num_qubits = num_qubits
        self.target_states = target_states
        self.iterations = iterations

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        n = input_qubits if input_qubits is not None else self.num_qubits
        return grover(n, self.target_states, self.iterations)


class QFTLayer(Layer):
    """A layer implementing Quantum Fourier Transform."""

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        n = input_qubits if input_qubits is not None else self.num_qubits
        return qft(n)
