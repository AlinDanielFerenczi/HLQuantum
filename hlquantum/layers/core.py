"""
hlquantum.layers.core
~~~~~~~~~~~~~~~~~~~~~

Core layer implementations.
"""

from __future__ import annotations
from typing import Optional
from hlquantum.layers.base import Layer
from hlquantum.circuit import Circuit


class CircuitLayer(Layer):
    """Wraps an existing QuantumCircuit as a layer."""

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        # If input_qubits is more than circuit.num_qubits, we might want to expand?
        # For now, just return the circuit.
        return self.circuit
