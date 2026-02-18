"""
hlquantum.layers.base
~~~~~~~~~~~~~~~~~~~~~

Base classes for quantum layers and modular circuit building.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from hlquantum.circuit import Circuit


class Layer(ABC):
    """Abstract base class for a quantum layer."""

    @abstractmethod
    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        """Build the module and return a QuantumCircuit."""
        pass

    def __or__(self, other: Layer) -> Sequential:
        """Pipe layers together."""
        if isinstance(other, Layer):
            return Sequential([self, other])
        return NotImplemented


class Sequential(Layer):
    """A container for a sequence of quantum layers."""

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        if not self.layers:
            raise ValueError("Sequential model must have at least one layer.")

        # Determine number of qubits if not provided
        if input_qubits is None:
            # We'll just build them one by one and let them determine their size
            # or rely on the first layer's requirement.
            # For simplicity, we assume the user knows what they are doing
            # or we take the max num_qubits from all built circuits.
            pass

        full_circuit: Optional[Circuit] = None
        for layer in self.layers:
            circuit = layer.build(input_qubits)
            if full_circuit is None:
                full_circuit = circuit
            else:
                full_circuit = full_circuit | circuit
            
            # Update input_qubits for next layer to match current circuit size
            input_qubits = full_circuit.num_qubits

        return full_circuit

    def __or__(self, other: Layer) -> Sequential:
        if isinstance(other, Sequential):
            return Sequential(self.layers + other.layers)
        if isinstance(other, Layer):
            return Sequential(self.layers + [other])
        return NotImplemented
