"""
hlquantum.layers.templates
~~~~~~~~~~~~~~~~~~~~~~~~~~

Predefined quantum templates and variational ansätze.
"""

from __future__ import annotations
from typing import List, Optional, Union
from hlquantum.layers.base import Layer
from hlquantum.circuit import Circuit, Parameter


class RealAmplitudes(Layer):
    """An ansatz consisting of single-qubit RY rotations and CX entanglers."""

    def __init__(self, num_qubits: int, reps: int = 1, entanglement: str = "full") -> None:
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        n = input_qubits if input_qubits is not None else self.num_qubits
        qc = Circuit(n)
        
        param_idx = 0
        
        # Initial rotation layer
        for i in range(n):
            qc.ry(i, Parameter(f"theta_{param_idx}"))
            param_idx += 1
            
        for r in range(self.reps):
            # Entanglement layer
            if self.entanglement == "full":
                for i in range(n):
                    for j in range(i + 1, n):
                        qc.cx(i, j)
            elif self.entanglement == "linear":
                for i in range(n - 1):
                    qc.cx(i, i + 1)
            
            # Rotation layer
            for i in range(n):
                qc.ry(i, Parameter(f"theta_{param_idx}"))
                param_idx += 1
                
        return qc


class HardwareEfficientAnsatz(Layer):
    """A general hardware-efficient ansatz with RX, RY, RZ and CX."""

    def __init__(self, num_qubits: int, reps: int = 1) -> None:
        self.num_qubits = num_qubits
        self.reps = reps

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        n = input_qubits if input_qubits is not None else self.num_qubits
        qc = Circuit(n)
        
        param_idx = 0
        for r in range(self.reps):
            for i in range(n):
                qc.rx(i, Parameter(f"theta_{param_idx}"))
                qc.ry(i, Parameter(f"theta_{param_idx + 1}"))
                param_idx += 2
            
            for i in range(n - 1):
                qc.cx(i, i + 1)
                
        return qc
