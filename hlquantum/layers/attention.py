"""
hlquantum.layers.attention
~~~~~~~~~~~~~~~~~~~~~~~~~

Quantum Transformers and Attention Mechanisms.
"""

from __future__ import annotations
from typing import List, Optional
from hlquantum.layers.base import Layer
from hlquantum.circuit import Circuit, Parameter


class QuantumMultiHeadAttention(Layer):
    """A Quantum Multi-Head Attention layer.
    
    This layer encodes classical data into quantum states and applies a 
    parameterized circuit to compute attention-like features.
    """

    def __init__(self, num_qubits: int, n_heads: int = 1):
        self.num_qubits = num_qubits
        self.n_heads = n_heads

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        n = input_qubits if input_qubits is not None else self.num_qubits
        qc = Circuit(n)
        
        # Parallel Attention Heads
        for head in range(self.n_heads):
            # Query/Key mapping (simplified as rotations)
            for i in range(n):
                qc.ry(i, Parameter(f"head_{head}_rot_{i}"))
            
            # Entangling layer (Attention mechanism)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            
        return qc


class QuantumTransformerBlock(Layer):
    """A high-level block containing Quantum Attention and Feed-Forward layers."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def build(self, input_qubits: Optional[int] = None) -> Circuit:
        n = input_qubits if input_qubits is not None else self.num_qubits
        
        # Combine Attention and variational layers
        from hlquantum.layers.templates import RealAmplitudes
        
        qc = QuantumMultiHeadAttention(n).build()
        qc = qc | RealAmplitudes(n, reps=1).build()
        
        return qc
