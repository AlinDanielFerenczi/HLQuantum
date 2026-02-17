"""
hlquantum.algorithms.grover
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grover's search algorithm implementation.
"""

from __future__ import annotations

import math
from typing import List
from hlquantum.circuit import Circuit


def grover(num_qubits: int, target_states: List[str], iterations: int = None) -> Circuit:
    """Generate a Grover's search circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the search space.
    target_states : list[str]
        List of target bitstrings to search for (e.g., ["101"]).
    iterations : int, optional
        Number of Grover iterations. Defaults to pi/4 * sqrt(2^n / m).

    Returns
    -------
    Circuit
        The Grover's algorithm circuit.
    """
    if iterations is None:
        iterations = int((math.pi / 4) * math.sqrt((2**num_qubits) / len(target_states)))

    qc = Circuit(num_qubits + 1) # n qubits + 1 ancilla
    
    # Initialize ancilla to |->
    qc.x(num_qubits).h(num_qubits)
    
    # Initial Superposition
    for i in range(num_qubits):
        qc.h(i)
        
    for _ in range(iterations):
        # 1. Oracle
        for state in target_states:
            _apply_bitstring_oracle(qc, state, num_qubits)
            
        # 2. Diffusion Operator (Inversion about the mean)
        for i in range(num_qubits):
            qc.h(i)
            qc.x(i)
            
        # Multi-controlled Z (simulated via ancilla)
        if num_qubits == 2:
            qc.h(1).cx(0, 1).h(1)
        elif num_qubits >= 3:
            # Simplification: use CCX if available or approximate
            # For 3 qubits: CCX(0,1,2)
            qc.ccx(0, 1, num_qubits) # This is a placeholder for a real n-controlled gate logic
            # In a real implementation we'd decompose n-controlled Z
        
        for i in range(num_qubits):
            qc.x(i)
            qc.h(i)
            
    # Measure
    for i in range(num_qubits):
        qc.measure(i)
        
    return qc


def _apply_bitstring_oracle(qc: Circuit, bitstring: str, ancilla: int):
    """Auxiliary to apply an oracle for a specific bitstring."""
    n = len(bitstring)
    # Apply X to qubits that should be 0
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '0':
            qc.x(i)
            
    # n-controlled NOT (Multi-Toffoli) to ancilla
    if n == 2:
        qc.ccx(0, 1, ancilla)
    elif n == 3:
        # Simplification
        qc.ccx(0, 1, ancilla) # Placeholder
    
    # Undo X
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '0':
            qc.x(i)
