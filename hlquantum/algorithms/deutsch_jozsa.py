"""
hlquantum.algorithms.deutsch_jozsa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deutsch-Jozsa algorithm implementation.
"""

from __future__ import annotations

from typing import Callable
from hlquantum.circuit import Circuit


def deutsch_jozsa(num_qubits: int, oracle: Callable[[Circuit], None]) -> Circuit:
    """Generate a Deutsch-Jozsa circuit.

    Parameters
    ----------
    num_qubits : int
        Number of input qubits (n).
    oracle : Callable[[Circuit], None]
        A function that applies the oracle to a Circuit of size n+1.

    Returns
    -------
    Circuit
        The Deutsch-Jozsa circuit.
    """
    qc = Circuit(num_qubits + 1)
    
    # 1. Initialize ancilla to |->
    qc.x(num_qubits).h(num_qubits)
    
    # 2. Apply Hadamard to all input qubits
    for i in range(num_qubits):
        qc.h(i)
        
    # 3. Apply the Oracle
    oracle(qc)
    
    # 4. Apply Hadamard to input qubits again
    for i in range(num_qubits):
        qc.h(i)
        
    # 5. Measure input qubits
    for i in range(num_qubits):
        qc.measure(i)
        
    return qc


def constant_oracle(qc: Circuit):
    """Example of a constant oracle (does nothing)."""
    pass


def balanced_oracle(qc: Circuit):
    """Example of a balanced oracle (CX from q0 to ancilla)."""
    qc.cx(0, qc.num_qubits - 1)
