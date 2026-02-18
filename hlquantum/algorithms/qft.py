"""
hlquantum.algorithms.qft
~~~~~~~~~~~~~~~~~~~~~~~~

Quantum Fourier Transform (QFT) implementation.
"""

from __future__ import annotations

import math
from hlquantum.circuit import Circuit


def qft(num_qubits: int, inverse: bool = False) -> Circuit:
    """Generate a Quantum Fourier Transform circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits to apply the QFT on.
    inverse : bool, optional
        If True, generate the Inverse QFT (IQFT).

    Returns
    -------
    Circuit
        The QFT or IQFT circuit.
    """
    qc = Circuit(num_qubits)
    
    if inverse:
        # Inverse QFT
        # 1. Swaps
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - i - 1)
        
        # 2. Gates in reverse
        for i in range(num_qubits):
            for j in range(i):
                angle = -math.pi / (2**(i - j))
                qc.rz(i, angle / 2) # Note: This is an approximation of controlled-Rz
                # In a real QFT we need CRz, which we can decompose or add to IR
                # For now, let's stick to the gates we have: cx, rz, h.
                # Controlled-Phase(theta) = Rz(theta/2) -> CX -> Rz(-theta/2) -> CX
                angle = math.pi / (2**(i - j))
                qc.rz(j, -angle/2)
                qc.cx(i, j)
                qc.rz(j, angle/2)
                qc.cx(i, j)
            qc.h(i)
    else:
        # Standard QFT
        for i in range(num_qubits - 1, -1, -1):
            qc.h(i)
            for j in range(i - 1, -1, -1):
                angle = math.pi / (2**(i - j))
                # Decomposing CP(angle) using CX and Rz
                qc.rz(j, angle/2)
                qc.cx(i, j)
                qc.rz(j, -angle/2)
                qc.cx(i, j)
                qc.rz(i, angle/2)
        
        # Swaps
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - i - 1)
            
    return qc
