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

    Notes
    -----
    Controlled-Phase(θ) is decomposed as:
        Rz(target, θ/2)  →  CX(control, target)  →  Rz(target, -θ/2)  →
        CX(control, target)  →  Rz(control, θ/2)
    """
    qc = Circuit(num_qubits)
    
    if inverse:
        # Inverse QFT — reverse of forward QFT
        # 1. Undo swaps
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - i - 1)
        
        # 2. Apply inverse rotations in reverse order
        for i in range(num_qubits):
            for j in range(i):
                angle = -math.pi / (2 ** (i - j))
                # Decompose controlled-phase(-angle) = CP(angle)†
                qc.rz(j, angle / 2)
                qc.cx(i, j)
                qc.rz(j, -angle / 2)
                qc.cx(i, j)
                qc.rz(i, angle / 2)
            qc.h(i)
    else:
        # Standard QFT
        for i in range(num_qubits - 1, -1, -1):
            qc.h(i)
            for j in range(i - 1, -1, -1):
                angle = math.pi / (2 ** (i - j))
                # Decompose CP(angle) into CX + Rz
                qc.rz(j, angle / 2)
                qc.cx(i, j)
                qc.rz(j, -angle / 2)
                qc.cx(i, j)
                qc.rz(i, angle / 2)
        
        # Swaps
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - i - 1)
            
    return qc


# ── User-friendly alias ──────────────────────────────────────────────────────
frequency_transform = qft
"""Alias for :func:`qft` — transform quantum amplitudes into the frequency domain."""
