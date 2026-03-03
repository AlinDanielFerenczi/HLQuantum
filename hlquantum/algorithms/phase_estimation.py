"""
hlquantum.algorithms.phase_estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantum Phase Estimation (QPE) implementation.
"""

from __future__ import annotations

from typing import Callable

from hlquantum.circuit import Circuit
from hlquantum.algorithms.qft import qft


def phase_estimation(
    num_evaluation_qubits: int,
    num_target_qubits: int,
    controlled_unitary: Callable[[Circuit, int, int], None]
) -> Circuit:
    """Generate a Quantum Phase Estimation circuit.

    Parameters
    ----------
    num_evaluation_qubits : int
        Number of qubits in the evaluation register (indices 0 to num_evaluation_qubits - 1).
    num_target_qubits : int
        Number of qubits in the target register (indices num_evaluation_qubits onwards).
    controlled_unitary : Callable[[Circuit, int, int], None]
        A function that applies the controlled unitary $C-U^{power}$.
        Signature: `def cu(qc: Circuit, control_qubit: int, power: int): ...`

    Returns
    -------
    Circuit
        The Phase Estimation circuit.
    """
    total_qubits = num_evaluation_qubits + num_target_qubits
    qc = Circuit(total_qubits)
    
    # 1. Initialize evaluation qubits in superposition
    for i in range(num_evaluation_qubits):
        qc.h(i)
        
    # 2. Apply controlled unitaries
    for j in range(num_evaluation_qubits):
        power = 2 ** j
        # C-U^(2^j) controlled on the j-th qubit 
        controlled_unitary(qc, j, power)
        
    # 3. Apply Inverse QFT to the evaluation register
    iqft_qc = qft(num_evaluation_qubits, inverse=True)
    
    # Composition __or__ concatenates gates onto the same qubit indices
    qc = qc | iqft_qc
    
    # 4. Measure evaluation qubits
    for i in range(num_evaluation_qubits):
        qc.measure(i)
        
    return qc

# ── User-friendly alias ──────────────────────────────────────────────────────
estimate_phase = phase_estimation
"""Alias for :func:`phase_estimation` — estimate the eigenvalue phase of a unitary operator."""
