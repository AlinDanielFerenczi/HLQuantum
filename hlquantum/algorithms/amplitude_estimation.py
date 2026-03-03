"""
hlquantum.algorithms.amplitude_estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantum Amplitude Estimation (AE) implementation.
"""

from __future__ import annotations

from typing import Callable

from hlquantum.circuit import Circuit
from hlquantum.algorithms.qft import qft


def amplitude_estimation(
    num_evaluation_qubits: int,
    num_state_qubits: int,
    state_preparation: Callable[[Circuit], None],
    controlled_grover: Callable[[Circuit, int, int], None]
) -> Circuit:
    """Generate a Quantum Amplitude Estimation circuit.

    Parameters
    ----------
    num_evaluation_qubits : int
        Number of evaluation qubits for phase estimation.
    num_state_qubits : int
        Number of target qubits for the state preparation.
    state_preparation : Callable[[Circuit], None]
        A function that applies the operator $A$ to prepare the state $|\psi\\rangle = A|0\\rangle$.
        Signature: `def prep(qc: Circuit): ...`
    controlled_grover : Callable[[Circuit, int, int], None]
        A function that applies $C-Q^{power}$, where $Q$ is the Grover operator.
        Signature: `def cq(qc: Circuit, control_qubit: int, power: int): ...`

    Returns
    -------
    Circuit
        The Amplitude Estimation circuit.
    """
    total_qubits = num_evaluation_qubits + num_state_qubits
    qc = Circuit(total_qubits)
    
    # 1. Prepare target state A|0>
    state_preparation(qc)
    
    # 2. Initialize evaluation qubits in superposition
    for i in range(num_evaluation_qubits):
        qc.h(i)
        
    # 3. Apply controlled Grover operations Q^(2^j)
    for j in range(num_evaluation_qubits):
        power = 2 ** j
        controlled_grover(qc, j, power)
        
    # 4. Apply Inverse QFT to the evaluation register
    iqft_qc = qft(num_evaluation_qubits, inverse=True)
    qc = qc | iqft_qc
    
    # 5. Measure evaluation qubits
    for i in range(num_evaluation_qubits):
        qc.measure(i)
        
    return qc


# ── User-friendly alias ──────────────────────────────────────────────────────
estimate_amplitude = amplitude_estimation
"""Alias for :func:`amplitude_estimation` — estimate the amplitude of marked states."""
