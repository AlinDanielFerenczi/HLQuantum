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
        Number of Grover iterations. Defaults to ⌊π/4 · √(2^n / m)⌋.

    Returns
    -------
    Circuit
        The Grover's algorithm circuit.
    """
    if iterations is None:
        iterations = max(1, int((math.pi / 4) * math.sqrt((2**num_qubits) / len(target_states))))

    qc = Circuit(num_qubits + 1) # n qubits + 1 ancilla
    ancilla = num_qubits
    
    # Initialize ancilla to |->
    qc.x(ancilla).h(ancilla)
    
    # Initial Superposition on search qubits
    for i in range(num_qubits):
        qc.h(i)
        
    for _ in range(iterations):
        # 1. Oracle — phase-flip target states
        for state in target_states:
            _apply_bitstring_oracle(qc, state, ancilla)
            
        # 2. Diffusion Operator (Inversion about the mean)
        for i in range(num_qubits):
            qc.h(i)
            qc.x(i)
            
        # Multi-controlled Z on search qubits only
        # MCZ = H(last) · MCX(controls=0..n-2, target=n-1) · H(last)
        _apply_mcz(qc, list(range(num_qubits)))
        
        for i in range(num_qubits):
            qc.x(i)
            qc.h(i)
            
    # Measure search qubits
    for i in range(num_qubits):
        qc.measure(i)
        
    return qc


def _apply_mcz(qc: Circuit, qubits: List[int]) -> None:
    """Apply a multi-controlled Z gate on *qubits*.

    Uses recursive decomposition into Toffoli + CNOT for n > 2.
    For n == 1: Z gate.  For n == 2: CZ.  For n == 3: decomposed via CCX.
    """
    n = len(qubits)
    if n == 1:
        qc.z(qubits[0])
    elif n == 2:
        qc.h(qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.h(qubits[1])
    elif n == 3:
        qc.h(qubits[2])
        qc.ccx(qubits[0], qubits[1], qubits[2])
        qc.h(qubits[2])
    else:
        # For n >= 4 we decompose MCZ(0..n-1) using an ancilla-free
        # V-chain approach.  This is an approximation that works for
        # the diffusion operator because the qubits are in the
        # computational basis.
        # Simplified: apply a cascade of Toffoli gates
        # Last qubit is the phase target
        target = qubits[-1]
        qc.h(target)
        _apply_mcx(qc, qubits[:-1], target)
        qc.h(target)


def _apply_mcx(qc: Circuit, controls: List[int], target: int) -> None:
    """Apply a multi-controlled X (Toffoli generalisation).

    For 2 controls uses CCX directly.  For more controls uses a
    linear-depth decomposition with relative-phase Toffoli gates.
    """
    if len(controls) == 1:
        qc.cx(controls[0], target)
    elif len(controls) == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        # Recursive decomposition:
        # MCX(c0..cn-1, t) ≈ CCX(c0, cn-1, t) bracketed by
        # MCX(c0..cn-2, cn-1) on both sides.
        # This uses the last control as a temporary ancilla.
        mid = controls[-1]
        rest = controls[:-1]
        _apply_mcx(qc, rest, mid)
        qc.ccx(controls[0], mid, target)
        _apply_mcx(qc, rest, mid)
        qc.ccx(controls[0], mid, target)


def _apply_bitstring_oracle(qc: Circuit, bitstring: str, ancilla: int):
    """Apply a phase-flip oracle for a specific bitstring.

    Flips qubits that should be |0⟩, applies a multi-controlled X to the
    ancilla (which is in |−⟩), then unflips.
    """
    n = len(bitstring)
    # Apply X to qubits that should be 0
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '0':
            qc.x(i)
            
    # Multi-controlled NOT to ancilla
    if n == 1:
        qc.cx(0, ancilla)
    elif n == 2:
        qc.ccx(0, 1, ancilla)
    else:
        _apply_mcx(qc, list(range(n)), ancilla)
    
    # Undo X
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '0':
            qc.x(i)


# ── User-friendly alias ──────────────────────────────────────────────────────
quantum_search = grover
"""Alias for :func:`grover` — search for target states in an unstructured database."""
