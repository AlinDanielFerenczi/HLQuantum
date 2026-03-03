"""Grover's search algorithm implementation."""

from __future__ import annotations

import math
from typing import List
from hlquantum.circuit import Circuit


def grover(num_qubits: int, target_states: List[str], iterations: int = None) -> Circuit:
    """Generate a Grover's search circuit."""
    if iterations is None:
        iterations = max(1, int((math.pi / 4) * math.sqrt((2**num_qubits) / len(target_states))))

    qc = Circuit(num_qubits + 1)
    ancilla = num_qubits
    qc.x(ancilla).h(ancilla)
    
    for i in range(num_qubits):
        qc.h(i)
        
    for _ in range(iterations):
        for state in target_states:
            _apply_bitstring_oracle(qc, state, ancilla)
            
        for i in range(num_qubits):
            qc.h(i)
            qc.x(i)
            
        _apply_mcz(qc, list(range(num_qubits)))
        
        for i in range(num_qubits):
            qc.x(i)
            qc.h(i)
            
    for i in range(num_qubits):
        qc.measure(i)
    return qc


def _apply_mcz(qc: Circuit, qubits: List[int]) -> None:
    """Apply multi-controlled Z on specified qubits."""
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
        target = qubits[-1]
        qc.h(target)
        _apply_mcx(qc, qubits[:-1], target)
        qc.h(target)


def _apply_mcx(qc: Circuit, controls: List[int], target: int) -> None:
    """Apply multi-controlled X."""
    if len(controls) == 1:
        qc.cx(controls[0], target)
    elif len(controls) == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        mid = controls[-1]
        rest = controls[:-1]
        _apply_mcx(qc, rest, mid)
        qc.ccx(controls[0], mid, target)
        _apply_mcx(qc, rest, mid)
        qc.ccx(controls[0], mid, target)


def _apply_bitstring_oracle(qc: Circuit, bitstring: str, ancilla: int):
    """Apply phase-flip oracle for a bitstring."""
    n = len(bitstring)
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '0':
            qc.x(i)
            
    if n == 1:
        qc.cx(0, ancilla)
    elif n == 2:
        qc.ccx(0, 1, ancilla)
    else:
        _apply_mcx(qc, list(range(n)), ancilla)
    
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '0':
            qc.x(i)


quantum_search = grover

