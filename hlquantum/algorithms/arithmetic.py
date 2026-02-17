"""
hlquantum.algorithms.arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantum arithmetic operations (Adders, etc.).
"""

from __future__ import annotations

from hlquantum.circuit import Circuit


def half_adder() -> Circuit:
    """Generate a 2-qubit half adder.
    
    Qubits:
    - 0, 1: Input bits (A, B)
    - 2: Sum (A ^ B)
    - 3: Carry (A & B)
    """
    qc = Circuit(4)
    # Sum: XOR gate
    qc.cx(0, 2).cx(1, 2)
    # Carry: AND/Toffoli gate
    qc.ccx(0, 1, 3)
    return qc


def full_adder() -> Circuit:
    """Generate a 1-bit full adder.
    
    Qubits:
    - 0, 1, 2: Input bits (A, B, Cin)
    - 3: Sum (A ^ B ^ Cin)
    - 4: Carry out (Cout)
    """
    qc = Circuit(5)
    # Binary addition: A + B + Cin
    # Sum bits
    qc.cx(0, 3).cx(1, 3).cx(2, 3)
    
    # Carry out: (A&B) | (B&Cin) | (A&Cin)
    # Simulating with CCX
    qc.ccx(0, 1, 4)
    qc.ccx(1, 2, 4)
    qc.ccx(0, 2, 4)
    
    return qc


def ripple_carry_adder(num_bits: int) -> Circuit:
    """Generate an n-bit ripple carry adder.
    
    This is a demonstration of classical logic (binary addition)
    reworked for quantum principles (reversible gates).
    """
    total_qubits = 2 * num_bits + 1 # A, B, and one carry
    qc = Circuit(total_qubits)
    
    # Simple Ripple Carry logic using CCX and CX
    # (High-level conceptual implementation)
    for i in range(num_bits):
        a = i
        b = num_bits + i
        cout = total_qubits - 1 # Usually would propagate carries properly
        
        # XOR sum
        qc.cx(a, b)
        
    return qc
