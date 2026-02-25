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
    """Generate an n-bit ripple-carry adder.

    Qubit layout:
        a[0..n-1]   — first operand
        b[0..n-1]   — second operand (sum is written here in-place)
        c[0..n]     — carry chain (c[0]=carry-in, c[n]=carry-out)

    Total qubits: 3*num_bits + 1
    After execution b[i] holds the i-th bit of (A+B) and c[n] holds the
    final carry-out.
    """
    n = num_bits
    total_qubits = 3 * n + 1  # a[n], b[n], c[n+1]
    qc = Circuit(total_qubits)

    def a(i: int) -> int:
        return i

    def b(i: int) -> int:
        return n + i

    def c(i: int) -> int:
        return 2 * n + i

    # Forward pass — propagate carries
    for i in range(n):
        # Carry: c[i+1] = MAJ(a[i], b[i], c[i])
        # Step 1: XOR partial sums
        qc.cx(a(i), b(i))          # b[i] ^= a[i]
        qc.cx(c(i), b(i))          # b[i] ^= c[i]  →  b[i] = a[i]⊕b[i]⊕c[i] (partial sum)
        # Step 2: Carry generation  c[i+1] = (a[i]·c[i]) ⊕ (b_orig[i]·c[i]) ⊕ (a[i]·b_orig[i])
        qc.ccx(a(i), c(i), c(i + 1))
        qc.cx(a(i), c(i))          # restore helper
        qc.ccx(b(i), c(i), c(i + 1))
        qc.cx(a(i), c(i))          # restore c[i]

    # b already holds the XOR (partial sum) from the forward pass.
    # The full sum bit is simply xor of a, b_orig, c_in which was
    # accumulated into b during the forward pass.  No additional
    # work needed — b[i] already equals Sum[i].\n
    return qc


# ── User-friendly aliases ────────────────────────────────────────────────────
add_two_bits = half_adder
"""Alias for :func:`half_adder` — add two single-bit inputs."""

add_three_bits = full_adder
"""Alias for :func:`full_adder` — add two bits with a carry-in."""

add_numbers = ripple_carry_adder
"""Alias for :func:`ripple_carry_adder` — add two n-bit numbers."""
