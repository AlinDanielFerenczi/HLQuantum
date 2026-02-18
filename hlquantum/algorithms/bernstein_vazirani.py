"""
hlquantum.algorithms.bernstein_vazirani
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bernstein-Vazirani algorithm implementation.
"""

from __future__ import annotations

from hlquantum.circuit import Circuit


def bernstein_vazirani(secret_bitstring: str) -> Circuit:
    """Generate a Bernstein-Vazirani circuit to find a secret bitstring.

    Parameters
    ----------
    secret_bitstring : str
        The secret bitstring (e.g., "1011") that the algorithm will discover.

    Returns
    -------
    Circuit
        The Bernstein-Vazirani circuit.
    """
    n = len(secret_bitstring)
    qc = Circuit(n + 1) # n qubits + 1 ancilla
    
    # Initialize ancilla to |->
    qc.x(n).h(n)
    
    # Apply Hadamard to all input qubits
    for i in range(n):
        qc.h(i)
        
    # Apply Oracle: CX from q_i to ancilla if bit_i is 1
    for i, bit in enumerate(reversed(secret_bitstring)):
        if bit == '1':
            qc.cx(i, n)
            
    # Apply Hadamard to input qubits again
    for i in range(n):
        qc.h(i)
        
    # Measure input qubits
    for i in range(n):
        qc.measure(i)
        
    return qc
