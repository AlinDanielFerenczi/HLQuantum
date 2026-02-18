"""
Demo of advanced algorithms: QAOA, GQE, and Quantum Transformers.
"""

import numpy as np
from hlquantum.algorithms import qaoa_solve, gqe_solve
from hlquantum.layers import QuantumTransformerBlock
from hlquantum.circuit import Circuit

def main():
    print("--- 1. QAOA (Max-Cut Example) ---")
    # Cost Hamiltonian for a simple 3-node triangle graph
    cost_hamiltonian = [
        {'qubits': (0, 1), 'weight': 1.0},
        {'qubits': (1, 2), 'weight': 1.0},
        {'qubits': (2, 0), 'weight': 1.0}
    ]
    
    # QAOA is CPU-intensive for optimization, so we show the setup
    print("Solving QAOA for Max-Cut...")
    # Mocking for speed in demo
    res = qaoa_solve(cost_hamiltonian, p=1, shots=100)
    print(f"QAOA Result: Energy {res['fun']:.4f}")

    print("\n--- 2. Quantum Transformer ---")
    tb = QuantumTransformerBlock(num_qubits=4)
    qc = tb.build()
    print(f"Transformer Circuit built with {len(qc.parameters)} trainable parameters.")
    print(f"Total Gates: {len(qc)}")

    print("\n--- 3. Generative Quantum Eigensolver ---")
    # GQE usually targets a probability distribution
    def simple_loss(result):
        # Target: maximize state '00'
        return -result.counts.get('00', 0)
    
    ansatz = Circuit(2).ry(0, "theta_0").ry(1, "theta_1")
    gqe_res = gqe_solve(ansatz, simple_loss)
    print(f"GQE optimized parameters: {gqe_res['x']}")

if __name__ == "__main__":
    main()
