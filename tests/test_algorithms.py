"""Tests for pre-implemented algorithms."""

import pytest
from hlquantum import Circuit, run, algorithms

class TestAlgorithms:
    def test_qft(self):
        c = algorithms.qft(3)
        assert c.num_qubits == 3
        # Should have H and rotation gates (decomposed to CX, Rz)
        assert len(c.gates) > 10 

    def test_bernstein_vazirani(self):
        secret = "101"
        c = algorithms.bernstein_vazirani(secret)
        assert c.num_qubits == 4 # 3 + 1 ancilla
        
    def test_grover(self):
        c = algorithms.grover(2, ["11"], iterations=1)
        assert c.num_qubits == 3 # 2 + 1 ancilla
        assert len(c.gates) > 5

    def test_deutsch_jozsa(self):
        from hlquantum.algorithms.deutsch_jozsa import constant_oracle
        c = algorithms.deutsch_jozsa(2, constant_oracle)
        assert c.num_qubits == 3

    def test_arithmetic(self):
        ha = algorithms.half_adder()
        assert ha.num_qubits == 4
        fa = algorithms.full_adder()
        assert fa.num_qubits == 5

    def test_vqe_concept(self):
        # We don't necessarily run the optimizer in CI to save time,
        # but we test the ansatz generation.
        from hlquantum.algorithms.vqe import hardware_efficient_ansatz
        ansatz = hardware_efficient_ansatz(2, [0.1, 0.2])
        assert ansatz.num_qubits == 2
        assert len(ansatz.gates) > 0
