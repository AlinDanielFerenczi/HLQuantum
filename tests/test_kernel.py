"""Tests for hlquantum.kernel."""

from hlquantum.kernel import Kernel, kernel
from hlquantum.circuit import QuantumCircuit


class TestKernelDecorator:
    def test_basic_kernel(self):
        @kernel(num_qubits=2)
        def bell(qc):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()

        assert isinstance(bell, Kernel)
        assert bell.name == "bell"
        assert bell.num_qubits == 2

    def test_circuit_property(self):
        @kernel(num_qubits=2)
        def bell(qc):
            qc.h(0)
            qc.cx(0, 1)

        circuit = bell.circuit
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2
        assert len(circuit) == 2

    def test_call_with_existing_circuit(self):
        @kernel(num_qubits=3)
        def add_h(qc):
            qc.h(0)

        qc = QuantumCircuit(3)
        qc.x(1)
        result = add_h(qc)
        assert result is qc
        assert len(qc) == 2

    def test_call_without_circuit(self):
        @kernel(num_qubits=1)
        def single(qc):
            qc.x(0)

        result = single()
        assert isinstance(result, QuantumCircuit)
        assert len(result) == 1

    def test_repr(self):
        @kernel(num_qubits=4)
        def ghz(qc):
            pass

        assert "ghz" in repr(ghz)
        assert "4" in repr(ghz)
