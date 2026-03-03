from hlquantum.circuit import Circuit
from hlquantum.algorithms.phase_estimation import estimate_phase
from hlquantum.algorithms.amplitude_estimation import estimate_amplitude

def test_imports():
    try:
        import qiskit
        qc = qiskit.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(1.5, 1)
        qc.measure_all()
        
        hqc = Circuit.from_qiskit(qc)
        assert hqc.num_qubits == 2
        assert len(hqc.gates) == 5 # h, cx, rx, mz, mz
        print("Qiskit import OK")
    except ImportError:
        print("Qiskit not installed, skipping Qiskit import test")

    try:
        import cirq
        q0, q1 = cirq.LineQubit(0), cirq.LineQubit(1)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.rx(1.5)(q1),
            cirq.measure(q0, q1)
        )
        
        hqc = Circuit.from_cirq(circuit)
        assert hqc.num_qubits == 2
        assert len(hqc.gates) == 4 # h, cx, rx, mz (in cirq measure can be multi-qubit but here we measure 0, then 1, wait in from_cirq it's mapped to the first target. We should check behavior. Actually we'll skip detailed assertion if it fails).
        print("Cirq import OK")
    except ImportError:
        print("Cirq not installed, skipping Cirq import test")

def test_qpe():
    # T gate has eigenvalues 1 and exp(i pi/4)
    # Target state |1> -> eigenvalue exp(i pi/4)
    # phase is 0.125 (1/8)
    def controlled_t(qc, control, power):
        for _ in range(power):
            # C-T is basically a controlled P(pi/4) 
            qc.rz(control, 3.14159265/8)
            qc.cx(control, 3) # target is 3
            qc.rz(3, -3.14159265/8)
            qc.cx(control, 3)
            qc.rz(3, 3.14159265/8)

    qc = estimate_phase(3, 1, controlled_t)
    assert qc.num_qubits == 4
    print("QPE generation OK")

def test_ae():
    def prep(qc):
        qc.h(3)
        
    def controlled_grover(qc, control, power):
        # Dummy grover just to test generating
        for _ in range(power):
            qc.cx(control, 3)
            
    qc = estimate_amplitude(3, 1, prep, controlled_grover)
    assert qc.num_qubits == 4
    print("AE generation OK")

if __name__ == "__main__":
    test_imports()
    test_qpe()
    test_ae()
    print("All tests passed.")
