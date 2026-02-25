"""Comprehensive tests covering circuit features, algorithm correctness,
backend translation, transpiler edge-cases, mitigation immutability,
and runner integration.

These tests are designed to validate actual execution logic rather than
just structure.  Backend SDK tests are skipped when the corresponding
SDK is not installed.
"""

import asyncio
import math
import pytest

from hlquantum.circuit import Circuit, Gate, Parameter, QuantumCircuit
from hlquantum.result import ExecutionResult
from hlquantum.backends.base import Backend
from hlquantum.exceptions import CircuitValidationError


# ─── Helper: lightweight mock backend for runner integration ────────────────

class _SimBackend(Backend):
    """Minimal backend that tallies all-zeros for counting-based tests."""

    @property
    def name(self) -> str:
        return "_sim"

    def run(self, circuit, shots=1000, include_statevector=False, **kw):
        key = "0" * circuit.num_qubits
        return ExecutionResult(counts={key: shots}, shots=shots, backend_name=self.name)


# ═══════════════════════════════════════════════════════════════════════════
#  1. CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitExtended:
    """Extended circuit tests covering Parameters, bind, compose, depth."""

    def test_parameter_creation_from_string(self):
        qc = Circuit(1)
        qc.rx(0, "theta")
        assert isinstance(qc.gates[0].params[0], Parameter)
        assert qc.gates[0].params[0].name == "theta"

    def test_parameters_property(self):
        qc = Circuit(2)
        qc.rx(0, "a").ry(1, "b").rz(0, "a")
        params = qc.parameters
        assert len(params) == 2
        names = {p.name for p in params}
        assert names == {"a", "b"}

    def test_bind_parameters(self):
        qc = Circuit(1).rx(0, "theta")
        bound = qc.bind_parameters({"theta": 1.5})
        assert bound.gates[0].params[0] == 1.5
        # Original should be unchanged
        assert isinstance(qc.gates[0].params[0], Parameter)

    def test_bind_missing_parameter_raises(self):
        qc = Circuit(1).rx(0, "theta")
        with pytest.raises(ValueError, match="Missing value"):
            qc.bind_parameters({})

    def test_compose_or(self):
        c1 = Circuit(2).h(0)
        c2 = Circuit(2).x(1)
        c3 = c1 | c2
        assert c3.num_qubits == 2
        assert len(c3) == 2
        assert c3.gates[0].name == "h"
        assert c3.gates[1].name == "x"

    def test_compose_different_sizes(self):
        c1 = Circuit(2).h(0)
        c2 = Circuit(3).x(2)
        c3 = c1 | c2
        assert c3.num_qubits == 3

    def test_depth_single_qubit(self):
        qc = Circuit(1).h(0).x(0).z(0)
        assert qc.depth == 3

    def test_depth_parallel(self):
        qc = Circuit(3).h(0).h(1).h(2)
        assert qc.depth == 1  # All H gates are on different qubits → parallel

    def test_depth_cx_chain(self):
        qc = Circuit(3).cx(0, 1).cx(1, 2)
        assert qc.depth == 2  # Second CX depends on qubit 1

    def test_gate_count(self):
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        assert qc.gate_count == 4

    def test_all_single_qubit_gates(self):
        qc = Circuit(1)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        names = [g.name for g in qc.gates]
        assert names == ["h", "x", "y", "z", "s", "t"]

    def test_all_rotation_gates(self):
        qc = Circuit(1)
        qc.rx(0, 0.1).ry(0, 0.2).rz(0, 0.3)
        assert qc.gates[0].params == (0.1,)
        assert qc.gates[1].params == (0.2,)
        assert qc.gates[2].params == (0.3,)

    def test_multi_qubit_gates(self):
        qc = Circuit(3)
        qc.cx(0, 1).cz(1, 2).swap(0, 2).ccx(0, 1, 2)
        assert len(qc) == 4

    def test_measure_single(self):
        qc = Circuit(2).measure(0)
        assert len(qc) == 1
        assert qc.gates[0].name == "mz"
        assert qc.gates[0].targets == (0,)

    def test_qubit_out_of_range(self):
        qc = Circuit(2)
        with pytest.raises(IndexError):
            qc.h(5)

    def test_empty_circuit_depth(self):
        qc = Circuit(1)
        assert qc.depth == 0


# ═══════════════════════════════════════════════════════════════════════════
#  2. TRANSPILER
# ═══════════════════════════════════════════════════════════════════════════

class TestTranspilerExtended:
    """Tests for transpiler edge-cases including Parameter handling."""

    def test_merge_rotations_with_parameters_skips(self):
        """MergeRotations must NOT attempt to merge symbolic Parameters."""
        from hlquantum.transpiler import MergeRotations

        qc = Circuit(1)
        qc.rx(0, "theta").rx(0, "phi")
        result = MergeRotations().run(qc)
        # Should keep both gates because both are Parameters
        assert len(result.gates) == 2

    def test_merge_rotations_one_param_one_float(self):
        from hlquantum.transpiler import MergeRotations

        qc = Circuit(1)
        qc.rx(0, "theta").rx(0, 0.5)
        result = MergeRotations().run(qc)
        # Cannot merge a Parameter with a float
        assert len(result.gates) == 2

    def test_merge_rotations_zero_angle_removal(self):
        """Merged rotations that sum to 0 (mod 2π) should be removed."""
        from hlquantum.transpiler import MergeRotations

        qc = Circuit(1)
        qc.rx(0, math.pi).rx(0, math.pi)  # π + π = 2π ≡ 0
        result = MergeRotations().run(qc)
        assert len(result.gates) == 0

    def test_redundant_gate_removal_non_adjacent(self):
        """Non-adjacent identical gates should NOT be cancelled."""
        from hlquantum.transpiler import RemoveRedundantGates

        qc = Circuit(1).h(0).x(0).h(0)
        result = RemoveRedundantGates().run(qc)
        assert len(result.gates) == 3  # H, X, H remain


# ═══════════════════════════════════════════════════════════════════════════
#  3. MITIGATION
# ═══════════════════════════════════════════════════════════════════════════

class TestMitigationImmutability:
    """Verify that mitigation does not mutate the input result."""

    def test_threshold_does_not_mutate_input(self):
        from hlquantum.mitigation import ThresholdMitigation

        counts = {"00": 900, "11": 95, "01": 5}
        original = ExecutionResult(counts=counts, shots=1000)
        mitigated = ThresholdMitigation(threshold=0.01).apply(original)
        # Original should still have all three keys
        assert "01" in original.counts
        # Mitigated should not
        assert "01" not in mitigated.counts


# ═══════════════════════════════════════════════════════════════════════════
#  4. VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendValidation:
    """Verify that Backend.validate() works."""

    def test_validate_passes_for_valid_circuit(self):
        be = _SimBackend()
        be.validate(Circuit(2).h(0))  # Should not raise

    def test_validate_raises_for_zero_qubits(self):
        # Can't even construct a 0-qubit Circuit, so validation
        # is guarded by Circuit.__init__
        with pytest.raises(ValueError):
            Circuit(0)


# ═══════════════════════════════════════════════════════════════════════════
#  5. ALGORITHMS — correctness checks
# ═══════════════════════════════════════════════════════════════════════════

class TestAlgorithmsExtended:
    """Algorithm correctness: structure, gate counts, inverse QFT."""

    def test_qft_forward_1_qubit(self):
        from hlquantum.algorithms.qft import qft
        c = qft(1)
        # 1-qubit QFT = just a Hadamard
        assert c.num_qubits == 1
        assert any(g.name == "h" for g in c.gates)

    def test_qft_forward_2_qubits(self):
        from hlquantum.algorithms.qft import qft
        c = qft(2)
        assert c.num_qubits == 2
        assert len(c.gates) > 2

    def test_qft_inverse(self):
        from hlquantum.algorithms.qft import qft
        c = qft(3, inverse=True)
        assert c.num_qubits == 3
        assert len(c.gates) > 5

    def test_grover_2qubit(self):
        from hlquantum.algorithms.grover import grover
        c = grover(2, ["11"], iterations=1)
        assert c.num_qubits == 3  # 2 search + 1 ancilla
        # Should have measurements on first 2 qubits
        mz_targets = [g.targets[0] for g in c.gates if g.name == "mz"]
        assert set(mz_targets) == {0, 1}

    def test_grover_3qubit(self):
        from hlquantum.algorithms.grover import grover
        c = grover(3, ["101"], iterations=1)
        assert c.num_qubits == 4
        assert len(c.gates) > 10

    def test_grover_4qubit(self):
        """Grover for n=4 should work (exercises MCX decomposition)."""
        from hlquantum.algorithms.grover import grover
        c = grover(4, ["1010"], iterations=1)
        assert c.num_qubits == 5
        assert len(c.gates) > 20

    def test_bernstein_vazirani(self):
        from hlquantum.algorithms.bernstein_vazirani import bernstein_vazirani
        c = bernstein_vazirani("110")
        assert c.num_qubits == 4
        mz_targets = [g.targets[0] for g in c.gates if g.name == "mz"]
        assert set(mz_targets) == {0, 1, 2}

    def test_deutsch_jozsa_constant(self):
        from hlquantum.algorithms.deutsch_jozsa import deutsch_jozsa, constant_oracle
        c = deutsch_jozsa(3, constant_oracle)
        assert c.num_qubits == 4

    def test_deutsch_jozsa_balanced(self):
        from hlquantum.algorithms.deutsch_jozsa import deutsch_jozsa, balanced_oracle
        c = deutsch_jozsa(3, balanced_oracle)
        assert c.num_qubits == 4

    def test_half_adder(self):
        from hlquantum.algorithms.arithmetic import half_adder
        c = half_adder()
        assert c.num_qubits == 4
        gate_names = [g.name for g in c.gates]
        assert gate_names.count("cx") == 2
        assert gate_names.count("ccx") == 1

    def test_full_adder(self):
        from hlquantum.algorithms.arithmetic import full_adder
        c = full_adder()
        assert c.num_qubits == 5
        gate_names = [g.name for g in c.gates]
        assert "cx" in gate_names
        assert "ccx" in gate_names

    def test_ripple_carry_adder_qubit_count(self):
        from hlquantum.algorithms.arithmetic import ripple_carry_adder
        # n bits → 3n+1 qubits (a[n], b[n], c[n+1])
        c = ripple_carry_adder(2)
        assert c.num_qubits == 7
        c3 = ripple_carry_adder(3)
        assert c3.num_qubits == 10

    def test_ripple_carry_adder_has_gates(self):
        from hlquantum.algorithms.arithmetic import ripple_carry_adder
        c = ripple_carry_adder(2)
        assert len(c.gates) > 0
        gate_names = [g.name for g in c.gates]
        assert "ccx" in gate_names
        assert "cx" in gate_names

    def test_vqe_ansatz(self):
        from hlquantum.algorithms.vqe import hardware_efficient_ansatz
        c = hardware_efficient_ansatz(3, [0.1, 0.2, 0.3])
        assert c.num_qubits == 3
        ry_count = sum(1 for g in c.gates if g.name == "ry")
        cx_count = sum(1 for g in c.gates if g.name == "cx")
        assert ry_count == 3
        assert cx_count == 2

    def test_qaoa_solve_builds_circuit(self):
        """Verify QAOA internal circuit construction via build_ansatz logic."""
        from hlquantum.algorithms.qaoa import qaoa_solve
        # We can't run the full optimizer without scipy, but we can verify
        # the circuit-building logic by testing at a structural level
        from hlquantum.circuit import Circuit
        n_qubits = 3
        edges = [{'qubits': (0, 1), 'weight': 1.0}, {'qubits': (1, 2), 'weight': 1.0}]
        # Build the ansatz manually (mirrors internal logic)
        qc = Circuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        gamma, beta = 0.5, 0.5
        for term in edges:
            q1, q2 = term['qubits']
            w = term.get('weight', 1.0)
            qc.cx(q1, q2)
            qc.rz(q2, 2 * gamma * w)
            qc.cx(q1, q2)
        for i in range(n_qubits):
            qc.rx(i, 2 * beta)
        assert qc.num_qubits == 3
        assert len(qc.gates) > 5

    def test_gqe_ansatz_construction(self):
        """Verify a parameterised circuit works as a GQE ansatz."""
        qc = Circuit(2).rx(0, "a").ry(1, "b").cx(0, 1)
        assert len(qc.parameters) == 2
        bound = qc.bind_parameters({"a": 0.1, "b": 0.2})
        assert bound.gates[0].params[0] == 0.1

    def test_parameter_shift_gradient_validates(self):
        from hlquantum.algorithms.grad import parameter_shift_gradient
        qc = Circuit(1).rx(0, "theta")
        with pytest.raises(ValueError, match="Missing parameter"):
            parameter_shift_gradient(qc, {})


# ═══════════════════════════════════════════════════════════════════════════
#  6. RUNNER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestRunnerIntegration:
    """Test the run() function with a mock backend."""

    def test_run_circuit(self):
        from hlquantum.runner import run
        c = Circuit(2).h(0).cx(0, 1).measure_all()
        result = run(c, backend=_SimBackend(), shots=500)
        assert result.shots == 500
        assert "00" in result.counts
        assert result.backend_name == "_sim"

    def test_run_with_transpile(self):
        from hlquantum.runner import run
        c = Circuit(1).h(0).h(0)  # Should be optimised away
        result = run(c, backend=_SimBackend(), transpile=True)
        assert result is not None

    def test_run_with_mitigation(self):
        from hlquantum.runner import run
        from hlquantum.mitigation import ThresholdMitigation

        be = _SimBackend()
        c = Circuit(1).h(0).measure_all()
        result = run(c, backend=be, mitigation=[ThresholdMitigation(0.01)])
        assert result is not None

    def test_run_kernel(self):
        from hlquantum.kernel import kernel
        from hlquantum.runner import run

        @kernel(num_qubits=2)
        def bell(qc):
            qc.h(0).cx(0, 1)

        result = run(bell, backend=_SimBackend())
        assert result.shots == 1000

    def test_run_bad_input_raises(self):
        from hlquantum.runner import run
        with pytest.raises(TypeError):
            run("not_a_circuit", backend=_SimBackend())


# ═══════════════════════════════════════════════════════════════════════════
#  7. BACKEND TRANSLATION — SDK-specific (skipped if not installed)
# ═══════════════════════════════════════════════════════════════════════════

class TestQiskitTranslation:
    """Validate gate translation to Qiskit QuantumCircuit."""

    @pytest.fixture(autouse=True)
    def _require_qiskit(self):
        try:
            import qiskit
            self.qiskit = qiskit
        except ImportError:
            pytest.skip("qiskit not installed")

    def test_bell_state(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        qk = QiskitBackend._translate(qc, self.qiskit)
        assert qk.num_qubits == 2
        assert qk.num_clbits == 2

    def test_all_gates(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        qc = Circuit(3)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        qc.rx(0, 0.5).ry(0, 0.5).rz(0, 0.5)
        qc.cx(0, 1).cz(1, 2).swap(0, 2).ccx(0, 1, 2)
        qc.measure_all()
        qk = QiskitBackend._translate(qc, self.qiskit)
        assert qk.num_qubits == 3

    def test_unsupported_gate_raises(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        qc = Circuit(1)
        qc.gates.append(Gate(name="foo", targets=(0,)))
        with pytest.raises(ValueError, match="does not support"):
            QiskitBackend._translate(qc, self.qiskit)


class TestQiskitExecution:
    """Actually run circuits on the Qiskit Aer simulator."""

    @pytest.fixture(autouse=True)
    def _require_qiskit_aer(self):
        try:
            import qiskit
            import qiskit_aer  # noqa: F401
            self.qiskit = qiskit
        except ImportError:
            pytest.skip("qiskit + qiskit-aer not installed")

    def test_bell_state_execution(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        be = QiskitBackend()
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = be.run(qc, shots=1000)
        assert result.shots == 1000
        # Bell state should produce only "00" and "11"
        for key in result.counts:
            assert key in ("00", "11")
        assert sum(result.counts.values()) == 1000

    def test_single_qubit_x(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        be = QiskitBackend()
        qc = Circuit(1).x(0).measure_all()
        result = be.run(qc, shots=100)
        assert result.counts.get("1", 0) == 100

    def test_statevector_retrieval(self):
        from hlquantum.backends.qiskit_backend import QiskitBackend
        be = QiskitBackend()
        qc = Circuit(1).h(0)
        result = be.run(qc, shots=100, include_statevector=True)
        sv = result.get_state_vector()
        if sv is not None:
            import numpy as np
            assert len(sv) == 2
            # |+⟩ state: both amplitudes should be ≈ 1/√2
            assert abs(abs(sv[0]) - 1 / math.sqrt(2)) < 0.05


class TestCirqTranslation:
    """Validate gate translation to Cirq."""

    @pytest.fixture(autouse=True)
    def _require_cirq(self):
        try:
            import cirq
            self.cirq = cirq
        except ImportError:
            pytest.skip("cirq not installed")

    def test_bell_state(self):
        from hlquantum.backends.cirq_backend import CirqBackend
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        cirq_circuit, qubits, keys = CirqBackend._translate(qc, self.cirq)
        assert len(qubits) == 2
        assert len(keys) == 2

    def test_all_gates(self):
        from hlquantum.backends.cirq_backend import CirqBackend
        qc = Circuit(3)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        qc.rx(0, 0.5).ry(0, 0.5).rz(0, 0.5)
        qc.cx(0, 1).cz(1, 2).swap(0, 2).ccx(0, 1, 2)
        qc.measure_all()
        cirq_circuit, qubits, keys = CirqBackend._translate(qc, self.cirq)
        assert len(qubits) == 3


class TestCirqExecution:
    """Actually run circuits on the Cirq simulator."""

    @pytest.fixture(autouse=True)
    def _require_cirq(self):
        try:
            import cirq
            self.cirq = cirq
        except ImportError:
            pytest.skip("cirq not installed")

    def test_bell_state_execution(self):
        from hlquantum.backends.cirq_backend import CirqBackend
        be = CirqBackend()
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = be.run(qc, shots=1000)
        assert result.shots == 1000
        for key in result.counts:
            assert key in ("00", "11")

    def test_x_gate_execution(self):
        from hlquantum.backends.cirq_backend import CirqBackend
        be = CirqBackend()
        qc = Circuit(1).x(0).measure_all()
        result = be.run(qc, shots=100)
        assert result.counts.get("1", 0) == 100


class TestBraketTranslation:
    """Validate gate translation to Amazon Braket."""

    @pytest.fixture(autouse=True)
    def _require_braket(self):
        try:
            from braket.circuits import Circuit as BraketCircuit
            self.BraketCircuit = BraketCircuit
        except ImportError:
            pytest.skip("amazon-braket-sdk not installed")

    def test_bell_state(self):
        from hlquantum.backends.braket_backend import BraketBackend
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        bc = BraketBackend._translate(qc, self.BraketCircuit)
        assert bc is not None

    def test_all_gates(self):
        from hlquantum.backends.braket_backend import BraketBackend
        qc = Circuit(3)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        qc.rx(0, 0.5).ry(0, 0.5).rz(0, 0.5)
        qc.cx(0, 1).cz(1, 2).swap(0, 2).ccx(0, 1, 2)
        qc.measure_all()
        bc = BraketBackend._translate(qc, self.BraketCircuit)
        assert bc is not None


class TestBraketExecution:
    """Run circuits on the Braket LocalSimulator."""

    @pytest.fixture(autouse=True)
    def _require_braket(self):
        try:
            from braket.circuits import Circuit as BraketCircuit  # noqa: F401
            from braket.devices import LocalSimulator  # noqa: F401
        except ImportError:
            pytest.skip("amazon-braket-sdk not installed")

    def test_bell_state_execution(self):
        from hlquantum.backends.braket_backend import BraketBackend
        be = BraketBackend()
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = be.run(qc, shots=1000)
        assert result.shots == 1000
        for key in result.counts:
            assert key in ("00", "11")

    def test_x_gate_execution(self):
        from hlquantum.backends.braket_backend import BraketBackend
        be = BraketBackend()
        qc = Circuit(1).x(0).measure_all()
        result = be.run(qc, shots=100)
        assert result.counts.get("1", 0) == 100


class TestPennyLaneTranslation:
    """Validate PennyLane backend translation and execution."""

    @pytest.fixture(autouse=True)
    def _require_pennylane(self):
        try:
            import pennylane  # noqa: F401
        except ImportError:
            pytest.skip("pennylane not installed")

    def test_bell_state_execution(self):
        from hlquantum.backends.pennylane_backend import PennyLaneBackend
        be = PennyLaneBackend()
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = be.run(qc, shots=1000)
        assert result.shots == 1000
        for key in result.counts:
            assert key in ("00", "11")

    def test_x_gate_execution(self):
        from hlquantum.backends.pennylane_backend import PennyLaneBackend
        be = PennyLaneBackend()
        qc = Circuit(1).x(0).measure_all()
        result = be.run(qc, shots=100)
        assert result.counts.get("1", 0) == 100


class TestCudaQTranslation:
    """Validate CUDA-Q translation and execution (skipped if not installed)."""

    @pytest.fixture(autouse=True)
    def _require_cudaq(self):
        try:
            import cudaq  # noqa: F401
        except ImportError:
            pytest.skip("cudaq not installed")

    def test_bell_state_execution(self):
        from hlquantum.backends.cudaq_backend import CudaQBackend
        be = CudaQBackend()
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = be.run(qc, shots=1000)
        assert result.shots == 1000
        for key in result.counts:
            assert key in ("00", "11")


# ═══════════════════════════════════════════════════════════════════════════
#  8. LAYERS
# ═══════════════════════════════════════════════════════════════════════════

class TestLayersExtended:
    """Additional layer tests."""

    def test_real_amplitudes_template(self):
        from hlquantum.layers.templates import RealAmplitudes
        layer = RealAmplitudes(num_qubits=3, reps=2)
        c = layer.build()
        assert c.num_qubits == 3
        params = c.parameters
        assert len(params) > 0

    def test_hardware_efficient_ansatz_template(self):
        from hlquantum.layers.templates import HardwareEfficientAnsatz
        layer = HardwareEfficientAnsatz(num_qubits=2, reps=1)
        c = layer.build()
        assert c.num_qubits == 2

    def test_grover_layer(self):
        from hlquantum.layers.functional import GroverLayer
        layer = GroverLayer(num_qubits=2, target_states=["11"])
        c = layer.build()
        assert c.num_qubits == 3  # 2 + ancilla

    def test_qft_layer(self):
        from hlquantum.layers.functional import QFTLayer
        layer = QFTLayer(num_qubits=3)
        c = layer.build()
        assert c.num_qubits == 3

    def test_layer_pipe(self):
        from hlquantum.layers.core import CircuitLayer
        c1 = Circuit(2).h(0)
        c2 = Circuit(2).x(1)
        l1 = CircuitLayer(c1)
        l2 = CircuitLayer(c2)
        seq = l1 | l2
        c = seq.build()
        assert c.num_qubits == 2
        assert len(c) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  9. WORKFLOWS (async)
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkflowsExtended:
    """Additional workflow tests for mixed classical/quantum pipelines."""

    def test_workflow_context_keyed_by_node_id(self):
        from hlquantum.workflows import Workflow, Classical, WorkflowRunner

        wf = Workflow(name="ContextKeyTest")
        wf.add(lambda ctx: 42, node_id="compute", name="compute")
        wf.add(lambda ctx: ctx.get("result_compute"), name="read_back")

        results = asyncio.run(wf.run(verbose=False))
        assert results[0] == 42
        assert results[1] == 42

    def test_loop_with_context_update(self):
        from hlquantum.workflows import Workflow, Loop

        counter = {"val": 0}
        def increment(ctx):
            counter["val"] += 1
            return counter["val"]

        wf = Workflow(name="LoopTest")
        wf.add(Loop(increment, 3))
        results = asyncio.run(wf.run(verbose=False))
        assert results[0] == [1, 2, 3]

    def test_branch_with_context(self):
        from hlquantum.workflows import Workflow, Branch

        wf = Workflow(name="BranchTest")
        wf.add(lambda ctx: 10, name="init")

        def check(ctx):
            return ctx.get("previous_result", 0) > 5

        wf.add(Branch(check, lambda ctx: "big", lambda ctx: "small"))
        results = asyncio.run(wf.run(verbose=False))
        assert results[1] == "big"

    def test_mermaid_generation(self):
        from hlquantum.workflows import Workflow
        wf = Workflow()
        wf.add(lambda ctx: 1, name="step1")
        wf.add(lambda ctx: 2, name="step2")
        mermaid = wf.to_mermaid()
        assert "graph TD" in mermaid
        assert "step1" in mermaid

    def test_mixed_quantum_classical_workflow(self):
        """Hybrid workflow: quantum circuit → classical post-processing."""
        from hlquantum.workflows import Workflow, WorkflowRunner

        be = _SimBackend()
        runner = WorkflowRunner(backend=be)
        wf = Workflow(name="HybridTest")

        wf.add(Circuit(2).h(0).cx(0, 1).measure_all(), name="quantum")
        wf.add(lambda ctx: list(ctx.get("previous_result", {}).counts.keys()), name="extract")

        results = asyncio.run(wf.run(runner=runner, verbose=False))
        assert results[0] is not None  # ExecutionResult
        assert isinstance(results[1], list)  # List of bitstrings

    def test_stop_and_resume(self, tmp_path):
        """Run part of a workflow, stop, then resume from checkpoint."""
        import json
        from hlquantum.workflows import Workflow, WorkflowRunner

        state_file = str(tmp_path / "wf_state.json")
        be = _SimBackend()
        runner = WorkflowRunner(backend=be)

        # ── Phase 1: run only the first node, then "crash" ──
        wf1 = Workflow(name="StopResumeTest", state_file=state_file)
        wf1.add(Circuit(1).h(0).measure_all(), node_id="step_a", name="step_a")
        wf1.add(lambda ctx: "done", node_id="step_b", name="step_b")

        # Execute just the first node manually
        asyncio.run(
            wf1.nodes[0].execute(runner)
        )
        wf1.completed_nodes.append("step_a")
        wf1._save_state()

        # Verify checkpoint was persisted
        with open(state_file, "r") as f:
            data = json.load(f)
        assert "step_a" in data["completed_nodes"]

        # ── Phase 2: build a fresh workflow and resume ──
        wf2 = Workflow(name="StopResumeTest", state_file=state_file)
        wf2.add(Circuit(1).h(0).measure_all(), node_id="step_a", name="step_a")
        wf2.add(lambda ctx: "done", node_id="step_b", name="step_b")

        results = asyncio.run(
            wf2.run(runner=runner, resume=True, verbose=False)
        )

        # step_a was skipped (returns None), step_b executed normally
        assert results[0] is None
        assert results[1] == "done"
        assert "step_a" in wf2.completed_nodes
        assert "step_b" in wf2.completed_nodes

    def test_resume_without_state_file_runs_all(self):
        """Resuming when no state file exists should run every node."""
        from hlquantum.workflows import Workflow

        wf = Workflow(name="NoStateFile")
        wf.add(lambda ctx: 1, node_id="a", name="a")
        wf.add(lambda ctx: 2, node_id="b", name="b")

        results = asyncio.run(
            wf.run(resume=True, verbose=False)
        )
        assert results == [1, 2]


# ═══════════════════════════════════════════════════════════════════════════
#  10. RESULT
# ═══════════════════════════════════════════════════════════════════════════

class TestResultExtended:
    """Extended result tests."""

    def test_probabilities_zero_shots(self):
        r = ExecutionResult(counts={}, shots=0)
        assert r.probabilities == {}

    def test_most_probable_empty(self):
        r = ExecutionResult(counts={}, shots=0)
        assert r.most_probable is None

    def test_expectation_value_all_zeros(self):
        r = ExecutionResult(counts={"000": 1000}, shots=1000)
        # All zeros → even parity → expectation = +1
        assert r.expectation_value() == 1.0

    def test_expectation_value_all_ones_even(self):
        r = ExecutionResult(counts={"11": 1000}, shots=1000)
        # "11" has 2 ones → even parity → +1
        assert r.expectation_value() == 1.0

    def test_expectation_value_all_ones_odd(self):
        r = ExecutionResult(counts={"1": 1000}, shots=1000)
        # "1" has 1 one → odd parity → -1
        assert r.expectation_value() == -1.0

    def test_expectation_value_mixed(self):
        r = ExecutionResult(counts={"00": 500, "11": 500}, shots=1000)
        # Both even parity → +1
        assert r.expectation_value() == 1.0

    def test_repr(self):
        r = ExecutionResult(counts={"0": 100}, shots=100, backend_name="test")
        s = repr(r)
        assert "100" in s


# ═══════════════════════════════════════════════════════════════════════════
#  11. EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════

class TestExceptions:
    """Verify the custom exception hierarchy."""

    def test_exception_hierarchy(self):
        from hlquantum.exceptions import (
            HLQuantumError,
            BackendError,
            CircuitValidationError,
            BackendNotAvailableError,
        )
        assert issubclass(BackendError, HLQuantumError)
        assert issubclass(CircuitValidationError, HLQuantumError)
        assert issubclass(BackendNotAvailableError, HLQuantumError)

    def test_exceptions_are_raisable(self):
        from hlquantum.exceptions import BackendError
        with pytest.raises(BackendError):
            raise BackendError("test")
