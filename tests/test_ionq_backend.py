"""Tests for hlquantum.backends.ionq_backend."""

from unittest.mock import MagicMock, patch

import pytest

from hlquantum.circuit import QuantumCircuit


class TestIonQBackendInit:
    """Test IonQBackend construction and properties (no network calls)."""

    def test_default_name(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        backend = IonQBackend()
        assert backend.name == "ionq (ionq_simulator)"

    def test_custom_backend_name(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        backend = IonQBackend(backend_name="ionq_qpu")
        assert "ionq_qpu" in backend.name

    def test_user_supplied_backend_name(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        mock_backend = MagicMock()
        mock_backend.name = "custom_ionq"
        backend = IonQBackend(backend=mock_backend)
        assert "custom_ionq" in backend.name

    def test_no_gpu_support(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        backend = IonQBackend()
        assert backend.supports_gpu is False

    def test_repr(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        backend = IonQBackend()
        r = repr(backend)
        assert "ionq" in r
        assert "[GPU]" not in r


class TestIonQTranslation:
    """Verify gate translation to a Qiskit QuantumCircuit."""

    def test_translate_bell_state(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()

        # We need qiskit imported for translation
        try:
            import qiskit
        except ImportError:
            pytest.skip("qiskit not installed")

        qk_circuit = IonQBackend._translate(qc, qiskit)
        # Should have h, cx, and 2 measures = 4 instructions
        assert qk_circuit.num_qubits == 2

    def test_translate_rotation_gates(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        qc = QuantumCircuit(1)
        qc.rx(0, 1.57).ry(0, 0.5).rz(0, 3.14)

        try:
            import qiskit
        except ImportError:
            pytest.skip("qiskit not installed")

        qk_circuit = IonQBackend._translate(qc, qiskit)
        assert qk_circuit.num_qubits == 1

    def test_translate_unsupported_gate_raises(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        qc = QuantumCircuit(1)
        # Manually add an unsupported gate
        from hlquantum.circuit import Gate
        qc.gates.append(Gate(name="unsupported_gate", targets=(0,)))

        try:
            import qiskit
        except ImportError:
            pytest.skip("qiskit not installed")

        with pytest.raises(ValueError, match="does not support gate"):
            IonQBackend._translate(qc, qiskit)

    def test_translate_all_single_qubit_gates(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        qc = QuantumCircuit(1)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)

        try:
            import qiskit
        except ImportError:
            pytest.skip("qiskit not installed")

        qk_circuit = IonQBackend._translate(qc, qiskit)
        assert qk_circuit.num_qubits == 1

    def test_translate_multi_qubit_gates(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        qc = QuantumCircuit(3)
        qc.cx(0, 1).cz(1, 2).swap(0, 2).ccx(0, 1, 2)

        try:
            import qiskit
        except ImportError:
            pytest.skip("qiskit not installed")

        qk_circuit = IonQBackend._translate(qc, qiskit)
        assert qk_circuit.num_qubits == 3


class TestIonQRequireHelpers:
    """Test the lazy-import helpers raise clear errors when deps are missing."""

    def test_require_qiskit_missing(self):
        from hlquantum.backends.ionq_backend import _require_qiskit

        with patch.dict("sys.modules", {"qiskit": None}):
            with pytest.raises(ImportError, match="Qiskit is required"):
                _require_qiskit()

    def test_require_ionq_provider_missing(self):
        from hlquantum.backends.ionq_backend import _require_ionq_provider

        with patch.dict("sys.modules", {"qiskit_ionq": None}):
            with pytest.raises(ImportError, match="qiskit-ionq is required"):
                _require_ionq_provider()


class TestIonQBackendRun:
    """Test the run() path with mocked IonQ provider and backend."""

    def test_run_with_mocked_backend(self):
        from hlquantum.backends.ionq_backend import IonQBackend

        # Create a mock Qiskit-compatible backend
        mock_result = MagicMock()
        mock_result.get_counts.return_value = {"00": 520, "11": 480}

        mock_job = MagicMock()
        mock_job.result.return_value = mock_result

        mock_backend = MagicMock()
        mock_backend.name = "mock_ionq"
        mock_backend.run.return_value = mock_job

        backend = IonQBackend(backend=mock_backend, transpile=False)

        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()

        try:
            import qiskit  # noqa: F401
        except ImportError:
            pytest.skip("qiskit not installed")

        result = backend.run(qc, shots=1000)

        assert result.shots == 1000
        assert result.backend_name == "ionq (mock_ionq)"
        assert "00" in result.counts
        assert "11" in result.counts
        assert result.counts["00"] + result.counts["11"] == 1000
