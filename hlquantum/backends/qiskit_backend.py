"""
hlquantum.backends.qiskit_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IBM Qiskit backend for HLQuantum.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hlquantum.backends.base import Backend
from hlquantum.circuit import Gate, QuantumCircuit
from hlquantum.gpu import GPUConfig
from hlquantum.result import ExecutionResult

logger = logging.getLogger(__name__)


def _require_qiskit():
    try:
        import qiskit
        return qiskit
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required for the QiskitBackend but is not installed.\n"
            "Install it with:  pip install qiskit qiskit-aer\n"
            "See https://qiskit.org for details."
        ) from exc


def _require_aer():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator
    except ImportError as exc:
        raise ImportError(
            "qiskit-aer is required for local simulation.\n"
            "Install it with:  pip install qiskit-aer"
        ) from exc


class QiskitBackend(Backend):
    """Execute HLQuantum circuits using IBM Qiskit."""

    def __init__(
        self,
        backend: Optional[Any] = None,
        transpile: bool = True,
        optimization_level: int = 1,
        gpu_config: Optional[GPUConfig] = None,
    ) -> None:
        self._user_backend = backend
        self._transpile = transpile
        self._optimization_level = optimization_level
        self._gpu_config = gpu_config or GPUConfig()

    @property
    def supports_gpu(self) -> bool:
        return True

    @property
    def name(self) -> str:
        if self._user_backend is not None:
            label = getattr(self._user_backend, "name", str(self._user_backend))
            return f"qiskit ({label})"
        suffix = " GPU" if self._gpu_config.enabled else ""
        return f"qiskit (aer_simulator{suffix})"

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        include_statevector: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Transpile & run *circuit* on the configured Qiskit backend."""
        qiskit = _require_qiskit()

        qk_circuit = self._translate(circuit, qiskit)

        # Resolve backend
        if self._user_backend is not None:
            qk_backend = self._user_backend
        else:
            qk_backend = self._build_aer_backend()

        # If statevector requested, add the save instruction (Aer specific)
        if include_statevector:
            # Check if backend supports save_statevector (effectively Aer)
            if hasattr(qk_circuit, "save_statevector"):
                qk_circuit.save_statevector()
            else:
                try:
                    import qiskit_aer.library as aer_lib
                    qk_circuit.append(aer_lib.SaveStatevector(circuit.num_qubits), qk_circuit.qubits)
                except (ImportError, Exception):
                    logger.warning("Backend may not support state vector retrieval.")

        # Apply CUDA_VISIBLE_DEVICES if configured
        if self._gpu_config.enabled:
            self._gpu_config.apply_env()

        # Optionally transpile
        if self._transpile:
            from qiskit import transpile as qk_transpile  # type: ignore[import-untyped]
            qk_circuit = qk_transpile(
                qk_circuit,
                backend=qk_backend,
                optimization_level=self._optimization_level,
            )

        logger.info(
            "Running %d-qubit circuit (%d gates) for %d shots on %s (GPU: %s, SV: %s)",
            circuit.num_qubits,
            len(circuit),
            shots,
            self.name,
            "enabled" if self._gpu_config.enabled else "disabled",
            include_statevector,
        )

        # Qiskit requires shots > 0 for run() usually, unless we skip sampling
        actual_shots = shots if shots > 0 else 1
        job = qk_backend.run(qk_circuit, shots=actual_shots, **kwargs)
        raw_result = job.result()

        # Counts
        counts: Dict[str, int] = {}
        if shots > 0:
            try:
                raw_counts = raw_result.get_counts()
                if isinstance(raw_counts, list): # handle multiple circuits if ever needed
                    raw_counts = raw_counts[0]
                for bitstring, count in raw_counts.items():
                    counts[bitstring.replace(" ", "")] = count
            except Exception:
                pass

        # Statevector
        state_vector = None
        if include_statevector:
            try:
                state_vector = raw_result.get_statevector()
            except Exception:
                logger.warning("Failed to retrieve state vector from result.")

        return ExecutionResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            raw=raw_result,
            state_vector=state_vector,
            metadata={"gpu_config": repr(self._gpu_config)},
        )

    def _build_aer_backend(self) -> Any:
        AerSimulator = _require_aer()
        aer_kwargs: Dict[str, Any] = {}
        if self._gpu_config.enabled:
            aer_kwargs["device"] = "GPU"
            if self._gpu_config.custatevec:
                aer_kwargs["cuStateVec_enable"] = True
            logger.info("Configuring AerSimulator with GPU (cuStateVec=%s)", self._gpu_config.custatevec)
        return AerSimulator(**aer_kwargs)

    @staticmethod
    def _translate(circuit: QuantumCircuit, qiskit: Any) -> Any:
        n = circuit.num_qubits
        qk = qiskit.QuantumCircuit(n, n)
        measure_targets = []

        for gate in circuit.gates:
            name = gate.name
            if name == "h":       qk.h(gate.targets[0])
            elif name == "x":     qk.x(gate.targets[0])
            elif name == "y":     qk.y(gate.targets[0])
            elif name == "z":     qk.z(gate.targets[0])
            elif name == "s":     qk.s(gate.targets[0])
            elif name == "t":     qk.t(gate.targets[0])
            elif name == "rx":    qk.rx(gate.params[0], gate.targets[0])
            elif name == "ry":    qk.ry(gate.params[0], gate.targets[0])
            elif name == "rz":    qk.rz(gate.params[0], gate.targets[0])
            elif name == "cx":    qk.cx(gate.controls[0], gate.targets[0])
            elif name == "cz":    qk.cz(gate.controls[0], gate.targets[0])
            elif name == "swap":  qk.swap(gate.targets[0], gate.targets[1])
            elif name == "ccx":   qk.ccx(gate.controls[0], gate.controls[1], gate.targets[0])
            elif name == "mz":    measure_targets.append(gate.targets[0])
            else: raise ValueError(f"QiskitBackend does not support gate: {name!r}")

        for t in measure_targets:
            qk.measure(t, t)
        return qk
