"""
hlquantum.backends.ionq_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IonQ backend for HLQuantum.

Supports execution on IonQ simulators and trapped-ion QPUs via the
``qiskit-ionq`` provider.  When no explicit backend object is supplied
the provider connects to the IonQ cloud simulator.

Prerequisites
~~~~~~~~~~~~~
* ``pip install qiskit qiskit-ionq``
* An IonQ API key, supplied either as the *api_key* constructor argument
  or through the ``IONQ_API_KEY`` / ``QISKIT_IONQ_API_TOKEN`` environment
  variable.

See https://docs.ionq.com and https://github.com/qiskit-partners/qiskit-ionq
for full documentation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from hlquantum.backends.base import Backend
from hlquantum.circuit import Gate, QuantumCircuit
from hlquantum.result import ExecutionResult

logger = logging.getLogger(__name__)

# Gate set natively supported by IonQ hardware (GPI / GPI2 / MS).
# We translate from the HLQuantum abstract gate set to the common Qiskit
# gate names and let the IonQ transpiler handle the rest.
_SUPPORTED_GATES = frozenset(
    {"h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap", "ccx", "mz"}
)


def _require_qiskit():
    """Import and return the ``qiskit`` package, raising a friendly error if missing."""
    try:
        import qiskit
        return qiskit
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required for the IonQBackend but is not installed.\n"
            "Install it with:  pip install qiskit\n"
            "See https://qiskit.org for details."
        ) from exc


def _require_ionq_provider(api_key: Optional[str] = None):
    """Import and return an ``IonQProvider`` instance.

    The *api_key* parameter takes precedence over the ``IONQ_API_KEY`` /
    ``QISKIT_IONQ_API_TOKEN`` environment variables.
    """
    try:
        from qiskit_ionq import IonQProvider  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "qiskit-ionq is required for the IonQBackend but is not installed.\n"
            "Install it with:  pip install qiskit-ionq\n"
            "See https://github.com/qiskit-partners/qiskit-ionq for details."
        ) from exc

    token = api_key or os.environ.get("IONQ_API_KEY") or os.environ.get("QISKIT_IONQ_API_TOKEN")
    if token:
        return IonQProvider(token=token)
    return IonQProvider()


class IonQBackend(Backend):
    """Execute HLQuantum circuits on IonQ hardware or simulators.

    Parameters
    ----------
    backend_name : str, optional
        Name of the IonQ backend to use (default ``"ionq_simulator"``).
        Common choices:

        * ``"ionq_simulator"`` – cloud-based ideal simulator
        * ``"ionq_qpu"``       – IonQ Aria / Forte trapped-ion QPU
    api_key : str, optional
        IonQ API key.  Falls back to the ``IONQ_API_KEY`` or
        ``QISKIT_IONQ_API_TOKEN`` environment variables when omitted.
    backend : object, optional
        A pre-configured Qiskit ``Backend`` object.  When provided,
        *backend_name* and *api_key* are ignored.
    transpile : bool, optional
        Whether to transpile the circuit before execution (default *True*).
    optimization_level : int, optional
        Qiskit transpiler optimisation level (0–3, default 1).
    """

    def __init__(
        self,
        backend_name: str = "ionq_simulator",
        api_key: Optional[str] = None,
        backend: Optional[Any] = None,
        transpile: bool = True,
        optimization_level: int = 1,
    ) -> None:
        self._backend_name = backend_name
        self._api_key = api_key
        self._user_backend = backend
        self._transpile = transpile
        self._optimization_level = optimization_level

    # ----- Backend interface ------------------------------------------------

    @property
    def name(self) -> str:
        if self._user_backend is not None:
            label = getattr(self._user_backend, "name", str(self._user_backend))
            return f"ionq ({label})"
        return f"ionq ({self._backend_name})"

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        include_statevector: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Translate, transpile and execute *circuit* on the IonQ backend.

        Parameters
        ----------
        circuit : QuantumCircuit
            The HLQuantum circuit to execute.
        shots : int, optional
            Number of measurement shots (default 1000).
        include_statevector : bool, optional
            If *True*, attempt to retrieve the state vector.  Note that
            state-vector retrieval is only supported on simulators.
        **kwargs
            Forwarded to the underlying Qiskit ``backend.run()`` call.
        """
        qiskit = _require_qiskit()
        qk_circuit = self._translate(circuit, qiskit)

        # Resolve backend
        qk_backend = self._resolve_backend()

        # Optional: statevector save instruction (simulator only)
        if include_statevector:
            try:
                qk_circuit.save_statevector()
            except Exception:
                logger.warning(
                    "State-vector save is not supported on this IonQ backend; "
                    "the result will not contain a state vector."
                )

        # Transpile
        if self._transpile:
            from qiskit import transpile as qk_transpile  # type: ignore[import-untyped]
            qk_circuit = qk_transpile(
                qk_circuit,
                backend=qk_backend,
                optimization_level=self._optimization_level,
            )

        logger.info(
            "Running %d-qubit circuit (%d gates) for %d shots on %s (SV: %s)",
            circuit.num_qubits,
            len(circuit),
            shots,
            self.name,
            include_statevector,
        )

        actual_shots = max(shots, 1)
        job = qk_backend.run(qk_circuit, shots=actual_shots, **kwargs)
        raw_result = job.result()

        # --- Counts --------------------------------------------------------
        counts: Dict[str, int] = {}
        if shots > 0:
            try:
                raw_counts = raw_result.get_counts()
                if isinstance(raw_counts, list):
                    raw_counts = raw_counts[0]
                for bitstring, count in raw_counts.items():
                    counts[bitstring.replace(" ", "")] = count
            except Exception:
                logger.warning("Could not extract measurement counts from the IonQ result.")

        # --- Statevector ---------------------------------------------------
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
            metadata={"ionq_backend": self._backend_name},
        )

    # ----- Helpers ----------------------------------------------------------

    def _resolve_backend(self) -> Any:
        """Return the Qiskit ``Backend`` object to use for execution."""
        if self._user_backend is not None:
            return self._user_backend
        provider = _require_ionq_provider(self._api_key)
        return provider.get_backend(self._backend_name)

    @staticmethod
    def _translate(circuit: QuantumCircuit, qiskit: Any) -> Any:
        """Convert an HLQuantum ``QuantumCircuit`` into a Qiskit ``QuantumCircuit``."""
        n = circuit.num_qubits
        qk = qiskit.QuantumCircuit(n, n)
        measure_targets: List[int] = []

        for gate in circuit.gates:
            name = gate.name
            t0 = gate.targets[0]
            if name == "h":       qk.h(t0)
            elif name == "x":     qk.x(t0)
            elif name == "y":     qk.y(t0)
            elif name == "z":     qk.z(t0)
            elif name == "s":     qk.s(t0)
            elif name == "t":     qk.t(t0)
            elif name == "rx":    qk.rx(gate.params[0], t0)
            elif name == "ry":    qk.ry(gate.params[0], t0)
            elif name == "rz":    qk.rz(gate.params[0], t0)
            elif name == "cx":    qk.cx(gate.controls[0], t0)
            elif name == "cz":    qk.cz(gate.controls[0], t0)
            elif name == "swap":  qk.swap(gate.targets[0], gate.targets[1])
            elif name == "ccx":   qk.ccx(gate.controls[0], gate.controls[1], t0)
            elif name == "mz":    measure_targets.append(t0)
            else:
                raise ValueError(f"IonQBackend does not support gate: {name!r}")

        for t in measure_targets:
            qk.measure(t, t)
        return qk
