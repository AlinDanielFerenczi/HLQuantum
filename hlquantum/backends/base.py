"""
hlquantum.backends.base
~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class that every HLQuantum backend must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from hlquantum.circuit import QuantumCircuit
from hlquantum.exceptions import CircuitValidationError
from hlquantum.gpu import GPUConfig
from hlquantum.result import ExecutionResult


class Backend(ABC):
    """Abstract quantum-execution backend."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def supports_gpu(self) -> bool:
        return False

    @property
    def gpu_config(self) -> Optional[GPUConfig]:
        return getattr(self, "_gpu_config", None)

    @abstractmethod
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        include_statevector: bool = False,
        **kwargs,
    ) -> ExecutionResult:
        """Execute *circuit* and return an :class:`ExecutionResult`.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to execute.
        shots : int, optional
            Number of measurement shots (default 1000).
        include_statevector : bool, optional
            If *True*, attempt to include the state vector in the result.
        **kwargs
            Backend-specific options.
        """
        ...

    def validate(self, circuit: QuantumCircuit) -> None:
        """Validate that *circuit* is executable on this backend.

        The default implementation checks that the circuit has at least one
        qubit.  Subclasses may override for backend-specific validation.

        Raises
        ------
        CircuitValidationError
            If the circuit fails validation.
        """
        if circuit.num_qubits < 1:
            raise CircuitValidationError(
                f"Circuit must have at least 1 qubit, got {circuit.num_qubits}"
            )

    def __repr__(self) -> str:
        gpu_tag = " [GPU]" if self.gpu_config and self.gpu_config.enabled else ""
        return f"<Backend: {self.name}{gpu_tag}>"
