"""
hlquantum.runner
~~~~~~~~~~~~~~~~~

Convenience helpers so users can do ``hlquantum.run(circuit)`` without
manually instantiating a backend.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from hlquantum.backends.base import Backend
from hlquantum.circuit import QuantumCircuit
from hlquantum.kernel import Kernel
from hlquantum.result import ExecutionResult
from hlquantum.transpiler import transpile as default_transpile
from hlquantum.mitigation import apply_mitigation

_default_backend: Optional[Backend] = None


def set_default_backend(backend: Backend) -> None:
    global _default_backend
    _default_backend = backend


def get_default_backend() -> Backend:
    global _default_backend
    if _default_backend is None:
        from hlquantum.backends.cudaq_backend import CudaQBackend
        _default_backend = CudaQBackend()
    return _default_backend


def run(
    circuit_or_kernel: Union[QuantumCircuit, Kernel],
    *,
    shots: int = 1000,
    include_statevector: bool = False,
    transpile: bool = False,
    mitigation: Optional[Union[List[Any], Any]] = None,
    backend: Optional[Backend] = None,
    **kwargs,
) -> ExecutionResult:
    """Execute a circuit or kernel and return the result.

    Parameters
    ----------
    circuit_or_kernel : Circuit | Kernel
        What to execute.
    shots : int, optional
        Number of measurement shots (default 1000).
    include_statevector : bool, optional
        If *True*, attempt to include the state vector in the result.
    transpile : bool, optional
        If *True*, apply HLQuantum's transpilation optimizations before running.
    mitigation : MitigationMethod | list[MitigationMethod], optional
        Error mitigation method(s) to apply to the results.
    backend : Backend, optional
        Override the default backend for this call.
    **kwargs
        Forwarded to :meth:`Backend.run`.
    """
    if isinstance(circuit_or_kernel, Kernel):
        circuit = circuit_or_kernel.circuit
    elif isinstance(circuit_or_kernel, QuantumCircuit):
        circuit = circuit_or_kernel
    else:
        raise TypeError(
            f"Expected a Circuit or Kernel, got {type(circuit_or_kernel).__name__}"
        )

    # 1. Transpile if requested
    if transpile:
        circuit = default_transpile(circuit)

    be = backend or get_default_backend()
    be.validate(circuit)
    
    # 2. Execute
    result = be.run(
        circuit, 
        shots=shots, 
        include_statevector=include_statevector, 
        **kwargs
    )

    # 3. Mitigate if requested
    if mitigation:
        from hlquantum.mitigation import MitigationMethod
        methods = mitigation if isinstance(mitigation, list) else [mitigation]
        result = apply_mitigation(result, methods)

    return result
