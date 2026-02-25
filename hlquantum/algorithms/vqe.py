"""
hlquantum.algorithms.vqe
~~~~~~~~~~~~~~~~~~~~~~~~

Variational Quantum Eigensolver (VQE) utilities.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from hlquantum.circuit import Circuit
from hlquantum.runner import run


def vqe_solve(
    ansatz: Union[Circuit, Callable[[List[float]], Circuit]],
    initial_params: List[float],
    optimizer: Optional[Callable] = None,
    shots: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """Solve an optimization problem using VQE.

    Parameters
    ----------
    ansatz : Circuit | Callable[[list[float]], Circuit]
        A parameterized circuit or a function returning a circuit.
    initial_params : list[float]
        Starting parameters for the optimizer.
    optimizer : Callable, optional
        A classical optimizer (defaults to scipy.optimize.minimize).
    shots : int, optional
        Number of shots for expectation value estimation.
    **kwargs
        Forwarded to the runner.

    Returns
    -------
    dict
        Optimization results including optimal parameters and minimum energy.
    """
    try:
        from scipy.optimize import minimize as _minimize
    except ImportError:
        raise ImportError(
            "scipy is required for vqe_solve but is not installed.\n"
            "Install it with:  pip install scipy"
        )

    def objective(params):
        if isinstance(ansatz, Circuit):
            # Map params list to the circuit's parameters
            param_map = {p: params[i] for i, p in enumerate(ansatz.parameters)}
            circuit = ansatz.bind_parameters(param_map)
        else:
            circuit = ansatz(params)
            
        # Ensure measurements are present
        if not any(g.name == "mz" for g in circuit.gates):
            circuit.measure_all()
            
        result = run(circuit, shots=shots, **kwargs)
        return result.expectation_value()

    # Use default COBYLA optimizer if none provided
    if optimizer is None:
        from scipy.optimize import minimize
        res = minimize(objective, initial_params, method='COBYLA')
    else:
        res = optimizer(objective, initial_params)

    return {
        "fun": getattr(res, "fun", res),
        "x": getattr(res, "x", initial_params),
        "raw": res
    }


def hardware_efficient_ansatz(num_qubits: int, params: List[float]) -> Circuit:
    """A standard Hardware-Efficient Ansatz (HEA) template.
    Usually consists of Ry/Rz layers and CX entanglers.
    """
    qc = Circuit(num_qubits)
    
    # Simple Ry rotation layer — one param per qubit
    for i in range(num_qubits):
        qc.ry(i, params[i] if i < len(params) else params[-1])
        
    # Entanglement layer
    if num_qubits > 1:
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
            
    return qc


# ── User-friendly aliases ────────────────────────────────────────────────────
find_minimum_energy = vqe_solve
"""Alias for :func:`vqe_solve` — find the minimum energy (ground state) of a system."""

variational_circuit = hardware_efficient_ansatz
"""Alias for :func:`hardware_efficient_ansatz` — build a parameterized variational circuit."""
