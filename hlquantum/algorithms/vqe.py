"""Variational Quantum Eigensolver (VQE) utilities."""

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
    """Solve an optimization problem using VQE."""
    try:
        from scipy.optimize import minimize as _minimize
    except ImportError:
        raise ImportError("scipy is required for vqe_solve. Install it with: pip install scipy")

    def objective(params):
        if isinstance(ansatz, Circuit):
            param_map = {p: params[i] for i, p in enumerate(ansatz.parameters)}
            circuit = ansatz.bind_parameters(param_map)
        else:
            circuit = ansatz(params)
            
        if not any(g.name == "mz" for g in circuit.gates):
            circuit.measure_all()
            
        result = run(circuit, shots=shots, **kwargs)
        return result.expectation_value()

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
    """A standard Hardware-Efficient Ansatz (HEA) template."""
    qc = Circuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(i, params[i] if i < len(params) else params[-1])
        
    if num_qubits > 1:
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    return qc


find_minimum_energy = vqe_solve
variational_circuit = hardware_efficient_ansatz

