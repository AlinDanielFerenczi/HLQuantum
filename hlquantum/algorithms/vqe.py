"""
hlquantum.algorithms.vqe
~~~~~~~~~~~~~~~~~~~~~~~~

Variational Quantum Eigensolver (VQE) utilities.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from hlquantum.circuit import Circuit
from hlquantum.runner import run


def vqe_solve(
    ansatz_fn: Callable[[List[float]], Circuit],
    initial_params: List[float],
    optimizer: Optional[Callable] = None,
    shots: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """Solve an optimization problem using VQE.

    Parameters
    ----------
    ansatz_fn : Callable[[list[float]], Circuit]
        A function that takes a list of parameters and returns an Ansatz Circuit.
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
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for vqe_solve. Install it with: pip install scipy")

    def objective(params):
        circuit = ansatz_fn(params)
        # Ensure measurements are present if requested
        if not any(g.name == "mz" for g in circuit.gates):
            circuit.measure_all()
            
        result = run(circuit, shots=shots, **kwargs)
        return result.expectation_value()

    # Use default COBYLA optimizer if none provided
    if optimizer is None:
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
    
    # Simple Ry rotation layer
    for i in range(num_qubits):
        # Using params sequentially
        p_idx = i % len(params)
        qc.ry(i, params[p_idx])
        
    # Entanglement layer
    if num_qubits > 1:
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
            
    return qc
