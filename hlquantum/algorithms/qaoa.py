"""
hlquantum.algorithms.qaoa
~~~~~~~~~~~~~~~~~~~~~~~~~

Quantum Approximate Optimization Algorithm (QAOA) implementation.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from hlquantum.circuit import Circuit, Parameter
from hlquantum.runner import run


def qaoa_solve(
    cost_hamiltonian: List[Dict[str, Any]],
    p: int = 1,
    optimizer: Optional[Callable] = None,
    shots: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """Solve an optimization problem using QAOA.

    Parameters
    ----------
    cost_hamiltonian : list[dict]
        Representation of the cost function (e.g., Max-Cut).
        Each dict should have 'qubits': (i, j) and 'weight': float.
    p : int
        Number of QAOA layers (steps).
    optimizer : Callable, optional
        A classical optimizer (defaults to scipy.optimize.minimize).
    shots : int, optional
        Number of shots for expectation value estimation.
    **kwargs
        Forwarded to the runner.

    Returns
    -------
    dict
        Optimization results including optimal beta/gamma and max-cut value.
    """
    # 1. Determine number of qubits
    all_qubits = set()
    for term in cost_hamiltonian:
        all_qubits.update(term['qubits'])
    n_qubits = max(all_qubits) + 1 if all_qubits else 0

    # 2. Build the QAOA Ansatz function
    def build_ansatz(params: List[float]) -> Circuit:
        gamma = params[:p]
        beta = params[p:]
        
        qc = Circuit(n_qubits)
        
        # Initial state: Hadamards on all qubits
        for i in range(n_qubits):
            qc.h(i)
            
        for step in range(p):
            # Cost Layer: e^(-i * gamma * H_C)
            # For each term ZZ_ij, we apply: CX(i,j), RZ(j, 2*gamma*w), CX(i,j)
            for term in cost_hamiltonian:
                q1, q2 = term['qubits']
                w = term.get('weight', 1.0)
                qc.cx(q1, q2)
                qc.rz(q2, 2 * gamma[step] * w)
                qc.cx(q1, q2)
            
            # Mixer Layer: e^(-i * beta * H_X)
            for i in range(n_qubits):
                qc.rx(i, 2 * beta[step])
                
        return qc

    # 3. Objective function for optimizer
    def objective(params):
        qc = build_ansatz(params)
        qc.measure_all()
        result = run(qc, shots=shots, **kwargs)
        # We minimize the cost (negative for max-cut)
        # Expectation result is a simplification
        return result.expectation_value()

    # 4. Optimization
    initial_params = [1.0] * (2 * p) # Gammas followed by Betas
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for qaoa_solve.")

    if optimizer is None:
        res = minimize(objective, initial_params, method='COBYLA')
    else:
        res = optimizer(objective, initial_params)

    return {
        "fun": getattr(res, "fun", res),
        "x": getattr(res, "x", initial_params),
        "raw": res,
        "n_qubits": n_qubits
    }
