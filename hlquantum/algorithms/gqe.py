"""
hlquantum.algorithms.gqe
~~~~~~~~~~~~~~~~~~~~~~~~

Generative Quantum Eigensolver (GQE) implementation.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
from hlquantum.circuit import Circuit, Parameter
from hlquantum.runner import run


def gqe_solve(
    ansatz: Circuit,
    loss_fn: Callable[[Any], float],
    optimizer: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """Solve an optimization problem using a Generative Quantum Eigensolver approach.

    GQE uses generative learning to find the ground state or approximate 
    the probability distribution of a Hamiltonian.
    """
    
    def objective(params):
        param_map = {p: params[i] for i, p in enumerate(ansatz.parameters)}
        qc = ansatz.bind_parameters(param_map)
        
        # Forward pass: Generate samples from the quantum state
        if not any(g.name == "mz" for g in qc.gates):
            qc.measure_all()
            
        result = run(qc, **kwargs)
        
        # The loss function compares generated distribution to target properties
        return loss_fn(result)

    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for gqe_solve.")

    # Initialize with random parameters
    n_params = len(ansatz.parameters)
    initial_params = [0.0] * n_params
    
    if optimizer is None:
        res = minimize(objective, initial_params, method='COBYLA')
    else:
        res = optimizer(objective, initial_params)

    return {
        "fun": getattr(res, "fun", res),
        "x": getattr(res, "x", initial_params),
        "raw": res
    }


# ── User-friendly alias ──────────────────────────────────────────────────────
learn_distribution = gqe_solve
"""Alias for :func:`gqe_solve` — learn a target probability distribution using a quantum circuit."""
