"""Quantum system dynamics and time evolution."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import scipy.integrate

from hlquantum.operators import Operator, ScalarOperator, TimeDependentOperator


def evolve(
    hamiltonian: Union[Operator, TimeDependentOperator],
    initial_state: np.ndarray,
    time_span: tuple[float, float],
    steps: int = 100
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate time evolution under a Hamiltonian: i d|ψ>/dt = H(t) |ψ>."""
    t0, tf = time_span
    t_eval = np.linspace(t0, tf, steps)
    
    def schrodinger_eq(t: float, y: np.ndarray) -> np.ndarray:
        return -1j * (hamiltonian.evaluate(t) @ y)
    
    res = scipy.integrate.solve_ivp(
        schrodinger_eq,
        [t0, tf],
        initial_state,
        t_eval=t_eval,
        method="RK45",
    )
    
    if not res.success:
        raise RuntimeError(f"Integration failed: {res.message}")
        
    states = [res.y[:, i] for i in range(res.y.shape[1])]
    return res.t, states

