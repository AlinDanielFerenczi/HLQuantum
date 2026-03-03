"""
hlquantum.optimizers.cobyla
~~~~~~~~~~~~~~~~~~~~~~~~~~~

COBYLA (Constrained Optimization BY Linear Approximations) Optimizer.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List
import numpy as np
import scipy.optimize as opt

from hlquantum.optimizers.optimizer import Optimizer, OptimizerResult


class COBYLA(Optimizer):
    """Constrained Optimization By Linear Approximation optimizer.
    
    COBYLA is a numerical optimization method for constrained problems
    where the derivative of the objective function is not known. It uses
    linear approximations to the objective function, making it robust and
    commonly used in VQA/VQE applications.
    """

    def __init__(self, maxiter: int = 1000, tol: float = 1e-4, rhobeg: float = 1.0):
        """
        Parameters
        ----------
        maxiter : int
            Maximum number of function evaluations.
        tol : float
            Final accuracy in the optimization (stopping criterion).
        rhobeg : float
            Reasonable initial changes to the variables.
        """
        self.maxiter = maxiter
        self.tol = tol
        self.rhobeg = rhobeg

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        res = opt.minimize(
            fun,
            x0,
            method="COBYLA",
            options={
                "maxiter": self.maxiter,
                "tol": self.tol,
                "rhobeg": self.rhobeg
            }
        )

        result = OptimizerResult()
        result.x = res.x
        result.fun = res.fun
        result.nfev = res.nfev
        # COBYLA doesn't return 'nit' exactly like others, but nfev is closely related
        result.nit = res.get('nit', res.nfev)
        
        return result
