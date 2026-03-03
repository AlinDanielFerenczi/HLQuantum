"""
hlquantum.optimizers.spsa
~~~~~~~~~~~~~~~~~~~~~~~~~

Simultaneous Perturbation Stochastic Approximation (SPSA) Optimizer.
Adapted for classical optimization of quantum circuits.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Callable, Optional, Tuple, List

from hlquantum.optimizers.optimizer import Optimizer, OptimizerResult

class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.
    
    SPSA is an gradient descent method for optimizing systems with multiple unknown 
    parameters. It is highly suited for noisy objective functions (like those evaluated
    on quantum hardware) because it requires only two function evaluations per iteration 
    to approximate the gradient, regardless of the parameter dimension.
    """

    def __init__(
        self,
        maxiter: int = 100,
        learning_rate: Optional[float] = None,
        perturbation: Optional[float] = None,
        alpha: float = 0.602,
        gamma: float = 0.101,
        c: float = 0.2,
        A: float = 0.0,
        a: float = None
    ):
        """
        Parameters
        ----------
        maxiter : int
            Maximum number of iterations. Total function evaluations will be 2 * maxiter.
        learning_rate : float, optional
            The scaling factor for the update step. Overrides `a` if provided.
        perturbation : float, optional
            The magnitude of the perturbation. Overrides `c` if provided.
        alpha : float
            Exponent of the learning rate power series.
        gamma : float
            Exponent of the perturbation power series.
        c : float
            Base perturbation magnitude.
        A : float
            Stability constant for learning rate.
        a : float, optional
            Base learning rate magnitude. If None, it will be auto-calibrated.
        """
        self.maxiter = maxiter
        self.alpha = alpha
        self.gamma = gamma
        self.c = perturbation if perturbation is not None else c
        self.A = A
        self.a = learning_rate if learning_rate is not None else a

    def _calibrate(self, fun: Callable[[np.ndarray], float], x0: np.ndarray) -> float:
        """Calibrate the base learning rate `a` if not provided."""
        dim = len(x0)
        target_magnitude = 2 * np.pi / 10
        steps = 25
        avg_magnitudes = 0.0
        
        for _ in range(steps):
            delta = 1 - 2 * np.random.binomial(1, 0.5, size=dim)
            plus = fun(x0 + self.c * delta)
            minus = fun(x0 - self.c * delta)
            avg_magnitudes += np.abs((plus - minus) / (2 * self.c))
            
        avg_magnitudes /= steps
        
        a = target_magnitude / avg_magnitudes if avg_magnitudes > 1e-10 else target_magnitude
        return a

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        x = np.asarray(x0)
        dim = len(x)
        
        # Calibration
        a = self.a
        if a is None:
            a = self._calibrate(fun, x)
            
        nfev = 0
        nit = 0
        
        for k in range(self.maxiter):
            # Compute current learning rate and perturbation
            ak = a / ((k + 1 + self.A) ** self.alpha)
            ck = self.c / ((k + 1) ** self.gamma)
            
            # Generate random perturbation (Bernoulli +-1)
            delta = 1 - 2 * np.random.binomial(1, 0.5, size=dim)
            
            # Evaluate objective function
            plus = fun(x + ck * delta)
            minus = fun(x - ck * delta)
            nfev += 2
            
            # Approximate gradient
            gradient = (plus - minus) / (2 * ck) * delta
            
            # Update parameters
            x = x - ak * gradient
            
            # Simple boundary enforcement if provided
            if bounds is not None:
                for idx, (lower, upper) in enumerate(bounds):
                    x[idx] = np.clip(x[idx], lower, upper)
                    
            nit += 1
            
        result = OptimizerResult()
        result.x = x
        result.fun = fun(x)
        result.nfev = nfev + 1 # +1 for the final evaluation
        result.nit = nit
        
        return result
