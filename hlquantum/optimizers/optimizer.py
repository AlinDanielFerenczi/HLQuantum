"""
hlquantum.optimizers.optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for classical optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Union
from collections import namedtuple

import numpy as np


class OptimizerResult:
    """The result of an optimization routine."""
    def __init__(self):
        self.x: np.ndarray = None            # The optimal point
        self.fun: float = None               # The optimal value
        self.nfev: int = None                # Number of objective function evaluations
        self.nit: int = None                 # Number of iterations


class Optimizer(ABC):
    """Base class for optimization algorithms."""

    @abstractmethod
    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        """Minimize the objective function.

        Parameters
        ----------
        fun : Callable[[np.ndarray], float]
            The objective function to minimize.
        x0 : np.ndarray
            Initial guess for the parameters.
        bounds : List[Tuple[float, float]], optional
            Optional bounds for the parameters.

        Returns
        -------
        OptimizerResult
            The results of the optimization.
        """
        pass
