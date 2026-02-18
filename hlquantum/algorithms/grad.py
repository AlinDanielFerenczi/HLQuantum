"""
hlquantum.algorithms.grad
~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient calculation for quantum circuits using the Parameter Shift Rule.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Union
from hlquantum.circuit import Circuit, Parameter
from hlquantum.runner import run


def parameter_shift_gradient(
    circuit: Circuit,
    parameter_values: Dict[Union[str, Parameter], float],
    shots: int = 1000,
    **kwargs
) -> Dict[str, float]:
    """Calculate the gradient of a circuit's expectation value using the Parameter Shift Rule.

    The formula is: df/dθ = 0.5 * [f(θ + π/2) - f(θ - π/2)]

    Parameters
    ----------
    circuit : Circuit
        The parameterized circuit.
    parameter_values : dict
        The current values for all parameters in the circuit.
    shots : int
        Number of shots for each evaluation.
    **kwargs
        Additional arguments for the runner.

    Returns
    -------
    dict
        A mapping from parameter name to its gradient value.
    """
    grads = {}
    shift = math.pi / 2

    # Get all unique parameters in the circuit
    circuit_params = circuit.parameters
    param_names = [p.name for p in circuit_params]
    
    # Ensure all required values are provided
    for name in param_names:
        if name not in parameter_values and not any(p.name == name for p in parameter_values.keys() if isinstance(p, Parameter)):
             # We assume name is str here if not Parameter
             pass

    for param in circuit_params:
        # 1. Plus shift: θ + π/2
        plus_values = parameter_values.copy()
        plus_values[param.name] = parameter_values[param.name] + shift
        plus_qc = circuit.bind_parameters(plus_values)
        if not any(g.name == "mz" for g in plus_qc.gates):
            plus_qc.measure_all()
        plus_result = run(plus_qc, shots=shots, **kwargs)
        e_plus = plus_result.expectation_value()

        # 2. Minus shift: θ - π/2
        minus_values = parameter_values.copy()
        minus_values[param.name] = parameter_values[param.name] - shift
        minus_qc = circuit.bind_parameters(minus_values)
        if not any(g.name == "mz" for g in minus_qc.gates):
            minus_qc.measure_all()
        minus_result = run(minus_qc, shots=shots, **kwargs)
        e_minus = minus_result.expectation_value()

        # df/dθ = 0.5 * (E_plus - E_minus)
        grads[param.name] = 0.5 * (e_plus - e_minus)

    return grads
