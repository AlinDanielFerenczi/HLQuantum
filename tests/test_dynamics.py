import numpy as np
import pytest
from numpy.testing import assert_allclose

from hlquantum.dynamics import evolve
from hlquantum.operators import ScalarOperator, spin_x, spin_z


def test_dynamics_static_hamiltonian():
    # A single qubit undergoing Rabi oscillation
    # H = (pi / 2) * X
    # It should map |0> to -i|1> after t=1
    H = (np.pi / 2) * spin_x()
    initial_state = np.array([1.0, 0.0], dtype=np.complex128)
    
    t_span = (0.0, 1.0)
    t_eval, states = evolve(H, initial_state, t_span, steps=10)
    
    # Final state at t=1
    final_state = states[-1]
    
    # Analytical solution: cos(pi/2) |0> - i * sin(pi/2) |1> = -i |1>
    expected_state = np.array([0.0, -1j])
    
    assert_allclose(final_state, expected_state, atol=1e-5)


def test_dynamics_time_dependent_hamiltonian():
    # Ensure evolve accepts TimeDependentOperator
    # We construct a scenario where driving only turns on slowly 
    X = spin_x()
    Z = spin_z()
    
    H0 = Z  # Static detuning
    driving = ScalarOperator(lambda t: t) # Ramps up linearily
    
    # The Hamiltonian is H(t) = Z + t * X
    # Using small time step execution
    initial_state = np.array([1.0, 0.0], dtype=np.complex128)
    
    try:
        t_eval, states = evolve(H0 + driving * X, initial_state, (0.0, 2.0), steps=20)
        assert len(states) == 20
        # The norm is preserved during unitary evolution
        for s in states:
            assert_allclose(np.linalg.norm(s), 1.0, atol=1e-5)
    except Exception as e:
        pytest.fail(f"Evolution failed with error: {e}")
