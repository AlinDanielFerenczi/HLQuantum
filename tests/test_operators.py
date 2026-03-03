from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hlquantum.operators import (
    Operator,
    ScalarOperator,
    TimeDependentOperator,
    annihilate,
    create,
    displace,
    number,
    parity,
    spin_minus,
    spin_plus,
    spin_x,
    spin_y,
    spin_z,
    squeeze,
)


def test_pauli_operators():
    # Spin operators match standard Pauli definitions
    assert_allclose(spin_x().matrix, [[0, 1], [1, 0]])
    assert_allclose(spin_y().matrix, [[0, -1j], [1j, 0]])
    assert_allclose(spin_z().matrix, [[1, 0], [0, -1]])
    
    assert spin_x().num_qubits == 1
    assert spin_y().num_qubits == 1
    assert spin_z().num_qubits == 1


def test_ladder_operators():
    # σ+ maps |1> -> |0> meaning matrix is [[0, 1], [0, 0]]
    assert_allclose(spin_plus().matrix, [[0, 1], [0, 0]])
    # σ- maps |0> -> |1> meaning matrix is [[0, 0], [1, 0]]
    assert_allclose(spin_minus().matrix, [[0, 0], [1, 0]])


def test_operator_algebra():
    X = spin_x()
    Y = spin_y()
    Z = spin_z()
    
    # X*X = I
    assert_allclose((X * X).matrix, np.eye(2))
    
    # X*Y = iZ
    assert_allclose((X * Y).matrix, 1j * Z.matrix)
    
    # Y*Y = I
    assert_allclose((Y * Y).matrix, np.eye(2))
    
    # Addition with scalar
    H = 2.0 * Z + 0.5 * X
    assert_allclose(H.matrix, [[2, 0.5], [0.5, -2]])


def test_bosonic_operators():
    dim = 4
    a = annihilate(dim)
    adag = create(dim)
    n = number(dim)
    
    # N |n> = n|n> roughly means diagonal is [0, 1, 2, ..., dim-1]
    assert_allclose(np.diag(n.matrix), np.arange(dim))
    
    # Commutator [a, a_dag] = 1 (approximately holding for truncated space)
    # Be careful at the boundary, only the first dim-1 elements are exactly 1
    comm = a * adag - adag * a
    assert_allclose(np.diag(comm.matrix)[:-1], np.ones(dim - 1))


def test_parity_operator():
    dim = 4
    p = parity(dim)
    # Parity should have eigenvalues 1, -1, 1, -1...
    assert_allclose(np.diag(p.matrix), [1, -1, 1, -1])


def test_time_dependent_operator():
    # Test definition and evaluation
    X = spin_x()
    Z = spin_z()
    
    # Construct a time dependent Hamiltonian H(t) = Z + cos(pi*t)*X
    f = ScalarOperator(lambda t: np.cos(np.pi * t))
    H = Z + f * X
    
    assert isinstance(H, TimeDependentOperator)
    
    # Evaluate at t=0
    # H(0) = Z + X
    assert_allclose(H.evaluate(0), Z.matrix + X.matrix)
    
    # Evaluate at t=1
    # H(1) = Z - X
    assert_allclose(H.evaluate(1), Z.matrix - X.matrix)
    
    # Evaluate at t=0.5
    # H(0.5) = Z
    assert_allclose(H.evaluate(0.5), Z.matrix, atol=1e-15)
