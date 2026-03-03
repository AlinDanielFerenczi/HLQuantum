import numpy as np
from hlquantum.optimizers import COBYLA, SPSA

def test_optimizers():
    # Simple quadratic function: f(x) = x_0^2 + x_1^2 + x_2^2
    def objective(x):
        return np.sum(x**2)
        
    x0 = np.array([1.0, 2.0, -1.5])
    
    # Test COBYLA
    cobyla = COBYLA(maxiter=200, tol=1e-5)
    res_cobyla = cobyla.minimize(objective, x0)
    
    # Test SPSA
    spsa = SPSA(maxiter=300, a=0.2, c=0.1)
    res_spsa = spsa.minimize(objective, x0)
    
    print(f"COBYLA: fun={res_cobyla.fun:.6f}, nfev={res_cobyla.nfev}, x={res_cobyla.x}")
    print(f"SPSA:   fun={res_spsa.fun:.6f}, nit={res_spsa.nit}, x={res_spsa.x}")
    
    assert res_cobyla.fun < 1e-3, "COBYLA did not converge"
    assert res_spsa.fun < 1e-2, "SPSA did not converge"
    print("Optimizer tests passed!")

if __name__ == "__main__":
    test_optimizers()
