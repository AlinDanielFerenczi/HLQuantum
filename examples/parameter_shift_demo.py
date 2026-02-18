"""
Example demonstrating the Parameter Shift Rule for calculating quantum gradients.
"""

import math
from hlquantum.circuit import Circuit, Parameter
from hlquantum.algorithms.grad import parameter_shift_gradient
from hlquantum.backends.base import Backend
from hlquantum.result import ExecutionResult

class MockBackend(Backend):
    """A mock backend that returns counts based on a simple sine-wave like expectation."""
    @property
    def name(self) -> str: return "SinusoidalMock"
    
    def run(self, circuit, shots=1000, **kwargs):
        # Determine expectation based on the first RX/RY parameter if present
        # This is purely for demonstration of the gradient logic
        val = 0.5
        for gate in circuit.gates:
            if gate.name in ["rx", "ry"] and gate.params:
                p = gate.params[0]
                # If bound, it's a float
                if isinstance(p, (float, int)):
                    val = math.cos(p) # f(x) = cos(x)
                    break
        
        # Map expectation [-1, 1] to counts
        count_0 = int(shots * (val + 1) / 2)
        return ExecutionResult(counts={"0": count_0, "1": shots - count_0}, shots=shots)

def main():
    print("--- Parameter Shift Rule Demo ---")
    
    # 1. Create a parameterized circuit
    # f(theta) = <psi|O|psi> where psi = Ry(theta)|0>
    qc = Circuit(1)
    qc.ry(0, Parameter("theta"))
    qc.measure_all()
    
    # 2. Define the point at which to calculate the gradient
    theta_val = math.pi / 4 # 45 degrees
    p_values = {"theta": theta_val}
    
    print(f"Calculating gradient at theta = {theta_val:.4f} radians")
    
    # 3. Calculate gradient using Parameter Shift Rule
    # The MockBackend is configured to return cos(theta)
    # Derivative of cos(theta) is -sin(theta)
    # At pi/4, -sin(pi/4) approx -0.7071
    backend = MockBackend()
    grads = parameter_shift_gradient(qc, p_values, backend=backend, shots=10000)
    
    print("\nResults:")
    for name, grad_val in grads.items():
        expected = -math.sin(theta_val)
        print(f"  Parameter: {name}")
        print(f"  Calculated Gradient: {grad_val:.4f}")
        print(f"  Analytical Gradient (-sin(theta)): {expected:.4f}")
        print(f"  Difference: {abs(grad_val - expected):.4f}")

if __name__ == "__main__":
    main()
