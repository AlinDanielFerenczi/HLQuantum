"""
hlquantum.transpiler
~~~~~~~~~~~~~~~~~~~~

Optimization and transpilation logic for quantum circuits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from hlquantum.circuit import Gate, QuantumCircuit


class TranspilationPass(ABC):
    """Abstract base class for a transpilation pass."""

    @abstractmethod
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Run the pass on the given circuit."""
        ...


class RemoveRedundantGates(TranspilationPass):
    """Removes consecutive gates that cancel each other out (e.g., H followed by H)."""

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        if not circuit.gates:
            return circuit

        new_circuit = QuantumCircuit(circuit.num_qubits)
        new_circuit.metadata = circuit.metadata.copy()
        
        # Simple optimization: filter out adjacent identical self-inverse gates
        # In a real production system, this would be more sophisticated
        i = 0
        while i < len(circuit.gates):
            gate = circuit.gates[i]
            
            # Check for self-inverse gates (H, X, Y, Z, CX, CZ, SWAP)
            if (i + 1 < len(circuit.gates) and 
                gate.name == circuit.gates[i+1].name and 
                gate.targets == circuit.gates[i+1].targets and 
                gate.controls == circuit.gates[i+1].controls and
                gate.name in ("h", "x", "y", "z", "cx", "cz", "swap")):
                i += 2  # Skip both
                continue
            
            new_circuit.gates.append(gate)
            i += 1
            
        return new_circuit


class MergeRotations(TranspilationPass):
    """Merges consecutive rotation gates on the same qubit."""

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        if len(circuit.gates) < 2:
            return circuit

        new_circuit = QuantumCircuit(circuit.num_qubits)
        new_circuit.metadata = circuit.metadata.copy()
        
        i = 0
        while i < len(circuit.gates):
            gate = circuit.gates[i]
            
            if (gate.name in ("rx", "ry", "rz") and 
                i + 1 < len(circuit.gates) and 
                circuit.gates[i+1].name == gate.name and 
                circuit.gates[i+1].targets == gate.targets):
                
                # Merge the angles
                new_angle = gate.params[0] + circuit.gates[i+1].params[0]
                merged_gate = Gate(
                    name=gate.name, 
                    targets=gate.targets, 
                    params=(new_angle,)
                )
                new_circuit.gates.append(merged_gate)
                i += 2
                continue
            
            new_circuit.gates.append(gate)
            i += 1
            
        return new_circuit


class Transpiler:
    """Orchestrates multiple transpilation passes."""

    def __init__(self, passes: Optional[List[TranspilationPass]] = None) -> None:
        self.passes = passes or [
            RemoveRedundantGates(),
            MergeRotations(),
        ]

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply all registered passes to the circuit."""
        optimized = circuit
        for p in self.passes:
            optimized = p.run(optimized)
        return optimized


# Default global transpiler
_default_transpiler = Transpiler()


def transpile(circuit: QuantumCircuit) -> QuantumCircuit:
    """Helper to transpile a circuit using the default transpiler."""
    return _default_transpiler.transpile(circuit)
