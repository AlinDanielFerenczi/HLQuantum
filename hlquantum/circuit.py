"""
hlquantum.circuit
~~~~~~~~~~~~~~~~~~

Backend-agnostic quantum circuit representation.

A QuantumCircuit is a simple, serialisable description of quantum gates
and measurements that can be handed to *any* backend for execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union


@dataclass(frozen=True)
class Parameter:
    """A symbolic parameter for a quantum gate."""
    name: str

    def __repr__(self) -> str:
        return f"${self.name}"


@dataclass
class Gate:
    """A single quantum gate operation."""

    name: str
    targets: Tuple[int, ...]
    controls: Tuple[int, ...] = ()
    params: Tuple[Union[float, Parameter], ...] = ()

    def __repr__(self) -> str:
        parts = [self.name]
        if self.controls:
            parts.append(f"controls={self.controls}")
        parts.append(f"targets={self.targets}")
        if self.params:
            param_repr = [str(p) if isinstance(p, Parameter) else f"{p:.3f}" for p in self.params]
            parts.append(f"params=({', '.join(param_repr)})")
        return f"Gate({', '.join(parts)})"


class Circuit:
    """High-level, backend-agnostic quantum circuit.

    Example
    -------
    >>> qc = Circuit(2)
    >>> qc.h(0).cx(0, 1).measure_all()
    >>> print(qc)
    Circuit(num_qubits=2, gates=3)
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.gates: List[Gate] = []
        self.metadata: Dict[str, Any] = {}

    # Alias for users coming from other frameworks
    @property
    def qubits(self):
        return range(self.num_qubits)

    # ------------------------------------------------------------------ #
    #  Single-qubit gates
    # ------------------------------------------------------------------ #

    def h(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="h", targets=(target,)))
        return self

    def x(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="x", targets=(target,)))
        return self

    def y(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="y", targets=(target,)))
        return self

    def z(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="z", targets=(target,)))
        return self

    def s(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="s", targets=(target,)))
        return self

    def t(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="t", targets=(target,)))
        return self

    # ------------------------------------------------------------------ #
    #  Parameterised single-qubit gates
    # ------------------------------------------------------------------ #

    def rx(self, target: int, angle: Union[float, Parameter, str]) -> "Circuit":
        self._validate_qubits(target)
        p = Parameter(angle) if isinstance(angle, str) else angle
        self.gates.append(Gate(name="rx", targets=(target,), params=(p,)))
        return self

    def ry(self, target: int, angle: Union[float, Parameter, str]) -> "Circuit":
        self._validate_qubits(target)
        p = Parameter(angle) if isinstance(angle, str) else angle
        self.gates.append(Gate(name="ry", targets=(target,), params=(p,)))
        return self

    def rz(self, target: int, angle: Union[float, Parameter, str]) -> "Circuit":
        self._validate_qubits(target)
        p = Parameter(angle) if isinstance(angle, str) else angle
        self.gates.append(Gate(name="rz", targets=(target,), params=(p,)))
        return self

    # ------------------------------------------------------------------ #
    #  Multi-qubit gates
    # ------------------------------------------------------------------ #

    def cx(self, control: int, target: int) -> "Circuit":
        self._validate_qubits(control, target)
        self.gates.append(Gate(name="cx", targets=(target,), controls=(control,)))
        return self

    def cz(self, control: int, target: int) -> "Circuit":
        self._validate_qubits(control, target)
        self.gates.append(Gate(name="cz", targets=(target,), controls=(control,)))
        return self

    def swap(self, q0: int, q1: int) -> "Circuit":
        self._validate_qubits(q0, q1)
        self.gates.append(Gate(name="swap", targets=(q0, q1)))
        return self

    def ccx(self, c0: int, c1: int, target: int) -> "Circuit":
        self._validate_qubits(c0, c1, target)
        self.gates.append(Gate(name="ccx", targets=(target,), controls=(c0, c1)))
        return self

    # ------------------------------------------------------------------ #
    #  Measurement
    # ------------------------------------------------------------------ #

    def measure(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="mz", targets=(target,)))
        return self

    def measure_all(self) -> "Circuit":
        for q in range(self.num_qubits):
            self.measure(q)
        return self

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    @property
    def depth(self) -> int:
        return len(self.gates)

    @property
    def parameters(self) -> List[Parameter]:
        """Returns a list of all unique parameters in the circuit."""
        params = []
        seen = set()
        for gate in self.gates:
            for p in gate.params:
                if isinstance(p, Parameter) and p.name not in seen:
                    params.append(p)
                    seen.add(p.name)
        return params

    def bind_parameters(self, value_dict: Dict[Union[str, Parameter], float]) -> "Circuit":
        """Returns a new circuit with parameters replaced by values."""
        # Normalize dict keys to names
        normalized_values = {}
        for k, v in value_dict.items():
            name = k.name if isinstance(k, Parameter) else k
            normalized_values[name] = v

        new_qc = Circuit(self.num_qubits)
        new_qc.metadata = self.metadata.copy()
        
        for gate in self.gates:
            new_params = []
            for p in gate.params:
                if isinstance(p, Parameter):
                    if p.name not in normalized_values:
                        raise ValueError(f"Missing value for parameter: {p.name}")
                    new_params.append(normalized_values[p.name])
                else:
                    new_params.append(p)
            
            new_gate = Gate(
                name=gate.name,
                targets=gate.targets,
                controls=gate.controls,
                params=tuple(new_params)
            )
            new_qc.gates.append(new_gate)
            
        return new_qc

    def _validate_qubits(self, *qubits: int) -> None:
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise IndexError(
                    f"Qubit index {q} out of range for circuit with "
                    f"{self.num_qubits} qubits."
                )

    def __or__(self, other: "Circuit") -> "Circuit":
        if not isinstance(other, Circuit):
            return NotImplemented
        
        # Determine the number of qubits for the combined circuit
        num_qubits = max(self.num_qubits, other.num_qubits)
        new_qc = Circuit(num_qubits)
        
        # Copy gates from self
        for gate in self.gates:
            new_qc.gates.append(gate)
            
        # Copy gates from other
        for gate in other.gates:
            new_qc.gates.append(gate)
            
        return new_qc

    def __repr__(self) -> str:
        return (
            f"Circuit(num_qubits={self.num_qubits}, "
            f"gates={len(self.gates)})"
        )

    def __len__(self) -> int:
        return len(self.gates)


# Alias for backward compatibility
QuantumCircuit = Circuit
