"""Backend-agnostic quantum circuit representation."""

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
    """High-level quantum circuit."""

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.gates: List[Gate] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def qubits(self):
        return range(self.num_qubits)

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

    def measure(self, target: int) -> "Circuit":
        self._validate_qubits(target)
        self.gates.append(Gate(name="mz", targets=(target,)))
        return self

    def measure_all(self) -> "Circuit":
        for q in range(self.num_qubits):
            self.measure(q)
        return self

    @property
    def depth(self) -> int:
        """Circuit depth (longest critical path)."""
        if not self.gates:
            return 0
        qubit_layers: Dict[int, int] = {}
        max_depth = 0
        for gate in self.gates:
            involved = list(gate.targets) + list(gate.controls)
            layer = max((qubit_layers.get(q, 0) for q in involved), default=0)
            for q in involved:
                qubit_layers[q] = layer + 1
            max_depth = max(max_depth, layer + 1)
        return max_depth

    @property
    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self.gates)

    @property
    def parameters(self) -> List[Parameter]:
        """Unique parameters in the circuit."""
        params = []
        seen = set()
        for gate in self.gates:
            for p in gate.params:
                if isinstance(p, Parameter) and p.name not in seen:
                    params.append(p)
                    seen.add(p.name)
        return params

    def bind_parameters(self, value_dict: Dict[Union[str, Parameter], float]) -> "Circuit":
        """Replace parameters with values and return a new circuit."""
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
                        raise ValueError(f"Missing parameter value: {p.name}")
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
                raise IndexError(f"Qubit index {q} out of range.")

    def __or__(self, other: "Circuit") -> "Circuit":
        if not isinstance(other, Circuit):
            return NotImplemented
        
        num_qubits = max(self.num_qubits, other.num_qubits)
        new_qc = Circuit(num_qubits)
        for gate in self.gates:
            new_qc.gates.append(gate)
        for gate in other.gates:
            new_qc.gates.append(gate)
        return new_qc

    def __repr__(self) -> str:
        return f"Circuit(num_qubits={self.num_qubits}, gates={len(self.gates)})"

    def __len__(self) -> int:
        return len(self.gates)

    @classmethod
    def from_qiskit(cls, qc: Any) -> "Circuit":
        """Import circuit from Qiskit."""
        new_qc = cls(qc.num_qubits)
        for instruction in qc.data:
            op, qargs = instruction.operation, instruction.qubits
            name = op.name
            try:
                targets = [qc.find_bit(q).index for q in qargs]
            except AttributeError:
                targets = [q.index for q in qargs]
            
            if name in ("rx", "ry", "rz", "p", "u1"):
                func_name = "rz" if name in ("p", "u1", "rz") else name
                getattr(new_qc, func_name)(targets[0], float(op.params[0]))
            elif name in ("cx", "cz"):
                getattr(new_qc, name)(targets[0], targets[1])
            elif name == "swap":
                new_qc.swap(targets[0], targets[1])
            elif name == "ccx":
                new_qc.ccx(targets[0], targets[1], targets[2])
            elif name in ("h", "x", "y", "z", "s", "t"):
                getattr(new_qc, name)(targets[0])
            elif name == "measure":
                new_qc.measure(targets[0])
            elif name == "barrier":
                pass
            else:
                raise ValueError(f"Unsupported Qiskit gate: {name}")
        return new_qc

    @classmethod
    def from_cirq(cls, circuit: Any) -> "Circuit":
        """Import circuit from Cirq."""
        import math
        qubits = sorted(list(circuit.all_qubits()))
        qubit_map = {q: i for i, q in enumerate(qubits)}
        new_qc = cls(max(1, len(qubits)))
        
        for moment in circuit:
            for op in moment:
                gate = op.gate
                targets = [qubit_map[q] for q in op.qubits]
                gate_str = str(gate).lower()
                
                if "measure" in gate_str:
                    new_qc.measure(targets[0])
                    continue
                
                if hasattr(gate, "exponent"):
                    angle = float(gate.exponent) * math.pi
                    if "rx" in gate_str or "xpow" in gate_str:
                        if angle == math.pi: new_qc.x(targets[0])
                        else: new_qc.rx(targets[0], angle)
                        continue
                    if "ry" in gate_str or "ypow" in gate_str:
                        if angle == math.pi: new_qc.y(targets[0])
                        else: new_qc.ry(targets[0], angle)
                        continue
                    if "rz" in gate_str or "zpow" in gate_str:
                        if angle == math.pi: new_qc.z(targets[0])
                        elif angle == math.pi / 2: new_qc.s(targets[0])
                        elif angle == math.pi / 4: new_qc.t(targets[0])
                        else: new_qc.rz(targets[0], angle)
                        continue
                
                if gate_str == "h":
                    new_qc.h(targets[0])
                elif gate_str in ("x", "y", "z", "s", "t"):
                    getattr(new_qc, gate_str)(targets[0])
                elif "cnot" in gate_str or gate_str == "cx":
                    new_qc.cx(targets[0], targets[1])
                elif gate_str == "cz":
                    new_qc.cz(targets[0], targets[1])
                elif gate_str == "swap":
                    new_qc.swap(targets[0], targets[1])
                elif "ccx" in gate_str or "toffoli" in gate_str:
                    new_qc.ccx(targets[0], targets[1], targets[2])
                else:
                    raise ValueError(f"Unsupported Cirq gate: {gate}")
        return new_qc


QuantumCircuit = Circuit

