"""
hlquantum.result
~~~~~~~~~~~~~~~~~

Unified result container returned by all backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """Result of a quantum circuit execution.

    Attributes
    ----------
    counts : dict[str, int]
        Measurement outcome counts, e.g. ``{"00": 502, "11": 498}``.
    shots : int
        Total number of shots executed.
    backend_name : str
        Name of the backend that produced this result.
    raw : Any
        Backend-specific raw result object (for advanced users).
    state_vector : Any, optional
        The state vector if requested and supported by the backend.
    metadata : dict
        Any additional metadata from the execution.
    """

    counts: Dict[str, int] = field(default_factory=dict)
    shots: int = 0
    backend_name: str = ""
    raw: Any = None
    state_vector: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Derived helpers
    # ------------------------------------------------------------------ #

    def get_state_vector(self) -> Any:
        """Return the state vector as a numpy array, if available."""
        if self.state_vector is None:
            return None
        import numpy as np
        return np.asarray(self.state_vector)

    @property
    def probabilities(self) -> Dict[str, float]:
        if self.shots == 0:
            return {}
        return {k: v / self.shots for k, v in self.counts.items()}

    @property
    def most_probable(self) -> Optional[str]:
        if not self.counts:
            return None
        return max(self.counts, key=self.counts.get)  # type: ignore[arg-type]

    def expectation_value(self) -> float:
        if self.shots == 0:
            return 0.0
        total = 0.0
        for bitstring, count in self.counts.items():
            parity = (-1) ** bitstring.count("1")
            total += parity * count
        return total / self.shots

    def __repr__(self) -> str:
        top = dict(sorted(self.counts.items(), key=lambda x: -x[1])[:5])
        return (
            f"ExecutionResult(shots={self.shots}, "
            f"backend={self.backend_name!r}, top_counts={top})"
        )
