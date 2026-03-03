"""Error mitigation hooks and post-processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from hlquantum.result import ExecutionResult


class MitigationMethod(ABC):
    """Base class for error mitigation techniques."""

    @abstractmethod
    def apply(self, result: ExecutionResult) -> ExecutionResult:
        """Apply mitigation to the result."""
        ...


class ThresholdMitigation(MitigationMethod):
    """Filters low-probability bitstrings."""

    def __init__(self, threshold: float = 0.005) -> None:
        self.threshold = threshold

    def apply(self, result: ExecutionResult) -> ExecutionResult:
        if not result.counts:
            return result

        from hlquantum.result import ExecutionResult as ER
        total_shots = result.shots
        new_counts = {
            k: v for k, v in result.counts.items() 
            if (v / total_shots) >= self.threshold
        }
        
        return ER(
            counts=new_counts,
            shots=result.shots,
            backend_name=result.backend_name,
            raw=result.raw,
            state_vector=result.state_vector,
            metadata=result.metadata,
        )


class ReadoutMitigation(MitigationMethod):
    """Readout Error Mitigation placeholder."""

    def apply(self, result: ExecutionResult) -> ExecutionResult:
        return result


def apply_mitigation(
    result: ExecutionResult, 
    methods: Optional[List[MitigationMethod]] = None
) -> ExecutionResult:
    """Apply a sequence of mitigation methods."""
    if not methods:
        return result
    for method in methods:
        result = method.apply(result)
    return result

