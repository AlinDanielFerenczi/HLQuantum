"""
hlquantum.mitigation
~~~~~~~~~~~~~~~~~~~~

Error mitigation hooks and post-processing for quantum results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from hlquantum.result import ExecutionResult


class MitigationMethod(ABC):
    """Abstract base class for an error mitigation technique."""

    @abstractmethod
    def apply(self, result: ExecutionResult) -> ExecutionResult:
        """Apply mitigation to the execution result."""
        ...


class ThresholdMitigation(MitigationMethod):
    """Simple mitigation that filters out low-probability bitstrings as noise."""

    def __init__(self, threshold: float = 0.005) -> None:
        self.threshold = threshold

    def apply(self, result: ExecutionResult) -> ExecutionResult:
        if not result.counts:
            return result

        total_shots = result.shots
        new_counts = {
            k: v for k, v in result.counts.items() 
            if (v / total_shots) >= self.threshold
        }
        
        # Re-scale to keep total shots consistent if needed, 
        # or leave as is to show filtered results.
        # Here we just update the counts.
        result.counts = new_counts
        return result


class ReadoutMitigation(MitigationMethod):
    """Placeholder for Readout Error Mitigation (Matrix Inversion/Calibration)."""

    def apply(self, result: ExecutionResult) -> ExecutionResult:
        # Real implementation would require a calibration matrix
        # For now, this is a hook for future expansion
        return result


def apply_mitigation(
    result: ExecutionResult, 
    methods: Optional[List[MitigationMethod]] = None
) -> ExecutionResult:
    """Helper to apply a sequence of mitigation methods to a result."""
    if not methods:
        return result
        
    for method in methods:
        result = method.apply(result)
    return result
