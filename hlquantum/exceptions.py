"""
hlquantum.exceptions
~~~~~~~~~~~~~~~~~~~~~

Custom exception hierarchy for HLQuantum.
"""


class HLQuantumError(Exception):
    """Base exception for all HLQuantum errors."""


class BackendError(HLQuantumError):
    """Raised when a backend encounters an execution error."""


class CircuitValidationError(HLQuantumError):
    """Raised when a circuit fails backend validation."""


class BackendNotAvailableError(HLQuantumError):
    """Raised when a requested backend is not installed or reachable."""
