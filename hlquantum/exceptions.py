"""Custom exception hierarchy."""


class HLQuantumError(Exception):
    """Base exception for all HLQuantum errors."""


class BackendError(HLQuantumError):
    """Raised on backend execution error."""


class CircuitValidationError(HLQuantumError):
    """Raised on circuit validation failure."""


class BackendNotAvailableError(HLQuantumError):
    """Raised when backend is unavailable."""

