# Error Mitigation API Reference

HLQuantum includes a pluggable error-mitigation pipeline. Each technique is a subclass of `MitigationMethod` and can be composed together via `apply_mitigation()`.

## Built-in Methods

| Class                 | Strategy                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| `ThresholdMitigation` | Discards bitstrings whose probability falls below a configurable threshold, treating them as noise. |
| `ReadoutMitigation`   | Placeholder for readout-error correction (extensible).                                              |

## Usage

```python
import hlquantum as hlq
from hlquantum.mitigation import ThresholdMitigation, apply_mitigation

# Apply during execution
result = hlq.run(circuit, mitigation=ThresholdMitigation(threshold=0.01))

# Or apply after the fact
raw = hlq.run(circuit)
clean = apply_mitigation(raw, [ThresholdMitigation(threshold=0.01)])
```

## Writing a Custom Mitigation Method

Subclass `MitigationMethod` and implement the `apply(result)` method:

```python
from hlquantum.mitigation import MitigationMethod

class MyMitigation(MitigationMethod):
    def apply(self, result):
        # Return a new or modified ExecutionResult
        ...
```

---

::: hlquantum.mitigation
