# Transpiler API Reference

The transpiler optimises circuits before execution by running a configurable sequence of transformation passes.

## Built-in Passes

| Pass                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `RemoveRedundantGates` | Cancels adjacent self-inverse gates (e.g. H-H, X-X). |
| `MergeRotations`       | Merges consecutive rotations around the same axis.   |

## Usage

```python
from hlquantum.transpiler import transpile, Transpiler
from hlquantum.transpiler import RemoveRedundantGates, MergeRotations

# Quick — use the default pass set
optimised = transpile(circuit)

# Custom — pick your own passes
t = Transpiler(passes=[RemoveRedundantGates(), MergeRotations()])
optimised = t.run(circuit)
```

## Writing a Custom Pass

Subclass `TranspilationPass` and implement the `run(circuit)` method:

```python
from hlquantum.transpiler import TranspilationPass

class MyPass(TranspilationPass):
    def run(self, circuit):
        # Return a new, optimised Circuit
        ...
```

---

::: hlquantum.transpiler
