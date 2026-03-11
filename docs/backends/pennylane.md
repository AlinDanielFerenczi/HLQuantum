# PennyLane Backend Configuration

The `PennyLaneBackend` enables executing HLQuantum arrays entirely via Xanadu's high-performance PennyLane ecosystem.

## Installation

Ensure you have the required extra installed:
```bash
pip install hlquantum[pennylane]
```
This installs `pennylane`.

## Default Qubits and Local Hardware

Without any configuration parameters, this uses `default.qubit` to generate rapid simulation on CPU processors. When running inside GPU devices, the `"lightning.gpu"` device enables highly optimized execution parameters using cuQuantum integration natively. Note that Xanadu plugin backends also leverage third-party device integrations transparently.

```python
from hlquantum.backends import PennyLaneBackend
import hlquantum as hlq

# Instantiate an optimized lightning backend simulator with GPU integration
backend = PennyLaneBackend(device_name="lightning.gpu")
result = hlq.run(circuit, backend=backend, shots=1000)
```

## PennyLane-Hosted Vendor Integrations

You can optionally run any supported PennyLane plugin with the `device_kwargs`.
Access configurations, like Amazon Braket tasks running natively over PennyLane workflows or Xanadu Photonic Quantum Computers, automatically retrieve environment variables matching the underlying technology, standardizing their execution profiles within `PennyLaneBackend` workflows.

```python
from hlquantum.backends import PennyLaneBackend

# Expose a separate QPU hardware via Xanadu PennyLane plugins.
backend = PennyLaneBackend(
    device_name="braket.aws.qubit",
    device_kwargs={
        "device_arn": "arn:aws:braket:::device/qpu/ionq/Aria-1",
        "s3_destination_folder": ("my-bucket", "prefix")
    }
)
```
When running on Xanadu's cloud platform, the `XANADU_CLOUD_API_KEY` token is required to interface with remote endpoints. Obtain an API key from [Xanadu Cloud](https://cloud.xanadu.ai/).
