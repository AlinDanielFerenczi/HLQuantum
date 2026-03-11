# Cirq Backend Configuration

The `CirqBackend` relies on Google's `cirq` and `qsim` to simulate quantum systems.

## Installation

Ensure you have the required extra installed:
```bash
pip install hlquantum[cirq]
```
This installs `cirq` and `qsimcirq`.

## Local Simulators

By default, the `CirqBackend` will attempt to use Google's fast open-source C++ quantum simulator `qsim`. If this is unavailable, it gracefully falls back to the native `cirq.Simulator`.

```python
from hlquantum.backends import CirqBackend
import hlquantum as hlq

backend = CirqBackend()
result = hlq.run(circuit, backend=backend, shots=1000)
```

## Running Google Cloud Services (Quantum Computing Service)

Cirq interfaces natively with Google's Quantum Computing Service hardware backends. To run on this endpoint, ensure you have:

1. **Google Cloud Project** with the appropriate quantum hardware capabilities enabled.
2. **Google Application Credentials**. Follow Google Cloud documentation to set `GOOGLE_APPLICATION_CREDENTIALS` on your workstation or server environment.

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/your/quantum-google-sa-key.json"
```

In python, initialize an active Google Cloud API Engine engine and construct a device wrapper representing the physical Google Sycamore hardware or cloud API simulators:

```python
import cirq_google
from hlquantum.backends import CirqBackend

# Obtain the Google Quantum Computing Engine with specific project
engine = cirq_google.Engine(project_id='my-google-quantum-project')

# Request physical processor from the platform
sycamore_processor = engine.get_processor('haselnut') # Example

# Wrap it in standard HLQuantum string parameters
backend = CirqBackend(backend=sycamore_processor)
```

## Hardware and Device Simulation

The backend also enables you to directly supply a mocked generic hardware target such as `cirq_google.Sycamore` to simulate native architectures directly in `CirqBackend` deployments:

```python
import cirq_google
from hlquantum.backends import CirqBackend

backend = CirqBackend(backend=cirq_google.Sycamore)
```
