# Qiskit Backend Configuration

The `QiskitBackend` enables you to connect to the extensive ecosystem supported by IBM Quantum and Qiskit. This backend supports both executing circuits on the high-performance local Aer simulators and real IBM Quantum hardware endpoints.

## Installation

Ensure you have the required extra installed:
```bash
pip install hlquantum[qiskit]
```
This installs `qiskit` and `qiskit-aer`.

## Local Simulator

By default, without passing a specific device or target, the `QiskitBackend` will use Qiskit's `AerSimulator`. This simulator runs locally and natively supports statevector measurements and noise models.

```python
from hlquantum.backends import QiskitBackend
import hlquantum as hlq

backend = QiskitBackend()
result = hlq.run(circuit, backend=backend, shots=1000)
```

## Hardware and IBM Cloud Executions

Connecting to real superconducting hardware requires an IBM Quantum Platform API token.

Get your token from the [IBM Quantum Dashboard](https://quantum.ibm.com/). Once acquired, you can save your token via the `qiskit-ibm-runtime` package which persists the configuration locally.

```bash
pip install qiskit-ibm-runtime
```

Save your authentication details locally using Python:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_IBM_QUANTUM_TOKEN", set_as_default=True)
```

Once saved, HLQuantum can seamlessly connect to hardware devices:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
from hlquantum.backends import QiskitBackend
import hlquantum as hlq

# Instantiate the service with your saved IBM credentials
service = QiskitRuntimeService()

# Automatically fetch the least busy hardware
ibm_backend = service.least_busy(min_num_qubits=2)

# Wrap it in standard HLQuantum Backend syntax
backend = QiskitBackend(backend=ibm_backend)
result = hlq.run(circuit, backend=backend)
```

## GPU Hardware Acceleration

Since IBM's Aer natively supports NVIDIA GPUs, HLQuantum bridges GPU enablement seamlessly. Set `GPUConfig(enabled=True)` across your instance. Ensure your CUDA drivers are appropriately installed for Qiskit Aer GPU implementations to activate.
