# IonQ Backend Configuration

The `IonQBackend` allows execution of your HLQuantum workloads natively on IonQ trapped-ion quantum computers or simulators over the IonQ Cloud API.

## Installation

Ensure you have the required extra installed:
```bash
pip install hlquantum[ionq]
```
This installs the `qiskit-ionq` provider plugin that HLQuantum relies on.

## API Key Authentication

Running on IonQ infrastructure requires an API key, which you can obtain from the [IonQ Quantum Cloud Console](https://cloud.ionq.com).

There are two primary ways to provide your setup with the key:

1. **Environment Variables**:
   You can place your credentials in the `IONQ_API_KEY` or `QISKIT_IONQ_API_TOKEN` environment variables. This is the recommended approach for production deployments.

   ```bash
   export IONQ_API_KEY="your-ionq-api-secret"
   ```

2. **Constructor Argument**:
   You can pass the token directly through Python during instantiation.

   ```python
   from hlquantum.backends import IonQBackend
   
   backend = IonQBackend(api_key="your-ionq-api-secret")
   ```

## Backends Available

- `ionq_simulator`: Ideal quantum simulator hosted in the IonQ cloud. It supports statevector retrieval. (Default setting).
- `ionq_qpu`: Executes circuits against physical IonQ Trapped-Ion Quantum Hardware (Aria, Forte). Only count metrics are available as it's physical hardware.

```python
from hlquantum.backends import IonQBackend
import hlquantum as hlq

# Instantiate an ionq_qpu backend instance
backend = IonQBackend(backend_name="ionq_qpu")
result = hlq.run(circuit, backend=backend, shots=1000)
```
