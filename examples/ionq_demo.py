"""
ionq_demo.py
~~~~~~~~~~~~

Demonstrates creating and running a simple Bell-state circuit on the
IonQ backend using HLQuantum.

Requirements
------------
    pip install qiskit qiskit-ionq

Authentication
--------------
Supply your IonQ API key in one of three ways:

1. Pass it directly:  ``IonQBackend(api_key="your-key")``
2. Set the env var:   ``IONQ_API_KEY=your-key``
3. Set the env var:   ``QISKIT_IONQ_API_TOKEN=your-key``

Targets
-------
* ``ionq_simulator`` – free cloud simulator (default)
* ``ionq_qpu``       – IonQ Aria / Forte trapped-ion QPU (paid)
"""

import logging

import hlquantum
from hlquantum import kernel
from hlquantum.backends import IonQBackend

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Define a simple Bell-state circuit using the @kernel decorator
# ──────────────────────────────────────────────────────────────────────────────

@kernel(num_qubits=2)
def bell(qc):
    """Create a maximally-entangled Bell state  |Φ⁺⟩ = (|00⟩ + |11⟩) / √2."""
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()


print(f"Circuit: {bell.circuit}")
print(f"  Qubits : {bell.circuit.num_qubits}")
print(f"  Gates  : {len(bell.circuit.gates)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Create the IonQ backend
#
#     By default the backend connects to the free IonQ cloud simulator.
#     Switch to real hardware with:
#         IonQBackend(backend_name="ionq_qpu", api_key="...")
# ──────────────────────────────────────────────────────────────────────────────

backend = IonQBackend(
    backend_name="ionq_simulator",
    # api_key="YOUR_IONQ_API_KEY",    # uncomment or set IONQ_API_KEY env var
)

print(f"Backend: {backend.name}")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Run the circuit
# ──────────────────────────────────────────────────────────────────────────────

SHOTS = 1000
result = hlquantum.run(bell, shots=SHOTS, backend=backend)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Inspect results
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'─' * 40}")
print(f"Shots         : {result.shots}")
print(f"Backend       : {result.backend_name}")
print(f"Counts        : {result.counts}")
print(f"Probabilities : {result.probabilities}")
print(f"Most probable : {result.most_probable}")
print(f"{'─' * 40}")

# For a perfect Bell state we expect roughly 50/50 between |00⟩ and |11⟩.
for state in ("00", "11"):
    pct = result.probabilities.get(state, 0) * 100
    print(f"  |{state}⟩  →  {pct:.1f}%")
