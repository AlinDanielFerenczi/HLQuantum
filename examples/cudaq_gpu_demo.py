"""
cudaq_gpu_demo.py
~~~~~~~~~~~~~~~~~

Demonstrates running the Bernstein-Vazirani algorithm on the CUDA-Q backend
with GPU-accelerated state-vector simulation.

Requirements
------------
    pip install cudaq

CUDA-Q target selection (handled automatically by GPUConfig):
  - ``nvidia``      – single GPU, FP32 (default when GPU is enabled)
  - ``nvidia-fp64`` – single GPU, FP64
  - ``nvidia-mqpu`` – multi-GPU
  - ``default``     – CPU-only (used as fallback here)
"""

import sys
import logging

from hlquantum.algorithms import find_hidden_pattern
from hlquantum.backends.cudaq_backend import CudaQBackend
from hlquantum.gpu import GPUConfig, GPUPrecision

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Choose a secret bitstring for the Bernstein-Vazirani algorithm
#     The algorithm should return exactly this string in a single query.
# ──────────────────────────────────────────────────────────────────────────────
SECRET = "10110"
print(f"Secret bitstring  : {SECRET}  ({len(SECRET)} qubits)")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Build the circuit
# ──────────────────────────────────────────────────────────────────────────────
circuit = find_hidden_pattern(SECRET)
print(f"Circuit           : {circuit.num_qubits} qubits, {len(circuit.gates)} gates")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Configure the GPU backend
#
#     GPUConfig(enabled=True)         → single GPU, FP32, CUDA-Q target "nvidia"
#     GPUConfig(enabled=True,
#               precision=FP64)       → CUDA-Q target "nvidia-fp64"
#     GPUConfig(enabled=True,
#               multi_gpu=True)       → CUDA-Q target "nvidia-mqpu"
#     GPUConfig(enabled=False)        → CUDA-Q target "default" (CPU)
#
#     If no physical GPU is present the script falls back to CPU automatically.
# ──────────────────────────────────────────────────────────────────────────────
try:
    gpu_cfg = GPUConfig(
        enabled=True,
        precision=GPUPrecision.FP32,   # use "nvidia" CUDA-Q target
        device_ids=[0],                # use GPU 0
    )
    backend = CudaQBackend(gpu_config=gpu_cfg)
    print(f"Backend           : {backend.name}")
    print(f"GPU config        : {gpu_cfg}")
except Exception as exc:
    print(f"GPU initialisation failed ({exc}), falling back to CPU backend.")
    backend = CudaQBackend(gpu_config=GPUConfig(enabled=False))
    print(f"Backend           : {backend.name}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Run
# ──────────────────────────────────────────────────────────────────────────────
SHOTS = 1024
print(f"\nRunning {SHOTS} shots …")

try:
    result = backend.run(circuit, shots=SHOTS)
except Exception as exc:
    # CUDA-Q may not be installed; give a helpful message and exit cleanly.
    print(
        f"\n[ERROR] Could not execute on CUDA-Q backend: {exc}\n"
        "Make sure CUDA-Q is installed:  pip install cudaq\n"
        "See https://nvidia.github.io/cuda-quantum for details."
    )
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Inspect results
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nBackend used      : {result.backend_name}")
print(f"Shots executed    : {result.shots}")

# The BV circuit measures only the n input qubits; the most probable outcome
# should be the secret bitstring with near-certainty.
top = result.most_probable
top_count = result.counts.get(top, 0)
top_prob = result.probabilities.get(top, 0.0)

print(f"\nMost probable state : |{top}⟩  —  {top_count}/{SHOTS} shots  ({top_prob:.1%})")
print(f"Expected secret     : |{SECRET}⟩")

# Show the full counts histogram (sorted by count, descending)
print("\nFull counts histogram:")
for state, count in sorted(result.counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count * 40 // SHOTS)
    print(f"  |{state}⟩  {count:5d}  {bar}")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Verify
# ──────────────────────────────────────────────────────────────────────────────
# The BV algorithm is deterministic: the measured input-qubit bits equal SECRET.
# We compare the top result against SECRET (ignoring any ancilla bit if present).
recovered = top[:len(SECRET)]  # in case ancilla is included in the bitstring
if recovered == SECRET:
    print(f"\n✓  Algorithm succeeded: recovered secret '{recovered}' correctly.")
else:
    print(f"\n✗  Unexpected result: got '{recovered}', expected '{SECRET}'.")
