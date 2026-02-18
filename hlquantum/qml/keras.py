"""
hlquantum.qml.keras
~~~~~~~~~~~~~~~~~~

Keras/TensorFlow integration for hybrid quantum-classical machine learning.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

import numpy as np
from hlquantum.circuit import Circuit, Parameter


def _require_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError("TensorFlow is required for QuantumLayer. Install it with: pip install tensorflow")


def _require_tfq():
    try:
        import tensorflow_quantum as tfq
        return tfq
    except ImportError:
        return None


class QuantumLayer:
    """A Keras-compatible layer wrapping a hlquantum circuit.
    
    If tensorflow-quantum is installed, it leverages TFQ's high-performance 
    differentiable simulator. Otherwise, it uses a custom gradient implementation 
    via the Parameter Shift Rule.
    """

    def __new__(cls, circuit: Circuit, **kwargs):
        tfq = _require_tfq()
        if tfq:
            return _TFQQuantumLayer(circuit, **kwargs)
        else:
            return _HlQuantumLayer(circuit, **kwargs)


class _HlQuantumLayer:
    """Fallback implementation using Parameter Shift Rule."""
    
    def __init__(self, circuit: Circuit, **kwargs):
        self.tf = _require_tf()
        self.circuit = circuit
        self.params = circuit.parameters
        
        # Initialize Keras Layer logic via composition or subclassing 
        # (Using composition for cleaner fallback logic)
        self.layer = self._build_layer(**kwargs)

    def _build_layer(self, **kwargs):
        tf = self.tf
        circuit = self.circuit
        params = self.params
        from hlquantum.algorithms.grad import parameter_shift_gradient

        class InnerLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.q_weights = self.add_weight(
                    name="q_weights",
                    shape=(len(params),),
                    initializer="random_normal",
                    trainable=True
                )

            @tf.custom_gradient
            def call(self, inputs):
                # hlquantum doesn't support batching natively in runner yet, 
                # so we do a simple loop or single call for now.
                
                def grad(dy):
                    # dy is the derivative from subsequent layers
                    weights_val = self.q_weights.numpy()
                    p_values = {params[i].name: weights_val[i] for i in range(len(params))}
                    
                    # Compute gradients via Parameter Shift
                    q_grads = parameter_shift_gradient(circuit, p_values)
                    
                    # Convert to tensor
                    grad_tensor = tf.constant([q_grads[p.name] for p in params], dtype=tf.float32)
                    
                    # Apply chain rule: dy * dq_weights
                    # Since dy might be (batch, 1) and grad_tensor is (n_params,)
                    # We might need to reduce dy or broadcast. 
                    # For simplicity, assume output is a scalar expectation.
                    return None, dy * grad_tensor

                # Forward pass
                weights_val = self.q_weights.numpy()
                p_values = {params[i].name: weights_val[i] for i in range(len(params))}
                
                from hlquantum.runner import run
                qc = circuit.bind_parameters(p_values)
                if not any(g.name == "mz" for g in qc.gates):
                    qc.measure_all()
                
                # Note: This is slow because it's synchronous runner inside TF
                res = run(qc)
                exp_val = res.expectation_value()
                
                return tf.constant([exp_val], dtype=tf.float32), grad

        return InnerLayer(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class _TFQQuantumLayer:
    """High-performance implementation using TensorFlow Quantum."""

    def __init__(self, circuit: Circuit, **kwargs):
        self.tf = _require_tf()
        self.tfq = _require_tfq()
        self.circuit = circuit
        
        # Convert hlquantum circuit to Cirq circuit for TFQ
        from hlquantum.backends.cirq_backend import CirqBackend
        cirq = CirqBackend._require_cirq()
        self.cirq_circuit, _, _ = CirqBackend._translate(circuit, cirq)
        
        self.layer = self._build_layer(**kwargs)

    def _build_layer(self, **kwargs):
        tf = self.tf
        tfq = self.tfq
        
        # TFQ requires PQC (Parameterized Quantum Circuit)
        # We need to identify symbols in the Cirq circuit
        import sympy
        symbols = [sympy.Symbol(p.name) for p in self.circuit.parameters]
        
        # Simple Expectation layer
        # For a full implementation, we'd need to handle operators (Hamiltonians)
        # Defaulting to Z expectation on all qubits
        import cirq
        qubits = cirq.LineQubit.range(self.circuit.num_qubits)
        operator = cirq.Z(qubits[0]) # Placeholder
        
        return tfq.layers.PQC(self.cirq_circuit, operator)

    def __call__(self, *args, **kwargs):
        return self.layer(*args, **kwargs)
