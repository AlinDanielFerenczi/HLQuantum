"""
hlquantum.qml.models
~~~~~~~~~~~~~~~~~~~

Pre-built high-level QML models.
"""

from __future__ import annotations
from typing import Optional
from hlquantum.circuit import Circuit
from hlquantum.layers.templates import RealAmplitudes
from hlquantum.qml.keras import QuantumLayer


def create_quantum_classifier(n_qubits: int, n_classes: int = 2):
    """Creates a high-level quantum-classical classifier using Keras.
    
    This model uses a RealAmplitudes ansatz as a quantum layer sandwiched 
    between classical dense layers.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required to create a QuantumClassifier.")

    ansatz = RealAmplitudes(n_qubits, reps=2)
    qc = ansatz.build()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_qubits,)),
        tf.keras.layers.Dense(n_qubits, activation='relu'),
        QuantumLayer(qc, name="quantum_embedding"),
        tf.keras.layers.Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    
    return model


def create_hybrid_cnn(input_shape: tuple, n_qubits: int):
    """Creates a hybrid model featuring a Quantum Layer as a 'filter' or feature extractor."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required to create a HybridCNN.")

    ansatz = RealAmplitudes(n_qubits, reps=1)
    qc = ansatz.build()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_qubits), # Bottleneck to quantum layer
        QuantumLayer(qc, name="quantum_bottleneck"),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model
