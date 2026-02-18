"""
Demo of High-Level Quantum-Classical Machine Learning using hlquantum's predefined models.
"""

import numpy as np
from hlquantum.qml import create_quantum_classifier

def main():
    print("--- HLQuantum Predefined Model Demo ---")
    
    n_qubits = 4
    n_classes = 3
    
    try:
        import tensorflow as tf
        print(f"\nBuilding hybrid classifier for {n_qubits} qubits and {n_classes} classes...")
        
        # Use the predefined high-level model
        model = create_quantum_classifier(n_qubits=n_qubits, n_classes=n_classes)
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        
        # Create dummy data
        # Inputs: features corresponding to n_qubits
        # Labels: integer class indices
        x_train = np.random.rand(5, n_qubits).astype(np.float32)
        y_train = np.random.randint(0, n_classes, size=(5,))
        
        print("\nRunning training on dummy data...")
        # Note: This will use the Parameter Shift Rule internally if TFQ is not installed
        history = model.fit(x_train, y_train, epochs=2, verbose=1)
        print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
        
    except ImportError as e:
        print(f"\nDemo skipped: {e}")
        print("To run this demo, install tensorflow: pip install tensorflow")
    except Exception as e:
        print(f"\nAn error occurred during model execution: {e}")

if __name__ == "__main__":
    main()
