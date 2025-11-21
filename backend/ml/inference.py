"""
Real-time Inference Service for Exercise Classification

This module provides real-time exercise classification from FlexTail sensor data.
It maintains a sliding window buffer and performs inference when sufficient data is collected.
"""

import torch
import numpy as np
from pathlib import Path
import json
from collections import deque
import threading


class ExerciseClassifierInference:
    """
    Real-time exercise classifier with sliding window buffer.

    This class manages a buffer of sensor readings and performs inference
    when enough data is available.
    """

    def __init__(
        self,
        model_path='models/trained/model_classification.pth',
        window_size=150,
        min_confidence=0.3,
        device=None
    ):
        """
        Args:
            model_path (str): Path to trained model checkpoint
            window_size (int): Number of timesteps for inference window
            min_confidence (float): Minimum confidence threshold
            device (torch.device): Device for inference (CPU/GPU)
        """
        self.model_path = Path(model_path)
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.device = device or torch.device('cpu')

        # Sensor data buffer (thread-safe)
        self.buffer = deque(maxlen=window_size)
        self.buffer_lock = threading.Lock()

        # Model and metadata
        self.model = None
        self.metadata = None
        self.id2label = {}
        self.is_loaded = False

        # Load model
        self.load_model()

    def load_model(self):
        """Load the trained model and metadata."""
        if not self.model_path.exists():
            print(f"Warning: Model not found at {self.model_path}")
            print("Please train a model first using train_model.py")
            return False

        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract metadata
            self.metadata = checkpoint['metadata']
            self.id2label = {int(k): v for k, v in self.metadata['id2label'].items()}

            # Create model
            model_type = checkpoint['model_type']

            if model_type == 'lstm':
                from ml.models.sensor_classifier import SensorClassifier
                self.model = SensorClassifier(
                    input_size=self.metadata['input_size'],
                    hidden_size=128,
                    num_layers=2,
                    num_classes=self.metadata['num_classes'],
                    dropout=0.3
                )
            elif model_type == 'cnn':
                from ml.models.sensor_classifier import LightweightSensorClassifier
                self.model = LightweightSensorClassifier(
                    input_size=self.metadata['input_size'],
                    num_classes=self.metadata['num_classes'],
                    dropout=0.3
                )

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True

            print(f"✓ Model loaded successfully")
            print(f"  Type: {model_type.upper()}")
            print(f"  Classes: {list(self.id2label.values())}")
            print(f"  Window size: {self.window_size}")
            print(f"  Device: {self.device}")

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def extract_features(self, measurement_data):
        """
        Extract feature vector from measurement data.

        Args:
            measurement_data (dict): Raw measurement from FlexTail sensor

        Returns:
            list: Feature vector [lumbarAngle, sagittal, lateral, twist, acceleration]
        """
        # Extract ALL 8 features in SAME ORDER as training data
        # These are in RADIANS (matching the training data format)
        features = [
            measurement_data.get('lumbarAngle', 0.0),      # angles.bend (radians)
            measurement_data.get('twist', 0.0),            # angles.twist (radians)
            measurement_data.get('lateral', 0.0),          # measurement.lateral_flexion (radians)
            measurement_data.get('sagittal', 0.0),         # measurement.sagittal_flexion (radians)
            measurement_data.get('lateralApprox', 0.0),    # measurement.calc_lateral_approx()
            measurement_data.get('sagittalApprox', 0.0),   # measurement.calc_sagittal_approx()
            measurement_data.get('acceleration', 0.0),     # magnitude of acc [x,y,z]
            measurement_data.get('gyro', 0.0)              # magnitude of gyro [x,y,z]
        ]

        return features

    def add_measurement(self, measurement_data):
        """
        Add a new measurement to the buffer.

        Args:
            measurement_data (dict): Measurement data from sensor

        Returns:
            bool: True if buffer is ready for inference
        """
        features = self.extract_features(measurement_data)

        with self.buffer_lock:
            self.buffer.append(features)
            return len(self.buffer) >= self.window_size

    def predict(self):
        """
        Perform inference on the current buffer.

        Returns:
            dict: Prediction results with exercise, confidence, and probabilities
                  Returns None if model not loaded or buffer not ready
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'exercise': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }

        with self.buffer_lock:
            if len(self.buffer) < self.window_size:
                return None

            # Convert buffer to numpy array
            window = np.array(list(self.buffer))

        # Normalize window
        window_mean = window.mean(axis=0, keepdims=True)
        window_std = window.std(axis=0, keepdims=True) + 1e-8
        window_normalized = (window - window_mean) / window_std

        # Convert to tensor
        x = torch.FloatTensor(window_normalized).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        # Extract results
        predicted_id = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_id].item()
        exercise = self.id2label[predicted_id]

        # Get all probabilities
        probabilities = {
            self.id2label[i]: probs[0, i].item()
            for i in range(len(self.id2label))
        }

        # Check confidence threshold
        if confidence < self.min_confidence:
            exercise = 'unknown'

        return {
            'exercise': exercise,
            'confidence': confidence,
            'probabilities': probabilities,
            'buffer_size': len(self.buffer)
        }

    def reset_buffer(self):
        """Clear the measurement buffer."""
        with self.buffer_lock:
            self.buffer.clear()

    def get_buffer_status(self):
        """Get current buffer status."""
        with self.buffer_lock:
            return {
                'current_size': len(self.buffer),
                'required_size': self.window_size,
                'ready': len(self.buffer) >= self.window_size,
                'percentage': int(100 * len(self.buffer) / self.window_size)
            }


class DummyClassifier:
    """
    Dummy classifier for testing when no trained model is available.

    This provides random predictions to demonstrate the integration.
    """

    def __init__(self, window_size=150):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.buffer_lock = threading.Lock()
        self.is_loaded = True

        self.id2label = {
            0: 'squat',
            1: 'deadlift',
            2: 'plank',
            3: 'pushup',
            4: 'row',
            5: 'burpee'
        }

        print("⚠ Using dummy classifier (no trained model)")
        print("  Train a model using train_model.py for real predictions")

    def add_measurement(self, measurement_data):
        """Add measurement to buffer."""
        with self.buffer_lock:
            self.buffer.append(measurement_data)
            return len(self.buffer) >= self.window_size

    def predict(self):
        """Return random prediction."""
        with self.buffer_lock:
            if len(self.buffer) < self.window_size:
                return None

        # Generate random prediction for demo
        import random
        predicted_id = random.randint(0, len(self.id2label) - 1)
        exercise = self.id2label[predicted_id]

        # Generate random probabilities
        probs = np.random.dirichlet(np.ones(len(self.id2label)))
        probs = sorted(probs, reverse=True)

        probabilities = {
            self.id2label[i]: float(probs[i])
            for i in range(len(self.id2label))
        }

        return {
            'exercise': exercise,
            'confidence': float(probs[0]),
            'probabilities': probabilities,
            'buffer_size': len(self.buffer),
            'is_dummy': True
        }

    def reset_buffer(self):
        """Clear buffer."""
        with self.buffer_lock:
            self.buffer.clear()

    def get_buffer_status(self):
        """Get buffer status."""
        with self.buffer_lock:
            return {
                'current_size': len(self.buffer),
                'required_size': self.window_size,
                'ready': len(self.buffer) >= self.window_size,
                'percentage': int(100 * len(self.buffer) / self.window_size)
            }


def create_classifier(model_path='models/trained/model_classification.pth', use_dummy=False):
    """
    Factory function to create a classifier.

    Args:
        model_path (str): Path to trained model
        use_dummy (bool): Force use of dummy classifier

    Returns:
        ExerciseClassifierInference or DummyClassifier
    """
    if use_dummy or not Path(model_path).exists():
        return DummyClassifier()
    else:
        return ExerciseClassifierInference(model_path=model_path)


if __name__ == "__main__":
    # Test the inference service
    print("=== Testing Inference Service ===\n")

    classifier = create_classifier()

    # Simulate sensor data
    for i in range(200):
        fake_measurement = {
            'timestamp': i * 0.02,
            'bend': np.random.randn() * 10,
            'pitch': np.random.randn() * 5,
            'roll': np.random.randn() * 5
        }

        ready = classifier.add_measurement(fake_measurement)

        if ready and i % 50 == 0:
            result = classifier.predict()
            if result:
                print(f"\nPrediction at timestep {i}:")
                print(f"  Exercise: {result['exercise']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Top 3 probabilities:")
                sorted_probs = sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for exercise, prob in sorted_probs[:3]:
                    print(f"    {exercise}: {prob:.2%}")

    print("\n=== Test Complete ===")
