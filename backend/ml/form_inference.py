"""
Real-Time Form Analysis Inference

Combines exercise classification + form quality analysis + AI coaching.
"""

import torch
import numpy as np
from collections import deque
from pathlib import Path
import json

from ml.models.form_classifier import MultiTaskClassifier, AICoach


class FormAnalyzer:
    """
    Real-time exercise form analyzer with AI coaching.

    Features:
    - Exercise classification
    - Form quality detection (good/bad)
    - Real-time coaching feedback
    - Rep counting and session summaries
    """

    def __init__(
        self,
        model_path='models/trained/form_model.pth',
        window_size=150,
        min_confidence=0.5
    ):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.buffer = deque(maxlen=window_size)

        self.model = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False

        # AI Coach
        self.coach = AICoach()

        # Session tracking
        self.rep_history = []
        self.current_exercise = None

        # Load model
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained multi-task model."""
        model_path = Path(model_path)
        metadata_path = model_path.parent / "form_metadata.json"

        if not model_path.exists():
            print(f"Form model not found at {model_path}")
            print("Please train the form model first: python ml/train_form_model.py")
            return

        try:
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                # Try loading from checkpoint
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.metadata = checkpoint.get('metadata', {})

            # Create model
            self.model = MultiTaskClassifier(
                input_size=self.metadata.get('input_size', 8),
                hidden_size=128,
                num_layers=2,
                num_exercises=self.metadata.get('num_exercises', 5),
                num_form_classes=2,
                dropout=0.0  # No dropout during inference
            ).to(self.device)

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.is_loaded = True
            print(f"Form model loaded successfully!")
            print(f"  Exercises: {list(self.metadata.get('exercise_label2id', {}).keys())}")
            print(f"  Form classes: good, bad")

        except Exception as e:
            print(f"Error loading form model: {e}")
            self.is_loaded = False

    def extract_features(self, measurement_data):
        """Extract features from measurement."""
        features = [
            measurement_data.get('lumbarAngle', 0.0),
            measurement_data.get('twist', 0.0),
            measurement_data.get('lateral', 0.0),
            measurement_data.get('sagittal', 0.0),
            measurement_data.get('lateralApprox', 0.0),
            measurement_data.get('sagittalApprox', 0.0),
            measurement_data.get('acceleration', 0.0),
            measurement_data.get('gyro', 0.0)
        ]
        return features

    def add_measurement(self, measurement_data):
        """Add measurement to buffer."""
        features = self.extract_features(measurement_data)
        self.buffer.append(features)
        return len(self.buffer) >= self.window_size

    def get_buffer_status(self):
        """Get current buffer status."""
        return {
            'current': len(self.buffer),
            'required': self.window_size,
            'ready': len(self.buffer) >= self.window_size,
            'percentage': (len(self.buffer) / self.window_size) * 100
        }

    def predict(self):
        """
        Predict exercise type, form quality, and get coaching feedback.

        Returns:
            dict: Complete analysis with exercise, form, and coaching
        """
        if not self.is_loaded or len(self.buffer) < self.window_size:
            return None

        try:
            # Prepare input
            window = np.array(list(self.buffer))

            # Normalize
            window_mean = window.mean(axis=0, keepdims=True)
            window_std = window.std(axis=0, keepdims=True) + 1e-8
            window_normalized = (window - window_mean) / window_std

            # To tensor
            x = torch.FloatTensor(window_normalized).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                ex_logits, form_logits = self.model(x)

                ex_probs = torch.softmax(ex_logits, dim=-1)[0]
                form_probs = torch.softmax(form_logits, dim=-1)[0]

                ex_pred = ex_probs.argmax().item()
                form_pred = form_probs.argmax().item()

                ex_confidence = ex_probs[ex_pred].item()
                form_confidence = form_probs[form_pred].item()

            # Get labels
            exercise = self.metadata['exercise_id2label'].get(str(ex_pred), 'unknown')
            form_quality = 'good' if form_pred == 0 else 'bad'

            # Get coaching feedback
            latest_features = {
                'lumbarAngle': self.buffer[-1][0],
                'twist': self.buffer[-1][1],
                'lateral': self.buffer[-1][2],
                'sagittal': self.buffer[-1][3]
            }

            coaching = self.coach.get_feedback(
                exercise=exercise,
                form_quality=form_quality,
                form_confidence=form_confidence,
                features=latest_features
            )

            # Track for session summary
            self.rep_history.append({
                'exercise': exercise,
                'form_quality': form_quality,
                'confidence': form_confidence
            })

            # Build response
            result = {
                'exercise': exercise,
                'exercise_confidence': ex_confidence,
                'exercise_probabilities': {
                    self.metadata['exercise_id2label'].get(str(i), f'class_{i}'): ex_probs[i].item()
                    for i in range(len(ex_probs))
                },
                'form_quality': form_quality,
                'form_confidence': form_confidence,
                'form_probabilities': {
                    'good': form_probs[0].item(),
                    'bad': form_probs[1].item()
                },
                'coaching': coaching,
                'session_stats': self.coach.get_rep_summary(self.rep_history)
            }

            return result

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def reset_session(self):
        """Reset session tracking."""
        self.rep_history = []
        self.buffer.clear()

    def get_session_summary(self):
        """Get complete session summary."""
        return self.coach.get_rep_summary(self.rep_history)


# Global instance
form_analyzer = None


def create_form_analyzer(model_path='models/trained/form_model.pth'):
    """Create global form analyzer instance."""
    global form_analyzer
    form_analyzer = FormAnalyzer(model_path=model_path)
    return form_analyzer


def get_form_analyzer():
    """Get global form analyzer instance."""
    global form_analyzer
    if form_analyzer is None:
        form_analyzer = FormAnalyzer()
    return form_analyzer
