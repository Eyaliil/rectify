"""
Multi-Task Exercise + Form Quality Classifier

Predicts both:
1. Exercise type (squat, deadlift, etc.)
2. Form quality (good vs bad)
"""

import torch
import torch.nn as nn


class MultiTaskClassifier(nn.Module):
    """
    LSTM-based multi-task classifier for exercise recognition and form analysis.

    Architecture:
    - Shared LSTM encoder (learns general movement patterns)
    - Exercise classification head
    - Form quality classification head
    """

    def __init__(
        self,
        input_size=8,
        hidden_size=128,
        num_layers=2,
        num_exercises=5,
        num_form_classes=2,
        dropout=0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Shared encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Exercise classification head
        self.exercise_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_exercises)
        )

        # Form quality classification head
        self.form_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_form_classes)
        )

    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)

        # Global average pooling
        pooled = attn_out.mean(dim=1)

        # Task-specific predictions
        exercise_logits = self.exercise_head(pooled)
        form_logits = self.form_head(pooled)

        return exercise_logits, form_logits

    def predict(self, x):
        """Get predictions with probabilities."""
        self.eval()
        with torch.no_grad():
            exercise_logits, form_logits = self.forward(x)

            exercise_probs = torch.softmax(exercise_logits, dim=-1)
            form_probs = torch.softmax(form_logits, dim=-1)

            exercise_pred = exercise_probs.argmax(dim=-1)
            form_pred = form_probs.argmax(dim=-1)

        return {
            'exercise_pred': exercise_pred,
            'exercise_probs': exercise_probs,
            'form_pred': form_pred,
            'form_probs': form_probs
        }


class AICoach:
    """
    AI Coach that provides real-time feedback based on exercise and form quality.
    """

    def __init__(self):
        # Exercise-specific coaching tips for bad form
        self.bad_form_tips = {
            'dl': [
                "Keep your back straight - avoid rounding your spine",
                "Push through your heels, not your toes",
                "Engage your core throughout the lift",
                "Keep the bar close to your body",
                "Don't let your knees cave inward"
            ],
            'squat': [
                "Keep your chest up and back straight",
                "Push your knees out over your toes",
                "Go deeper - aim for parallel or below",
                "Keep your weight on your heels",
                "Engage your core before descending"
            ],
            'ohp': [
                "Keep your core tight - don't arch your back",
                "Press the bar in a straight line overhead",
                "Tuck your chin slightly as the bar passes",
                "Lock out fully at the top",
                "Keep your wrists straight"
            ],
            'pushup': [
                "Keep your body in a straight line",
                "Don't let your hips sag or pike up",
                "Lower your chest all the way down",
                "Keep your elbows at 45 degrees",
                "Engage your core throughout"
            ],
            'row': [
                "Pull with your elbows, not your hands",
                "Squeeze your shoulder blades together",
                "Keep your back flat, avoid rounding",
                "Control the weight on the way down",
                "Don't use momentum - stay controlled"
            ]
        }

        # Good form encouragement
        self.good_form_tips = {
            'dl': ["Great deadlift form! Keep it up!", "Perfect hip hinge pattern!", "Excellent back position!"],
            'squat': ["Beautiful squat depth!", "Great knee tracking!", "Perfect form - keep going!"],
            'ohp': ["Strong overhead press!", "Great core stability!", "Excellent lockout!"],
            'pushup': ["Perfect pushup form!", "Great body alignment!", "Excellent range of motion!"],
            'row': ["Great rowing technique!", "Perfect back position!", "Excellent scapular control!"]
        }

        # Form metrics explanations
        self.form_metrics = {
            'lumbar_angle': {
                'name': 'Spine Position',
                'good_range': (-0.3, 0.3),
                'tip': 'Keep your spine neutral'
            },
            'lateral_flexion': {
                'name': 'Side Bend',
                'good_range': (-0.1, 0.1),
                'tip': 'Stay balanced, avoid leaning'
            },
            'twist': {
                'name': 'Rotation',
                'good_range': (-0.1, 0.1),
                'tip': 'Keep hips and shoulders aligned'
            }
        }

    def get_feedback(self, exercise, form_quality, form_confidence, features=None):
        """
        Generate coaching feedback based on predictions.

        Args:
            exercise (str): Predicted exercise type
            form_quality (str): 'good' or 'bad'
            form_confidence (float): Confidence score 0-1
            features (dict): Optional sensor features for detailed analysis

        Returns:
            dict: Coaching feedback
        """
        import random

        feedback = {
            'exercise': exercise,
            'form_quality': form_quality,
            'confidence': form_confidence,
            'message': '',
            'tips': [],
            'metrics': {}
        }

        if form_quality == 'bad':
            # Get exercise-specific tips
            tips = self.bad_form_tips.get(exercise, ["Focus on your form"])
            feedback['tips'] = random.sample(tips, min(2, len(tips)))
            feedback['message'] = f"Form needs improvement! {feedback['tips'][0]}"
            feedback['color'] = 'warning'  # Orange/yellow

        else:
            # Good form encouragement
            tips = self.good_form_tips.get(exercise, ["Great form!"])
            feedback['tips'] = [random.choice(tips)]
            feedback['message'] = feedback['tips'][0]
            feedback['color'] = 'success'  # Green

        # Analyze specific metrics if features provided
        if features:
            feedback['metrics'] = self._analyze_metrics(features, exercise)

        return feedback

    def _analyze_metrics(self, features, exercise):
        """Analyze individual sensor metrics."""
        metrics = {}

        lumbar = features.get('lumbarAngle', 0)
        lateral = features.get('lateral', 0)
        twist = features.get('twist', 0)

        # Spine position analysis
        if abs(lumbar) > 0.5:  # Excessive bend
            metrics['spine'] = {
                'status': 'warning',
                'value': f"{abs(lumbar)*57.3:.1f}°",
                'tip': 'Reduce spinal flexion'
            }
        else:
            metrics['spine'] = {
                'status': 'good',
                'value': f"{abs(lumbar)*57.3:.1f}°",
                'tip': 'Good spine position'
            }

        # Lateral balance
        if abs(lateral) > 0.15:
            metrics['balance'] = {
                'status': 'warning',
                'value': f"{abs(lateral)*57.3:.1f}°",
                'tip': 'Keep weight centered'
            }
        else:
            metrics['balance'] = {
                'status': 'good',
                'value': f"{abs(lateral)*57.3:.1f}°",
                'tip': 'Good balance'
            }

        # Rotation check
        if abs(twist) > 0.15:
            metrics['rotation'] = {
                'status': 'warning',
                'value': f"{abs(twist)*57.3:.1f}°",
                'tip': 'Reduce rotation'
            }
        else:
            metrics['rotation'] = {
                'status': 'good',
                'value': f"{abs(twist)*57.3:.1f}°",
                'tip': 'Good alignment'
            }

        return metrics

    def get_rep_summary(self, rep_history):
        """
        Summarize form quality over multiple reps.

        Args:
            rep_history: List of form predictions for each rep

        Returns:
            dict: Summary statistics
        """
        if not rep_history:
            return {'total_reps': 0, 'good_reps': 0, 'form_score': 0}

        good_reps = sum(1 for r in rep_history if r['form_quality'] == 'good')
        total_reps = len(rep_history)
        form_score = (good_reps / total_reps) * 100

        return {
            'total_reps': total_reps,
            'good_reps': good_reps,
            'bad_reps': total_reps - good_reps,
            'form_score': form_score,
            'grade': self._get_grade(form_score)
        }

    def _get_grade(self, score):
        """Convert score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
