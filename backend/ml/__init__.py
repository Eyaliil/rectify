"""
Machine Learning module for Rectify

This module provides AI-powered exercise classification using FlexTail sensor data.
"""

from .inference import ExerciseClassifierInference, DummyClassifier, create_classifier

__all__ = [
    'ExerciseClassifierInference',
    'DummyClassifier',
    'create_classifier'
]
