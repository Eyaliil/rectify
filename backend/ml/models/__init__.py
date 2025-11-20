"""
Neural network models for exercise classification
"""

from .sensor_classifier import SensorClassifier, LightweightSensorClassifier

__all__ = [
    'SensorClassifier',
    'LightweightSensorClassifier'
]
