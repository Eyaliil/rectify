"""
Exercise Classification Model using LSTM with Attention

This model classifies exercises from FlexTail sensor data using a bidirectional
LSTM with multi-head attention mechanism.
"""

import torch
import torch.nn as nn


class SensorClassifier(nn.Module):
    """
    LSTM-based classifier for exercise recognition from time-series sensor data.

    Architecture:
    - Bidirectional LSTM for temporal feature extraction
    - Multi-head attention for focusing on important timesteps
    - Fully connected layers for classification

    Args:
        input_size (int): Number of input features (default: 5)
            - lumbarAngle, sagittal, lateral, twist, acceleration
        hidden_size (int): LSTM hidden dimension (default: 128)
        num_layers (int): Number of LSTM layers (default: 2)
        num_classes (int): Number of exercise classes (default: 6)
        dropout (float): Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_size=5,
        hidden_size=128,
        num_layers=2,
        num_classes=6,
        dropout=0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        # LSTM processing
        # Output shape: (batch, seq_len, hidden_size*2)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Self-attention
        # Output shape: (batch, seq_len, hidden_size*2)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling across time dimension
        # Output shape: (batch, hidden_size*2)
        pooled = attn_out.mean(dim=1)

        # Classification
        # Output shape: (batch, num_classes)
        logits = self.classifier(pooled)

        return logits

    def predict_proba(self, x):
        """
        Get class probabilities.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            probs: Probability distribution of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


class LightweightSensorClassifier(nn.Module):
    """
    Lightweight CNN-based classifier for faster inference.

    This model uses 1D convolutions instead of LSTM for reduced computational cost.
    Suitable for real-time inference on CPU or mobile devices.
    """

    def __init__(
        self,
        input_size=5,
        num_classes=6,
        dropout=0.3
    ):
        super().__init__()

        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Conv block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Conv block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        # Transpose for Conv1d: (batch, features, sequence_length)
        x = x.transpose(1, 2)

        # Convolutions
        features = self.conv_layers(x)

        # Classification
        logits = self.classifier(features)

        return logits


if __name__ == "__main__":
    # Test the model
    model = SensorClassifier()

    # Create dummy input: (batch=4, sequence=150, features=5)
    x = torch.randn(4, 150, 5)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test lightweight model
    light_model = LightweightSensorClassifier()
    output_light = light_model(x)
    print(f"\nLightweight model output shape: {output_light.shape}")
    print(f"Lightweight model parameters: {sum(p.numel() for p in light_model.parameters()):,}")
