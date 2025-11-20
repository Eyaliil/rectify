# 🤗 Hugging Face Integration Guide for Rectify

## Table of Contents
1. [Understanding Hugging Face Ecosystem](#understanding-hugging-face-ecosystem)
2. [Custom Model Creation](#custom-model-creation)
3. [Training Your Model](#training-your-model)
4. [Deploying to Hugging Face Hub](#deploying-to-hugging-face-hub)
5. [Integration with Web App](#integration-with-web-app)
6. [Alternative Approaches](#alternative-approaches)

---

## Understanding Hugging Face Ecosystem

### What is Hugging Face?

Hugging Face is an AI platform that provides:
- **Pre-trained models** for NLP, computer vision, audio, and more
- **Transformers library** for easy model usage
- **Model Hub** for sharing and hosting models
- **Datasets library** for accessing thousands of datasets
- **Inference API** for deploying models without infrastructure

### Key Concepts for Beginners

#### 1. **Transformers Library**
The main Python library for working with models:
```python
from transformers import AutoModel, AutoTokenizer

# Load a pre-trained model
model = AutoModel.from_pretrained("model-name")
```

#### 2. **Model Types**
- **Sequence Classification**: Categorizing text/sequences (e.g., exercise classification)
- **Token Classification**: Labeling individual tokens (e.g., movement phase detection)
- **Time Series**: Analyzing temporal data (e.g., sensor readings over time)

#### 3. **Model Hub**
A repository where you can:
- Browse 400,000+ pre-trained models
- Upload your custom models
- Version control your models with git
- Share models publicly or privately

#### 4. **Inference API**
- Free tier: Limited requests per month
- Pro tier: $9/month for higher limits
- Serverless deployment (no GPU management needed)

---

## Custom Model Creation

### For Rectify: Exercise Classification from Sensor Data

**Our Goal:** Create a model that classifies exercises (squat, deadlift, plank, pushup) from FlexTail sensor readings.

### Data Format

**Input Features** (from FlexTail sensor):
```python
{
    'lumbarAngle': float,      # Lumbar spine angle (radians)
    'sagittal': float,         # Forward/backward lean (radians)
    'lateral': float,          # Left/right lean (radians)
    'twist': float,            # Rotation (radians)
    'acceleration': float,     # Movement magnitude
}
```

**Output Classes:**
```python
['squat', 'deadlift', 'plank', 'pushup', 'row', 'burpee']
```

### Recommended Model Architecture

For time-series sensor data, you have several options:

#### Option 1: **Simple Sequential Model** (Easiest)
- Input: Time window of sensor readings (e.g., 100 timesteps × 5 features)
- Architecture: LSTM/GRU layers → Dense layers → Softmax
- Best for: Getting started quickly

#### Option 2: **Transformer-based** (Most Powerful)
- Input: Sequence of sensor embeddings
- Architecture: Time Series Transformer
- Best for: Complex patterns, highest accuracy

#### Option 3: **CNN-based** (Efficient)
- Input: Sensor readings as 1D signal
- Architecture: Conv1D layers → Pooling → Dense
- Best for: Fast inference, mobile deployment

---

## Training Your Model

### Step 1: Prepare Your Dataset

Create a dataset from your sensor recordings:

```python
# backend/ml/prepare_dataset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset():
    """
    Collect sensor data for each exercise type.
    Each sample is a time window (e.g., 3 seconds at 50Hz = 150 timesteps)
    """
    data = []
    labels = []

    # Example: Load from RSF files or CSV
    for exercise in ['squat', 'deadlift', 'plank', 'pushup']:
        # Load sensor recordings for this exercise
        recordings = load_exercise_data(exercise)

        for recording in recordings:
            # Extract windows
            windows = extract_windows(recording, window_size=150, stride=50)
            data.extend(windows)
            labels.extend([exercise] * len(windows))

    return train_test_split(data, labels, test_size=0.2)

def extract_windows(data, window_size=150, stride=50):
    """Extract sliding windows from sensor data."""
    windows = []
    for i in range(0, len(data) - window_size, stride):
        window = data[i:i+window_size]
        # Normalize each feature
        window = (window - window.mean(axis=0)) / (window.std(axis=0) + 1e-8)
        windows.append(window)
    return windows
```

### Step 2: Choose Your Approach

#### **Approach A: Fine-tune a Pre-trained Model** (Recommended)

Use a time-series transformer from Hugging Face:

```python
# backend/ml/train_model.py
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from torch.utils.data import Dataset

class SensorDataset(Dataset):
    def __init__(self, data, labels, label2id):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor([label2id[l] for l in labels])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_values': self.data[idx],
            'labels': self.labels[idx]
        }

def train_exercise_classifier():
    # Load data
    X_train, X_val, y_train, y_val = create_dataset()

    # Define label mappings
    label2id = {'squat': 0, 'deadlift': 1, 'plank': 2, 'pushup': 3}
    id2label = {v: k for k, v in label2id.items()}

    # Create datasets
    train_dataset = SensorDataset(X_train, y_train, label2id)
    val_dataset = SensorDataset(X_val, y_val, label2id)

    # Initialize model
    # Note: We'll create a custom model since there's no pre-trained for this
    from models.sensor_classifier import SensorClassifier
    model = SensorClassifier(
        input_size=5,           # 5 sensor features
        hidden_size=128,
        num_layers=2,
        num_classes=len(label2id)
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train!
    trainer.train()

    # Save the model
    model.save_pretrained("./exercise_classifier")

    return model, label2id, id2label
```

#### **Approach B: Train from Scratch** (More Control)

Create a custom PyTorch model:

```python
# backend/ml/models/sensor_classifier.py
import torch
import torch.nn as nn

class SensorClassifier(nn.Module):
    """LSTM-based classifier for exercise recognition."""

    def __init__(self, input_size=5, hidden_size=128, num_layers=2, num_classes=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=4,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, features)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size*2)

        # Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        pooled = attn_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits
```

### Step 3: Training Loop (if not using Trainer)

```python
def train_model(model, train_loader, val_loader, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_values'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    return model
```

---

## Deploying to Hugging Face Hub

### Step 1: Install Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 2: Prepare Model Card

Create `README.md` in your model directory:

```markdown
---
language: en
tags:
- exercise-classification
- time-series
- sensor-data
license: mit
datasets:
- rectify-flextail-exercises
metrics:
- accuracy
- f1
---

# Exercise Classification Model

This model classifies exercises (squat, deadlift, plank, pushup) from FlexTail sensor data.

## Model Details

- **Architecture**: Bidirectional LSTM with Multi-head Attention
- **Input**: Time-series sensor data (5 features × 150 timesteps)
- **Output**: Exercise class probabilities

## Usage

\`\`\`python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("your-username/exercise-classifier")
# ... inference code
\`\`\`

## Training Data

- FlexTail sensor recordings from 50 participants
- 6 exercise types with ~500 samples each
- 3-second windows at 50Hz sampling rate
```

### Step 3: Upload to Hub

```python
from huggingface_hub import HfApi, create_repo

# Create repository
repo_name = "your-username/exercise-classifier"
create_repo(repo_name, private=False)

# Upload model
api = HfApi()
api.upload_folder(
    folder_path="./exercise_classifier",
    repo_id=repo_name,
    repo_type="model"
)
```

---

## Integration with Web App

### Backend Integration (Flask)

Update `backend/app.py`:

```python
# backend/app.py
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import torch
import numpy as np
from ml.models.sensor_classifier import SensorClassifier

# Initialize model
model = SensorClassifier(input_size=5, hidden_size=128, num_layers=2, num_classes=4)
model.load_state_dict(torch.load('ml/models/best_model.pth'))
model.eval()

# Label mappings
id2label = {0: 'squat', 1: 'deadlift', 2: 'plank', 3: 'pushup'}

# Sliding window buffer
sensor_buffer = []
WINDOW_SIZE = 150  # 3 seconds at 50Hz

@socketio.on('sensor_data')
def handle_sensor_data(data):
    """Receive sensor data and perform real-time classification."""
    global sensor_buffer

    # Extract features
    features = [
        data['lumbarAngle'],
        data['sagittal'],
        data['lateral'],
        data['twist'],
        data['acceleration']
    ]

    # Add to buffer
    sensor_buffer.append(features)

    # Keep only last WINDOW_SIZE samples
    if len(sensor_buffer) > WINDOW_SIZE:
        sensor_buffer.pop(0)

    # Classify when we have enough data
    if len(sensor_buffer) == WINDOW_SIZE:
        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor([sensor_buffer])

            # Normalize
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

            # Inference
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            # Get prediction
            predicted_id = probs.argmax(dim=1).item()
            confidence = probs[0, predicted_id].item()
            exercise = id2label[predicted_id]

            # Send result to frontend
            emit('classification_result', {
                'exercise': exercise,
                'confidence': confidence,
                'probabilities': {
                    id2label[i]: probs[0, i].item()
                    for i in range(len(id2label))
                }
            })
```

### Frontend Integration

Update `public/main.js`:

```javascript
// public/main.js

class FlexTailSocketBridge {
    constructor() {
        this.socket = io('http://localhost:5000');
        this.setupListeners();
    }

    setupListeners() {
        // Listen for classification results from AI model
        this.socket.on('classification_result', (data) => {
            console.log('AI Classification:', data);

            // Update UI with AI prediction
            this.updateExerciseDisplay(data.exercise, data.confidence);
            this.updateProbabilities(data.probabilities);
        });

        // ... existing listeners
    }

    handleMeasurement(data) {
        // Send sensor data to backend for classification
        this.socket.emit('sensor_data', data);

        // Also update visualization
        if (window.app && window.app.exerciseAnalyzer) {
            window.app.exerciseAnalyzer.analyzeMeasurement(data);
        }
    }

    updateExerciseDisplay(exercise, confidence) {
        const display = document.getElementById('detected-exercise');
        if (display) {
            display.innerHTML = `
                <div class="ai-prediction">
                    <h3>${exercise.toUpperCase()}</h3>
                    <p>Confidence: ${(confidence * 100).toFixed(1)}%</p>
                    <div class="confidence-bar">
                        <div class="fill" style="width: ${confidence * 100}%"></div>
                    </div>
                </div>
            `;
        }
    }

    updateProbabilities(probs) {
        const container = document.getElementById('class-probabilities');
        if (container) {
            container.innerHTML = Object.entries(probs)
                .map(([exercise, prob]) => `
                    <div class="prob-item">
                        <span>${exercise}</span>
                        <span>${(prob * 100).toFixed(1)}%</span>
                    </div>
                `)
                .join('');
        }
    }
}
```

---

## Alternative Approaches

### Option 1: Hugging Face Inference API (No Local Model)

**Pros**: No model hosting, serverless, automatic scaling
**Cons**: Requires internet, API costs

```python
# backend/app.py
import requests

HF_API_TOKEN = "your_token_here"
MODEL_URL = "https://api-inference.huggingface.co/models/your-username/exercise-classifier"

def classify_with_api(sensor_data):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(MODEL_URL, headers=headers, json={"inputs": sensor_data})
    return response.json()
```

### Option 2: TensorFlow.js (Browser-based)

**Pros**: No backend needed, instant inference
**Cons**: Limited model size, browser performance

```javascript
// public/aiModel.js
import * as tf from '@tensorflow/tfjs';

class ExerciseClassifier {
    async loadModel() {
        this.model = await tf.loadLayersModel('/models/exercise_model/model.json');
    }

    async predict(sensorWindow) {
        const input = tf.tensor3d([sensorWindow]);
        const prediction = await this.model.predict(input);
        const probs = await prediction.data();
        return probs;
    }
}
```

### Option 3: ONNX Runtime (Cross-platform)

**Pros**: Fast inference, works everywhere
**Cons**: Need to convert model

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": sensor_data})
```

---

## Next Steps

1. **Collect Training Data**: Record FlexTail sensor data for each exercise
2. **Train Your Model**: Use the provided scripts
3. **Test Locally**: Verify model accuracy before deployment
4. **Deploy**: Upload to Hugging Face Hub or use local inference
5. **Integrate**: Update Flask backend and frontend code
6. **Monitor**: Track prediction accuracy and retrain as needed

---

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Time Series Classification](https://huggingface.co/blog/time-series-transformers)
