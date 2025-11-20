# 🚀 AI Integration Quick Start Guide

This guide will get you started with the AI-powered exercise classification feature for Rectify.

## 📋 Prerequisites

- Python 3.10+
- Node.js 16+
- FlexTail sensor (for real data collection)
- ~2GB disk space for ML dependencies

---

## Step 1: Install Dependencies

### Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Transformers (Hugging Face library)
- scikit-learn (data processing)
- numpy, pandas (data manipulation)

**Note:** PyTorch installation may take 5-10 minutes depending on your system.

---

## Step 2: Collect Training Data

### Option A: Use Sample Synthetic Data (For Testing)

```bash
cd backend
python ml/prepare_dataset.py
```

This creates synthetic sensor data for 6 exercise types. **Replace this with real data for production.**

### Option B: Collect Real FlexTail Data

1. **Record exercises using your FlexTail sensor**
   - Wear the FlexTail sensor
   - Perform each exercise (squat, deadlift, plank, etc.)
   - Record for at least 30 seconds per exercise
   - Repeat 5-10 times per exercise type

2. **Organize your recordings:**

```
backend/data/recordings/
├── squat/
│   ├── recording1.csv
│   ├── recording2.csv
│   └── recording3.rsf
├── deadlift/
│   ├── recording1.csv
│   └── recording2.csv
├── plank/
│   ├── recording1.csv
│   └── recording2.csv
├── pushup/
│   ├── recording1.csv
│   └── recording2.csv
├── row/
│   └── recording1.csv
└── burpee/
    └── recording1.csv
```

**CSV Format:**
```csv
timestamp,lumbarAngle,sagittal,lateral,twist,acceleration
0.00,0.25,0.10,-0.05,0.02,0.45
0.02,0.26,0.11,-0.04,0.01,0.48
...
```

3. **Process the dataset:**

```bash
python ml/prepare_dataset.py
```

This will:
- Load all recordings
- Extract time windows (3-second windows)
- Normalize the data
- Split into train/validation/test sets
- Save to `data/processed/`

---

## Step 3: Train Your Model

### Quick Training (Default Settings)

```bash
cd backend
python ml/train_model.py
```

### Advanced Training (Custom Parameters)

```bash
# LSTM model (better accuracy)
python ml/train_model.py --model lstm --epochs 100 --batch-size 32 --lr 0.001

# CNN model (faster inference)
python ml/train_model.py --model cnn --epochs 50 --batch-size 64 --lr 0.001
```

**Training Parameters:**
- `--model`: Model architecture (`lstm` or `cnn`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--data-dir`: Path to processed data (default: `data/processed`)

**Training Output:**
```
=== Training Started ===
Epochs: 50
Batch size: 32
Learning rate: 0.001

Epoch 1/50 | Train Loss: 1.7534 Acc: 35.23% | Val Loss: 1.5123 Acc: 42.10% | LR: 0.001000
Epoch 2/50 | Train Loss: 1.3421 Acc: 52.45% | Val Loss: 1.2234 Acc: 58.34% | LR: 0.000995
...
Epoch 50/50 | Train Loss: 0.1234 Acc: 95.67% | Val Loss: 0.2345 Acc: 92.34% | LR: 0.000050

=== Training Complete ===
Best validation accuracy: 92.34% (Epoch 48)
```

**Training Time:**
- CPU: 10-30 minutes
- GPU: 2-5 minutes

**Output Files:**
- `backend/ml/models/trained/best_model.pth` - Trained model
- `backend/ml/models/trained/metadata.json` - Model configuration
- `confusion_matrix.png` - Classification performance
- `training_curves.png` - Loss and accuracy plots

---

## Step 4: Run the Application

### Start Backend

```bash
cd backend
python app.py
```

Expected output:
```
⚠ Using dummy classifier (no trained model)  # Before training
✓ AI classifier initialized                   # After training
✓ Model loaded successfully
  Type: LSTM
  Classes: ['squat', 'deadlift', 'plank', 'pushup', 'row', 'burpee']
  Window size: 150
  Device: cpu

Starting FlexTail Sensor Web Interface...
Server running on http://0.0.0.0:5000
```

### Start Frontend

```bash
cd ..  # Back to root directory
node server.js
```

Expected output:
```
Server running on http://localhost:4000
```

---

## Step 5: Use AI Classification

### In the Web Interface

1. **Open browser:** Navigate to `http://localhost:4000`

2. **Select exercise type** from the landing page

3. **Connect FlexTail sensor:**
   - Enter MAC address (optional)
   - Select hardware version (5.0 or 6.0)
   - Click "Connect"

4. **Enable AI Classification:**
   - Look for the "🤖 AI Exercise Classification" panel
   - Click "Enable AI" button
   - Wait for buffer to fill (150 samples = 3 seconds at 50Hz)

5. **Start exercising:**
   - Begin your exercise
   - Watch real-time AI predictions appear
   - See confidence scores and probabilities

### Understanding the UI

**AI Panel Components:**
```
🤖 AI Exercise Classification
├── Status Indicator: Shows if AI is active
├── Enable/Disable Button: Toggle AI classification
├── Data Buffer: Progress bar showing data collection
├── Prediction: Current exercise prediction
├── Confidence: Model's confidence (0-100%)
└── Probabilities: All class probabilities
```

**Confidence Colors:**
- 🟢 **Green (>70%):** High confidence - reliable prediction
- 🟡 **Yellow (40-70%):** Medium confidence - reasonable prediction
- 🔴 **Red (<40%):** Low confidence - uncertain prediction

---

## 📊 Model Performance Tips

### Improving Accuracy

1. **Collect More Data:**
   - Record 10+ samples per exercise
   - Include different people, speeds, and variations
   - Ensure balanced dataset (equal samples per class)

2. **Data Quality:**
   - Perform exercises with correct form
   - Maintain consistent sensor placement
   - Avoid transitional movements between exercises

3. **Training Adjustments:**
   - Increase epochs for better convergence
   - Try different architectures (LSTM vs CNN)
   - Adjust learning rate if training is unstable

4. **Window Size:**
   - Larger windows (200-300) for complex exercises
   - Smaller windows (100-150) for fast movements
   - Adjust in `ml/prepare_dataset.py`

### Expected Performance

**With Synthetic Data (Demo):**
- Random predictions (~16% accuracy for 6 classes)
- Used for testing UI and integration

**With Real Data (Minimum):**
- 5 recordings/exercise: ~60-70% accuracy
- 10 recordings/exercise: ~75-85% accuracy
- 20+ recordings/exercise: ~85-95% accuracy

**Professional Dataset:**
- 50+ participants
- 10+ recordings/exercise/person
- 95%+ accuracy achievable

---

## 🔧 Troubleshooting

### "Model not found" Error

**Problem:** Backend shows "⚠ Using dummy classifier"

**Solution:**
1. Train a model first: `python ml/train_model.py`
2. Verify model exists: `ls -la backend/ml/models/trained/best_model.pth`
3. Check file permissions

### Low Accuracy

**Problem:** Model predictions are incorrect

**Solution:**
1. Collect more training data (10+ samples per exercise)
2. Verify data quality (check CSV files)
3. Train for more epochs: `python ml/train_model.py --epochs 100`
4. Try different architecture: `--model cnn` or `--model lstm`

### Slow Inference

**Problem:** AI predictions are laggy

**Solution:**
1. Use CNN model: `python ml/train_model.py --model cnn`
2. Reduce window size in `ml/inference.py`
3. Use GPU if available

### Buffer Not Filling

**Problem:** AI buffer stays at 0%

**Solution:**
1. Ensure sensor is connected and streaming
2. Check backend console for errors
3. Verify AI is enabled in UI
4. Refresh browser and reconnect

### Import Errors

**Problem:** `ImportError: No module named 'torch'`

**Solution:**
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Next Steps

### Deploy to Hugging Face Hub

1. **Create Hugging Face account:** https://huggingface.co/join

2. **Install CLI:**
```bash
pip install huggingface_hub
huggingface-cli login
```

3. **Upload model:**
```python
from huggingface_hub import HfApi, create_repo

# Create repository
repo_name = "your-username/rectify-exercise-classifier"
create_repo(repo_name, private=False)

# Upload model
api = HfApi()
api.upload_folder(
    folder_path="./backend/ml/models/trained",
    repo_id=repo_name,
    repo_type="model"
)
```

4. **Use hosted model:**
Update `backend/ml/inference.py` to download from Hub:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/rectify-exercise-classifier",
    filename="best_model.pth"
)
```

### Production Deployment

1. **Optimize model:**
   - Use model quantization for faster inference
   - Convert to ONNX for cross-platform support
   - Deploy on GPU servers for scale

2. **Cloud hosting:**
   - Backend: AWS EC2, Google Cloud Run, Azure
   - Model: Hugging Face Inference API
   - Frontend: Netlify, Vercel, AWS S3

3. **Monitoring:**
   - Track prediction accuracy over time
   - Collect user feedback on predictions
   - Retrain periodically with new data

---

## 📚 Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Time Series Classification](https://huggingface.co/blog/time-series-transformers)
- [Full Guide](./HUGGINGFACE_GUIDE.md)

---

## 🆘 Getting Help

**Issues?** Check the troubleshooting section above.

**Questions?** See the detailed guide: [docs/HUGGINGFACE_GUIDE.md](./HUGGINGFACE_GUIDE.md)

**Bugs?** Open an issue on GitHub with:
- Error messages
- Backend logs
- Steps to reproduce
