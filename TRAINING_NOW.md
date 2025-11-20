# 🎯 Ready to Train? Here's What To Do!

You have your dataset ready - great! Here's the exact process:

---

## Option 1: Automated Training (Recommended)

### Just run this one command:

```bash
./quick_train.sh
```

This will:
1. ✅ Check your dataset
2. ✅ Process the data
3. ✅ Train the model
4. ✅ Save it for use

**Time:** 10-30 minutes (depending on your CPU)

---

## Option 2: Manual Step-by-Step

### Step 1: Put Your Data in the Right Place

Copy your recordings into these folders **by exercise type**:

```bash
# Example: If you have squat recordings
cp /path/to/your/squat/*.csv backend/data/recordings/squat/

# For each exercise type you have
cp /path/to/your/deadlift/*.csv backend/data/recordings/deadlift/
cp /path/to/your/plank/*.csv backend/data/recordings/plank/
# ... etc
```

**❓ Where is your dataset?** Tell me the path and I'll help!

### Step 2: Process Your Dataset

```bash
cd backend
python ml/prepare_dataset.py
```

**What this does:**
- Loads all your recordings
- Extracts 3-second windows
- Normalizes the data
- Splits into train/val/test (70%/15%/15%)

**You'll see output like:**
```
Found 3 exercise types: ['squat', 'deadlift', 'plank']
Processing squat...
  recording1.csv: 25 windows
  recording2.csv: 28 windows
...
Total windows: 450
Train: 315 | Val: 68 | Test: 67
```

### Step 3: Train Your Model

**Basic training (LSTM, 50 epochs):**
```bash
python ml/train_model.py
```

**Fast training (CNN, 30 epochs):**
```bash
python ml/train_model.py --model cnn --epochs 30
```

**High accuracy (LSTM, 100 epochs):**
```bash
python ml/train_model.py --model lstm --epochs 100
```

**You'll see progress:**
```
Epoch 1/50 | Train Loss: 1.75 Acc: 35% | Val Loss: 1.51 Acc: 42%
Epoch 2/50 | Train Loss: 1.34 Acc: 52% | Val Loss: 1.22 Acc: 58%
...
Epoch 50/50 | Train Loss: 0.12 Acc: 96% | Val Loss: 0.23 Acc: 92%

✅ Best validation accuracy: 92% (Epoch 48)
```

### Step 4: Test It!

```bash
# Terminal 1: Start backend
cd backend
python app.py
```

```bash
# Terminal 2: Start frontend
cd /Users/donghyun/All/aihack/web
node server.js
```

Open `http://localhost:4000` and click **"Enable AI"**! 🚀

---

## 📊 What To Expect With Small Dataset

**If you have 3-5 recordings per exercise:**
- Expected accuracy: 50-70%
- It will work, but might confuse similar exercises
- Recommendation: Try to get 5-10 recordings per exercise

**If you have 5-10 recordings per exercise:**
- Expected accuracy: 70-85%
- Good enough for testing and demos
- Should work well for distinct exercises

**If you have 10+ recordings per exercise:**
- Expected accuracy: 85-95%
- Production-ready!

---

## 💡 Tips for Small Datasets

### 1. **Data Augmentation** (built into training)
The training script automatically:
- Uses different time windows from same recording
- Normalizes data to handle variations
- Applies dropout to prevent overfitting

### 2. **Train Longer**
With small data, training longer helps:
```bash
python ml/train_model.py --epochs 100
```

### 3. **Try Both Models**
```bash
# LSTM - better for small datasets
python ml/train_model.py --model lstm --epochs 100

# CNN - faster, might work better if exercises are very different
python ml/train_model.py --model cnn --epochs 50
```

### 4. **Check Your Data Quality**
Make sure:
- ✅ Sensor was worn correctly
- ✅ Exercises performed properly
- ✅ No transitional movements
- ✅ Consistent sampling rate

---

## 🐛 Troubleshooting

### "No data files found"
```bash
# Check what you have:
ls -la backend/data/recordings/*/

# You should see .csv or .rsf files in subdirectories
```

### "Dataset processing failed"
- Check CSV format: `timestamp,lumbarAngle,sagittal,lateral,twist,acceleration`
- Make sure files aren't empty
- Verify column names match exactly

### "Low accuracy"
- Train longer: `--epochs 100`
- Collect more data (even 2-3 more recordings helps!)
- Try different model: `--model cnn` or `--model lstm`

### "Training takes forever"
- Expected: 10-30 minutes on CPU
- To speed up: use `--model cnn` (2x faster)
- Or reduce epochs: `--epochs 30`

---

## 🎓 Understanding the Training Output

```
Epoch 10/50 | Train Loss: 0.85 Acc: 68% | Val Loss: 0.92 Acc: 65%
```

- **Train Loss** (lower = better): How well model fits training data
- **Train Acc** (higher = better): Accuracy on training data
- **Val Loss** (lower = better): How well model generalizes
- **Val Acc** (higher = better): Accuracy on unseen data

**Good signs:**
- ✅ Both losses decreasing
- ✅ Val accuracy increasing
- ✅ Val accuracy close to train accuracy

**Warning signs:**
- ⚠️ Train acc high (>95%) but val acc low (<70%) = Overfitting
- ⚠️ Both accuracies stuck at ~20-30% = Need more epochs or better data

---

## ❓ Still Stuck?

**Tell me:**
1. Where is your dataset located?
2. What file format? (CSV or RSF)
3. How many recordings per exercise?

I'll help you get it working! 🚀
