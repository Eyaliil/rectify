#!/bin/bash
# Quick training script for Rectify AI

set -e

echo "========================================="
echo "  Rectify AI - Quick Training"
echo "========================================="
echo ""

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies (this may take 5-10 minutes)..."
    pip install -r requirements.txt
fi

echo ""
echo "========================================="
echo "  Step 1: Checking Dataset"
echo "========================================="
echo ""

# Count files
total_files=0
for dir in data/recordings/*/; do
    if [ -d "$dir" ]; then
        exercise=$(basename "$dir")
        count=$(find "$dir" -type f \( -name "*.csv" -o -name "*.rsf" \) | wc -l | tr -d ' ')
        if [ "$count" -gt 0 ]; then
            echo "  $exercise: $count files"
            total_files=$((total_files + count))
        fi
    fi
done

if [ "$total_files" -eq 0 ]; then
    echo ""
    echo "❌ No data files found!"
    echo ""
    echo "Please add your recordings to:"
    echo "  backend/data/recordings/<exercise_name>/"
    echo ""
    echo "Supported formats: .csv, .rsf"
    exit 1
fi

echo ""
echo "✓ Found $total_files recording files"

echo ""
echo "========================================="
echo "  Step 2: Processing Dataset"
echo "========================================="
echo ""

python ml/prepare_dataset.py

if [ ! -f "data/processed/X_train.npy" ]; then
    echo ""
    echo "❌ Dataset processing failed!"
    exit 1
fi

echo ""
echo "✓ Dataset processed successfully"

echo ""
echo "========================================="
echo "  Step 3: Training Model"
echo "========================================="
echo ""
echo "This will take 10-30 minutes on CPU..."
echo ""

# Default to LSTM with 50 epochs, can be overridden
MODEL_TYPE=${1:-lstm}
EPOCHS=${2:-50}

python ml/train_model.py --model "$MODEL_TYPE" --epochs "$EPOCHS"

if [ ! -f "ml/models/trained/best_model.pth" ]; then
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "  ✅ Training Complete!"
echo "========================================="
echo ""
echo "Model saved to: backend/ml/models/trained/best_model.pth"
echo ""
echo "Next steps:"
echo "1. Start backend:  cd backend && python app.py"
echo "2. Start frontend: node server.js"
echo "3. Open browser:   http://localhost:4000"
echo "4. Enable AI and test your model!"
echo ""
