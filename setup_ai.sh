#!/bin/bash
# Setup script for Rectify AI Integration

set -e

echo "========================================="
echo "  Rectify AI Integration Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Check Node.js
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠ Node.js not found (frontend will not work)${NC}"
else
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js $NODE_VERSION detected${NC}"
fi

echo ""
echo "========================================="
echo "  Installing Backend Dependencies"
echo "========================================="
echo ""

cd backend

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing Python packages (this may take 5-10 minutes)..."
pip install -r requirements.txt

echo -e "${GREEN}✓ All dependencies installed${NC}"

echo ""
echo "========================================="
echo "  Setting up ML Directory Structure"
echo "========================================="
echo ""

# Create necessary directories
mkdir -p ml/models/trained
mkdir -p data/recordings/squat
mkdir -p data/recordings/deadlift
mkdir -p data/recordings/plank
mkdir -p data/recordings/pushup
mkdir -p data/recordings/row
mkdir -p data/recordings/burpee
mkdir -p data/processed

echo -e "${GREEN}✓ Directory structure created${NC}"

echo ""
echo "========================================="
echo "  Creating Sample Data (Optional)"
echo "========================================="
echo ""

read -p "Generate synthetic sample data for testing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating sample data..."
    python ml/prepare_dataset.py
    echo -e "${GREEN}✓ Sample data created${NC}"
else
    echo -e "${YELLOW}⚠ Skipping sample data generation${NC}"
    echo "To collect real data, place recordings in:"
    echo "  backend/data/recordings/<exercise_name>/"
fi

echo ""
echo "========================================="
echo "  Training Model (Optional)"
echo "========================================="
echo ""

if [ -d "data/processed" ] && [ "$(ls -A data/processed)" ]; then
    read -p "Train model now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Training model (this may take 10-30 minutes on CPU)..."
        python ml/train_model.py --epochs 50
        echo -e "${GREEN}✓ Model training complete${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping model training${NC}"
        echo "To train later, run:"
        echo "  cd backend"
        echo "  python ml/train_model.py"
    fi
else
    echo -e "${YELLOW}⚠ No training data found${NC}"
    echo "Prepare dataset first by running:"
    echo "  cd backend"
    echo "  python ml/prepare_dataset.py"
fi

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the backend server:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "2. Start the frontend server (in a new terminal):"
echo "   node server.js"
echo ""
echo "3. Open your browser:"
echo "   http://localhost:4000"
echo ""
echo "4. See the Quick Start Guide for more info:"
echo "   docs/AI_QUICK_START.md"
echo ""
echo -e "${GREEN}Happy training! 🚀${NC}"
echo ""
