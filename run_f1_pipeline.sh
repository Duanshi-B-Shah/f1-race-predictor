#!/bin/bash

# =============================
# F1 Race Predictor Pipeline
# Optimized for Your Structure
# =============================

# Exit on any error
set -e

# --------------------------
# 1. Navigate to Project Root
# --------------------------
echo "🚀 Starting F1 Race Predictor Pipeline..."
cd "$(dirname "$0")"  # Ensures execution from f1-race-predictor/

# --------------------------
# 2. Setup Virtual Environment
# --------------------------
echo "🧰 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies from your requirements.txt
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# --------------------------
# 3. Run Pipeline Stages
# --------------------------

echo "📥 Step 1/4: Fetching F1 data from OpenF1 API..."
python src/data_fetcher.py

echo "🏋️ Step 2/4: Training the model..."
python src/train.py

echo "📈 Step 3/4: Generating evaluation plots..."
python src/evaluate.py

echo "🌟 Step 4/4: Launching Streamlit demo app..."
streamlit run src/app.py