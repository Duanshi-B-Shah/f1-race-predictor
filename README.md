# F1 Race Position Predictor

An ML project that predicts Formula 1 race finishing positions using historical data, qualifying results, and circuit characteristics.

## Project Overview

This project demonstrates:
- Data collection from public APIs (OpenF1 API — https://openf1.org)
- Feature engineering from raw motorsport data
- Model training and evaluation (XGBoost vs baseline)
- Interactive prediction demo (Streamlit)

## Setup

```bash
cd f1-race-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Fetch Data
```bash
python src/data_fetcher.py
```
Downloads F1 race data (2023–2026 seasons) from the OpenF1 API.

### 2. Build Features & Train Model
```bash
python src/train.py
```
Runs feature engineering, trains the model, and saves it to `models/`.

### 3. Evaluate
```bash
python src/evaluate.py
```
Generates evaluation metrics and plots in `outputs/`.

### 4. Demo App
```bash
streamlit run src/app.py
```
Interactive UI to predict race finishing positions.

## Features Used

| Feature | Description |
|---|---|
| `grid_position` | Qualifying/grid position |
| `driver_avg_finish` | Driver's rolling average finish (last 5 races) |
| `constructor_avg_finish` | Constructor's rolling average finish (last 5 races) |
| `circuit_driver_avg` | Driver's historical average at this circuit |
| `driver_dnf_rate` | Driver's DNF rate over recent races |
| `grid_position_change` | Typical positions gained/lost from grid |

## Project Structure

```
f1-race-predictor/
├── src/
│   ├── data_fetcher.py    # OpenF1 API data collection
│   ├── features.py        # Feature engineering pipeline
│   ├── train.py           # Model training
│   ├── evaluate.py        # Evaluation & plots
│   └── app.py             # Streamlit demo
├── data/                  # Raw + processed data (gitignored)
├── models/                # Saved models (gitignored)
├── outputs/               # Evaluation plots
├── requirements.txt
└── README.md
```
