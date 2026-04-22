# F1 Race Predictor — Interview Prep Guide

This document walks through every component of the project so you can confidently explain it in an interview. Each section covers what the code does, why it's built that way, and what questions to expect.

---

## 1. Data Collection (`src/data_fetcher.py`)

### What it does
Fetches race data from the OpenF1 API (https://openf1.org) for seasons 2023–2026. For each race, it pulls:
- **Sessions** — finds all Race-type sessions per year
- **Drivers** — full name, team name, 3-letter acronym per session
- **Starting grid** — grid positions from qualifying
- **Session results** — final positions, DNF/DNS/DSQ flags

The raw JSON is saved per season, then parsed into a flat CSV with one row per driver per race.

### Key design choices
- **Rate limiting**: OpenF1 free tier allows 3 req/s and 30 req/min. We use 2.5s delay between requests to stay safe.
- **Fallback logic**: Some sessions don't have `starting_grid` data (e.g., Sprint races). We fall back to the `position` endpoint and grab the earliest timestamp as the grid order.
- **Retry with backoff**: On 429 (rate limited), we wait progressively longer (5s, 10s, 15s...). On 404, we return empty data gracefully.
- **Schema design**: The CSV includes both machine-readable IDs (`driver_id: "VER"`, `circuit_id: "monza"`) and human-readable names (`driver_name: "Max VERSTAPPEN"`, `circuit_name: "Monza, Italy"`). The model trains on IDs; the UI shows names.

### Pros
- Clean separation of raw data (JSON) and processed data (CSV)
- Resilient to API failures — won't crash on missing data
- Idempotent — can re-run safely

### Cons
- Slow due to rate limiting (~10 min for all seasons)
- No incremental fetch — re-downloads everything each time
- OpenF1 only has data from 2023, limiting training data volume

### Interview questions to expect
- *"Why not cache API responses?"* — We do save raw JSON. Re-running `parse_results_to_dataframe()` alone skips the API entirely.
- *"How would you handle API changes?"* — Version the raw data, add schema validation on the parsed output.
- *"What if you needed more historical data?"* — Could supplement with another source (e.g., f1api.dev or Kaggle datasets) and normalize the schema.

---

## 2. Feature Engineering (`src/features.py`)

### What it does
Takes the flat race results CSV and computes 6 ML features per driver per race:

1. **`grid_position`** — where they start (raw input, strongest signal)
2. **`driver_avg_finish`** — rolling mean of their last 5 finishes
3. **`constructor_avg_finish`** — rolling mean of their team's last 5 finishes
4. **`circuit_driver_avg`** — their historical average at this specific circuit
5. **`driver_dnf_rate`** — proportion of DNFs in last 10 races
6. **`grid_position_change`** — average positions gained/lost from grid over last 5 races

### Key design choices
- **`shift(1)` everywhere** — This is critical. Every rolling feature is shifted by one race so the model never sees the current race's result when making a prediction. Without this, you'd have data leakage and artificially inflated metrics.
- **Rolling windows (5 and 10)** — Short enough to capture recent form, long enough to smooth out noise. 5 races ≈ quarter of a season.
- **`min_periods=1`** — Allows features to be computed even for a driver's first race (uses whatever data is available).
- **NaN handling** — `circuit_driver_avg` falls back to `driver_avg_finish` when a driver hasn't raced at a circuit before. Remaining NaNs filled with column medians.

### Pros
- All features are interpretable — you can explain each one to a non-technical interviewer
- No data leakage by design
- Captures both driver skill and car performance separately

### Cons
- Only 6 features — could add weather, tire strategy, qualifying times
- Rolling windows are fixed — could be hyperparameters
- Circuit-driver average is sparse for new circuits or new drivers
- Doesn't capture race-specific context (wet weather, safety cars, penalties)

### Interview questions to expect
- *"What is data leakage and how did you prevent it?"* — Using future information to predict the past. We prevent it with `shift(1)` on all rolling features and time-based train/test splits.
- *"Why rolling averages instead of all-time averages?"* — F1 performance changes rapidly (car upgrades, driver transfers). Recent form is more predictive than career averages.
- *"What features would you add next?"* — Weather (OpenF1 has it), tire compound/stint data, qualifying lap times, safety car probability per circuit.

---

## 3. Model Training (`src/train.py`)

### What it does
Trains an XGBoost regressor to predict `finish_position` (integer 1–20).

**Training process:**
1. Load processed CSV, run feature engineering
2. Time-based 80/20 split (first 80% of races = train, last 20% = test)
3. Train baseline: predict finish = grid position (MAE ~2.84)
4. Train XGBoost with tuned hyperparameters
5. Run 5-fold time-series cross-validation for robust evaluation
6. Retrain final model on all data, save to disk

### Key design choices
- **Time-based split, not random** — F1 data is sequential. A random split would put 2025 races in training and 2023 races in test, leaking future team/driver performance into the model.
- **XGBoost over other models** — Gradient-boosted trees are the gold standard for tabular data with <10K rows. Neural nets would overfit. Linear regression can't capture non-linear interactions (e.g., a specific driver at a specific circuit).
- **Baseline comparison** — Always compare against a simple baseline. If the model can't beat "finish where you qualify," it's not useful.
- **TimeSeriesSplit for CV** — Standard k-fold would also leak. TimeSeriesSplit ensures each fold only trains on past data and validates on future data.

### Hyperparameters
```
n_estimators=200    # number of boosting rounds
max_depth=6         # tree depth (controls complexity)
learning_rate=0.1   # step size shrinkage
subsample=0.8       # row sampling per tree (reduces overfitting)
colsample_bytree=0.8 # feature sampling per tree
```

### Pros
- Beats the baseline consistently (~0.3 positions better MAE)
- Fast to train (<5 seconds)
- Time-series-aware evaluation prevents optimistic metrics
- Model is small and portable (joblib serialization)

### Cons
- No hyperparameter tuning (grid search / Bayesian optimization)
- No experiment tracking (MLflow, W&B)
- Single model — no ensemble or stacking
- Limited training data (~1800 rows across 4 seasons)

### Interview questions to expect
- *"Why not use a neural network?"* — With ~1800 rows and 6 features, tree-based models dominate. NNs need much more data and offer no advantage on structured tabular data.
- *"How would you tune hyperparameters?"* — Optuna or scikit-learn's RandomizedSearchCV with TimeSeriesSplit as the CV strategy.
- *"What's the MAE mean in practical terms?"* — On average, the prediction is ~2.5 positions off. So if we predict P5, the actual finish is typically P3–P7.
- *"How do you handle model drift?"* — Retrain after each race weekend with new data. Monitor MAE on recent predictions vs actuals.

---

## 4. Evaluation (`src/evaluate.py`)

### What it does
Generates 4 diagnostic plots on the held-out test set:

1. **Feature importance** — which inputs the model relies on most (bar chart)
2. **Predicted vs actual** — scatter plot with perfect-prediction diagonal line
3. **Error distribution** — histogram of prediction errors (should be centered at 0)
4. **MAE by grid group** — accuracy broken down by Top 3 / 4-10 / 11-20 grid positions

### What to look for in the plots
- **Feature importance**: `grid_position` will dominate. If `driver_avg_finish` or `constructor_avg_finish` have meaningful importance, the model is learning beyond just "finish where you start."
- **Predicted vs actual**: Points should cluster around the diagonal. Spread increases for midfield positions (harder to predict).
- **Error distribution**: Should be roughly symmetric around 0. A skew would indicate systematic bias.
- **MAE by grid**: Front-runners (Top 3) are easier to predict. Midfield (4-10) is chaotic. Backmarkers (11-20) are moderately predictable.

### Pros
- Comprehensive evaluation beyond a single number
- Plots are interview-ready — can screenshot for presentations
- Reveals where the model struggles (midfield unpredictability)

### Cons
- No per-driver or per-circuit breakdown
- No comparison against other model types
- Static plots — not interactive

---

## 5. Streamlit App (`src/app.py`)

### What it does
Interactive web UI where you:
1. Select a driver (full name), circuit (full name), and grid position
2. Click "Predict" to get the predicted finish position
3. See an interactive Plotly chart of the driver's race history (grid vs finish over time)
4. Expand a detailed results table with color-coded position changes

### Key design choices
- **Plotly over matplotlib** — Interactive: hover to see circuit name, positions gained. Zoom, pan, export. Much better for a demo.
- **Inverted Y-axis** — P1 at the top, like a real standings board. Intuitive for F1 fans and non-fans alike.
- **Color coding** — Green vertical bars = positions gained, red = positions lost. Gold stars = podium finishes. Makes patterns immediately visible.
- **Full names everywhere** — "Max VERSTAPPEN" not "VER", "Monza, Italy" not "monza". Accessible to people who don't follow F1.

### Pros
- Visually impressive for demos and interviews
- Interactive chart encourages exploration
- Shows both prediction and historical context
- Works on any device with a browser

### Cons
- Features are computed from the full dataset at app load time (not from a feature store)
- No caching of predictions
- Single-race prediction only — no batch or season simulation

---

## 6. Deployment (`Dockerfile`, `deploy/`)

### What it does
- **Dockerfile**: Packages the app, processed data, and trained model into a container
- **CloudFormation**: Creates VPC, ALB, ECS Fargate cluster, security groups — full infrastructure as code
- **deploy.sh**: One-command deployment (build → push to ECR → deploy stack)
- **buildspec.yml**: AWS CodeBuild spec for CI/CD (build Docker image from GitHub)

### Key design choices
- **ECS Fargate over EC2** — No server management. Pay only for running tasks. Scale to zero by setting desired count to 0.
- **ALB in front** — Public DNS, health checks, can add HTTPS later with ACM certificate.
- **Model baked into image** — Simple for a demo project. In production, you'd pull from S3 or a model registry.

### Pros
- Fully reproducible infrastructure (CloudFormation)
- Publicly accessible URL for sharing
- Container ensures consistent environment

### Cons
- No HTTPS (would need ACM certificate + Route53 domain)
- No CI/CD pipeline (manual build trigger)
- Model updates require rebuilding the image
- ~$30/month running cost

---

## Common Interview Questions (Cross-Cutting)

### "Walk me through the project end to end."
"I built an ML pipeline that predicts F1 race finishing positions. Data comes from the OpenF1 API — I fetch race results, grid positions, and driver info for 2023–2026. I engineer 6 features: grid position, rolling driver and team averages, circuit-specific history, DNF rate, and position-change tendency. All features use shift-by-one to prevent data leakage. I train an XGBoost regressor with time-series cross-validation, beating a grid-equals-finish baseline by about 0.3 positions MAE. The model is served through a Streamlit app with interactive Plotly charts, containerized with Docker, and deployable to AWS ECS Fargate."

### "What was the hardest part?"
"Data quality from the API. Some sessions don't have starting grid data, some return 404s, and the rate limits are tight. I had to build fallback logic (using the position endpoint when starting_grid is missing) and progressive retry with backoff for rate limiting."

### "What would you do differently with more time?"
"Three things: (1) Add weather and tire strategy features — OpenF1 has both. (2) Set up MLflow for experiment tracking so I can compare model variants systematically. (3) Build an automated retraining pipeline that triggers after each race weekend and deploys the updated model."

### "How would this scale to production?"
"The model itself is tiny and fast (<1ms inference). For production: store features in a feature store (e.g., Feast or SageMaker Feature Store), serve the model behind an API (FastAPI or SageMaker endpoint), add monitoring for prediction drift, and set up automated retraining with new race data."

### "Why XGBoost and not [other model]?"
"For tabular data with ~1800 rows and 6 features, gradient-boosted trees consistently outperform other approaches. Linear models can't capture non-linear interactions. Neural nets need more data and offer no advantage here. Random forests are competitive but typically slightly worse than boosted trees on structured data. XGBoost also gives feature importance for free, which helps interpretability."
