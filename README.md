---
title: Delivery Time Prediction
emoji: 🍕
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: ui/app.py
pinned: false
---


### 🍕 Delivery Time Prediction
#### A friendly Gradio demo (XGBoost) that predicts delivery time from order characteristics

- Short: XGBoost regression model that estimates delivery time (minutes) from features like distance, weather, traffic, vehicle type and courier experience.
- Purpose: Give quick, explainable delivery-time estimates to support logistics decisions or power a demo UI on Hugging Face Spaces.

#### Dataset (quick facts)
- Shape: 1000 rows × 9 columns
- Train / Test split: 795 / 199
- Converted to Pandas for preprocessing and modeling
- Missing values:
  - Weather: 30
  - Traffic_Level: 30
  - Courier_Experience: 30

- Main columns (used in the model)
  - Order_ID
  - Distance_km
  - Weather
  - Traffic_Level
  - Vehicle_Type
  - Preparation_Time_min
  - Courier_Experience_yrs
  - Delivery_Time_min (target)

- Example preview (first 5 rows — one column omitted for brevity):
  - 522, 7.93, Windy,  Low,  <other_col>, Scooter, 12, 1.0, 43
  - 738, 16.42, Clear, Medium, <other_col>, Bike,    20, 2.0, 84
  - 741, 9.52, Foggy,  Low,  <other_col>, Scooter, 28, 1.0, 59
  - 661, 7.44, Rainy,  Medium, <other_col>, Scooter, 5, 1.0, 37
  - 412, 19.03, Clear, Low,  <other_col>, Bike,    16, 5.0, 68

#### Model & performance
- Algorithm: XGBoost regressor (scikit-learn API)
- CV (k-fold) MAE: 7.15 ± 0.27 minutes
- Final metrics:
  - MAE (train): 3.78 min
  - MAE (test): 6.37 min
  - RMSE (test): 9.55 min
  - R² (test): 0.816

#### How it works (pipeline)
- Data ingestion → convert to Pandas
- Missing-value handling (imputation / simple strategy for categorical/continuous)
- Categorical encoding for Weather, Traffic_Level, Vehicle_Type
- Train/test split (795 / 199)
- XGBoost training with cross-validation (MAE objective)
- Evaluate on hold-out test set and surface results in the Gradio UI

#### What you’ll find in this Space
- Interactive Gradio app (app.py) to:
  - Input feature values (distance, weather, traffic, vehicle, prep time, courier experience)
  - See predicted delivery time (minutes)
  - Optionally show confidence or explanation (if SHAP/feature importances added)
- Model served behind the UI (fast inference, suitable for demo / prototyping)

#### Run locally
- Recommended (example):
  - Create venv: python -m venv .venv && source .venv/bin/activate
  - Install deps: pip install -r requirements.txt
  - Run: python app.py
- Notes:
  - app.py is built with Gradio (sdk_version: 4.0.0)
  - If you include a saved model artifact, name it consistently (e.g., model.pkl) and load with joblib/pickle

#### Inputs & expected ranges (suggested)
- Distance_km: 0.5 – 20.0
- Weather: categorical — Clear, Rainy, Windy, Foggy, ...
- Traffic_Level: categorical — Low, Medium, High
- Vehicle_Type: categorical — Bike, Scooter, Car, ...
- Preparation_Time_min: integer, typically 5 – 30
- Courier_Experience_yrs: float/integer, typically 0 – 9

#### Tech stack
- Model: XGBoost
- Serving / UI: Gradio
- Preprocessing & training: pandas, scikit-learn
- Language: Python

#### Tips & caveats
- Model generalization depends on dataset coverage (geography, vehicle fleets, traffic patterns).
- Missing categorical values (Weather, Traffic_Level, Courier_Experience) were present — choose an appropriate imputation strategy for production.
- Consider recalibrating or retraining with new data periodically to keep predictions accurate.

