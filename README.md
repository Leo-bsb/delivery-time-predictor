# 🍕 Delivery Time Predictor

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&style=flat-square)
![XGBoost](https://img.shields.io/badge/xgboost-1.6-orange?logo=xgboost&style=flat-square)
![Gradio](https://img.shields.io/badge/gradio-3.20-purple?style=flat-square)
![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square) 
![Last commit](https://img.shields.io/github/last-commit/leo-bsb/delivery-time-predictor?style=flat-square)

### Friendly XGBoost model + Gradio demo to estimate food delivery time (link to live demo below)

- Live demo: [Try the app on Hugging Face Spaces](https://leo-bsb-delivery-time-predictor.hf.space/)  
- Short purpose: quick, explainable estimates of delivery time (minutes) from order & courier features — great for demos, prototyping logistic UIs, or lightweight decision support.

⚠️ Note: This application is currently in Brazilian Portuguese (Português-BR). All inputs, outputs, and UI elements are in Portuguese.

#### Quick summary (what this repo contains)
- Model: XGBoost regressor trained with the [Food Delivery Time Prediction dataset](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction).
- Serving/UI: Gradio app (app.py) with interactive inputs, plots, and example cases.
- Artifacts: saved model file `delivery_model.pkl` (contains model, encoders, metrics).
- Visuals in the app: feature importance, predicted vs real, and error distribution.

![Uploading Captura de tela de 2025-11-12 01-27-46.png…]()


#### Dataset & features (pipeline overview)
- Main input columns used:
  - Distance_km
  - Weather (categorical → encoded)
  - Traffic_Level (categorical → encoded)
  - Time_of_Day (categorical → encoded)
  - Vehicle_Type (categorical → encoded)
  - Preparation_Time_min
  - Courier_Experience_yrs
  - Delivery_Time_min (target)
- Extra engineered features:
  - distance_per_prep = Distance_km / Preparation_Time_min
  - rush_hour = 1 if Time_of_Day in [Morning, Afternoon] else 0
- Robust categorical encoding: manual mapping per category with fallback -1 for unseen categories.

#### Model & performance
- Algorithm: XGBoostRegressor (scikit-learn API)
- Training: 80/20 split + 5-fold CV (scoring = MAE)
- Reported metrics:
  - CV MAE: ~7.15 ± 0.27 min
  - MAE (train): 3.78 min
  - MAE (test): 6.37 min
  - RMSE (test): 9.55 min
  - R² (test): 0.816

#### Repo structure (suggested)
- app.py (Gradio UI + glue code)
- data/ (optional: raw / processed CSV)
- models/
  - delivery_model.pkl
- requirements.txt
- README.md

#### How to run locally (quick)
- Create & activate venv:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows PowerShell
```
- Install dependencies (example):
```bash
pip install -r requirements.txt
# or
pip install polars pandas numpy scikit-learn xgboost plotly gradio pickle5
```
- Run app:
```bash
python app.py
```
- The script launches a Gradio UI (it uses interface.launch(share=True) in the demo). For local only run, remove `share=True` or set `debug=False` per your needs.

#### Example programmatic usage (load saved model)
- Load model from `delivery_model.pkl`:
```python
import pickle
with open('delivery_model.pkl', 'rb') as f:
    artifact = pickle.load(f)
model = artifact['model']
category_maps = artifact['category_maps']
mae_test = artifact['mae']
```
- Use a safe encoding helper:
```python
def safe_encode(value, mapping):
    return mapping.get(value, -1)
```
- Build input vector matching feature order in `artifact['feature_cols']` and call `model.predict`.

#### Gradio app notes (what the UI provides)
- Input controls: sliders + dropdowns for distance, weather, traffic, time of day, vehicle, prep time, courier experience.
- Tabs in UI:
  - Prediction (with examples)
  - Feature importance (Plotly)
  - Predicted vs Real (Plotly)
  - Error distribution (Plotly)
  - Model metrics & config
- Example values and safe defaults are automatically read from the dataset.


#### Tips for improvement (ideas)
- Add SHAP explanations for per-prediction interpretability.
- Replace simple imputation with more robust strategies (or model-based imputers).
- Add calibration / quantile models for formal prediction intervals.
- Use a pipeline + joblib to persist preprocessor + model together (better for production).


#### License
- MIT
