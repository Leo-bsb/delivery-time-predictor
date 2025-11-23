---
title: "Delivery Time Predictor — XGBoost + Gradio"
emoji: 🚚
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
---



# 🍕 Delivery Time Prediction — XGBoost + Gradio

<p align="center">
  <img src="https://img.shields.io/badge/Model-XGBoost-blue.svg" />
  <img src="https://img.shields.io/badge/UI-Gradio-orange.svg" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  <img src="https://img.shields.io/badge/Status-Production%20Demo-brightgreen.svg" />
</p>

**🔗 Live Demo:** [https://huggingface.co/spaces/leo-bsb/Delivery-Time-Predictor](https://huggingface.co/spaces/leo-bsb/Delivery-Time-Predictor)

A lightweight XGBoost regression model that predicts **delivery time (minutes)** from operational features such as distance, weather, traffic conditions, vehicle type, and courier experience.
Built to support logistics decisions and serve as a clean, interactive demo via **Gradio**.

---

## 📦 Overview

* **Goal:** Provide fast and explainable delivery-time estimates.
* **Model:** XGBoost Regressor (scikit-learn API).
* **Interface:** Gradio app with clean UI for quick experimentation.
* **Use cases:**

  * Delivery/logistics prototyping
  * Time-estimation demos
  * Teaching regression + feature encoding + model serving

---

## 📊 Dataset Summary

* **Size:** 1,000 rows × 9 columns
* **Train / Test:** 795 / 199
* **Converted to Pandas** for preprocessing and modeling
* **Missing values:**

  * Weather: 30
  * Traffic_Level: 30
  * Courier_Experience: 30

**Key columns:**

* Distance_km
* Weather
* Traffic_Level
* Vehicle_Type
* Preparation_Time_min
* Courier_Experience_yrs
* Delivery_Time_min *(target)*

**Example rows (one column omitted):**

```
522, 7.93, Windy,  Low,    Scooter, 12, 1.0, 43
738, 16.42, Clear, Medium, Bike,    20, 2.0, 84
741, 9.52,  Foggy, Low,    Scooter, 28, 1.0, 59
661, 7.44,  Rainy, Medium, Scooter,  5, 1.0, 37
412, 19.03, Clear, Low,    Bike,    16, 5.0, 68
```

---

## 📈 Model Performance

* **CV (k-fold) MAE:** 7.15 ± 0.27 min
* **Final metrics:**

  * MAE (train): **3.78 min**
  * MAE (test): **6.37 min**
  * RMSE (test): **9.55 min**
  * R² (test): **0.816**

---

## 🧪 Pipeline Summary

1. Load dataset → convert to Pandas
2. Handle missing values (categorical + numeric strategies)
3. Encode Weather, Traffic_Level, Vehicle_Type
4. Train-test split (795 / 199)
5. Train XGBoost with CV (MAE objective)
6. Evaluate on hold-out test set
7. Serve model in Gradio for interactive inference

---

## 🎛️ Gradio App Features

The **`app.py`** interface allows you to:

* Input all model features (distance, weather, traffic, vehicle type, prep time, courier experience)
* Get predicted **delivery time (minutes)**
* (Optional) Extend with SHAP/feature importances for explainability
* Fast inference → ideal for demos or prototypes

Live version:
👉 [https://huggingface.co/spaces/leo-bsb/Delivery-Time-Predictor](https://huggingface.co/spaces/leo-bsb/Delivery-Time-Predictor)

---

## 🖥️ Run Locally

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Notes:

* The UI uses **Gradio 4.x**
* If using a serialized model file, keep a consistent name (e.g., `model.pkl`) and load via `joblib` or `pickle`

---

## 📥 Input Ranges (suggested)

* **Distance_km:** 0.5–20.0
* **Weather:** Clear / Rainy / Windy / Foggy / ...
* **Traffic_Level:** Low / Medium / High
* **Vehicle_Type:** Bike / Scooter / Car / ...
* **Preparation_Time_min:** 5–30
* **Courier_Experience_yrs:** 0–9

---

## 🧱 Tech Stack

* **Model:** XGBoost
* **UI / Serving:** Gradio
* **Preprocessing:** pandas, scikit-learn
* **Language:** Python


