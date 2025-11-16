# app.py — API FastAPI + UI Gradio servidos juntos em /ui

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn
import numpy as np
import pandas as pd
import plotly.express as px
import pickle

# -----------------------------------------------------
# 1. Inicialização da API
# -----------------------------------------------------

app = FastAPI(
    title="Delivery Time Predictor",
    description="Unified API + Gradio UI",
    version="2.0.0"
)

# -----------------------------------------------------
# 2. Carrega modelo
# -----------------------------------------------------

def load_model(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return (
        data["model"],
        data["category_maps"],
        data["feature_cols"],
        data["mae"],
        data["r2"]
    )

model, category_maps, feature_cols, mae_test, r2_test = load_model("delivery_model.pkl")

# -----------------------------------------------------
# 3. Funções auxiliares
# -----------------------------------------------------

def safe_encode(value, mapping):
    return mapping.get(value, -1)

def preprocess(data, maps):
    distance_per_prep = data["distance"] / max(data["prep_time"], 1e-6)
    rush_hour = int(data["time_of_day"] in ["Morning", "Afternoon"])

    features = [
        data["distance"],
        safe_encode(data["weather"], maps["Weather"]),
        safe_encode(data["traffic"], maps["Traffic_Level"]),
        safe_encode(data["time_of_day"], maps["Time_of_Day"]),
        safe_encode(data["vehicle"], maps["Vehicle_Type"]),
        data["prep_time"],
        data["experience"],
        distance_per_prep,
        rush_hour
    ]

    return np.array(features).reshape(1, -1)

# -----------------------------------------------------
# 4. Endpoints da API
# -----------------------------------------------------

@app.get("/")
def root():
    return {"status": "online", "ui": "/ui"}

@app.post("/predict")
def predict(payload: dict):
    X = preprocess(payload, category_maps)
    pred = float(model.predict(X)[0])
    return {
        "predicted_time_min": pred,
        "confidence_lower_bound": max(0.0, pred - mae_test),
        "confidence_upper_bound": pred + mae_test
    }

@app.get("/metrics")
def metrics():
    return {"mae_test": mae_test, "r2_test": r2_test}

@app.get("/feature_importance")
def feature_importance():
    imp = model.feature_importances_.tolist()
    return dict(zip(feature_cols, imp))

# -----------------------------------------------------
# 5. Interface Gradio
# -----------------------------------------------------

def gradio_predict(distance, weather, traffic, time_of_day, vehicle, prep_time, experience):
    payload = {
        "distance": distance,
        "weather": weather,
        "traffic": traffic,
        "time_of_day": time_of_day,
        "vehicle": vehicle,
        "prep_time": prep_time,
        "experience": experience
    }

    result = predict(payload)

    return f"""
    ⏱️ **Estimativa:** {result['predicted_time_min']:.1f} min  
    📉 Intervalo: {result['confidence_lower_bound']:.1f} – {result['confidence_upper_bound']:.1f} min  
    🎯 MAE: ±{mae_test:.1f} min  
    R²: {r2_test:.3f}  
    """

# Gráfico de importância
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="📊 Importância das Features"
)

with gr.Blocks(title="Delivery Time Predictor") as gradio_app:

    gr.Markdown("# 🍕 Delivery Time Predictor")

    with gr.Tab("Predição"):
        distance = gr.Slider(0.5, 20, value=5)
        weather = gr.Dropdown(['Clear', 'Fog', 'Rainy', 'Sandstorms', 'Stormy', 'Sunny', 'Windy'])
        traffic = gr.Dropdown(['High', 'Jam', 'Low', 'Medium'])
        time_of_day = gr.Dropdown(['Afternoon', 'Evening', 'Morning', 'Night'])
        vehicle = gr.Dropdown(['Bike', 'Electric Scooter', 'Motorcycle', 'Scooter'])
        prep = gr.Slider(5, 60, value=15)
        exp = gr.Slider(0, 15, value=3)

        out = gr.Markdown()

        gr.Button("Calcular").click(
            gradio_predict,
            [distance, weather, traffic, time_of_day, vehicle, prep, exp],
            out
        )

    with gr.Tab("Importância"):
        gr.Plot(fig_importance)

# -----------------------------------------------------
# 6. Rota /ui → interface Gradio embutida
# -----------------------------------------------------

@app.get("/ui", response_class=HTMLResponse)
def ui():
    """
    Retorna o HTML da interface Gradio.
    """
    return gradio_app.launch(
        inline=True,
        share=False,
        prevent_thread_lock=True,
        server_name="0.0.0.0",
        server_port=None,
    )

# -----------------------------------------------------
# 7. Execução
# -----------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
