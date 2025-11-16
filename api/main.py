import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# --- Configuração da API ---
app = FastAPI(
    title="API de Predição de Tempo de Entrega",
    description="Microserviço para estimar tempo de entrega de delivery.",
    version="1.0.0"
)

# --- Carregamento do Modelo via HuggingFace Hub ---

def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="leo-bsb/delivery-time-model",
            filename="delivery_model.pkl"
        )

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        required_keys = ["model", "category_maps", "feature_cols", "mae", "r2"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Chave ausente no pickle: {key}")

        return (
            data["model"],
            data["category_maps"],
            data["feature_cols"],
            data["mae"],
            data["r2"],
        )

    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")

model, category_maps, feature_cols, mae_test, r2_test = load_model()

# --- Schemas Pydantic ---

class InputSchema(BaseModel):
    distance: float
    weather: str
    traffic: str
    time_of_day: str
    vehicle: str
    prep_time: float
    experience: float


class OutputSchema(BaseModel):
    predicted_time_min: float
    confidence_lower_bound: float
    confidence_upper_bound: float

# --- Funções auxiliares ---

def safe_encode(value, mapping):
    return mapping.get(value, -1)

def preprocess(data: InputSchema, maps: dict) -> np.ndarray:
    distance_per_prep = data.distance / max(data.prep_time, 1e-6)
    rush_hour = int(data.time_of_day in ["Morning", "Afternoon"])

    weather_enc = safe_encode(data.weather, maps["Weather"])
    traffic_enc = safe_encode(data.traffic, maps["Traffic_Level"])
    tod_enc = safe_encode(data.time_of_day, maps["Time_of_Day"])
    vehicle_enc = safe_encode(data.vehicle, maps["Vehicle_Type"])

    features = [
        data.distance,
        weather_enc,
        traffic_enc,
        tod_enc,
        vehicle_enc,
        data.prep_time,
        data.experience,
        distance_per_prep,
        rush_hour,
    ]

    return np.array(features).reshape(1, -1)


# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "online", "service": "Delivery Time Predictor"}

@app.post("/predict", response_model=OutputSchema)
def predict(data: InputSchema):
    features = preprocess(data, category_maps)
    prediction = float(model.predict(features)[0])

    lower = max(0.0, prediction - mae_test)
    upper = prediction + mae_test

    return OutputSchema(
        predicted_time_min=prediction,
        confidence_lower_bound=lower,
        confidence_upper_bound=upper
    )

@app.get("/metrics")
def metrics():
    return {"mae_test": mae_test, "r2_test": r2_test}

@app.get("/feature_importance")
def feature_importance():
    imp = model.feature_importances_.tolist()

    if len(imp) != len(feature_cols):
        return {"error": "Dimensão inconsistente entre features e importâncias."}

    return dict(zip(feature_cols, imp))
