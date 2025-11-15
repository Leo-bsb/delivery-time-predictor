import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd # Necessário para o XGBoost mesmo com numpy

# --- Configuração do App ---
app = FastAPI(
    title="API de Predição de Tempo de Entrega",
    description="Microserviço para estimar tempo de entrega de delivery.",
    version="1.0.0"
)

# --- Carregamento do Modelo ---

# Função auxiliar para carregar o modelo de forma segura
def load_model(path: str):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = data['model']
        category_maps = data['category_maps']
        feature_cols = data['feature_cols']
        mae = data['mae']
        r2 = data['r2']
        
        print("✅ Modelo e metadados carregados com sucesso.")
        print(f"   -> MAE: {mae:.2f}, R²: {r2:.3f}")
        print(f"   -> Features: {feature_cols}")
        
        return model, category_maps, feature_cols, mae, r2
        
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo '{path}' não encontrado.")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ ERRO ao carregar o modelo: {e}")
        return None, None, None, None, None

# Carrega na inicialização da API
# O Dockerfile garantirá que este arquivo esteja no path
model, category_maps, feature_cols, mae_test, r2_test = load_model('delivery_model.pkl')

if model is None:
    raise RuntimeError("Falha ao carregar o 'delivery_model.pkl'. A API não pode iniciar.")

# --- Schemas Pydantic (Validação) ---

class InputSchema(BaseModel):
    distance: float
    weather: str
    traffic: str
    time_of_day: str
    vehicle: str
    prep_time: float
    experience: float

    class Config:
        schema_extra = {
            "example": {
                "distance": 10.5,
                "weather": "Rainy",
                "traffic": "High",
                "time_of_day": "Night",
                "vehicle": "Motorcycle",
                "prep_time": 25.0,
                "experience": 8.0
            }
        }

class OutputSchema(BaseModel):
    predicted_time_min: float
    confidence_lower_bound: float
    confidence_upper_bound: float

# --- Funções Auxiliares ---

def safe_encode(value, mapping):
    """Codifica o valor usando o mapa, com fallback para -1."""
    return mapping.get(value, -1)

def preprocess(data: InputSchema, maps: dict) -> np.ndarray:
    """Transforma os dados de entrada no formato que o modelo espera."""
    
    # Feature Engineering
    distance_per_prep = data.distance / max(data.prep_time, 1e-6)
    rush_hour = int(data.time_of_day in ['Morning', 'Afternoon'])

    # Encoding
    weather_enc = safe_encode(data.weather, maps['Weather'])
    traffic_enc = safe_encode(data.traffic, maps['Traffic_Level'])
    tod_enc = safe_encode(data.time_of_day, maps['Time_of_Day'])
    vehicle_enc = safe_encode(data.vehicle, maps['Vehicle_Type'])
    
    # Monta o vetor de features na ordem correta
    # (Baseado em `feature_cols` salvo no pickle)
    features = [
        data.distance,
        weather_enc,
        traffic_enc,
        tod_enc,
        vehicle_enc,
        data.prep_time,
        data.experience,
        distance_per_prep,
        rush_hour
    ]
    
    return np.array(features).reshape(1, -1)

# --- Endpoints da API ---

@app.get("/")
def read_root():
    return {"status": "online", "service": "Delivery Time Predictor API"}

@app.post("/predict", response_model=OutputSchema)
def post_predict(data: InputSchema):
    """Recebe os dados de entrada, processa e retorna a predição."""
    
    # 1. Pré-processamento
    features_array = preprocess(data, category_maps)
    
    # 2. Predição
    prediction = float(model.predict(features_array)[0])
    
    # 3. Faixa de confiança (baseada no MAE)
    lower_bound = max(0.0, prediction - mae_test)
    upper_bound = prediction + mae_test
    
    return OutputSchema(
        predicted_time_min=prediction,
        confidence_lower_bound=lower_bound,
        confidence_upper_bound=upper_bound
    )

@app.get("/metrics")
def get_metrics():
    """Retorna as métricas de avaliação do modelo."""
    return {
        "mae_teste": mae_test,
        "r2_score_teste": r2_test
    }

@app.get("/feature_importance")
def get_feature_importance():
    """Retorna a importância de cada feature (XGBoost)."""
    importance = model.feature_importances_
    
    # CORREÇÃO: Converte o array numpy para uma lista Python
    importance_list = importance.tolist()
    
    if len(feature_cols) == len(importance_list):
        return dict(zip(feature_cols, importance_list)) # <-- CORRIGIDO
    else:
        return {"error": "Disparidade entre features e valores de importância."}

# --- Execução local (para testes) ---
if __name__ == "__main__":
    print("🚀 Iniciando API localmente com Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000)