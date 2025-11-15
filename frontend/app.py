import gradio as gr
import requests
import plotly.express as px
import pandas as pd
import sys

# Define a URL da nossa API local (rodando no mesmo container)
API_URL = "http://127.0.0.1:8000"

# --- Funções para carregar dados da API ---

def load_metrics_from_api():
    """Carrega as métricas (MAE, R²) da API."""
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            data = response.json()
            return data.get('mae_teste'), data.get('r2_score_teste')
    except requests.ConnectionError:
        print("❌ ERRO: Não foi possível conectar à API em /metrics.")
        return None, None
    return None, None

def load_importance_from_api():
    """Carrega a importância das features e cria o gráfico Plotly."""
    try:
        response = requests.get(f"{API_URL}/feature_importance")
        if response.status_code == 200:
            data = response.json()
            if 'error' in data:
                return None
            
            # Criar DataFrame para o Plotly
            importance_df = pd.DataFrame({
                'Feature': data.keys(),
                'Importance': data.values()
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='📊 Importância das Features',
                color='Importance',
                color_continuous_scale='viridis'
            )
            return fig
    except requests.ConnectionError:
        print("❌ ERRO: Não foi possível conectar à API em /feature_importance.")
        return None
    return None

# --- Carrega os dados estáticos na inicialização do app ---
MAE_TEST, R2_TEST = load_metrics_from_api()
FIG_IMPORTANCE = load_importance_from_api()

# --- Função de Predição (Chamando a API) ---

def predict_delivery_time(distance, weather, traffic, time_of_day,
                          vehicle, prep_time, experience):
    """Função que o Gradio chama, que por sua vez chama a API."""
    
    # 1. Monta o payload JSON para a API
    payload = {
        "distance": distance,
        "weather": weather,
        "traffic": traffic,
        "time_of_day": time_of_day,
        "vehicle": vehicle,
        "prep_time": prep_time,
        "experience": experience
    }
    
    # 2. Faz a requisição POST para o endpoint /predict
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            # 3. Processa a resposta da API
            data = response.json()
            prediction = data['predicted_time_min']
            lower_bound = data['confidence_lower_bound']
            upper_bound = data['confidence_upper_bound']

            result = f"""
            ⏱️ **Tempo Estimado de Entrega:** {prediction:.1f} minutos
            
            📊 **Faixa de Confiança:** {lower_bound:.1f} - {upper_bound:.1f} min
            
            💡 **Explicação:** O tempo foi calculado com base nas condições fornecidas.
            
            🎯 **Métricas do Modelo:** MAE ±{MAE_TEST:.1f} min | R² {R2_TEST:.3f}
            """
            return result
            
        else:
            return f"Erro da API: {response.status_code} - {response.text}"
            
    except requests.ConnectionError as e:
        return f"❌ ERRO DE CONEXÃO: Não foi possível conectar à API. Verifique se ela está rodando.\nDetalhes: {e}"
    except Exception as e:
        return f"❌ ERRO Inesperado: {e}"

# --- Definição da Interface Gradio ---

# Opções para dropdowns (você pode carregar isso de um config ou da API também)
weather_options = ['Clear', 'Fog', 'Rainy', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']
traffic_options = ['High', 'Jam', 'Low', 'Medium']
time_options = ['Afternoon', 'Evening', 'Morning', 'Night']
vehicle_options = ['Bike', 'Electric Scooter', 'Motorcycle', 'Scooter']

default_weather = "Clear"
default_traffic = "Low"
default_time = "Morning"
default_vehicle = "Motorcycle"

with gr.Blocks(theme="soft", title="🍕 Delivery Time Predictor") as interface:
    
    gr.Markdown("# 🍕 Sistema de Previsão de Tempo de Entrega (v2 - API)")
    gr.Markdown("### Arquitetura desacoplada com FastAPI e Gradio")
    
    with gr.Tab("🎯 Predição"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📝 Insira os dados da entrega:")
                distance = gr.Slider(0.5, 20, value=5, step=0.1, label="📍 Distância (km)")
                weather = gr.Dropdown(weather_options, label="🌤️ Clima", value=default_weather)
                traffic = gr.Dropdown(traffic_options, label="🚦 Nível de Tráfego", value=default_traffic)
                time_of_day = gr.Dropdown(time_options, label="🕐 Período do Dia", value=default_time)
                vehicle = gr.Dropdown(vehicle_options, label="🛵 Tipo de Veículo", value=default_vehicle)
                prep_time = gr.Slider(5, 60, value=15, step=1, label="👨‍🍳 Tempo de Preparo (min)")
                experience = gr.Slider(0, 15, value=3, step=0.5, label="⭐ Experiência do Entregador (anos)")
                
                btn = gr.Button("🚀 Calcular Tempo de Entrega", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("#### 📊 Resultado da Predição:")
                output = gr.Markdown("Aguardando entrada...")
        
        btn.click(
            fn=predict_delivery_time,
            inputs=[distance, weather, traffic, time_of_day, vehicle, prep_time, experience],
            outputs=output
        )
        
        gr.Markdown("#### 💡 Exemplos:")
        gr.Examples(
            examples=[
                # Adicionamos 'default_weather' (ou "Clear") na segunda posição
                [5.0, default_weather, "High", "Afternoon", "Bike", 15, 3],
                [12.5, "Rainy", "High", "Night", "Scooter", 20, 5],
                [3.2, "Clear", "Low", "Morning", "Motorcycle", 10, 7]
            ],
            # CORREÇÃO: A lista 'inputs' deve ter TODOS os 7 componentes
            inputs=[distance, weather, traffic, time_of_day, vehicle, prep_time, experience],
            outputs=output,
            fn=predict_delivery_time,
            cache_examples=False
        )
    
    with gr.Tab("📊 Importância das Features"):
        gr.Markdown("### 📈 Quais fatores mais influenciam o tempo de entrega?")
        gr.Markdown("*(Estes dados são carregados da API na inicialização do app)*")
        if FIG_IMPORTANCE:
            gr.Plot(FIG_IMPORTANCE)
        else:
            gr.Markdown("❌ **Não foi possível carregar o gráfico de importância.**")
    
    with gr.Tab("ℹ️ Métricas do Modelo"):
        gr.Markdown("### 📈 Desempenho do Modelo")
        gr.Markdown("*(Estes dados são carregados da API na inicialização do app)*")
        if MAE_TEST and R2_TEST:
            gr.Markdown(f"""
            | Métrica | Valor |
            |---------|-------|
            | **MAE Teste** | {MAE_TEST:.2f} min |
            | **R² Score** | {R2_TEST:.3f} |
            """)
        else:
            gr.Markdown("❌ **Não foi possível carregar as métricas do modelo.**")
            
    # Removemos a aba "Predito vs Real", pois o frontend não tem
    # acesso ao dataset de teste (y_test, y_pred), o que é uma
    # boa prática de desacoplamento.

# Permite que o Gradio seja iniciado pela linha de comando
# com os argumentos do 'run.sh'
if __name__ == "__main__":
    server_port = 7860
    server_name = "127.0.0.1"

    # Processa argumentos da linha de comando
    if '--server_port' in sys.argv:
        server_port = int(sys.argv[sys.argv.index('--server_port') + 1])
    if '--server_name' in sys.argv:
        server_name = sys.argv[sys.argv.index('--server_name') + 1]

    print(f"✅ Iniciando Gradio em {server_name}:{server_port}")
    interface.launch(server_name=server_name, server_port=server_port, share=False)