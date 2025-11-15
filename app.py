import polars as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
import pickle


# ============================================
# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# ============================================

df = pl.read_csv('Food_Delivery_Times.csv')

print("📊 Shape do dataset:", df.shape)
print("\n🔍 Primeiras linhas:")
print(df.head())
print("\n📈 Estatísticas descritivas:")
print(df.describe())
print("\n❌ Valores nulos:")
print(df.null_count())

df = df.with_columns(
    pl.col('Courier_Experience_yrs').fill_null(
        pl.when(pl.col('Vehicle_Type') == 'Motorcycle').then(6.0)
          .when(pl.col('Vehicle_Type') == 'Bike').then(4.5)
          .when(pl.col('Vehicle_Type') == 'Scooter').then(3.0)
          .otherwise(pl.col('Courier_Experience_yrs'))
    ).alias('Courier_Experience_yrs')
)


# ============================================
# 2. LIMPEZA, PREPARAÇÃO E FEATURE ENGINEERING
# ============================================

df_pd = df.to_pandas()
print("\n✅ Dados convertidos para Pandas")

required_cols = [
    'Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day',
    'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs',
    'Delivery_Time_min'
]
missing = [c for c in required_cols if c not in df_pd.columns]
if missing:
    raise ValueError(f"Colunas faltando no dataset: {missing}")

# Trata outliers do target (IQR)
Q1 = df_pd['Delivery_Time_min'].quantile(0.25)
Q3 = df_pd['Delivery_Time_min'].quantile(0.75)
IQR = Q3 - Q1
df_pd = df_pd[
    (df_pd['Delivery_Time_min'] >= Q1 - 1.5 * IQR) &
    (df_pd['Delivery_Time_min'] <= Q3 + 1.5 * IQR)
].copy()

# Feature engineering adicional
df_pd['distance_per_prep'] = df_pd['Distance_km'] / np.clip(df_pd['Preparation_Time_min'], 1e-6, None)
df_pd['rush_hour'] = df_pd['Time_of_Day'].isin(['Morning', 'Afternoon']).astype(int)

# Encoding robusto
categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
category_maps = {}
for col in categorical_cols:
    uniques = sorted(df_pd[col].dropna().unique())
    mapping = {v: i for i, v in enumerate(uniques)}
    category_maps[col] = mapping
    df_pd[f'{col}_encoded'] = df_pd[col].map(mapping).fillna(-1).astype(int)

# Features do modelo
feature_cols = [
    'Distance_km',
    'Weather_encoded',
    'Traffic_Level_encoded',
    'Time_of_Day_encoded',
    'Vehicle_Type_encoded',
    'Preparation_Time_min',
    'Courier_Experience_yrs',
    'distance_per_prep',
    'rush_hour'
]

X = df_pd[feature_cols]
y = df_pd['Delivery_Time_min']

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📦 Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

# ============================================
# 3. TREINAMENTO + CROSS-VALIDATION
# ============================================

print("\n🚀 Treinando modelo XGBoost...")

model = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Cross-Validation (MAE)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"MAE CV médio: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Fit final no treino
model.fit(X_train, y_train)

# Predições
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Métricas
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"\n📊 MÉTRICAS DO MODELO:")
print(f"   MAE Treino: {mae_train:.2f} min")
print(f"   MAE Teste: {mae_test:.2f} min")
print(f"   RMSE Teste: {rmse_test:.2f} min")
print(f"   R² Teste: {r2_test:.3f}")


def generate_rule_based_explanation(distance, weather, traffic, time_of_day, 
                                     vehicle, prep_time, experience, 
                                     predicted_time, top_factors):
    """Gera explicação baseada em regras"""
    
    explanations = []
    
    # Analisa distância
    if distance > 10:
        explanations.append(f"a distância de {distance:.1f}km é considerável")
    elif distance < 3:
        explanations.append("a distância é curta")
    
    # Analisa clima
    if weather in ['Rainy', 'Stormy', 'Fog']:
        explanations.append(f"o clima {weather.lower()} reduz a velocidade")
    
    # Analisa tráfego
    if traffic == 'High':
        explanations.append("o tráfego está pesado")
    elif traffic == 'Low':
        explanations.append("o tráfego está leve")
    
    # Analisa horário
    if time_of_day in ['Morning', 'Afternoon']:
        explanations.append("é horário de rush")
    
    # Analisa veículo
    vehicle_speed = {'Bike': 'lento', 'Scooter': 'moderado', 'Motorcycle': 'rápido'}
    if vehicle in vehicle_speed:
        explanations.append(f"o veículo ({vehicle}) é {vehicle_speed[vehicle]}")
    
    # Analisa experiência
    if experience < 2:
        explanations.append("o entregador tem pouca experiência")
    elif experience > 7:
        explanations.append("o entregador é experiente")
    
    # Ordena fatores por valor decrescente
    sorted_factors = sorted(top_factors.items(), key=lambda x: x[1], reverse=True)
    
    # Pega os dois fatores mais relevantes
    if len(sorted_factors) >= 2:
        factor_text = f" Os fatores mais relevantes são: {sorted_factors[0][0]} ({sorted_factors[0][1]:.0%}) e {sorted_factors[1][0]} ({sorted_factors[1][1]:.0%})."
    elif len(sorted_factors) == 1:
        factor_text = f" O fator mais relevante é: {sorted_factors[0][0]} ({sorted_factors[0][1]:.0%})."
    else:
        factor_text = ""
    
    # Monta explicação final
    if len(explanations) >= 2:
        main_explanation = f"O tempo de {predicted_time:.0f} minutos se deve principalmente porque {explanations[0]} e {explanations[1]}."
    elif len(explanations) == 1:
        main_explanation = f"O tempo de {predicted_time:.0f} minutos se deve principalmente porque {explanations[0]}."
    else:
        main_explanation = f"O tempo estimado é {predicted_time:.0f} minutos considerando as condições normais de entrega."
    
    return main_explanation + factor_text


# ============================================
# 4. VISUALIZAÇÕES (CONVERSÃO EXPLÍCITA PARA LISTAS!)
# ============================================

# Importância das features
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

fig_importance = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    title='📊 Importância das Features',
    color='Importance',
    color_continuous_scale='viridis'
)

# Predito vs Real - CONVERSÃO EXPLÍCITA PARA LISTA!
y_test_list = y_test.values.tolist() if hasattr(y_test, 'values') else list(y_test)
y_pred_test_list = y_pred_test.tolist() if hasattr(y_pred_test, 'tolist') else list(y_pred_test)

print(f"\n🔍 Debug - y_test: {len(y_test_list)} pontos")
print(f"🔍 Debug - y_pred_test: {len(y_pred_test_list)} pontos")
print(f"🔍 Debug - Amostra y_test: {y_test_list[:3]}")
print(f"🔍 Debug - Amostra y_pred: {y_pred_test_list[:3]}")

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=y_test_list,
    y=y_pred_test_list,
    mode='markers',
    marker=dict(color='blue', opacity=0.6, size=8),
    name='Predições',
    text=[f'Real: {r:.1f}<br>Pred: {p:.1f}' for r, p in zip(y_test_list, y_pred_test_list)],
    hovertemplate='%{text}<extra></extra>'
))

min_axis = min(min(y_test_list), min(y_pred_test_list))
max_axis = max(max(y_test_list), max(y_pred_test_list))

fig_pred.add_trace(go.Scatter(
    x=[min_axis, max_axis],
    y=[min_axis, max_axis],
    mode='lines',
    line=dict(color='red', dash='dash', width=2),
    name='Linha Ideal'
))
fig_pred.update_layout(
    title=f'🎯 Tempo Predito vs Real ({len(y_test_list)} pontos)',
    xaxis_title='Tempo Real (min)',
    yaxis_title='Tempo Predito (min)',
    height=500,
    showlegend=True
)

# Distribuição dos erros - CONVERSÃO EXPLÍCITA E CRIAÇÃO DO DATAFRAME!
errors = (np.array(y_test_list) - np.array(y_pred_test_list)).tolist()
print(f"🔍 Debug - Erros: {len(errors)} valores, amostra: {errors[:3]}")

# CORREÇÃO: Criar DataFrame para o histograma
errors_df = pd.DataFrame({'Erro': errors})

fig_errors = px.histogram(
    errors_df,
    x='Erro',
    nbins=50,
    title=f'📉 Distribuição dos Erros de Predição (n={len(errors)})',
    labels={'Erro': 'Erro (min)', 'count': 'Frequência'}
)
fig_errors.update_traces(marker_color='indianred', marker_line_color='darkred', marker_line_width=1)
fig_errors.update_layout(height=500)

# ============================================
# 5. INTERFACE GRADIO
# ============================================

def safe_encode(value, mapping):
    return mapping.get(value, -1)

def predict_delivery_time(distance, weather, traffic, time_of_day,
                          vehicle, prep_time, experience):
    """Função de predição para o Gradio com explicação inteligente"""

    # Cria features derivadas
    distance_per_prep = distance / max(prep_time, 1e-6)
    rush_hour = int(time_of_day in ['Morning', 'Afternoon'])

    # Encoding via mapas
    weather_enc = safe_encode(weather, category_maps['Weather'])
    traffic_enc = safe_encode(traffic, category_maps['Traffic_Level'])
    tod_enc = safe_encode(time_of_day, category_maps['Time_of_Day'])
    vehicle_enc = safe_encode(vehicle, category_maps['Vehicle_Type'])

    # Monta vetor de entrada
    features = np.array([[
        distance,
        weather_enc,
        traffic_enc,
        tod_enc,
        vehicle_enc,
        prep_time,
        experience,
        distance_per_prep,
        rush_hour
    ]])

    # Predição
    prediction = float(model.predict(features)[0])
    
    # Calcula importância relativa das features para ESTA predição
    feature_values = {
        'Distância': min(distance / 20.0, 1.0),
        'Clima': 1.0 if weather in ['Rainy', 'Stormy', 'Fog'] else 0.3,
        'Tráfego': {'Low': 0.2, 'Medium': 0.5, 'High': 1.0}.get(traffic, 0.5),
        'Horário': 1.0 if time_of_day in ['Morning', 'Afternoon'] else 0.3,
        'Veículo': {'Bike': 0.3, 'Scooter': 0.6, 'Motorcycle': 1.0}.get(vehicle, 0.5),
        'Tempo de preparo': min(prep_time / 60.0, 1.0),
        'Experiência': max(0, 1.0 - (experience / 15.0))
    }
    
    # GERA EXPLICAÇÃO (Baseada em regras)
    explanation = generate_rule_based_explanation(
        distance, weather, traffic, time_of_day,
        vehicle, prep_time, experience,
        prediction, feature_values
    )

    # Faixa de confiança
    lower_bound = max(0.0, prediction - mae_test)
    upper_bound = prediction + mae_test

    # CORREÇÃO: Adicionar a explicação ao resultado
    result = f"""
⏱️ **Tempo Estimado de Entrega:** {prediction:.1f} minutos

📊 **Faixa de Confiança:** {lower_bound:.1f} - {upper_bound:.1f} min

💡 **Explicação:** {explanation}

🎯 **Métricas do Modelo:** MAE ±{mae_test:.1f} min | R² {r2_test:.3f}
"""
    return result

# Opções para dropdowns
weather_options = sorted(df_pd['Weather'].dropna().unique().tolist())
traffic_options = sorted(df_pd['Traffic_Level'].dropna().unique().tolist())
time_options = sorted(df_pd['Time_of_Day'].dropna().unique().tolist())
vehicle_options = sorted(df_pd['Vehicle_Type'].dropna().unique().tolist())

default_weather = weather_options[0] if weather_options else ""
default_traffic = traffic_options[0] if traffic_options else ""
default_time = time_options[0] if time_options else ""
default_vehicle = vehicle_options[0] if vehicle_options else ""

# Interface com abas
with gr.Blocks(theme="soft", title="🍕 Delivery Time Predictor") as interface:
    
    gr.Markdown("# 🍕 Sistema de Previsão de Tempo de Entrega")
    gr.Markdown("### Preveja o tempo de entrega do seu pedido")
    
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
                output = gr.Markdown()
        
        btn.click(
            fn=predict_delivery_time,
            inputs=[distance, weather, traffic, time_of_day, vehicle, prep_time, experience],
            outputs=output
        )
        
        gr.Markdown("#### 💡 Exemplos:")
        gr.Examples(
            examples=[
                [5.0, default_weather, "High", "Afternoon", "Bike", 15, 3],
                [12.5, "Rainy", "High", "Night", "Scooter", 20, 5],
                [3.2, "Clear", "Low", "Morning", "Motorcycle", 10, 7]
            ],
            inputs=[distance, weather, traffic, time_of_day, vehicle, prep_time, experience],
            outputs=output,
            fn=predict_delivery_time,
            cache_examples=False
        )
    
    with gr.Tab("📊 Importância das Features"):
        gr.Markdown("### 📈 Quais fatores mais influenciam o tempo de entrega?")
        gr.Plot(fig_importance)
    
    with gr.Tab("🎯 Predito vs Real"):
        gr.Markdown("### 🔍 Comparação entre valores preditos e reais")
        gr.Plot(fig_pred)
    
    with gr.Tab("ℹ️ Métricas do Modelo"):
        gr.Markdown(f"""
        ### 📈 Desempenho do Modelo
        
        | Métrica | Valor |
        |---------|-------|
        | **MAE Treino** | {mae_train:.2f} min |
        | **MAE Teste** | {mae_test:.2f} min |
        | **RMSE Teste** | {rmse_test:.2f} min |
        | **R² Score** | {r2_test:.3f} |
        | **Registros** | {len(df_pd)} entregas |
        | **CV MAE** | {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} min |
        
        ---
        
        ### 🔧 Configuração
        
        - **Algoritmo:** XGBoost Regressor
        - **Features:** {len(feature_cols)}

        ---
        
        ### 📝 Features: {', '.join(feature_cols)}
        """)

interface.launch(share=True, debug=True)

# ============================================
# 6. SALVAR MODELO
# ============================================

with open('delivery_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'category_maps': category_maps,
        'feature_cols': feature_cols,
        'mae': mae_test,
        'r2': r2_test
    }, f)

print("\n💾 Modelo salvo como 'delivery_model.pkl'")