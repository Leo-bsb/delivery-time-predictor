#!/bin/bash

# Inicia a API FastAPI em background
echo "🚀 Iniciando API FastAPI na porta 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Aguarda alguns segundos para a API iniciar
sleep 5

# Inicia o Frontend Gradio em foreground
# O Hugging Face Spaces usa a porta 7860 por padrão
echo "🚀 Iniciando Frontend Gradio na porta 7860..."
python frontend/app.py --server_port 7860 --server_name 0.0.0.0