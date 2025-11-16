#!/bin/bash

set -e  # se qualquer comando falhar, o container cai (bom para CI/CD)

echo "🚀 Iniciando API FastAPI na porta 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

API_PID=$!

# Aguarda API subir
sleep 4

echo "🚀 Iniciando Frontend Gradio na porta 7860..."
python frontend/app.py --server_name 0.0.0.0 --server_port 7860

# Se o frontend morrer, mata API
kill $API_PID
