# 1. Imagem base — mais estável para libs científicas
FROM python:3.10

# 2. Diretório de trabalho
WORKDIR /app

# 3. Instala dependências de sistema (essencial para numpy/xgboost/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Copia requirements e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia os arquivos do projeto (exceto ignorados no .dockerignore)
COPY . .

# 6. Garante que o script está executável
RUN chmod +x ./run.sh

# 7. Exponha apenas a porta da API
EXPOSE 8000

# 8. Comando de inicialização
CMD ["./run.sh"]
