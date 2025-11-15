# 1. Imagem base
FROM python:3.10-slim

# 2. Define o diretório de trabalho
WORKDIR /app

# 3. Copia e instala as dependências
# (Copiamos o requirements.txt primeiro para aproveitar o cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia todo o código do projeto para dentro do container
COPY . .

# 5. Torna o script de inicialização executável
RUN chmod +x ./run.sh

# 6. Define a porta que o container vai expor
# O FastAPI rodará na 8000, o Gradio na 7860
EXPOSE 8000 7860

# 7. Comando para iniciar o container
CMD ["./run.sh"]