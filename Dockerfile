FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero: aprovecha la caché de capas de Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY matchpredictor ./matchpredictor
RUN mkdir -p data/raw

EXPOSE 8000

# Por defecto descarga los datos. Los CSV quedan en el volumen montado en /app/data:
#   docker run --rm -v "$(pwd)/data:/app/data" match-predictor
#   (la web se arranca con docker compose up web; ver docker-compose.yml)
CMD ["python", "-m", "matchpredictor.data.download"]
