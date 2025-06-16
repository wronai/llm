FROM python:3.10-slim

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Skopiuj requirements
COPY requirements.txt .

# Zainstaluj zależności Python
RUN pip install --no-cache-dir -r requirements.txt

# Pobierz model FastText
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Skopiuj kod aplikacji
COPY . .

# Utwórz katalogi
RUN mkdir -p /app/wronai_data /app/cache /app/logs

# Zmienne środowiskowe
ENV PYTHONPATH=/app
ENV HF_DATASETS_CACHE=/app/cache

# Uruchom pipeline
CMD ["python", "collect_wronai_data.py"]