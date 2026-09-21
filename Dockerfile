# Image de base légère avec Python
FROM python:3.11-slim

# Dépendances système nécessaires à OpenCV et à la compilation de certains paquets
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python d'abord (meilleure mise en cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY app.py utils.py ./

# Le cache Hugging Face sera stocké dans un volume pour ne pas re-télécharger
# le modèle à chaque redémarrage du conteneur
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
