FROM python:3.12.7-slim

# mediapipe's Tasks API (used for pose/face detection) links against GL
# libraries even with the CPU delegate; Render's native Python runtime
# doesn't have them, so this needs a Docker image where we can install them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libgbm1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN bash scripts/fetch_catalog_assets.sh

EXPOSE 10000
CMD ["sh", "-c", "gunicorn --chdir backend --bind 0.0.0.0:${PORT:-10000} --timeout 120 app:app"]
