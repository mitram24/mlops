# --- Serving image: small, only what's needed to run the FastAPI model service --------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/data/06_models

WORKDIR /app

# Install slim serving dependencies first (better layer caching).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application code + the trained model artefacts produced by `kedro run`.
COPY src/ ./src/
COPY data/06_models/ ./data/06_models/

EXPOSE 8000

# Basic container healthcheck hitting the API's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "mlops_player_rating.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
