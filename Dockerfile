FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ARG VITE_API_BASE=
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVE_FRONTEND=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics_config \
    MPLCONFIGDIR=/app/.matplotlib_config

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-react-demo.txt ./
RUN pip install --upgrade pip && pip install -r requirements-react-demo.txt

COPY backend ./backend
COPY data ./data
COPY dataset ./dataset
COPY models ./models
COPY reports ./reports
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

EXPOSE 7860

CMD ["uvicorn", "backend.api_server:app", "--host", "0.0.0.0", "--port", "7860"]
