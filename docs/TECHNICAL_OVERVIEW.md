# Energy MLOps Workshop — Technical Overview

This document explains the repository architecture, key components, and how everything works together. It focuses on the MLOps engineering aspects: serving, containerization, orchestration, scaling, and observability. Use it as the maintainer’s guide and a deep-dive reference.

## Repository Layout
- `src/backend/`: Backend service (FastAPI) for anomaly detection
- `src/ui/`: Streamlit UI for interactive demo and visualization
- `scripts/`: Utility scripts (dev tools, load generator)
- `docker/`: Container build files and Docker Compose
- `k8s/`: Kubernetes manifests with Kustomize
- `models/`: Model artifacts (generated at runtime if not present)
- `docs/`: Documentation (this file + workshop guide)

## Runtime Stack
- **API**: FastAPI + Uvicorn
- **Model**: IsolationForest (scikit-learn) trained on synthetic data
- **Metrics**: Prometheus client metrics endpoint (`/metrics`)
- **UI**: Streamlit, polls backend and visualizes results
- **Packaging/Deps**: `uv` with `pyproject.toml` and `uv.lock`
- **Containers**: Python 3.13 slim, `uv` for dependency install
- **Orchestration**: Kubernetes (kind workflow), HPA based on CPU

## Backend Service
- `src/backend/main.py`: Runs Uvicorn and respects `WORKERS` env for process count.
- `src/backend/app.py`: FastAPI app wiring.
  - Endpoints:
    - `GET /health`: readiness probe with status and feature list
    - `POST /predict`: batch predictions; returns anomaly flags and scores
    - `GET /metrics`: Prometheus metrics exposition
  - Metrics:
    - `requests_total{path,method,status}`: request counter
    - `request_latency_seconds{path,method}`: request latency
    - `inference_latency_seconds`: model inference latency
    - `anomalies_total`: predicted anomaly count
  - Error handling: global exception handler logs and increments `500` counts
- `src/backend/model.py`: Model utilities
  - `ModelConfig`: feature order and IsolationForest params
  - `train_isolation_forest`: trains model on synthetic “normal” data
  - `ensure_model(path, train_if_missing)`: loads or creates `models/iso_forest.joblib`
  - `predict_batch`: returns `(flags, scores)` for an input matrix
- `src/backend/sim.py`: Generator that emits noisy sensor readings with configurable anomaly rate.
- `src/backend/api.py`: Pydantic schemas for requests/responses.

## Streamlit UI
- `src/ui/main.py`: Core UI app
  - Sidebar controls: backend URL, batch size, interval, anomaly rate
  - Actions: Generate Batch, Start/Stop Streaming, Inject Extreme Reading
  - Visualization: rolling charts of voltage/current/frequency with anomaly overlay
  - Diagnostics: shows simulator source; warns if inline fallback is used
- `src/ui/sim_local.py`: Lightweight simulator to keep UI image lean.
  - Import order: `src.ui.sim_local` → `src.backend.sim` → inline fallback

## Configuration
- Backend env vars:
  - `WORKERS`: Uvicorn worker processes (default `1`)
  - `MODEL_DIR`: directory for the model artifact (default `models`)
  - `MODEL_NAME`: artifact filename (default `iso_forest.joblib`)
  - `TRAIN_IF_MISSING`: `1` to auto-train on startup if artifact missing
- UI env vars:
  - `BACKEND_URL`: base URL of backend (`http://backend:8000` in Compose)
  - `STREAMLIT_SERVER_PORT`: port; default 8501

## Local Development
- Install: `uv sync` (or `pip install -e .[dev]`)
- Run API: `python -m src.backend.main` or `uvicorn src.backend.app:app --host 0.0.0.0 --port 8000`
- Run UI: `streamlit run src/ui/main.py`
- Lint/format: `python scripts/dev.py fix`
- Pre-commit hook: `python scripts/dev.py install_hooks`

## Containers
- `docker/Dockerfile.backend`:
  - Python 3.13 slim, install `uv`, `uv sync --frozen`, run Uvicorn
  - Non-root `app` user; `/models` volume for artifact
- `docker/Dockerfile.ui`:
  - Python 3.13 slim, install `uv`, copies only UI sources, installs `streamlit` + `requests`
  - Lean UI image; avoids full ML stack
- `docker/compose.yaml`:
  - Services: `backend` (8000), `ui` (8501)
  - UI uses `BACKEND_URL=http://backend:8000`

## Kubernetes
- Manifests in `k8s/`:
  - `namespace.yaml`: `anomaly`
  - `backend-deployment.yaml`: probes, resource limits, emptyDir for `/models`
  - `backend-service.yaml`: ClusterIP
  - `backend-hpa.yaml`: CPU-based HPA (60% target, 1–5 replicas)
  - `ui-deployment.yaml`: UI pointing to `http://anomaly-backend:8000`
  - `ui-service.yaml`: ClusterIP
  - `kustomization.yaml`: `kubectl apply -k k8s/`
- kind workflow: build images, `kind load docker-image`, apply manifests, port-forward to access services.

## Load Generation
- `scripts/load.py`:
  - Threaded HTTP load generator to `POST /predict`
  - Args: `--url`, `--rps`, `--duration`, `--batch-size`, `--workers`, `--anomaly-rate`
  - Outputs success/error counts and latency stats (avg/p95/p99)

## Observability
- App metrics at `/metrics` (Prometheus format)
- Track: request counts, latencies, inference time, anomaly totals
- K8s HPA requires `metrics-server`

## Performance Tuning
- Increase Uvicorn workers: `WORKERS=2` or higher per CPU
- Scale replicas: HPA or `kubectl scale` for backend deployment
- Tune resource requests/limits for predictable HPA behavior
- Balance batch size and target RPS in load tests

## Security & Hardening (Pointers)
- Non-root container user
- Resource requests/limits
- Probes for liveness/readiness
- Production: ingress with TLS, authn/z, secret management

## Extensibility
- Swap or extend model in `src/backend/model.py` (preserve `predict_batch` signature)
- Add WS/streaming or brokers (Kafka/NATS) + KEDA
- Add monitoring stack (Prometheus/Grafana) via Helm
- Progressive delivery via Argo Rollouts

## Troubleshooting
- UI flat lines/no anomalies:
  - Check “Simulator source” caption; avoid `none`
  - Increase anomaly rate; use “Inject Extreme Reading”
- Load all errors:
  - Verify backend `/health`; confirm target URL; set `WORKERS`
- Build space issues:
  - `docker system prune -af --volumes`; enlarge Docker Desktop disk image

## API Reference
- `GET /health` → `{ "status": "ok", "model": "isolation_forest", "features": [...] }`
- `POST /predict` → `{ "predictions": [{"anomaly": bool, "score": float}, ...], "anomalies": int, "total": int }`
- `GET /metrics` → Prometheus exposition format

## Key Files
- `src/backend/app.py`
- `src/backend/main.py`
- `src/backend/api.py`
- `src/backend/model.py`
- `src/backend/sim.py`
- `src/ui/main.py`
- `src/ui/sim_local.py`
- `scripts/load.py`
- `docker/Dockerfile.backend`
- `docker/Dockerfile.ui`
- `docker/compose.yaml`
- `k8s/*`

