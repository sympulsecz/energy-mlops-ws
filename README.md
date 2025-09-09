# energy-mlops-ws
MLOps Workshop for Energy Sector: a simple lightweight intro to "From IPython Notebook to Prod"

Quickstart (Backend)
- Install deps: `uv sync --extra dev` (or `pip install -e .[dev]`)
- Run API: `uv run python -m src.backend.main` or `uv run uvicorn src.backend.app:app --host 0.0.0.0 --port 8000`
- Health check: `GET http://localhost:8000/health`
- Predict: `POST http://localhost:8000/predict` with `{ "readings": [{"voltage":230, "current":10, "frequency":50}] }`
- Metrics: `GET http://localhost:8000/metrics`

The first run trains a small IsolationForest on synthetic data and saves `models/iso_forest.joblib`.

Quickstart (UI)
- Start backend first (see above)
- Launch UI: `uv run streamlit run src/ui/main.py`
- In the sidebar, set Backend URL (default `http://localhost:8000`), choose batch size, interval, and anomaly rate.
- Click "Generate Batch" to send one batch, or "Start Streaming" to continuously send batches and visualize anomalies.

Containerize Backend
- Build image: `docker build -f docker/Dockerfile.backend -t anomaly-backend .`
- Run container: `docker run --rm -p 8000:8000 -e TRAIN_IF_MISSING=1 -e MODEL_DIR=/models -v $(pwd)/models:/models anomaly-backend`
  - The volume mount persists the model artifact to your host `./models`.
  - API available at `http://localhost:8000`.

Containerize UI
- Build image: `docker build -f docker/Dockerfile.ui -t anomaly-ui .`
- Run container: `docker run --rm -p 8501:8501 -e BACKEND_URL=http://localhost:8000 anomaly-ui`
  - UI available at `http://localhost:8501`
  - By default, the UI points to `BACKEND_URL`. Override as needed.

Docker Compose (both services)
- From `docker/` directory: `docker compose up --build`
  - Backend: `http://localhost:8000`
  - UI: `http://localhost:8501` (configured with `BACKEND_URL=http://backend:8000`)

Kubernetes (kind) Quickstart
- Create a cluster: `kind create cluster --name anomaly`
- Install metrics-server (for HPA):
  - `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`
  - For kind, add insecure kubelet TLS flag:
    - `kubectl -n kube-system patch deploy metrics-server --type='json' -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'`
  - Verify rollout and metrics:
    - `kubectl -n kube-system rollout status deploy/metrics-server`
    - `kubectl top nodes`
- Build images locally:
  - `docker build -f docker/Dockerfile.backend -t anomaly-backend:latest .`
  - `docker build -f docker/Dockerfile.ui -t anomaly-ui:latest .`
- Load images into kind:
  - `kind load docker-image anomaly-backend:latest --name anomaly`
  - `kind load docker-image anomaly-ui:latest --name anomaly`
- Deploy manifests:
  - `kubectl apply -k k8s/`
- Redeploy latest images after changes:
  - `./scripts/kind_redeploy.sh` (rebuilds, kind-loads, and restarts deployments)
  - Or manually: `kubectl -n anomaly rollout restart deploy anomaly-backend anomaly-ui`
- Access services (port-forward):
  - Backend: `kubectl -n anomaly port-forward svc/anomaly-backend 8000:8000`
  - UI: `kubectl -n anomaly port-forward svc/anomaly-ui 8501:8501`
- Observe HPA:
  - `kubectl -n anomaly get hpa -w`
  - `kubectl -n anomaly get pods -w`

Notes
- Images use `imagePullPolicy: IfNotPresent`; with kind, use `kind load docker-image ...` after building.
- Backend HPA targets 60% CPU utilization; adjust requests/limits as needed in `k8s/backend-deployment.yaml`.

Documentation
- Technical overview: `docs/TECHNICAL_OVERVIEW.md`
- Workshop guide: `docs/WORKSHOP_GUIDE.md`

Load Generation (local or k8s via port-forward)
- Local backend (Docker or bare):
  - `python scripts/load.py --url http://localhost:8000 --rps 100 --duration 60 --batch-size 64 --workers 16`
- K8s with port-forward:
  - In one terminal: `kubectl -n anomaly port-forward svc/anomaly-backend 8000:8000`
  - In another: `python scripts/load.py --url http://localhost:8000 --rps 150 --duration 120 --batch-size 64 --workers 32`
  - Observe HPA scaling with `kubectl -n anomaly get hpa -w` and pods with `kubectl -n anomaly get pods -w`.

End-to-End Test Plan
- Local (bare Python):
  1) `uv sync --extra dev` (or `pip install -e .[dev]`)
  2) Backend: `uv run python -m src.backend.main`
  3) UI: `uv run streamlit run src/ui/main.py`
  4) Use UI to generate batches/stream; optionally run `python scripts/load.py ...` for sustained load.
- Docker Compose:
  1) `cd docker && docker compose up --build`
  2) UI at `http://localhost:8501` (talks to backend)
  3) Optionally run load: `uv run python ../scripts/load.py --url http://localhost:8000 --rps 100 --duration 60`
- Kubernetes (kind):
  1) `kind create cluster --name anomaly`
  2) Install metrics-server (see above) and verify `kubectl top nodes`
  3) Build images and load into kind (see above)
  4) `kubectl apply -k k8s/`
  5) Port-forward UI: `kubectl -n anomaly port-forward svc/anomaly-ui 8501:8501`
  6) (Optional) Port-forward backend and run `uv run python scripts/load.py ...` to drive HPA
  7) Watch scaling: `kubectl -n anomaly get hpa -w` and `kubectl -n anomaly get pods -w`
