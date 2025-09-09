# MLOps in Energy — Workshop Guide

This guide is the hands-on path for participants to build, deploy, and scale a real-time grid anomaly detection microservice with a Streamlit UI. It favors practical tooling over modeling theory.

## Learning Objectives
- Understand model serving with FastAPI and Uvicorn
- Containerize services with Docker and manage deps using `uv`
- Deploy to a local Kubernetes cluster and scale with HPA
- Expose app metrics and reason about performance under load
- Build a simple UI that exercises and visualizes the service end-to-end

## Prerequisites
- Installed locally:
  - Docker Desktop (4+ CPUs, 6–8 GB RAM)
  - `kubectl`, `kind`, `helm` (optional), `k9s` (optional), `kubectx`
  - Python 3.13 and `uv` (or use Docker only)
- Clone repository and ensure `uv.lock` is present

## Workshop Flow (2 sessions)
- Session 1 (Local + Containers)
  - Explore repo, run backend locally
  - Run Streamlit UI locally
  - Containerize and run via Docker Compose
- Session 2 (Kubernetes + Scaling)
  - Build and load images into kind
  - Deploy manifests with HPA
  - Generate load and watch autoscaling
  - Optional: monitoring, variants

## 1) Explore the Repository
- Key components:
  - `src/backend/`: API, model, simulator
  - `src/ui/`: Streamlit UI and local simulator
  - `scripts/`: dev helpers and load generator
  - `docker/`: Dockerfiles and Compose
  - `k8s/`: namespace, deployments, services, HPA, Kustomize
- Examine:
  - `src/backend/app.py` (endpoints and metrics)
  - `src/backend/model.py` (artifact handling and predictions)
  - `src/ui/main.py` (controls, charts, prediction loop)

## 2) Run Locally (No Containers)
- Install Python deps:
  - `uv sync` (or `pip install -e .[dev]`)
- Start API:
  - `uv run python -m src.backend.main`
  - Docs: `GET http://localhost:8000/docs`
- Start UI (in a new terminal):
  - `streamlit run src/ui/main.py`
  - Use sidebar controls, click “Start Streaming”. If anomalies are rare, increase “Anomaly rate” or click “Inject Extreme Reading”.

## 3) Run with Docker Compose
- From the `docker/` directory:
  - `docker compose up --build`
  - UI at `http://localhost:8501`
  - Backend at `http://localhost:8000`
- The UI is configured with `BACKEND_URL=http://backend:8000` in Compose.
- Exercise:
  - Use the UI to stream data and watch anomalies increment.
  - Try a higher batch size and shorter interval; observe responsiveness.

## 4) Deploy to Kubernetes (kind)
- Create a cluster:
  - `kind create cluster --name anomaly`
- Install and patch metrics-server (for HPA on kind):
  - `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`
  - Add insecure TLS flag for local kubelets (kind clusters):
    - `kubectl -n kube-system patch deploy metrics-server --type='json' -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'`
  - Rollout and verify availability:
    - `kubectl -n kube-system rollout status deploy/metrics-server`
    - `kubectl get apiservices | grep metrics`
    - Now `kubectl top nodes` and `kubectl top pods -A` should work
- Build images on your host:
  - `docker build -f docker/Dockerfile.backend -t anomaly-backend:latest .`
  - `docker build -f docker/Dockerfile.ui -t anomaly-ui:latest .`
- Load images into kind:
  - `kind load docker-image anomaly-backend:latest --name anomaly`
  - `kind load docker-image anomaly-ui:latest --name anomaly`
- Deploy manifests:
  - `kubectl apply -k k8s/`
- If you rebuild images and don’t see changes in K8s:
  - Use the helper: `./scripts/kind_redeploy.sh`
  - Or: rebuild → kind load → `kubectl -n anomaly rollout restart deploy anomaly-backend anomaly-ui`
- Access services via port-forward:
  - Backend: `kubectl -n anomaly port-forward svc/anomaly-backend 8000:8000`
- UI: `kubectl -n anomaly port-forward svc/anomaly-ui 8501:8501`
- Open the UI and stream data; confirm anomalies appear.
  - Tip: enable “New connection each request” in the UI sidebar to improve request distribution across pods (disables HTTP keep-alive pinning).

## 5) Generate Load and Observe Autoscaling
- In one terminal, keep port-forward to backend:
  - `kubectl -n anomaly port-forward svc/anomaly-backend 8000:8000`
- In another terminal, run the load generator:
  - `uv run python scripts/load.py --url http://localhost:8000 --rps 150 --duration 120 --batch-size 64 --workers 32 --no-keepalive`
- Watch HPA and pods:
  - `kubectl -n anomaly get hpa -w`
  - `kubectl -n anomaly get pods -w`
- Discussion prompts:
  - How does batch size affect average latency and throughput?
  - How quickly does HPA react to load? Does it scale down?
  - What are the tradeoffs between more workers vs. more replicas?

## 6) Metrics and Observability
- Visit `http://localhost:8000/metrics` to see Prometheus metrics
- Key metrics:
  - `requests_total{path,method,status}`
  - `request_latency_seconds_bucket` / `_sum` / `_count`
  - `inference_latency_seconds_*`
  - `anomalies_total`
- Optional (if Helm available):
  - Install kube-prometheus-stack to explore metrics in Grafana
  - Create a basic dashboard for request rate and latency

## 7) Exercises & Variants
- Tune the backend:
  - Add `WORKERS=2` and compare latency under load
  - Increase HPA max replicas to 10; compare scale behavior
- Change model thresholding:
  - Modify contamination rate in `ModelConfig` and re-train
  - Observe impact on false positives/negatives
- LitServe variant (stretch):
  - Swap FastAPI for a LitServe service while preserving the API contract
- Persistence:
  - Replace `emptyDir` with a PVC for model artifacts

## Reference Commands
- Local API: `python -m src.backend.main`
- Local UI: `streamlit run src/ui/main.py`
- Compose: `cd docker && docker compose up --build`
- kind cluster: `kind create cluster --name anomaly`
- Apply k8s: `kubectl apply -k k8s/`
- Port-forward UI: `kubectl -n anomaly port-forward svc/anomaly-ui 8501:8501`
- Load: `python scripts/load.py --url http://localhost:8000 --rps 100 --duration 60 --batch-size 64 --workers 16`

## Cleanup
- Compose: `docker compose down` (in the `docker/` folder)
- kind: `kind delete cluster --name anomaly`
- Docker cache: `docker system prune -af --volumes`
