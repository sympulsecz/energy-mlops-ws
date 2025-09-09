#!/usr/bin/env bash
set -euo pipefail

# Rebuild local images, load them into a kind cluster, and restart deployments
# Usage: ./scripts/kind_redeploy.sh [cluster-name]

CLUSTER_NAME=${1:-anomaly}

echo "[1/5] Building images..."
docker build -f docker/Dockerfile.backend -t anomaly-backend:latest .
docker build -f docker/Dockerfile.ui -t anomaly-ui:latest .

echo "[2/5] Loading images into kind cluster '$CLUSTER_NAME'..."
kind load docker-image anomaly-backend:latest --name "$CLUSTER_NAME"
kind load docker-image anomaly-ui:latest --name "$CLUSTER_NAME"

echo "[3/5] Ensuring namespace exists..."
kubectl apply -f k8s/namespace.yaml >/dev/null

echo "[4/5] Restarting deployments to pick up new images..."
kubectl -n anomaly rollout restart deploy anomaly-backend || true
kubectl -n anomaly rollout restart deploy anomaly-ui || true

echo "[5/5] Waiting for rollout to finish..."
kubectl -n anomaly rollout status deploy/anomaly-backend
kubectl -n anomaly rollout status deploy/anomaly-ui

echo "Done. Deployed latest images to kind cluster '$CLUSTER_NAME'."

