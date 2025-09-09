#!/usr/bin/env bash
set -euo pipefail

# Helper to install and patch metrics-server for kind clusters.
# Usage: ./scripts/setup_kind_metrics.sh

NS=kube-system
echo "Applying metrics-server components..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

echo "Patching metrics-server to allow insecure kubelet TLS (kind)..."
kubectl -n "$NS" patch deploy metrics-server \
  --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' || true

echo "Waiting for metrics-server rollout..."
kubectl -n "$NS" rollout status deploy/metrics-server

echo "Checking APIService availability..."
kubectl get apiservices | grep metrics || true

echo "Testing metrics API..."
kubectl top nodes || echo "Metrics API not ready yet; retry in ~30s"

echo "Done."

