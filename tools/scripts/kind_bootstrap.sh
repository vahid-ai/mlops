#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-dfp-kind}"
KIND_CONFIG="${KIND_CONFIG:-${ROOT_DIR}/infra/k8s/kind/kind-config.yaml}"
KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${ROOT_DIR}/infra/k8s/kind/manifests}"

if ! command -v kind >/dev/null 2>&1; then
  echo "kind is required. Install from https://kind.sigs.k8s.io/docs/user/quick-start/." >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required. Install from https://kubernetes.io/docs/tasks/tools/." >&2
  exit 1
fi

if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG}"
fi

kubectl apply -k "${KUSTOMIZE_DIR}"
kubectl wait --namespace dfp --for=condition=available deployment/minio deployment/redis deployment/lakefs deployment/mlflow deployment/lakefs-postgres --timeout=180s

echo "Kind cluster '${CLUSTER_NAME}' is ready."
echo "Endpoints: LakeFS http://localhost:8000, MinIO http://localhost:9000, MLflow http://localhost:5000, Redis 127.0.0.1:6379"
