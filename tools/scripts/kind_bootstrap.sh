#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-dfp-kind}"
NAMESPACE="${NAMESPACE:-dfp}"
KIND_CONFIG="${KIND_CONFIG:-${ROOT_DIR}/infra/k8s/kind/kind-config.yaml}"
KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${ROOT_DIR}/infra/k8s/kind/manifests}"
PORT_FORWARD="${PORT_FORWARD:-1}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required (kind uses Docker). Install Docker Desktop or another Docker runtime." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable. Start your Docker runtime, then verify \`docker ps\` works." >&2
  exit 1
fi

if ! command -v kind >/dev/null 2>&1; then
  echo "kind is required. Install from https://kind.sigs.k8s.io/docs/user/quick-start/." >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required. Install from https://kubernetes.io/docs/tasks/tools/." >&2
  exit 1
fi

if [ ! -f "${KIND_CONFIG}" ]; then
  echo "KIND_CONFIG not found: ${KIND_CONFIG}" >&2
  exit 1
fi

if [ ! -d "${KUSTOMIZE_DIR}" ]; then
  echo "KUSTOMIZE_DIR not found: ${KUSTOMIZE_DIR}" >&2
  exit 1
fi

if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG}"
fi

kubectl apply -k "${KUSTOMIZE_DIR}"

kubectl -n "${NAMESPACE}" rollout status deployment/minio --timeout=180s
kubectl -n "${NAMESPACE}" rollout status deployment/redis --timeout=180s
kubectl -n "${NAMESPACE}" rollout status deployment/lakefs --timeout=180s
kubectl -n "${NAMESPACE}" rollout status deployment/mlflow --timeout=180s
kubectl -n "${NAMESPACE}" rollout status deployment/lakefs-postgres --timeout=180s

echo "Kind cluster '${CLUSTER_NAME}' is ready."

if [ "${PORT_FORWARD}" = "1" ] || [ "${PORT_FORWARD}" = "true" ]; then
  if ! bash "${ROOT_DIR}/tools/scripts/kind_portforward.sh" start; then
    echo "Port-forwarding failed; check logs under ${ROOT_DIR}/.task/port-forwards/${CLUSTER_NAME}/${NAMESPACE}/" >&2
  fi
  echo "Endpoints: LakeFS http://localhost:8000, MinIO http://localhost:9000, MLflow http://localhost:5050, Redis 127.0.0.1:6379"
else
  echo "Port-forwarding is disabled (set PORT_FORWARD=1 to enable)."
  echo "Run: task port-forward"
fi
