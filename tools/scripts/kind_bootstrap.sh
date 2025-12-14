#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-dfp-kind}"
NAMESPACE="${NAMESPACE:-dfp}"
KIND_CONFIG="${KIND_CONFIG:-${ROOT_DIR}/infra/k8s/kind/kind-config.yaml}"
KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${ROOT_DIR}/infra/k8s/kind/manifests}"
PORT_FORWARD="${PORT_FORWARD:-1}"
WITH_SPARK_OPERATOR="${WITH_SPARK_OPERATOR:-0}"
WITH_FEAST="${WITH_FEAST:-0}"
WITH_KFP="${WITH_KFP:-0}"

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

if [ "${WITH_SPARK_OPERATOR}" = "1" ] || [ "${WITH_SPARK_OPERATOR}" = "true" ]; then
  kubectl apply -k "${ROOT_DIR}/infra/k8s/kind/addons/spark-operator"
  kubectl -n "${NAMESPACE}" rollout status deployment/spark-operator --timeout=180s
fi

if [ "${WITH_FEAST}" = "1" ] || [ "${WITH_FEAST}" = "true" ]; then
  kubectl apply -k "${ROOT_DIR}/infra/k8s/kind/addons/feast"
  kubectl -n "${NAMESPACE}" rollout status deployment/feast-feature-server --timeout=180s
fi

if [ "${WITH_KFP}" = "1" ] || [ "${WITH_KFP}" = "true" ]; then
  bash "${ROOT_DIR}/tools/scripts/kfp_install.sh"
fi

echo "Kind cluster '${CLUSTER_NAME}' is ready."

if [ "${PORT_FORWARD}" = "1" ] || [ "${PORT_FORWARD}" = "true" ]; then
  if ! bash "${ROOT_DIR}/tools/scripts/kind_portforward.sh" start; then
    echo "Port-forwarding failed; check logs under ${ROOT_DIR}/.task/port-forwards/${CLUSTER_NAME}/${NAMESPACE}/" >&2
  fi
  echo "Endpoints: LakeFS http://localhost:8000, MinIO API http://localhost:19000, MinIO Console http://localhost:19001, MLflow http://localhost:5050, Redis 127.0.0.1:16379, Feast http://localhost:16566"
else
  echo "Port-forwarding is disabled (set PORT_FORWARD=1 to enable)."
  echo "Run: task port-forward"
fi
