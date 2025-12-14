#!/usr/bin/env bash
set -euo pipefail

KFP_VERSION="${KFP_VERSION:-2.1.3}"
KFP_NAMESPACE="${KFP_NAMESPACE:-kubeflow}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required." >&2
  exit 1
fi

echo "Installing Kubeflow Pipelines ${KFP_VERSION} into namespace ${KFP_NAMESPACE} ..."

kubectl get ns "${KFP_NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${KFP_NAMESPACE}"

# Uses upstream Kustomize manifests; requires network access to fetch the remote Git repo.
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${KFP_VERSION}"

echo "KFP install applied. Wait for pods:"
echo "  kubectl -n ${KFP_NAMESPACE} get pods"
echo "When ready, port-forward the UI:"
echo "  kubectl -n ${KFP_NAMESPACE} port-forward svc/ml-pipeline-ui 8080:80"

