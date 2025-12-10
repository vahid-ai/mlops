# Kind setup

This overlay runs the local data/metadata plane (MinIO, LakeFS, MLflow, Redis) on a Kind cluster and keeps service wiring compatible with future cloud deployments.

## Run locally
1) Create the cluster and deploy services: `tools/scripts/kind_bootstrap.sh` (uses `infra/k8s/kind/kind-config.yaml` and `infra/k8s/kind/manifests`).
2) Access endpoints: LakeFS `http://localhost:8000`, MinIO `http://localhost:9000`, MLflow `http://localhost:5000`, Redis `127.0.0.1:6379`.
3) Tear down: `kind delete cluster --name dfp-kind`.

## Cloud parity
- Replace NodePort services with LoadBalancer/Ingress in a cloud overlay while reusing the same manifests for pods/config.
- Swap MinIO for your cloud object store and adjust `LAKEFS_BLOCKSTORE_*` and MLflow S3 envs.
- Switch LakeFS/MLflow backends from sqlite/ephemeral to managed Postgres by patching the deployments or using Helm charts with the same values.
