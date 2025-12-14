# Kubernetes/Kind
 
 Issue: ImagePullBackOff is happening because the Spark Operator manifest points at an image tag that doesn’t exist in GHCR.

 Debug Log:
  - The pod was trying to pull ghcr.io/kubeflow/spark-operator:v1beta2-1.3.9-3.5.0 and containerd returned not found.
  - I confirmed the available tags from inside the kind node (v1beta2-1.4.2-3.5.0, v1beta2-1.4.3-3.5.0) and updated the
    manifest to use a real tag: infra/k8s/kind/addons/spark-operator/spark-operator-deployment.yaml:19.
  - After kubectl apply + rollout restart, spark-operator is now Running (1/1).

  If you hit this again, the fastest way to see the real pull error in kind is:

  - docker exec <kind-node> crictl pull <image>:<tag> (it prints the underlying not found / 401 / TLS / DNS error even
    when kubectl describe pod is vague).


# Spark
Issue: Feast was CrashLooping for two separate reasons in the kind “addon” manifests that task feast:up applies:

  - Bad container invocation: the manifest was passing serve ... as raw args, so the image entrypoint tried to exec serve (not a binary). Fixed by running the actual CLI and pointing it at
    the mounted config: infra/k8s/kind/addons/feast/feast-deployment.yaml:19.
  - Spark offline store config + missing dependency: once it could run feast serve, it crashed loading offline_store: type: spark because the feastdev/feature-server:latest image doesn’t
    include pyspark (ModuleNotFoundError: No module named 'pyspark'). Fixed by switching the kind addon’s feature_store.yaml to offline_store: type: file so the feature server can start:
    infra/k8s/kind/addons/feast/feast-config.yaml:23.

  After these changes, the dfp/feast-feature-server pod is Running (1/1) and the server is listening on :6566.

  If you want the feature-server pod itself to keep offline_store: spark, you’ll need a custom image (or another image) that includes pyspark (and likely Spark/Iceberg deps); the upstream
  feastdev/feature-server tags I checked don’t include it.