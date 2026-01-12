# Spark Operator for Kubernetes

Spark Operator enables running Apache Spark applications as Kubernetes-native resources using the `SparkApplication` Custom Resource Definition (CRD).

## Version Information

- **Operator Image**: `ghcr.io/kubeflow/spark-operator:v1beta2-1.4.3-3.5.0`
- **API Version**: `v1beta2`
- **Supported Spark Version**: 3.5.x
- **Namespace**: `dfp`
- **Service Account**: `spark-operator`

## Components

### Files in this directory:

1. **`spark-operator-crds.yaml`** - Custom Resource Definitions
   - `SparkApplication` - Defines a Spark application to run
   - `ScheduledSparkApplication` - Defines a scheduled Spark application

2. **`spark-operator-rbac.yaml`** - RBAC configuration
   - ServiceAccount: `spark-operator`
   - ClusterRole with permissions for pods, services, configmaps, etc.
   - ClusterRoleBinding

3. **`spark-operator-deployment.yaml`** - Operator deployment
   - Single replica deployment
   - Watches all namespaces (namespace="" arg)
   - Webhook disabled for simplicity

4. **`kustomization.yaml`** - Kustomize configuration
   - Ties all resources together
   - Sets target namespace to `dfp`

## Deployment

### Deploy Spark Operator:
```bash
# Using kubectl
kubectl apply -k infra/k8s/kind/addons/spark-operator/

# Using task
task spark-operator:up

# Verify deployment
kubectl -n dfp get deployment spark-operator
kubectl -n dfp rollout status deployment/spark-operator --timeout=180s
```

### Remove Spark Operator:
```bash
# Using kubectl
kubectl delete -k infra/k8s/kind/addons/spark-operator/

# Using task
task spark-operator:down
```

## Spark Image Compatibility

The Spark Operator version `v1beta2-1.4.3-3.5.0` is built for **Spark 3.5.0**. When creating SparkApplications, use compatible Spark images:

### Recommended Spark Images:
- `apache/spark:3.5.7-python3` (default, recommended)
- `bitnami/spark:3.5.0`
- `apache/spark:3.5.0`
- `gcr.io/spark-operator/spark:v3.5.0`

### Required Dependencies (for Iceberg + S3):
When using Iceberg with S3/MinIO, include these packages in your SparkApplication:
- `org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0`
- `org.apache.hadoop:hadoop-aws:3.3.4`
- `com.amazonaws:aws-java-sdk-bundle:1.12.262`
- `org.apache.spark:spark-avro_2.12:3.5.0`

## Usage Example

### Creating a SparkApplication:

```yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: my-spark-job
  namespace: dfp
spec:
  type: Python
  mode: cluster
  pythonVersion: "3"
  sparkVersion: "3.5.3"
  image: "apache/spark:3.5.7-python3"
  mainApplicationFile: "local:///opt/spark/work-dir/job.py"

  driver:
    cores: 1
    memory: "2g"
    serviceAccount: spark-operator

  executor:
    instances: 2
    cores: 1
    memory: "2g"
```

### Using the Python API:

See `orchestration/kubeflow/dfp_kfp/components/kronodroid_spark_operator_transform_component.py` for a complete example.

```python
from orchestration.kubeflow.dfp_kfp.components.kronodroid_spark_operator_transform_component import (
    KronodroidSparkOperatorConfig,
    run,
)

cfg = KronodroidSparkOperatorConfig(
    namespace="dfp",
    spark_image="apache/spark:3.5.7-python3",
    timeout_seconds=1800,
)

success = run(cfg)
```

## Monitoring

### Check SparkApplication status:
```bash
# List all SparkApplications
kubectl -n dfp get sparkapplication

# Describe a specific SparkApplication
kubectl -n dfp describe sparkapplication <app-name>

# Get detailed status
kubectl -n dfp get sparkapplication <app-name> -o yaml

# View driver logs
kubectl -n dfp logs <app-name>-driver

# View executor logs
kubectl -n dfp logs <app-name>-exec-1
```

### Common Issues:

**ImagePullBackOff**:
- Ensure the Spark image exists and is accessible
- For kind clusters, pre-load the image: `kind load docker-image <image> --name dfp-kind`

**Permission Errors**:
- Verify the ServiceAccount has proper RBAC permissions
- Check that `spark-operator` ServiceAccount exists in the namespace

**Application Stuck in SUBMITTED**:
- Check Spark Operator logs: `kubectl -n dfp logs deployment/spark-operator`
- Verify driver pod status: `kubectl -n dfp get pods -l spark-role=driver`

## Architecture

```
User/Script
    ↓
kubectl apply SparkApplication
    ↓
Spark Operator (controller)
    ↓
Creates: Driver Pod → Executor Pods
    ↓
SparkApplication status updates
```

The Spark Operator watches for SparkApplication resources and:
1. Creates a driver pod with the Spark driver process
2. The driver creates executor pods as needed
3. Updates the SparkApplication status throughout the lifecycle
4. Cleans up resources when the application completes

## References

- [Spark Operator GitHub](https://github.com/kubeflow/spark-operator)
- [Spark Operator Documentation](https://github.com/kubeflow/spark-operator/blob/master/docs/quick-start-guide.md)
- [SparkApplication API Reference](https://github.com/kubeflow/spark-operator/blob/master/docs/api-docs.md)
