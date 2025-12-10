# Data Lineage

Feature tables live in Iceberg and are versioned via LakeFS branches/commits. Feast feature views reference commit-aware table URIs so MLflow runs can capture exact snapshots.
