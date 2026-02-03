"""Kubeflow Pipelines helpers for Kubernetes plumbing.

These helpers wrap `kfp.kubernetes.*` calls so pipelines stay readable and
credential wiring is consistent across components.
"""

from __future__ import annotations

from kfp import kubernetes
from kfp.dsl import PipelineTask

from orchestration.kubeflow.dfp_kfp.config import (
    DEFAULT_FEAST_CONFIGMAP_NAME,
    DEFAULT_FEAST_MOUNT_PATH,
    LAKEFS_SECRET_KEY_TO_ENV,
    MINIO_SECRET_KEY_TO_ENV,
)


def use_minio_credentials(task: PipelineTask, *, secret_name: str) -> PipelineTask:
    """Inject MinIO credentials into a task via env vars."""
    kubernetes.use_secret_as_env(
        task=task,
        secret_name=secret_name,
        secret_key_to_env=MINIO_SECRET_KEY_TO_ENV,
    )
    return task


def use_lakefs_credentials(task: PipelineTask, *, secret_name: str) -> PipelineTask:
    """Inject LakeFS credentials into a task via env vars."""
    kubernetes.use_secret_as_env(
        task=task,
        secret_name=secret_name,
        secret_key_to_env=LAKEFS_SECRET_KEY_TO_ENV,
    )
    return task


def mount_feast_repo(
    task: PipelineTask,
    *,
    config_map_name: str = DEFAULT_FEAST_CONFIGMAP_NAME,
    mount_path: str = DEFAULT_FEAST_MOUNT_PATH,
) -> PipelineTask:
    """Mount a ConfigMap containing a Feast repo at `mount_path`."""
    kubernetes.use_config_map_as_volume(
        task=task,
        config_map_name=config_map_name,
        mount_path=mount_path,
    )
    return task

