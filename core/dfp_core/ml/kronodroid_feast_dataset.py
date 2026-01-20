"""Kronodroid training data access via Feast FeatureViews."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class KronodroidDatasetSpec:
    lakefs_repository: str
    lakefs_ref: str
    iceberg_catalog: str
    iceberg_database: str
    source_table: str
    split: str
    feast_project: str
    feast_feature_view: str
    feature_names: tuple[str, ...]
    max_rows: int | None = None

    def stable_id(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _set_lakefs_env(repo: str, ref: str) -> None:
    os.environ["LAKEFS_REPOSITORY"] = repo
    os.environ["LAKEFS_BRANCH"] = ref


def _envsubst(value: str) -> str:
    import re

    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        return os.environ.get(var, "")

    return re.sub(r"\$\{([^}]+)\}", repl, value)


def _spark_session_from_feature_store_yaml(feature_store_yaml_path: str):
    import yaml
    from pyspark.sql import SparkSession

    with open(feature_store_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    spark_conf = ((cfg.get("offline_store") or {}).get("spark_conf") or {}).copy()

    builder = SparkSession.builder.appName("dfp-feast-kronodroid")
    for key, raw_value in spark_conf.items():
        key = _envsubst(str(key))
        value = _envsubst(str(raw_value))
        if key and value:
            builder = builder.config(key, value)

    return builder.getOrCreate()


def _repo_root() -> Path:
    # core/dfp_core/ml/kronodroid_feast_dataset.py -> repo root
    return Path(__file__).resolve().parents[3]


def _ensure_dfp_feast_importable() -> None:
    feast_repo = _repo_root() / "feature_stores" / "feast_store"
    sys.path.insert(0, str(feast_repo))


def ensure_feast_registry(
    *,
    feature_store_yaml_path: str,
    only_feature_views: list[str] | None = None,
) -> None:
    """Create/update the Feast registry inside the current environment.

    This is important inside KFP pods where the registry isn't persisted across runs.
    """
    from feast import FeatureStore
    from feast.repo_config import RepoConfig

    _ensure_dfp_feast_importable()
    import dfp_feast  # type: ignore[import-not-found]

    repo_cfg = RepoConfig.from_yaml(feature_store_yaml_path)
    store = FeatureStore(config=repo_cfg)

    objects: list[Any] = [dfp_feast.malware_sample, dfp_feast.malware_family]
    feature_views: dict[str, Any] = {
        "kronodroid_autoencoder_features": dfp_feast.kronodroid_autoencoder_features,
        "malware_sample_features": dfp_feast.malware_sample_features,
        "malware_family_features": dfp_feast.malware_family_features,
        "malware_batch_features": dfp_feast.malware_batch_features,
    }
    for name, fv in feature_views.items():
        if only_feature_views is None or name in only_feature_views:
            objects.append(fv)

    store.apply(objects)


def load_split_as_pandas(
    *,
    feature_store_yaml_path: str,
    lakefs_repository: str,
    lakefs_ref: str,
    iceberg_catalog: str,
    iceberg_database: str,
    source_table: str,
    split: str,
    feast_feature_refs: list[str],
    entity_column: str = "sample_id",
    timestamp_column: str = "event_timestamp",
    split_column: str = "dataset_split",
    max_rows: int | None = None,
    ensure_registry: bool = False,
) -> "Any":
    """Load one split via Feast (Spark offline store) into a pandas DataFrame."""
    _set_lakefs_env(lakefs_repository, lakefs_ref)

    from feast import FeatureStore
    from feast.repo_config import RepoConfig
    from pyspark.sql import functions as F

    if ensure_registry and feast_feature_refs:
        ensure_feast_registry(
            feature_store_yaml_path=feature_store_yaml_path,
            only_feature_views=[feast_feature_refs[0].split(":", 1)[0]],
        )

    repo_cfg = RepoConfig.from_yaml(feature_store_yaml_path)
    store = FeatureStore(config=repo_cfg)

    spark = _spark_session_from_feature_store_yaml(feature_store_yaml_path)
    full_table = f"{iceberg_catalog}.{iceberg_database}.{source_table}"

    entity_sdf = (
        spark.table(full_table)
        .where(F.col(split_column) == F.lit(split))
        .select(entity_column, timestamp_column)
    )
    if max_rows is not None:
        entity_sdf = entity_sdf.limit(int(max_rows))

    entity_df = entity_sdf.toPandas()
    hist = store.get_historical_features(entity_df=entity_df, features=feast_feature_refs)
    return hist.to_df()


def pandas_to_tensor_dataset(df, feature_columns: Iterable[str]):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    features = df[list(feature_columns)].to_numpy(dtype=np.float32, copy=True)
    x = torch.from_numpy(features)
    return TensorDataset(x)


def resolve_feature_columns(df, feast_feature_view: str, feature_names: list[str]) -> list[str]:
    """Resolve Feast output column names for requested features."""
    resolved: list[str] = []
    for name in feature_names:
        if name in df.columns:
            resolved.append(name)
            continue
        candidates = (
            f"{feast_feature_view}__{name}",
            f"{feast_feature_view}_{name}",
        )
        for cand in candidates:
            if cand in df.columns:
                resolved.append(cand)
                break
        else:
            raise KeyError(f"Missing feature column {name!r} (tried {candidates!r})")
    return resolved


def try_get_iceberg_snapshot_id(
    *,
    feature_store_yaml_path: str | None = None,
    iceberg_catalog: str,
    iceberg_database: str,
    table: str,
) -> str | None:
    """Best-effort snapshot id lookup (requires Iceberg metadata tables)."""
    try:
        from pyspark.sql import SparkSession
    except Exception:
        return None

    spark = (
        _spark_session_from_feature_store_yaml(feature_store_yaml_path)
        if feature_store_yaml_path is not None
        else SparkSession.builder.getOrCreate()
    )
    full_table = f"{iceberg_catalog}.{iceberg_database}.{table}.snapshots"
    try:
        row = spark.sql(f"SELECT max(snapshot_id) AS snapshot_id FROM {full_table}").collect()[0]
        snapshot_id = row["snapshot_id"]
        return str(snapshot_id) if snapshot_id is not None else None
    except Exception:
        return None
