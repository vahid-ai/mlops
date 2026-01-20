"""Dataset loaders for training and evaluation."""

from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:  # pragma: no cover - tooling only
    np = None
    pd = None
    torch = None
    Dataset = object
    DataLoader = None


class DatasetLoader:
    """Generic dataset loader for Iceberg/Avro backed by LakeFS."""

    def __init__(self, source: str):
        self.source = source

    def load(self) -> Any:
        # Placeholder for loading from Iceberg/Avro backed by LakeFS
        return []


class FeastAutoencoderDataset(Dataset):
    """PyTorch Dataset for autoencoder training from Feast features.

    Loads features from a pandas DataFrame (retrieved from Feast or Iceberg)
    and applies optional z-score normalization.

    Args:
        features_df: DataFrame with features
        feature_columns: List of feature column names to use
        normalize: Whether to apply z-score normalization
        mean: Optional pre-computed mean for normalization
        std: Optional pre-computed std for normalization
    """

    def __init__(
        self,
        features_df: "pd.DataFrame",
        feature_columns: List[str],
        normalize: bool = True,
        mean: Optional["np.ndarray"] = None,
        std: Optional["np.ndarray"] = None,
    ):
        if torch is None:
            raise RuntimeError("torch not installed")

        self.feature_columns = feature_columns

        # Extract feature values
        data = features_df[feature_columns].values.astype(np.float32)

        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)

        # Normalization
        if normalize and mean is None:
            # Compute normalization parameters
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8  # Avoid division by zero
            data = (data - self.mean) / self.std
        elif mean is not None and std is not None:
            # Use provided normalization parameters
            self.mean = mean
            self.std = std
            data = (data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> "torch.Tensor":
        return self.data[idx]

    def get_normalization_params(self) -> Optional[Dict[str, "np.ndarray"]]:
        """Return normalization parameters for inference."""
        if self.mean is not None:
            return {"mean": self.mean, "std": self.std}
        return None


def create_dataloaders_from_dataframes(
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    feature_columns: List[str],
    batch_size: int = 512,
    num_workers: int = 4,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[Dict[str, "np.ndarray"]]]:
    """Create DataLoaders from pre-split DataFrames.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        feature_columns: List of feature column names
        batch_size: Batch size for DataLoaders
        num_workers: Number of data loading workers
        normalize: Whether to apply z-score normalization

    Returns:
        Tuple of (train_loader, val_loader, test_loader, normalization_params)
    """
    # Create training dataset first to get normalization params
    train_dataset = FeastAutoencoderDataset(
        features_df=train_df,
        feature_columns=feature_columns,
        normalize=normalize,
    )

    norm_params = train_dataset.get_normalization_params()

    # Create val/test datasets with training normalization
    val_dataset = FeastAutoencoderDataset(
        features_df=val_df,
        feature_columns=feature_columns,
        normalize=normalize,
        mean=norm_params["mean"] if norm_params else None,
        std=norm_params["std"] if norm_params else None,
    )

    test_dataset = FeastAutoencoderDataset(
        features_df=test_df,
        feature_columns=feature_columns,
        normalize=normalize,
        mean=norm_params["mean"] if norm_params else None,
        std=norm_params["std"] if norm_params else None,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, norm_params


def load_splits_from_iceberg(
    spark_session: Any,
    catalog: str,
    database: str,
    table: str,
    feature_columns: List[str],
    split_column: str = "dataset_split",
    max_rows_per_split: Optional[int] = None,
) -> Tuple[Dict[str, "pd.DataFrame"], Dict[str, Any]]:
    """Load train/val/test splits from an Iceberg table.

    Args:
        spark_session: Active SparkSession
        catalog: Iceberg catalog name
        database: Database name
        table: Table name
        feature_columns: List of feature column names
        split_column: Column containing split labels (train/validation/test)
        max_rows_per_split: Optional limit per split (for testing)

    Returns:
        Tuple of (splits dict, lineage metadata dict)
    """
    full_table = f"{catalog}.{database}.{table}"
    df = spark_session.read.table(full_table)

    # Get Iceberg snapshot ID for lineage
    snapshot_id = None
    try:
        snapshots_df = spark_session.sql(
            f"SELECT snapshot_id FROM {full_table}.snapshots "
            f"ORDER BY committed_at DESC LIMIT 1"
        )
        row = snapshots_df.first()
        if row:
            snapshot_id = str(row["snapshot_id"])
    except Exception:
        pass

    # Select relevant columns
    columns = ["sample_id", "event_timestamp", split_column] + feature_columns
    available_columns = [c for c in columns if c in df.columns]
    df = df.select(*available_columns)

    # Convert to pandas splits
    splits = {}
    sample_counts = {}

    for split_name in ["train", "validation", "test"]:
        split_df = df.filter(df[split_column] == split_name)
        if max_rows_per_split:
            split_df = split_df.limit(max_rows_per_split)
        pdf = split_df.toPandas()
        splits[split_name] = pdf
        sample_counts[f"{split_name}_samples"] = len(pdf)

    lineage = {
        "iceberg_table": full_table,
        "iceberg_snapshot_id": snapshot_id,
        **sample_counts,
    }

    return splits, lineage
