"""KronoDroid dataset ingestion pipeline using dlt."""

import os
from pathlib import Path

import dlt

from .kaggle_source import kronodroid_source
from .minio_destination import (
    LakeFSConfig,
    MinioConfig,
    ensure_minio_bucket,
    get_lakefs_destination,
    get_minio_destination,
)


def run_kronodroid_pipeline(
    destination: str = "minio",
    dataset_name: str = "kronodroid_raw",
    lakefs_branch: str | None = None,
) -> dlt.Pipeline:
    """Run the KronoDroid data ingestion pipeline.

    Downloads the KronoDroid-2021 dataset from Kaggle and loads it into
    either MinIO directly or via LakeFS for versioning.

    Args:
        destination: Either 'minio' or 'lakefs'
        dataset_name: Name for the dataset in the destination
        lakefs_branch: LakeFS branch name (only used if destination='lakefs')

    Returns:
        The dlt pipeline object with run info

    Example:
        >>> pipeline = run_kronodroid_pipeline(destination='minio')
        >>> print(pipeline.last_trace.last_normalize_info)
    """
    kaggle_token = os.getenv("KAGGLE_API_TOKEN")

    if destination == "lakefs":
        config = LakeFSConfig.from_env()
        if lakefs_branch:
            config.branch = lakefs_branch
        dest = get_lakefs_destination(config)
        pipeline_name = f"kronodroid_lakefs_{config.branch}"
    else:
        config = MinioConfig.from_env()
        ensure_minio_bucket(config)
        dest = get_minio_destination(config)
        pipeline_name = "kronodroid_minio"

    pipeline = dlt.pipeline(
        pipeline_name=pipeline_name,
        destination=dest,
        dataset_name=dataset_name,
    )

    source = kronodroid_source(kaggle_api_token=kaggle_token)

    info = pipeline.run(source)
    print(f"Pipeline completed: {info}")

    return pipeline


def run_kronodroid_to_parquet(
    output_dir: str | Path = "data/kronodroid",
    kaggle_token: str | None = None,
) -> Path:
    """Download KronoDroid dataset and save as Parquet files locally.

    This is useful for local development and testing before pushing
    to MinIO/LakeFS.

    Args:
        output_dir: Directory to save Parquet files
        kaggle_token: Kaggle API token (uses env var if not provided)

    Returns:
        Path to the output directory
    """
    import tempfile
    import zipfile

    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if kaggle_token:
        os.environ["KAGGLE_KEY"] = kaggle_token
    elif "KAGGLE_API_TOKEN" in os.environ:
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmpdir:
        api.dataset_download_files(
            "dhoogla/kronodroid-2021",
            path=tmpdir,
            unzip=True,
        )

        for csv_file in Path(tmpdir).rglob("*.csv"):
            df = pd.read_csv(csv_file)
            table_name = csv_file.stem.lower().replace("-", "_").replace(" ", "_")
            parquet_path = output_path / f"{table_name}.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"Saved {len(df)} rows to {parquet_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run KronoDroid data ingestion pipeline"
    )
    parser.add_argument(
        "--destination",
        choices=["minio", "lakefs", "local"],
        default="minio",
        help="Destination for the data",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="LakeFS branch (only for lakefs destination)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/kronodroid",
        help="Output directory (only for local destination)",
    )

    args = parser.parse_args()

    if args.destination == "local":
        output = run_kronodroid_to_parquet(output_dir=args.output_dir)
        print(f"Data saved to: {output}")
    else:
        pipeline = run_kronodroid_pipeline(
            destination=args.destination,
            lakefs_branch=args.branch,
        )
        print(f"Pipeline state: {pipeline.state}")
