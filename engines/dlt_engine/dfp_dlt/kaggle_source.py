"""dlt source for Kaggle datasets."""

import os
import tempfile
from pathlib import Path
from typing import Iterator

import dlt
import pandas as pd
from dlt.sources import DltResource

from .kaggle_compat import patch_kagglesdk_user_agent


def _setup_kaggle_auth() -> None:
    """Set up Kaggle authentication from environment variables.

    Supports both new-style API tokens (KAGGLE_API_TOKEN with KGAT_ prefix)
    and legacy username/key pairs.
    """
    import json

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # If kaggle.json already exists, use it
    if kaggle_json.exists():
        return

    # Try to get credentials from environment
    api_token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    username = os.environ.get("KAGGLE_USERNAME")

    if not api_token:
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY, "
            "or create ~/.kaggle/kaggle.json"
        )

    # For KGAT_ tokens, use the token directly
    if api_token.startswith("KGAT_"):
        # New-style API token - set it in environment for kaggle package
        os.environ["KAGGLE_API_TOKEN"] = api_token
        # Also create a temporary kaggle.json with placeholder values
        # The kaggle package will use the API token instead
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": "_token_", "key": api_token}))
        kaggle_json.chmod(0o600)
    elif username:
        # Legacy username/key pair
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": username, "key": api_token}))
        kaggle_json.chmod(0o600)


def _download_kaggle_dataset(dataset_slug: str, download_dir: Path) -> Path:
    """Download a Kaggle dataset using the Kaggle API.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g., 'dhoogla/kronodroid-2021')
        download_dir: Directory to download files to

    Returns:
        Path to the extracted dataset directory
    """
    _setup_kaggle_auth()
    patch_kagglesdk_user_agent()

    # NOTE: The `kaggle` package authenticates at import time.
    import kaggle  # type: ignore[import-not-found]

    kaggle.api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True)

    return download_dir


def _read_data_files(directory: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Read all CSV and Parquet files from a directory.

    Yields:
        Tuples of (filename_stem, dataframe)
    """
    # Try parquet files first (more common in Kaggle datasets)
    for parquet_file in directory.rglob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        yield parquet_file.stem, df

    # Also check for CSV files
    for csv_file in directory.rglob("*.csv"):
        df = pd.read_csv(csv_file)
        yield csv_file.stem, df


@dlt.source(name="kaggle")
def kaggle_dataset_source(
    dataset_slug: str,
    kaggle_api_token: str | None = None,
) -> Iterator[DltResource]:
    """dlt source for loading Kaggle datasets.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g., 'dhoogla/kronodroid-2021')
        kaggle_api_token: Optional Kaggle API token. If not provided, uses
            KAGGLE_API_TOKEN env var or ~/.kaggle/kaggle.json

    Yields:
        DltResource for each data file (CSV or Parquet) in the dataset
    """
    if kaggle_api_token:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_api_token

    with tempfile.TemporaryDirectory() as tmpdir:
        download_dir = Path(tmpdir)
        _download_kaggle_dataset(dataset_slug, download_dir)

        for table_name, df in _read_data_files(download_dir):
            @dlt.resource(
                name=table_name,
                write_disposition="replace",
                primary_key=None,
            )
            def _table_resource(data: pd.DataFrame = df) -> Iterator[dict]:
                for _, row in data.iterrows():
                    yield row.to_dict()

            yield _table_resource()


@dlt.source(name="kaggle_kronodroid")
def kronodroid_source(
    kaggle_api_token: str | None = None,
) -> Iterator[DltResource]:
    """dlt source specifically for the KronoDroid-2021 dataset.

    This source downloads the KronoDroid Android malware detection dataset
    and yields resources for each data file (emulator and real device data).

    Args:
        kaggle_api_token: Optional Kaggle API token

    Yields:
        DltResource for emulator and real_device data
    """
    dataset_slug = "dhoogla/kronodroid-2021"

    if kaggle_api_token:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_api_token

    with tempfile.TemporaryDirectory() as tmpdir:
        download_dir = Path(tmpdir)
        _download_kaggle_dataset(dataset_slug, download_dir)

        # Find all data files (parquet or csv)
        data_files = list(download_dir.rglob("*.parquet")) + list(download_dir.rglob("*.csv"))

        for data_file in data_files:
            table_name = data_file.stem.lower().replace("-", "_").replace(" ", "_")

            # Read based on file type
            if data_file.suffix == ".parquet":
                df = pd.read_parquet(data_file)
            else:
                df = pd.read_csv(data_file)

            # Add metadata columns
            df["_source_file"] = data_file.name
            df["_ingestion_timestamp"] = pd.Timestamp.now(tz="UTC")

            @dlt.resource(
                name=table_name,
                write_disposition="replace",
                columns={
                    "_source_file": {"data_type": "text"},
                    "_ingestion_timestamp": {"data_type": "timestamp"},
                },
            )
            def _kronodroid_resource(data: pd.DataFrame = df) -> Iterator[dict]:
                for _, row in data.iterrows():
                    yield row.to_dict()

            yield _kronodroid_resource()
