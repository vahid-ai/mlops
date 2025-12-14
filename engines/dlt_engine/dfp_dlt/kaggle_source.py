"""dlt source for Kaggle datasets."""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import dlt
import pandas as pd
from dlt.sources import DltResource


def _download_kaggle_dataset(dataset_slug: str, download_dir: Path) -> Path:
    """Download a Kaggle dataset using the Kaggle API.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g., 'dhoogla/kronodroid-2021')
        download_dir: Directory to download files to

    Returns:
        Path to the extracted dataset directory
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # Fix for User-Agent being None - explicitly set it after authentication
    if api.configuration.user_agent is None:
        api.configuration.user_agent = "Kaggle/1.0"

    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True)

    return download_dir


def _read_csv_files(directory: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Read all CSV files from a directory.

    Yields:
        Tuples of (filename_stem, dataframe)
    """
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
        DltResource for each CSV file in the dataset
    """
    if kaggle_api_token:
        os.environ["KAGGLE_KEY"] = kaggle_api_token

    with tempfile.TemporaryDirectory() as tmpdir:
        download_dir = Path(tmpdir)
        _download_kaggle_dataset(dataset_slug, download_dir)

        for table_name, df in _read_csv_files(download_dir):
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
        os.environ["KAGGLE_KEY"] = kaggle_api_token

    with tempfile.TemporaryDirectory() as tmpdir:
        download_dir = Path(tmpdir)
        _download_kaggle_dataset(dataset_slug, download_dir)

        for csv_file in download_dir.rglob("*.csv"):
            table_name = csv_file.stem.lower().replace("-", "_").replace(" ", "_")
            df = pd.read_csv(csv_file)

            # Add metadata columns
            df["_source_file"] = csv_file.name
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
