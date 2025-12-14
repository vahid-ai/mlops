"""KronoDroid dataset ingestion pipeline using dlt.

Ingests data from Kaggle and writes Avro format to MinIO/LakeFS
for consumption by Spark and dbt-spark.
"""

import os
from pathlib import Path

import dlt

from .kaggle_source import kronodroid_source
from .minio_destination import (
    LakeFSConfig,
    MinioConfig,
    ensure_lakefs_repository,
    ensure_minio_bucket,
    get_avro_loader_config,
    get_lakefs_destination,
    get_minio_destination,
)


def run_kronodroid_pipeline(
    destination: str = "minio",
    dataset_name: str = "kronodroid_raw",
    lakefs_branch: str | None = None,
    file_format: str = "avro",
) -> dlt.Pipeline:
    """Run the KronoDroid data ingestion pipeline.

    Downloads the KronoDroid-2021 dataset from Kaggle and loads it into
    either MinIO directly or via LakeFS for versioning.

    Data is written in Avro format for optimal Spark consumption.

    Args:
        destination: Either 'minio' or 'lakefs'
        dataset_name: Name for the dataset in the destination
        lakefs_branch: LakeFS branch name (only used if destination='lakefs')
        file_format: Output format ('avro' recommended for Spark)

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
        # Ensure LakeFS repository and branch exist before writing
        ensure_lakefs_repository(config)
        dest = get_lakefs_destination(config, file_format=file_format)
        pipeline_name = f"kronodroid_lakefs_{config.branch}"
    else:
        config = MinioConfig.from_env()
        ensure_minio_bucket(config)
        dest = get_minio_destination(config, file_format=file_format)
        pipeline_name = "kronodroid_minio"

    # Get Avro-specific loader configuration
    loader_config = get_avro_loader_config() if file_format == "avro" else {}

    pipeline = dlt.pipeline(
        pipeline_name=pipeline_name,
        destination=dest,
        dataset_name=dataset_name,
    )

    source = kronodroid_source(kaggle_api_token=kaggle_token)

    # Run with Avro loader format
    info = pipeline.run(
        source,
        loader_file_format=file_format,
    )
    print(f"Pipeline completed: {info}")
    print(f"  - Format: {file_format}")
    print(f"  - Dataset: {dataset_name}")

    return pipeline


def run_kronodroid_to_avro(
    output_dir: str | Path = "data/kronodroid",
    kaggle_token: str | None = None,
) -> Path:
    """Download KronoDroid dataset and save as Avro files locally.

    This is useful for local development and testing before pushing
    to MinIO/LakeFS.

    Args:
        output_dir: Directory to save Avro files
        kaggle_token: Kaggle API token (uses env var if not provided)

    Returns:
        Path to the output directory
    """
    import tempfile

    import pandas as pd

    from .kaggle_compat import patch_kagglesdk_user_agent

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if kaggle_token:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_token

    patch_kagglesdk_user_agent()
    import kaggle  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle.api.dataset_download_files(
            "dhoogla/kronodroid-2021",
            path=tmpdir,
            unzip=True,
        )

        for csv_file in Path(tmpdir).rglob("*.csv"):
            df = pd.read_csv(csv_file)
            table_name = csv_file.stem.lower().replace("-", "_").replace(" ", "_")
            
            # Write as Avro using fastavro
            avro_path = output_path / f"{table_name}.avro"
            _write_df_to_avro(df, avro_path)
            print(f"Saved {len(df)} rows to {avro_path}")

    return output_path


def _write_df_to_avro(df, path: Path) -> None:
    """Write a pandas DataFrame to Avro format.

    Args:
        df: pandas DataFrame
        path: Output path for Avro file
    """
    import fastavro
    from fastavro.schema import parse_schema

    # Convert DataFrame to records
    records = df.to_dict(orient="records")

    # Infer Avro schema from DataFrame dtypes
    fields = []
    for col, dtype in df.dtypes.items():
        avro_type = _pandas_dtype_to_avro(dtype)
        fields.append({"name": str(col), "type": ["null", avro_type]})

    schema = {
        "type": "record",
        "name": "kronodroid",
        "fields": fields,
    }
    parsed_schema = parse_schema(schema)

    with open(path, "wb") as f:
        fastavro.writer(f, parsed_schema, records, codec="snappy")


def _pandas_dtype_to_avro(dtype) -> str:
    """Convert pandas dtype to Avro type.

    Args:
        dtype: pandas dtype

    Returns:
        Avro type string
    """
    dtype_str = str(dtype)
    if "int" in dtype_str:
        return "long"
    elif "float" in dtype_str:
        return "double"
    elif "bool" in dtype_str:
        return "boolean"
    else:
        return "string"


# Keep legacy function for backwards compatibility
def run_kronodroid_to_parquet(
    output_dir: str | Path = "data/kronodroid",
    kaggle_token: str | None = None,
) -> Path:
    """Download KronoDroid dataset and save as Parquet files locally.

    DEPRECATED: Use run_kronodroid_to_avro() for Spark compatibility.

    Args:
        output_dir: Directory to save Parquet files
        kaggle_token: Kaggle API token (uses env var if not provided)

    Returns:
        Path to the output directory
    """
    import tempfile
    import warnings

    import pandas as pd

    from .kaggle_compat import patch_kagglesdk_user_agent

    warnings.warn(
        "run_kronodroid_to_parquet is deprecated, use run_kronodroid_to_avro",
        DeprecationWarning,
        stacklevel=2,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if kaggle_token:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_token

    patch_kagglesdk_user_agent()
    import kaggle  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle.api.dataset_download_files(
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
    parser.add_argument(
        "--format",
        choices=["avro", "parquet", "jsonl"],
        default="avro",
        help="Output file format (default: avro for Spark compatibility)",
    )

    args = parser.parse_args()

    if args.destination == "local":
        if args.format == "avro":
            output = run_kronodroid_to_avro(output_dir=args.output_dir)
        else:
            output = run_kronodroid_to_parquet(output_dir=args.output_dir)
        print(f"Data saved to: {output}")
    else:
        pipeline = run_kronodroid_pipeline(
            destination=args.destination,
            lakefs_branch=args.branch,
            file_format=args.format,
        )
        print(f"Pipeline state: {pipeline.state}")
