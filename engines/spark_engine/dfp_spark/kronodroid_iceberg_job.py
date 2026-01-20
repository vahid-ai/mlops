#!/usr/bin/env python3
"""
Kronodroid Iceberg transformation job.

This PySpark job mirrors the dbt kronodroid models and writes Iceberg tables
with Avro as the data file format, tracked via LakeFS Iceberg REST catalog.

Transformations:
- Staging: stg_kronodroid__{emulator,real_device,combined}
- Marts: fct_malware_samples, dim_malware_families, fct_training_dataset

Usage:
    spark-submit kronodroid_iceberg_job.py \
        --minio-bucket dlt-data \
        --lakefs-branch main \
        --catalog-name lakefs
"""

import argparse
import os
import sys
from typing import TYPE_CHECKING, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from functools import reduce

if TYPE_CHECKING:
    pass

SYSCALL_FEATURE_COUNT = 20


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Kronodroid Iceberg transformations"
    )
    parser.add_argument(
        "--minio-bucket",
        default=os.getenv("MINIO_BUCKET_NAME", "dlt-data"),
        help="MinIO bucket containing raw Parquet files",
    )
    parser.add_argument(
        "--minio-prefix",
        default="kronodroid_raw",
        help="Prefix path within the bucket for raw data",
    )
    parser.add_argument(
        "--lakefs-repository",
        default=os.getenv("LAKEFS_REPOSITORY", "kronodroid"),
        help="LakeFS repository name",
    )
    parser.add_argument(
        "--lakefs-branch",
        default=os.getenv("LAKEFS_BRANCH", "main"),
        help="LakeFS branch to write to",
    )
    parser.add_argument(
        "--catalog-name",
        default="lakefs",
        help="Iceberg catalog name configured in Spark",
    )
    parser.add_argument(
        "--staging-database",
        default="stg_kronodroid",
        help="Database name for staging tables",
    )
    parser.add_argument(
        "--marts-database",
        default="kronodroid",
        help="Database name for mart tables",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit rows per source table for testing (0 = no limit)",
    )
    return parser.parse_args()


def read_raw_data(
    spark: SparkSession,
    minio_bucket: str,
    prefix: str,
    table_name: str,
    limit: int = 0,
) -> DataFrame:
    """Read raw data files from storage (supports JSONL and Parquet).

    Args:
        spark: SparkSession instance
        minio_bucket: MinIO bucket name
        prefix: Path prefix (e.g., kronodroid_raw)
        table_name: Table name as written by dlt
        limit: Maximum rows to read (0 = no limit, for testing)

    Returns:
        DataFrame with raw data
    """
    # dlt writes data to bucket/dataset_name/table_name/*
    # Try JSONL first (dlt default), fall back to Parquet
    jsonl_path = f"s3a://{minio_bucket}/{prefix}/{table_name}/*.jsonl.gz"
    parquet_path = f"s3a://{minio_bucket}/{prefix}/{table_name}/*.parquet"

    print(f"Attempting to read from: {jsonl_path}")
    try:
        df = spark.read.json(jsonl_path)
    except Exception as e:
        print(f"JSONL read failed: {e}")
        print(f"Falling back to Parquet: {parquet_path}")
        df = spark.read.parquet(parquet_path)

    if limit > 0:
        print(f"  Limiting to {limit} rows for testing")
        df = df.limit(limit)

    return df


def create_stg_emulator(raw_df: DataFrame) -> DataFrame:
    """Create stg_kronodroid__emulator staging table.

    Mirrors dbt model: adds data_source and _dbt_loaded_at columns.
    """
    return raw_df.withColumn(
        "data_source", F.lit("emulator")
    ).withColumn(
        "_dbt_loaded_at", F.current_timestamp()
    )


def create_stg_real_device(raw_df: DataFrame) -> DataFrame:
    """Create stg_kronodroid__real_device staging table.

    Mirrors dbt model: adds data_source and _dbt_loaded_at columns.
    """
    return raw_df.withColumn(
        "data_source", F.lit("real_device")
    ).withColumn(
        "_dbt_loaded_at", F.current_timestamp()
    )


def create_stg_combined(
    stg_emulator: DataFrame,
    stg_real_device: DataFrame,
) -> DataFrame:
    """Create stg_kronodroid__combined staging table.

    Mirrors dbt model: unions emulator and real_device, adds sample_id.
    """
    combined = stg_emulator.unionByName(stg_real_device, allowMissingColumns=True)

    # Add sample_id using row_number
    # Handle case where _ingestion_timestamp might not exist
    if "_ingestion_timestamp" in combined.columns:
        order_cols = [F.col("data_source"), F.col("_ingestion_timestamp")]
    else:
        order_cols = [F.col("data_source")]

    window = Window.orderBy(*order_cols)
    return combined.withColumn("sample_id", F.row_number().over(window))


def ensure_kronodroid_feature_columns(df: DataFrame) -> DataFrame:
    """Normalize Kronodroid feature column set for downstream (Feast/ML).

    The Kaggle Kronodroid dataset can vary slightly by version/source. To keep
    Feast FeatureViews stable, we ensure key columns exist and are typed.
    """
    # Ensure syscall_1_normalized ... syscall_20_normalized exist and are Float32-ish.
    for i in range(1, SYSCALL_FEATURE_COUNT + 1):
        col_name = f"syscall_{i}_normalized"
        if col_name not in df.columns:
            df = df.withColumn(col_name, F.lit(0.0))
        df = df.withColumn(col_name, F.col(col_name).cast("float"))

    syscall_cols = [F.col(f"syscall_{i}_normalized") for i in range(1, SYSCALL_FEATURE_COUNT + 1)]
    syscall_total = reduce(lambda a, b: a + b, syscall_cols, F.lit(0.0))

    df = df.withColumn("syscall_total", syscall_total.cast("double"))
    df = df.withColumn(
        "syscall_mean",
        (F.col("syscall_total") / F.lit(float(SYSCALL_FEATURE_COUNT))).cast("double"),
    )

    # Align label column name with Feast FeatureView (prefer existing label).
    if "label" not in df.columns and "is_malware" in df.columns:
        df = df.withColumn("label", F.col("is_malware").cast("long"))
    elif "label" in df.columns:
        df = df.withColumn("label", F.col("label").cast("long"))

    # Optional metadata expected by Feast definitions.
    if "malware_family" not in df.columns:
        for candidate in ("family", "family_name", "malware_family_name"):
            if candidate in df.columns:
                df = df.withColumn("malware_family", F.col(candidate).cast("string"))
                break
        else:
            df = df.withColumn("malware_family", F.lit("unknown"))

    if "first_seen_year" not in df.columns and "event_timestamp" in df.columns:
        df = df.withColumn("first_seen_year", F.year(F.col("event_timestamp")).cast("long"))

    return df


def create_fct_malware_samples(stg_combined: DataFrame) -> DataFrame:
    """Create fct_malware_samples fact table.

    Mirrors dbt model: casts sample_id, adds event_timestamp and feature_timestamp.
    """
    df = stg_combined.withColumn(
        "sample_id", F.col("sample_id").cast("string")
    )

    # Add event_timestamp: coalesce(_ingestion_timestamp, current_timestamp)
    if "_ingestion_timestamp" in df.columns:
        df = df.withColumn(
            "event_timestamp",
            F.coalesce(F.col("_ingestion_timestamp"), F.current_timestamp()),
        )
    else:
        df = df.withColumn("event_timestamp", F.current_timestamp())

    # Add feature_timestamp from _dbt_loaded_at
    df = df.withColumn("feature_timestamp", F.col("_dbt_loaded_at"))

    return ensure_kronodroid_feature_columns(df)


def create_dim_malware_families(stg_combined: DataFrame) -> DataFrame:
    """Create dim_malware_families dimension table.

    Mirrors dbt model: aggregates by data_source with statistics.
    """
    # Aggregate by data_source
    if "_ingestion_timestamp" in stg_combined.columns:
        agg_df = stg_combined.groupBy("data_source").agg(
            F.count("*").alias("total_samples"),
            F.min("_ingestion_timestamp").alias("first_ingestion"),
            F.max("_ingestion_timestamp").alias("last_ingestion"),
        )
    else:
        agg_df = stg_combined.groupBy("data_source").agg(
            F.count("*").alias("total_samples"),
            F.current_timestamp().alias("first_ingestion"),
            F.current_timestamp().alias("last_ingestion"),
        )

    # Build final dimension table
    return agg_df.select(
        F.col("data_source").alias("family_id"),
        F.col("data_source").alias("family_name"),
        F.lit(True).alias("is_data_source"),
        F.col("total_samples"),
        F.col("total_samples").alias("unique_samples"),
        F.when(F.col("data_source") == "emulator", F.col("total_samples"))
        .otherwise(0)
        .alias("emulator_count"),
        F.when(F.col("data_source") == "real_device", F.col("total_samples"))
        .otherwise(0)
        .alias("real_device_count"),
        F.year(F.col("first_ingestion")).alias("earliest_year"),
        F.year(F.col("last_ingestion")).alias("latest_year"),
        F.lit(1).alias("year_span"),
        F.current_timestamp().alias("_dbt_loaded_at"),
    ).orderBy(F.desc("total_samples"))


def create_fct_training_dataset(fct_malware_samples: DataFrame) -> DataFrame:
    """Create fct_training_dataset fact table.

    Mirrors dbt model: adds deterministic train/validation/test split.
    """
    # Deterministic split based on hash(sample_id) % 100
    # ~70% train, ~15% validation, ~15% test
    df = fct_malware_samples.withColumn(
        "hash_value", F.abs(F.hash(F.col("sample_id"))) % 100
    )

    df = df.withColumn(
        "dataset_split",
        F.when(F.col("hash_value") < 70, "train")
        .when(F.col("hash_value") < 85, "validation")
        .otherwise("test"),
    )

    # Drop the intermediate hash column
    return df.drop("hash_value")


def write_iceberg_table(
    df: DataFrame,
    catalog: str,
    database: str,
    table: str,
    mode: str = "overwrite",
) -> None:
    """Write DataFrame as an Iceberg table with Avro format.

    Args:
        df: DataFrame to write
        catalog: Iceberg catalog name
        database: Database/namespace name
        table: Table name
        mode: Write mode (overwrite, append)
    """
    full_table_name = f"{catalog}.{database}.{table}"
    print(f"Writing Iceberg table: {full_table_name}")

    # Write with Avro format (matches repo defaults; enables consistent file type across jobs)
    df.writeTo(full_table_name).tableProperty(
        "write.format.default", "avro"
    ).tableProperty(
        "write.avro.compression-codec", "snappy"
    ).using("iceberg").createOrReplace()

    print(f"Successfully wrote {df.count()} rows to {full_table_name}")


def ensure_databases(spark: SparkSession, catalog: str, databases: List[str]) -> None:
    """Ensure Iceberg databases exist.

    Args:
        spark: SparkSession instance
        catalog: Catalog name
        databases: List of database names to create
    """
    for db in databases:
        full_db = f"{catalog}.{db}"
        print(f"Creating database if not exists: {full_db}")
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {full_db}")


def main() -> int:
    """Main entry point for the Kronodroid Iceberg job."""
    args = get_args()

    print("=" * 60)
    print("Kronodroid Iceberg Transformation Job")
    print("=" * 60)
    print(f"  MinIO bucket: {args.minio_bucket}")
    print(f"  MinIO prefix: {args.minio_prefix}")
    print(f"  LakeFS repository: {args.lakefs_repository}")
    print(f"  LakeFS branch: {args.lakefs_branch}")
    print(f"  Catalog: {args.catalog_name}")
    print(f"  Staging database: {args.staging_database}")
    print(f"  Marts database: {args.marts_database}")
    if args.test_limit > 0:
        print(f"  Test limit: {args.test_limit} rows per table")
    print("=" * 60)

    # Get or create SparkSession (assumes it's configured externally or via session.py)
    spark = SparkSession.builder.getOrCreate()

    try:
        # Ensure databases exist
        ensure_databases(
            spark,
            args.catalog_name,
            [args.staging_database, args.marts_database],
        )

        # Step 1: Read raw data from storage
        # dlt uses these table names from the Kronodroid Kaggle dataset
        print("\n[1/7] Reading raw emulator data...")
        raw_emulator = read_raw_data(
            spark, args.minio_bucket, args.minio_prefix, "kronodroid_2021_emu_v1",
            limit=args.test_limit,
        )
        print(f"  Rows: {raw_emulator.count()}")

        print("\n[2/7] Reading raw real_device data...")
        raw_real_device = read_raw_data(
            spark, args.minio_bucket, args.minio_prefix, "kronodroid_2021_real_v1",
            limit=args.test_limit,
        )
        print(f"  Rows: {raw_real_device.count()}")

        # Step 2: Create staging tables
        print("\n[3/7] Creating stg_kronodroid__emulator...")
        stg_emulator = create_stg_emulator(raw_emulator)
        write_iceberg_table(
            stg_emulator,
            args.catalog_name,
            args.staging_database,
            "stg_kronodroid__emulator",
        )

        print("\n[4/7] Creating stg_kronodroid__real_device...")
        stg_real_device = create_stg_real_device(raw_real_device)
        write_iceberg_table(
            stg_real_device,
            args.catalog_name,
            args.staging_database,
            "stg_kronodroid__real_device",
        )

        print("\n[5/7] Creating stg_kronodroid__combined...")
        stg_combined = create_stg_combined(stg_emulator, stg_real_device)
        write_iceberg_table(
            stg_combined,
            args.catalog_name,
            args.staging_database,
            "stg_kronodroid__combined",
        )

        # Step 3: Create mart tables
        print("\n[6/7] Creating fct_malware_samples...")
        fct_samples = create_fct_malware_samples(stg_combined)
        write_iceberg_table(
            fct_samples,
            args.catalog_name,
            args.marts_database,
            "fct_malware_samples",
        )

        print("\n[7/7] Creating dim_malware_families...")
        dim_families = create_dim_malware_families(stg_combined)
        write_iceberg_table(
            dim_families,
            args.catalog_name,
            args.marts_database,
            "dim_malware_families",
        )

        print("\n[8/7] Creating fct_training_dataset...")
        fct_training = create_fct_training_dataset(fct_samples)
        write_iceberg_table(
            fct_training,
            args.catalog_name,
            args.marts_database,
            "fct_training_dataset",
        )

        print("\n" + "=" * 60)
        print("Kronodroid Iceberg transformations completed successfully!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nERROR: Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
