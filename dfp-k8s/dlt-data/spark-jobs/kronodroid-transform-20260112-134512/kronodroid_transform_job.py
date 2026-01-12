#!/usr/bin/env python3
"""Spark (PySpark) job: Kronodroid raw → Iceberg marts on LakeFS.

This job is intended to run via the Kubeflow Spark Operator (SparkApplication),
as an alternative to `dbt-spark` for Step #2 in the Kronodroid pipeline.

Inputs (MinIO via S3A):
  - s3a://{RAW_BUCKET}/{RAW_DATASET}/kronodroid_2021_emu_v1/...
  - s3a://{RAW_BUCKET}/{RAW_DATASET}/kronodroid_2021_real_v1/...

Outputs (Iceberg via configured catalog + LakeFS warehouse):
  - lakefs_catalog.dfp.stg_kronodroid__emulator
  - lakefs_catalog.dfp.stg_kronodroid__real_device
  - lakefs_catalog.dfp.stg_kronodroid__combined
  - lakefs_catalog.dfp.fct_malware_samples
  - lakefs_catalog.dfp.dim_malware_families
  - lakefs_catalog.dfp.fct_training_dataset
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession, functions as F


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _s3a_path(bucket: str, *parts: str) -> str:
    clean = "/".join(p.strip("/") for p in parts if p)
    return f"s3a://{bucket}/{clean}" if clean else f"s3a://{bucket}"


def _ensure_iceberg_db(spark: SparkSession, catalog: str, database: str) -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.{database}")


def main() -> None:
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    raw_bucket = _env("RAW_BUCKET", _env("MINIO_BUCKET_NAME", "dlt-data"))
    raw_dataset = _env("RAW_DATASET", "kronodroid_raw")
    raw_format = _env("RAW_FORMAT", "parquet")

    emulator_table = _env("KRONODROID_EMULATOR_TABLE", "kronodroid_2021_emu_v1")
    real_device_table = _env("KRONODROID_REAL_DEVICE_TABLE", "kronodroid_2021_real_v1")

    iceberg_catalog = _env("ICEBERG_CATALOG", "lakefs_catalog")
    iceberg_database = _env("ICEBERG_DATABASE", "dfp")

    _ensure_iceberg_db(spark, iceberg_catalog, iceberg_database)

    def read_raw(table: str):
        return spark.read.format(raw_format).load(
            _s3a_path(raw_bucket, raw_dataset, table)
        )

    emulator_raw = read_raw(emulator_table)
    real_raw = read_raw(real_device_table)

    stg_emulator = emulator_raw.select(
        F.expr("CAST(uuid() AS STRING)").alias("_row_id"),
        F.col("package").alias("app_package"),
        F.col("malware").cast("int").alias("is_malware"),
        F.lit("emulator").alias("data_source"),
        F.current_timestamp().alias("_dbt_loaded_at"),
    )
    stg_real = real_raw.select(
        F.expr("CAST(uuid() AS STRING)").alias("_row_id"),
        F.col("package").alias("app_package"),
        F.col("malware").cast("int").alias("is_malware"),
        F.lit("real_device").alias("data_source"),
        F.current_timestamp().alias("_dbt_loaded_at"),
    )

    stg_combined = stg_emulator.unionByName(stg_real).select(
        F.concat(F.col("_row_id"), F.lit("_"), F.col("data_source")).alias("sample_id"),
        "app_package",
        "is_malware",
        "data_source",
        "_dbt_loaded_at",
    )

    fct_malware_samples = stg_combined.select(
        "sample_id",
        "app_package",
        "is_malware",
        "data_source",
        F.col("_dbt_loaded_at").alias("event_timestamp"),
    )

    dim_malware_families = (
        stg_combined.groupBy("data_source", "is_malware")
        .agg(
            F.count(F.lit(1)).alias("sample_count"),
            F.countDistinct("app_package").alias("unique_apps"),
        )
        .select(
            F.concat(
                F.col("data_source"),
                F.lit("_"),
                F.when(F.col("is_malware") == 1, F.lit("malware")).otherwise(
                    F.lit("benign")
                ),
            ).alias("category_id"),
            "data_source",
            F.when(F.col("is_malware") == 1, F.lit("malware")).otherwise(
                F.lit("benign")
            ).alias("label_name"),
            "is_malware",
            "sample_count",
            "unique_apps",
            F.current_timestamp().alias("_dbt_loaded_at"),
        )
        .orderBy(F.col("data_source"), F.col("is_malware").desc())
    )

    fct_training_dataset = fct_malware_samples.select(
        "*",
        F.when(
            (F.expr("ABS(xxhash64(sample_id)) % 100") < 70), F.lit("train")
        )
        .when((F.expr("ABS(xxhash64(sample_id)) % 100") < 85), F.lit("validation"))
        .otherwise(F.lit("test"))
        .alias("dataset_split"),
    )

    def write_iceberg(df, table: str) -> None:
        full_name = f"{iceberg_catalog}.{iceberg_database}.{table}"
        df.writeTo(full_name).createOrReplace()

    write_iceberg(stg_emulator, "stg_kronodroid__emulator")
    write_iceberg(stg_real, "stg_kronodroid__real_device")
    write_iceberg(stg_combined, "stg_kronodroid__combined")
    write_iceberg(fct_malware_samples, "fct_malware_samples")
    write_iceberg(dim_malware_families, "dim_malware_families")
    write_iceberg(fct_training_dataset, "fct_training_dataset")


if __name__ == "__main__":
    main()

