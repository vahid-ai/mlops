{# ========================================================================
   dbt Macros for Spark + Iceberg + LakeFS
   
   Data Flow:
   - dlt ingests from Kaggle → writes Avro to MinIO (s3a://dlt-data/...)
   - dbt-spark reads Avro from MinIO → transforms → writes Iceberg to LakeFS
   - Feast reads Iceberg tables from LakeFS
   ======================================================================== #}


{# ------------------------------------------------------------------------
   MinIO Avro Readers (for raw data from dlt)
   dlt writes Avro files to MinIO bucket
   ------------------------------------------------------------------------ #}

{% macro read_avro_from_minio(path) %}
    {# Read Avro files from MinIO using Spark's Avro reader #}
    {# Note: S3A credentials configured in Spark session via profiles.yml #}
    {% set bucket = var('raw_bucket', 'dlt-data') %}
    {% set minio_endpoint = env_var('MINIO_ENDPOINT_URL', 'http://localhost:19000') %}
    
    {# Use spark.read.format('avro') via SQL table function #}
    (
        SELECT * FROM avro.`s3a://{{ bucket }}/{{ path }}`
    )
{% endmacro %}


{% macro read_kronodroid_emulator() %}
    {# Read the Kronodroid emulator dataset from MinIO (Avro format) #}
    {% set dataset = var('raw_dataset', 'kronodroid_raw') %}
    {{ read_avro_from_minio(dataset ~ '/kronodroid_2021_emu_v1') }}
{% endmacro %}


{% macro read_kronodroid_real_device() %}
    {# Read the Kronodroid real device dataset from MinIO (Avro format) #}
    {% set dataset = var('raw_dataset', 'kronodroid_raw') %}
    {{ read_avro_from_minio(dataset ~ '/kronodroid_2021_real_v1') }}
{% endmacro %}


{# ------------------------------------------------------------------------
   Iceberg Table Helpers
   ------------------------------------------------------------------------ #}

{% macro get_iceberg_table_path(table_name) %}
    {# Get fully qualified Iceberg table name #}
    {% set catalog = var('iceberg_catalog', 'lakefs_catalog') %}
    {% set database = var('iceberg_database', 'dfp') %}
    {{ catalog }}.{{ database }}.{{ table_name }}
{% endmacro %}


{% macro get_lakefs_s3a_path(table_name) %}
    {# Get S3A path for an Iceberg table on LakeFS #}
    {% set lakefs_repo = var('lakefs_repo', 'kronodroid') %}
    {% set lakefs_branch = var('lakefs_branch', 'main') %}
    s3a://{{ lakefs_repo }}/{{ lakefs_branch }}/iceberg/dfp/{{ table_name }}
{% endmacro %}


{# ------------------------------------------------------------------------
   Spark SQL Compatibility Helpers
   ------------------------------------------------------------------------ #}

{% macro spark_hash(column) %}
    {# Spark-compatible hash function (replaces DuckDB hash()) #}
    xxhash64({{ column }})
{% endmacro %}


{% macro spark_current_timestamp() %}
    {# Spark current_timestamp() #}
    current_timestamp()
{% endmacro %}


{# ------------------------------------------------------------------------
   Legacy DuckDB Macros (kept for reference, not used with Spark)
   ------------------------------------------------------------------------ #}

{% macro configure_s3_for_minio() %}
    {# DEPRECATED: DuckDB S3 configuration - not needed for Spark #}
    {# Spark S3A config is set in profiles.yml spark_config #}
    -- No-op for Spark: S3A configuration handled by SparkSession
{% endmacro %}


{% macro configure_s3_for_lakefs() %}
    {# DEPRECATED: DuckDB S3 configuration for LakeFS #}
    -- No-op for Spark: S3A configuration handled by SparkSession
{% endmacro %}
