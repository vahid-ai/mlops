{{
    config(
        materialized='table',
        tags=['staging', 'kronodroid'],
        file_format='iceberg'
    )
}}

{#
    Staging model for Kronodroid emulator data.

    Reads Avro data from MinIO (ingested by dlt from Kaggle).
    The dataset contains ~300 columns with syscall counts and a malware label.
    
    Note: Uses Spark SQL syntax (no DuckDB-specific features).
#}

with source as (
    select 
        *,
        -- Generate unique row ID
        CAST(uuid() AS STRING) as _row_id
    from {{ read_kronodroid_emulator() }}
),

cleaned as (
    select
        _row_id,
        
        -- App identifier (rename package to app_package)
        package as app_package,

        -- Target label (ensure integer type)
        CAST(malware AS INT) as is_malware,

        -- Metadata
        'emulator' as data_source,
        current_timestamp() as _dbt_loaded_at

    from source
)

select * from cleaned
