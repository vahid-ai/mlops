{{
    config(
        materialized='table',
        tags=['staging', 'kronodroid'],
        file_format='iceberg'
    )
}}

{#
    Combined staging model for Kronodroid data.

    Unions emulator and real device data with a unique sample_id.
    Uses standard UNION ALL (Spark compatible) instead of DuckDB's UNION ALL BY NAME.
    
    The sample_id is generated from _row_id + data_source for uniqueness.
#}

with emulator as (
    select 
        _row_id,
        app_package,
        is_malware,
        data_source,
        _dbt_loaded_at
    from {{ ref('stg_kronodroid__emulator') }}
),

real_device as (
    select 
        _row_id,
        app_package,
        is_malware,
        data_source,
        _dbt_loaded_at
    from {{ ref('stg_kronodroid__real_device') }}
),

combined as (
    -- Standard UNION ALL (columns must match exactly)
    select * from emulator
    union all
    select * from real_device
)

select
    -- Generate unique sample ID by combining row ID and data source
    CONCAT(_row_id, '_', data_source) as sample_id,
    app_package,
    is_malware,
    data_source,
    _dbt_loaded_at
from combined
