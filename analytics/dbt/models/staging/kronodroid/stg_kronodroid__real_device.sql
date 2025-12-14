{{
    config(
        materialized='view',
        tags=['staging', 'kronodroid']
    )
}}

{#
    Staging model for Kronodroid real device data.

    The KronoDroid dataset contains 289 dynamic features (system calls) and
    200 static features. This model selects all columns and adds standardized
    metadata columns.
#}

with source as (
    select * from {{ source('kronodroid_raw', 'real_device') }}
),

cleaned as (
    select
        *,
        -- Data source indicator
        'real_device' as data_source,
        -- Standardized timestamp
        current_timestamp as _dbt_loaded_at
    from source
)

select * from cleaned
