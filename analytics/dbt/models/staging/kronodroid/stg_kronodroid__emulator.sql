{{
    config(
        materialized='view',
        tags=['staging', 'kronodroid']
    )
}}

{#
    Staging model for Kronodroid emulator data.

    The KronoDroid dataset contains 289 dynamic features (system calls) and
    200 static features. This model selects all columns and adds standardized
    metadata columns.

    Since column names may vary between dataset versions, we use SELECT *
    and add our metadata columns.
#}

with source as (
    select * from {{ source('kronodroid_raw', 'emulator') }}
),

cleaned as (
    select
        *,
        -- Data source indicator
        'emulator' as data_source,
        -- Standardized timestamp
        current_timestamp as _dbt_loaded_at
    from source
)

select * from cleaned
