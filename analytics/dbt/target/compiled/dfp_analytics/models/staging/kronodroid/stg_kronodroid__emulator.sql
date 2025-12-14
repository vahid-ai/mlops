



with source as (
    select * from "dbt_lakefs"."kronodroid_raw"."emulator"
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