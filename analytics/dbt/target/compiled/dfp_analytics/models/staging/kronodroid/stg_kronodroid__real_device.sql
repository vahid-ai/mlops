



with source as (
    select * from "dbt_lakefs"."kronodroid_raw"."real_device"
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