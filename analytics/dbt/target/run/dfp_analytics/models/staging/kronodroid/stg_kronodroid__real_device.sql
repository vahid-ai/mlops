
  
  create view "dbt_lakefs"."main_stg_kronodroid"."stg_kronodroid__real_device__dbt_tmp" as (
    



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
  );
