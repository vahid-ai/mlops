{{
    config(
        materialized='table',
        tags=['staging', 'kronodroid']
    )
}}

{#
    Combined staging model for Kronodroid data.

    Unions emulator and real device data with a unique sample_id.
    The sample_id is generated from hash + data_source to handle
    cases where the same APK appears in both datasets.
#}

with emulator as (
    select * from {{ ref('stg_kronodroid__emulator') }}
),

real_device as (
    select * from {{ ref('stg_kronodroid__real_device') }}
),

combined as (
    select * from emulator
    union all
    select * from real_device
)

select
    *,
    -- Generate a unique ID combining row number and source
    -- (hash column name varies by dataset version)
    row_number() over (order by data_source, _ingestion_timestamp) as sample_id
from combined
