{{
    config(
        materialized='table',
        tags=['mart', 'kronodroid', 'ml', 'feast'],
        file_format='iceberg'
    )
}}

{#
    Training dataset with train/val/test splits.

    Uses deterministic splitting for reproducible ML experiments.
    Includes all features from the fact table plus split assignment.
    
    Note: Uses Spark SQL syntax - xxhash64() instead of DuckDB hash().
#}

with samples as (
    select * from {{ ref('fct_malware_samples') }}
),

with_split as (
    select
        *,
        -- Deterministic split based on sample_id hash
        -- ~70% train, ~15% validation, ~15% test
        -- Use xxhash64 (Spark) instead of hash (DuckDB)
        -- ABS handles potential negative hash values
        case
            when ABS(xxhash64(sample_id)) % 100 < 70 then 'train'
            when ABS(xxhash64(sample_id)) % 100 < 85 then 'validation'
            else 'test'
        end as dataset_split

    from samples
)

select * from with_split
