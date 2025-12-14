{{
    config(
        materialized='table',
        tags=['mart', 'kronodroid', 'ml', 'feast']
    )
}}

{#
    Training dataset with train/val/test splits.

    Uses deterministic splitting for reproducible ML experiments.
    Includes all features from the fact table plus split assignment.
#}

with samples as (
    select * from {{ ref('fct_malware_samples') }}
),

with_split as (
    select
        *,
        -- Deterministic split based on sample_id
        -- ~70% train, ~15% validation, ~15% test
        case
            when abs(hash(sample_id)) % 100 < 70 then 'train'
            when abs(hash(sample_id)) % 100 < 85 then 'validation'
            else 'test'
        end as dataset_split

    from samples
)

select
    sample_id,

    -- All original columns except the ones we're explicitly selecting
    * exclude (sample_id),

    -- Split assignment
    dataset_split

from with_split
