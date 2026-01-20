## KronoDroid (2021) — Feast feature contract + Spark feature engineering

This doc describes the **KronoDroid** Android malware dataset features that are **used by Feast** (as defined in `feature_stores/feast_store/dfp_feast/kronodroid_features.py`) and, for each feature, what **Spark transformation** is applied in the Iceberg pipeline (or whether it is **passthrough** / **on-demand**).

### Data flow (tables)

- **Raw (dlt, Kaggle)**: `dhoogla/kronodroid-2021` is ingested into two raw tables:
  - `kronodroid_2021_emu_v1` (emulator)
  - `kronodroid_2021_real_v1` (real device)
  - Ingestion adds `_source_file` and `_ingestion_timestamp` (see `engines/dlt_engine/dfp_dlt/kaggle_source.py`).
- **Spark/Iceberg (LakeFS-tracked)**: `engines/spark_engine/dfp_spark/kronodroid_iceberg_job.py` writes:
  - Staging: `stg_kronodroid.stg_kronodroid__{emulator,real_device,combined}`
  - Marts: `kronodroid.fct_malware_samples`, `kronodroid.fct_training_dataset`, `kronodroid.dim_malware_families`
- **Feast** reads:
  - Sample features from `kronodroid.fct_training_dataset` (timestamp field: `event_timestamp`)
  - Family features from `kronodroid.dim_malware_families` (timestamp field: `_dbt_loaded_at`)

### Spark transformations that affect the Feast tables

These are the **only** Spark transformations currently applied by the Iceberg job (everything else is carried through from the raw dataset):

- **`data_source`**: added via `withColumn("data_source", lit("emulator"|"real_device"))`
- **`_dbt_loaded_at`**: added via `withColumn("_dbt_loaded_at", current_timestamp())`
- **`sample_id`**: generated via `row_number().over(Window.orderBy(data_source, _ingestion_timestamp?))`
- **`event_timestamp`**: `coalesce(_ingestion_timestamp, current_timestamp())` (or `current_timestamp()` if missing)
- **`feature_timestamp`**: copied from `_dbt_loaded_at`
- **`dataset_split`**: deterministic split based on `abs(hash(sample_id)) % 100`
- **`dim_malware_families` aggregates**: `groupBy("data_source").agg(count, min/max ingestion)` plus simple derived columns

### Feature set 1: malware sample features (`FeatureView`: `malware_sample_features`)

**Entity key**

- **`sample_id`**
  - **Meaning**: unique identifier for a sample (Feast join key for entity `malware_sample`)
  - **Table**: `kronodroid.fct_training_dataset.sample_id`
  - **Spark transform**: generated in staging combined:
    - `create_stg_combined`: `withColumn("sample_id", row_number().over(Window.orderBy(data_source, _ingestion_timestamp?)))`

**Metadata + training split**

- **`label`** (Feast schema name)
  - **Meaning**: target label, `1=malware`, `0=benign`
  - **Table**: `kronodroid.fct_training_dataset.label`
  - **Spark transform**: **passthrough** (Spark does not currently rename/cast it)
  - **Upstream expectation**: raw tables contain `label` (see `analytics/dbt/models/staging/kronodroid/_kronodroid_sources.yml`)

- **`malware_family`** (Feast schema name)
  - **Meaning**: malware family name (usually null/empty for benign)
  - **Table**: `kronodroid.fct_training_dataset.malware_family`
  - **Spark transform**: **passthrough / not created in Spark today**
  - **Upstream expectation**: raw tables contain `family` (not `malware_family`)
  - **Intended Spark mapping** (if aligning schema):
    - `withColumn("malware_family", col("family"))` or `withColumnRenamed("family", "malware_family")`

- **`first_seen_year`** (Feast schema name)
  - **Meaning**: year sample was first seen
  - **Table**: `kronodroid.fct_training_dataset.first_seen_year`
  - **Spark transform**: **passthrough / not created in Spark today**
  - **Upstream expectation**: raw tables contain `year` (not `first_seen_year`)
  - **Intended Spark mapping** (if aligning schema):
    - `withColumn("first_seen_year", col("year").cast("bigint"))` (or rename then cast)

- **`data_source`**
  - **Meaning**: `emulator` or `real_device`
  - **Table**: `kronodroid.fct_training_dataset.data_source`
  - **Spark transform**:
    - `create_stg_emulator`: `withColumn("data_source", lit("emulator"))`
    - `create_stg_real_device`: `withColumn("data_source", lit("real_device"))`

- **`dataset_split`**
  - **Meaning**: deterministic `train` / `validation` / `test` assignment
  - **Table**: `kronodroid.fct_training_dataset.dataset_split`
  - **Spark transform** (`create_fct_training_dataset`):
    - `hash_value = abs(hash(col("sample_id"))) % 100`
    - `when(hash_value < 70, "train").when(hash_value < 85, "validation").otherwise("test")`
    - then drops `hash_value`

**Dynamic syscall features (top-20)**

Feast expects **20 numeric syscall features** named `syscall_1_normalized` … `syscall_20_normalized`.

- **`syscall_{i}_normalized`** for \(i=1..20\)
  - **Meaning**: normalized syscall feature \(i\) (dynamic behavior)
  - **Table**: `kronodroid.fct_training_dataset.syscall_{i}_normalized`
  - **Spark transform**: **passthrough / not created in Spark today**
  - **Intended Spark mapping** (canonical formula mirrors `analytics/dbt/macros/kronodroid_features.sql`):
    - `withColumn(f"syscall_{i}_normalized", coalesce(col(f"syscall_{i}").cast("double"), lit(0.0)).cast("float"))`

**Aggregated syscall features**

- **`syscall_total`**
  - **Meaning**: sum of the 20 syscall features
  - **Table**: `kronodroid.fct_training_dataset.syscall_total`
  - **Spark transform**: **passthrough / not created in Spark today**
  - **Intended Spark mapping** (sum of normalized cols; mirrors dbt macro `sum_syscall_features`):
    - `withColumn("syscall_total", col("syscall_1_normalized") + ... + col("syscall_20_normalized"))`

- **`syscall_mean`**
  - **Meaning**: mean of the 20 syscall features
  - **Table**: `kronodroid.fct_training_dataset.syscall_mean`
  - **Spark transform**: **passthrough / not created in Spark today**
  - **Intended Spark mapping** (mirrors dbt macro `mean_syscall_features`):
    - `withColumn("syscall_mean", col("syscall_total") / lit(20.0))`

**Feast-required timestamps (not in the FeatureView schema, but required in the table)**

- **`event_timestamp`** (timestamp field for the SparkSource)
  - **Table**: `kronodroid.fct_training_dataset.event_timestamp`
  - **Spark transform** (`create_fct_malware_samples`):
    - if `_ingestion_timestamp` exists: `withColumn("event_timestamp", coalesce(col("_ingestion_timestamp"), current_timestamp()))`
    - else: `withColumn("event_timestamp", current_timestamp())`

- **`feature_timestamp`** (freshness tracking)
  - **Table**: `kronodroid.fct_training_dataset.feature_timestamp`
  - **Spark transform** (`create_fct_malware_samples`): `withColumn("feature_timestamp", col("_dbt_loaded_at"))`

### Feature set 2: malware family features (`FeatureView`: `malware_family_features`)

**Entity key**

- **`family_id`**
  - **Meaning**: unique identifier for a family/statistics row (Feast join key for entity `malware_family`)
  - **Table**: `kronodroid.dim_malware_families.family_id`
  - **Spark transform** (`create_dim_malware_families`): `select(col("data_source").alias("family_id"), ...)`

**Family-statistics fields**

> Note: the current Spark/dbt pipeline builds this “family” dimension **by `data_source`**, not by malware family label.

- **`family_name`**
  - **Meaning**: display name for the family/statistics row
  - **Table**: `kronodroid.dim_malware_families.family_name`
  - **Spark transform**: `select(col("data_source").alias("family_name"))`

- **`is_malware_family`** (Feast schema name)
  - **Meaning**: whether this row represents a malware family (vs benign)
  - **Table**: `kronodroid.dim_malware_families.is_malware_family`
  - **Spark transform**: **not created in Spark today**
  - **Current pipeline output**: `is_data_source` is created instead:
    - `select(lit(True).alias("is_data_source"))`

- **`total_samples`**
  - **Meaning**: total sample count for the group
  - **Table**: `kronodroid.dim_malware_families.total_samples`
  - **Spark transform**: `groupBy("data_source").agg(count("*").alias("total_samples"))`

- **`unique_samples`**
  - **Meaning**: unique sample count for the group
  - **Table**: `kronodroid.dim_malware_families.unique_samples`
  - **Spark transform**: currently `alias("unique_samples")` of `total_samples` (no distinct count)

- **`emulator_count`**
  - **Meaning**: count of emulator samples
  - **Table**: `kronodroid.dim_malware_families.emulator_count`
  - **Spark transform**:
    - `when(col("data_source") == "emulator", col("total_samples")).otherwise(0)`

- **`real_device_count`**
  - **Meaning**: count of real-device samples
  - **Table**: `kronodroid.dim_malware_families.real_device_count`
  - **Spark transform**:
    - `when(col("data_source") == "real_device", col("total_samples")).otherwise(0)`

- **`earliest_year`**
  - **Meaning**: earliest ingestion year for the group
  - **Table**: `kronodroid.dim_malware_families.earliest_year`
  - **Spark transform**:
    - `min(_ingestion_timestamp)` then `year(first_ingestion)`

- **`latest_year`**
  - **Meaning**: latest ingestion year for the group
  - **Table**: `kronodroid.dim_malware_families.latest_year`
  - **Spark transform**:
    - `max(_ingestion_timestamp)` then `year(last_ingestion)`

- **`year_span`**
  - **Meaning**: span of years represented
  - **Table**: `kronodroid.dim_malware_families.year_span`
  - **Spark transform**: currently constant `lit(1)`

- **`_dbt_loaded_at`** (timestamp field for the SparkSource)
  - **Table**: `kronodroid.dim_malware_families._dbt_loaded_at`
  - **Spark transform**: `current_timestamp()`

### Feature set 3: derived (on-demand) features (`OnDemandFeatureView`: `malware_derived_features`)

These are **not Spark transformations**. They are computed by Feast at request time in Python/NumPy (see `feature_stores/feast_store/dfp_feast/kronodroid_features.py`).

- **`syscall_variance`**
  - **Meaning**: variance across the available syscall feature columns (`syscall_1_normalized`..`syscall_20_normalized`)
  - **Computation**: `np.var(syscall_data, axis=1)`

- **`is_high_activity`**
  - **Meaning**: heuristic flag for high syscall activity
  - **Computation**: `int(sum(syscall_i_normalized) > 100)`

### Notes / current schema mismatches (important)

- **Feast expects `malware_family` + `first_seen_year`, but the raw dataset columns are `family` + `year`.**
  - The current Spark job does not rename these columns, so unless the raw dataset already includes `malware_family`/`first_seen_year`, Feast lookups may fail or return nulls.
- **Feast expects `is_malware_family`, but Spark/dbt currently outputs `is_data_source` and groups by `data_source`.**
  - The current `dim_malware_families` is effectively a “data source stats” dimension, not malware-family stats.
- **Feast expects `syscall_*_normalized`, `syscall_total`, `syscall_mean`, but Spark/dbt currently does not compute them explicitly.**
  - The canonical intended formulas exist as dbt macros in `analytics/dbt/macros/kronodroid_features.sql`; the Spark equivalents are listed above.
