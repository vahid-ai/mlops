{# ========================================================================
   Spark-specific dbt macros
   
   Helper macros for Spark SQL compatibility and Iceberg table operations.
   ======================================================================== #}


{# ------------------------------------------------------------------------
   Column Selection Helpers (replaces DuckDB's * EXCLUDE syntax)
   ------------------------------------------------------------------------ #}

{% macro select_columns_except(relation, exclude_columns) %}
    {# Select all columns except specified ones (Spark doesn't have EXCLUDE) #}
    {% set columns = adapter.get_columns_in_relation(relation) %}
    {% set result_columns = [] %}
    {% for col in columns %}
        {% if col.name not in exclude_columns %}
            {% do result_columns.append(col.name) %}
        {% endif %}
    {% endfor %}
    {{ result_columns | join(', ') }}
{% endmacro %}


{# ------------------------------------------------------------------------
   Type Casting Helpers
   ------------------------------------------------------------------------ #}

{% macro safe_cast_float(column) %}
    {# Safely cast to float with null handling #}
    CAST(COALESCE({{ column }}, 0) AS DOUBLE)
{% endmacro %}


{% macro safe_cast_int(column) %}
    {# Safely cast to integer with null handling #}
    CAST(COALESCE({{ column }}, 0) AS INT)
{% endmacro %}


{# ------------------------------------------------------------------------
   Hash Functions (Spark-compatible)
   ------------------------------------------------------------------------ #}

{% macro deterministic_hash(column) %}
    {# Deterministic hash for partitioning/splitting #}
    {# xxhash64 returns a signed BIGINT in Spark #}
    xxhash64(CAST({{ column }} AS STRING))
{% endmacro %}


{% macro hash_mod_100(column) %}
    {# Hash modulo 100 for train/val/test splits #}
    {# Use ABS to handle negative hash values #}
    ABS(xxhash64(CAST({{ column }} AS STRING))) % 100
{% endmacro %}


{# ------------------------------------------------------------------------
   Iceberg-specific Operations
   ------------------------------------------------------------------------ #}

{% macro create_iceberg_table_statement(table_name, schema, partition_by=none) %}
    {# Generate CREATE TABLE statement for Iceberg #}
    {% set catalog = var('iceberg_catalog', 'lakefs_catalog') %}
    {% set database = var('iceberg_database', 'dfp') %}
    
    CREATE TABLE IF NOT EXISTS {{ catalog }}.{{ database }}.{{ table_name }} (
        {{ schema }}
    )
    USING iceberg
    {% if partition_by %}
    PARTITIONED BY ({{ partition_by }})
    {% endif %}
    TBLPROPERTIES (
        'write.format.default' = 'avro',
        'write.parquet.compression-codec' = 'snappy'
    )
{% endmacro %}


{% macro merge_into_iceberg(target_table, source_query, merge_keys, update_columns) %}
    {# Generate MERGE INTO statement for Iceberg upserts #}
    {% set catalog = var('iceberg_catalog', 'lakefs_catalog') %}
    {% set database = var('iceberg_database', 'dfp') %}
    
    MERGE INTO {{ catalog }}.{{ database }}.{{ target_table }} AS target
    USING ({{ source_query }}) AS source
    ON {% for key in merge_keys %}target.{{ key }} = source.{{ key }}{% if not loop.last %} AND {% endif %}{% endfor %}
    WHEN MATCHED THEN UPDATE SET
        {% for col in update_columns %}{{ col }} = source.{{ col }}{% if not loop.last %}, {% endif %}{% endfor %}
    WHEN NOT MATCHED THEN INSERT *
{% endmacro %}


{# ------------------------------------------------------------------------
   Timestamp Helpers
   ------------------------------------------------------------------------ #}

{% macro add_load_timestamp() %}
    {# Add a load timestamp column #}
    current_timestamp() AS _dbt_loaded_at
{% endmacro %}


{% macro add_event_timestamp() %}
    {# Add event timestamp for Feast compatibility #}
    current_timestamp() AS event_timestamp
{% endmacro %}
