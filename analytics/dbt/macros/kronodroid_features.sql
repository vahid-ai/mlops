{% macro generate_syscall_columns(prefix='syscall', count=20, suffix='_normalized') %}
    {# Generate a list of syscall feature columns #}
    {% for i in range(1, count + 1) %}
        coalesce(cast({{ prefix }}_{{ i }} as float), 0.0) as {{ prefix }}_{{ i }}{{ suffix }}{% if not loop.last %},{% endif %}
    {% endfor %}
{% endmacro %}


{% macro sum_syscall_features(prefix='syscall', count=20, suffix='_normalized') %}
    {# Generate sum of all syscall features #}
    (
        {% for i in range(1, count + 1) %}
            {{ prefix }}_{{ i }}{{ suffix }}{% if not loop.last %} + {% endif %}
        {% endfor %}
    )
{% endmacro %}


{% macro mean_syscall_features(prefix='syscall', count=20, suffix='_normalized') %}
    {# Generate mean of all syscall features #}
    {{ sum_syscall_features(prefix, count, suffix) }} / {{ count }}.0
{% endmacro %}


{% macro select_syscall_columns(prefix='syscall', count=20, suffix='_normalized') %}
    {# Select existing syscall columns #}
    {% for i in range(1, count + 1) %}
        {{ prefix }}_{{ i }}{{ suffix }}{% if not loop.last %},{% endif %}
    {% endfor %}
{% endmacro %}


{% macro coalesce_feature_columns(columns, default_value=0.0) %}
    {# Apply coalesce to a list of columns #}
    {% for col in columns %}
        coalesce(cast({{ col }} as float), {{ default_value }}) as {{ col }}_safe{% if not loop.last %},{% endif %}
    {% endfor %}
{% endmacro %}
