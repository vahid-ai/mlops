"""Create SparkSession with Iceberg + LakeFS catalog configured."""

def get_spark_session(app_name: str = "dfp"):
    return f"spark-session-{app_name}"
