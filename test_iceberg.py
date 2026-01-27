
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder \
    .appName("test-iceberg") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.lakefs", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.lakefs.type", "hadoop") \
    .config("spark.sql.catalog.lakefs.warehouse", "s3a://kronodroid/dev") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.bucket.kronodroid.endpoint", "http://localhost:8000") \
    .config("spark.hadoop.fs.s3a.bucket.kronodroid.access.key", "AKIAJFUXKT62JAV6MTXQ") \
    .config("spark.hadoop.fs.s3a.bucket.kronodroid.secret.key", "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") \
    .config("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1,org.apache.iceberg:iceberg-aws:1.10.1,org.apache.hadoop:hadoop-aws:3.4.1,software.amazon.awssdk:bundle:2.29.51") \
    .getOrCreate()

try:
    spark.sql("SHOW TABLES IN lakefs.kronodroid").show()
    df = spark.sql("SELECT * FROM lakefs.kronodroid.fct_training_dataset LIMIT 5")
    df.show()
    print("TABLE FOUND AND READ SUCCESSFUL")
except Exception as e:
    print(f"FAILED TO READ TABLE: {e}")
finally:
    spark.stop()
