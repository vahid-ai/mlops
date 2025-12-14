#!/usr/bin/env python3
"""Export dbt mart tables from DuckDB to LakeFS as parquet files.

This script reads the transformed tables from DuckDB (output of dbt build)
and exports them to LakeFS via S3 API for Feast to consume.

Usage:
    python export_dbt_to_lakefs.py --duckdb-path data/dbt_lakefs.duckdb
    python export_dbt_to_lakefs.py --tables fct_training_dataset dim_malware_families
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_s3_client(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
):
    """Create an S3 client for LakeFS."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def export_table_to_lakefs(
    duckdb_path: Path,
    table_name: str,
    schema: str,
    lakefs_repo: str,
    lakefs_branch: str,
    lakefs_endpoint: str,
    lakefs_key: str,
    lakefs_secret: str,
) -> str:
    """Export a single table from DuckDB to LakeFS as parquet.

    Args:
        duckdb_path: Path to DuckDB database file
        table_name: Name of the table to export
        schema: DuckDB schema containing the table
        lakefs_repo: LakeFS repository name
        lakefs_branch: LakeFS branch name
        lakefs_endpoint: LakeFS S3 gateway endpoint
        lakefs_key: LakeFS access key
        lakefs_secret: LakeFS secret key

    Returns:
        S3 path where the parquet file was written
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tempfile

    # Connect to DuckDB and read the table
    conn = duckdb.connect(str(duckdb_path), read_only=True)

    try:
        # Get the table as Arrow table for efficient parquet writing
        query = f'SELECT * FROM "{schema}"."{table_name}"'
        arrow_table = conn.execute(query).fetch_arrow_table()
    finally:
        conn.close()

    # Write to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
        pq.write_table(arrow_table, tmp_path)

    # Upload to LakeFS
    s3_client = get_s3_client(lakefs_endpoint, lakefs_key, lakefs_secret)

    # LakeFS uses repo as bucket, branch/path as key
    s3_key = f"{lakefs_branch}/dbt/{table_name}/data.parquet"

    try:
        s3_client.upload_file(tmp_path, lakefs_repo, s3_key)
    finally:
        os.unlink(tmp_path)

    s3_path = f"s3://{lakefs_repo}/{s3_key}"
    print(f"Exported {schema}.{table_name} ({len(arrow_table)} rows) -> {s3_path}")

    return s3_path


def list_dbt_tables(duckdb_path: Path) -> list[tuple[str, str]]:
    """List all tables in the DuckDB database created by dbt.

    Returns:
        List of (schema, table_name) tuples
    """
    import duckdb

    conn = duckdb.connect(str(duckdb_path), read_only=True)

    try:
        # Get all tables from information_schema
        result = conn.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """).fetchall()
        return result
    finally:
        conn.close()


def export_all_marts(
    duckdb_path: Path,
    tables: list[str] | None,
    lakefs_repo: str,
    lakefs_branch: str,
    lakefs_endpoint: str,
    lakefs_key: str,
    lakefs_secret: str,
) -> dict[str, str]:
    """Export all mart tables to LakeFS.

    Args:
        duckdb_path: Path to DuckDB database
        tables: Specific tables to export (None = export all marts)
        lakefs_repo: LakeFS repository
        lakefs_branch: LakeFS branch
        lakefs_endpoint: LakeFS endpoint
        lakefs_key: LakeFS access key
        lakefs_secret: LakeFS secret

    Returns:
        Dict mapping table names to their S3 paths
    """
    all_tables = list_dbt_tables(duckdb_path)

    if not all_tables:
        print(f"No tables found in {duckdb_path}")
        return {}

    print(f"Found {len(all_tables)} tables in DuckDB:")
    for schema, table in all_tables:
        print(f"  - {schema}.{table}")

    # Filter to mart tables (or specific tables if provided)
    mart_schemas = ["main_kronodroid", "kronodroid"]
    tables_to_export = []

    for schema, table_name in all_tables:
        if tables:
            # Export only specified tables
            if table_name in tables:
                tables_to_export.append((schema, table_name))
        else:
            # Export mart tables
            if schema in mart_schemas or table_name.startswith(("fct_", "dim_")):
                tables_to_export.append((schema, table_name))

    if not tables_to_export:
        print("No mart tables to export")
        return {}

    print(f"\nExporting {len(tables_to_export)} tables to LakeFS:")

    exported = {}
    for schema, table_name in tables_to_export:
        try:
            s3_path = export_table_to_lakefs(
                duckdb_path=duckdb_path,
                table_name=table_name,
                schema=schema,
                lakefs_repo=lakefs_repo,
                lakefs_branch=lakefs_branch,
                lakefs_endpoint=lakefs_endpoint,
                lakefs_key=lakefs_key,
                lakefs_secret=lakefs_secret,
            )
            exported[table_name] = s3_path
        except Exception as e:
            print(f"ERROR exporting {schema}.{table_name}: {e}")

    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Export dbt tables from DuckDB to LakeFS"
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=PROJECT_ROOT / "analytics" / "dbt" / "data" / "dbt_lakefs.duckdb",
        help="Path to DuckDB database file",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help="Specific tables to export (default: all mart tables)",
    )
    parser.add_argument(
        "--lakefs-repo",
        default=os.getenv("LAKEFS_REPOSITORY", "kronodroid"),
        help="LakeFS repository name",
    )
    parser.add_argument(
        "--lakefs-branch",
        default=os.getenv("LAKEFS_BRANCH", "dev"),
        help="LakeFS branch name",
    )

    args = parser.parse_args()

    # Load LakeFS credentials from environment
    lakefs_endpoint = os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
    lakefs_key = os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    lakefs_secret = os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")

    if not lakefs_key or not lakefs_secret:
        print("ERROR: LAKEFS_ACCESS_KEY_ID and LAKEFS_SECRET_ACCESS_KEY must be set")
        sys.exit(1)

    if not args.duckdb_path.exists():
        print(f"ERROR: DuckDB database not found: {args.duckdb_path}")
        sys.exit(1)

    print("=" * 60)
    print("Exporting dbt tables to LakeFS")
    print("=" * 60)
    print(f"  DuckDB: {args.duckdb_path}")
    print(f"  LakeFS: {args.lakefs_repo}/{args.lakefs_branch}")
    print(f"  Endpoint: {lakefs_endpoint}")

    exported = export_all_marts(
        duckdb_path=args.duckdb_path,
        tables=args.tables,
        lakefs_repo=args.lakefs_repo,
        lakefs_branch=args.lakefs_branch,
        lakefs_endpoint=lakefs_endpoint,
        lakefs_key=lakefs_key,
        lakefs_secret=lakefs_secret,
    )

    print("\n" + "=" * 60)
    print(f"Exported {len(exported)} tables to LakeFS")
    print("=" * 60)

    return 0 if exported else 1


if __name__ == "__main__":
    sys.exit(main())
