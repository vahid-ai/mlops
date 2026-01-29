
If I’m using LakeFS with minio to store data as iceberg tables with Spark, feast and mlflow, what is the best and most efficient way to create, version and track training, validation and testing datasets. Ideally I want to be able to trace back from a model in the mlflow model registry, to the exact datasets and feature transformed datasets that was used to create the model. Since, the technologies used here can overlap in data versioning and quality functionality, what would be the best method? For example, do I create the training, Val and test sets as separate iceberg tables and track with LakeFS and create feast feature views for them? Or do I create the datasets from the original iceberg tables by defining feast views over them? If I want to keep the same dataset splits, but change the feature engineering method, so I make another LakeFS branch to do it? I also want to minimize space/storage

Jan 14

Architected layered versioning strategy across overlapping ML stack technologies.
16s






This is a great architectural question, and you're right that the overlapping capabilities create confusion. Let me break down a recommended approach that leverages each tool's strengths while avoiding redundant versioning.

Core Principle: Single Source of Truth for Each Concern
Concern	Primary Tool	Why
Data versioning	LakeFS	Atomic commits across multiple tables, branches for experimentation
Table time-travel	Iceberg	Convenience queries, but treat as secondary to LakeFS commits
Feature definitions	Feast	Point-in-time joins, online/offline consistency
Experiment lineage	MLflow	Links everything together via logged parameters
Recommended Architecture
1. Dataset Splits: Don't Duplicate Data
Instead of creating separate train/val/test Iceberg tables, store a split assignment table or column:

python
# Split assignment table (tiny - just keys + assignment)
# └── ml_datasets.splits
#     ├── entity_key: string
#     ├── event_timestamp: timestamp
#     ├── split: string  ("train" | "val" | "test")
#     ├── split_version: string  (e.g., "v1_stratified_80_10_10")
#     └── split_seed: int

# Your source data stays in canonical tables
# └── features.user_transactions
# └── features.user_profiles
```

This approach means changing splits just updates a small index table, not copying TBs of feature data.

### 2. LakeFS Branching Strategy
```
main (production)
├── feature-eng/v1  (tag after merge)
├── feature-eng/v2  (tag after merge)
│
└── experiments/
    ├── exp-123-new-embeddings  (branch, delete after experiment)
    └── exp-124-different-splits (branch, delete after experiment)
Key insight: Use LakeFS tags (immutable) for anything you need to trace back to, and branches for experimentation. When an experiment succeeds, merge to main and tag.

python
# After successful experiment
lakefs.tag_create(
    repository="ml-data",
    tag="model-v2.3.0-data",  # matches your model version
    ref="main"
)
3. Feast's Role: Feature Definitions, Not Storage
Feast should define how to retrieve features, not be the primary versioning mechanism:

python
# feast_repo/feature_views.py (version this in Git, not LakeFS)

user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="txn_count_7d", dtype=Int64),
        Field(name="avg_purchase_30d", dtype=Float64),
    ],
    source=IcebergSource(
        table="lakefs://ml-data/main/features.user_features",
        timestamp_field="event_timestamp",
    ),
    # Feast handles point-in-time correctness
)

# Feature service bundles features for a model
model_v2_features = FeatureService(
    name="model_v2_features",
    features=[user_features, item_features],
)
Don't create separate Feast feature views for train/val/test. Instead, use the same feature view and filter at retrieval time:

python
# Training data retrieval
training_df = store.get_historical_features(
    entity_df=splits_df[splits_df.split == "train"],  # filter here
    features=["model_v2_features:txn_count_7d", ...],
).to_df()
4. Complete Lineage via MLflow
Log everything needed to reconstruct the exact training state:

python
with mlflow.start_run():
    # Data lineage
    mlflow.log_param("lakefs_commit", lakefs_client.refs.get("main").commit_id)
    mlflow.log_param("lakefs_tag", "model-v2.3.0-data")  # if tagged
    
    # Split definition
    mlflow.log_param("split_version", "v1_stratified_80_10_10")
    mlflow.log_param("split_seed", 42)
    
    # Feature engineering version (Git commit of feast repo)
    mlflow.log_param("feast_repo_commit", get_git_commit("feast_repo/"))
    mlflow.log_param("feast_feature_service", "model_v2_features")
    
    # Optional: log split statistics for quick reference
    mlflow.log_metrics({
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    })
    
    # Train and log model
    model = train(...)
    mlflow.sklearn.log_model(model, "model")
5. Handling Feature Engineering Changes
Scenario: Same splits, different feature transformations.

python
# Option A: LakeFS branch (recommended for experimentation)
lakefs.branch_create(
    repository="ml-data",
    branch="exp-new-normalization",
    source="main"
)

# Run new feature pipeline writing to this branch
spark.conf.set("spark.sql.catalog.lakefs.ref", "exp-new-normalization")
run_feature_pipeline()  # writes to features.* tables

# If successful, merge and tag
lakefs.merge("exp-new-normalization", "main")
lakefs.tag_create("feature-eng-v3")

# Option B: New table versions (for permanent parallel versions)
# Only if you need both versions in production simultaneously
# └── features.user_features_v2  (new normalization)
# └── features.user_features     (original)
```

### Storage Optimization Summary

| Strategy | Storage Impact |
|----------|----------------|
| Split index table vs. duplicated tables | **~95% reduction** for splits |
| LakeFS branches (copy-on-write) | Only changed files duplicated |
| Iceberg snapshots | Metadata only, shared data files |
| Feature views over source tables | Zero duplication |

### Putting It All Together: Reproducibility Flow
```
MLflow Model Registry
    └── model: fraud-detector-v2.3
        └── params:
            ├── lakefs_tag: "model-v2.3.0-data"
            ├── split_version: "v1_stratified"
            ├── feast_feature_service: "fraud_model_v2"
            └── feast_repo_commit: "abc123"

↓ To reproduce training data exactly:

1. lakefs checkout ml-data --ref model-v2.3.0-data
2. feast apply (at commit abc123)
3. store.get_historical_features(
       entity_df=splits.filter(split_version="v1_stratified", split="train"),
       features=["fraud_model_v2:*"]
   )
What NOT To Do
Don't create train/val/test as separate Iceberg tables duplicating all features
Don't use LakeFS branches as permanent version markers (use tags)
Don't version feature transformation logic in LakeFS (keep it in Git with Feast definitions)
Don't rely on Iceberg snapshots as your primary lineage mechanism (they can be expired for compaction)
Would you like me to dig deeper into any specific part of this—perhaps the Feast + Iceberg integration details, or the exact MLflow artifact logging pattern?

Yes

Jan 14

Orchestrated comprehensive deep-dives into integration and logging patterns.




Let me dive deep into both areas.

Part 1: Feast + Iceberg + LakeFS Integration
The Challenge
Feast doesn't have a native Iceberg source out of the box, and adding LakeFS complicates the URI resolution. Here's how to build a production-grade integration:

Custom Iceberg Data Source for Feast
python
# feast_plugins/iceberg_source.py

from feast import FileSource
from feast.data_source import DataSource
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.repo_config import RepoConfig
from feast.infra.offline_stores.offline_store import OfflineStore
from pyspark.sql import SparkSession
from typing import Callable, Dict, List, Optional, Union
from datetime import datetime
import pyarrow as pa

class IcebergSource(DataSource):
    """
    Feast data source backed by Iceberg tables, with LakeFS support.
    """
    
    def __init__(
        self,
        *,
        table: str,  # e.g., "lakefs://repo/branch/catalog.schema.table"
        timestamp_field: str,
        created_timestamp_column: Optional[str] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        owner: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._table = table
        self._timestamp_field = timestamp_field
        self._created_timestamp_column = created_timestamp_column
        
        super().__init__(
            name=name or table,
            timestamp_field=timestamp_field,
            created_timestamp_column=created_timestamp_column,
            field_mapping=field_mapping,
            description=description,
            tags=tags,
            owner=owner,
        )
    
    @property
    def table(self) -> str:
        return self._table
    
    @staticmethod
    def from_proto(data_source: DataSourceProto) -> "IcebergSource":
        custom_options = data_source.custom_options
        return IcebergSource(
            table=custom_options.configuration["table"],
            timestamp_field=data_source.timestamp_field,
            created_timestamp_column=data_source.created_timestamp_column,
            field_mapping=dict(data_source.field_mapping),
            description=data_source.description,
            tags=dict(data_source.tags),
            owner=data_source.owner,
            name=data_source.name,
        )
    
    def to_proto(self) -> DataSourceProto:
        data_source_proto = DataSourceProto(
            name=self.name,
            type=DataSourceProto.CUSTOM_SOURCE,
            timestamp_field=self.timestamp_field,
            created_timestamp_column=self.created_timestamp_column or "",
            description=self.description or "",
            owner=self.owner or "",
        )
        data_source_proto.custom_options.configuration["table"] = self._table
        return data_source_proto
    
    def get_table_query_string(self, ref_override: Optional[str] = None) -> str:
        """
        Parse LakeFS URI and return Spark-compatible table reference.
        Allows runtime ref override for reproducibility.
        """
        if self._table.startswith("lakefs://"):
            # Parse: lakefs://repo/ref/catalog.schema.table
            parts = self._table.replace("lakefs://", "").split("/", 2)
            repo, ref, table_path = parts[0], parts[1], parts[2]
            
            if ref_override:
                ref = ref_override
            
            # Return format depends on your Spark catalog configuration
            return f"lakefs_{repo}_{ref}.{table_path}"
        return self._table
Custom Offline Store Implementation
python
# feast_plugins/iceberg_offline_store.py

from feast.infra.offline_stores.offline_store import (
    OfflineStore,
    RetrievalJob,
    RetrievalMetadata,
)
from feast.feature_view import FeatureView
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.repo_config import RepoConfig, FeastConfigBaseModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Optional, Union
from datetime import datetime
import pyarrow as pa
from dataclasses import dataclass

class IcebergOfflineStoreConfig(FeastConfigBaseModel):
    type: str = "feast_plugins.iceberg_offline_store.IcebergOfflineStore"
    spark_master: str = "local[*]"
    lakefs_endpoint: str = "http://localhost:8000"
    lakefs_access_key: str = ""
    lakefs_secret_key: str = ""
    warehouse_path: str = "s3a://warehouse/"
    
    # For reproducibility
    default_lakefs_ref: Optional[str] = None


@dataclass
class IcebergRetrievalJob(RetrievalJob):
    """Wraps a Spark DataFrame for lazy evaluation."""
    
    _spark_df: DataFrame
    _metadata: RetrievalMetadata
    _lakefs_commit: str
    
    def to_df(self) -> "pd.DataFrame":
        return self._spark_df.toPandas()
    
    def to_arrow(self) -> pa.Table:
        return pa.Table.from_pandas(self.to_df())
    
    @property
    def metadata(self) -> RetrievalMetadata:
        return self._metadata
    
    @property
    def lakefs_commit(self) -> str:
        """Expose the exact commit used for MLflow logging."""
        return self._lakefs_commit


class IcebergOfflineStore(OfflineStore):
    
    @staticmethod
    def _get_spark_session(config: RepoConfig) -> SparkSession:
        offline_config: IcebergOfflineStoreConfig = config.offline_store
        
        spark = (
            SparkSession.builder
            .master(offline_config.spark_master)
            .config("spark.jars.packages", ",".join([
                "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0",
                "io.lakefs:hadoop-lakefs-assembly:0.2.4",
            ]))
            # Iceberg catalog configuration
            .config("spark.sql.catalog.lakefs", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.lakefs.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
            .config("spark.sql.catalog.lakefs.uri", f"{offline_config.lakefs_endpoint}/api/v1")
            .config("spark.sql.catalog.lakefs.warehouse", offline_config.warehouse_path)
            # LakeFS S3 gateway configuration
            .config("spark.hadoop.fs.s3a.endpoint", offline_config.lakefs_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", offline_config.lakefs_access_key)
            .config("spark.hadoop.fs.s3a.secret.key", offline_config.lakefs_secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .getOrCreate()
        )
        return spark
    
    @staticmethod
    def get_historical_features(
        config: RepoConfig,
        feature_views: List[FeatureView],
        feature_refs: List[str],
        entity_df: Union["pd.DataFrame", str],
        registry: "Registry",
        project: str,
        full_feature_names: bool = False,
        lakefs_ref: Optional[str] = None,  # Override for reproducibility
    ) -> IcebergRetrievalJob:
        """
        Point-in-time correct feature retrieval from Iceberg tables.
        
        The lakefs_ref parameter is critical for reproducibility - it pins
        the exact data version used.
        """
        spark = IcebergOfflineStore._get_spark_session(config)
        offline_config: IcebergOfflineStoreConfig = config.offline_store
        
        # Resolve LakeFS reference
        effective_ref = lakefs_ref or offline_config.default_lakefs_ref or "main"
        
        # Get the exact commit SHA for this ref (for MLflow logging)
        lakefs_commit = IcebergOfflineStore._resolve_lakefs_commit(
            offline_config, effective_ref
        )
        
        # Convert entity_df to Spark
        if isinstance(entity_df, str):
            entity_spark_df = spark.sql(entity_df)
        else:
            entity_spark_df = spark.createDataFrame(entity_df)
        
        # Ensure entity_df has event_timestamp
        if "event_timestamp" not in entity_spark_df.columns:
            raise ValueError("entity_df must contain 'event_timestamp' column")
        
        result_df = entity_spark_df
        
        for feature_view in feature_views:
            source: IcebergSource = feature_view.batch_source
            
            # Get table with specific LakeFS ref
            table_ref = source.get_table_query_string(ref_override=effective_ref)
            feature_df = spark.table(table_ref)
            
            # Point-in-time join
            result_df = IcebergOfflineStore._point_in_time_join(
                entity_df=result_df,
                feature_df=feature_df,
                feature_view=feature_view,
                feature_refs=[f for f in feature_refs if f.startswith(f"{feature_view.name}:")],
                full_feature_names=full_feature_names,
            )
        
        metadata = RetrievalMetadata(
            features=[],
            keys=[],
            min_event_timestamp=None,
            max_event_timestamp=None,
        )
        
        return IcebergRetrievalJob(
            _spark_df=result_df,
            _metadata=metadata,
            _lakefs_commit=lakefs_commit,
        )
    
    @staticmethod
    def _point_in_time_join(
        entity_df: DataFrame,
        feature_df: DataFrame,
        feature_view: FeatureView,
        feature_refs: List[str],
        full_feature_names: bool,
    ) -> DataFrame:
        """
        Implements point-in-time correct join with TTL support.
        
        This ensures no future data leaks into training features.
        """
        entity_keys = [e.name for e in feature_view.entities]
        timestamp_field = feature_view.batch_source.timestamp_field
        ttl_seconds = feature_view.ttl.total_seconds() if feature_view.ttl else None
        
        # Select only needed feature columns
        feature_names = [ref.split(":")[1] for ref in feature_refs]
        select_cols = entity_keys + [timestamp_field] + feature_names
        feature_df = feature_df.select(*select_cols)
        
        # Alias to avoid column name conflicts
        feature_df = feature_df.alias("features")
        entity_df = entity_df.alias("entities")
        
        # Join condition: entity keys match AND feature timestamp <= entity timestamp
        join_conditions = [
            F.col(f"entities.{key}") == F.col(f"features.{key}")
            for key in entity_keys
        ]
        join_conditions.append(
            F.col(f"features.{timestamp_field}") <= F.col("entities.event_timestamp")
        )
        
        # Optional TTL: feature must be within TTL window
        if ttl_seconds:
            join_conditions.append(
                F.col(f"features.{timestamp_field}") >= 
                F.col("entities.event_timestamp") - F.expr(f"INTERVAL {int(ttl_seconds)} SECONDS")
            )
        
        # Join and get latest feature value per entity
        joined = entity_df.join(feature_df, join_conditions, "left")
        
        # Window to get most recent feature row
        window = Window.partitionBy(
            *[F.col(f"entities.{k}") for k in entity_keys],
            F.col("entities.event_timestamp")
        ).orderBy(F.col(f"features.{timestamp_field}").desc())
        
        joined = joined.withColumn("_pit_rank", F.row_number().over(window))
        joined = joined.filter(F.col("_pit_rank") == 1).drop("_pit_rank")
        
        # Rename feature columns
        for feat in feature_names:
            new_name = f"{feature_view.name}__{feat}" if full_feature_names else feat
            joined = joined.withColumn(new_name, F.col(f"features.{feat}"))
        
        # Drop redundant columns
        drop_cols = [f"features.{c}" for c in select_cols]
        joined = joined.drop(*drop_cols)
        
        return joined
    
    @staticmethod
    def _resolve_lakefs_commit(config: IcebergOfflineStoreConfig, ref: str) -> str:
        """Resolve a LakeFS ref (branch/tag) to its commit SHA."""
        import requests
        
        response = requests.get(
            f"{config.lakefs_endpoint}/api/v1/repositories/ml-data/refs/{ref}",
            auth=(config.lakefs_access_key, config.lakefs_secret_key),
        )
        response.raise_for_status()
        return response.json()["commit_id"]
Feature Repository Configuration
yaml
# feast_repo/feature_store.yaml

project: ml_features
provider: local
registry: 
  registry_type: sql
  path: postgresql://feast:feast@localhost:5432/feast_registry

offline_store:
  type: feast_plugins.iceberg_offline_store.IcebergOfflineStore
  spark_master: spark://spark-master:7077
  lakefs_endpoint: http://lakefs:8000
  lakefs_access_key: ${LAKEFS_ACCESS_KEY}
  lakefs_secret_key: ${LAKEFS_SECRET_KEY}
  warehouse_path: s3a://ml-data/warehouse/
  default_lakefs_ref: main

online_store:
  type: redis
  connection_string: redis://localhost:6379

entity_key_serialization_version: 2
Feature View Definitions
python
# feast_repo/features/user_features.py

from datetime import timedelta
from feast import Entity, FeatureView, Field, FeatureService
from feast.types import Float64, Int64, String
from feast_plugins.iceberg_source import IcebergSource

# Entities
user = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Unique user identifier",
)

item = Entity(
    name="item_id", 
    join_keys=["item_id"],
)

# Feature Views
user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    ttl=timedelta(days=90),  # Features older than 90 days are stale
    schema=[
        Field(name="txn_count_7d", dtype=Int64),
        Field(name="txn_count_30d", dtype=Int64),
        Field(name="avg_txn_amount_7d", dtype=Float64),
        Field(name="avg_txn_amount_30d", dtype=Float64),
        Field(name="max_txn_amount_30d", dtype=Float64),
        Field(name="txn_amount_std_30d", dtype=Float64),
        Field(name="unique_merchants_30d", dtype=Int64),
    ],
    source=IcebergSource(
        table="lakefs://ml-data/main/features.user_transaction_features",
        timestamp_field="feature_timestamp",
    ),
    online=True,
    tags={"team": "fraud", "tier": "critical"},
)

user_profile_features = FeatureView(
    name="user_profile_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="account_age_days", dtype=Int64),
        Field(name="is_verified", dtype=Int64),
        Field(name="credit_score_bucket", dtype=String),
        Field(name="lifetime_value_bucket", dtype=String),
    ],
    source=IcebergSource(
        table="lakefs://ml-data/main/features.user_profiles",
        timestamp_field="updated_at",
    ),
    online=True,
)

# Feature Services bundle features for specific models
fraud_model_v1 = FeatureService(
    name="fraud_model_v1",
    features=[
        user_transaction_features,
        user_profile_features,
    ],
    description="Features for fraud detection model v1",
    tags={"model": "fraud_detector", "version": "1"},
)

fraud_model_v2 = FeatureService(
    name="fraud_model_v2",
    features=[
        user_transaction_features[[  # Subset of features
            "txn_count_7d",
            "avg_txn_amount_7d", 
            "txn_amount_std_30d",
        ]],
        user_profile_features,
    ],
    description="Features for fraud detection model v2 - reduced feature set",
    tags={"model": "fraud_detector", "version": "2"},
)
Part 2: Complete MLflow Lineage Pattern
The Reproducibility Contract
Every model in your registry should be fully reproducible from logged parameters alone. Here's the comprehensive pattern:

python
# ml_training/lineage.py

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import hashlib
import json
import subprocess
from datetime import datetime

@dataclass
class DataLineage:
    """Complete data lineage information for reproducibility."""
    
    # LakeFS state
    lakefs_repository: str
    lakefs_ref: str  # branch or tag name
    lakefs_commit: str  # resolved SHA
    
    # Split definition
    split_table: str
    split_version: str
    split_seed: int
    
    # Feast state
    feast_repo_path: str
    feast_repo_commit: str
    feast_feature_service: str
    
    # Timestamps
    data_retrieved_at: str
    
    # Dataset statistics
    train_samples: int
    val_samples: int
    test_samples: int
    
    # Feature schema hash (detect schema drift)
    feature_schema_hash: str
    
    def to_mlflow_params(self) -> Dict[str, Any]:
        """Flatten to MLflow-compatible params."""
        return {
            "data/lakefs_repository": self.lakefs_repository,
            "data/lakefs_ref": self.lakefs_ref,
            "data/lakefs_commit": self.lakefs_commit,
            "data/split_table": self.split_table,
            "data/split_version": self.split_version,
            "data/split_seed": self.split_seed,
            "data/feast_repo_path": self.feast_repo_path,
            "data/feast_repo_commit": self.feast_repo_commit,
            "data/feast_feature_service": self.feast_feature_service,
            "data/retrieved_at": self.data_retrieved_at,
            "data/feature_schema_hash": self.feature_schema_hash,
        }
    
    def to_mlflow_metrics(self) -> Dict[str, int]:
        """Dataset sizes as metrics for easy comparison."""
        return {
            "data/train_samples": self.train_samples,
            "data/val_samples": self.val_samples,
            "data/test_samples": self.test_samples,
        }


@dataclass  
class FeatureEngineeringLineage:
    """Tracks feature transformation logic version."""
    
    # Git commit of your feature engineering code
    feature_pipeline_repo: str
    feature_pipeline_commit: str
    
    # Specific transformation versions (if you version transforms separately)
    transform_versions: Dict[str, str]
    
    # Hash of transformation parameters
    transform_params_hash: str
    
    def to_mlflow_params(self) -> Dict[str, Any]:
        return {
            "features/pipeline_repo": self.feature_pipeline_repo,
            "features/pipeline_commit": self.feature_pipeline_commit,
            "features/transform_params_hash": self.transform_params_hash,
            **{f"features/transform/{k}": v for k, v in self.transform_versions.items()},
        }


def get_git_commit(repo_path: str) -> str:
    """Get current git commit SHA."""
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def compute_schema_hash(df) -> str:
    """Compute deterministic hash of DataFrame schema."""
    schema_str = json.dumps(
        [(c, str(df.schema[c].dataType)) for c in sorted(df.columns)],
        sort_keys=True
    )
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
Complete Training Script with Full Lineage
python
# ml_training/train_with_lineage.py

import mlflow
from mlflow.tracking import MlflowClient
from feast import FeatureStore
from pyspark.sql import SparkSession
import pandas as pd
from datetime import datetime
from typing import Tuple
import lakefs_client
from lakefs_client.api import refs_api, tags_api

from lineage import DataLineage, FeatureEngineeringLineage, get_git_commit, compute_schema_hash


class ReproducibleTrainer:
    """
    Training wrapper that ensures complete lineage tracking.
    """
    
    def __init__(
        self,
        feast_repo_path: str,
        lakefs_endpoint: str,
        lakefs_access_key: str,
        lakefs_secret_key: str,
        mlflow_tracking_uri: str,
    ):
        self.feast_store = FeatureStore(repo_path=feast_repo_path)
        self.feast_repo_path = feast_repo_path
        
        # LakeFS client
        configuration = lakefs_client.Configuration(
            host=lakefs_endpoint,
            username=lakefs_access_key,
            password=lakefs_secret_key,
        )
        self.lakefs_client = lakefs_client.ApiClient(configuration)
        self.refs_api = refs_api.RefsApi(self.lakefs_client)
        self.tags_api = tags_api.TagsApi(self.lakefs_client)
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    def get_training_data(
        self,
        feature_service: str,
        lakefs_ref: str,
        split_version: str,
        repository: str = "ml-data",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataLineage]:
        """
        Retrieve training data with complete lineage tracking.
        """
        # Resolve LakeFS ref to commit
        ref_info = self.refs_api.get_branch(repository, lakefs_ref)
        lakefs_commit = ref_info.commit_id
        
        # Get Feast repo commit
        feast_commit = get_git_commit(self.feast_repo_path)
        
        # Load split definitions
        spark = SparkSession.builder.getOrCreate()
        splits_df = spark.table(f"lakefs_{repository}_{lakefs_ref}.ml_datasets.splits")
        splits_df = splits_df.filter(f"split_version = '{split_version}'").toPandas()
        
        split_seed = int(splits_df["split_seed"].iloc[0])
        
        # Retrieve features for each split using Feast
        # Note: We pass lakefs_ref to pin the exact data version
        train_entities = splits_df[splits_df["split"] == "train"][["user_id", "event_timestamp"]]
        val_entities = splits_df[splits_df["split"] == "val"][["user_id", "event_timestamp"]]
        test_entities = splits_df[splits_df["split"] == "test"][["user_id", "event_timestamp"]]
        
        # Get features with point-in-time correctness
        train_job = self.feast_store.get_historical_features(
            entity_df=train_entities,
            features=self.feast_store.get_feature_service(feature_service).features,
            lakefs_ref=lakefs_ref,  # Custom parameter for our IcebergOfflineStore
        )
        
        val_job = self.feast_store.get_historical_features(
            entity_df=val_entities,
            features=self.feast_store.get_feature_service(feature_service).features,
            lakefs_ref=lakefs_ref,
        )
        
        test_job = self.feast_store.get_historical_features(
            entity_df=test_entities,
            features=self.feast_store.get_feature_service(feature_service).features,
            lakefs_ref=lakefs_ref,
        )
        
        train_df = train_job.to_df()
        val_df = val_job.to_df()
        test_df = test_job.to_df()
        
        # Build lineage
        lineage = DataLineage(
            lakefs_repository=repository,
            lakefs_ref=lakefs_ref,
            lakefs_commit=lakefs_commit,
            split_table=f"{repository}.ml_datasets.splits",
            split_version=split_version,
            split_seed=split_seed,
            feast_repo_path=self.feast_repo_path,
            feast_repo_commit=feast_commit,
            feast_feature_service=feature_service,
            data_retrieved_at=datetime.utcnow().isoformat(),
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            feature_schema_hash=compute_schema_hash(train_df),
        )
        
        return train_df, val_df, test_df, lineage
    
    def train_and_log(
        self,
        model_name: str,
        feature_service: str,
        lakefs_ref: str,
        split_version: str,
        model_params: dict,
        train_fn,  # Callable that takes (train_df, val_df, params) -> model
        feature_engineering_lineage: Optional[FeatureEngineeringLineage] = None,
    ):
        """
        Complete training with full lineage logging.
        """
        # Get data with lineage
        train_df, val_df, test_df, data_lineage = self.get_training_data(
            feature_service=feature_service,
            lakefs_ref=lakefs_ref,
            split_version=split_version,
        )
        
        with mlflow.start_run() as run:
            # Log all lineage
            mlflow.log_params(data_lineage.to_mlflow_params())
            mlflow.log_metrics(data_lineage.to_mlflow_metrics())
            
            if feature_engineering_lineage:
                mlflow.log_params(feature_engineering_lineage.to_mlflow_params())
            
            # Log model hyperparameters
            mlflow.log_params({f"model/{k}": v for k, v in model_params.items()})
            
            # Train
            model = train_fn(train_df, val_df, model_params)
            
            # Evaluate
            train_metrics = self._evaluate(model, train_df, prefix="train")
            val_metrics = self._evaluate(model, val_df, prefix="val")
            test_metrics = self._evaluate(model, test_df, prefix="test")
            
            mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})
            
            # Log model with signature
            signature = mlflow.models.infer_signature(
                train_df.drop(columns=["user_id", "event_timestamp", "label"]),
                model.predict(train_df.drop(columns=["user_id", "event_timestamp", "label"]))
            )
            
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                registered_model_name=model_name,
            )
            
            # Log complete lineage as artifact (for complex queries)
            lineage_artifact = {
                "data": asdict(data_lineage),
                "feature_engineering": asdict(feature_engineering_lineage) if feature_engineering_lineage else None,
                "model_params": model_params,
                "run_id": run.info.run_id,
            }
            mlflow.log_dict(lineage_artifact, "lineage/complete_lineage.json")
            
            # Create reproducibility script
            self._log_reproduction_script(data_lineage, feature_engineering_lineage, model_params)
            
            return run.info.run_id
    
    def _log_reproduction_script(
        self,
        data_lineage: DataLineage,
        fe_lineage: Optional[FeatureEngineeringLineage],
        model_params: dict,
    ):
        """Generate a script that reproduces this exact training run."""
        script = f'''#!/bin/bash
# Reproduction script for training run
# Generated: {datetime.utcnow().isoformat()}

# 1. Checkout LakeFS to exact commit
lakefs log lakefs://{data_lineage.lakefs_repository}/{data_lineage.lakefs_commit} -n 1
echo "Ensure your Spark is configured to read from ref: {data_lineage.lakefs_ref}"
echo "Or use commit directly: {data_lineage.lakefs_commit}"

# 2. Checkout Feast repo to exact commit
cd {data_lineage.feast_repo_path}
git checkout {data_lineage.feast_repo_commit}
feast apply

# 3. Checkout feature pipeline to exact commit (if applicable)
{"cd " + fe_lineage.feature_pipeline_repo + " && git checkout " + fe_lineage.feature_pipeline_commit if fe_lineage else "# No separate feature pipeline tracked"}

# 4. Run training with same parameters
python train.py \\
    --feature-service {data_lineage.feast_feature_service} \\
    --lakefs-ref {data_lineage.lakefs_commit} \\
    --split-version {data_lineage.split_version} \\
    {" ".join([f"--{k} {v}" for k, v in model_params.items()])}
'''
        mlflow.log_text(script, "lineage/reproduce.sh")
    
    def _evaluate(self, model, df, prefix: str) -> dict:
        """Evaluate model and return prefixed metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        X = df.drop(columns=["user_id", "event_timestamp", "label"])
        y = df["label"]
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        return {
            f"{prefix}/accuracy": accuracy_score(y, y_pred),
            f"{prefix}/precision": precision_score(y, y_pred),
            f"{prefix}/recall": recall_score(y, y_pred),
            f"{prefix}/f1": f1_score(y, y_pred),
            f"{prefix}/roc_auc": roc_auc_score(y, y_prob),
        }
    
    def tag_successful_run(
        self,
        run_id: str,
        model_version: str,
        repository: str = "ml-data",
    ):
        """
        After a successful experiment, tag LakeFS for permanent reference.
        """
        client = MlflowClient()
        run = client.get_run(run_id)
        
        lakefs_commit = run.data.params["data/lakefs_commit"]
        tag_name = f"model-{model_version}-data"
        
        # Create LakeFS tag
        self.tags_api.create_tag(
            repository=repository,
            tag_creation={
                "id": tag_name,
                "ref": lakefs_commit,
            }
        )
        
        # Update MLflow run with tag reference
        client.log_param(run_id, "data/lakefs_tag", tag_name)
        
        print(f"Created LakeFS tag '{tag_name}' pointing to {lakefs_commit}")
Usage Example
python
# experiments/train_fraud_model.py

from ml_training.train_with_lineage import ReproducibleTrainer
from ml_training.lineage import FeatureEngineeringLineage, get_git_commit
from sklearn.ensemble import GradientBoostingClassifier

trainer = ReproducibleTrainer(
    feast_repo_path="/path/to/feast_repo",
    lakefs_endpoint="http://lakefs:8000",
    lakefs_access_key="access_key",
    lakefs_secret_key="secret_key",
    mlflow_tracking_uri="http://mlflow:5000",
)

# Feature engineering lineage (if you have a separate feature pipeline)
fe_lineage = FeatureEngineeringLineage(
    feature_pipeline_repo="/path/to/feature_pipeline",
    feature_pipeline_commit=get_git_commit("/path/to/feature_pipeline"),
    transform_versions={
        "normalization": "minmax_v2",
        "encoding": "target_encoding_v1",
    },
    transform_params_hash="abc123",  # Hash of transform config
)

def train_model(train_df, val_df, params):
    X_train = train_df.drop(columns=["user_id", "event_timestamp", "label"])
    y_train = train_df["label"]
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model

# Run training with complete lineage
run_id = trainer.train_and_log(
    model_name="fraud-detector",
    feature_service="fraud_model_v2",
    lakefs_ref="main",  # Or specific branch/tag
    split_version="v1_stratified_80_10_10",
    model_params={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
    },
    train_fn=train_model,
    feature_engineering_lineage=fe_lineage,
)

# If this run is good, tag it permanently
trainer.tag_successful_run(run_id, model_version="2.3.0")
Querying Lineage from Model Registry
python
# Given a model version, reconstruct everything

from mlflow.tracking import MlflowClient

client = MlflowClient()

def get_full_lineage(model_name: str, version: str) -> dict:
    """Retrieve complete lineage for a registered model version."""
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    
    params = run.data.params
    
    return {
        "model": {
            "name": model_name,
            "version": version,
            "run_id": model_version.run_id,
        },
        "data": {
            "lakefs_tag": params.get("data/lakefs_tag"),
            "lakefs_commit": params["data/lakefs_commit"],
            "split_version": params["data/split_version"],
            "split_seed": params["data/split_seed"],
        },
        "features": {
            "feast_feature_service": params["data/feast_feature_service"],
            "feast_repo_commit": params["data/feast_repo_commit"],
            "schema_hash": params["data/feature_schema_hash"],
        },
        "reproduction_command": f"""
lakefs checkout lakefs://ml-data/{params.get('data/lakefs_tag', params['data/lakefs_commit'])}
cd feast_repo && git checkout {params['data/feast_repo_commit']} && feast apply
python train.py --feature-service {params['data/feast_feature_service']} ...
        """.strip(),
    }

# Usage
lineage = get_full_lineage("fraud-detector", "23")
print(lineage)
```

---

## Summary: The Complete Flow
```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT TIME                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. LakeFS main branch ──────────────────┐                          │
│         │                                 │                          │
│         ▼                                 ▼                          │
│  ┌─────────────┐                  ┌─────────────────┐               │
│  │ Iceberg     │                  │ Split Index     │               │
│  │ Feature     │                  │ Table           │               │
│  │ Tables      │                  │ (tiny)          │               │
│  └─────────────┘                  └─────────────────┘               │
│         │                                 │                          │
│         └──────────────┬──────────────────┘                          │
│                        ▼                                             │
│              ┌──────────────────┐                                    │
│              │ Feast            │◄─── Feature definitions           │
│              │ Feature Service  │     (Git versioned)               │
│              └──────────────────┘                                    │
│                        │                                             │
│                        ▼ Point-in-time join                         │
│              ┌──────────────────┐                                    │
│              │ Training Data    │                                    │
│              │ (ephemeral)      │                                    │
│              └──────────────────┘                                    │
│                        │                                             │
│                        ▼                                             │
│              ┌──────────────────┐                                    │
│              │ MLflow Run       │                                    │
│              │ ├─ lakefs_commit │                                    │
│              │ ├─ split_version │                                    │
│              │ ├─ feast_commit  │                                    │
│              │ └─ model         │                                    │
│              └──────────────────┘                                    │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                     PROMOTION TO PRODUCTION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  lakefs tag create "model-v2.3.0-data" @ commit                     │
│                        │                                             │
│                        ▼                                             │
│              ┌──────────────────┐                                    │
│              │ MLflow Model     │                                    │
│              │ Registry         │                                    │
│              │ fraud-detector   │──► Production                     │
│              │ version 23       │                                    │
│              └──────────────────┘                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
This gives you complete traceability from any model version back to the exact data, features, and code that created it—while minimizing storage through split indexes and LakeFS copy-on-write semantics.







Claude is AI and can make mistakes. Please double-check responses.
