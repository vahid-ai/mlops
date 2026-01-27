#!/usr/bin/env python3
"""Script to clear all Kubeflow Pipelines, Runs, and Experiments."""

import os
import sys

try:
    import kfp
    from kubernetes import client as k8s_client, config
    from kubernetes.client.rest import ApiException
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

def main():
    # Use KFP_HOST from environment or default to localhost:8080 (standard for port-forward)
    kfp_host = os.getenv("KFP_HOST", "http://localhost:8080")
    print(f"Connecting to Kubeflow Pipelines at {kfp_host}...")
    
    try:
        kfp_client = kfp.Client(host=kfp_host)
    except Exception as e:
        print(f"Failed to connect to KFP: {e}")
        sys.exit(1)

    # 1. Delete all runs
    print("Listing all runs...")
    try:
        runs = kfp_client.list_runs(page_size=100)
        if hasattr(runs, 'runs') and runs.runs:
            print(f"Found {len(runs.runs)} runs. Deleting...")
            for run in runs.runs:
                print(f"  - Deleting run: {run.name} ({run.id})")
                kfp_client.delete_run(run.id)
        else:
            print("No runs found.")
    except Exception as e:
        print(f"Error listing/deleting runs: {e}")

    # 2. Delete all pipelines
    print("Listing all pipelines...")
    try:
        pipelines = kfp_client.list_pipelines(page_size=100)
        if hasattr(pipelines, 'pipelines') and pipelines.pipelines:
            print(f"Found {len(pipelines.pipelines)} pipelines. Deleting...")
            for pipeline in pipelines.pipelines:
                print(f"  - Deleting pipeline: {pipeline.name} ({pipeline.id})")
                kfp_client.delete_pipeline(pipeline.id)
        else:
            print("No pipelines found.")
    except Exception as e:
        print(f"Error listing/deleting pipelines: {e}")

    # 3. Delete all experiments (except Default)
    print("Listing all experiments...")
    try:
        experiments = kfp_client.list_experiments(page_size=100)
        if hasattr(experiments, 'experiments') and experiments.experiments:
            print(f"Found {len(experiments.experiments)} experiments. Deleting...")
            for exp in experiments.experiments:
                if exp.name == "Default":
                    print(f"  - Skipping 'Default' experiment ({exp.id})")
                    continue
                print(f"  - Deleting experiment: {exp.name} ({exp.id})")
                try:
                    kfp_client.delete_experiment(exp.id)
                except Exception as e:
                    print(f"    Error deleting experiment {exp.name}: {e}")
        else:
            print("No experiments found.")
    except Exception as e:
        print(f"Error listing/deleting experiments: {e}")

    # 4. Delete Kubernetes services and pods
    print("\nCleaning up Kubernetes resources...")
    namespace = os.getenv("NAMESPACE", "dfp")
    
    try:
        config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception as e:
            print(f"Failed to load K8s config: {e}")
            return

    core_api = k8s_client.CoreV1Api()
    prefix = "kronodroid-iceberg"

    # Delete Services
    print(f"Searching for services with prefix '{prefix}' in namespace '{namespace}'...")
    try:
        svcs = core_api.list_namespaced_service(namespace)
        for svc in svcs.items:
            if svc.metadata.name.startswith(prefix):
                print(f"  - Deleting service: {svc.metadata.name}")
                core_api.delete_namespaced_service(svc.metadata.name, namespace)
    except Exception as e:
        print(f"Error deleting services: {e}")

    # Delete Pods
    print(f"Searching for pods with prefix '{prefix}' in namespace '{namespace}'...")
    try:
        pods = core_api.list_namespaced_pod(namespace)
        for pod in pods.items:
            if pod.metadata.name.startswith(prefix):
                print(f"  - Deleting pod: {pod.metadata.name}")
                core_api.delete_namespaced_pod(pod.metadata.name, namespace)
    except Exception as e:
        print(f"Error deleting pods: {e}")

    print("\nKFP cleanup complete.")

if __name__ == "__main__":
    main()
