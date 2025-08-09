# infra/scripts/fetch_manifest.py
import os, sys, json, pathlib, requests
from mlflow.tracking import MlflowClient

def main():
    if len(sys.argv) < 2:
        print("ERROR: run_name arg required", file=sys.stderr); sys.exit(1)
    run_name = sys.argv[1]

    mlflow_uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    exp_name   = os.environ.get("EXPERIMENT_NAME", "complaint-embeddings")

    client = MlflowClient(tracking_uri=mlflow_uri)
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"ERROR: experiment not found: {exp_name}", file=sys.stderr); sys.exit(2)

    runs = client.search_runs(
        [exp.experiment_id],
        f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        print(f"ERROR: run_name not found: {run_name}", file=sys.stderr); sys.exit(3)

    run_id = runs[0].info.run_id

    # Ask MLflow server to give us the artifact (server will generate a presigned URL)
    url = f"{mlflow_uri}/api/2.0/mlflow/artifacts/download"
    params = {"run_id": run_id, "path": "faiss/manifest.json"}
    resp = requests.get(url, params=params, allow_redirects=True, timeout=60)
    if resp.status_code != 200:
        print(f"ERROR: MLflow artifact download failed: HTTP {resp.status_code} - {resp.text[:300]}", file=sys.stderr)
        sys.exit(4)

    out_dir = pathlib.Path("/tmp/faiss_manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.json"
    out_path.write_bytes(resp.content)

    print(json.dumps({"manifest_path": str(out_path), "run_id": run_id}))

if __name__ == "__main__":
    main()
