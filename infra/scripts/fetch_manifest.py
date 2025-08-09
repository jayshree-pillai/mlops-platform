# infra/scripts/fetch_manifest.py
import os, sys, json, pathlib
from mlflow.tracking import MlflowClient

def main():
    if len(sys.argv) < 2:
        print("ERROR: run_name arg required", file=sys.stderr); sys.exit(1)
    run_name = sys.argv[1]

    uri = os.environ["MLFLOW_TRACKING_URI"]
    exp_name = os.environ.get("EXPERIMENT_NAME", "complaint-embeddings")

    client = MlflowClient(tracking_uri=uri)
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

    run = runs[0]
    out_dir = "/tmp/faiss_manifest"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # The manifest was logged at artifact_path="faiss"
    manifest_path = client.download_artifacts(run.info.run_id, "faiss/manifest.json", out_dir)

    print(json.dumps({"manifest_path": manifest_path, "run_id": run.info.run_id}))

if __name__ == "__main__":
    main()
