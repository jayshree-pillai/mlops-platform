# infra/scripts/fetch_manifest.py
import os, sys, json, pathlib, traceback
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
    run_id = run.info.run_id

    # See what's actually there
    root = client.list_artifacts(run_id, path="")
    root_entries = [f"{x.path}{'/' if x.is_dir else ''}" for x in root]

    out_dir = "/tmp/faiss_manifest"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    last_err = None
    for cand in ("faiss/manifest.json", "manifest.json"):
        try:
            p = client.download_artifacts(run_id, cand, out_dir)
            print(json.dumps({"manifest_path": p, "run_id": run_id, "tried": [cand], "root": root_entries}))
            return
        except Exception as e:
            last_err = e

    print(f"ERROR: manifest.json not found at expected paths. Root artifacts: {root_entries}", file=sys.stderr)
    if last_err:
        print(f"Last error: {last_err}", file=sys.stderr)
    sys.exit(4)

if __name__ == "__main__":
    main()
