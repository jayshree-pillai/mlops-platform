# tools/prompt_promote.py
import os, json, argparse, shutil, yaml, hashlib
from pathlib import Path
import mlflow

PROMPT_DIR = Path(__file__).resolve().parents[1] / "modules" / "complaint_triage" / "app" / "prompt" / "versions"
DEPLOY_DIR = Path(__file__).resolve().parents[1] / "modules" / "complaint_triage" / "app" / "prompt"

def phash(spec): 
    import json, hashlib
    payload = json.dumps({"system":spec.get("system",""),
                          "instruction":spec.get("instruction",""),
                          "template":spec.get("template",""),
                          "params":spec.get("params",{})}, sort_keys=True, separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=False, help="(optional) MLflow run to fetch prompt_name from")
    ap.add_argument("--prompt", choices=["v1","v2"], required=False, help="Skip MLflow and promote this prompt name")
    args = ap.parse_args()

    if args.prompt:
        name = args.prompt
    else:
        run = mlflow.get_run(args.run_id)
        name = run.data.params.get("prompt_name")
        assert name, "run has no prompt_name param"

    src = PROMPT_DIR / f"triage_{name}.yml"
    spec = yaml.safe_load(open(src))
    pv = phash(spec)

    # write "current.yml" for deployment bundle
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    dst = DEPLOY_DIR / "current.yml"
    shutil.copyfile(src, dst)

    print("\nPromotion complete ✅")
    print(f"Prompt: {name}")
    print(f"Prompt Version: {pv}")
    print(f"Wrote: {dst}")
    print("\nSet this on the serving box and restart:")
    print(f"  echo 'PROMPT_VERSION={name}@{pv}' | sudo tee -a /etc/rag-api.env")
    print("  sudo systemctl daemon-reload && sudo systemctl restart rag-api\n")

if __name__ == "__main__":
    main()
