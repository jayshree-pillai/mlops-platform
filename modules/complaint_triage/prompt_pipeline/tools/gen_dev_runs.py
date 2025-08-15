#!/usr/bin/env python3
# tools/gen_dev_runs.py — Generate NDJSON runs from a query file using a chosen prompt version.
# Works for DEV, VAL, or TEST; it's just a wrapper around runners/query_rag.py.
# Example:
# python3 tools/gen_dev_runs.py --queries data/dev_s.jsonl --version summarize_v0        --k 3 --temp 0.0 --out runs/dev/dev_s_v0_k3_t00.ndjson
# python3 tools/gen_dev_runs.py --queries data/dev_s.jsonl --version summarize_v1_fewshot --k 3 --temp 0.0 --out runs/dev/dev_s_v1_k3_t00.ndjson
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def load_queries(path: Path):
    qs = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                q = (obj.get("query") or obj.get("text") or "").strip()
                if q:
                    qs.append(q)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    qs.append(q)
    return qs

def run_one(query: str, version: str, k: int, temp: float, drop_top: int, runner: Path, env: dict,
            retries: int = 2, backoff: float = 1.7):
    cmd = ["python3", str(runner), "--q", query, "--version", version, "--k", str(k), "--temp", str(temp), "--json"]
    if drop_top and drop_top > 0:
        cmd.extend(["--drop-top", str(drop_top)])
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
        if attempt >= retries:
            return json.dumps({"error": True, "version": version, "k": k, "temp": temp,
                               "stderr": (p.stderr or "")[-400:], "query_head": query[:160]})
        time.sleep(backoff ** attempt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Path to .jsonl or .txt file with queries")
    ap.add_argument("--version", required=True, help="prompt version (e.g., summarize_v0 or summarize_v1_fewshot)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", required=True, help="Output NDJSON path")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("OPENAI_BATCH", "6")))
    ap.add_argument("--drop-top", type=int, default=0, help="Drop top-N retrieved chunks before prompting")
    ap.add_argument("--runner", default="runners/query_rag.py", help="Path to the single-query runner")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]  # .../prompt_pipeline
    runner = (root / args.runner).resolve()
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    # Make the package importable (prompt_pipeline under modules/complaint_triage)
    env = os.environ.copy()
    pkg = str((root.parent).resolve())  # .../modules/complaint_triage
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + (os.pathsep if env.get("PYTHONPATH") else "") + pkg)

    queries = load_queries(Path(args.queries))
    if not queries:
        print("No queries loaded.", file=sys.stderr)
        sys.exit(2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w", encoding="utf-8").close()

    with ThreadPoolExecutor(max_workers=args.batch_size) as ex:
        futs = [ex.submit(run_one, q, args.version, args.k, args.temp, args.drop_top, runner, env) for q in queries]
        for i, fut in enumerate(as_completed(futs), 1):
            line = fut.result()
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if i % 50 == 0:
                print(f"  wrote {i}/{len(queries)}")

    print(f"[gen_dev_runs] done -> {args.out}")

if __name__ == "__main__":
    main()
