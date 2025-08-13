# batch_run.py — run ONE config at a time, in parallel batches
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
                    q = (obj.get("query") or "").strip()
                    if q:
                        qs.append(q)
                except Exception:
                    pass
    else:
        qs = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return qs

def run_one(query, version, k, temp, runner_path: Path, env, retries=3, backoff=1.5):
    cmd = [sys.executable, str(runner_path), "--q", query, "--version", version,
           "--k", str(k), "--temp", str(temp), "--json"]
    if drop_top and int(drop_top) > 0:
        cmd += ["--drop-top", str(drop_top)]
    attempt = 0
    while True:
        attempt += 1
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
        if attempt >= retries:
            return json.dumps({"error": True, "version": version, "k": k, "temp": temp,
                               "stderr": (p.stderr or "")[-400:], "query_head": query[:160]})
        time.sleep(backoff ** attempt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Path to dev/val/test .jsonl or .txt")
    ap.add_argument("--version", required=True, help="prompt version name (e.g., summarize_v1_fewshot)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("OPENAI_BATCH", "6")),
                    help="parallel workers; can also set OPENAI_BATCH env")
    ap.add_argument("--out", required=True, help="Output NDJSON path")
    ap.add_argument("--drop-top", type=int, default=0, help="Drop top-N retrieved chunks before prompting")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    runner = root / "runners" / "query_rag.py"

    futs = [ex.submit(run_one, q, args.version, args.k, args.temp, args.drop_top, runner, env) for q in queries]

    # Make package importable (prompt_pipeline lives under modules/complaint_triage/)
    env = os.environ.copy()
    pkg = str((root / ".." / ".." / "complaint_triage").resolve())
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + (os.pathsep if env.get("PYTHONPATH") else "") + pkg)

    queries = load_queries(Path(args.queries))
    print(f"[batch_run] loaded {len(queries)} queries from {args.queries}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("", encoding="utf-8")  # truncate

    with ThreadPoolExecutor(max_workers=args.batch_size) as ex:
        futs = [ex.submit(run_one, q, args.version, args.k, args.temp, runner, env) for q in queries]
        for i, fut in enumerate(as_completed(futs), 1):
            line = fut.result()
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if i % 50 == 0:
                print(f"  wrote {i}/{len(queries)}")

    print(f"[batch_run] done → {args.out}")

if __name__ == "__main__":
    main()
