import json, argparse, math, mlflow, yaml, os, time
from pathlib import Path
from jinja2 import Template
from openai import OpenAI

def elo_update(ra, rb, outcome, k=16):
    ea = 1/(1+10**((rb-ra)/400)); eb = 1-ea
    return ra + k*(outcome-ea), rb + k*((1-outcome)-eb)

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))

def _load_ndjson(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

def _get_query(rec: dict) -> str:
    # Try common locations; fall back to empty string
    return (
        rec.get("query")
        or (rec.get("lineage") or {}).get("query")
        or (rec.get("input") or {}).get("q")
        or ""
    )

def _get_hits(rec: dict):
    # Support both shapes: top-level "hits" OR nested "retriever.hits"
    return rec.get("hits") or (rec.get("retriever") or {}).get("hits") or []

def _ctx_from_hits(hits, max_chars=6000):
    parts = []
    for h in hits:
        t = h.get("text") if isinstance(h, dict) else str(h)
        if not t: continue
        parts.append(t)
        if sum(len(p) for p in parts) >= max_chars:
            break
    ctx = "\n\n".join(parts)
    return ctx[:max_chars]

def _answer_str(x, max_chars=5000):
    if isinstance(x, str):
        return x[:max_chars]
    try:
        return json.dumps(x, ensure_ascii=False)[:max_chars]
    except Exception:
        return str(x)[:max_chars]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="NDJSON A")
    ap.add_argument("--b", required=True, help="NDJSON B")
    ap.add_argument("--judge", default="config/judge.yml")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--strict-match", action="store_true",
                    help="If set, skip pairs where A.query != B.query")
    ap.add_argument("--ctx-max", type=int, default=6000)
    ap.add_argument("--ans-max", type=int, default=5000)
    args = ap.parse_args()

    A = _load_ndjson(Path(args.a))
    B = _load_ndjson(Path(args.b))
    n = min(len(A), len(B))

    cfg = yaml.safe_load(open(args.judge, "r", encoding="utf-8"))
    judge_model = cfg["model"]
    tpl = Template(cfg["pairwise_prompt"])
    client = OpenAI()

    wins = ties = 0
    ra = rb = 1200.0
    judged = 0

    for i in range(n):
        a, b = A[i], B[i]
        qa, qb = _get_query(a), _get_query(b)
        if args.strict_match and qa != qb:
            continue  # skip mismatched queries if strict mode

        ctx = _ctx_from_hits(_get_hits(a), max_chars=args.ctx_max)
        pa = _answer_str(a.get("answer", ""), max_chars=args.ans_max)
        pb = _answer_str(b.get("answer", ""), max_chars=args.ans_max)

        pr = tpl.render(query=(qa or qb or ""), context=ctx, a=pa, b=pb)
        jr = client.chat.completions.create(
            model=judge_model,
            temperature=0.0,
            messages=[{"role": "user", "content": pr}],
        )
        v = (jr.choices[0].message.content or "").strip().upper()
        judged += 1
        if v.startswith("A"):
            wins += 1; ra, rb = elo_update(ra, rb, 1.0)
        elif v.startswith("B"):
            ra, rb = elo_update(ra, rb, 0.0)
        else:
            ties += 1; ra, rb = elo_update(ra, rb, 0.5)

    wr = wins / max(1, judged)
    lo, hi = wilson_ci(wins, max(1, judged))
    mlflow.set_experiment(args.exp or "rag-prompt-evals")
    with mlflow.start_run(run_name=f"PAIR_{Path(args.a).stem}_vs_{Path(args.b).stem}"):
        mlflow.log_params({
            "A": Path(args.a).name,
            "B": Path(args.b).name,
            "judge_model": judge_model,
            "strict_match": bool(args.strict_match),
            "ctx_max": args.ctx_max,
            "ans_max": args.ans_max,
        })
        mlflow.log_metrics({
            "win_rate_A_over_B": wr,
            "win_rate_ci_low": lo,
            "win_rate_ci_high": hi,
            "elo_A": ra,
            "elo_B": rb,
            "ties": ties,
            "n_pairs_judged": judged,
        })
    print(f"win_rate={wr:.3f} [{lo:.3f},{hi:.3f}] eloA={ra:.1f} eloB={rb:.1f} ties={ties} n={judged}")

if __name__ == "__main__":
    main()
