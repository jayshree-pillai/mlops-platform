# tools/prompt_eval.py
import os, json, time, hashlib, yaml, argparse, pandas as pd
from pathlib import Path
from openai import OpenAI
from sklearn.preprocessing import normalize
import numpy as np, faiss, pickle, mlflow
from jinja2 import Environment, BaseLoader
env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

# replace OUT_DIR...PICKLE_PATH block with:
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "semantic_index")
INDEX_PATH   = f"{ARTIFACT_DIR}/index.faiss"
PCA_PATH     = f"{ARTIFACT_DIR}/pca.transform"
PICKLE_PATH  = f"{ARTIFACT_DIR}/index.pkl"

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "rag" / "versions"
PROMPT_DIR = Path(os.environ.get("PROMPT_DIR", str(DEFAULT_PROMPT_DIR)))

index = faiss.read_index(INDEX_PATH)
pca   = faiss.read_VectorTransform(PCA_PATH)
data  = pickle.load(open(PICKLE_PATH, "rb"))
texts = data.get("texts", [])
metas = data.get("metadata", [{} for _ in texts])

assert isinstance(index, faiss.IndexFlatIP), "Index must be FlatIP (cosine on unit vectors)."
assert pca.d_out == index.d, f"PCA out dim {pca.d_out} != index dim {index.d}"


client = OpenAI()
def phash(obj):
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except:
        s = str(obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
def load_prompt(name):
    """
    Minimal tweak: look for {name}.yml, summarize_{name}.yml, triage_{name}.yml in PROMPT_DIR.
    Keep your existing field usage (system/instruction/template). If spec has jinja_template,
    render it via Jinja (if available) instead of .format.
    """
    cands = [f"{name}.yml", f"summarize_{name}.yml", f"triage_{name}.yml"]
    for fn in cands:
        p = PROMPT_DIR / fn
        if p.exists():
            spec = yaml.safe_load(open(p, "r", encoding="utf-8"))
            return spec or {}, p
    raise FileNotFoundError(f"Prompt '{name}' not found in {PROMPT_DIR}")

def embed(client, q: str):
    if not q.strip():
        # faiss PCA exposes d_out; safe zero vec
        return np.zeros((1, pca.d_out), dtype=np.float32)

    e = client.embeddings.create(model="text-embedding-3-large", input=[q])
    v = np.asarray(e.data[0].embedding, dtype=np.float32)[None, :]
    v = normalize(v, axis=1).astype(np.float32)   # L2 before PCA
    v = pca.apply_py(v)                           # same PCA as index build
    v = normalize(v, axis=1).astype(np.float32)   # L2 after PCA (for IP/cosine)
    return v

def read_dataset(path):
    from pathlib import Path
    import json
    p = Path(path)
    rows = []
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                o = json.loads(ln)
                q = (o.get('query') or o.get('text') or o.get('complaint') or '').strip()
                kws = o.get('keywords') or []
                if isinstance(kws, str):  # allow "a;b;c"
                    kws = [x.strip() for x in kws.split(";") if x.strip()]
                if q:
                    rows.append({"query": q, "keywords": ";".join(kws)})
        import pandas as pd
        df = pd.DataFrame(rows)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix in (".tsv", ".tab"):
        df = pd.read_csv(p, sep="\t")
    else:
        raise ValueError(f"Unsupported dataset: {p}")
    if "keywords" not in df.columns:
        df["keywords"] = ""
    return df

def score_emb_cosine(ans: str, ctx: str) -> float:
    if not ans.strip() or not ctx.strip(): return 0.0
    r = client.embeddings.create(model=os.environ.get("EMBED_MODEL","text-embedding-3-large"), input=[ans, ctx])
    v = np.vstack([np.array(d.embedding, dtype=np.float32) for d in r.data])
    v = normalize(v)
    return float(np.dot(v[0], v[1]))

def score_llm_judge(ans: str, ctx: str) -> float:
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    prompt = (
        "You are a strict evaluator.\n"
        "Given CONTEXT and ANSWER, return a number 0.0 to 1.0 for:\n"
        "- Groundedness (no hallucinations)\n"
        "- Relevance to the query implied by the context\n"
        "Return only the number.\n\n"
        f"CONTEXT:\n{ctx}\n\nANSWER:\n{ans}\n"
    )
    r = client.chat.completions.create(
        model=judge_model, temperature=0.0,
        messages=[{"role":"user","content":prompt}]
    )
    try:
        s = float(r.choices[0].message.content.strip())
        return max(0.0, min(1.0, s))
    except:
        return 0.0

def score_keywords(ans: str, kw_str: str) -> float:
    kws = [k.strip().lower() for k in (kw_str or "").split(";") if k.strip()]
    if not kws:
        return 0.0
    a = ans.lower()
    hit = sum(1 for k in kws if k in a)
    return hit / len(kws)
def build_messages(spec, pv, query, contexts):
    # Jinja path if present; else fall back to instruction/template
    system = (spec.get("system") or "").strip()
    user = None

    jpath = spec.get("jinja_template")
    if jpath:
        try:
            jfile = (PROMPT_DIR / jpath) if not os.path.isabs(jpath) else Path(jpath)
            jtxt = jfile.read_text(encoding="utf-8")
            user = env.from_string(jtxt).render(query=query, contexts=contexts)
        except Exception:
            user = None

    if user is None:
        instruction = (spec.get("instruction") or "").strip()
        template    = (spec.get("template") or "{query}").strip()
        user = ((instruction + "\n\n") if instruction else "") + template.format(
            context="\n\n".join(contexts), query=query
        )

    msgs = []
    if system:
        msgs.append({"role": "system", "content": f"{system} [pv={pv}]"})
    msgs.append({"role": "user", "content": user})
    return msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt", required=True, help="e.g. summarize_v0 or v1")
    ap.add_argument("--exp", default=os.environ.get("MLFLOW_EXPERIMENT","rag-prompt-evals"))
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=0.0)
    ap.add_argument("--judge", choices=["keywords","emb_cosine","llm_judge"], default="keywords")
    ap.add_argument("--sample_n", type=int, default=10)
    args = ap.parse_args()

    spec, spec_path = load_prompt(args.prompt)
    pv = phash(spec)
    model = spec.get("model", "gpt-3.5-turbo")
    temp  = float(spec.get("temperature", 0.2))
    topk  = args.topk or int(spec.get("top_k", 4))

    df = read_dataset(args.dataset)

    # CHANGED: ensure experiment set BEFORE starting run
    mlflow.set_experiment(args.exp)
    t0 = time.time()
    results = []
    llm_ms = []
    retrieved = []

    for i, row in df.iterrows():
        q  = (row.get("query") or "").strip()
        kw = (row.get("keywords") or "").strip()

        # retrieval
        qv = embed(client,q)
        D, I = index.search(qv, topk)

        hits = []
        for rnk, (d, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0: continue
            if d >= args.thresh:
                txt = texts[idx] if idx < len(texts) else ""
                meta = metas[idx] if idx < len(metas) else {}
                hits.append({"text": txt, "meta": meta, "score": float(d), "rank": int(rnk)})
        contexts = [h["text"] for h in hits]
        retrieved.append(len(hits))

        # build messages + call LLM
        msgs = build_messages(spec, pv, q, contexts)
        t1 = time.time()
        r = client.chat.completions.create(model=model, temperature=temp, messages=msgs)
        t2 = time.time()
        ans = r.choices[0].message.content.strip()
        llm_ms.append((t2 - t1) * 1000.0)

        # NEW: tokens + light format/refusal checks (for gates)
        usage = getattr(r, "usage", None)
        ptok = getattr(usage, "prompt_tokens", 0) if usage else 0
        ctok = getattr(usage, "completion_tokens", 0) if usage else 0

        fmt_ok = True
        refusal_ok = None
        pj = None
        if ans.startswith("{"):
            try:
                pj = json.loads(ans)
            except Exception:
                fmt_ok = False
        should_refuse = (len(hits) == 0)
        if pj is not None:
            try:
                refusal_ok = (
                    (should_refuse and ((pj.get("bullets")==[]) or (float(pj.get("confidence",1.0))==0.0)))
                    or (not should_refuse and bool(pj.get("bullets")))
                )
            except Exception:
                refusal_ok = None

        # judge
        if args.judge == "keywords":
            s = score_keywords(ans, kw)
        elif args.judge == "emb_cosine":
            s = score_emb_cosine(ans, "\n\n".join(contexts))
        else:
            s = score_llm_judge(ans, "\n\n".join(contexts))

        results.append({
            "i": int(i), "query": q, "keywords": kw, "answer": ans,
            "hits": hits, "llm_ms": llm_ms[-1],
            "prompt_tokens": ptok, "completion_tokens": ctok,
            "fmt_ok": fmt_ok, "refusal_ok": refusal_ok, "score": float(s)
        })

    avg_score = float(np.mean([r["score"] for r in results])) if results else 0.0
    avg_llm   = float(np.mean(llm_ms)) if llm_ms else 0.0
    avg_ret   = float(np.mean(retrieved)) if retrieved else 0.0
    fmt_err   = 1.0 - (sum(1 for r in results if r["fmt_ok"]) / max(1,len(results)))
    bad_ref   = 1.0 - (sum(1 for r in results if (r["refusal_ok"] is True) or (r["refusal_ok"] is None)) / max(1,len(results)))
    avg_ptok  = float(np.mean([r["prompt_tokens"] for r in results])) if results else 0.0
    avg_ctok  = float(np.mean([r["completion_tokens"] for r in results])) if results else 0.0

    out_dir = Path("prompt_eval_reports"); out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.prompt}-{pv}.json"

    with mlflow.start_run(run_name=f"{args.prompt}-{pv}") as run:
        mlflow.log_params({
            "dataset": Path(args.dataset).name, "prompt": args.prompt,
            "prompt_version": pv, "prompt_file": str(PROMPT_DIR),
            "model": model, "temperature": temp, "topk": topk,
            "thresh": args.thresh, "judge": args.judge
        })
        mlflow.log_metrics({
            "avg_score": avg_score, "avg_llm_ms": avg_llm, "avg_retrieved": avg_ret,
            "format_error_rate": fmt_err, "bad_refusal_rate": bad_ref,
            "avg_prompt_tokens": avg_ptok, "avg_completion_tokens": avg_ctok,
            "n": len(results), "wall_ms": round((time.time()-t0)*1000.0,1)
        })

        out.write_text(json.dumps({"summary":{
            "prompt": args.prompt, "prompt_version": pv,
            "avg_score": avg_score, "avg_llm_ms": avg_llm, "avg_retrieved": avg_ret,
            "format_error_rate": fmt_err, "bad_refusal_rate": bad_ref,
            "avg_prompt_tokens": avg_ptok, "avg_completion_tokens": avg_ctok
        }, "rows": results}, indent=2))
        mlflow.log_artifact(str(out))

        print(json.dumps({"prompt": args.prompt, "prompt_version": pv,
                          "avg_score": round(avg_score,3),
                          "avg_llm_ms": round(avg_llm,1),
                          "avg_retrieved": round(avg_ret,2),
                          "format_error_rate": round(fmt_err,4),
                          "bad_refusal_rate": round(bad_ref,4),
                          "run_id": run.info.run_id}, indent=2))

if __name__ == "__main__":
    main()