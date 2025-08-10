# tools/prompt_eval.py
import os, json, time, hashlib, yaml, argparse, pandas as pd
from pathlib import Path
from openai import OpenAI
from sklearn.preprocessing import normalize
import numpy as np, faiss, pickle, mlflow

# --- artifacts (same as your build) ---
OUT_DIR = "semantic_index"
INDEX_PATH = f"{OUT_DIR}/index.faiss"
PCA_PATH   = f"{OUT_DIR}/pca.transform"
PICKLE_PATH= f"{OUT_DIR}/index.pkl"

index = faiss.read_index(INDEX_PATH)
pca   = faiss.read_VectorTransform(PCA_PATH)
data  = pickle.load(open(PICKLE_PATH, "rb"))
texts, metas = data["texts"], data["metadata"]

PROMPT_DIR = Path(__file__).resolve().parents[1] / "modules" / "complaint_triage" / "app" / "prompt" / "versions"

def phash(spec: dict) -> str:
    payload = json.dumps(
        {"system": spec.get("system",""), "instruction": spec.get("instruction",""),
         "template": spec.get("template",""), "params": spec.get("params",{})},
        sort_keys=True, separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]

def load_prompt(name: str):
    p = PROMPT_DIR / f"triage_{name}.yml"
    spec = yaml.safe_load(open(p))
    spec["params"] = spec.get("params",{}) or {}
    return spec, p

def embed(client, q: str):
    e = client.embeddings.create(model="text-embedding-3-large", input=[q])
    v = np.array(e.data[0].embedding, dtype=np.float32).reshape(1,-1)
    v = normalize(v, axis=1)
    v = pca.apply_py(v)
    v = normalize(v, axis=1)
    return v

def score_answer(ans: str, kw: list[str]) -> float:
    # dumb, but quick: fraction of expected keywords present
    if not kw: return 0.0
    a = ans.lower()
    hit = sum(1 for k in kw if k.lower() in a)
    return hit / len(kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with columns: query,keywords (semicolon sep)")
    ap.add_argument("--prompt", choices=["v1","v2"], required=True)
    ap.add_argument("--exp", default="prompt-engg")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=0.35)
    args = ap.parse_args()

    spec, spec_path = load_prompt(args.prompt)
    pv = phash(spec)
    chat_model = spec["params"].get("model", "gpt-3.5-turbo")
    temp = float(spec["params"].get("temperature", 0.2))
    client = OpenAI()

    df = pd.read_csv(args.dataset)  # query, keywords
    results = []
    t0 = time.time()

    with mlflow.start_run(run_name=f"{args.prompt}-{pv}") as run:
        mlflow.set_experiment(args.exp)
        mlflow.log_params({
            "prompt_name": args.prompt,
            "prompt_version": pv,
            "chat_model": chat_model,
            "temperature": temp,
            "topk": args.topk,
            "thresh": args.thresh
        })
        mlflow.log_artifact(spec_path, artifact_path="prompt")
        # evaluate
        for _, row in df.iterrows():
            q = str(row["query"]).strip()
            kws = [x.strip() for x in str(row.get("keywords","")).split(";") if x.strip()]

            # retrieve
            v = embed(client, q)
            scores, idx = index.search(v, args.topk)
            hits = [(texts[i], metas[i], float(scores[0][r]))
                    for r,i in enumerate(idx[0]) if scores[0][r] >= args.thresh]
            context = "\n".join([t for t,_,_ in hits])

            # prompt
            sys = (spec.get("system","") + f" [meta prompt={pv}]").strip()
            ins = spec.get("instruction","").strip()
            tmpl= spec.get("template","{query}")
            user = (ins + "\n\n" if ins else "") + tmpl.format(context=context, query=q)

            t1 = time.time()
            resp = client.chat.completions.create(model=chat_model, temperature=temp,
                        messages=[{"role":"system","content":sys},{"role":"user","content":user}])
            t2 = time.time()
            ans = resp.choices[0].message.content.strip()
            s   = score_answer(ans, kws)

            results.append({
                "query": q, "keywords": ";".join(kws), "retrieved": len(hits),
                "max_score": max([h[2] for h in hits], default=0.0),
                "llm_ms": round((t2-t1)*1000,1), "score": s, "answer": ans
            })

        # aggregate
        avg_score = sum(r["score"] for r in results)/max(len(results),1)
        avg_llm   = sum(r["llm_ms"] for r in results)/max(len(results),1)
        avg_ret   = sum(r["retrieved"] for r in results)/max(len(results),1)

        mlflow.log_metrics({
            "avg_score": avg_score,
            "avg_llm_ms": avg_llm,
            "avg_retrieved": avg_ret,
            "n": len(results),
            "wall_ms": round((time.time()-t0)*1000,1)
        })

        out = Path("prompt_eval_reports")/f"{args.prompt}-{pv}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary":{
            "prompt": args.prompt, "prompt_version": pv,
            "avg_score": avg_score, "avg_llm_ms": avg_llm, "avg_retrieved": avg_ret
        }, "rows": results}, indent=2))
        mlflow.log_artifact(str(out))

        print(json.dumps({"prompt": args.prompt, "prompt_version": pv,
                          "avg_score": round(avg_score,3),
                          "avg_llm_ms": round(avg_llm,1),
                          "avg_retrieved": round(avg_ret,2),
                          "run_id": run.info.run_id}, indent=2))

if __name__ == "__main__":
    main()
