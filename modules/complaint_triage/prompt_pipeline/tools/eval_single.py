import os, json, argparse, statistics as st, yaml, mlflow, random
from pathlib import Path
from jinja2 import Template

def load_yaml(p): return yaml.safe_load(open(p, "r", encoding="utf-8"))

def p95(xs): 
    xs=sorted(xs); 
    return xs[max(0, int(0.95*(len(xs)-1)))]

def cost(tokens_in, tokens_out, price):
    return (tokens_in/1000.0)*price["input_per_1k"] + (tokens_out/1000.0)*price["output_per_1k"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--exp", default="prompt-evals")
    ap.add_argument("--judge", default="config/judge.yml")
    ap.add_argument("--gates", default="config/gates.yml")
    ap.add_argument("--costs", default="config/costs.yml")
    ap.add_argument("--sample-audit", type=int, default=None)
    args = ap.parse_args()

    judge_cfg = load_yaml(args.judge)
    gates = load_yaml(args.gates)
    costs = load_yaml(args.costs)

    # lazy OpenAI client
    import openai
    client = openai.OpenAI()

    triad_tpl = Template(judge_cfg["triad_prompt"])

    rows=[]; lat=[]; pt=[]; ct=[]; fmt_ok=0; refus_ok=0
    with open(args.ndjson, "r", encoding="utf-8") as f:
        for ln in f:
            o=json.loads(ln)
            ans = o.get("answer") or o.get("content") or o.get("output") or ""
            hits = o.get("hits") or ((o.get("retriever") or {}).get("hits") or [])
            ctx = "\n\n".join([(h.get("text", "") if isinstance(h, dict) else str(h)) for h in hits])
            q   = o.get("query") or o.get("prompt") or ""
            # format validity (JSON outputs)
            ok=True
            try:
                _ = json.loads(ans) if ans.strip().startswith("{") else None
            except: 
                ok=False
            fmt_ok += int(ok)
            # refusal accuracy (if no hits, expect empty bullets or low confidence)
            should_refuse = (len(o.get("hits",[]))==0)
            ro = None
            try:
                jo = json.loads(ans) if ans.strip().startswith("{") else {}
                ro = ( (should_refuse and (jo.get("bullets")==[] or float(jo.get("confidence",1.0))==0.0))
                       or (not should_refuse) )
            except:
                ro = not should_refuse
            refus_ok += int(bool(ro))

            # triad judge
            jprompt = triad_tpl.render(query=q, context=ctx[:6000], ans=ans[:6000])
            jr = client.chat.completions.create(
                model=judge_cfg["model"], temperature=judge_cfg["temperature"],
                messages=[{"role":"user","content":jprompt}]
            )
            try:
                scores = json.loads(jr.choices[0].message.content)
            except:
                scores = {"faithfulness":0.0, "answer_relevance":0.0, "context_relevance":0.0}
            rows.append(scores)

            # timings/tokens if present
            tt = o.get("timing_tokens") or {}
            lat.append(o.get("latency_ms", o.get("latency", tt.get("latency_ms", 0))))
            pt.append(o.get("prompt_tokens", (o.get("timing_tokens") or {}).get("prompt_tokens", 0)))
            ct.append(o.get("completion_tokens", (o.get("timing_tokens") or {}).get("completion_tokens", 0)))

    n=len(rows)
    faith = [r["faithfulness"] for r in rows]
    ansrel= [r["answer_relevance"] for r in rows]
    ctxrel= [r["context_relevance"] for r in rows]
    fmt_err = 1 - (fmt_ok / max(1,n))
    bad_ref = 1 - (refus_ok / max(1,n))
    p95_lat = p95([x for x in lat if isinstance(x,(int,float))]) if lat else 0
    avg_lat = sum(lat)/max(1,len(lat))
    avg_pt  = sum(pt)/max(1,len(pt)); avg_ct=sum(ct)/max(1,len(ct))
    model_name = judge_cfg["model"]
    c = cost(avg_pt, avg_ct, costs.get(model_name, {"input_per_1k":0.0,"output_per_1k":0.0}))

    gen_model = None
    if rows:
        # try to read lineage from any NDJSON line that has it
        try:
            sample = json.loads(open(args.ndjson, "r", encoding="utf-8").readline())
            gen_model = (sample.get("lineage") or {}).get("gen_model")
        except:
            pass

    gen_prices = costs.get(gen_model or "gpt-3.5-turbo", {"input_per_1k": 0.0, "output_per_1k": 0.0})
    judge_prices = costs.get(judge_cfg["model"], {"input_per_1k": 0.0, "output_per_1k": 0.0})

    cost_gen_per_q = (avg_pt / 1000.0) * gen_prices["input_per_1k"] + (avg_ct / 1000.0) * gen_prices["output_per_1k"]
    # judge tokens aren’t in NDJSON; leave as 0 or add separate judge-token logging if you want
    cost_judge_per_q = 0.0

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "rag-prompt-evals"))
    with mlflow.start_run(run_name=f"{Path(args.ndjson).stem}") as run:
        mlflow.log_params({
          "split": args.split,
          "judge_model": judge_cfg["model"],
          "judge_temp": judge_cfg["temperature"],
         "gen_model": gen_model or "gpt-3.5-turbo"
        })
        mlflow.log_metrics({
          "faithfulness_mean": sum(faith)/max(1,n),
          "answer_rel_mean": sum(ansrel)/max(1,n),
          "context_rel_mean": sum(ctxrel)/max(1,n),
          "format_error_rate": fmt_err,
          "bad_refusal_rate": bad_ref,
          "p95_latency_ms": p95_lat,
          "avg_latency_ms": avg_lat,
          "avg_prompt_tokens": avg_pt,
          "avg_completion_tokens": avg_ct,
          "cost_per_query_usd": c,
          "n": n,
          "cost_gen_per_query_usd": cost_gen_per_q,
          "cost_judge_per_query_usd": cost_judge_per_q
        })
        # artifacts
        jd = Path("artifacts"); jd.mkdir(exist_ok=True)
        # store configs for reproducibility
        mlflow.log_artifact(args.judge)
        mlflow.log_artifact(args.gates)
        mlflow.log_artifact(args.costs)
        # store a small audit sample
        k = (args.sample_audit or load_yaml(args.gates)["sample_for_audit"])
        sample_path = jd / f"audit_{Path(args.ndjson).stem}.json"
        # naive: sample K raw lines
        with open(args.ndjson, "r", encoding="utf-8") as f, open(sample_path,"w",encoding="utf-8") as w:
            lines=f.readlines(); random.shuffle(lines)
            for ln in lines[:k]: w.write(ln)
        mlflow.log_artifact(str(sample_path))

if __name__=="__main__":
    main()
