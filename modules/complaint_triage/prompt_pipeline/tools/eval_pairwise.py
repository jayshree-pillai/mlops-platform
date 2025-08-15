import json, argparse, math, mlflow, yaml
from pathlib import Path
from jinja2 import Template
from openai import OpenAI

def elo_update(ra, rb, outcome, k=16):
    ea = 1/(1+10**((rb-ra)/400)); eb = 1-ea
    return ra + k*(outcome-ea), rb + k*((1-outcome)-eb)

def wilson_ci(k, n, z=1.96):
    if n==0: return (0.0,0.0)
    p=k/n; denom=1+z*z/n
    center=(p + z*z/(2*n))/denom
    margin=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return (max(0,center-margin), min(1,center+margin))

ap=argparse.ArgumentParser()
ap.add_argument("--a", required=True)   # NDJSON A
ap.add_argument("--b", required=True)   # NDJSON B
ap.add_argument("--judge", default="config/judge.yml")
ap.add_argument("--exp", default=None)
args=ap.parse_args()

A=[json.loads(l) for l in open(args.a, "r", encoding="utf-8")]
B=[json.loads(l) for l in open(args.b, "r", encoding="utf-8")]
n=min(len(A),len(B))

cfg=yaml.safe_load(open(args.judge,"r",encoding="utf-8"))
judge_model=cfg["model"]
tpl = Template(cfg["pairwise_prompt"])
client=OpenAI()

wins=ties=0; ra=rb=1200.0
for i in range(n):
    qa=A[i].get("query",""); qb=B[i].get("query","")
    if qa!=qb: continue
    ctx="\n\n".join([h.get("text","") for h in A[i].get("hits",[])])
    pa=A[i].get("answer",""); pb=B[i].get("answer","")
    pr=tpl.render(query=qa, context=ctx[:6000], a=pa[:5000], b=pb[:5000])
    jr=client.chat.completions.create(model=judge_model, temperature=0.0, messages=[{"role":"user","content":pr}])
    v=jr.choices[0].message.content.strip().upper()
    if v.startswith("A"): wins+=1; ra,rb=elo_update(ra,rb,1.0)
    elif v.startswith("B"):          ra,rb=elo_update(ra,rb,0.0)
    else:             ties+=1;       ra,rb=elo_update(ra,rb,0.5)

wr = wins/max(1,n); lo,hi=wilson_ci(wins, max(1,n))
mlflow.set_experiment(args.exp or "rag-prompt-evals")
with mlflow.start_run(run_name=f"PAIR_{Path(args.a).stem}_vs_{Path(args.b).stem}"):
    mlflow.log_params({"A":Path(args.a).name,"B":Path(args.b).name,"judge_model":judge_model})
    mlflow.log_metrics({"win_rate_A_over_B":wr,"win_rate_ci_low":lo,"win_rate_ci_high":hi,"elo_A":ra,"elo_B":rb,"ties":ties,"n":n})
print(f"win_rate={wr:.3f} [{lo:.3f},{hi:.3f}] eloA={ra:.1f} eloB={rb:.1f} ties={ties} n={n}")
