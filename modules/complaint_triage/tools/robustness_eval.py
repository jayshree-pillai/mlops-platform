import json, argparse, csv, yaml
from jinja2 import Template
from openai import OpenAI
ap=argparse.ArgumentParser()
ap.add_argument("--base", required=True)      # NDJSON (winner)
ap.add_argument("--ablated", required=True)   # NDJSON (drop-top=1)
ap.add_argument("--judge", default="config/judge.yml")
ap.add_argument("--out", default="artifacts/robustness_table.csv")
args=ap.parse_args()

cfg=yaml.safe_load(open(args.judge,"r",encoding="utf-8"))
triad=Template(cfg["triad_prompt"])
client=OpenAI()

def means(path):
    R=[json.loads(l) for l in open(path,"r",encoding="utf-8")]
    f=a=c=0.0; n=0
    for r in R:
        q=r.get("query",""); ctx="\n\n".join([h.get("text","") for h in r.get("hits",[])])
        ans=r.get("answer","")
        jr=client.chat.completions.create(model=cfg["model"], temperature=0.0,
            messages=[{"role":"user","content":triad.render(query=q, context=ctx[:6000], ans=ans[:6000])}])
        try: s=json.loads(jr.choices[0].message.content)
        except: s={"faithfulness":0.0,"answer_relevance":0.0,"context_relevance":0.0}
        f+=s["faithfulness"]; a+=s["answer_relevance"]; c+=s["context_relevance"]; n+=1
    return f/max(1,n), a/max(1,n), c/max(1,n)

fb,ab,cb=means(args.base); fa,aa,ca=means(args.ablated)
from pathlib import Path; Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out,"w",newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(["metric","paraphrase","ablation_drop1","delta"])
    for (n,x,y) in [("faithfulness",fb,fa),("answer_relevance",ab,aa),("context_relevance",cb,ca)]:
        w.writerow([n, round(x,4), round(y,4), round(y-x,4)])
print("Wrote", args.out)
