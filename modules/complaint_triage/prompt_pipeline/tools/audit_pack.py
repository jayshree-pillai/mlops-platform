import json, argparse, random, textwrap
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument("--ndjson", required=True)
ap.add_argument("--out", default="artifacts/audit_pack.md")
ap.add_argument("--k", type=int, default=25)
args=ap.parse_args()

rows=[json.loads(l) for l in open(args.ndjson, "r", encoding="utf-8")]
random.shuffle(rows); rows=rows[:args.k]
md=["# Audit Pack"]
for r in rows:
    md.append("## Query\n"+(r.get("query","")))
    hits=r.get("hits",[])
    md.append("\n### Retrieved (top 3)\n"+ "\n".join([f"- {(h.get('meta') or {}).get('source','')}: {textwrap.shorten(h.get('text',''),220)}" for h in hits[:3]]))
    md.append("\n### Answer\n```\n"+(r.get("answer",""))+"\n```")
    md.append("\n### Lineage\n- "+json.dumps(r.get("lineage",{})))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n\n".join(md), encoding="utf-8")
print("Wrote", args.out)
