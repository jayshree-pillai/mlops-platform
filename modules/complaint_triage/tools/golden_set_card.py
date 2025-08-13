import json, argparse, collections, statistics as st, math
from pathlib import Path

def load_jsonl(p): 
    return [json.loads(l) for l in open(p,"r",encoding="utf-8")]

def lens(ds, key):
    return [len((x.get(key) or x.get("query") or "")) for x in ds]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="artifacts/golden_set_card.md")
    args=ap.parse_args()

    val=load_jsonl(args.val); test=load_jsonl(args.test)
    Path("artifacts").mkdir(exist_ok=True)
    def describe(name, ds):
        L = lens(ds, "query")
        return f"""### {name}
- n = {len(ds)}
- len(query): mean={st.mean(L):.1f}, p50={st.median(L):.0f}, p95={sorted(L)[int(0.95*(len(L)-1))]:.0f}
- sources (if present): top-10 = {collections.Counter([h.get('source','') for x in ds for h in x.get('hits',[]) if isinstance(h, dict)]).most_common(10)}
- edge cases: empty_context={sum(1 for x in ds if not x.get('hits'))}, very_long_query={sum(1 for x in ds if len((x.get('query') or ''))>500)}
"""
    md = "# Golden-set Card\n" + describe("VAL", val) + "\n" + describe("TEST", test)
    Path(args.out).write_text(md, encoding="utf-8")
    print("Wrote", args.out)

if __name__=="__main__":
    main()
