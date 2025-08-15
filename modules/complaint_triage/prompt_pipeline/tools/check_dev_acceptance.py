#!/usr/bin/env python3
# Simple DEV acceptance checker for RAG NDJSON outputs.
# Usage:
#   python3 tools/check_dev_acceptance.py \
#     --in runs/dev/dev_s_v0_k4_t00.ndjson \
#     --report runs/dev/dev_s_v0_k4_t00.report.json \
#     --max-format-error 0.02 --max-refusal 0.05 --max-p95 3000 --max-avg-tokens 800
import argparse, json, math
from statistics import mean

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input NDJSON file")
    ap.add_argument("--report", dest="report", required=False, help="Where to write JSON report")
    ap.add_argument("--max-format-error", type=float, default=0.02)
    ap.add_argument("--max-refusal", type=float, default=0.05)
    ap.add_argument("--max-p95", type=float, default=3000.0)
    ap.add_argument("--max-avg-tokens", type=float, default=800.0)
    return ap.parse_args()
def coerce(ans):
    if isinstance(ans,str):
        try: ans=json.loads(ans)
        except: return {}
    return ans if isinstance(ans,dict) else {}

def is_valid_answer(ans):
    b = ans.get("bullets"); e = ans.get("evidence"); c = ans.get("confidence")
    if not isinstance(b, list) or not isinstance(e, list) or not isinstance(c, (int, float)):
        return False
    # refusal shape is allowed as valid schema
    if len(b) == 0 and len(e) == 0 and c <= 0.05:
        return True
    # normal shape
    if not (3 <= len(b) <= 6):
        return False
    if not all(isinstance(x, str) and x.strip() for x in b):
        return False
    if not all(isinstance(ev, dict) and ev.get("doc_id") and ev.get("span") for ev in e):
        return False
    return True

def is_refusal(ans):
    b = ans.get("bullets"); e = ans.get("evidence"); c = ans.get("confidence")
    return isinstance(b, list) and isinstance(e, list) and len(b) == 0 and len(e) == 0 and isinstance(c, (int, float)) and c <= 0.05

def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k); c = f + (0 if k.is_integer() else 1)
    if f == c or c >= len(xs):
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)

def main():
    args = parse_args()
    tot = 0; valids = 0; refusals = 0
    lat = []; toks = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            tot += 1
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ans = coerce(obj.get("answer", {}))
            if is_valid_answer(ans):
                valids += 1
                if is_refusal(ans):
                    refusals += 1
            tt = obj.get("timing_tokens", {})
            lm = tt.get("latency_ms"); tk = tt.get("total_tokens")
            if isinstance(lm, (int, float)): lat.append(float(lm))
            if isinstance(tk, (int, float)): toks.append(float(tk))

    fmt_err = 1.0 - (valids / tot) if tot else 1.0
    ref_rate = (refusals / tot) if tot else 1.0
    p95 = percentile(lat, 95) if lat else None
    avg_tokens = (mean(toks) if toks else None)

    report = {
        "file": args.inp,
        "total": tot,
        "format_error_rate": None if tot==0 else round(fmt_err, 4),
        "refusal_rate": None if tot==0 else round(ref_rate, 4),
        "p95_latency_ms": None if p95 is None else round(p95, 1),
        "avg_total_tokens": None if avg_tokens is None else round(avg_tokens, 1),
        "thresholds": {
            "format_error_rate_max": args.max_format_error,
            "refusal_rate_max": args.max_refusal,
            "p95_latency_ms_max": args.max_p95,
            "avg_total_tokens_max": args.max_avg_tokens
        },
        "pass": (
            tot>0 and
            (fmt_err <= args.max_format_error) and
            (ref_rate <= args.max_refusal) and
            (p95 is None or p95 <= args.max_p95) and
            (avg_tokens is None or avg_tokens <= args.max_avg_tokens)
        )
    }

    j = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as g:
            g.write(j+"\n")
    print(j)

if __name__ == "__main__":
    main()
