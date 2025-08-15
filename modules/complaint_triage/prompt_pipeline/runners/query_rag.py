#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
from jinja2 import Environment, BaseLoader

from prompt_pipeline.utils.prompt_manager import load_active_prompt, load_prompt
from prompt_pipeline.utils.loader import get_retriever
from prompt_pipeline.utils.openai_client import ask_llm
from prompt_pipeline.utils.json_schemas import STRICT_RAG_JSON_SCHEMA

def _minify_text(s: str) -> str:
    # collapse all whitespace to single spaces; preserves content
    return " ".join((s or "").split())
def render_template(jinja_text: str, **kw) -> str:
    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(jinja_text)
    return tmpl.render(**kw)

def main():
    ap = argparse.ArgumentParser(description="RAG query runner")
    ap.add_argument("--q", required=True, help="User query")
    ap.add_argument("--version", default=None, help="Prompt version name (overrides config)")
    ap.add_argument("--k", type=int, default=None, help="Top-K contexts (override)")
    ap.add_argument("--temp", type=float, default=None, help="LLM temperature override")
    ap.add_argument("--json", action="store_true", help="Print only JSON result")
    ap.add_argument("--min-avg-top3", type=float, default=0.10, help="Low-confidence avg_top3 threshold")
    ap.add_argument("--min-margin", type=float, default=0.00, help="Low-confidence margin threshold")
    ap.add_argument("--min-keep", type=float, default=0.05, help="Very low bar to keep any hit")
    ap.add_argument("--drop-top", type=int, default=0, help="Drop top N retrieved chunks before prompting")
    args = ap.parse_args()

    # 1) Load prompt spec
    spec = load_prompt("rag", args.version) if args.version else load_active_prompt("rag")

    # 2) Retrieve contexts (keep alignment between contexts and ctx_meta)
    retriever = get_retriever()
    k = args.k or getattr(spec, "top_k", None) or 4
    hits = retriever.retrieve(args.q, k=k)
    if args.drop_top > 0:
        hits = hits[args.drop_top:]

    # keep 1:1 mapping; minify whitespace only
    triplets = [(t, meta, score) for (t, meta, score) in hits]

    contexts = [_minify_text(t) for (t, _, _) in triplets]
    ctx_meta = [{"score": score, **(meta or {})} for (_, meta, score) in triplets]

    # Retrieval quality signals (use triplets, not hits)
    top_scores = [s for (_, _, s) in triplets[:3]]
    avg_top3 = (sum(top_scores) / len(top_scores)) if top_scores else 0.0
    margin = (top_scores[0] - top_scores[1]) if len(top_scores) >= 2 else 0.0
    low_confidence = (avg_top3 < args.min_avg_top3 and margin < args.min_margin)

    # Early refusal if nothing is usable
    min_keep = args.min_keep
    has_any_relevant = any(score >= min_keep for (_, _, score) in hits)
    # Early refusal ONLY if there are truly no hits
    if not triplets:
        refusal = {"bullets": [], "confidence": 0.0, "note": "No relevant information found"}
        out = {
            "answer": json.dumps(refusal, ensure_ascii=False),
            "prompt_version": spec.name,
            "llm_model": None,
            "model_version": None,
            "low_confidence": True,
            "retrieval_stats": {"avg_top3": avg_top3, "margin": margin},
            "lineage": {
                "prompt_version": spec.name,
                "topk": k,
                "temperature": args.temp if args.temp is not None else getattr(spec, "temperature", 0.0),
                "gen_model": getattr(spec, "model", None),
            },
            "retriever": {
                "faiss_version": retriever.version,
                "ntotal": retriever.info()["ntotal"],
                "k": k,
                "hits": [{"text": t, "meta": m} for t, m in zip(contexts, ctx_meta)],
            },
            "timing_tokens": {},
        }
        print(json.dumps(out if args.json else out, ensure_ascii=False, indent=None if args.json else 2))
        return

    # Token/latency clamps
    def _truncate_contexts(texts: List[str], max_chars=500, max_ctx=3) -> List[str]:
        out = []
        for t in texts[:max_ctx]:
            out.append(t if len(t) <= max_chars else t[:max_chars])
        return out
    contexts = _truncate_contexts(contexts, max_chars=700, max_ctx=3)

    # 3) Render prompt
    user_prompt = render_template(
        spec.jinja,
        query=args.q,
        contexts=contexts,
        few_shots=getattr(spec, "few_shots", []),
    )

    # 4) Call LLM with JSON mode enforced
    text, info = ask_llm(
        system=(spec.system or ""),
        user_prompt=user_prompt + "\n\nRespond ONLY with JSON that matches the schema.",
        model=spec.model,
        temperature=0.0 if args.temp is None else args.temp,  # deterministic DEV
        metadata={
            "task": "rag",
            "prompt_version": spec.name,
            "faiss_version": retriever.version,
            "artifact_dir": retriever.info().get("artifact_dir"),
            "top_k": k,
        },
        response_format=STRICT_RAG_JSON_SCHEMA,
        max_tokens=180,  # <-- new: keep completion tight; JSON is small anyway
    )

    # 5) Guarantee valid JSON in output (salvage braces if needed)
    def _force_json(s: str) -> str:
        if not s:
            return json.dumps({"bullets": [], "confidence": 0.0, "evidence": []}, ensure_ascii=False)
        try:
            json.loads(s)
            return s
        except Exception:
            pass
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            frag = s[start:end+1]
            try:
                json.loads(frag)
                return frag
            except Exception:
                pass
        return json.dumps({"bullets": [], "confidence": 0.0, "evidence": []}, ensure_ascii=False)

    safe_text = _force_json(text)

    # 6) Emit result
    out = {
        "answer": safe_text,
        "prompt_version": spec.name,
        "llm_model": info["model"],
        "model_version": info["model_version"],
        "low_confidence": low_confidence,
        "retrieval_stats": {"avg_top3": avg_top3, "margin": margin},
        "lineage": {
            "prompt_version": spec.name,
            "topk": k,
            "temperature": args.temp if args.temp is not None else 0.0,
            "gen_model": info["model"],
        },
        "retriever": {
            "faiss_version": retriever.version,
            "ntotal": retriever.info()["ntotal"],
            "k": k,
            "hits": [{"text": t, "meta": m} for t, m in zip(contexts, ctx_meta)],
        },
        "timing_tokens": {
            "latency_ms": info["latency_ms"],
            "prompt_tokens": info["prompt_tokens"],
            "completion_tokens": info["completion_tokens"],
            "total_tokens": info["total_tokens"],
        },
    }
    print(json.dumps(out if args.json else out, ensure_ascii=False, indent=None if args.json else 2))

if __name__ == "__main__":
    main()
