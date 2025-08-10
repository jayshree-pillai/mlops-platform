#Reads ARTIFACT_DIR from env.
#Uses utils.loader.retriever(k) → returns top‑k docs.
#Loads template from prompts/rag/<version>.jinja (version from config/versions.yml or --version flag).
#Renders prompt (Jinja) with {query, contexts}.
#Calls OpenAI via utils.openai_client.ask_llm() and prints JSON: {answer, contexts, latency_ms, tokens_in/out, prompt_version}.
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from jinja2 import Environment, BaseLoader

from prompt_pipeline.utils.prompt_manager import load_active_prompt, load_prompt
from prompt_pipeline.utils.loader import get_retriever
from prompt_pipeline.utils.openai_client import ask_llm

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
    args = ap.parse_args()

    # 1) Load prompt spec
    if args.version:
        spec = load_prompt("rag", args.version)
    else:
        spec = load_active_prompt("rag")

    # 2) Retrieve contexts
    retriever = get_retriever()
    k = args.k or spec.top_k or 4
    hits = retriever.retrieve(args.q, k=k)
    contexts = [t for (t, meta, score) in hits]
    ctx_meta = [{"score": score, **(meta or {})} for (t, meta, score) in hits]

    # 3) Render Jinja with few-shots (if any)
    user_prompt = render_template(
        spec.jinja,
        query=args.q,
        contexts=contexts,
        few_shots=spec.few_shots,
    )

    # 4) Ask LLM
    text, info = ask_llm(
        system=spec.system or "",
        user_prompt=user_prompt,
        model=spec.model,
        temperature=args.temp if args.temp is not None else spec.temperature,
        metadata={
            "task": "rag",
            "prompt_version": spec.name,
            "faiss_version": retriever.version,
            "artifact_dir": retriever.info().get("artifact_dir"),
            "top_k": k,
        },
    )

    # 5) Build output
    out = {
        "answer": text,
        "prompt_version": spec.name,
        "llm_model": info["model"],
        "model_version": info["model_version"],
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
