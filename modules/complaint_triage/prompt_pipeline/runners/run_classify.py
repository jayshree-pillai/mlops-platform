#Same loader, optionally appends retrieved snippets to the prompt when using *_llm_rag template.
#Uses config/labels.yml for target labels + thresholds.
#Emits JSON: {label, score, rationale, prompt_version, latency/tokens}.