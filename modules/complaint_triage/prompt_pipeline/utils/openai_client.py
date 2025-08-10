from __future__ import annotations
import time, yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]   # .../prompt_pipeline
CONFIG_DIR = BASE_DIR / "config"

with open(CONFIG_DIR / "base.yml", "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

LLM_MODEL = _CFG.get("llm_model", "gpt-3.5-turbo")
MODEL_VERSION = _CFG.get("model_version", "text-embedding-3-large + gpt-3.5-turbo")
DEFAULT_TEMP = float(_CFG.get("default_temperature", 0.2))

_client = OpenAI()

def ask_llm(
    system: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, info) where info has usage + timings + model metadata.
    """
    model = model or LLM_MODEL
    temperature = DEFAULT_TEMP if temperature is None else float(temperature)

    t0 = time.time()
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    dt = (time.time() - t0) * 1000.0

    msg = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    info = {
        "latency_ms": round(dt, 1),
        "model": model,
        "model_version": MODEL_VERSION,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "metadata": metadata or {},
    }
    return msg, info
