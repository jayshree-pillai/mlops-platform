import os, yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # .../prompt_pipeline
PROMPTS_DIR = BASE_DIR / "prompts"
CONFIG_DIR = BASE_DIR / "config"

class PromptSpec:
    def __init__(self, meta: dict, jinja_text: str, folder: Path):
        self.meta = meta                  # entire YAML
        self.jinja = jinja_text           # Jinja template string
        self.folder = folder              # versions/ folder path
        self.name = meta.get("name")
        self.task = meta.get("task")
        self.version = meta.get("version")
        self.model = meta.get("model")
        self.top_k = meta.get("top_k", 4)
        self.temperature = meta.get("temperature", 0.2)
        self.few_shots = meta.get("few_shots", [])
        self.system = meta.get("system", "")
        self.output_schema = meta.get("output_schema", "json")

def _read_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_versions_map() -> dict:
    """config/versions.yml -> {'rag': 'summarize_v1_fewshot', ...}"""
    return _read_yaml(CONFIG_DIR / "versions.yml")

def load_prompt(task: str, version_name: str) -> PromptSpec:
    """
    task: 'rag' | 'classify' | 'text2sql'
    version_name: e.g., 'summarize_v1_fewshot'
    Expects:
      prompts/{task}/versions/{version_name}.yml
      prompts/{task}/versions/{version_name}.jinja
    """
    vdir = PROMPTS_DIR / task / "versions"
    yml_path = vdir / f"{version_name}.yml"
    jinja_path = vdir / f"{version_name}.jinja"

    if not yml_path.exists():
        raise FileNotFoundError(f"Missing YAML: {yml_path}")
    if not jinja_path.exists():
        raise FileNotFoundError(f"Missing Jinja: {jinja_path}")

    meta = _read_yaml(yml_path)
    jinja_text = jinja_path.read_text(encoding="utf-8")
    return PromptSpec(meta, jinja_text, vdir)

def load_active_prompt(task: str) -> PromptSpec:
    """Reads config/versions.yml and loads the active version for the task."""
    versions = load_versions_map()
    version_name = versions.get(task)
    if not version_name:
        raise KeyError(f"No active version set for task='{task}' in versions.yml")
    return load_prompt(task, version_name)
