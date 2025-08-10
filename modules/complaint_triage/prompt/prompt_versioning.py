from dataclasses import dataclass, asdict
import hashlib, json, os, yaml

@dataclass(frozen=True)
class PromptSpec:
    system: str
    instruction: str
    template: str
    params: dict

    def version(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

def load_prompt(path: str) -> PromptSpec:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    params = cfg.get('params', {}) or {}
    params.setdefault('model', os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
    return PromptSpec(
        system=cfg['system'],
        instruction=cfg['instruction'],
        template=cfg['template'],
        params=params
    )
