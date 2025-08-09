from prompt.prompt_versioning import load_prompt

base = "prompt/versions"
p1 = load_prompt(f"{base}/triage_v1.yml")
p2 = load_prompt(f"{base}/triage_v2.yml")

print("v1 prompt_version:", p1.version())
print("v2 prompt_version:", p2.version())
print("different?:", p1.version() != p2.version())
