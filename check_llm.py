"""Run first:  python check_llm.py     Tests the API key and each model named in config.yaml, and prints the REAL error if any."""
import os, sys
from src.config import CFG
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set in this terminal.\n  Git Bash:   export ANTHROPIC_API_KEY=sk-ant-...\n  PowerShell: $env:ANTHROPIC_API_KEY='sk-ant-...'\n  cmd:        set ANTHROPIC_API_KEY=sk-ant-...")
import anthropic
c = anthropic.Anthropic()
for role, model in CFG["llm"].items():
    try:
        r = c.messages.create(model=model, max_tokens=20, messages=[{"role": "user", "content": "Reply with the word OK"}])
        print(f"[ok]   {role:12s} {model}: {r.content[0].text.strip()}")
    except Exception as e:
        print(f"[FAIL] {role:12s} {model}\n       {type(e).__name__}: {e}")
print("\nIf a model FAILS with not_found, change its name in config.yaml to one your account can use.")
