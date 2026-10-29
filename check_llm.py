"""Run first:  python check_llm.py   One quick attempt per model (30 s timeout). Prints the server's real answer."""
import os, sys, time
from src.config import CFG
L = CFG["llm"]; prov = L.get("provider", "anthropic")
print(f"provider: {prov}" + (f"   base_url: {L.get('base_url')}" if prov == "openai_compatible" else ""), flush=True)
env = "ANTHROPIC_API_KEY" if prov == "anthropic" else (L.get("api_key_env") or "")
if env and (len(env) > 40 or not env.replace("_", "").isalnum() or env != env.upper()):
    sys.exit(f"config.yaml api_key_env must be a variable NAME like GEMINI_API_KEY, not the key itself. Got: {env[:6]}...")
if env and not os.environ.get(env) and "localhost" not in L.get("base_url", ""):
    sys.exit(f"{env} is not set in this terminal.  Git Bash:  export {env}=your_key")
for role in [k for k in L if k.endswith("_model")]:
    model = L[role]; print(f"testing {role}: {model} ...", flush=True)
    try:
        if prov == "openai_compatible":
            import requests
            r = requests.post(L["base_url"].rstrip("/") + "/chat/completions", timeout=30,
                json=dict(model=model, max_tokens=30, messages=[{"role": "user", "content": "Reply with the single word OK"}]),
                headers={"Authorization": "Bearer " + os.environ.get(env or "LLM_API_KEY", "none")})
            print(f"   HTTP {r.status_code}: {r.text[:300]}")
        else:
            from src.llm import chat
            print("   [ok]", chat("You are terse.", "Reply with the single word OK", model=model, max_tokens=30, tag=str(time.time())).strip()[:40])
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {str(e)[:300]}")
print("\nHTTP 200 with a 'choices' text = working. 404 = wrong model name. 429 = free quota/rate limit. 401/403 = key problem.")
