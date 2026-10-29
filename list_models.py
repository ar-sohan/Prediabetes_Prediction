"""Lists the model IDs your provider/key can use.  python list_models.py"""
import os, sys, requests
from src.config import CFG
L = CFG["llm"]; env = L.get("api_key_env") or ""
if not os.environ.get(env, ""): sys.exit(f"{env} is not set. Git Bash:  export {env}=your_key")
r = requests.get(L["base_url"].rstrip("/") + "/models", headers={"Authorization": "Bearer " + os.environ[env]}, timeout=30)
print("HTTP", r.status_code)
if r.status_code != 200: sys.exit(r.text[:300])
ids = sorted(m["id"] for m in r.json().get("data", []))
print("\n".join(ids))
print("\nPick a chat model (names like llama-...). Ignore ones with 'whisper', 'tts', 'guard', 'embed'.")
