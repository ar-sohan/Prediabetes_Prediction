"""One LLM wrapper: retries, on-disk cache, full logging (reproducibility). Needs ANTHROPIC_API_KEY."""
import os, json, time, hashlib, re, pathlib
from .config import ROOT, CFG
CACHE = ROOT / "logs" / "llm_cache"; CACHE.mkdir(parents=True, exist_ok=True)
_client = None
import threading
_lock = threading.Lock(); _last = [0.0]

class APIError(Exception):
    def __init__(self, status_code, msg): super().__init__(f"HTTP {status_code}: {msg}"); self.status_code = status_code

def _throttle():
    gap = float(CFG["llm"].get("min_interval_sec", 0) or 0)       # free tiers limit requests/minute
    if gap <= 0: return
    with _lock:
        wait = _last[0] + gap - time.time()
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()

def _openai_compatible(model, system, messages, temperature, max_tokens):
    """Works with Ollama (local), Groq, OpenRouter, Google Gemini's OpenAI-compatible endpoint, etc."""
    import requests
    c = CFG["llm"]; key = os.environ.get(c.get("api_key_env") or "LLM_API_KEY", "none")
    body = dict(model=model, max_tokens=max_tokens + int(c.get("token_headroom", 0) or 0), temperature=temperature,   # reasoning models spend tokens thinking
                messages=([{"role": "system", "content": system}] if system else []) + messages)
    if c.get("reasoning_effort"): body["reasoning_effort"] = c["reasoning_effort"]     # optional: low | medium | high
    r = requests.post(c["base_url"].rstrip("/") + "/chat/completions", json=body, timeout=120,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    if r.status_code != 200: raise APIError(r.status_code, r.text[:400])
    return r.json()["choices"][0]["message"].get("content") or ""

def _cli():
    global _client
    if _client is None:
        import anthropic; _client = anthropic.Anthropic()
    return _client

def _accepts(name):
    """True if the installed SDK's messages.create() takes this argument (newer SDKs/models drop 'temperature')."""
    import inspect
    ps = inspect.signature(_cli().messages.create).parameters
    return name in ps or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values())

def chat(system, messages, model=None, temperature=0.0, max_tokens=1200, tag=""):
    model = model or CFG["llm"]["system_model"]
    if isinstance(messages, str): messages = [{"role": "user", "content": messages}]
    key = hashlib.sha256(json.dumps([model, system, messages, temperature, max_tokens, tag], sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists(): return json.load(open(f))["text"]
    for i in range(6):
        try:
            _throttle()
            if CFG["llm"].get("provider", "anthropic") == "openai_compatible":
                text = _openai_compatible(model, system, messages, temperature, max_tokens); break
            kw = dict(model=model, system=system, messages=messages, max_tokens=max_tokens)
            if _accepts("temperature"): kw["temperature"] = temperature
            r = _cli().messages.create(**kw)
            text = r.content[0].text; break
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404) or i == 5: raise     # not retryable: show the REAL error (key, model name, credits)
            w = min(10 * (i + 1), 60) if status == 429 else 2 ** i      # rate limit: wait longer
            print(f"  [retry {i + 1}/5] {str(e)[:200]}  -> waiting {w}s", flush=True)
            time.sleep(w)
    json.dump(dict(model=model, system=system, messages=messages, temperature=temperature, text=text), open(f, "w"))
    return text

def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)          # reasoning models (e.g. qwen, deepseek) add these
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1: s, e = t.find("["), t.rfind("]")
    return json.loads(t[s:e + 1])

def chat_json(system, messages, **kw):
    last = ""
    for i in range(3):
        last = chat(system, messages, tag=f"try{i}", **kw)     # API errors propagate with their real message
        try: return parse_json(last)
        except Exception: continue
    raise ValueError("LLM did not return valid JSON. Last output was:\n" + last[:600])
