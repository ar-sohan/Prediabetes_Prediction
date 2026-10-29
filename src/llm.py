"""One LLM wrapper: retries, on-disk cache, full logging (reproducibility). Needs ANTHROPIC_API_KEY."""
import os, json, time, hashlib, re, pathlib
from .config import ROOT, CFG
CACHE = ROOT / "logs" / "llm_cache"; CACHE.mkdir(parents=True, exist_ok=True)
_client = None

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
    for i in range(5):
        try:
            kw = dict(model=model, system=system, messages=messages, max_tokens=max_tokens)
            if _accepts("temperature"): kw["temperature"] = temperature
            r = _cli().messages.create(**kw)
            text = r.content[0].text; break
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404) or i == 4: raise     # not retryable: show the REAL error (key, model name, credits)
            time.sleep(2 ** i)
    json.dump(dict(model=model, system=system, messages=messages, temperature=temperature, text=text), open(f, "w"))
    return text

def parse_json(text):
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
