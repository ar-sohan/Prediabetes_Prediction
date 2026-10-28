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

def chat(system, messages, model=None, temperature=0.0, max_tokens=1200, tag=""):
    model = model or CFG["llm"]["system_model"]
    if isinstance(messages, str): messages = [{"role": "user", "content": messages}]
    key = hashlib.sha256(json.dumps([model, system, messages, temperature, max_tokens, tag], sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists(): return json.load(open(f))["text"]
    for i in range(5):
        try:
            r = _cli().messages.create(model=model, system=system, messages=messages,
                                       temperature=temperature, max_tokens=max_tokens)
            text = r.content[0].text; break
        except Exception as e:
            if i == 4: raise
            time.sleep(2 ** i)
    json.dump(dict(model=model, system=system, messages=messages, temperature=temperature, text=text), open(f, "w"))
    return text

def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1: s, e = t.find("["), t.rfind("]")
    return json.loads(t[s:e + 1])

def chat_json(system, messages, **kw):
    for i in range(3):
        try: return parse_json(chat(system, messages, tag=f"try{i}", **kw))
        except Exception: continue
    raise ValueError("LLM did not return valid JSON")
