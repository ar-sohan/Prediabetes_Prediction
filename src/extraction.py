"""Chat transcript -> structured features (null if not stated)."""
import pandas as pd
from pydantic import BaseModel, Field, create_model
from typing import Optional
from .config import P, CFG
from .llm import chat_json

class Measured(BaseModel):
    value: Optional[float] = None
    confidence: float = Field(0.0, ge=0, le=1)
    evidence: Optional[str] = None

def load_spec(chat_only=True):
    """Uses data/feature_spec_core.csv (core model features + value labels) when it exists."""
    core = P("feature_spec").parent / "feature_spec_core.csv"
    if core.exists():
        s = pd.read_csv(core)
        if "value_labels" not in s: s["value_labels"] = ""
        s["value_labels"] = s["value_labels"].fillna("")
        return s
    s = pd.read_csv(P("feature_spec")); s["value_labels"] = ""
    return s[s.chat_class.isin(["easy", "noisy"])] if chat_only else s

def parse_labels(s):
    d = {}
    for part in (s if isinstance(s, str) else "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            try: d[float(k.strip())] = v.strip()
            except ValueError: pass
    return d

def fmt_value(r, v):
    """Human wording of a true value for the conversation simulator (label if coded, else number + unit)."""
    lab = parse_labels(getattr(r, "value_labels", ""))
    return lab[float(v)] if float(v) in lab else f"{v:g} {r.unit}".strip()

def make_schema(spec):
    return create_model("Extraction", **{f: (Measured, Measured()) for f in spec.feature})

def render(turns):
    return "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)

def build_prompt(spec):
    feats = "\n".join(f"- {r.feature} ({r.unit}): {r.description}" +
                      (f" Codes: {r.value_labels}. Return the numeric code." if r.value_labels else "") for r in spec.itertuples())
    return ("You extract health-related facts from a chat between a USER and a COACH.\n"
            "Rules: use ONLY what the USER explicitly states about themselves. If a feature is not stated, value=null. "
            "NEVER guess or infer. Convert units to the unit shown. For vague amounts ('a few'), give a best numeric estimate "
            "with confidence <= 0.5. Binary features: 1=yes, 0=no. Put the user's own words in 'evidence'.\n"
            f"Features:\n{feats}\n\n"
            'Return ONLY JSON: {"<feature>": {"value": number|null, "confidence": 0-1, "evidence": string|null}, ...}')

def extract(turns, spec=None, model=None):
    spec = spec if spec is not None else load_spec()
    Schema = make_schema(spec)
    for _ in range(3):
        try:
            raw = chat_json(build_prompt(spec), render(turns), model=model or CFG["llm"]["system_model"])
            out = Schema.model_validate({k: raw.get(k, {}) for k in spec.feature}).model_dump()
            for r in spec.itertuples():                      # range check -> drop impossible values
                v = out[r.feature]["value"]
                if v is not None and not (r.min <= v <= r.max): out[r.feature] = Measured().model_dump()
            return out
        except Exception:
            continue
    return {f: Measured().model_dump() for f in spec.feature}
