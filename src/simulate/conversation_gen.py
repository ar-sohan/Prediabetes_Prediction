"""Row of true features -> synthetic chat with controlled noise. Uses the SIMULATOR model (not the system model)."""
import numpy as np
from ..llm import chat_json
from ..config import CFG

CONDITIONS = {
  "clean":   dict(frac=1.0, style="Natural, polite teenager/young adult. Give exact numbers.", vague=False),
  "vague":   dict(frac=1.0, style="Casual. Give approximate or rounded amounts ('around 60 kilos', 'a few sodas'), never exact.", vague=True),
  "partial": dict(frac=0.5, style="Natural. Mention only the allowed facts, exact numbers.", vague=False),
  "messy":   dict(frac=0.7, style="Slang, typos, off-topic tangents, short replies, one self-correction. Numbers may be approximate.", vague=True),
}

def generate(truth: dict, spec, condition, seed=0):
    cfg = CONDITIONS[condition]; rng = np.random.default_rng(seed)
    feats = [f for f in spec.feature if f in truth and truth[f] == truth[f]]
    k = max(1, int(round(cfg["frac"] * len(feats))))
    revealed = list(rng.choice(feats, k, replace=False))
    desc = {r.feature: f"{r.description} ({r.unit})" for r in spec.itertuples()}
    facts = "\n".join(f"- {desc[f]}: {truth[f]:g}" for f in revealed)
    sys = ("You write realistic chat transcripts between a young person (USER) and a friendly health coach (COACH). "
           "The USER reveals ONLY the facts listed, naturally across turns, in their own words, and invents NO other health numbers. "
           f"Style: {cfg['style']} 6-10 turns. Return ONLY JSON: {{\"turns\": [{{\"role\": \"user\"|\"coach\", \"content\": str}}, ...]}}")
    out = chat_json(sys, f"FACTS TO REVEAL:\n{facts}", model=CFG["llm"]["sim_model"], temperature=0.7, max_tokens=1500)
    return out["turns"], revealed
