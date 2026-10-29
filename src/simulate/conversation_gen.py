"""Row of true features -> synthetic chat with controlled noise. Uses the SIMULATOR model (not the system model)."""
import numpy as np
from ..llm import chat_json
from ..config import CFG
from ..extraction import fmt_value

CONDITIONS = {
  "clean":   dict(n=8, style="Natural, polite teenager/young adult. Give exact numbers.", vague=False),
  "vague":   dict(n=8, style="Casual. Give approximate or rounded amounts ('around 60 kilos', 'a few sodas'), never exact.", vague=True),
  "partial": dict(n=4, style="Natural. Mention only the allowed facts, exact numbers.", vague=False),
  "messy":   dict(n=6, style="Slang, typos, off-topic tangents, short replies, one self-correction. Numbers may be approximate.", vague=True),
}

def generate(truth: dict, spec, condition, seed=0):
    cfg = CONDITIONS[condition]; rng = np.random.default_rng(seed)
    feats = [f for f in spec.feature if f in truth and truth[f] == truth[f]]
    k = min(cfg["n"], len(feats))                       # one session reveals only a few facts; memory accumulates them
    revealed = list(rng.choice(feats, k, replace=False))
    facts = "\n".join(f"- {r.description}: {fmt_value(r, truth[r.feature])}" for r in spec.itertuples() if r.feature in revealed)
    sys = ("You write realistic chat transcripts between a young person (USER) and a friendly health coach (COACH). "
           "The USER reveals ONLY the facts listed, naturally across turns, in their own words, and invents NO other health numbers. "
           f"Style: {cfg['style']} 8-14 turns. Return ONLY JSON: {{\"turns\": [{{\"role\": \"user\"|\"coach\", \"content\": str}}, ...]}}")
    out = chat_json(sys, f"FACTS TO REVEAL:\n{facts}", model=CFG["llm"]["sim_model"], temperature=0.7, max_tokens=3000)
    return out["turns"], revealed
