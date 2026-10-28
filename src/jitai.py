"""JITAI: components + decision rules. Intervention = instruction to the LLM, not a fixed script."""
from dataclasses import dataclass
import random

INTERVENTIONS = {
  "diet":      "Warmly ask if they'd like a small goal about sugary drinks this week. Offer 2 concrete options. Don't lecture.",
  "sleep":     "Share one practical sleep tip and ask if it feels doable.",
  "activity":  "Suggest one enjoyable way to add 10 minutes of movement, linked to their interests.",
  "clinic":    "Gently suggest a check-up with a doctor or clinic. Do NOT state or imply a diagnosis.",
  "none":      None,
}
JITAI_SPEC = {
  "distal_outcome": "Lower progression to prediabetes/diabetes",
  "proximal_outcomes": ["sugary drinks/week", "activity min/week", "sleep hours", "follow-up completion"],
  "decision_points": "session start; daily check (simulation)",
  "tailoring_variables": ["risk tier", "need per behavior", "days since last intervention", "receptivity", "burden", "safety flags"],
  "options": list(INTERVENTIONS),
}

@dataclass
class Ctx:
    risk_tier: str = "low"            # low|moderate|high|insufficient_data
    needs: dict = None                # observed need per behavior in [0,1]
    days_since_last: int = 99
    receptive: bool = True
    followup_done: bool = False
    safety_flag: bool = False
    day: int = 0

def decide_tailored(c: Ctx, min_gap=2):
    if c.safety_flag: return "none"                       # safety overrides all
    if c.days_since_last < min_gap or not c.receptive: return "none"   # burden control
    if c.risk_tier == "high" and not c.followup_done: return "clinic"
    if c.needs:
        beh, val = max(c.needs.items(), key=lambda kv: kv[1])
        if val > 0.5 and (c.risk_tier != "low" or val > 0.7): return beh
    return "none"

def decide_none(c, rng=None): return "none"
def decide_random(c, rng=None):
    rng = rng or random; return rng.choice(["diet", "sleep", "activity", "clinic"]) if rng.random() < 0.3 else "none"
def decide_fixed(c, rng=None):
    return ["diet", "sleep", "activity"][(c.day // 3) % 3] if c.day % 3 == 0 else "none"
POLICIES = {"none": decide_none, "random": decide_random, "fixed": decide_fixed,
            "tailored": lambda c, rng=None: decide_tailored(c)}
