"""Simulated user response model for the JITAI experiment. ALL parameters are assumptions -> report + sensitivity analysis."""
import numpy as np
from dataclasses import dataclass, field
from ..jitai import Ctx, POLICIES

BEHS = ["diet", "sleep", "activity"]
def sig(x): return 1 / (1 + np.exp(-x))

@dataclass
class SimUser:
    base_risk: float; receptivity: float; needs: dict           # true need per behavior in [0,1]
    health: float = 0.3; burden: float = 0.0; followup: bool = False
    last: int = 99; msgs: int = 0; first_escalation: int = None
    def risk(self): return float(np.clip(self.base_risk * (1.3 - 0.6 * self.health), 0, 1))
    def tier(self): r = self.risk(); return "high" if r >= 0.6 else ("moderate" if r >= 0.3 else "low")

def make_users(n, rng):
    return [SimUser(base_risk=float(rng.beta(2, 3)), receptivity=float(rng.beta(3, 2)),
                    needs={b: float(rng.beta(2, 2)) for b in BEHS}) for _ in range(n)]

def run_policy(policy, n_users=200, days=90, seed=0, effect=0.08, engage_bias=-1.0, obs_noise=0.15, fatigue=0.15):
    rng = np.random.default_rng(seed); users = make_users(n_users, rng); pol = POLICIES[policy]
    h0 = [u.health for u in users]
    for d in range(days):
        for u in users:
            u.burden = max(0.0, u.burden - 0.05); u.last += 1
            obs = {b: float(np.clip(u.needs[b] + rng.normal(0, obs_noise), 0, 1)) for b in BEHS}   # noisy observation
            ctx = Ctx(risk_tier=u.tier(), needs=obs, days_since_last=u.last, receptive=bool(u.receptivity - 0.5 * u.burden > 0.35),
                      followup_done=u.followup, day=d)
            a = pol(ctx, rng)
            if a == "none": continue
            u.msgs += 1; u.last = 0
            fit = 0.6 if a == "clinic" else u.needs[a]
            engaged = rng.random() < sig(engage_bias + 2.5 * u.receptivity + 2.0 * fit - 2.0 * u.burden)
            u.burden += fatigue
            if engaged:
                if a == "clinic":
                    if rng.random() < 0.35: u.followup = True; u.first_escalation = u.first_escalation or d
                else:
                    u.health = min(1.0, u.health + effect * fit); u.needs[a] = max(0.0, u.needs[a] - 0.5 * effect)
    hi = [u for u in users if u.base_risk >= 0.6]
    return dict(policy=policy, health_gain=float(np.mean([u.health - h for u, h in zip(users, h0)])),
                msgs_per_week=float(np.mean([u.msgs for u in users]) / (days / 7)),
                mean_final_burden=float(np.mean([u.burden for u in users])),
                high_risk_escalated=float(np.mean([u.followup for u in hi])) if hi else float("nan"),
                time_to_escalation=float(np.nanmean([u.first_escalation for u in hi if u.first_escalation is not None] or [np.nan])))
