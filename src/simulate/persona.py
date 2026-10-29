"""Scripted multi-session personas with traps: value change, correction, goal, long gap, deletion request."""
import numpy as np
from ..llm import chat_json
from ..config import CFG
from ..extraction import fmt_value

BEHAV = ["PAminWk", "ScreenTime", "AddSugars", "MealsOut"]   # behaviours that can change over time
HOBBIES = ["football", "gaming", "drawing", "swimming", "guitar", "cooking", "cycling"]

def make_personas(test_df, spec, n=20, seed=0, n_sessions=9):
    rng = np.random.default_rng(seed); feats = list(spec.feature)
    behav = [b for b in BEHAV if b in feats]; out = []
    for i, ridx in enumerate(rng.choice(len(test_df), n, replace=False)):
        row = test_df.iloc[ridx]; base = {f: float(row[f]) for f in feats if f in row and row[f] == row[f]}
        change_f = behav[i % len(behav)] if behav else next(iter(base))
        old = base[change_f]; new = round(old * (1.5 if change_f == "PAminWk" else 0.6), 1)
        del_f = next((f for f in ("FamIncC", "HIns4", "BMXWAIST") if f in base), [f for f in base if f != change_f][0])
        goal = {"AddSugars": "cut down on sugary drinks and sweets", "ScreenTime": "cut screen time before bed",
                "PAminWk": "be active 20 minutes most days", "MealsOut": "eat out less often"}.get(change_f, "improve one healthy habit")
        out.append(dict(id=f"p{i:02d}", hobby=HOBBIES[i % len(HOBBIES)], base=base, change_feature=change_f, old=old, new=new,
                        goal=goal, delete_feature=del_f, sessions=[
            dict(idx=1, reveal=base),
            dict(idx=2, reveal={}, note="small talk about hobby; no new health numbers"),
            dict(idx=4, reveal={change_f: old}, goal=goal),
            dict(idx=6, reveal={change_f: new}, note="reports progress on the goal"),
            dict(idx=7, reveal={}, note="long gap since last chat (3 months); casual check-in"),
            dict(idx=9, reveal={}, delete=del_f)]))
    return out

def gen_session(p, s, spec):
    desc = {r.feature: f"{r.description} ({r.unit})" for r in spec.itertuples()}
    rows = {r.feature: r for r in spec.itertuples()}
    facts = "\n".join(f"- {rows[f].description}: {fmt_value(rows[f], v)}" for f, v in s["reveal"].items() if f in rows) or "(no new health facts)"
    extra = ""
    if s.get("goal"): extra += f"\nThe USER sets this goal in this session: '{s['goal']}'."
    if s.get("delete"): extra += f"\nThe USER explicitly asks the coach to FORGET what they said about: {desc[s['delete']]}. Do not restate its value."
    if s.get("note"): extra += f"\nContext: {s['note']}."
    sys = ("You write one realistic chat session between a young person (USER, hobby: " + p["hobby"] + ") and a friendly health coach (COACH). "
           "The USER reveals ONLY the listed facts and invents no other health numbers. 6-8 turns. "
           'Return ONLY JSON: {"turns": [{"role": "user"|"coach", "content": str}, ...]}')
    return chat_json(sys, f"FACTS:\n{facts}{extra}", model=CFG["llm"]["sim_model"], temperature=0.7, max_tokens=3000)["turns"]
