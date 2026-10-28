"""Chat agent = safety + memory brief + JITAI instruction -> LLM."""
from . import safety
from .llm import chat
from .jitai import INTERVENTIONS
from .config import CFG

MEM_RULES = ("Use the MEMORY block naturally. Do not ask again for facts listed there. If you lack a fact, say you don't know it. "
             "Never invent facts about the user. If the user asked to forget something, you must not mention it.")

def respond(message, history=None, memory_brief="", intervention="none", model=None):
    flags = safety.check_user(message)
    for f in ("crisis", "emergency"):
        if f in flags: return safety.FIXED[f], flags
    sys = safety.BASE_RULES + "\n" + MEM_RULES
    for f in ("eating", "diagnosis"):
        if f in flags: sys += "\n" + safety.PROMPT_RULES[f]
    if memory_brief: sys += f"\n\nMEMORY:\n{memory_brief}"
    instr = INTERVENTIONS.get(intervention)
    if instr and "eating" not in flags: sys += f"\n\nINTERVENTION THIS TURN (weave in naturally): {instr}"
    msgs = (history or []) + [{"role": "user", "content": message}]
    out = chat(sys, msgs, model=model or CFG["llm"]["system_model"], temperature=0.0, max_tokens=400)
    if safety.check_assistant(out):
        out = "I can't diagnose conditions, but I'd really encourage a check-up with a doctor who can look at this properly."
        flags.append("diagnosis_blocked")
    return out, flags
