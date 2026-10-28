"""Rule-based guardrails (fast, auditable). Extend the patterns for your red-team set."""
import re
CRISIS = re.compile(r"\b(kill myself|suicid\w*|end(ing)? my life|want to die|self[- ]harm|hurt(ing)? myself|cut(ting)? myself)", re.I)
EATING = re.compile(r"\b(stop eating|starv(e|ing) myself|skip (all )?meals to lose|purge|make myself (throw up|vomit)|"
                    r"laxatives? to lose|hate my body|so fat|need to lose \d+ ?(kg|pounds|lbs) fast|anorex|bulimi)", re.I)
EMERGENCY = re.compile(r"\b(can'?t stop vomiting|vomiting (a lot|and)|extremely thirsty|confus(ed|ion)|fruity breath|"
                       r"passed out|unconscious|trouble breathing|severe stomach pain)\b", re.I)
DIAG_REQ = re.compile(r"\b(do i have|am i|i have|i know i have|confirm i have) (a )?(pre-?)?diabet|diagnos|\b(insulin|metformin) (dose|dosage)|how (much|many) units", re.I)
DIAG_OUT = re.compile(r"\byou (have|likely have|probably have|are) (type ?[12] )?(pre-?)?diabet", re.I)

FIXED = {
  "crisis": "I'm really sorry you're feeling this way. You deserve support right now. Please contact a local emergency number or a crisis line "
            "(in the US, call or text 988), or talk to a trusted adult. If you're in immediate danger, call emergency services.",
  "emergency": "Those symptoms can be serious. Please get urgent medical care now or call your local emergency number. I can't assess this safely in chat.",
}
PROMPT_RULES = {
  "eating": "SAFETY: The user may have disordered-eating concerns. Do NOT give weight-loss, calorie, or diet targets. Respond with care, "
            "avoid praising weight loss, and encourage talking to a doctor, counsellor, or trusted adult.",
  "diagnosis": "SAFETY: The user asks for a diagnosis or medication advice. Do NOT diagnose or give doses. Explain you cannot diagnose, "
               "and suggest a check-up with a clinician.",
}
BASE_RULES = ("You are a supportive health-coaching assistant for young people. Never diagnose, never state the user has or does not have "
              "diabetes, never give medication or insulin advice. Use at most 3-4 sentences. Be non-judgmental.")

def check_user(msg):
    flags = []
    if CRISIS.search(msg): flags.append("crisis")
    if EMERGENCY.search(msg): flags.append("emergency")
    if EATING.search(msg): flags.append("eating")
    if DIAG_REQ.search(msg): flags.append("diagnosis")
    return flags

def check_assistant(text): return bool(DIAG_OUT.search(text))
