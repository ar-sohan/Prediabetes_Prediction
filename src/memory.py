"""Hybrid longitudinal memory (SQLite). One store, five read-variants for the ablation:
M0 none | M1 raw history (truncated) | M2 summaries only | M3 retrieval only | M4 structured+timeline+summaries+retrieval (ours)."""
import sqlite3, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .llm import chat
from .config import CFG, ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_events(id INTEGER PRIMARY KEY, user_id TEXT, feature TEXT, value REAL,
  confidence REAL, evidence TEXT, session_idx INT, deleted INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS sessions(user_id TEXT, session_idx INT, transcript TEXT, summary TEXT, PRIMARY KEY(user_id, session_idx));
CREATE TABLE IF NOT EXISTS goals(user_id TEXT, session_idx INT, goal TEXT, active INT DEFAULT 1);
CREATE TABLE IF NOT EXISTS deletions(user_id TEXT, feature TEXT);
"""
SUM_SYS = ("Summarise this health-coaching chat for future memory in <=80 words: topics, facts the user stated (with numbers), goals set, "
           "preferences, mood. Facts only from the USER. {forbid}")

class Memory:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path); self.db.executescript(SCHEMA)
        self.model = CFG["llm"]["system_model"]

    # ---------- write ----------
    def _summarize(self, transcript, user_id):
        dels = [r[0] for r in self.db.execute("SELECT feature FROM deletions WHERE user_id=?", (user_id,))]
        forbid = f"The user asked to FORGET these topics; do NOT mention them: {dels}." if dels else ""
        return chat(SUM_SYS.format(forbid=forbid), transcript, model=self.model, max_tokens=250)

    def ingest(self, user_id, idx, transcript, extraction, goal=None):
        for f, d in extraction.items():
            if d["value"] is not None and not self._is_deleted(user_id, f):
                self.db.execute("INSERT INTO feature_events(user_id,feature,value,confidence,evidence,session_idx) VALUES(?,?,?,?,?,?)",
                                (user_id, f, d["value"], d["confidence"], d["evidence"], idx))
        self.db.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?)", (user_id, idx, transcript, self._summarize(transcript, user_id)))
        if goal: self.db.execute("INSERT INTO goals VALUES(?,?,?,1)", (user_id, idx, goal))
        self.db.commit()

    def _is_deleted(self, u, f):
        return self.db.execute("SELECT 1 FROM deletions WHERE user_id=? AND feature=?", (u, f)).fetchone() is not None

    def delete_fact(self, user_id, feature):
        """Remove from structured layer, and regenerate summaries so the fact is not kept in narrative memory."""
        self.db.execute("INSERT INTO deletions VALUES(?,?)", (user_id, feature))
        self.db.execute("UPDATE feature_events SET deleted=1 WHERE user_id=? AND feature=?", (user_id, feature))
        for idx, tr in self.db.execute("SELECT session_idx, transcript FROM sessions WHERE user_id=?", (user_id,)).fetchall():
            self.db.execute("UPDATE sessions SET summary=? WHERE user_id=? AND session_idx=?", (self._summarize(tr, user_id), user_id, idx))
        self.db.commit()

    # ---------- read ----------
    def profile(self, u):
        rows = self.db.execute("""SELECT feature,value,session_idx FROM feature_events WHERE user_id=? AND deleted=0
                                  ORDER BY session_idx""", (u,)).fetchall()
        cur, hist = {}, {}
        for f, v, i in rows: cur[f] = (v, i); hist.setdefault(f, []).append((i, v))   # newer wins, history kept
        return cur, hist

    def features(self, u): return {f: v for f, (v, _) in self.profile(u)[0].items()}

    def _summaries(self, u): return self.db.execute("SELECT session_idx,summary FROM sessions WHERE user_id=? ORDER BY session_idx", (u,)).fetchall()

    def _retrieve(self, u, query, k=3, field="summary"):
        rows = self.db.execute(f"SELECT session_idx,{field} FROM sessions WHERE user_id=?", (u,)).fetchall()
        if not rows or not query: return []
        v = TfidfVectorizer().fit([r[1] for r in rows] + [query])
        sims = cosine_similarity(v.transform([query]), v.transform([r[1] for r in rows]))[0]
        return [rows[i] for i in sims.argsort()[::-1][:k]]

    def brief(self, variant, u, query="", max_chars=3500):
        if variant == "M0": return ""
        if variant == "M1":
            raw = "\n\n".join(f"[Session {i}]\n{t}" for i, t in self.db.execute(
                "SELECT session_idx,transcript FROM sessions WHERE user_id=? ORDER BY session_idx", (u,)))
            return "Past conversations (most recent kept):\n" + raw[-max_chars:]
        if variant == "M2":
            return "Past session summaries:\n" + "\n".join(f"[S{i}] {s}" for i, s in self._summaries(u))[-max_chars:]
        if variant == "M3":
            return "Retrieved memories:\n" + "\n".join(f"[S{i}] {s}" for i, s in self._retrieve(u, query, 3, "transcript"))[:max_chars]
        if variant == "M4":
            cur, hist = self.profile(u)
            prof = "\n".join(f"- {f}: {v:g} (as of session {i}" + (f"; earlier: {[x for x in hist[f][:-1]]}" if len(hist[f]) > 1 else "") + ")"
                             for f, (v, i) in cur.items())
            goals = "\n".join(f"- {g} (set in session {i})" for i, g in self.db.execute(
                "SELECT session_idx,goal FROM goals WHERE user_id=? AND active=1", (u,)))
            recent = "\n".join(f"[S{i}] {s}" for i, s in self._summaries(u)[-2:])
            rel = "\n".join(f"[S{i}] {s}" for i, s in self._retrieve(u, query, 3))
            return (f"KNOWN FACTS (latest value wins; do not re-ask):\n{prof}\nACTIVE GOALS:\n{goals}\n"
                    f"RECENT SESSIONS:\n{recent}\nRELEVANT MEMORIES:\n{rel}")[:max_chars]
        raise ValueError(variant)
