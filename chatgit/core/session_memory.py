"""
Multi-Turn Session-Aware Retrieval Memory — Novelty 3

Maintains per-session retrieval state across conversation turns to enable:

1. Redundancy suppression
   Chunks retrieved recently get a score penalty — avoids showing the same
   code block across multiple turns.

2. Session-zone coherence bonus
   Files active in recent turns receive a mild retrieval boost so follow-up
   questions naturally continue exploring the same topic area.

3. Discussed-function bonus
   Functions explicitly mentioned in prior responses get a boost when they
   appear in candidates for a new query, supporting drill-down conversations.

4. Code co-reference resolution
   Short queries with pronouns ("what does it return?", "the other one") are
   expanded with the last-discussed function / active file as a context hint,
   improving retrieval accuracy on follow-up turns.
"""

import re
from collections import defaultdict
from typing import Dict, List


class SessionRetrievalMemory:
    """
    Retrieval memory for a single chat session.
    Reset by calling .reset() when a new repository is loaded.
    """

    REDUNDANCY_PENALTY_SAME_TURN   = 0.25   # chunk retrieved this turn
    REDUNDANCY_PENALTY_LAST_TURN   = 0.55   # chunk retrieved 1 turn ago
    REDUNDANCY_PENALTY_OLDER       = 0.82   # 2+ turns ago
    SESSION_ZONE_BONUS             = 0.05   # gentle file-level coherence bonus
    DISCUSSED_FUNC_BONUS           = 1.20   # multiplier for discussed functions
    RECENCY_DECAY                  = 0.80   # per-turn decay on active-file weights

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        self.turn: int = 0
        # chunk_id -> list of turn numbers it was retrieved
        self._retrieved: Dict[str, List[int]] = defaultdict(list)
        # file -> current zone weight (decays each turn)
        self._active_files: Dict[str, float] = {}
        # functions discussed across the session (most recent last)
        self._discussed_fns: List[str] = []
        # full query history: [(turn, query_text)] — used for temporal back-refs
        self._query_history: List[tuple] = []

    # ------------------------------------------------------------------
    # Record a completed turn
    # ------------------------------------------------------------------

    def record_query(self, query: str):
        """Call at the START of a turn (before retrieval) to log the raw query."""
        self._query_history.append((self.turn + 1, query.strip()))
        # Keep bounded
        if len(self._query_history) > 50:
            self._query_history = self._query_history[-50:]

    def record_turn(self, query: str, retrieved_chunks: List[dict], response: str):
        """
        Call after the LLM response is generated to update memory.

        retrieved_chunks: list of dicts with keys:
            'file'         : str  (file_name metadata)
            'node_name'    : str  (function/class name)
            'matched_funcs': list[str]
        """
        self.turn += 1

        # Decay existing active files
        decayed = {f: w * self.RECENCY_DECAY for f, w in self._active_files.items()}
        self._active_files = {f: w for f, w in decayed.items() if w > 0.05}

        for chunk in retrieved_chunks:
            fname     = chunk.get("file", "")
            node_name = chunk.get("node_name", "")
            chunk_id  = f"{fname}::{node_name}"

            self._retrieved[chunk_id].append(self.turn)
            # Fresh weight for this file
            self._active_files[fname] = max(self._active_files.get(fname, 0.0), 1.0)

            # Any retrieved node is now "discussed" — enables same-referent
            # continuation exemption on follow-up EXPLAIN/DEBUG turns
            if node_name and node_name not in self._discussed_fns:
                self._discussed_fns.append(node_name)
            for fn in chunk.get("matched_funcs", []):
                if fn and fn not in self._discussed_fns:
                    self._discussed_fns.append(fn)

        # Keep discussed functions list bounded
        self._discussed_fns = self._discussed_fns[-20:]

    # ------------------------------------------------------------------
    # Apply session scores to candidates
    # ------------------------------------------------------------------

    def apply_session_scores(self, candidates: List[dict],
                             intent: str = "") -> List[dict]:
        """
        Adjust candidate scores in-place based on session memory.
        Should be called AFTER the initial vector + hybrid scoring,
        BEFORE cross-encoder reranking.

        intent: current query intent string (e.g. "summarize", "explain").
                When intent=="summarize", a softer penalty is applied so that
                module-level architecture summaries remain retrievable across
                turns.
        """
        if self.turn == 0:
            return candidates  # no history yet

        for c in candidates:
            meta      = c["snippet"].metadata
            fname     = meta.get("file_name", "")
            node_name = meta.get("node_name", "")
            node_type = meta.get("node_type", "")
            chunk_id  = f"{fname}::{node_name}"

            # --- Redundancy penalty ---
            # Exemptions (no penalty applied):
            # 1. module_summary chunks: always exempt
            # 2. class chunks for SUMMARIZE: "overview of X" legitimately
            #    re-retrieves the same class definition across turns
            # 3. Same-referent continuation: if intent is explain/debug and
            #    this chunk's function was discussed in a previous turn, the
            #    user is asking a follow-up about the SAME function — the chunk
            #    is the correct answer and must not be suppressed.
            is_class_for_summarize = (node_type == "class"
                                      and intent == "summarize")
            is_same_referent_continuation = (
                intent in ("explain", "debug")
                and node_name in self._discussed_fns
            )
            if (chunk_id in self._retrieved
                    and node_type != "module_summary"
                    and not is_class_for_summarize
                    and not is_same_referent_continuation):
                last_seen  = max(self._retrieved[chunk_id])
                turns_ago  = self.turn - last_seen
                if intent == "summarize":
                    c["score"] *= 0.92
                elif turns_ago == 0:
                    c["score"] *= self.REDUNDANCY_PENALTY_SAME_TURN
                elif turns_ago == 1:
                    c["score"] *= self.REDUNDANCY_PENALTY_LAST_TURN
                else:
                    c["score"] *= self.REDUNDANCY_PENALTY_OLDER

            # --- Session-zone bonus ---
            if fname in self._active_files:
                c["score"] *= (1.0 + self.SESSION_ZONE_BONUS * self._active_files[fname])

            # --- Discussed-function bonus ---
            for fn in c.get("matched_funcs", []):
                if fn in self._discussed_fns:
                    c["score"] *= self.DISCUSSED_FUNC_BONUS
                    break

        return candidates

    # ------------------------------------------------------------------
    # Co-reference resolution
    # ------------------------------------------------------------------

    # Simple pronoun patterns → expand with last discussed function
    _PRONOUN_PATTERNS = [
        r"\bit\b",
        r"\bthis function\b",
        r"\bthe other one\b",
        r"\bthe same\b",
        r"\bthat function\b",
        r"\bthis class\b",
        r"\bthe above\b",
    ]

    # Temporal back-reference patterns → expand with FIRST or EARLY query
    _TEMPORAL_FIRST_PATTERNS = [
        r"\bin the beginning\b",
        r"\bat the (very )?start\b",
        r"\bmy first question\b",
        r"\bthe first (query|question|thing)\b",
        r"\binitially\b",
        r"\boriginally\b",
        r"\bfirst asked\b",
        r"\basked (at|in) the beginning\b",
        r"\bthe function i asked\b",
    ]

    # Repeat / "again" patterns → expand with previous query
    _REPEAT_PATTERNS = [
        r"^again[?!.]?\s*$",
        r"^repeat\s*$",
        r"\bsame (question|query)\b",
        r"\bask again\b",
        r"\bone more time\b",
        r"^more[?!.]?\s*$",
    ]

    def resolve_coreferences(self, query: str) -> str:
        """
        Expand queries with session context based on three resolution strategies:

        1. Temporal back-references ("the function I asked in the beginning")
           → resolved to the FIRST query in the session history
        2. Repeat queries ("again?", "more?")
           → resolved to the PREVIOUS query
        3. Pronoun references ("it", "this function")
           → resolved to the last discussed function
        """
        if not self._discussed_fns and not self._active_files and not self._query_history:
            return query

        q = query.strip()
        q_lower = q.lower()

        # ── Strategy 1: Temporal back-references ──────────────────────────
        for pattern in self._TEMPORAL_FIRST_PATTERNS:
            if re.search(pattern, q_lower):
                if self._query_history:
                    first_query = self._query_history[0][1]
                    q = f"{q} [referring to earlier question: \"{first_query}\"]"
                    # Also pin the first discussed function if available
                    if self._discussed_fns:
                        q += f" [first function discussed: '{self._discussed_fns[0]}']"
                return q

        # ── Strategy 2: Repeat / "again" queries ──────────────────────────
        for pattern in self._REPEAT_PATTERNS:
            if re.search(pattern, q_lower):
                if len(self._query_history) >= 2:
                    prev_query = self._query_history[-2][1]  # the turn before this one
                    q = f"{prev_query} [follow-up: {q}]"
                elif self._query_history:
                    q = self._query_history[-1][1]
                return q

        # ── Strategy 3: Pronoun resolution (short queries only) ───────────
        if len(q.split()) < 10:
            for pattern in self._PRONOUN_PATTERNS:
                if re.search(pattern, q_lower):
                    if self._discussed_fns:
                        last_fn = self._discussed_fns[-1]
                        q = f"{q} [context: function '{last_fn}']"
                    break

        # ── Fallback: bare short query → hint with most active file ───────
        if len(q.split()) < 5 and self._active_files:
            top_file = max(self._active_files, key=self._active_files.get)
            q = f"{q} [file context: {top_file}]"

        return q

    # ------------------------------------------------------------------
    # Session summary for the prompt
    # ------------------------------------------------------------------

    def get_session_summary(self) -> str:
        """
        One-paragraph summary of session state to include in the LLM prompt.
        Helps the model give coherent follow-up answers.
        """
        if self.turn == 0:
            return ""

        lines = ["\n## Session Context (Multi-Turn Memory)"]
        if self._query_history:
            lines.append(f"- First question asked: \"{self._query_history[0][1]}\"")
            if len(self._query_history) > 1:
                lines.append(f"- Previous question: \"{self._query_history[-2][1]}\"")
        if self._discussed_fns:
            fns = ", ".join(f"`{f}`" for f in self._discussed_fns[-5:])
            lines.append(f"- Recently discussed functions: {fns}")
        if self._active_files:
            top = sorted(self._active_files.items(), key=lambda x: x[1], reverse=True)[:3]
            files = ", ".join(f"`{f}`" for f, _ in top)
            lines.append(f"- Active topic files: {files}")
        lines.append(f"- Conversation turn: {self.turn}")
        return "\n".join(lines)
