"""
Git-History Volatility Analyzer — Novelty 1

Computes per-file volatility profiles from git commit history:
  - change_frequency : number of commits touching each file (normalized)
  - recency_score    : exponential decay of days since last change
  - author_diversity : number of unique contributors (capped at 5)
  - co_change_map    : files that frequently co-change (logical coupling)

These signals feed a retrieval weight that complements PageRank:
  - Stable code  → boosted for fact/locate queries (reliable, well-established)
  - Recent code  → boosted for "what changed / latest" queries
  - Co-changed files are soft-linked for retrieval expansion
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class GitVolatilityAnalyzer:
    """Extracts code-volatility signals from git history for retrieval weighting."""

    # Weight parameters for the composite volatility score
    _W_FREQ    = 0.50
    _W_RECENCY = 0.30
    _W_AUTHORS = 0.20
    _RECENCY_HALFLIFE_DAYS = 90   # exponential decay half-life

    def __init__(self):
        self.file_change_count: Dict[str, int]      = {}
        self.file_last_change:  Dict[str, datetime] = {}
        self.file_authors:      Dict[str, set]       = {}
        self.co_change_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._analyzed = False
        self._max_changes = 1
        self._now = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, repo_path: str) -> bool:
        """
        Walk git log (up to 500 commits) and populate all volatility maps.
        Returns True on success, False if repo has no git history.
        """
        try:
            from git import Repo
            repo = Repo(repo_path)
        except Exception as exc:
            print(f"[GitAnalyzer] Cannot open repo: {exc}")
            return False

        try:
            commits = list(repo.iter_commits("HEAD", max_count=500))
        except Exception as exc:
            print(f"[GitAnalyzer] Cannot read commits: {exc}")
            return False

        if not commits:
            print("[GitAnalyzer] No commits found — skipping volatility analysis.")
            return False

        print(f"[GitAnalyzer] Analyzing {len(commits)} commits...")
        for commit in commits:
            self._process_commit(commit)

        if self.file_change_count:
            self._max_changes = max(self.file_change_count.values())

        self._analyzed = True
        print(f"[GitAnalyzer] Tracked {len(self.file_change_count)} files.")
        return True

    def get_volatility_score(self, file_path: str) -> float:
        """
        Composite volatility ∈ [0, 1].
        High → frequently changed, recent, many authors.
        """
        if not self._analyzed:
            return 0.0

        freq   = self.file_change_count.get(file_path, 0) / self._max_changes
        recency = self._recency(file_path)
        diversity = min(len(self.file_authors.get(file_path, set())) / 5.0, 1.0)

        return (self._W_FREQ * freq
                + self._W_RECENCY * recency
                + self._W_AUTHORS * diversity)

    def get_stability_score(self, file_path: str) -> float:
        """Inverse of volatility — high = stable, well-established code."""
        return 1.0 - self.get_volatility_score(file_path)

    def get_retrieval_weight(self, file_path: str, recency_focused: bool = False) -> float:
        """
        Returns a multiplicative retrieval weight ≥ 1.0.

        recency_focused=True  → boost recently changed files  (user asked "what changed")
        recency_focused=False → boost stable files            (fact / locate queries)
        """
        if not self._analyzed:
            return 1.0

        if not self.file_change_count:
            # No history data — return neutral weight
            return 1.0

        if recency_focused:
            return 1.0 + self._recency(file_path)
        else:
            # Volatile files (many authors, recent changes) may contain active bugs/features
            # Stable files get a small boost for reliability on fact/locate queries
            v = self.get_volatility_score(file_path)
            return 1.0 + 0.4 * (1.0 - v)

    def get_co_changed_files(self, file_path: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """Files that most frequently co-change with *file_path* (logical coupling)."""
        if not self._analyzed or file_path not in self.co_change_map:
            return []
        coupled = sorted(
            self.co_change_map[file_path].items(),
            key=lambda x: x[1], reverse=True
        )
        return coupled[:top_n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_commit(self, commit):
        try:
            committed_dt = datetime.fromtimestamp(
                commit.committed_date, tz=timezone.utc
            )
            changed_files = list(commit.stats.files.keys())
            author_email  = (commit.author.email or "") if commit.author else ""

            for fpath in changed_files:
                self.file_change_count[fpath] = self.file_change_count.get(fpath, 0) + 1
                if (fpath not in self.file_last_change
                        or committed_dt > self.file_last_change[fpath]):
                    self.file_last_change[fpath] = committed_dt
                self.file_authors.setdefault(fpath, set())
                if author_email:
                    self.file_authors[fpath].add(author_email)

            # Co-change: O(n²) within a commit — fine for ≤ 500 commits
            for i, f1 in enumerate(changed_files):
                for f2 in changed_files[i + 1:]:
                    self.co_change_map[f1][f2] += 1
                    self.co_change_map[f2][f1] += 1
        except Exception:
            pass  # never crash on a single commit

    def _recency(self, file_path: str) -> float:
        """Exponential-decay recency score ∈ [0, 1]."""
        if file_path not in self.file_last_change:
            return 0.0
        days_old = (self._now - self.file_last_change[file_path]).days
        return math.exp(-days_old / self._RECENCY_HALFLIFE_DAYS)
