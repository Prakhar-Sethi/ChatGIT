"""
Hybrid Importance Scorer — Novelty 2

Replaces the static PageRank boost with a weighted combination of:

    hybrid(node, query) = α · PageRank(node)  +  (1-α) · QC-Attention(node, query)

where:
  PageRank(node)        — global structural centrality (query-agnostic)
  QC-Attention(node, q) — 1-hop graph attention conditioned on query embedding
  α                     — query-conditioned mixing weight

QC-Attention:
  - Each node is embedded by its short function name using the repo's BGE model.
  - At query time: self_sim = cosine(query_emb, node_emb)
  - 1-hop propagation: attention = 0.6·self_sim + 0.4·mean(neighbor_sim)
  - This makes importance dynamic: auth nodes score high for auth queries,
    DB nodes score high for database queries — without any training data.

Alpha selection (query-conditioned):
  - Broad/architectural queries → α ≈ 0.70  (trust global PageRank more)
  - Specific/locate queries     → α ≈ 0.25  (trust local attention more)
  - Default (mixed)             → α ≈ 0.50

Note: Node embeddings are computed once at repo-load time (batch embedding).
      Scoring at query time uses pre-built embeddings + a single query embed call.
"""

import numpy as np
from typing import Dict, Optional, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_BROAD_KW = [
    "architecture", "overview", "structure", "how does", "explain",
    "design", "pattern", "workflow", "pipeline", "system", "high level",
    "big picture", "overall", "entire", "whole", "module",
]
_SPECIFIC_KW = [
    "find", "locate", "where", "which file", "which line", "show me",
    "exact", "definition of", "declared", "defined", "implemented in",
    "what line", "where is", "find the",
]

def _query_alpha(query: str) -> float:
    """
    Returns α ∈ [0.25, 0.70]:
      broad query  → higher α (PageRank dominates)
      specific query → lower α (local attention dominates)
    """
    q = query.lower()
    broad   = sum(1 for kw in _BROAD_KW    if kw in q)
    specific = sum(1 for kw in _SPECIFIC_KW if kw in q)
    if broad > specific:
        return 0.70
    if specific > broad:
        return 0.25
    return 0.50


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HybridImportanceScorer:
    """
    Query-conditioned hybrid importance scorer for code nodes.

    Usage:
        scorer = HybridImportanceScorer(function_graph)
        scorer.build(pagerank_dict, embed_model)   # once at repo load
        scores = scorer.score_all(query, query_emb) # at each query
    """

    # Maximum number of nodes to embed (by PageRank rank) to bound startup time
    _MAX_NODES_TO_EMBED = 500

    def __init__(self, function_graph):
        """
        Args:
            function_graph : NetworkX DiGraph from CodePageRankAnalyzer
        """
        self.graph = function_graph
        self._node_embs: Dict[str, np.ndarray] = {}
        self._pagerank:  Dict[str, float]      = {}
        self._built = False

    # ------------------------------------------------------------------
    # Build phase (called once after repo load)
    # ------------------------------------------------------------------

    def build(self, pagerank_scores: Dict[str, float], embed_model) -> None:
        """
        Pre-compute node embeddings from short function names.
        Only embeds the top-N nodes by PageRank to keep startup fast.

        Args:
            pagerank_scores : {qualified_name: score} from get_function_pagerank()
            embed_model     : LlamaIndex / LangchainEmbedding with get_text_embedding_batch()
        """
        self._pagerank = pagerank_scores

        nodes = list(self.graph.nodes())
        if not nodes:
            return

        # Limit to top-N by PageRank (embed the most important ones first)
        nodes_ranked = sorted(nodes, key=lambda n: pagerank_scores.get(n, 0), reverse=True)
        nodes_to_embed = nodes_ranked[: self._MAX_NODES_TO_EMBED]

        short_names = [n.split("::")[-1] for n in nodes_to_embed]
        print(f"[HybridScorer] Embedding {len(short_names)} function names...")

        try:
            embs = embed_model.get_text_embedding_batch(short_names, show_progress=False)
            self._node_embs = {
                node: np.array(emb)
                for node, emb in zip(nodes_to_embed, embs)
            }
            self._built = True
            print("[HybridScorer] Node embeddings ready.")
        except Exception as exc:
            print(f"[HybridScorer] Batch embedding failed ({exc}). Falling back to PageRank only.")

    # ------------------------------------------------------------------
    # Scoring phase (called per query)
    # ------------------------------------------------------------------

    def score(self, node: str, query_emb: np.ndarray,
              alpha: Optional[float] = None) -> float:
        """
        Hybrid importance for a single node given a pre-computed query embedding.

        Returns:
            float ∈ [0, ∞)  — combines global PageRank + local graph attention
        """
        pr = self._pagerank.get(node, 0.0)

        if not self._built or node not in self._node_embs:
            return pr   # graceful fallback

        if alpha is None:
            alpha = 0.50

        # --- 1-hop graph attention ---
        self_sim = max(0.0, _cosine(query_emb, self._node_embs[node]))

        neighbors = (
            list(self.graph.predecessors(node))
            + list(self.graph.successors(node))
        )
        neighbor_sims = [
            max(0.0, _cosine(query_emb, self._node_embs[nb]))
            for nb in neighbors
            if nb in self._node_embs
        ]
        neighbor_score = float(np.mean(neighbor_sims)) if neighbor_sims else 0.0

        attention = 0.6 * self_sim + 0.4 * neighbor_score

        # --- Hybrid combination ---
        return alpha * pr + (1.0 - alpha) * attention

    def score_all(self, query: str, query_emb: np.ndarray) -> Dict[str, float]:
        """
        Score every node in the graph for a given query.
        Alpha is inferred automatically from query intent.

        Returns:
            {qualified_node_name: hybrid_score}
        """
        alpha = _query_alpha(query)
        return {
            node: self.score(node, query_emb, alpha)
            for node in self.graph.nodes()
        }
