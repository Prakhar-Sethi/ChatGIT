"""
Full evaluation: ChatGIT vs all baselines.
Uses 150 real multi-turn conversations (30 per repo × 5 repos).

Systems evaluated:
  1. BM25                      — lexical baseline
  2. BM25-SlidingWindow        — BM25 with one round of query augmentation
                                 (NOT RepoCoder; see baselines.py for distinction)
  3. ConvAwareRAG              — VanillaRAG + previous query appended; isolates N3
  4. VanillaRAG (BGE)          — dense, no graph signals, no session state
  5. ChatGIT (Session+Intent)  — N3+N4 only, no graph components
  6. ChatGIT (Full)            — all five novelties (N1-N5)

Metrics: MRR, P@1, Recall@5, NDCG@5, cross-turn redundancy
"""

import sys, os, json, time
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Patch tiktoken
import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4

import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from chatgit.core.chunker import chunk_repository
from chatgit.core.intent_classifier import classify_intent
from chatgit.core.session_memory import SessionRetrievalMemory
from chatgit.core.git_analyzer import GitVolatilityAnalyzer
from chatgit.core.graph.pagerank import CodePageRankAnalyzer
from chatgit.core.graph.hybrid_importance import HybridImportanceScorer
from evaluation.baselines import BM25, BM25SlidingWindow
from evaluation.eval_retrieval import evaluate_retrieval, print_retrieval_report
from evaluation.statistical_tests import full_comparison_report, print_comparison_table

_REPO_BASE = os.environ.get("CHATGIT_REPO_BASE", "/tmp")
REPOS = {
    "flask":    os.environ.get("CHATGIT_REPO_FLASK",    os.path.join(_REPO_BASE, "flask_bench")),
    "requests": os.environ.get("CHATGIT_REPO_REQUESTS", os.path.join(_REPO_BASE, "requests_bench")),
    "click":    os.environ.get("CHATGIT_REPO_CLICK",    os.path.join(_REPO_BASE, "click_bench")),
    "fastapi":  os.environ.get("CHATGIT_REPO_FASTAPI",  os.path.join(_REPO_BASE, "fastapi_bench")),
    "celery":   os.environ.get("CHATGIT_REPO_CELERY",   os.path.join(_REPO_BASE, "celery_bench")),
}
REPOS = {k: v for k, v in REPOS.items() if os.path.isdir(v) or
         print(f"  [SKIP] {k}: path not found ({v})", file=sys.stderr) or False}
CONVERSATIONS_PATH = os.environ.get(
    "CHATGIT_CONVS_PATH",
    os.path.join(_project_root, "data", "convcodebench", "eval_conversations.jsonl")
)
SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "example",
             "__pycache__", ".git", "build", "dist"}


# ── Chunk all repos ──────────────────────────────────────────────────────────

def chunk_all_repos():
    all_chunks = {}
    for repo_id, path in REPOS.items():
        print(f"  Chunking {repo_id}...", end=" ", flush=True)
        t0 = time.time()
        docs = chunk_repository(path)
        chunks, seen = [], set()
        for d in docs:
            fname = d.metadata["file_name"]
            parts = set(fname.replace("\\", "/").split("/"))
            if parts & SKIP_DIRS:
                continue
            cid  = f"{fname}::{d.metadata['node_name']}"
            text = d.text if hasattr(d, "text") else d.page_content
            obj  = {"id": cid, "text": text[:600],
                    "file":      fname,
                    "node_type": d.metadata.get("node_type", ""),
                    "node_name": d.metadata.get("node_name", "")}
            chunks.append(obj)
            if cid not in seen:
                seen.add(cid)
        all_chunks[repo_id] = {"list": chunks,
                                "by_id": {c["id"]: c for c in chunks}}
        print(f"{len(chunks)} chunks  ({time.time()-t0:.1f}s)")
    return all_chunks


# ── GT fuzzy matching ────────────────────────────────────────────────────────

def fuzzy_match_gt(gt_id: str, by_id: dict) -> list:
    if gt_id in by_id:
        return [gt_id]
    # Partial path + name match
    parts = gt_id.split("::")
    if len(parts) != 2:
        return []
    file_part, name_part = parts
    bare = name_part.split(".")[-1]
    hits = []
    for cid, c in by_id.items():
        cfile = c["file"]
        cname = c["node_name"]
        file_ok = (not file_part or
                   file_part in cfile or cfile in file_part or
                   cfile.split("/")[-1] == file_part.split("/")[-1])
        name_ok = (cname == bare or cname == name_part)
        if file_ok and name_ok:
            hits.append(cid)
    if not hits and bare and len(bare) > 3:
        for cid, c in by_id.items():
            if c["node_name"] == bare:
                hits.append(cid)
    return hits[:3]


# ── Load conversations ────────────────────────────────────────────────────────

def load_conversations(all_chunks):
    convs_raw = []
    with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs_raw.append(json.loads(line))

    queries, sessions = [], []
    gt_hit = gt_miss = 0

    for conv in convs_raw:
        rid = conv["repo_id"]
        if rid not in all_chunks:
            continue
        by_id = all_chunks[rid]["by_id"]
        sess_turns = []
        for turn in conv["turns"]:
            matched_gt = []
            for g in turn.get("ground_truth_chunks", []):
                m = fuzzy_match_gt(g, by_id)
                if m:
                    matched_gt.extend(m); gt_hit += 1
                else:
                    gt_miss += 1
            # dedup
            seen_set = set()
            matched_gt = [x for x in matched_gt
                          if not (x in seen_set or seen_set.add(x))]

            qid = f"{conv['conversation_id']}_t{turn['turn_id']}"
            row = (qid, turn["query"], matched_gt,
                   turn.get("intent", "unknown"), rid,
                   conv["conversation_id"])
            queries.append(row)
            sess_turns.append({"qid": qid,
                                "query": turn["query"],
                                "gt":    matched_gt,
                                "intent": turn.get("intent", "unknown")})
        sessions.append({"conv_id": conv["conversation_id"],
                         "repo_id": rid,
                         "turns":   sess_turns})

    total = gt_hit + gt_miss
    print(f"  GT mapping: {gt_hit}/{total} ({100*gt_hit/max(total,1):.0f}% matched)")
    return queries, sessions


# ── Build index ──────────────────────────────────────────────────────────────

def build_index(all_chunks, embed_model):
    index = {}
    for rid, data in all_chunks.items():
        chunks = data["list"]
        repo_path = REPOS[rid]
        print(f"  Indexing {rid} ({len(chunks)} chunks)...", end=" ", flush=True)
        t0 = time.time()

        texts = [c["text"] for c in chunks]
        embs  = embed_model.encode(texts, batch_size=64,
                                   show_progress_bar=False,
                                   normalize_embeddings=True).astype(np.float32)

        bm25 = BM25().fit(chunks)

        # BM25-SlidingWindow: BM25 with one round of query augmentation
        # (NOTE: this is NOT RepoCoder from Zhang et al. 2023 — see baselines.py)
        bm25sw_bm25 = BM25SlidingWindow(base_retriever=BM25()).fit(chunks)

        # BM25-SlidingWindow: BGE base (iterative dense retrieval)
        class _DenseRetriever:
            def __init__(self, _embs, _chunks, _model):
                self._e = _embs; self._c = _chunks; self._m = _model
            def fit(self, _): return self
            def retrieve_ids(self, query, k=10):
                qe = self._m.encode([query], normalize_embeddings=True)[0].astype(np.float32)
                s  = self._e @ qe
                return [self._c[i]["id"] for i in np.argsort(-s)[:k]]
        dense_ret = _DenseRetriever(embs, chunks, embed_model)
        bm25sw_bge = BM25SlidingWindow(base_retriever=dense_ret)
        bm25sw_bge._chunks_by_id = {c["id"]: c["text"] for c in chunks}

        # Git volatility
        git_az = GitVolatilityAnalyzer()
        git_az.analyze(repo_path)

        # PageRank + hybrid scorer — skip for large repos to keep eval fast
        pagerank = None; hybrid_scorer = HybridImportanceScorer(None)
        if len(chunks) <= 2000:
            try:
                pr = CodePageRankAnalyzer()
                pr.analyze_repository(repo_path)
                pr_dict = dict(pr.get_function_pagerank())
                hs = HybridImportanceScorer(pr.function_graph)
                top_nodes = sorted(pr_dict, key=pr_dict.get, reverse=True)[:300]
                short_names = [n.split("::")[-1] for n in top_nodes]
                ne = embed_model.encode(short_names, batch_size=64,
                                        show_progress_bar=False,
                                        normalize_embeddings=True)
                hs._pagerank = pr_dict
                hs._node_embs = {n: np.array(e) for n, e in zip(top_nodes, ne)}
                hs._built = True
                pagerank = pr; hybrid_scorer = hs
            except Exception as ex:
                print(f"[PR warn: {ex}]", end=" ")
        else:
            print(f"[PR skipped: {len(chunks)} chunks]", end=" ")

        index[rid] = {
            "chunks": chunks, "embs": embs,
            "bm25": bm25,
            "bm25sw_bm25": bm25sw_bm25,
            "bm25sw_bge":  bm25sw_bge,
            "git_az": git_az,
            "hybrid_scorer": hybrid_scorer,
            "pagerank": pagerank,
        }
        print(f"{time.time()-t0:.1f}s")
    return index


# ── Retrieval runners ────────────────────────────────────────────────────────

def run_bm25(index, queries, k=10):
    preds = []
    for qid, query, gt, intent, rid, _ in queries:
        if rid not in index or not gt: continue
        retrieved = index[rid]["bm25"].retrieve_ids(query, k)
        preds.append({"query_id": qid, "retrieved": retrieved,
                      "ground_truth": gt, "intent": intent})
    return preds


def run_repocoder_bm25(index, queries, k=10):
    """BM25-SlidingWindow (BM25 base): NOT RepoCoder — see baselines.py for distinction."""
    preds = []
    for qid, query, gt, intent, rid, _ in queries:
        if rid not in index or not gt: continue
        retrieved = index[rid]["bm25sw_bm25"].retrieve_ids(query, k)
        preds.append({"query_id": qid, "retrieved": retrieved,
                      "ground_truth": gt, "intent": intent})
    return preds


def run_repocoder_bge(index, queries, embed_model, k=10):
    """BM25-SlidingWindow (BGE base): dense round-1, augment query with top snippet, dense round-2.
    NOT RepoCoder — see baselines.py for distinction."""
    preds = []
    for qid, query, gt, intent, rid, _ in queries:
        if rid not in index or not gt: continue
        ix = index[rid]
        chunks, embs = ix["chunks"], ix["embs"]

        # Round 1: dense retrieval
        q1 = embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims1 = embs @ q1
        top1_ids = [chunks[i]["id"] for i in np.argsort(-sims1)[:5]]

        # Augment: append top-1 chunk text to query
        top1_text = next((c["text"][:250] for c in chunks
                          if c["id"] == top1_ids[0]), "")
        aug_query  = query + " " + top1_text

        # Round 2: dense retrieval with augmented query
        q2 = embed_model.encode([aug_query], normalize_embeddings=True)[0].astype(np.float32)
        sims2 = embs @ q2
        top2_idx = np.argsort(-sims2)[:k]

        # Merge: round2 preferred, fill from round1
        seen = set()
        merged = []
        for i in top2_idx:
            cid = chunks[i]["id"]
            if cid not in seen:
                seen.add(cid); merged.append(cid)
        for cid in top1_ids:
            if cid not in seen and len(merged) < k:
                seen.add(cid); merged.append(cid)

        preds.append({"query_id": qid, "retrieved": merged[:k],
                      "ground_truth": gt, "intent": intent})
    return preds


def run_vanilla_rag(index, queries, embed_model, k=10):
    preds = []
    for qid, query, gt, intent, rid, _ in queries:
        if rid not in index or not gt: continue
        ix = index[rid]
        qe = embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims = ix["embs"] @ qe
        top  = [ix["chunks"][i]["id"] for i in np.argsort(-sims)[:k]]
        preds.append({"query_id": qid, "retrieved": top,
                      "ground_truth": gt, "intent": intent})
    return preds


def run_chatgit_n3n4(index, queries, embed_model, sessions, k=10):
    """Session memory (N3) + intent-adaptive retrieval (N4). No graph signals."""
    by_conv = defaultdict(list)
    for row in queries: by_conv[row[5]].append(row)

    preds = []
    for conv_id, conv_rows in by_conv.items():
        rid = conv_rows[0][4]
        if rid not in index: continue
        ix = index[rid]
        chunks, embs = ix["chunks"], ix["embs"]
        mem = SessionRetrievalMemory()

        for qid, query, gt, intent, _, _ in conv_rows:
            if not gt: continue
            cfg      = classify_intent(query)
            resolved = mem.resolve_coreferences(query)
            qe = embed_model.encode([resolved], normalize_embeddings=True)[0].astype(np.float32)
            sims = (embs @ qe).copy()

            # N4: granularity boost
            for i, c in enumerate(chunks):
                nt = c["node_type"]
                if nt == "module_summary" and cfg.granularity == "module":
                    sims[i] *= cfg.granularity_boost
                elif nt == "function" and cfg.granularity == "function":
                    sims[i] *= 1.15

            top_idx = np.argsort(-sims)[:cfg.top_k]

            # N3: session scoring
            scored = []
            for i in top_idx:
                rid2  = chunks[i]["id"]
                score = float(sims[i])
                fname = chunks[i]["file"]
                if rid2 in mem._retrieved:
                    score *= mem.REDUNDANCY_PENALTY_LAST_TURN
                if fname in mem._active_files:
                    score *= (1.0 + mem.SESSION_ZONE_BONUS
                              * mem._active_files[fname])
                scored.append((rid2, score))
            scored.sort(key=lambda x: -x[1])
            retrieved = [r for r, _ in scored[:k]]

            preds.append({"query_id": qid, "retrieved": retrieved,
                          "ground_truth": gt, "intent": intent})
            mem.record_turn(query,
                [{"file": r.split("::")[0] if "::" in r else r,
                  "node_name": r.split("::")[-1] if "::" in r else r,
                  "matched_funcs": []} for r in retrieved],
                f"[{intent}]")
    return preds


def run_chatgit_full(index, queries, embed_model, sessions, k=10):
    """All components: N3+N4+N1+N2+N5."""
    by_conv = defaultdict(list)
    for row in queries: by_conv[row[5]].append(row)

    preds = []
    for conv_id, conv_rows in by_conv.items():
        rid = conv_rows[0][4]
        if rid not in index: continue
        ix      = index[rid]
        chunks  = ix["chunks"]
        embs    = ix["embs"]
        git_az  = ix["git_az"]
        hyb_sc  = ix["hybrid_scorer"]
        pr      = ix["pagerank"]
        mem     = SessionRetrievalMemory()

        for qid, query, gt, intent, _, _ in conv_rows:
            if not gt: continue
            cfg      = classify_intent(query)
            resolved = mem.resolve_coreferences(query)

            qe   = embed_model.encode([resolved], normalize_embeddings=True)[0].astype(np.float32)
            sims = (embs @ qe).copy()

            # N4: granularity boost
            for i, c in enumerate(chunks):
                nt = c["node_type"]
                if nt == "module_summary" and cfg.granularity == "module":
                    sims[i] *= cfg.granularity_boost
                elif nt == "function" and cfg.granularity == "function":
                    sims[i] *= 1.15
                elif nt == "class" and intent == "explain":
                    sims[i] *= 1.10

            # N2: hybrid scores
            hybrid_scores = {}
            if hyb_sc._built and hyb_sc.graph is not None:
                try:
                    hybrid_scores = hyb_sc.score_all(resolved, qe)
                except Exception:
                    pass

            # N1: recency focus
            recency = any(kw in resolved.lower() for kw in
                          ["recent","changed","latest","updated","new","modified"])

            # N5: call-neighbourhood boost
            nb_boost = {}
            if pr is not None:
                seed_idx = np.argsort(-sims)[:5]
                for i in seed_idx:
                    psim = float(sims[i])
                    if psim < 0.3: continue
                    c     = chunks[i]
                    qname = f"{c['file']}::{c['node_name']}"
                    if qname not in pr.function_graph: continue
                    nbrs = (list(pr.function_graph.successors(qname))[:2]
                           + list(pr.function_graph.predecessors(qname))[:2])
                    for nb in nbrs:
                        nb_short = nb.split("::")[-1]
                        for j, nc in enumerate(chunks):
                            if (nc["id"] == nb
                                    or nc["id"].endswith(f"::{nb_short}")):
                                nb_boost[nc["id"]] = (
                                    nb_boost.get(nc["id"], 0) + 0.06 * psim)

            pool = min(cfg.top_k * 2, len(chunks))
            top_idx = np.argsort(-sims)[:pool]

            scored = []
            for i in top_idx:
                cid   = chunks[i]["id"]
                fname = chunks[i]["file"]
                score = float(sims[i])

                # N1
                score *= git_az.get_retrieval_weight(fname, recency)

                # N2
                h = hybrid_scores.get(
                    f"{fname}::{chunks[i]['node_name']}", 0.0)
                if h > 0.3:
                    score *= (1.0 + 0.05 * h)

                # N5
                score += nb_boost.get(cid, 0.0)

                # N3
                if cid in mem._retrieved:
                    score *= mem.REDUNDANCY_PENALTY_LAST_TURN
                if fname in mem._active_files:
                    score *= (1.0 + mem.SESSION_ZONE_BONUS
                              * mem._active_files[fname])

                scored.append((cid, score))

            scored.sort(key=lambda x: -x[1])
            retrieved = [r for r, _ in scored[:k]]

            preds.append({"query_id": qid, "retrieved": retrieved,
                          "ground_truth": gt, "intent": intent})
            mem.record_turn(query,
                [{"file": r.split("::")[0] if "::" in r else r,
                  "node_name": r.split("::")[-1] if "::" in r else r,
                  "matched_funcs": []} for r in retrieved],
                f"[{intent}]")
    return preds


# ── Redundancy rate ──────────────────────────────────────────────────────────

def redundancy_rate(preds, sessions):
    pm = {p["query_id"]: p["retrieved"] for p in preds}
    total = redundant = 0
    for sess in sessions:
        seen = set()
        for turn in sess["turns"]:
            retrieved = pm.get(turn["qid"], [])
            for cid in retrieved:
                total += 1
                if cid in seen: redundant += 1
            seen.update(retrieved)
    return redundant / total if total else 0.0


# ── Print results table ──────────────────────────────────────────────────────

def print_table(results, metrics, extra=None):
    col_w = 14
    header = f"  {'System':<30}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print("  " + "-" * (30 + col_w * len(metrics)))
    for name, res in results.items():
        row = f"  {name:<30}"
        for m in metrics:
            s = res["summary"].get(m, {})
            v = s.get("mean", 0.0)
            ci = (s.get("ci_hi", v) - s.get("ci_lo", v)) / 2
            row += f"  {v:.4f}±{ci:.4f}"
        if extra and name in extra:
            row += f"  {extra[name]}"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  ChatGIT Evaluation — Multi-Turn Code QA")
    print(f"  Repos: {list(REPOS.keys())}")
    print("=" * 72)

    print("\n[1/4] Chunking repositories...")
    all_chunks = chunk_all_repos()

    print("\n[2/4] Loading evaluation conversations...")
    queries, sessions = load_conversations(all_chunks)
    queries_gt = [q for q in queries if q[2]]
    print(f"  Total turns: {len(queries)} | With GT: {len(queries_gt)} "
          f"| Conversations: {len(sessions)}")

    from collections import Counter
    ic = Counter(q[3] for q in queries_gt)
    print(f"  Intent dist: {dict(sorted(ic.items()))}")

    print("\n[3/4] Building embeddings and indices...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5",
                                      cache_folder=os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
                                      device="cpu")
    index = build_index(all_chunks, embed_model)

    print("\n[4/4] Running retrieval systems...")
    k = 10
    t0 = time.time()

    bm25_p         = run_bm25(index, queries_gt, k)
    bm25sw_bm25_p  = run_repocoder_bm25(index, queries_gt, k)
    bm25sw_bge_p   = run_repocoder_bge(index, queries_gt, embed_model, k)
    vanilla_p      = run_vanilla_rag(index, queries_gt, embed_model, k)
    cg_n3n4_p      = run_chatgit_n3n4(index, queries_gt, embed_model, sessions, k)
    cg_full_p      = run_chatgit_full(index, queries_gt, embed_model, sessions, k)
    print(f"  Done in {time.time()-t0:.1f}s")

    systems = {
        "BM25":                      bm25_p,
        "BM25-SlidingWindow (BM25)": bm25sw_bm25_p,
        "BM25-SlidingWindow (BGE)":  bm25sw_bge_p,
        "VanillaRAG (BGE)":          vanilla_p,
        "ChatGIT (Session+Intent)":  cg_n3n4_p,
        "ChatGIT (Full)":            cg_full_p,
    }

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5] Evaluating...")
    results = {}
    for name, preds in systems.items():
        results[name] = evaluate_retrieval(preds, ks=[1, 5, 10],
                                           n_bootstrap=2000)

    # ── Main table ────────────────────────────────────────────────────────────
    metrics = ["mrr", "p@1", "recall@5", "ndcg@5", "success@5"]
    rr = {name: f"  redund={redundancy_rate(preds, sessions):.3f}"
          for name, preds in systems.items()}

    print("\n" + "=" * 72)
    print(f"  MAIN RESULTS — {len(queries_gt)} turns, {len(sessions)} conversations, "
          f"{len(REPOS)} repos")
    print("=" * 72)
    print_table(results, metrics, extra=rr)

    # ── Per-intent ────────────────────────────────────────────────────────────
    intents = ["locate", "explain", "debug", "summarize"]
    print(f"\n  PER-INTENT MRR")
    header = f"  {'System':<30}" + "".join(f"{i:>14}" for i in intents)
    print(header)
    print("  " + "-" * (30 + 14 * len(intents)))
    for name, res in results.items():
        pi  = res.get("per_intent", {})
        row = f"  {name:<30}"
        for intent in intents:
            v = pi.get(intent, {}).get("mrr", 0.0)
            row += f"  {v:>12.4f}"
        print(row)

    # ── Per-repo MRR ──────────────────────────────────────────────────────────
    print(f"\n  PER-REPOSITORY MRR")
    key_systems = ["BM25", "BM25-SlidingWindow (BGE)", "VanillaRAG (BGE)",
                   "ChatGIT (Full)"]
    header = f"  {'Repo':<14}" + "".join(f"{s[:14]:>16}" for s in key_systems)
    print(header)
    print("  " + "-" * (14 + 16 * len(key_systems)))

    per_repo = defaultdict(lambda: defaultdict(list))
    for name, preds in systems.items():
        if name not in key_systems: continue
        for p in preds:
            repo = p["query_id"].split("_conv_")[0]
            mrr  = next((1 / i for i, c in enumerate(p["retrieved"], 1)
                         if c in p["ground_truth"]), 0.0)
            per_repo[repo][name].append(mrr)
    for repo in sorted(per_repo):
        row = f"  {repo:<14}"
        for s in key_systems:
            v = np.mean(per_repo[repo].get(s, [0.0]))
            row += f"  {v:>14.4f}"
        print(row)

    # ── Statistical significance ───────────────────────────────────────────────
    print(f"\n  STATISTICAL SIGNIFICANCE (ChatGIT Full vs. each baseline)")
    cg_map  = {p["query_id"]: p for p in cg_full_p}
    reports = []
    for name, preds in [
        ("BM25",                      bm25_p),
        ("BM25-SlidingWindow (BM25)", bm25sw_bm25_p),
        ("BM25-SlidingWindow (BGE)",  bm25sw_bge_p),
        ("VanillaRAG (BGE)",          vanilla_p),
        ("ChatGIT (Session+Intent)",  cg_n3n4_p),
    ]:
        oth_map = {p["query_id"]: p for p in preds}
        common  = sorted(set(cg_map) & set(oth_map))
        if len(common) < 2: continue
        for metric in ["mrr", "recall@5", "ndcg@5"]:
            a = np.array([cg_map[q][metric]  for q in common])
            b = np.array([oth_map[q][metric] for q in common])
            reports.append(full_comparison_report(
                a, b, metric_name=f"{metric} vs {name}",
                system_a_name="ChatGIT", system_b_name=name))
    print_comparison_table(reports)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    save = {}
    for name, res in results.items():
        save[name] = {
            m: {"mean":  res["summary"].get(m, {}).get("mean", 0),
                "ci_lo": res["summary"].get(m, {}).get("ci_lo", 0),
                "ci_hi": res["summary"].get(m, {}).get("ci_hi", 0)}
            for m in metrics
        }
        save[name]["per_intent"] = {
            i: {"mrr": res.get("per_intent", {}).get(i, {}).get("mrr", 0)}
            for i in intents
        }
        save[name]["redundancy_rate"] = redundancy_rate(
            systems[name], sessions)

    with open("results/full_eval_results.json", "w") as f:
        json.dump(save, f, indent=2)
    print("\n  Saved → results/full_eval_results.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
