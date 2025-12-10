"""
Real benchmark test using Flask repo + ConvCodeBench sample conversations.
Runs BM25, TF-IDF, VanillaRAG, and ChatGIT (no-LLM retrieval only) baselines.
"""

import sys, json, time, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from collections import defaultdict

from chatgit.core.chunker import chunk_repository
from chatgit.core.intent_classifier import classify_intent
from chatgit.core.session_memory import SessionRetrievalMemory

from evaluation.baselines import BM25, TFIDFRetriever, VanillaRAG, RepoCoderStyle
from evaluation.eval_retrieval import evaluate_retrieval, print_retrieval_report
from evaluation.statistical_tests import full_comparison_report, print_comparison_table

FLASK_REPO = os.environ.get("CHATGIT_REPO_FLASK",
                            os.path.join(os.environ.get("CHATGIT_REPO_BASE", "/tmp"), "flask_bench"))
CONVERSATIONS_PATH = os.environ.get(
    "CHATGIT_CONVS_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "convcodebench", "sample_conversations.jsonl")
)

# ── Ground truth re-mapped to actual chunker node_names ──────────────────────
# (verified from actual chunker output above)
GT_REMAPPED = {
    "flask_conv_001": [
        # Turn 0: Where is Flask class defined?
        {"gt_chunks": ["src/flask/app.py::Flask"],
         "gt_files":  ["src/flask/app.py"],
         "query":     "Where is the Flask application class defined?",
         "intent":    "locate"},
        # Turn 1: How does it handle URL routing?
        {"gt_chunks": ["src/flask/app.py::add_url_rule",
                       "src/flask/sansio/app.py::add_url_rule"],
         "gt_files":  ["src/flask/app.py", "src/flask/sansio/app.py"],
         "query":     "How does it handle URL routing?",
         "intent":    "explain"},
        # Turn 2: What happens when route throws exception?
        {"gt_chunks": ["src/flask/app.py::handle_exception",
                       "src/flask/app.py::handle_http_exception"],
         "gt_files":  ["src/flask/app.py"],
         "query":     "What happens when a route throws an exception?",
         "intent":    "debug"},
    ],
    "flask_conv_002": [
        {"gt_chunks": ["src/flask/app.py::Flask",
                       "src/flask/blueprints.py::Blueprint",
                       "src/flask/wrappers.py::Request"],
         "gt_files":  ["src/flask/app.py", "src/flask/blueprints.py", "src/flask/wrappers.py"],
         "query":     "Give me an overview of the Flask codebase architecture",
         "intent":    "summarize"},
        {"gt_chunks": ["src/flask/blueprints.py::Blueprint",
                       "src/flask/sansio/blueprints.py::BlueprintSetupState"],
         "gt_files":  ["src/flask/blueprints.py"],
         "query":     "What is the role of blueprints specifically?",
         "intent":    "explain"},
        {"gt_chunks": ["src/flask/sansio/app.py::_find_error_handler",
                       "src/flask/sansio/blueprints.py::errorhandler"],
         "gt_files":  ["src/flask/sansio/app.py"],
         "query":     "How do blueprints handle their own error handlers differently from the app?",
         "intent":    "explain"},
    ],
    "requests_conv_001": [
        {"gt_chunks": ["requests/sessions.py::Session"],
         "gt_files":  ["requests/sessions.py"],
         "query":     "Where is the Session class defined in requests?",
         "intent":    "locate"},
        {"gt_chunks": ["requests/sessions.py::send",
                       "requests/cookies.py::RequestsCookieJar"],
         "gt_files":  ["requests/sessions.py", "requests/cookies.py"],
         "query":     "How does it manage cookies across requests?",
         "intent":    "explain"},
        {"gt_chunks": ["requests/sessions.py::rebuild_auth",
                       "requests/sessions.py::resolve_redirects"],
         "gt_files":  ["requests/sessions.py"],
         "query":     "Is there a bug where cookies from redirected requests persist incorrectly?",
         "intent":    "debug"},
    ],
}


def build_chunks_for_flask():
    print("\n[Test] Chunking Flask repository...")
    t0 = time.time()
    docs = chunk_repository(FLASK_REPO)
    print(f"[Test] {len(docs)} chunks created in {time.time()-t0:.1f}s")

    # Build lookup structures
    chunks_list = []
    chunk_by_id = {}
    for d in docs:
        cid = f"{d.metadata['file_name']}::{d.metadata['node_name']}"
        text = d.text if hasattr(d, 'text') else d.page_content
        obj = {"id": cid, "text": text,
               "file": d.metadata['file_name'],
               "node_type": d.metadata.get('node_type',''),
               "node_name": d.metadata.get('node_name','')}
        chunks_list.append(obj)
        # Only keep first chunk per id (multiple chunks per large function)
        if cid not in chunk_by_id:
            chunk_by_id[cid] = obj

    return chunks_list, chunk_by_id


def make_retrieval_preds(retriever, queries, chunk_by_id, k=10):
    """Run a retriever and build predictions list."""
    preds = []
    for qid, query, gt_chunks, intent in queries:
        retrieved = retriever.retrieve_ids(query, k=k)
        preds.append({
            "query_id":     qid,
            "retrieved":    retrieved,
            "ground_truth": gt_chunks,
            "intent":       intent,
        })
    return preds


def make_chatgit_preds(docs, queries, k=10):
    """
    ChatGIT retrieval using:
    - N4: intent-based top_k
    - BGE embeddings for dense retrieval
    - Module summary chunk boost
    - N3: session memory redundancy suppression
    """
    from sentence_transformers import SentenceTransformer
    _hf_cache = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder=_hf_cache)

    print("[Test] Building ChatGIT embeddings (BGE)...")
    t0 = time.time()

    flask_docs = docs  # already filtered to flask src before call
    texts = [d['text'][:500] for d in flask_docs]

    emb_matrix = embed_model.encode(texts, batch_size=64, show_progress_bar=False,
                                     normalize_embeddings=True).astype(np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix /= norms

    print(f"[Test] Embedded {len(flask_docs)} chunks in {time.time()-t0:.1f}s")

    session_mem = SessionRetrievalMemory()
    preds = []

    for qid, query, gt_chunks, intent in queries:
        # N4: classify intent
        cfg = classify_intent(query)
        use_k = min(cfg.top_k, k)

        # N3: resolve coreference
        resolved_query = session_mem.resolve_coreferences(query)

        # Dense retrieval
        q_emb = embed_model.encode([resolved_query], normalize_embeddings=True)[0].astype(np.float32)
        sims = emb_matrix @ q_emb

        # N4: granularity boost for module_summary chunks
        for i, d in enumerate(flask_docs):
            if d['node_type'] == 'module_summary' and cfg.granularity == 'module':
                sims[i] *= cfg.granularity_boost
            elif d['node_type'] == 'function' and cfg.granularity == 'function':
                sims[i] *= cfg.granularity_boost

        top_idx = np.argsort(-sims)[:use_k]
        retrieved_ids = [flask_docs[i]['id'] for i in top_idx]

        # N3: apply session-memory redundancy suppression inline
        penalised = []
        for j, rid in enumerate(retrieved_ids):
            score = float(sims[top_idx[j]])
            fname = rid.split("::")[0] if "::" in rid else rid
            # Penalise previously retrieved chunks
            if rid in session_mem._retrieved:
                score *= session_mem.REDUNDANCY_PENALTY_LAST_TURN
            # Boost active session files
            if fname in session_mem._active_files:
                score *= (1.0 + session_mem.SESSION_ZONE_BONUS * session_mem._active_files[fname])
            penalised.append((rid, score))
        penalised.sort(key=lambda x: -x[1])
        retrieved_ids = [rid for rid, _ in penalised]

        preds.append({
            "query_id":     qid,
            "retrieved":    retrieved_ids,
            "ground_truth": gt_chunks,
            "intent":       intent,
        })

        # N3: record turn
        # N3: record_turn expects list of dicts
        retrieved_dicts = [
            {"file": rid.split("::")[0] if "::" in rid else rid,
             "node_name": rid.split("::")[-1] if "::" in rid else rid,
             "matched_funcs": []}
            for rid in retrieved_ids
        ]
        session_mem.record_turn(query, retrieved_dicts, f"Answer about {query[:50]}")

    return preds


def main():
    print("=" * 70)
    print("  ChatGIT Real Benchmark — Flask Repository")
    print("=" * 70)

    # Build chunks
    chunks_list, chunk_by_id = build_chunks_for_flask()

    # Build flat query list from GT
    queries = []
    for conv_id, turns in GT_REMAPPED.items():
        if conv_id.startswith("requests"):
            continue  # Flask only for this run
        for ti, turn in enumerate(turns):
            queries.append((
                f"{conv_id}_t{ti}",
                turn["query"],
                turn["gt_chunks"],
                turn["intent"],
            ))

    print(f"\n[Test] Queries: {len(queries)}")
    print(f"[Test] Unique queries:")
    for qid, q, gt, intent in queries:
        hits = sum(1 for g in gt if g in chunk_by_id)
        print(f"  [{intent:10}] {q[:55]:<55}  GT found: {hits}/{len(gt)}")

    # ── Fit baselines ──────────────────────────────────────────────────────
    flask_chunks = [c for c in chunks_list if c['file'].startswith('src/flask')]
    print(f"\n[Test] Flask src chunks for retrieval: {len(flask_chunks)}")

    print("\n[Test] Fitting BM25...")
    bm25 = BM25().fit(flask_chunks)

    print("[Test] Fitting TF-IDF...")
    tfidf = TFIDFRetriever().fit(flask_chunks)

    print("[Test] Fitting BM25-SlidingWindow (NOT RepoCoder — see baselines.py)...")
    bm25sw = RepoCoderStyle(base_retriever=TFIDFRetriever()).fit(flask_chunks)

    # ── Run retrievals ─────────────────────────────────────────────────────
    k = 10
    systems = {}

    print("\n[Test] Running BM25...")
    systems["BM25"] = make_retrieval_preds(bm25, queries, chunk_by_id, k=k)

    print("[Test] Running TF-IDF (VanillaRAG proxy)...")
    systems["TF-IDF/VanillaRAG"] = make_retrieval_preds(tfidf, queries, chunk_by_id, k=k)

    print("[Test] Running BM25-SlidingWindow...")
    systems["BM25-SlidingWindow"] = make_retrieval_preds(bm25sw, queries, chunk_by_id, k=k)

    print("[Test] Running ChatGIT (BGE + N3 + N4)...")
    chatgit_preds = make_chatgit_preds(flask_chunks, queries, k=k)
    systems["ChatGIT"] = chatgit_preds

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REAL RESULTS")
    print("=" * 70)

    ks = [1, 5, 10]
    results = {}
    for name, preds in systems.items():
        res = evaluate_retrieval(preds, ks=ks, n_bootstrap=2000)
        results[name] = res
        print_retrieval_report(res, title=f"{name}")

    # ── Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-SYSTEM COMPARISON TABLE")
    print("=" * 70)
    metrics = ["mrr", "recall@5", "ndcg@5", "p@1", "success@5"]

    header = f"  {'System':<22}" + "".join(f"  {m:>12}" for m in metrics)
    print(header)
    print("  " + "-" * 70)
    for name, res in results.items():
        summary = res["summary"]
        row = f"  {name:<22}"
        for m in metrics:
            s = summary.get(m, {})
            val = s.get("mean", 0.0)
            ci_lo = s.get("ci_lo", 0.0)
            ci_hi = s.get("ci_hi", 0.0)
            row += f"  {val:>6.4f} ±{(ci_hi-ci_lo)/2:>5.4f}"
        print(row)
    print()

    # ── Per-intent breakdown ───────────────────────────────────────────────
    print("  PER-INTENT MRR BREAKDOWN")
    print("  " + "-" * 70)
    intents = sorted({t[3] for t in queries})
    for name, res in results.items():
        per_intent = res.get("per_intent", {})
        row = f"  {name:<22}"
        for intent in intents:
            mrr = per_intent.get(intent, {}).get("mrr", 0.0)
            row += f"  {intent}: {mrr:.4f}"
        print(row)
    print()

    # ── Statistical tests: ChatGIT vs each baseline ────────────────────────
    print("  STATISTICAL SIGNIFICANCE (ChatGIT vs baselines)")
    print("  " + "-" * 70)
    chatgit_per_q = {r["query_id"]: r for r in results["ChatGIT"]["per_query"]}
    reports = []
    for name in ["BM25", "TF-IDF/VanillaRAG", "BM25-SlidingWindow"]:
        other_per_q = {r["query_id"]: r for r in results[name]["per_query"]}
        common = sorted(set(chatgit_per_q) & set(other_per_q))
        for metric in ["mrr", "recall@5"]:
            a = np.array([chatgit_per_q[q][metric] for q in common])
            b = np.array([other_per_q[q][metric] for q in common])
            report = full_comparison_report(
                a, b, metric_name=metric,
                system_a_name="ChatGIT",
                system_b_name=name
            )
            reports.append(report)
    print_comparison_table(reports)

    # ── Save results ──────────────────────────────────────────────────────
    import json
    import os
    os.makedirs("results", exist_ok=True)
    save = {}
    for name, res in results.items():
        save[name] = {
            m: {"mean": res["summary"][m]["mean"],
                "ci_lo": res["summary"][m]["ci_lo"],
                "ci_hi": res["summary"][m]["ci_hi"]}
            for m in metrics if m in res["summary"]
        }
    with open("results/real_benchmark_results.json", "w") as f:
        json.dump(save, f, indent=2)
    print("  Results saved to results/real_benchmark_results.json")


if __name__ == "__main__":
    main()
