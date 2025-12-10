"""
Latency & efficiency benchmarks for ChatGIT.

Measures:
  - Repository indexing time (chunking + embedding)
  - Per-query retrieval latency (mean, p50, p95, p99)
  - Index memory footprint (embedding matrix size)
  - Throughput (queries/second)

Repos tested: flask (small), requests (medium), fastapi (large)
"""

import sys, time, os, json
import numpy as np
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Pre-import torch on Windows to prevent c10.dll DLL initialization failure
import torch  # noqa: F401 — must come before sentence_transformers / transformers

import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4

from chatgit.core.chunker import chunk_repository
from chatgit.core.intent_classifier import classify_intent
from chatgit.core.session_memory import SessionRetrievalMemory
from evaluation.baselines import BM25
from sentence_transformers import SentenceTransformer

_REPO_BASE = os.environ.get("CHATGIT_REPO_BASE", "/tmp")
_repo_paths = {
    "flask":    os.environ.get("CHATGIT_REPO_FLASK",   os.path.join(_REPO_BASE, "flask_bench")),
    "requests": os.environ.get("CHATGIT_REPO_REQUESTS",os.path.join(_REPO_BASE, "requests_bench")),
    "fastapi":  os.environ.get("CHATGIT_REPO_FASTAPI", os.path.join(_REPO_BASE, "fastapi_bench")),
}
REPOS = {k: (v, sz) for (k, sz), v in zip(
    [("flask","small"), ("requests","medium"), ("fastapi","large")],
    _repo_paths.values()
) if os.path.isdir(v)}

SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "example",
             "__pycache__", ".git", "build", "dist"}

SAMPLE_QUERIES = [
    "Where is the main application class defined?",
    "How does routing work?",
    "What happens when an exception is raised in a route?",
    "How are middlewares handled?",
    "Explain how sessions are managed",
    "What is the blueprint architecture?",
    "How does request parsing work?",
    "How are cookies handled across redirects?",
    "Show me the error handling mechanism",
    "Where is authentication implemented?",
]


def benchmark_indexing(repo_path, embed_model):
    """Measure full indexing time: chunking + embedding."""
    # Chunking
    t0 = time.perf_counter()
    docs = chunk_repository(repo_path)
    chunk_time = time.perf_counter() - t0

    chunks = []
    for d in docs:
        fname = d.metadata["file_name"]
        parts = set(fname.replace("\\", "/").split("/"))
        if parts & SKIP_DIRS:
            continue
        text = d.text if hasattr(d, "text") else d.page_content
        chunks.append({"id": f"{fname}::{d.metadata['node_name']}",
                       "text": text[:600],
                       "file": fname,
                       "node_type": d.metadata.get("node_type", ""),
                       "node_name": d.metadata.get("node_name", "")})

    # Embedding
    texts = [c["text"] for c in chunks]
    t1 = time.perf_counter()
    embs = embed_model.encode(texts, batch_size=256,
                              show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)
    embed_time = time.perf_counter() - t1

    mem_mb = embs.nbytes / (1024 * 1024)

    return {
        "n_chunks": len(chunks),
        "chunk_time_s": round(chunk_time, 3),
        "embed_time_s": round(embed_time, 3),
        "total_index_time_s": round(chunk_time + embed_time, 3),
        "index_mem_mb": round(mem_mb, 2),
        "chunks_per_sec": round(len(chunks) / (chunk_time + embed_time), 1),
        "chunks": chunks,
        "embs": embs,
    }


def benchmark_retrieval(chunks, embs, embed_model, n_warmup=3, n_trials=50):
    """Measure per-query retrieval latency."""
    queries = (SAMPLE_QUERIES * ((n_trials // len(SAMPLE_QUERIES)) + 2))[:n_trials + n_warmup]

    # BM25 baseline latency
    bm25 = BM25().fit(chunks)
    bm25_latencies = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        bm25.retrieve_ids(q, k=10)
        lat = time.perf_counter() - t0
        if i >= n_warmup:
            bm25_latencies.append(lat * 1000)  # ms

    # VanillaRAG latency (encode + cosine)
    vanilla_latencies = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        qe = embed_model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        sims = embs @ qe
        np.argsort(-sims)[:10]
        lat = time.perf_counter() - t0
        if i >= n_warmup:
            vanilla_latencies.append(lat * 1000)

    # ChatGIT latency (encode + cosine + intent + session memory)
    chatgit_latencies = []
    mem = SessionRetrievalMemory()
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        cfg = classify_intent(q)
        resolved = mem.resolve_coreferences(q)
        qe = embed_model.encode([resolved], normalize_embeddings=True)[0].astype(np.float32)
        sims = (embs @ qe).copy()
        for j, c in enumerate(chunks):
            nt = c["node_type"]
            if nt == "module_summary" and cfg.granularity == "module":
                sims[j] *= cfg.granularity_boost
            elif nt == "function" and cfg.granularity == "function":
                sims[j] *= 1.15
        top_idx = np.argsort(-sims)[:cfg.top_k]
        scored = []
        for j in top_idx:
            rid = chunks[j]["id"]
            score = float(sims[j])
            fname = chunks[j]["file"]
            if rid in mem._retrieved:
                score *= mem.REDUNDANCY_PENALTY_LAST_TURN
            if fname in mem._active_files:
                score *= (1.0 + mem.SESSION_ZONE_BONUS * mem._active_files[fname])
            scored.append((rid, score))
        scored.sort(key=lambda x: -x[1])
        lat = time.perf_counter() - t0
        if i >= n_warmup:
            chatgit_latencies.append(lat * 1000)
        if i % 5 == 0:
            retrieved_dicts = [{"file": r.split("::")[0] if "::" in r else r,
                                "node_name": r.split("::")[-1] if "::" in r else r,
                                "matched_funcs": []} for r, _ in scored[:10]]
            mem.record_turn(q, retrieved_dicts, f"[bench] {q[:30]}")

    def stats(lats):
        a = np.array(lats)
        return {
            "mean_ms":   round(float(np.mean(a)), 2),
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms":    round(float(np.percentile(a, 95)), 2),
            "p99_ms":    round(float(np.percentile(a, 99)), 2),
            "qps":       round(1000.0 / float(np.mean(a)), 1),
        }

    return {
        "n_trials": n_trials,
        "BM25":       stats(bm25_latencies),
        "VanillaRAG": stats(vanilla_latencies),
        "ChatGIT":    stats(chatgit_latencies),
    }


def main():
    print("=" * 70)
    print("  ChatGIT — Latency & Efficiency Benchmarks")
    print("=" * 70)

    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5",
                                      cache_folder=os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")))

    indexing_results = {}
    retrieval_results = {}

    for repo_id, (repo_path, size_label) in REPOS.items():
        print(f"\n[{repo_id}] ({size_label})")
        print(f"  Benchmarking indexing...", flush=True)
        idx = benchmark_indexing(repo_path, embed_model)
        indexing_results[repo_id] = {k: v for k, v in idx.items()
                                     if k not in ("chunks", "embs")}
        indexing_results[repo_id]["size_label"] = size_label
        print(f"    {idx['n_chunks']} chunks | "
              f"chunk: {idx['chunk_time_s']}s | "
              f"embed: {idx['embed_time_s']}s | "
              f"total: {idx['total_index_time_s']}s | "
              f"mem: {idx['index_mem_mb']}MB")

        print(f"  Benchmarking retrieval (n=50 queries)...", flush=True)
        ret = benchmark_retrieval(idx["chunks"], idx["embs"], embed_model,
                                  n_warmup=5, n_trials=50)
        retrieval_results[repo_id] = ret
        for sys_name, s in ret.items():
            if isinstance(s, dict):
                print(f"    {sys_name:<14} mean={s['mean_ms']}ms  "
                      f"p95={s['p95_ms']}ms  {s['qps']} qps")

    # ── Pretty table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INDEXING BENCHMARKS")
    print("=" * 70)
    print(f"  {'Repo':<12}  {'Size':>8}  {'Chunks':>8}  "
          f"{'ChunkT(s)':>10}  {'EmbedT(s)':>10}  "
          f"{'TotalT(s)':>10}  {'MemMB':>8}  {'C/s':>8}")
    print("  " + "-" * 82)
    for rid, r in indexing_results.items():
        print(f"  {rid:<12}  {r['size_label']:>8}  {r['n_chunks']:>8}  "
              f"{r['chunk_time_s']:>10.2f}  {r['embed_time_s']:>10.2f}  "
              f"{r['total_index_time_s']:>10.2f}  {r['index_mem_mb']:>8.1f}  "
              f"{r['chunks_per_sec']:>8.0f}")

    print(f"\n{'='*70}")
    print(f"  RETRIEVAL LATENCY (ms) — {list(REPOS)[0]}–{list(REPOS)[-1]}")
    print("=" * 70)
    for repo_id in REPOS:
        ret = retrieval_results[repo_id]
        print(f"\n  {repo_id}:")
        print(f"  {'System':<14}  {'Mean':>8}  {'Median':>8}  {'P95':>8}  {'P99':>8}  {'QPS':>8}")
        print("  " + "-" * 62)
        for sys_name, s in ret.items():
            if isinstance(s, dict):
                print(f"  {sys_name:<14}  {s['mean_ms']:>8.2f}  "
                      f"{s['median_ms']:>8.2f}  {s['p95_ms']:>8.2f}  "
                      f"{s['p99_ms']:>8.2f}  {s['qps']:>8.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = {"indexing": indexing_results, "retrieval": retrieval_results}
    with open("results/latency_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved → results/latency_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
