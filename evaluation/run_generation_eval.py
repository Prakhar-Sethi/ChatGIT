"""
Lightweight generation quality evaluation for ConvCodeBench.

Uses retrieved chunk text as hypothesis and reference_answer from the
JSONL as gold. Reports CodeBLEU-proxy, ROUGE-L, and Edit Similarity.

This provides an offline upper-bound on retrieval-grounded generation
quality without requiring live LLM API calls.

Usage:
    python -m evaluation.run_generation_eval
"""

import sys, os, json
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch  # noqa: F401
import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4

import numpy as np
from sentence_transformers import SentenceTransformer
from chatgit.core.chunker import chunk_repository
from evaluation.eval_generation import evaluate_generation

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_BASE = os.environ.get("CHATGIT_REPO_BASE", "/tmp")
REPOS = {
    "flask":    os.environ.get("CHATGIT_REPO_FLASK",    os.path.join(_REPO_BASE, "flask_bench")),
    "requests": os.environ.get("CHATGIT_REPO_REQUESTS", os.path.join(_REPO_BASE, "requests_bench")),
    "click":    os.environ.get("CHATGIT_REPO_CLICK",    os.path.join(_REPO_BASE, "click_bench")),
    "fastapi":  os.environ.get("CHATGIT_REPO_FASTAPI",  os.path.join(_REPO_BASE, "fastapi_bench")),
    "celery":   os.environ.get("CHATGIT_REPO_CELERY",   os.path.join(_REPO_BASE, "celery_bench")),
}
REPOS = {k: v for k, v in REPOS.items() if os.path.isdir(v)}

CONVERSATIONS_PATH = os.path.join(
    _project_root, "data", "convcodebench", "sample_conversations.jsonl"
)

SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "example",
             "benchmarks", "scripts", "contrib", "extras", "tools", "__pycache__"}


def build_chunk_index(repo_id, repo_path, embed_model):
    docs = chunk_repository(repo_path)
    chunks, texts = [], []
    for d in docs:
        fname = d.metadata.get("file_name", "").replace("\\", "/")
        parts = set(fname.split("/"))
        if parts & SKIP_DIRS:
            continue
        text = d.text if hasattr(d, "text") else d.page_content
        cid = f"{fname}::{d.metadata.get('node_name', '')}"
        chunks.append({"id": cid, "text": text})
        texts.append(text[:600])

    embs = embed_model.encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    return chunks, embs


def retrieve_top1(query: str, chunks, embs, embed_model) -> str:
    qe = embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sims = embs @ qe
    top_idx = int(np.argmax(sims))
    return chunks[top_idx]["text"][:800]


def main():
    print("=" * 60)
    print("  ConvCodeBench — Generation Quality Evaluation")
    print("=" * 60)

    # Load conversations — only those with reference_answer + repo in REPOS
    convs = []
    with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["repo_id"] not in REPOS:
                continue
            # Only use manually-annotated conversations with real reference answers
            if c.get("metadata", {}).get("annotator_id", "").startswith("AUTO"):
                continue
            convs.append(c)

    print(f"Evaluating {len(convs)} manually-annotated conversations")

    embed_model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        cache_folder=os.environ.get("HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
        device="cpu"
    )

    # Build chunk indices per repo
    indices = {}
    for repo_id, repo_path in REPOS.items():
        print(f"  Indexing {repo_id}...", end=" ", flush=True)
        chunks, embs = build_chunk_index(repo_id, repo_path, embed_model)
        indices[repo_id] = (chunks, embs)
        print(f"{len(chunks)} chunks")

    # Build eval pairs
    eval_pairs = []
    for conv in convs:
        repo_id = conv["repo_id"]
        chunks, embs = indices[repo_id]
        for turn in conv["turns"]:
            ref = turn.get("reference_answer", "").strip()
            if not ref or len(ref) < 20:
                continue
            query = turn["query"]
            hypothesis = retrieve_top1(query, chunks, embs, embed_model)
            eval_pairs.append({
                "query_id": f"{conv['conversation_id']}_t{turn['turn_id']}",
                "hypothesis": hypothesis,
                "reference": ref,
                "intent": turn.get("intent", "unknown"),
            })

    print(f"\nEvaluating {len(eval_pairs)} turns...")
    results = evaluate_generation(eval_pairs)

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("  GENERATION QUALITY METRICS")
    print("=" * 60)
    print(f"  ROUGE-L F1:        {summary['rouge_l']['mean']:.3f} "
          f"(CI: {summary['rouge_l']['ci_lo']:.3f}–{summary['rouge_l']['ci_hi']:.3f})")
    print(f"  CodeBLEU-proxy:    {summary['code_bleu']['mean']:.3f} "
          f"(CI: {summary['code_bleu']['ci_lo']:.3f}–{summary['code_bleu']['ci_hi']:.3f})")
    print(f"  Edit Similarity:   {summary['edit_similarity']['mean']:.3f}")
    print(f"  BERTScore-proxy:   {summary['bertscore_f1']['mean']:.3f}")
    if "per_intent" in results:
        print("\n  Per-intent ROUGE-L and CodeBLEU:")
        for intent, m in sorted(results["per_intent"].items()):
            rl = m.get('rouge_l', 0)
            cb = m.get('code_bleu', 0)
            print(f"    {intent:<12} ROUGE-L={rl:.3f}  CodeBLEU={cb:.3f}")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved → results/generation_results.json")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
