"""
Turn-position analysis for ChatGIT.

Measures how retrieval quality changes across conversation turns (turn 0, 1, 2, 3+).
This directly validates N3 (session memory): later turns should benefit more
from context about what was already retrieved.

Also measures: conversation-level coherence (how many turns in a conversation
had at least one hit in @5).
"""

import sys, os, json, time
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch  # noqa: F401 — must come before sentence_transformers / transformers

import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4

import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from chatgit.core.chunker import chunk_repository
from chatgit.core.intent_classifier import classify_intent
from chatgit.core.session_memory import SessionRetrievalMemory
from evaluation.baselines import BM25
from evaluation.eval_retrieval import evaluate_retrieval

_REPO_BASE = os.environ.get("CHATGIT_REPO_BASE", "/tmp")
REPOS = {
    "flask":    os.environ.get("CHATGIT_REPO_FLASK",    os.path.join(_REPO_BASE, "flask_bench")),
    "requests": os.environ.get("CHATGIT_REPO_REQUESTS", os.path.join(_REPO_BASE, "requests_bench")),
    "click":    os.environ.get("CHATGIT_REPO_CLICK",    os.path.join(_REPO_BASE, "click_bench")),
    "fastapi":  os.environ.get("CHATGIT_REPO_FASTAPI",  os.path.join(_REPO_BASE, "fastapi_bench")),
    "celery":   os.environ.get("CHATGIT_REPO_CELERY",   os.path.join(_REPO_BASE, "celery_bench")),
}
REPOS = {k: v for k, v in REPOS.items() if os.path.isdir(v)}
CONVERSATIONS_PATH = os.environ.get(
    "CHATGIT_CONVS_PATH",
    os.path.join(_project_root, "data", "convcodebench", "sample_conversations.jsonl")
)
SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "example",
             "__pycache__", ".git", "build", "dist"}


def chunk_repo(repo_path):
    docs = chunk_repository(repo_path)
    chunks = []
    for d in docs:
        fname = d.metadata["file_name"]
        parts = set(fname.replace("\\", "/").split("/"))
        if parts & SKIP_DIRS:
            continue
        cid  = f"{fname}::{d.metadata['node_name']}"
        text = d.text if hasattr(d, "text") else d.page_content
        chunks.append({"id": cid, "text": text[:600],
                       "file": fname,
                       "node_type": d.metadata.get("node_type", ""),
                       "node_name": d.metadata.get("node_name", "")})
    return chunks


def fuzzy_match_gt(gt_id, by_id):
    if gt_id in by_id:
        return [gt_id]
    parts = gt_id.split("::")
    if len(parts) != 2:
        return []
    file_part, name_part = parts
    bare = name_part.split(".")[-1]
    hits = []
    for cid, c in by_id.items():
        file_ok = (not file_part or file_part in c["file"] or
                   c["file"].split("/")[-1] == file_part.split("/")[-1])
        name_ok = (c["node_name"] == bare or c["node_name"] == name_part)
        if file_ok and name_ok:
            hits.append(cid)
    if not hits and bare and len(bare) > 3:
        for cid, c in by_id.items():
            if c["node_name"] == bare:
                hits.append(cid)
    return hits[:3]


def mrr(retrieved, ground_truth):
    for i, rid in enumerate(retrieved, 1):
        if rid in ground_truth:
            return 1.0 / i
    return 0.0


def success_at_k(retrieved, ground_truth, k=5):
    return 1.0 if any(r in ground_truth for r in retrieved[:k]) else 0.0


def main():
    print("=" * 70)
    print("  ChatGIT — Turn-Position Analysis")
    print("=" * 70)

    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5",
                                      cache_folder=os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")))

    # Load and index repos
    print("\n[1/3] Indexing repositories...")
    index = {}
    for rid, path in REPOS.items():
        print(f"  {rid}...", end=" ", flush=True)
        t0 = time.time()
        chunks = chunk_repo(path)
        by_id  = {c["id"]: c for c in chunks}
        texts  = [c["text"] for c in chunks]
        embs   = embed_model.encode(texts, batch_size=64,
                                    show_progress_bar=False,
                                    normalize_embeddings=True,
                                    device="cpu").astype(np.float32)
        bm25   = BM25().fit(chunks)
        index[rid] = {"chunks": chunks, "by_id": by_id, "embs": embs, "bm25": bm25}
        print(f"{len(chunks)} chunks  ({time.time()-t0:.1f}s)")

    # Load conversations, keeping turn-position info
    print("\n[2/3] Loading conversations...")
    convs_raw = []
    with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs_raw.append(json.loads(line))

    # per_turn_position[system][turn_pos] = list of (mrr, success@5) per query
    metrics_by_pos = {
        "VanillaRAG": defaultdict(list),
        "BM25":       defaultdict(list),
        "ChatGIT":    defaultdict(list),
    }
    # conversation-level: did this conversation have ALL turns hit?
    conv_coherence = {"VanillaRAG": [], "BM25": [], "ChatGIT": []}
    # Track how many turns were included per conversation
    conv_lengths = []

    print("\n[3/3] Running retrieval by turn position...")
    n_gt_turns = 0
    for conv in convs_raw:
        rid = conv["repo_id"]
        if rid not in index:
            continue
        ix = index[rid]

        # Resolve GT for each turn
        turns_gt = []
        for turn in conv["turns"]:
            matched_gt = []
            for g in turn.get("ground_truth_chunks", []):
                m = fuzzy_match_gt(g, ix["by_id"])
                matched_gt.extend(m)
            seen = set()
            matched_gt = [x for x in matched_gt if not (x in seen or seen.add(x))]
            if matched_gt:
                turns_gt.append((turn["query"], matched_gt,
                                 turn.get("intent", "unknown"),
                                 turn["turn_id"]))

        if not turns_gt:
            continue
        conv_lengths.append(len(turns_gt))
        n_gt_turns += len(turns_gt)

        # VanillaRAG (no session memory)
        vr_success = []
        for query, gt, intent, turn_id in turns_gt:
            pos = min(turn_id, 3)   # 0,1,2,3+
            qe  = embed_model.encode([query], normalize_embeddings=True, device="cpu")[0].astype(np.float32)
            sims = ix["embs"] @ qe
            top  = [ix["chunks"][i]["id"] for i in np.argsort(-sims)[:10]]
            m    = mrr(top, gt)
            s5   = success_at_k(top, gt, 5)
            metrics_by_pos["VanillaRAG"][pos].append((m, s5))
            vr_success.append(s5)
        conv_coherence["VanillaRAG"].append(float(np.mean(vr_success)))

        # BM25 (no session memory)
        bm_success = []
        for query, gt, intent, turn_id in turns_gt:
            pos  = min(turn_id, 3)
            top  = ix["bm25"].retrieve_ids(query, k=10)
            m    = mrr(top, gt)
            s5   = success_at_k(top, gt, 5)
            metrics_by_pos["BM25"][pos].append((m, s5))
            bm_success.append(s5)
        conv_coherence["BM25"].append(float(np.mean(bm_success)))

        # ChatGIT(Routed) — intent-aware routing, processes turns IN ORDER
        # Routing table (matches run_convcodebench.py):
        #   EXPLAIN   → pure cosine (no N3, no N4): explanations need the same
        #               chunk as the prior LOCATE turn, so N3 must not penalise it
        #   LOCATE    → N3 (redundancy penalty) + N4 (function granularity boost)
        #   DEBUG     → N3 (redundancy penalty; session context helps find callers)
        #   SUMMARIZE → N4 only (module-level granularity boost; no redundancy penalty)
        #   unknown   → N3 + N4 (safe default)
        mem = SessionRetrievalMemory()
        cg_success = []
        for query, gt, intent, turn_id in turns_gt:
            pos = min(turn_id, 3)
            cfg = classify_intent(query)

            # use the ground-truth intent label when available (oracle routing)
            routed_intent = intent if intent in ("locate", "explain", "debug", "summarize") \
                            else (cfg.intent if hasattr(cfg, "intent") else "unknown")

            resolved = mem.resolve_coreferences(query)
            qe  = embed_model.encode([resolved], normalize_embeddings=True, device="cpu")[0].astype(np.float32)
            sims = (ix["embs"] @ qe).copy()

            use_n3 = routed_intent in ("locate", "debug", "unknown")
            use_n4 = routed_intent in ("locate", "summarize", "unknown")

            # N4: granularity boost (LOCATE/SUMMARIZE only)
            if use_n4:
                for i, c in enumerate(ix["chunks"]):
                    nt = c["node_type"]
                    if nt == "module_summary" and cfg.granularity == "module":
                        sims[i] *= cfg.granularity_boost
                    elif nt == "function" and cfg.granularity == "function":
                        sims[i] *= 1.15

            top_idx = np.argsort(-sims)[:cfg.top_k]

            # N3: session scoring (LOCATE/DEBUG only — NOT EXPLAIN)
            if use_n3:
                scored = []
                for i in top_idx:
                    rid2  = ix["chunks"][i]["id"]
                    score = float(sims[i])
                    fname = ix["chunks"][i]["file"]
                    if rid2 in mem._retrieved:
                        score *= mem.REDUNDANCY_PENALTY_LAST_TURN
                    if fname in mem._active_files:
                        score *= (1.0 + mem.SESSION_ZONE_BONUS * mem._active_files[fname])
                    scored.append((rid2, score))
                scored.sort(key=lambda x: -x[1])
                top_cg = [r for r, _ in scored[:10]]
            else:
                top_cg = [ix["chunks"][i]["id"] for i in top_idx[:10]]

            m  = mrr(top_cg, gt)
            s5 = success_at_k(top_cg, gt, 5)
            metrics_by_pos["ChatGIT"][pos].append((m, s5))
            cg_success.append(s5)

            mem.record_turn(query,
                [{"file": r.split("::")[0] if "::" in r else r,
                  "node_name": r.split("::")[-1] if "::" in r else r,
                  "matched_funcs": []} for r in top_cg],
                f"[{intent}]")
        conv_coherence["ChatGIT"].append(float(np.mean(cg_success)))

    print(f"\n  Evaluated {n_gt_turns} GT turns across {len(convs_raw)} conversations")

    # ── Turn-Position Table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TURN-POSITION MRR (turn 0=first, 3=4th+)")
    print("=" * 70)
    pos_labels = {0: "Turn 0 (1st)", 1: "Turn 1 (2nd)",
                  2: "Turn 2 (3rd)", 3: "Turn 3+ (4th+)"}
    header = f"  {'System':<16}" + "".join(f"  {pos_labels[p]:>14}" for p in sorted(pos_labels))
    print(header)
    print("  " + "-" * 72)
    for sys_name, by_pos in metrics_by_pos.items():
        row = f"  {sys_name:<16}"
        for p in sorted(pos_labels):
            vals = by_pos.get(p, [])
            if vals:
                mrr_mean = np.mean([v[0] for v in vals])
                row += f"  {mrr_mean:>14.4f}"
            else:
                row += f"  {'n/a':>14}"
        print(row)

    # ── Turn-Position Success@5 ───────────────────────────────────────────────
    print(f"\n  TURN-POSITION SUCCESS@5")
    print(header)
    print("  " + "-" * 72)
    for sys_name, by_pos in metrics_by_pos.items():
        row = f"  {sys_name:<16}"
        for p in sorted(pos_labels):
            vals = by_pos.get(p, [])
            if vals:
                s5_mean = np.mean([v[1] for v in vals])
                row += f"  {s5_mean:>14.4f}"
            else:
                row += f"  {'n/a':>14}"
        print(row)

    # ── Conversation Coherence ────────────────────────────────────────────────
    print(f"\n  CONVERSATION-LEVEL COHERENCE (mean success@5 per conversation)")
    print(f"  (higher = more turns in conversation had a hit in top-5)")
    print(f"  {'System':<16}  {'Mean':>8}  {'Std':>8}  {'% perfect convs':>16}")
    print("  " + "-" * 52)
    for sys_name, scores in conv_coherence.items():
        a = np.array(scores)
        perfect = np.mean(a == 1.0) * 100
        print(f"  {sys_name:<16}  {np.mean(a):>8.4f}  {np.std(a):>8.4f}  {perfect:>15.1f}%")

    # ── N3 Redundancy by position ──────────────────────────────────────────────
    print(f"\n  QUERY COUNT PER TURN POSITION")
    for p in sorted(pos_labels):
        n = len(metrics_by_pos["ChatGIT"].get(p, []))
        print(f"  {pos_labels[p]}: {n} queries")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = {}
    for sys_name, by_pos in metrics_by_pos.items():
        out[sys_name] = {}
        for p, vals in by_pos.items():
            out[sys_name][str(p)] = {
                "mrr": float(np.mean([v[0] for v in vals])) if vals else 0.0,
                "success@5": float(np.mean([v[1] for v in vals])) if vals else 0.0,
                "n": len(vals)
            }
    out["conv_coherence"] = {
        sys_name: {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores)),
            "pct_perfect": float(np.mean(np.array(scores) == 1.0) * 100)
        }
        for sys_name, scores in conv_coherence.items()
    }
    with open("results/turn_position_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved -> results/turn_position_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
