"""
LLM-as-Judge proxy evaluation for ChatGIT.

Uses Groq (Llama-3.3-70B) to rate the top-1 retrieved chunk for
ChatGIT(Routed) vs VanillaRAG on relevance and completeness.
This serves as a proxy for human evaluation.

Usage:
    GROQ_API_KEY=<key> python -m evaluation.llm_judge_eval

Output: results/llm_judge_results.json
"""

import sys, os, json, time, re, urllib.request
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch  # noqa: F401
import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4

import numpy as np
# Use REST API directly to avoid httpx/groq SDK version conflicts
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
from sentence_transformers import SentenceTransformer
from chatgit.core.chunker import chunk_repository
from chatgit.core.intent_classifier import classify_intent
from chatgit.core.session_memory import SessionRetrievalMemory

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
             "__pycache__", ".git", "build", "dist"}

JUDGE_PROMPT = """You are an expert software engineer evaluating a code retrieval system.

The user asked: "{query}"

The system retrieved this code chunk:
```
{chunk}
```

Rate this retrieved chunk on two criteria (score 1-5):
1. RELEVANCE: How relevant is this chunk to the user's query?
   1=completely irrelevant, 3=somewhat relevant, 5=exactly what was needed
2. COMPLETENESS: How completely does this chunk answer the query?
   1=answers nothing, 3=partially answers, 5=fully answers

Reply with ONLY this JSON format (no explanation):
{{"relevance": <1-5>, "completeness": <1-5>}}"""

SET_JUDGE_PROMPT = """You are an expert software engineer evaluating a code retrieval system.

The user asked: "{query}"

The system returned these 5 code chunks (in ranked order):
{chunks_text}

Rate this retrieved SET of chunks on two criteria (score 1-5):
1. COVERAGE: Taken together, how well do these 5 chunks cover what the user needs?
   1=none of them help, 3=partially covered, 5=fully covered by the set
2. DIVERSITY: How diverse and non-redundant are these 5 chunks?
   1=all chunks are repetitive/duplicate, 3=some overlap, 5=all chunks add unique information

Reply with ONLY this JSON format (no explanation):
{{"coverage": <1-5>, "diversity": <1-5>}}"""


def chunk_repo(repo_path):
    docs = chunk_repository(repo_path)
    chunks = []
    for d in docs:
        fname = d.metadata["file_name"].replace("\\", "/")
        if set(fname.split("/")) & SKIP_DIRS:
            continue
        cid = f"{fname}::{d.metadata['node_name']}"
        text = d.text if hasattr(d, "text") else d.page_content
        chunks.append({"id": cid, "text": text[:600],
                       "file": fname,
                       "node_type": d.metadata.get("node_type", ""),
                       "node_name": d.metadata.get("node_name", "")})
    return chunks


def retrieve_vanilla(query, chunks, embs, embed_model):
    qe = embed_model.encode([query], normalize_embeddings=True, device="cpu")[0].astype(np.float32)
    sims = embs @ qe
    idx = int(np.argmax(sims))
    return chunks[idx]["text"]


def retrieve_vanilla_top5(query, chunks, embs, embed_model):
    qe = embed_model.encode([query], normalize_embeddings=True, device="cpu")[0].astype(np.float32)
    sims = embs @ qe
    top5 = np.argsort(-sims)[:5]
    return [chunks[i]["text"] for i in top5]


def retrieve_chatgit(query, intent, chunks, embs, embed_model, mem):
    cfg = classify_intent(query)
    resolved = mem.resolve_coreferences(query)
    qe = embed_model.encode([resolved], normalize_embeddings=True, device="cpu")[0].astype(np.float32)
    sims = (embs @ qe).copy()

    use_n3 = intent in ("locate", "debug", "unknown")
    use_n4 = intent in ("locate", "summarize", "unknown")

    if use_n4:
        for i, c in enumerate(chunks):
            nt = c["node_type"]
            if nt == "module_summary" and cfg.granularity == "module":
                sims[i] *= cfg.granularity_boost
            elif nt == "function" and cfg.granularity == "function":
                sims[i] *= 1.15

    top_idx = np.argsort(-sims)[:cfg.top_k]

    if use_n3:
        scored = []
        for i in top_idx:
            rid = chunks[i]["id"]
            score = float(sims[i])
            fname = chunks[i]["file"]
            if rid in mem._retrieved:
                score *= mem.REDUNDANCY_PENALTY_LAST_TURN
            if fname in mem._active_files:
                score *= (1.0 + mem.SESSION_ZONE_BONUS * mem._active_files[fname])
            scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        top1_idx = scored[0][0]
    else:
        top1_idx = top_idx[0]

    top10 = [chunks[i]["id"] for i in (top_idx if not use_n3 else [s[0] for s in scored[:10]])]
    mem.record_turn(query,
        [{"file": r.split("::")[0] if "::" in r else r,
          "node_name": r.split("::")[-1] if "::" in r else r,
          "matched_funcs": []} for r in top10],
        f"[{intent}]")
    return chunks[top1_idx]["text"]


def retrieve_chatgit_top5(query, intent, chunks, embs, embed_model, mem):
    cfg = classify_intent(query)
    resolved = mem.resolve_coreferences(query)
    qe = embed_model.encode([resolved], normalize_embeddings=True, device="cpu")[0].astype(np.float32)
    sims = (embs @ qe).copy()

    use_n3 = intent in ("locate", "debug", "unknown")
    use_n4 = intent in ("locate", "summarize", "unknown")

    if use_n4:
        for i, c in enumerate(chunks):
            nt = c["node_type"]
            if nt == "module_summary" and cfg.granularity == "module":
                sims[i] *= cfg.granularity_boost
            elif nt == "function" and cfg.granularity == "function":
                sims[i] *= 1.15

    top_idx = np.argsort(-sims)[:cfg.top_k]

    if use_n3:
        scored = []
        for i in top_idx:
            rid = chunks[i]["id"]
            score = float(sims[i])
            fname = chunks[i]["file"]
            _penalty = (0.75 if intent == "debug"
                        else mem.REDUNDANCY_PENALTY_LAST_TURN)
            if rid in mem._retrieved:
                score *= _penalty
            if fname in mem._active_files:
                score *= (1.0 + mem.SESSION_ZONE_BONUS * mem._active_files[fname])
            scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        top5_indices = [s[0] for s in scored[:5]]
        top10_ids = [chunks[s[0]]["id"] for s in scored[:10]]
    else:
        top5_indices = list(top_idx[:5])
        top10_ids = [chunks[i]["id"] for i in top_idx[:10]]

    mem.record_turn(query,
        [{"file": r.split("::")[0] if "::" in r else r,
          "node_name": r.split("::")[-1] if "::" in r else r,
          "matched_funcs": []} for r in top10_ids],
        f"[{intent}]")
    return [chunks[i]["text"] for i in top5_indices]


def judge_chunk(api_key, query, chunk_text, model="llama-3.3-70b-versatile"):
    prompt = JUDGE_PROMPT.format(query=query, chunk=chunk_text[:800])
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50,
    }).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                GROQ_API_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "python-httpx/0.27.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            raw = data["choices"][0]["message"]["content"].strip()
            m = re.search(r'\{[^}]+\}', raw)
            if m:
                scores = json.loads(m.group())
                return float(scores.get("relevance", 3)), float(scores.get("completeness", 3))
        except Exception as e:
            print(f"    Judge error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return 3.0, 3.0  # fallback neutral


def judge_set(api_key, query, chunk_texts, model="llama-3.3-70b-versatile"):
    """Rate top-5 retrieved chunk SET on coverage + diversity (set-level quality)."""
    chunks_formatted = "\n\n".join(
        f"[Chunk {i+1}]\n```\n{t[:400]}\n```" for i, t in enumerate(chunk_texts)
    )
    prompt = SET_JUDGE_PROMPT.format(query=query, chunks_text=chunks_formatted)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50,
    }).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                GROQ_API_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "python-httpx/0.27.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            raw = data["choices"][0]["message"]["content"].strip()
            m = re.search(r'\{[^}]+\}', raw)
            if m:
                scores = json.loads(m.group())
                return float(scores.get("coverage", 3)), float(scores.get("diversity", 3))
        except Exception as e:
            print(f"    Set-judge error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return 3.0, 3.0  # fallback neutral


def main():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Run with: GROQ_API_KEY=<key> python -m evaluation.llm_judge_eval")
        sys.exit(1)

    print("=" * 60)
    print("  ChatGIT -- LLM-as-Judge Evaluation (Groq Llama-3.3-70B)")
    print("=" * 60)

    embed_model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        cache_folder=os.environ.get("HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
        device="cpu"
    )

    print("\n[1/3] Indexing repos...")
    indices = {}
    for repo_id, repo_path in REPOS.items():
        print(f"  {repo_id}...", end=" ", flush=True)
        chunks = chunk_repo(repo_path)
        texts = [c["text"] for c in chunks]
        embs = embed_model.encode(texts, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False, device="cpu").astype(np.float32)
        indices[repo_id] = {"chunks": chunks, "embs": embs}
        print(f"{len(chunks)} chunks")

    print("\n[2/3] Loading conversations (manually-annotated only)...")
    convs = []
    with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["repo_id"] not in indices:
                continue
            if c.get("metadata", {}).get("annotator_id", "").startswith("AUTO"):
                continue
            convs.append(c)
    print(f"  {len(convs)} manually-annotated conversations")

    # Sample up to 30 turns for judging (balance intents)
    import random
    random.seed(42)
    all_turns = []
    for conv in convs:
        for turn in conv["turns"]:
            all_turns.append((conv["repo_id"], conv["conversation_id"], turn))
    random.shuffle(all_turns)
    sample = all_turns[:30]

    print(f"\n[3/3] Running LLM judge on {len(sample)} turns...")
    results = []
    for repo_id, conv_id, turn in sample:
        query = turn["query"]
        intent = turn.get("intent", "unknown")
        ix = indices[repo_id]

        top1_vanilla = retrieve_vanilla(query, ix["chunks"], ix["embs"], embed_model)
        top5_vanilla = retrieve_vanilla_top5(query, ix["chunks"], ix["embs"], embed_model)

        mem1 = SessionRetrievalMemory()
        top1_chatgit = retrieve_chatgit(query, intent, ix["chunks"], ix["embs"], embed_model, mem1)
        mem5 = SessionRetrievalMemory()
        top5_chatgit = retrieve_chatgit_top5(query, intent, ix["chunks"], ix["embs"], embed_model, mem5)

        print(f"  [{intent}] {query[:55]}...", end=" ", flush=True)
        rel_v, comp_v = judge_chunk(api_key, query, top1_vanilla)
        rel_c, comp_c = judge_chunk(api_key, query, top1_chatgit)
        cov_v, div_v  = judge_set(api_key, query, top5_vanilla)
        cov_c, div_c  = judge_set(api_key, query, top5_chatgit)
        print(f"V=({rel_v:.0f},{comp_v:.0f},cov={cov_v:.0f},div={div_v:.0f}) "
              f"C=({rel_c:.0f},{comp_c:.0f},cov={cov_c:.0f},div={div_c:.0f})")

        results.append({
            "conv_id": conv_id,
            "query": query,
            "intent": intent,
            "vanilla_rel": rel_v,
            "vanilla_comp": comp_v,
            "chatgit_rel": rel_c,
            "chatgit_comp": comp_c,
            "vanilla_cov": cov_v,
            "vanilla_div": div_v,
            "chatgit_cov": cov_c,
            "chatgit_div": div_c,
        })
        time.sleep(1.0)  # rate limit (4 calls per turn)

    # Summary — per-chunk (top-1) metrics
    vanilla_rel  = np.mean([r["vanilla_rel"]  for r in results])
    vanilla_comp = np.mean([r["vanilla_comp"] for r in results])
    chatgit_rel  = np.mean([r["chatgit_rel"]  for r in results])
    chatgit_comp = np.mean([r["chatgit_comp"] for r in results])
    # Summary — set-level (top-5) metrics
    vanilla_cov = np.mean([r["vanilla_cov"] for r in results])
    vanilla_div = np.mean([r["vanilla_div"] for r in results])
    chatgit_cov = np.mean([r["chatgit_cov"] for r in results])
    chatgit_div = np.mean([r["chatgit_div"] for r in results])

    print("\n" + "=" * 68)
    print("  LLM-AS-JUDGE RESULTS (Llama-3.3-70B, n=30 turns)")
    print("=" * 68)
    print(f"  {'System':<20} {'Relevance':>10} {'Completeness':>14} {'Coverage':>10} {'Diversity':>10}")
    print("  " + "-" * 66)
    print(f"  {'VanillaRAG(BGE)':<20} {vanilla_rel:>10.2f}/5  {vanilla_comp:>10.2f}/5"
          f"  {vanilla_cov:>8.2f}/5  {vanilla_div:>8.2f}/5")
    print(f"  {'ChatGIT(Routed)':<20} {chatgit_rel:>10.2f}/5  {chatgit_comp:>10.2f}/5"
          f"  {chatgit_cov:>8.2f}/5  {chatgit_div:>8.2f}/5")
    print(f"  {'Delta':<20} {chatgit_rel-vanilla_rel:>+10.2f}    {chatgit_comp-vanilla_comp:>+10.2f}"
          f"    {chatgit_cov-vanilla_cov:>+6.2f}    {chatgit_div-vanilla_div:>+6.2f}")

    # Per-intent
    intents_seen = sorted(set(r["intent"] for r in results))
    print("\n  Per-intent (rel, comp, cov, div):")
    for intent in intents_seen:
        sub = [r for r in results if r["intent"] == intent]
        vr = np.mean([r["vanilla_rel"] for r in sub])
        vc = np.mean([r["vanilla_comp"] for r in sub])
        cr = np.mean([r["chatgit_rel"] for r in sub])
        cc = np.mean([r["chatgit_comp"] for r in sub])
        vd = np.mean([r["vanilla_div"] for r in sub])
        cd = np.mean([r["chatgit_div"] for r in sub])
        print(f"    {intent:<12} V=({vr:.2f},{vc:.2f},div={vd:.2f}) C=({cr:.2f},{cc:.2f},div={cd:.2f})  n={len(sub)}")

    os.makedirs("results", exist_ok=True)
    out = {
        "n_turns": len(results),
        "model": "llama-3.3-70b-versatile",
        "summary": {
            "vanilla_relevance": float(vanilla_rel),
            "vanilla_completeness": float(vanilla_comp),
            "chatgit_relevance": float(chatgit_rel),
            "chatgit_completeness": float(chatgit_comp),
            "vanilla_coverage": float(vanilla_cov),
            "vanilla_diversity": float(vanilla_div),
            "chatgit_coverage": float(chatgit_cov),
            "chatgit_diversity": float(chatgit_div),
        },
        "per_intent": {},
        "turns": results,
    }
    for intent in intents_seen:
        sub = [r for r in results if r["intent"] == intent]
        out["per_intent"][intent] = {
            "vanilla_relevance": float(np.mean([r["vanilla_rel"] for r in sub])),
            "chatgit_relevance": float(np.mean([r["chatgit_rel"] for r in sub])),
            "vanilla_completeness": float(np.mean([r["vanilla_comp"] for r in sub])),
            "chatgit_completeness": float(np.mean([r["chatgit_comp"] for r in sub])),
            "vanilla_diversity": float(np.mean([r["vanilla_div"] for r in sub])),
            "chatgit_diversity": float(np.mean([r["chatgit_div"] for r in sub])),
            "n": len(sub),
        }
    with open("results/llm_judge_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved -> results/llm_judge_results.json")
    print("=" * 68)
    return out


if __name__ == "__main__":
    main()
