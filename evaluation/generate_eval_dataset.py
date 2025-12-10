"""
Generate programmatic ConvCodeBench conversations from all benchmark repos.
Produces 8 conversations per repo × 5 repos = 40 new conversations appended
to data/convcodebench/sample_conversations.jsonl.

Each conversation follows a LOCATE → EXPLAIN → DEBUG template centred on a
single substantive function, with ground-truth chunk IDs extracted directly
from chunk_repository output (guaranteed exact GT matches).
"""

import sys, os, json, random, re
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch  # noqa: F401 — must come before sentence_transformers on Windows
import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4

from chatgit.core.chunker import chunk_repository

# ── Repo paths (same as run_convcodebench) ────────────────────────────────────
_REPO_BASE = os.environ.get("CHATGIT_REPO_BASE", "/tmp")
REPOS = {
    "flask":    os.environ.get("CHATGIT_REPO_FLASK",    os.path.join(_REPO_BASE, "flask_bench")),
    "requests": os.environ.get("CHATGIT_REPO_REQUESTS", os.path.join(_REPO_BASE, "requests_bench")),
    "click":    os.environ.get("CHATGIT_REPO_CLICK",    os.path.join(_REPO_BASE, "click_bench")),
    "fastapi":  os.environ.get("CHATGIT_REPO_FASTAPI",  os.path.join(_REPO_BASE, "fastapi_bench")),
    "celery":   os.environ.get("CHATGIT_REPO_CELERY",   os.path.join(_REPO_BASE, "celery_bench")),
}
REPOS = {k: v for k, v in REPOS.items() if os.path.isdir(v)}

DOMAIN_MAP = {
    "flask": "web_framework",
    "requests": "library",
    "click": "cli_tool",
    "fastapi": "web_framework",
    "celery": "devops",
}

CONVERSATIONS_PATH = os.path.join(
    _project_root, "data", "convcodebench", "sample_conversations.jsonl"
)

SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "example",
             "benchmarks", "scripts", "contrib", "extras", "tools",
             "__pycache__", "tutorial", "tutorials", "fixtures",
             "migrations", "compat", "vendor", "_vendor"}

SKIP_NAME_PATTERNS = [
    r"^__\w+__$",          # dunder methods
    r"^test_",             # test functions
    r"^setUp$", r"^tearDown$",
    r"^_$",                # single underscore
    r"^conftest",
    r"^module_level$",     # auto-generated module-level chunks
]

SKIP_FILE_PATTERNS = [
    r"test_\w+\.py$",      # test files by name
    r"conftest\.py$",
    r"setup\.py$", r"setup\.cfg$",
    r"CONTRIBUTING", r"CHANGELOG", r"README", r"LICENSE",
    r"\.md$", r"\.rst$", r"\.txt$",
    r"tutorial\d*",        # fastapi tutorial files
    r"_winconsole",        # platform-specific internals
    r"compat\.py$",
]

# Templates
LOCATE_T = [
    "Where is {func} defined in {repo}?",
    "Find the {func} function — which file is it in?",
    "Locate the implementation of {func} in the {repo} source.",
    "Which module contains {func}?",
    "Where can I find {func} in the codebase?",
]
EXPLAIN_T = [
    "How does {func} work?",
    "Explain what {func} does and how it's used.",
    "Walk me through the logic inside {func}.",
    "What is the purpose of {func}?",
    "How is {func} implemented under the hood?",
]
DEBUG_T = [
    "What could cause {func} to fail or raise an exception?",
    "Are there edge cases in {func} that could cause bugs?",
    "Why might {func} return an unexpected result?",
    "What happens in {func} when the input is None or invalid?",
    "How would I debug an issue originating from {func}?",
]
SUMMARIZE_T = [
    "Give me an overview of the {file} module in {repo}.",
    "Summarize the key classes and functions in {file}.",
    "What is the role of {file} in the {repo} architecture?",
]


def should_skip_name(name: str) -> bool:
    for pat in SKIP_NAME_PATTERNS:
        if re.match(pat, name):
            return True
    return False


def should_skip_file(fpath: str) -> bool:
    for pat in SKIP_FILE_PATTERNS:
        if re.search(pat, fpath, re.IGNORECASE):
            return True
    return False


def select_candidates(repo_id: str, repo_path: str, n: int = 8) -> list:
    """Return n diverse, substantive chunk objects from the repo."""
    docs = chunk_repository(repo_path)
    candidates = []
    for d in docs:
        meta = d.metadata
        fname = meta.get("file_name", "").replace("\\", "/")
        parts = set(fname.split("/"))
        if parts & SKIP_DIRS:
            continue
        if should_skip_file(fname):
            continue
        node_name = meta.get("node_name", "")
        node_type = meta.get("node_type", "")
        if node_type in ("module_summary",):
            continue
        if should_skip_name(node_name):
            continue
        text = d.text if hasattr(d, "text") else d.page_content
        if len(text.strip()) < 80:
            continue
        candidates.append({
            "chunk_id": f"{fname}::{node_name}",
            "file": fname,
            "basename": fname.split("/")[-1],
            "node_name": node_name,
            "node_type": node_type,
            "text": text,
        })

    # One candidate per file for diversity
    random.seed(repo_id)  # deterministic per repo
    random.shuffle(candidates)
    seen_files = set()
    selected = []
    for c in candidates:
        if c["file"] not in seen_files:
            selected.append(c)
            seen_files.add(c["file"])
        if len(selected) >= n:
            break
    # Fill from same file if not enough
    if len(selected) < n:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= n:
                break
    return selected[:n]


def make_conversation(repo_id: str, domain: str, chunk: dict,
                      conv_idx: int) -> dict:
    fn = chunk["node_name"]
    file_bn = chunk["basename"]
    conv_id = f"{repo_id}_conv_{conv_idx:03d}"

    locate_q  = random.choice(LOCATE_T).format(func=fn, repo=repo_id, file=file_bn)
    explain_q = random.choice(EXPLAIN_T).format(func=fn, repo=repo_id, file=file_bn)
    debug_q   = random.choice(DEBUG_T).format(func=fn, repo=repo_id, file=file_bn)

    turns = [
        {
            "turn_id": 0,
            "query": locate_q,
            "intent": "locate",
            "requires_context": False,
            "coreferences": [],
            "ground_truth_chunks": [chunk["chunk_id"]],
            "ground_truth_files":  [chunk["file"]],
            "reference_answer":
                f"`{fn}` is implemented in `{chunk['file']}`. "
                f"It is a {chunk['node_type']} in the {repo_id} library.",
            "context_snippets": [chunk["text"][:200]],
            "difficulty": "easy",
            "notes": "programmatic-auto",
        },
        {
            "turn_id": 1,
            "query": explain_q,
            "intent": "explain",
            "requires_context": True,
            "coreferences": [{"pronoun": "it", "referent": fn, "referent_turn_id": 0}],
            "ground_truth_chunks": [chunk["chunk_id"]],
            "ground_truth_files":  [chunk["file"]],
            "reference_answer":
                f"`{fn}` (in `{chunk['file']}`) — see the source for full details.",
            "context_snippets": [chunk["text"][:400]],
            "difficulty": "medium",
            "notes": "programmatic-auto",
        },
        {
            "turn_id": 2,
            "query": debug_q,
            "intent": "debug",
            "requires_context": True,
            "coreferences": [{"pronoun": "it", "referent": fn, "referent_turn_id": 0}],
            "ground_truth_chunks": [chunk["chunk_id"]],
            "ground_truth_files":  [chunk["file"]],
            "reference_answer":
                f"Examine `{fn}` in `{chunk['file']}` for edge-case handling and "
                f"exception paths.",
            "context_snippets": [],
            "difficulty": "hard",
            "notes": "programmatic-auto",
        },
    ]

    return {
        "conversation_id": conv_id,
        "repo_id": repo_id,
        "language": "python",
        "complexity_tier": "medium",
        "domain": domain,
        "turns": turns,
        "metadata": {
            "intent_sequence": ["locate", "explain", "debug"],
            "has_coreference": True,
            "topic_shift": False,
            "annotator_id": "AUTO",
            "annotation_date": "2026-03-19",
            "verified_by": "AUTO",
        },
    }


def load_existing_ids() -> set:
    existing = set()
    if not os.path.exists(CONVERSATIONS_PATH):
        return existing
    with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    c = json.loads(line)
                    existing.add(c.get("conversation_id", ""))
                except json.JSONDecodeError:
                    pass
    return existing


def main(n_per_repo: int = 8, dry_run: bool = False):
    print(f"Generating {n_per_repo} conversations per repo × {len(REPOS)} repos")
    existing_ids = load_existing_ids()
    print(f"Existing conversations: {len(existing_ids)}")

    new_convs = []
    for repo_id, repo_path in REPOS.items():
        domain = DOMAIN_MAP.get(repo_id, "library")
        print(f"\n[{repo_id}] Selecting candidates from {repo_path}...")
        candidates = select_candidates(repo_id, repo_path, n=n_per_repo)
        print(f"  Selected {len(candidates)} candidates")

        # Start conv_idx after existing convs for this repo
        existing_for_repo = sum(
            1 for cid in existing_ids if cid.startswith(f"{repo_id}_conv_")
        )
        for i, chunk in enumerate(candidates):
            idx = existing_for_repo + i + 1
            conv = make_conversation(repo_id, domain, chunk, idx)
            if conv["conversation_id"] in existing_ids:
                print(f"  SKIP (already exists): {conv['conversation_id']}")
                continue
            new_convs.append(conv)
            print(f"  + {conv['conversation_id']}  [{chunk['node_name']}]  ({chunk['file'].split('/')[-1]})")

    print(f"\nTotal new conversations: {len(new_convs)}")
    if dry_run:
        print("DRY RUN — not writing anything.")
        return

    if new_convs:
        with open(CONVERSATIONS_PATH, "a", encoding="utf-8") as f:
            for conv in new_convs:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        print(f"Appended {len(new_convs)} conversations to {CONVERSATIONS_PATH}")
    else:
        print("Nothing to append.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="conversations per repo")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(n_per_repo=args.n, dry_run=args.dry_run)
