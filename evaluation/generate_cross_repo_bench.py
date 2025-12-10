"""
Generate single-turn LOCATE + SUMMARIZE conversations with
programmatic ground truth for new repos.

Usage:
    python -m evaluation.generate_cross_repo_bench \
        --repos tornado_bench scrapy_bench \
        --n_locate 40 --n_summarize 15 \
        --out data/convcodebench/cross_repo_conversations.jsonl
"""

import sys, os, json, argparse, random
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chatgit.core.chunker as _ck
_ck._count_tokens = lambda text: len(text) // 4
from chatgit.core.chunker import chunk_repository

SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "example",
             "__pycache__", ".git", "build", "dist", "benchmarks"}

LOCATE_TEMPLATES = [
    "Where is the {name} function defined?",
    "Where is {name} implemented in the codebase?",
    "Find the definition of {name}.",
    "Which file contains the {name} function?",
    "Locate the {name} function for me.",
]

SUMMARIZE_TEMPLATES = [
    "Give me an overview of the {name} class.",
    "Summarize what the {name} class does.",
    "What is the purpose of the {name} class?",
    "Explain the {name} module at a high level.",
    "What does the {name} class provide?",
]


def get_repo_id(repo_path):
    return os.path.basename(repo_path.rstrip("/\\")).replace("_bench", "")


def build_conversations(repo_path, n_locate=40, n_summarize=15, seed=42):
    random.seed(seed)
    repo_id = get_repo_id(repo_path)

    docs = chunk_repository(repo_path)
    functions, classes = [], []
    for d in docs:
        fname = d.metadata.get("file_name", "").replace("\\", "/")
        parts = set(fname.split("/"))
        if parts & SKIP_DIRS:
            continue
        nt = d.metadata.get("node_type", "")
        nn = d.metadata.get("node_name", "")
        if not nn or nn.startswith("_") or len(nn) < 3:
            continue
        text = d.text if hasattr(d, "text") else d.page_content
        cid = f"{fname}::{nn}"
        entry = {"cid": cid, "name": nn, "file": fname, "text_len": len(text)}
        if nt == "function" and len(text) > 100:
            functions.append(entry)
        elif nt in ("class", "module_summary") and len(text) > 50:
            classes.append(entry)

    # Deduplicate by name (keep longest)
    by_name = {}
    for e in functions:
        if e["name"] not in by_name or e["text_len"] > by_name[e["name"]]["text_len"]:
            by_name[e["name"]] = e
    functions = sorted(by_name.values(), key=lambda x: -x["text_len"])

    by_name = {}
    for e in classes:
        if e["name"] not in by_name or e["text_len"] > by_name[e["name"]]["text_len"]:
            by_name[e["name"]] = e
    classes = sorted(by_name.values(), key=lambda x: -x["text_len"])

    # Sample top-N (most substantial)
    sel_funcs = functions[:min(n_locate, len(functions))]
    sel_classes = classes[:min(n_summarize, len(classes))]
    random.shuffle(sel_funcs)
    random.shuffle(sel_classes)

    convs = []

    for i, e in enumerate(sel_funcs):
        tmpl = LOCATE_TEMPLATES[i % len(LOCATE_TEMPLATES)]
        query = tmpl.format(name=e["name"])
        conv_id = f"{repo_id}_locate_{i:03d}"
        convs.append({
            "conversation_id": conv_id,
            "repo_id": repo_id,
            "language": "python",
            "complexity_tier": "tier1",
            "domain": "programmatic_locate",
            "turns": [{
                "turn_id": 0,
                "query": query,
                "intent": "locate",
                "requires_context": False,
                "coreferences": [],
                "ground_truth_chunks": [e["cid"]],
                "ground_truth_files": [e["file"]],
                "difficulty": "medium",
            }],
            "metadata": {
                "annotator_id": "AUTO_PROG",
                "intent_sequence": ["locate"],
                "has_coreference": False,
                "topic_shift": False,
            }
        })

    for i, e in enumerate(sel_classes):
        tmpl = SUMMARIZE_TEMPLATES[i % len(SUMMARIZE_TEMPLATES)]
        query = tmpl.format(name=e["name"])
        conv_id = f"{repo_id}_summarize_{i:03d}"
        convs.append({
            "conversation_id": conv_id,
            "repo_id": repo_id,
            "language": "python",
            "complexity_tier": "tier3",
            "domain": "programmatic_summarize",
            "turns": [{
                "turn_id": 0,
                "query": query,
                "intent": "summarize",
                "requires_context": False,
                "coreferences": [],
                "ground_truth_chunks": [e["cid"]],
                "ground_truth_files": [e["file"]],
                "difficulty": "medium",
            }],
            "metadata": {
                "annotator_id": "AUTO_PROG",
                "intent_sequence": ["summarize"],
                "has_coreference": False,
                "topic_shift": False,
            }
        })

    return convs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", required=True)
    parser.add_argument("--n_locate", type=int, default=40)
    parser.add_argument("--n_summarize", type=int, default=15)
    parser.add_argument("--out", default="data/convcodebench/cross_repo_conversations.jsonl")
    args = parser.parse_args()

    all_convs = []
    for repo_path in args.repos:
        if not os.path.isdir(repo_path):
            print(f"  SKIP {repo_path} (not found)")
            continue
        print(f"  Processing {repo_path}...", flush=True)
        convs = build_conversations(repo_path, args.n_locate, args.n_summarize)
        print(f"    -> {len(convs)} conversations ({args.n_locate} LOCATE + {args.n_summarize} SUMMARIZE)")
        all_convs.extend(convs)

    out_path = os.path.join(_project_root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_convs:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(all_convs)} conversations -> {out_path}")
    n_turns = sum(len(c["turns"]) for c in all_convs)
    print(f"Total turns: {n_turns}")
    intents = {}
    for c in all_convs:
        for t in c["turns"]:
            intents[t["intent"]] = intents.get(t["intent"], 0) + 1
    print(f"Intent dist: {intents}")


if __name__ == "__main__":
    main()
