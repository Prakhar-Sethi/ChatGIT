"""
Build a proper multi-turn evaluation dataset from real repository code.

For each repo we:
  1. Parse AST to find real function/class nodes
  2. Generate questions of 4 intent types from actual code
  3. Group into multi-turn conversations (3-4 turns each)
  4. Record ground-truth chunk IDs

Outputs: data/convcodebench/eval_conversations.jsonl
"""

import ast
import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

random.seed(42)

_REPO_BASE = os.environ.get("CHATGIT_REPO_BASE", "/tmp")
REPOS = {
    "flask":    os.environ.get("CHATGIT_REPO_FLASK",    os.path.join(_REPO_BASE, "flask_bench")),
    "requests": os.environ.get("CHATGIT_REPO_REQUESTS", os.path.join(_REPO_BASE, "requests_bench")),
    "click":    os.environ.get("CHATGIT_REPO_CLICK",    os.path.join(_REPO_BASE, "click_bench")),
    "fastapi":  os.environ.get("CHATGIT_REPO_FASTAPI",  os.path.join(_REPO_BASE, "fastapi_bench")),
    "celery":   os.environ.get("CHATGIT_REPO_CELERY",   os.path.join(_REPO_BASE, "celery_bench")),
}

OUTPUT_PATH = "data/convcodebench/eval_conversations.jsonl"

# ── Python AST node extractor ──────────────────────────────────────────────

def extract_python_nodes(repo_path: str, skip_dirs=None) -> List[Dict]:
    """Return list of {file, name, type, docstring, calls, lineno} from repo."""
    if skip_dirs is None:
        skip_dirs = {"tests", "test", "docs", "doc", "examples", "migrations",
                     "__pycache__", ".git", "build", "dist", "node_modules"}
    nodes = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            rel   = os.path.relpath(fpath, repo_path)
            try:
                src = open(fpath, encoding="utf-8", errors="ignore").read()
                tree = ast.parse(src)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc  = ast.get_docstring(node) or ""
                    args = [a.arg for a in node.args.args]
                    # collect called names
                    calls = []
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                calls.append(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                calls.append(child.func.attr)
                    nodes.append({
                        "file":     rel,
                        "name":     node.name,
                        "type":     "function",
                        "args":     args,
                        "doc":      doc[:200],
                        "calls":    list(set(calls))[:6],
                        "lineno":   node.lineno,
                        "chunk_id": f"{rel}::{node.name}",
                    })
                elif isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    methods = [n.name for n in ast.walk(node)
                               if isinstance(n, (ast.FunctionDef,
                                                 ast.AsyncFunctionDef))][:6]
                    nodes.append({
                        "file":     rel,
                        "name":     node.name,
                        "type":     "class",
                        "args":     [],
                        "doc":      doc[:200],
                        "calls":    methods,
                        "lineno":   node.lineno,
                        "chunk_id": f"{rel}::{node.name}",
                    })
    return nodes


# ── Question templates ─────────────────────────────────────────────────────

LOCATE_TEMPLATES = [
    "Where is `{name}` defined?",
    "Which file contains the `{name}` {type}?",
    "Find the definition of `{name}`.",
    "Where is `{name}` implemented in the codebase?",
    "What file and line defines `{name}`?",
]

EXPLAIN_TEMPLATES = [
    "How does `{name}` work?",
    "Explain what `{name}` does.",
    "What is the purpose of `{name}`?",
    "Walk me through the logic of `{name}`.",
    "What does `{name}` return and when?",
]

EXPLAIN_FOLLOWUP_TEMPLATES = [
    "What does it call internally?",
    "How does it handle errors?",
    "What are its arguments?",
    "Can you explain the return value?",
    "What calls this function?",
]

DEBUG_TEMPLATES = [
    "What could go wrong when calling `{name}`?",
    "What exceptions can `{name}` raise?",
    "How would I debug an issue in `{name}`?",
    "What are the edge cases in `{name}`?",
]

SUMMARIZE_TEMPLATES = [
    "Give me an overview of `{file_module}`.",
    "What does the `{file_module}` module do?",
    "Summarise the key classes in `{file_module}`.",
    "What is the responsibility of `{file_module}`?",
]

COREF_PRONOUNS = ["it", "this function", "the function", "that", "this class"]


def make_question(template: str, node: Dict) -> str:
    file_module = Path(node["file"]).stem
    return template.format(
        name=node["name"],
        type=node["type"],
        file_module=file_module,
    )


# ── Conversation builder ────────────────────────────────────────────────────

def build_conversations(repo_id: str, nodes: List[Dict],
                        n_convs: int = 20) -> List[Dict]:
    """
    Build n_convs multi-turn conversations.
    Each conversation: 3-4 turns covering LOCATE → EXPLAIN → DEBUG or
    EXPLAIN → FOLLOWUP → SUMMARIZE patterns.
    """
    if not nodes:
        return []

    # Keep only non-trivial nodes (has a docstring or has calls)
    useful = [n for n in nodes
              if (n["doc"] or n["calls"]) and len(n["name"]) > 2
              and not n["name"].startswith("_")]
    if not useful:
        useful = nodes

    random.shuffle(useful)
    conversations = []
    conv_idx = 0

    # Pattern 1: LOCATE → EXPLAIN (with pronoun) → DEBUG
    for node in useful[:n_convs // 2]:
        turns = []
        # Turn 0: locate
        q0 = make_question(random.choice(LOCATE_TEMPLATES), node)
        turns.append({
            "turn_id":   0,
            "query":     q0,
            "intent":    "locate",
            "requires_context": False,
            "coreferences": [],
            "ground_truth_chunks": [node["chunk_id"]],
            "ground_truth_files":  [node["file"]],
        })
        # Turn 1: explain (may use pronoun coreference)
        use_coref = random.random() < 0.5
        if use_coref:
            pronoun = random.choice(COREF_PRONOUNS)
            q1 = f"How does {pronoun} work?"
            coref = [{"pronoun": pronoun, "referent": node["name"],
                      "referent_turn_id": 0}]
        else:
            q1 = make_question(random.choice(EXPLAIN_TEMPLATES), node)
            coref = []
        turns.append({
            "turn_id":   1,
            "query":     q1,
            "intent":    "explain",
            "requires_context": use_coref,
            "coreferences": coref,
            "ground_truth_chunks": [node["chunk_id"]],
            "ground_truth_files":  [node["file"]],
        })
        # Turn 2: debug or followup
        q2 = make_question(random.choice(DEBUG_TEMPLATES), node)
        turns.append({
            "turn_id":   2,
            "query":     q2,
            "intent":    "debug",
            "requires_context": False,
            "coreferences": [],
            "ground_truth_chunks": [node["chunk_id"]],
            "ground_truth_files":  [node["file"]],
        })
        # Turn 3 (optional): summarize the file
        if random.random() < 0.6:
            file_module = Path(node["file"]).stem
            q3 = random.choice(SUMMARIZE_TEMPLATES).format(
                name=node["name"], type=node["type"],
                file_module=file_module)
            # GT for summarize = module summary chunk
            sum_gt = f"{node['file']}::__module_summary__"
            turns.append({
                "turn_id":   3,
                "query":     q3,
                "intent":    "summarize",
                "requires_context": False,
                "coreferences": [],
                "ground_truth_chunks": [sum_gt, node["chunk_id"]],
                "ground_truth_files":  [node["file"]],
            })
        conv_idx += 1
        conversations.append({
            "conversation_id": f"{repo_id}_conv_{conv_idx:03d}",
            "repo_id":   repo_id,
            "language":  "python",
            "turns":     turns,
        })

    # Pattern 2: EXPLAIN → FOLLOWUP (again?) → SUMMARIZE
    for node in useful[n_convs // 2: n_convs]:
        turns = []
        q0 = make_question(random.choice(EXPLAIN_TEMPLATES), node)
        turns.append({
            "turn_id":   0,
            "query":     q0,
            "intent":    "explain",
            "requires_context": False,
            "coreferences": [],
            "ground_truth_chunks": [node["chunk_id"]],
            "ground_truth_files":  [node["file"]],
        })
        q1 = random.choice(EXPLAIN_FOLLOWUP_TEMPLATES)
        turns.append({
            "turn_id":   1,
            "query":     q1,
            "intent":    "explain",
            "requires_context": True,
            "coreferences": [{"pronoun": "it", "referent": node["name"],
                               "referent_turn_id": 0}],
            "ground_truth_chunks": [node["chunk_id"]],
            "ground_truth_files":  [node["file"]],
        })
        if node["calls"]:
            callee = node["calls"][0]
            q2 = f"Where is `{callee}` defined?"
            turns.append({
                "turn_id":   2,
                "query":     q2,
                "intent":    "locate",
                "requires_context": False,
                "coreferences": [],
                "ground_truth_chunks": [f"::{callee}"],  # fuzzy match
                "ground_truth_files":  [],
            })
        else:
            q2 = make_question(random.choice(DEBUG_TEMPLATES), node)
            turns.append({
                "turn_id":   2,
                "query":     q2,
                "intent":    "debug",
                "requires_context": False,
                "coreferences": [],
                "ground_truth_chunks": [node["chunk_id"]],
                "ground_truth_files":  [node["file"]],
            })
        conv_idx += 1
        conversations.append({
            "conversation_id": f"{repo_id}_conv_{conv_idx:03d}",
            "repo_id":   repo_id,
            "language":  "python",
            "turns":     turns,
        })

    return conversations


def main():
    all_convs = []
    for repo_id, repo_path in REPOS.items():
        print(f"  Parsing {repo_id}...", end=" ", flush=True)
        nodes = extract_python_nodes(repo_path)
        print(f"{len(nodes)} nodes", end=" → ", flush=True)
        convs = build_conversations(repo_id, nodes, n_convs=30)
        print(f"{len(convs)} conversations")
        all_convs.extend(convs)

    random.shuffle(all_convs)
    os.makedirs("data/convcodebench", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for c in all_convs:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(all_convs)} conversations → {OUTPUT_PATH}")
    # Intent distribution
    from collections import Counter
    ic = Counter()
    for c in all_convs:
        for t in c["turns"]:
            ic[t["intent"]] += 1
    print(f"Intent dist: {dict(ic)}")


if __name__ == "__main__":
    main()
