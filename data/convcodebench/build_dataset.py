"""
ConvCodeBench Dataset Construction Script.

Pipeline:
  1. Clone repositories at pinned commit hashes (from repo_manifest.json)
  2. Run ChatGIT indexing on each repo to get chunk IDs
  3. Generate conversation seeds using templates per intent
  4. Export to JSONL in the schema defined by schema.json

Usage:
    python data/convcodebench/build_dataset.py \
        --manifest data/convcodebench/repo_manifest.json \
        --output   data/convcodebench/conversations.jsonl \
        --n_convs_per_repo 20 \
        --clone_dir /tmp/convcodebench_repos
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
import re


# ---------------------------------------------------------------------------
# Conversation templates by intent
# ---------------------------------------------------------------------------

LOCATE_TEMPLATES = [
    "Where is {entity} defined?",
    "Which file contains the implementation of {entity}?",
    "Find the definition of {entity}.",
    "What line is {entity} declared on?",
    "Where can I find the {entity} function?",
    "Which file implements {entity}?",
    "Locate the {entity} class for me.",
    "Show me where {entity} is implemented.",
]

EXPLAIN_TEMPLATES = [
    "How does {entity} work?",
    "What does {entity} do?",
    "Explain the purpose of {entity}.",
    "Walk me through {entity}.",
    "What is the role of {entity}?",
    "How is {entity} used in this codebase?",
    "Describe how {entity} handles {aspect}.",
    "What happens when {entity} is called with {aspect}?",
]

SUMMARIZE_TEMPLATES = [
    "Give me an overview of the {module} module.",
    "Summarize the architecture of {repo}.",
    "What are the main components of {module}?",
    "Explain the high-level structure of {repo}.",
    "What does the {module} module contain?",
    "How do the components of {repo} work together?",
    "Give me a big-picture view of {module}.",
]

DEBUG_TEMPLATES = [
    "Why does {entity} fail when {condition}?",
    "There's a bug in {entity} — it doesn't handle {condition} correctly.",
    "Why is {entity} not working when {condition}?",
    "I'm getting an error in {entity}: {error_msg}. How do I fix it?",
    "What's wrong with {entity} when {condition}?",
    "Debug {entity} — it crashes on {condition}.",
]

FOLLOWUP_TEMPLATES = {
    "locate": [
        "How does {entity} work exactly?",
        "What calls {entity}?",
        "Where is {entity} used in the rest of the codebase?",
    ],
    "explain": [
        "What happens if {aspect} is missing?",
        "Can you show me an example?",
        "How does this interact with {other_entity}?",
    ],
    "summarize": [
        "What is the role of {entity} specifically?",
        "How does {entity} fit into the architecture?",
        "Tell me more about {entity}.",
    ],
    "debug": [
        "What is the root cause?",
        "How would you fix it?",
        "Are there similar bugs elsewhere in the codebase?",
    ],
}

COREFERENCE_REWRITES = {
    "locate":    ["it", "the function", "this"],
    "explain":   ["it", "this", "the method", "that class"],
    "summarize": ["the module", "this component", "they"],
    "debug":     ["it", "this bug", "the issue"],
}


# ---------------------------------------------------------------------------
# Repository cloning
# ---------------------------------------------------------------------------

def clone_repo(url: str, commit: str, dest_dir: str) -> Optional[str]:
    """
    Clone a repo at a specific commit hash or tag.
    Returns the path to the cloned directory, or None on failure.
    """
    repo_name = url.rstrip("/").split("/")[-1]
    clone_path = os.path.join(dest_dir, repo_name)

    if os.path.exists(clone_path):
        print(f"  [build] Already exists: {clone_path}")
        return clone_path

    try:
        print(f"  [build] Cloning {url}...")
        subprocess.run(
            ["git", "clone", "--depth=1", f"--branch={commit}", url, clone_path],
            check=True, capture_output=True, timeout=120
        )
        return clone_path
    except subprocess.CalledProcessError:
        # Try cloning main and checking out the commit
        try:
            subprocess.run(
                ["git", "clone", url, clone_path],
                check=True, capture_output=True, timeout=300
            )
            subprocess.run(
                ["git", "-C", clone_path, "checkout", commit],
                check=True, capture_output=True, timeout=30
            )
            return clone_path
        except Exception as e:
            print(f"  [build] ERROR cloning {url}: {e}")
            return None


# ---------------------------------------------------------------------------
# Entity extraction from repo
# ---------------------------------------------------------------------------

def extract_entities(repo_path: str, language: str) -> Dict[str, List[str]]:
    """
    Extract named entities (functions, classes, modules) from a repository.
    Returns {functions: [...], classes: [...], modules: [...]}
    """
    functions, classes, modules = [], [], []

    if language == "python":
        import ast as ast_mod
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {
                'venv', '__pycache__', '.git', 'node_modules', 'build', 'dist'
            }]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                modules.append(fname.replace(".py", ""))
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                    tree = ast_mod.parse(source)
                    for node in ast_mod.walk(tree):
                        if isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef)):
                            functions.append(node.name)
                        elif isinstance(node, ast_mod.ClassDef):
                            classes.append(node.name)
                except Exception:
                    pass
    else:
        # Generic: scan for common patterns
        ext_map = {
            "javascript": ".js", "typescript": ".ts",
            "java": ".java", "go": ".go"
        }
        ext = ext_map.get(language, ".py")
        func_re = re.compile(r'\b(?:function|def|func)\s+(\w+)\s*\(', re.M)
        class_re = re.compile(r'\bclass\s+(\w+)', re.M)
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', 'build'}]
            for fname in files:
                if not fname.endswith(ext):
                    continue
                fpath = os.path.join(root, fname)
                modules.append(fname.split(".")[0])
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        src = f.read()
                    functions.extend(func_re.findall(src))
                    classes.extend(class_re.findall(src))
                except Exception:
                    pass

    # Deduplicate and filter trivial names
    def _filter(names):
        skip = {'__init__', 'test', 'main', 'setup', 'teardown', 'helper', 'util', 'base'}
        seen = set()
        result = []
        for n in names:
            if n.lower() not in skip and n not in seen and len(n) > 2:
                seen.add(n)
                result.append(n)
        return result

    return {
        "functions": _filter(functions)[:200],
        "classes":   _filter(classes)[:100],
        "modules":   _filter(list(set(modules)))[:50],
    }


# ---------------------------------------------------------------------------
# Conversation generation
# ---------------------------------------------------------------------------

def generate_conversation(
    repo_id: str,
    repo_url: str,
    commit: str,
    language: str,
    domain: str,
    complexity: str,
    entities: Dict[str, List[str]],
    conv_index: int,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """
    Generate a single multi-turn conversation from extracted entities.
    """
    functions = entities.get("functions", [])
    classes   = entities.get("classes",   [])
    modules   = entities.get("modules",   [])

    if not functions and not classes:
        return None

    # Pick intents for a 2-4 turn conversation
    n_turns = rng.randint(2, 4)
    intent_patterns = [
        ["locate", "explain"],
        ["locate", "explain", "debug"],
        ["summarize", "explain", "explain"],
        ["explain", "debug"],
        ["locate", "explain", "explain", "debug"],
        ["summarize", "explain", "debug"],
    ]
    intents = rng.choice(intent_patterns)[:n_turns]

    turns = []
    prev_entity = None
    has_coreference = False

    for ti, intent in enumerate(intents):
        entity = rng.choice(functions + classes) if (functions or classes) else "module"
        module = rng.choice(modules) if modules else repo_id
        aspect = rng.choice(["None values", "empty input", "concurrent calls", "large data"])
        error_msg = rng.choice(["TypeError", "ValueError", "AttributeError", "IndexError"])
        other_entity = rng.choice(functions + classes) if len(functions) > 1 else "the rest of the code"

        # Decide whether to use coreference
        use_coref = ti > 0 and prev_entity and rng.random() < 0.4

        template_map = {
            "locate":    LOCATE_TEMPLATES,
            "explain":   EXPLAIN_TEMPLATES,
            "summarize": SUMMARIZE_TEMPLATES,
            "debug":     DEBUG_TEMPLATES,
        }
        templates = template_map[intent]
        template = rng.choice(templates)

        try:
            if use_coref:
                coref_word = rng.choice(COREFERENCE_REWRITES[intent])
                if intent == "locate":
                    query = f"Where is {coref_word} used in the codebase?"
                elif intent in ("explain", "debug"):
                    query = template.format(
                        entity=coref_word, aspect=aspect, condition=aspect,
                        error_msg=error_msg, module=module, repo=repo_id,
                        other_entity=other_entity
                    )
                else:
                    query = template.format(
                        entity=coref_word, module=module, repo=repo_id
                    )
                coreferences = [{"pronoun": coref_word, "referent": prev_entity, "referent_turn_id": ti - 1}]
                has_coreference = True
            else:
                query = template.format(
                    entity=entity, aspect=aspect, condition=aspect,
                    error_msg=error_msg, module=module, repo=repo_id,
                    other_entity=other_entity
                )
                coreferences = []
        except KeyError:
            query = template.replace("{entity}", entity).replace("{module}", module).replace("{repo}", repo_id)
            coreferences = []

        # Generate placeholder chunk IDs — to be filled by annotators
        chunk_id = f"{repo_id}/{rng.choice(['src', 'lib', ''])}/{entity.lower()}.py::{entity}".strip("/")
        turns.append({
            "turn_id":            ti,
            "query":              query,
            "intent":             intent,
            "requires_context":   use_coref or (ti > 0 and rng.random() < 0.3),
            "coreferences":       coreferences,
            "ground_truth_chunks": [chunk_id],
            "ground_truth_files":  [chunk_id.split("::")[0]],
            "reference_answer":    f"[TO BE ANNOTATED] Answer about {entity} in {repo_id}",
            "context_snippets":    [],
            "difficulty":          rng.choice(["easy", "medium", "hard"]),
            "notes":               "",
        })
        prev_entity = entity

    intent_sequence = [t["intent"] for t in turns]
    has_topic_shift = len(set(t["intent"] for t in turns)) == len(turns)

    return {
        "conversation_id": f"{repo_id}_conv_{conv_index:03d}",
        "repo_id":         repo_id,
        "repo_url":        repo_url,
        "commit_hash":     commit,
        "language":        language,
        "complexity_tier": complexity,
        "domain":          domain,
        "turns":           turns,
        "metadata": {
            "intent_sequence": intent_sequence,
            "has_coreference": has_coreference,
            "topic_shift":     has_topic_shift,
            "annotator_id":    "SEED",
            "annotation_date": "2025-10-01",
            "verified_by":     "PENDING",
        },
    }


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_dataset(
    manifest_path: str,
    output_path: str,
    n_convs_per_repo: int = 20,
    clone_dir: Optional[str] = None,
    max_repos: Optional[int] = None,
    seed: int = 42,
) -> None:
    """
    Build the ConvCodeBench dataset.
    """
    rng = random.Random(seed)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    repos = manifest["repositories"]
    if max_repos:
        repos = repos[:max_repos]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for repo in repos:
            repo_id    = repo["id"]
            repo_url   = repo["url"]
            commit     = repo["commit"]
            language   = repo["language"]
            domain     = repo["domain"]
            complexity = repo["complexity"]

            print(f"\n[build] Processing {repo_id} ({language}, {complexity})...")

            # Try to clone or use existing path
            repo_path = None
            if clone_dir:
                cloned = clone_repo(repo_url, commit, clone_dir)
                if cloned:
                    repo_path = cloned

            # Extract entities (from cloned repo if available, otherwise generate stubs)
            if repo_path and os.path.exists(repo_path):
                entities = extract_entities(repo_path, language)
                print(f"  [build] Extracted {len(entities['functions'])} functions, "
                      f"{len(entities['classes'])} classes, {len(entities['modules'])} modules")
            else:
                print(f"  [build] No local repo — generating stub entities for {repo_id}")
                entities = {
                    "functions": [f"process_{repo_id}", f"handle_request", f"validate_input",
                                  f"parse_config", f"setup_{repo_id}", f"run", f"execute",
                                  f"create_{repo_id.replace('-', '_')}", f"load", f"save"],
                    "classes":   [f"{repo_id.title().replace('-', '')}Client",
                                  f"BaseHandler", f"Config", f"Manager"],
                    "modules":   [repo_id.replace("-", "_"), "utils", "core", "config", "models"],
                }

            # Generate conversations
            n_generated = 0
            for ci in range(n_convs_per_repo * 3):  # Over-generate to handle failures
                if n_generated >= n_convs_per_repo:
                    break
                conv = generate_conversation(
                    repo_id, repo_url, commit, language, domain, complexity,
                    entities, ci, rng
                )
                if conv:
                    out_f.write(json.dumps(conv) + "\n")
                    n_generated += 1
                    total_written += 1

            print(f"  [build] Generated {n_generated} conversations for {repo_id}")

    print(f"\n[build] Dataset construction complete.")
    print(f"[build] Total conversations written: {total_written}")
    print(f"[build] Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ConvCodeBench dataset")
    parser.add_argument("--manifest",          default="data/convcodebench/repo_manifest.json")
    parser.add_argument("--output",            default="data/convcodebench/conversations_seed.jsonl")
    parser.add_argument("--n_convs_per_repo",  type=int, default=20)
    parser.add_argument("--clone_dir",         default=None,
                        help="Directory to clone repos into. If None, uses stubs.")
    parser.add_argument("--max_repos",         type=int, default=None)
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        manifest_path=args.manifest,
        output_path=args.output,
        n_convs_per_repo=args.n_convs_per_repo,
        clone_dir=args.clone_dir,
        max_repos=args.max_repos,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
