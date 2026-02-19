"""
Token-aware AST chunker for ChatGIT.

Creates semantically-meaningful chunks from code files that:
- Respect function and class boundaries (using Python AST, regex for others)
- Include rich metadata: file path, start_line, end_line, node_type, node_name
- Split large nodes into overlapping sub-chunks (max 6 per node)
- Use token counts for consistent sizing
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from llama_index.core import Document          # llama_index >= 0.10 (namespaced packages)
except (ImportError, OSError):
    try:
        from llama_index import Document           # llama_index < 0.10 (monolithic package)
    except (ImportError, OSError):
        class Document:                            # minimal shim when llama_index unavailable
            """Minimal Document shim — same interface used by chunker.py."""
            __slots__ = ("text", "metadata")
            def __init__(self, text: str = "", metadata: dict = None):
                self.text = text
                self.metadata = metadata or {}
            @property
            def page_content(self) -> str:
                return self.text

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_TOKENIZER.encode(text))
except Exception:
    def _count_tokens(text: str) -> int:
        return len(text) // 4


MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64
MAX_CHUNKS_PER_NODE = 6

SKIP_DIRS = {'venv', '__pycache__', '.git', 'node_modules', '.venv', 'build', 'dist', '.idea', '.vscode'}
PYTHON_EXTS = {'.py'}
OTHER_CODE_EXTS = {'.js', '.jsx', '.ts', '.tsx', '.java', '.swift', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'}
TEXT_EXTS = {'.md', '.txt', '.rst'}


def _split_lines_into_chunks(
    lines: List[str],
    start_line: int,
    file_path: str,
    node_type: str,
    node_name: str,
) -> List[Document]:
    """Split a list of source lines into token-bounded, overlapping Document chunks."""
    if not lines:
        return []

    chunks = []
    current_lines: List[str] = []
    current_tokens = 0
    chunk_start = start_line

    for line in lines:
        line_tokens = _count_tokens(line)

        if current_tokens + line_tokens > MAX_CHUNK_TOKENS and current_lines:
            # Emit chunk
            text = "".join(current_lines)
            chunks.append(Document(
                text=text,
                metadata={
                    "file_name": file_path,
                    "start_line": chunk_start,
                    "end_line": chunk_start + len(current_lines) - 1,
                    "node_type": node_type,
                    "node_name": node_name,
                    "chunk_index": len(chunks),
                }
            ))

            if len(chunks) >= MAX_CHUNKS_PER_NODE:
                return chunks

            # Build overlap from tail of current chunk
            overlap: List[str] = []
            overlap_tokens = 0
            for ol in reversed(current_lines):
                ot = _count_tokens(ol)
                if overlap_tokens + ot <= OVERLAP_TOKENS:
                    overlap.insert(0, ol)
                    overlap_tokens += ot
                else:
                    break

            chunk_start = chunk_start + len(current_lines) - len(overlap)
            current_lines = overlap + [line]
            current_tokens = overlap_tokens + line_tokens
        else:
            current_lines.append(line)
            current_tokens += line_tokens

    # Emit final chunk
    if current_lines and len(chunks) < MAX_CHUNKS_PER_NODE:
        text = "".join(current_lines)
        chunks.append(Document(
            text=text,
            metadata={
                "file_name": file_path,
                "start_line": chunk_start,
                "end_line": chunk_start + len(current_lines) - 1,
                "node_type": node_type,
                "node_name": node_name,
                "chunk_index": len(chunks),
            }
        ))

    return chunks


def _make_module_summary(relative_path: str, functions: List[str],
                         classes: List[str], imports: List[str]) -> Optional[Document]:
    """
    Generate a module-level summary Document for a source file.
    Stored with node_type='module_summary' — boosted by the SUMMARIZE intent.
    """
    if not functions and not classes:
        return None

    parts = [f"# Module Summary: {relative_path}"]
    if functions:
        parts.append(
            f"## Functions ({len(functions)})\n"
            + "\n".join(f"- {f}" for f in functions[:40])
        )
    if classes:
        parts.append(
            "## Classes\n" + "\n".join(f"- {c}" for c in classes[:20])
        )
    if imports:
        unique_imports = list(dict.fromkeys(imports))[:20]
        parts.append("## Imports\n" + "\n".join(f"- {i}" for i in unique_imports))

    return Document(
        text="\n\n".join(parts),
        metadata={
            "file_name": relative_path,
            "start_line": 1,
            "end_line": 0,
            "node_type": "module_summary",
            "node_name": f"{Path(relative_path).stem}_summary",
            "chunk_index": 0,
        },
    )


def chunk_python_file(file_path: str, relative_path: str) -> List[Document]:
    """Chunk a Python file using the AST for semantic boundaries."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=relative_path)
    except SyntaxError:
        return chunk_generic_file(file_path, relative_path)
    except Exception as e:
        print(f"[Chunker] Error reading {file_path}: {e}")
        return []

    all_lines = source.splitlines(keepends=True)
    chunks: List[Document] = []
    covered: set = set()

    fn_names:  List[str] = []
    cls_names: List[str] = []
    imports:   List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # ast.end_lineno available in Python 3.8+
            if not hasattr(node, 'end_lineno'):
                continue
            start_idx = node.lineno - 1
            end_idx = node.end_lineno  # exclusive
            node_lines = all_lines[start_idx:end_idx]
            node_type = "class" if isinstance(node, ast.ClassDef) else "function"
            node_chunks = _split_lines_into_chunks(
                node_lines,
                start_line=node.lineno,
                file_path=relative_path,
                node_type=node_type,
                node_name=node.name,
            )
            chunks.extend(node_chunks)
            covered.update(range(start_idx, end_idx))

            if isinstance(node, ast.ClassDef):
                cls_names.append(node.name)
            else:
                doc = ast.get_docstring(node) or ""
                first_doc = doc.split("\n")[0][:80] if doc else ""
                fn_names.append(f"{node.name}(): {first_doc}" if first_doc else node.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    # Module-level code not inside any function/class
    module_lines = [(i, l) for i, l in enumerate(all_lines) if i not in covered]
    if module_lines:
        first_idx = module_lines[0][0]
        just_lines = [l for _, l in module_lines]
        module_chunks = _split_lines_into_chunks(
            just_lines,
            start_line=first_idx + 1,
            file_path=relative_path,
            node_type="module",
            node_name="module_level",
        )
        chunks.extend(module_chunks)

    # Novelty 4: append module-level summary chunk
    summary = _make_module_summary(relative_path, fn_names, cls_names, imports)
    if summary:
        chunks.append(summary)

    return chunks


def _get_func_positions(content: str, ext: str) -> List[Dict[str, Any]]:
    """Return sorted list of {name, line, pos} for function/class definitions."""
    patterns_map = {
        frozenset({'.js', '.jsx', '.ts', '.tsx'}): [
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
            r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(',
            r'(?:export\s+)?class\s+(\w+)',
        ],
        frozenset({'.java'}): [
            r'(?:public|protected|private|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws[^{]+)?\{',
            r'(?:public|protected|private)?\s*class\s+(\w+)',
        ],
        frozenset({'.swift'}): [
            r'func\s+(\w+)',
            r'(?:class|struct|enum)\s+(\w+)',
        ],
        frozenset({'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'}): [
            r'(?:\w+(?:\s*[*&])?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{',
            r'class\s+(\w+)',
        ],
    }
    SKIP = {'if', 'while', 'for', 'switch', 'catch', 'return', 'else', 'do'}

    active = []
    for key, pats in patterns_map.items():
        if ext in key:
            active = pats
            break

    found = []
    for pat in active:
        for m in re.finditer(pat, content, re.MULTILINE):
            name = m.group(1)
            if name not in SKIP:
                line_num = content[:m.start()].count('\n') + 1
                found.append({'name': name, 'line': line_num, 'pos': m.start()})

    found.sort(key=lambda x: x['pos'])
    # Deduplicate by pos
    seen = set()
    deduped = []
    for f in found:
        if f['pos'] not in seen:
            seen.add(f['pos'])
            deduped.append(f)
    return deduped


def chunk_generic_file(file_path: str, relative_path: str) -> List[Document]:
    """Chunk a non-Python source file using regex-detected function boundaries."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
    except Exception as e:
        print(f"[Chunker] Cannot read {file_path}: {e}")
        return []

    all_lines = source.splitlines(keepends=True)
    ext = Path(file_path).suffix.lower()
    functions = _get_func_positions(source, ext)

    if not functions:
        # No identifiable functions — chunk whole file
        return _split_lines_into_chunks(
            all_lines,
            start_line=1,
            file_path=relative_path,
            node_type="file",
            node_name=Path(file_path).name,
        )

    chunks: List[Document] = []
    fn_names: List[str] = []

    for i, func in enumerate(functions):
        start_line = func['line']
        end_line = functions[i + 1]['line'] - 1 if i + 1 < len(functions) else len(all_lines)
        func_lines = all_lines[start_line - 1:end_line]
        func_chunks = _split_lines_into_chunks(
            func_lines,
            start_line=start_line,
            file_path=relative_path,
            node_type="function",
            node_name=func['name'],
        )
        chunks.extend(func_chunks)
        fn_names.append(func['name'])

    # Novelty 4: append module-level summary chunk
    summary = _make_module_summary(relative_path, fn_names, [], [])
    if summary:
        chunks.append(summary)

    return chunks


def chunk_repository(repo_path: str) -> List[Document]:
    """
    Chunk all supported source files in a repository.
    Returns LlamaIndex Documents with rich metadata.
    """
    repo = Path(repo_path)
    all_docs: List[Document] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            try:
                relative = str(fpath.relative_to(repo))
            except ValueError:
                continue

            try:
                if ext in PYTHON_EXTS:
                    docs = chunk_python_file(str(fpath), relative)
                elif ext in OTHER_CODE_EXTS:
                    docs = chunk_generic_file(str(fpath), relative)
                elif ext in TEXT_EXTS:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    if text.strip():
                        docs = [Document(
                            text=text,
                            metadata={
                                "file_name": relative,
                                "start_line": 1,
                                "end_line": text.count('\n') + 1,
                                "node_type": "documentation",
                                "node_name": fname,
                                "chunk_index": 0,
                            }
                        )]
                    else:
                        docs = []
                else:
                    docs = []

                all_docs.extend(docs)

            except Exception as e:
                print(f"[Chunker] Error processing {fpath}: {e}")

    print(f"[Chunker] Created {len(all_docs)} chunks from {repo_path}")
    return all_docs
