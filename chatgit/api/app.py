import os
import re
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import torch  # noqa: F401  — must be imported before llama_index on Windows

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LlamaIndex
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core import Document
from llama_index.embeddings.langchain import LangchainEmbedding

# ChromaDB
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Groq
from groq import Groq

# GitPython
from git import Repo, GitCommandError

# Core modules
from chatgit.core.embeddings import load_embedding_model
from chatgit.core.ast_parser import generate_repo_ast
from chatgit.core.chunker import chunk_repository
from chatgit.core.reranker import rerank
from chatgit.core.graph.dependency import FunctionDependencyAnalyzer
from chatgit.core.snippets import ImprovedCodeSnippetExtractor
from chatgit.core.graph.pagerank import CodePageRankAnalyzer

# ── Novelty modules ────────────────────────────────────────────────────────
from chatgit.core.git_analyzer import GitVolatilityAnalyzer          # N1
from chatgit.core.graph.hybrid_importance import HybridImportanceScorer  # N2
from chatgit.core.session_memory import SessionRetrievalMemory        # N3
from chatgit.core.intent_classifier import classify_intent            # N4
# Novelty 5 (bidirectional call-context) is implemented inline below

load_dotenv()

# Token counting
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    print("Tokenizer initialized (cl100k_base)")
except Exception as e:
    TOKENIZER = None
    print(f"Warning: tiktoken not available ({e})")

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", Path.home() / ".chatgit_cache" / "chroma_db"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RepositoryLoadSchema(BaseModel):
    github_url: str

class MessagePayload(BaseModel):
    message: str
    enhance_code: bool = True

class RepoStatistics(BaseModel):
    total_files: int
    total_functions: int
    total_classes: int
    total_packages: int

# ---------------------------------------------------------------------------
# Global server context
# ---------------------------------------------------------------------------

class ServerContext:
    def __init__(self):
        self.repository_root: Optional[str] = None
        self.repo_key:        Optional[str] = None
        self.search_index:    Optional[VectorStoreIndex] = None
        self.code_ast:        Optional[Dict[str, Any]] = None
        self.graph_analyzer:  Optional[CodePageRankAnalyzer] = None
        self.conversation_log: List[Dict[str, str]] = []
        self.llm_client:      Optional[Groq] = None
        self.services_initialized: bool = False
        self.chroma_client:   Optional[chromadb.PersistentClient] = None

        # ── Novelty instances ──────────────────────────────────────────
        self.git_analyzer:    Optional[GitVolatilityAnalyzer] = None   # N1
        self.hybrid_scorer:   Optional[HybridImportanceScorer] = None  # N2
        self.retrieval_memory: SessionRetrievalMemory = SessionRetrievalMemory()  # N3

    def clear_session(self):
        self.repository_root = None
        self.repo_key        = None
        self.search_index    = None
        self.code_ast        = None
        self.graph_analyzer  = None
        self.conversation_log = []
        self.git_analyzer    = None
        self.hybrid_scorer   = None
        self.retrieval_memory.reset()           # N3: reset retrieval memory
        self.services_initialized = False

session = ServerContext()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def initialize_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY not found!")
        return None
    try:
        client = Groq(api_key=key)
        print("Groq client initialized")
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize Groq: {e}")
        return None

def initialize_embedder():
    model = load_embedding_model(device="cpu")
    return LangchainEmbedding(model)

def ensure_services():
    if not session.services_initialized:
        print("Lazy loading AI models...")
        session.llm_client  = initialize_llm()
        Settings.embed_model = initialize_embedder()
        session.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        session.services_initialized = True
        print("AI models loaded.")

def sanitize_collection_name(name: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    if len(sanitized) < 3:
        sanitized += "_repo"
    return sanitized[:63]

def determine_temperature(query: str) -> float:
    q = query.lower()
    if any(k in q for k in ['explain', 'how', 'why', 'what if', 'suggest',
                              'recommend', 'describe', 'compare', 'best way']):
        return 0.3
    if any(k in q for k in ['find', 'show', 'where', 'which file', 'locate',
                              'what does', 'list', 'get']):
        return 0.1
    return 0.2

def extract_github_segments(url: str):
    found = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    return found.groups() if found else (None, None)

def build_file_tree(base_path):
    lines = ["# Project File Tree\n"]
    base = Path(base_path)
    ignored = {'venv', '__pycache__', '.git', 'node_modules', '.venv'}
    for root, folders, filenames in os.walk(base):
        folders[:] = [d for d in folders if d not in ignored]
        rel_path = Path(root).relative_to(base)
        depth = len(rel_path.parts)
        spacer = "  " * depth
        current_folder = rel_path.name if rel_path.parts else "root"
        lines.append(f"{spacer}[{current_folder}]")
        for fname in sorted(filenames)[:20]:
            lines.append(f"{spacer}  - {fname}")
    return "\n".join(lines)

def _count_tokens(text: str) -> int:
    if TOKENIZER:
        return int(len(TOKENIZER.encode(text)) * 1.2)
    return len(text) // 3

def _is_recency_focused(query: str) -> bool:
    """Detect queries asking about recent changes (Novelty 1)."""
    kw = ['recent', 'changed', 'latest', 'updated', 'new', 'last commit',
          'what changed', 'modification', 'modified']
    q = query.lower()
    return any(k in q for k in kw)

# ---------------------------------------------------------------------------
# Novelty 5: Bidirectional call-context neighborhood builder
# ---------------------------------------------------------------------------

def _build_neighborhood_context(
    diverse_results: List[dict],
    analyzer: Optional[CodePageRankAnalyzer],
    repo_root: str,
    max_neighbors: int = 2,
    max_lines: int = 40,
) -> str:
    """
    For each retrieved function, pull its immediate callers and callees
    from the call graph and include a brief code excerpt as auxiliary context.

    This gives the LLM execution-context (how a function is called AND what
    it calls), drastically reducing hallucination on relational questions.
    """
    if not analyzer or not repo_root:
        return ""

    seen_nodes: set = set()
    blocks: List[str] = []

    for res in diverse_results[:4]:   # limit neighbourhood expansion to top-4
        meta  = res["snippet"].metadata
        fname = meta.get("file_name", "")
        fn_names = res.get("matched_funcs", [])

        for fn_name in fn_names[:2]:
            qname = f"{fname}::{fn_name}"
            if qname not in analyzer.function_graph:
                continue

            callers = list(analyzer.function_graph.predecessors(qname))[:max_neighbors]
            callees = list(analyzer.function_graph.successors(qname))[:max_neighbors]

            for neighbor_qname in callers + callees:
                if neighbor_qname in seen_nodes:
                    continue
                seen_nodes.add(neighbor_qname)

                parts = neighbor_qname.split("::")
                if len(parts) != 2:
                    continue
                nb_file, nb_func = parts
                direction = "caller" if neighbor_qname in callers else "callee"

                # Try to read a short excerpt from the actual file
                full_path = Path(repo_root) / nb_file
                excerpt = ""
                if full_path.exists():
                    try:
                        nb_info = analyzer.function_info.get(neighbor_qname, {})
                        start_ln = nb_info.get("line", 1)
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                            all_lines = fh.readlines()
                        excerpt_lines = all_lines[start_ln - 1: start_ln - 1 + max_lines]
                        excerpt = "".join(excerpt_lines)
                    except Exception:
                        pass

                if excerpt:
                    label = "calls" if direction == "callee" else "called by"
                    blocks.append(
                        f"**`{fn_name}` {label} `{nb_func}` "
                        f"({nb_file}, line {analyzer.function_info.get(neighbor_qname,{}).get('line','?')})**\n"
                        f"```\n{excerpt[:600]}\n```"
                    )

    if not blocks:
        return ""
    return "\n\n### Call Neighborhood (Novelty: Bidirectional Context)\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan(server):
    print("Application startup complete. Models are lazy-loaded on first use.")
    yield
    print("Application shutting down.")

app = FastAPI(lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "active"}


@app.post("/api/load_repo")
async def ingest_repository(payload: RepositoryLoadSchema):
    ensure_services()
    url = payload.github_url
    user, project = extract_github_segments(url)

    if not user or not project:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")

    try:
        workspace = Path(os.getenv("WORKSPACE_DIR", Path.home() / "Documents" / "github_repos"))
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            test_file = workspace / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise HTTPException(status_code=500, detail="Cannot write to workspace directory.")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Filesystem error: {e}")

        stats_info = shutil.disk_usage(workspace)
        if stats_info.free < 100 * 1024 * 1024:
            raise HTTPException(status_code=507, detail="Insufficient disk space (need 100MB+)")

        target_path = workspace / project
        already_up_to_date = False

        if not target_path.exists():
            print(f"Cloning {url}...")
            Repo.clone_from(url, str(target_path))
        else:
            print("Repository exists. Pulling latest changes...")
            try:
                repo = Repo(str(target_path))
                pull_result = repo.remotes.origin.pull()
                fetch_info = pull_result[0] if pull_result else None
                already_up_to_date = (fetch_info is not None and fetch_info.flags == 4)
                print("Up to date." if already_up_to_date else "Updated.")
            except GitCommandError as e:
                print(f"Warning: git pull failed: {e}. Using local version.")
                already_up_to_date = True

        repo_key = sanitize_collection_name(f"{user}_{project}")

        chroma_collection = session.chroma_client.get_or_create_collection(
            name=repo_key, metadata={"hnsw:space": "cosine"}
        )
        reuse_index = already_up_to_date and chroma_collection.count() > 0
        ast_data = None

        if reuse_index:
            print(f"Reusing existing vector index ({chroma_collection.count()} chunks).")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            vector_db    = VectorStoreIndex.from_vector_store(vector_store)
        else:
            if chroma_collection.count() > 0:
                print("Rebuilding index (repo updated)...")
                session.chroma_client.delete_collection(repo_key)
                chroma_collection = session.chroma_client.get_or_create_collection(
                    name=repo_key, metadata={"hnsw:space": "cosine"}
                )

            print("Token-aware AST chunking...")
            documents = chunk_repository(str(target_path))

            tree_text = build_file_tree(target_path)
            documents.append(Document(
                text=tree_text,
                metadata={"file_name": "STRUCTURE.md", "node_type": "meta",
                          "node_name": "file_tree", "start_line": 1,
                          "end_line": 0, "chunk_index": 0}
            ))

            print("Parsing AST...")
            ast_data = generate_repo_ast(str(target_path))
            stats    = ast_data.get("stats", {})
            func_list = "\n".join(
                [f"- {fn['name']} ({fn['file']})" for fn in ast_data.get("functions", [])[:50]]
            )
            class_list = "\n".join(
                [f"- {cl['name']} ({cl['file']})" for cl in ast_data.get("classes", [])[:50]]
            )
            overview_text = (
                f"# Codebase Overview\n\n**Metrics:**\n"
                f"- Files: {stats.get('total_files',0)}\n"
                f"- Functions: {stats.get('total_functions',0)}\n"
                f"- Classes: {stats.get('total_classes',0)}\n"
                f"- Packages: {stats.get('total_packages',0)}\n\n"
                f"**Key Functions:**\n{func_list}\n\n**Key Classes:**\n{class_list}\n"
            )
            documents.append(Document(
                text=overview_text,
                metadata={"file_name": "OVERVIEW.md", "node_type": "meta",
                          "node_name": "overview", "start_line": 1,
                          "end_line": 0, "chunk_index": 0}
            ))

            print("Building vector index (ChromaDB)...")
            vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_db = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, show_progress=True
            )
            print(f"Indexed {len(documents)} chunks.")

        print("Running PageRank analysis...")
        if ast_data is None:
            ast_data = generate_repo_ast(str(target_path))
        pagerank = CodePageRankAnalyzer()
        pagerank.analyze_repository(str(target_path))

        # ── Novelty 1: git volatility analysis ──────────────────────────
        print("Running git volatility analysis (Novelty 1)...")
        git_analyzer = GitVolatilityAnalyzer()
        git_analyzer.analyze(str(target_path))

        # ── Novelty 2: build hybrid importance scorer ────────────────────
        print("Building hybrid importance scorer (Novelty 2)...")
        func_pr_dict = dict(pagerank.get_function_pagerank())
        hybrid_scorer = HybridImportanceScorer(pagerank.function_graph)
        hybrid_scorer.build(func_pr_dict, Settings.embed_model)

        # ── Commit to session ────────────────────────────────────────────
        session.repository_root = str(target_path)
        session.repo_key        = repo_key
        session.code_ast        = ast_data
        session.graph_analyzer  = pagerank
        session.search_index    = vector_db
        session.git_analyzer    = git_analyzer       # N1
        session.hybrid_scorer   = hybrid_scorer      # N2
        session.retrieval_memory.reset()             # N3: fresh memory per repo
        session.conversation_log = []

        return {"status": "success", "message": f"Loaded {project}", "repo_name": project}

    except HTTPException:
        raise
    except Exception as ex:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/api/current_repo")
async def get_active_repo():
    if not session.repository_root:
        return {"repo_name": None}
    return {"repo_name": Path(session.repository_root).name}

@app.post("/api/clear_repo")
async def reset_session():
    session.clear_session()
    return {"status": "cleared"}

@app.get("/api/stats")
async def fetch_statistics():
    if not session.code_ast:
        return {}
    return session.code_ast.get("stats", {})

@app.get("/api/structure")
async def fetch_structure():
    if not session.code_ast:
        return {"files": {}}
    return session.code_ast.get("files", {})

@app.get("/api/pagerank/files")
async def get_top_files():
    if not session.graph_analyzer:
        return []
    try:
        ranked = session.graph_analyzer.get_file_pagerank()[:10]
        return [{"name": f, "score": s} for f, s in ranked]
    except Exception as e:
        print(f"[API] Error in get_top_files: {e}")
        return []

@app.get("/api/pagerank/hubs_authorities")
async def get_network_metrics():
    """Hub/authority via HITS on file graph (falls back to degree count)."""
    if not session.graph_analyzer:
        return {"hubs": [], "authorities": []}
    try:
        hits = session.graph_analyzer.get_file_hits_scores(top_n=10)
        hubs  = [{"name": n, "score": s} for n, s in hits["hubs"]        if s > 0]
        auths = [{"name": n, "score": s} for n, s in hits["authorities"] if s > 0]
        # Fallback to degree-count if HITS returned nothing
        if not hubs:
            raw_hubs  = session.graph_analyzer.get_hub_files(10)
            raw_auths = session.graph_analyzer.get_authority_files(10)
            hubs  = [{"name": f, "score": c} for f, c in raw_hubs  if c > 0]
            auths = [{"name": f, "score": c} for f, c in raw_auths if c > 0]
        return {"hubs": hubs, "authorities": auths}
    except Exception as e:
        print(f"[API] Error in get_network_metrics: {e}")
        return {"hubs": [], "authorities": []}

@app.get("/api/pagerank/functions")
async def get_top_functions():
    if not session.graph_analyzer:
        return []
    items = session.graph_analyzer.get_function_pagerank()[:10]
    return [{"name": f, "score": s} for f, s in items]

@app.get("/api/pagerank/central_functions")
async def get_centrality_metrics():
    if not session.graph_analyzer:
        return []
    items = session.graph_analyzer.get_central_functions(10)
    return [{"name": f, "score": s} for f, s in items if s > 0]

@app.get("/api/pagerank/modules")
async def get_module_importance():
    if not session.graph_analyzer:
        return []
    items = session.graph_analyzer.get_import_pagerank()[:10]
    return [{"name": m, "score": s, "is_local": m.endswith(".py")} for m, s in items]

@app.get("/api/hits")
async def get_hits_analysis():
    """
    Return HITS hub and authority scores for both files and functions.
    Hubs   = orchestrators / entry points (high out-degree importance)
    Authorities = core utilities / shared logic (high in-degree importance)
    """
    if not session.graph_analyzer:
        return {"files": {"hubs": [], "authorities": []},
                "functions": {"hubs": [], "authorities": []}}
    try:
        file_hits = session.graph_analyzer.get_file_hits_scores(top_n=10)
        func_hits = session.graph_analyzer.get_hits_scores(top_n=10)
        # Serialize tuples to dicts for JSON
        def _fmt(pairs):
            return [{"name": n, "score": round(s, 6)} for n, s in pairs]
        return {
            "files": {
                "hubs":        _fmt(file_hits["hubs"]),
                "authorities": _fmt(file_hits["authorities"]),
            },
            "functions": {
                "hubs":        _fmt(func_hits["hubs"]),
                "authorities": _fmt(func_hits["authorities"]),
            },
        }
    except Exception as exc:
        import traceback; traceback.print_exc()
        return {"error": str(exc)}

@app.get("/api/call_graph")
async def retrieve_call_graph(target_function: Optional[str] = None):
    if not session.graph_analyzer:
        return {"error": "No repo loaded"}
    try:
        nodes = sorted(list(session.graph_analyzer.function_graph.nodes()))
        return {"functions": nodes}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/call_graph/visualize")
async def generate_graph_data(body: Dict[str, Any] = Body(...)):
    if not session.graph_analyzer:
        return {"error": "No repo loaded"}

    focus = body.get("target")
    if focus == "Show All":
        focus = None

    try:
        fg = session.graph_analyzer.function_graph
        node_list = [{"id": n, "label": n.split("::")[-1]} for n in fg.nodes()]
        edge_list = [{"source": u, "target": v} for u, v in fg.edges()]

        meta = {}
        if focus:
            if "::" not in focus:
                candidates = [n for n in fg.nodes() if n.endswith(f"::{focus}")]
                focus_qualified = candidates[0] if candidates else focus
            else:
                focus_qualified = focus

            deps    = list(fg.successors(focus_qualified))   if focus_qualified in fg else []
            callers = list(fg.predecessors(focus_qualified)) if focus_qualified in fg else []
            meta = {
                "target":       focus_qualified,
                "dependencies": [d.split("::")[-1] for d in deps[:10]],
                "callers":      [c.split("::")[-1] for c in callers[:10]],
            }

        return {"nodes": node_list, "edges": edge_list, "details": meta}

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Chat endpoint — all 5 novelties integrated
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def process_chat(payload: MessagePayload):
    ensure_services()
    if not session.search_index:
        raise HTTPException(status_code=400, detail="Repository not loaded")

    raw_query = payload.message
    session.conversation_log.append({"role": "user", "content": raw_query})

    try:
        analyzer  = session.graph_analyzer
        ast_data  = session.code_ast

        # ── Novelty 3: log query BEFORE resolution (enables temporal back-refs)
        session.retrieval_memory.record_query(raw_query)

        # ── Novelty 3: co-reference resolution ──────────────────────────
        query = session.retrieval_memory.resolve_coreferences(raw_query)
        if query != raw_query:
            print(f"[SessionMemory] Resolved query: {query}")

        # ── Novelty 4: classify intent → retrieval config ────────────────
        cfg = classify_intent(query)
        print(f"[IntentClassifier] Intent={cfg.intent}, top_k={cfg.top_k}, "
              f"granularity={cfg.granularity}, neighbourhood={cfg.include_neighborhood}")

        # ── Step 1: vector search with intent-based top-k ────────────────
        retriever = session.search_index.as_retriever(similarity_top_k=cfg.top_k)
        results   = retriever.retrieve(query)

        # ── Novelty 2: get query embedding for hybrid scoring ─────────────
        query_emb = np.array(Settings.embed_model.get_text_embedding(query))

        # Pre-compute hybrid scores for all nodes (query-conditioned)
        hybrid_scores: Dict[str, float] = {}
        if session.hybrid_scorer and session.hybrid_scorer._built:
            hybrid_scores = session.hybrid_scorer.score_all(query, query_emb)
            print(f"[HybridScorer] Computed {len(hybrid_scores)} node scores.")

        # ── Novelty 1: check if query is recency-focused ──────────────────
        recency_focused = _is_recency_focused(query)

        # PageRank maps (used as fallback / file-level signal)
        file_pr_map = dict(analyzer.get_file_pagerank())    if analyzer else {}

        # ── Step 2: score candidates ─────────────────────────────────────
        candidates       = []
        context_metadata = {}

        for item in results:
            fname    = item.metadata.get("file_name", "unknown")
            content  = item.text
            base_score = item.score if item.score else 1.0

            matched_funcs = []
            node_hybrid   = 0.0
            related_funcs = []

            if ast_data and fname.endswith((".py", ".js", ".ts", ".java", ".cpp")):
                file_funcs = [f for f in ast_data.get("functions", []) if f["file"] == fname]
                for f in file_funcs:
                    fn_name = f["name"]
                    if re.search(r'\b' + re.escape(fn_name) + r'\b', content):
                        matched_funcs.append(fn_name)
                        qname = f"{fname}::{fn_name}"

                        # Novelty 2: use hybrid score instead of raw PageRank
                        h = hybrid_scores.get(qname, 0.0)
                        node_hybrid = max(node_hybrid, h)

                        if analyzer and qname in analyzer.function_graph:
                            succs = list(analyzer.function_graph.successors(qname))[:3]
                            preds = list(analyzer.function_graph.predecessors(qname))[:3]
                            related_funcs.extend([s.split("::")[-1] for s in succs])
                            related_funcs.extend([p.split("::")[-1] for p in preds])

            # Fall back to file-level PageRank if no function matched
            if not matched_funcs:
                node_hybrid = file_pr_map.get(fname, 0.0)

            # Novelty 1: apply git volatility weight
            vol_weight = 1.0
            if session.git_analyzer:
                vol_weight = session.git_analyzer.get_retrieval_weight(
                    fname, recency_focused=recency_focused
                )

            # Novelty 2: hybrid boost (replaces plain pagerank * 10)
            final_score = base_score * (1.0 + node_hybrid * 10.0) * vol_weight

            # Novelty 4: granularity-adaptive boost ──────────────────────
            node_type = item.metadata.get("node_type", "")
            if node_type == "module_summary":
                if cfg.granularity == "module":
                    # SUMMARIZE intent: mild boost; actual GT chunks are class
                    # definitions, so don't over-boost module summaries
                    final_score *= 1.20
                elif cfg.intent in ("explain", "debug"):
                    final_score *= 1.10
            elif node_type == "class":
                if cfg.granularity == "module":
                    # SUMMARIZE intent: class definitions ARE the ground truth
                    # for architecture / overview questions — boost them strongly
                    final_score *= 1.40
                elif cfg.intent in ("explain",):
                    final_score *= 1.10
            elif cfg.granularity == "statement":
                # Prefer smaller chunks (few lines = statement level)
                chunk_lines = (item.metadata.get("end_line", 0)
                               - item.metadata.get("start_line", 0))
                if 0 < chunk_lines <= 10:
                    final_score *= cfg.granularity_boost
            elif cfg.granularity == "function" and node_type == "function":
                final_score *= cfg.granularity_boost
            # debug / mixed: no specific boost

            candidates.append({
                "snippet":       item,
                "score":         final_score,
                "matched_funcs": matched_funcs,
                "related":       list(set(related_funcs)),
                "hybrid_score":  node_hybrid,
                "vol_weight":    vol_weight,
            })

            if fname not in context_metadata:
                context_metadata[fname] = {
                    "functions": matched_funcs,
                    "pagerank":  file_pr_map.get(fname, 0),
                }

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # ── Novelty 3: apply session memory scores ────────────────────────
        candidates = session.retrieval_memory.apply_session_scores(
            candidates, intent=cfg.intent)
        # Re-sort after session adjustments
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # ── Step 3: cross-encoder reranking (intent-based rerank_n) ──────
        top_results = rerank(query, candidates, top_n=cfg.rerank_n)

        # ── Step 4: file-diversity cap (intent-based max_per_file) ───────
        diverse_results: List[dict] = []
        file_count: Dict[str, int]  = {}
        for res in top_results:
            fname = res["snippet"].metadata.get("file_name", "unknown")
            if file_count.get(fname, 0) < cfg.max_per_file:
                diverse_results.append(res)
                file_count[fname] = file_count.get(fname, 0) + 1

        # ── Novelty 5: bidirectional call-context neighborhood ────────────
        neighborhood_ctx = ""
        if cfg.include_neighborhood:
            neighborhood_ctx = _build_neighborhood_context(
                diverse_results, analyzer, session.repository_root or ""
            )
            if neighborhood_ctx:
                print("[CallNeighbour] Added bidirectional call context.")

        # ── Step 5: build context block with token budget ─────────────────
        by_file: Dict[str, list] = {}
        for res in diverse_results:
            fname = res["snippet"].metadata.get("file_name", "unknown")
            by_file.setdefault(fname, []).append(res)

        context_blocks = []
        token_count    = 0

        for fname, file_results in by_file.items():
            file_block = [f"### File: `{fname}`"]
            file_pr    = file_pr_map.get(fname, 0)
            if file_pr > 0.01:
                file_block.append(f"**PageRank Score:** {file_pr:.4f}")

            # Novelty 1: show volatility info if relevant
            if session.git_analyzer and session.git_analyzer._analyzed:
                vol = session.git_analyzer.get_volatility_score(fname)
                if vol > 0.3:
                    file_block.append(f"**Volatility Score:** {vol:.2f} (actively modified)")

            for res in file_results:
                content = res["snippet"].text
                meta    = res["snippet"].metadata
                line_info = ""
                if meta.get("start_line") and meta.get("end_line"):
                    line_info = f" (Lines {meta['start_line']}-{meta['end_line']})"

                meta_parts = []
                if res["matched_funcs"]:
                    meta_parts.append(f"Functions: {', '.join(res['matched_funcs'])}")
                if res["related"]:
                    meta_parts.append(f"Calls: {', '.join(res['related'][:5])}")
                if meta.get("node_name") and meta.get("node_type") not in ("meta", "module"):
                    meta_parts.append(
                        f"{meta.get('node_type','').capitalize()}: {meta.get('node_name','')}"
                    )

                if meta_parts:
                    file_block.append(f"\n**{' | '.join(meta_parts)}{line_info}**")
                elif line_info:
                    file_block.append(f"\n**{line_info.strip()}**")

                file_block.append(f"```\n{content}\n```")

            block_text = "\n".join(file_block)
            est = _count_tokens(block_text)
            if token_count + est < cfg.token_budget:
                context_blocks.append(block_text)
                token_count += est
            else:
                break

        context_block = "\n\n---\n\n".join(context_blocks)

        # Append call-neighbourhood context (Novelty 5)
        if neighborhood_ctx:
            context_block += neighborhood_ctx

        # ── Step 6: build prompt ─────────────────────────────────────────
        stats = (session.code_ast or {}).get("stats", {})

        file_index = "\n".join(
            [f"- `{fname}`: {len(items)} snippet(s)" for fname, items in by_file.items()]
        )

        history_turns = session.conversation_log[:-1]
        recent_history = history_turns[-6:] if len(history_turns) > 6 else history_turns
        history_block = ""
        if recent_history:
            history_lines = []
            for msg in recent_history:
                role    = "User" if msg["role"] == "user" else "Assistant"
                preview = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
                history_lines.append(f"**{role}:** {preview}")
            history_block = "\n\n## Conversation History\n" + "\n\n".join(history_lines)

        # Novelty 3: session summary in the prompt
        session_summary = session.retrieval_memory.get_session_summary()

        final_prompt = f"""You are ChatGIT, an expert code analysis assistant.

# Repository Context

## Statistics
- Total Files: {stats.get('total_files', 0)}
- Total Functions: {stats.get('total_functions', 0)}
- Total Classes: {stats.get('total_classes', 0)}
- Total Packages: {stats.get('total_packages', 0)}

## Retrieval Intent: {cfg.intent.upper()} (granularity: {cfg.granularity})

## Retrieved Files (Ranked by Hybrid Importance + Cross-Encoder)
{file_index}

## Code Snippets
{context_block}
{history_block}
{session_summary}

# Current Query
{query}

# Instructions
1. Answer using ONLY the code provided above.
2. Always specify the exact filename when referencing code.
3. Include line numbers when available.
4. If prior conversation is relevant, refer to it naturally.
5. If information is incomplete, say so — do not hallucinate.
6. Use code blocks with the correct language tag.

Answer:"""

        # ── Step 7: LLM inference ────────────────────────────────────────
        chat_messages = [
            {
                "role": "system",
                "content": (
                    "You are ChatGIT, an expert code assistant. "
                    "Always respond in plain markdown prose — use headings, bullet points, and bold text for structure. "
                    "Only use fenced code blocks (``` ```) for actual code snippets, never for explanatory text. "
                    "Do NOT wrap your entire answer in a code block. "
                    "Always cite exact filenames and line numbers. "
                    "Be precise and grounded in the provided code."
                ),
            },
            {"role": "user", "content": final_prompt},
        ]

        if session.llm_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Groq LLM client is not initialized. "
                    "Please set GROQ_API_KEY in your .env file and restart the server."
                )
            )

        temp = determine_temperature(query)
        completion = session.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_messages,
            temperature=temp,
            max_tokens=2048,
            stream=False,
        )
        answer = completion.choices[0].message.content

        # ── Step 8: enhance with precise line numbers ────────────────────
        if payload.enhance_code and session.repository_root:
            try:
                enhancer = ImprovedCodeSnippetExtractor(session.repository_root)
                answer   = enhancer.enhance_response(
                    answer, session.repository_root, context_metadata=context_metadata
                )
            except Exception as e:
                print(f"Enhancement failed: {e}")

        session.conversation_log.append({"role": "assistant", "content": answer})

        # ── Novelty 3: update retrieval memory ───────────────────────────
        retrieved_info = [
            {
                "file":         r["snippet"].metadata.get("file_name", ""),
                "node_name":    r["snippet"].metadata.get("node_name", ""),
                "matched_funcs": r.get("matched_funcs", []),
            }
            for r in diverse_results
        ]
        session.retrieval_memory.record_turn(raw_query, retrieved_info, answer)

        return {
            "response": answer,
            "history":  session.conversation_log,
            "metadata": {
                "files_used":      list(by_file.keys()),
                "total_snippets":  len(diverse_results),
                "reranked":        True,
                "intent":          cfg.intent,             # N4
                "granularity":     cfg.granularity,        # N4
                "session_turn":    session.retrieval_memory.turn,  # N3
                "neighbourhood":   bool(neighborhood_ctx), # N5
            },
        }

    except Exception as err:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Chat error:\n{error_detail}")
        error_resp = f"Error processing request: {str(err)}"
        if "413" in str(err) or "context" in str(err).lower():
            error_resp += "\n\nContext too large. Try a more specific question."
        session.conversation_log.append({"role": "assistant", "content": error_resp})
        return {"response": error_resp, "history": session.conversation_log}


@app.get("/api/chat/history")
async def fetch_history():
    return session.conversation_log


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
