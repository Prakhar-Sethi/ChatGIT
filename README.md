# ChatGIT

**Chat with any GitHub repository using AI — powered by RAG, PageRank, and AST-aware code understanding.**

ChatGIT is a research system that lets you ask natural language questions about any public GitHub codebase. Load a repo URL and start chatting — the system retrieves the most relevant code context, ranks it by structural importance, and generates precise, grounded answers via Llama 3.1.

---

## Features

- **Multi-turn conversation** with co-reference resolution across turns (e.g., "What does *it* do?" after asking about a function)
- **Intent-aware retrieval** — LOCATE, EXPLAIN, SUMMARIZE, and DEBUG queries each get a tailored retrieval strategy
- **Git-volatility weighting** — recently changed, frequently touched files rank higher in retrieval
- **Query-conditioned PageRank** — structural importance is combined with query relevance at inference time
- **Bidirectional call-graph context** — callers and callees of matched functions are injected into the prompt
- **Semantic redundancy penalty** — chunks seen in recent turns are down-ranked to promote exploration
- **Cross-encoder reranking** — final retrieval pass with ms-marco-MiniLM-L-6-v2
- **Multi-language AST parsing** — Python, JavaScript/TypeScript, Java, Swift, C/C++
- **Interactive visualizations** — PageRank scores, call graphs, file structure explorer
- **Precise code citations** — every answer includes file paths and line numbers

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  N3 – Session Memory: co-reference resolution + pronoun expansion│
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  N4 – Intent Classifier: LOCATE / EXPLAIN / SUMMARIZE / DEBUG    │
│        → sets top_k, max_per_file, call-graph toggle             │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ChromaDB Vector Search (BGE-small-en-v1.5 embeddings)           │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  N2 – Hybrid Importance: query-conditioned PageRank + embedding  │
│  N1 – Git Volatility: frequency 50%, recency 30%, authors 20%    │
│  N3 – Redundancy penalty: same-turn ×0.30, last-turn ×0.60      │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Cross-encoder Reranker (ms-marco-MiniLM-L-6-v2)                 │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  N5 – Call-graph Context: bidirectional callers + callees        │
│        injected into prompt (not retrieved)                      │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Groq API – Llama 3.1-8B-Instant generation                      │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Snippet Extractor: adds exact file paths + line numbers         │
└──────────────────────────────────────────────────────────────────┘
```

### Repository Loading Pipeline

When you submit a GitHub URL, the backend runs:

1. `git clone` via GitPython
2. **AST parsing** (`chatgit/core/ast_parser.py`) — extracts functions/classes per file
3. **Token-aware chunking** (`chatgit/core/chunker.py`) — 512-token chunks, 64-token overlap
4. **Embedding** (`chatgit/core/embeddings.py`) — BGE-small-en-v1.5 via HuggingFace
5. **Vector indexing** — ChromaDB persistent store (keyed by repo path)
6. **PageRank analysis** (`chatgit/core/graph/pagerank.py`) — file/function/import graphs via NetworkX
7. **Git volatility analysis** (`chatgit/core/git_analyzer.py`) — 500-commit lookback

### Intent Retrieval Parameters

| Intent | top_k | max_per_file | Call-graph |
|--------|-------|--------------|------------|
| LOCATE | 25 | 2 | Off |
| EXPLAIN | 20 | 3 | On |
| SUMMARIZE | 12 | 5 | Off |
| DEBUG | 30 | 4 | On |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.12 |
| Vector DB | ChromaDB |
| Embeddings | BGE-small-en-v1.5 (HuggingFace) |
| RAG framework | LlamaIndex |
| LLM | Llama 3.1-8B-Instant via Groq API |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| Graph analysis | NetworkX |
| Frontend | React, Vite |
| Graph viz | vis-network |
| Containerization | Docker |

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Git
- [Groq API Key](https://console.groq.com/)

### 1. Clone the repo

```bash
git clone https://github.com/Prakhar-Sethi/ChatGIT.git
cd ChatGIT
```

### 2. Backend setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### 3. Frontend setup

```bash
cd chatgit-react/frontend
npm install
```

### 4. Run

**Terminal 1 — Backend:**
```bash
# From project root, with venv active
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
# From chatgit-react/frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Docker (optional)

```bash
docker build -t chatgit .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here chatgit
```

---

## Usage

### Load a repository

Paste any public GitHub URL, e.g.:

```
https://github.com/pallets/flask
https://github.com/tiangolo/fastapi
https://github.com/psf/requests
```

Click **Load Repository**. Parsing, indexing, and graph analysis run automatically (~30–90 seconds depending on repo size).

### Chat

Ask anything about the codebase:

```
How is authentication implemented?
What does the request lifecycle look like?
Show me all database models.
Find every place where user input is validated.
How does error handling work in this project?
```

Answers include the relevant code snippets, exact file paths, and line numbers.

### Dashboard

The Dashboard tab shows:
- Total files, functions, classes, packages
- Top files by PageRank score
- Top functions by call frequency
- Module-level dependency rankings

### Call Graph

The Call Graph tab lets you select any function and visualize its callers and callees interactively.

---

## Project Structure

```
ChatGIT/
├── api.py                          # ASGI entrypoint (thin shim → chatgit/api/app.py)
├── requirements.txt
├── Dockerfile
├── .env.example
│
├── chatgit/                        # Main Python package
│   ├── api/
│   │   └── app.py                  # FastAPI app, all routes, ServerContext
│   ├── core/
│   │   ├── ast_parser.py           # Multi-language AST parsing
│   │   ├── chunker.py              # Token-aware chunking
│   │   ├── embeddings.py           # BGE embedding model loader
│   │   ├── git_analyzer.py         # N1: Git volatility scorer
│   │   ├── intent_classifier.py    # N4: Intent routing (LOCATE/EXPLAIN/SUMMARIZE/DEBUG)
│   │   ├── intent_clf.pkl          # Trained LinearSVC classifier
│   │   ├── reranker.py             # Cross-encoder reranker
│   │   ├── session_memory.py       # N3: Session retrieval memory
│   │   ├── snippets.py             # Precise line-number annotation
│   │   └── graph/
│   │       ├── dependency.py       # N5: Function dependency analyzer
│   │       ├── hybrid_importance.py # N2: Query-conditioned PageRank
│   │       └── pagerank.py         # Static PageRank + HITS analysis
│   └── utils/
│
├── chatgit-react/
│   └── frontend/
│       └── src/
│           ├── App.jsx             # Global state, localStorage persistence
│           ├── config.js           # API base URL
│           └── components/
│               ├── Chat.jsx        # Markdown + syntax-highlighted chat
│               ├── Dashboard.jsx   # PageRank / HITS metrics
│               ├── CallGraph.jsx   # vis-network interactive graph
│               ├── Sidebar.jsx     # Repo loading + navigation
│               └── StructureExplorer.jsx
│
├── evaluation/                     # Benchmark & evaluation scripts
│   ├── run_convcodebench.py        # Main ConvCodeBench evaluation
│   ├── run_benchmark.py            # Full benchmark suite
│   ├── ablation.py                 # Per-novelty ablation (N1–N5)
│   ├── llm_judge_eval.py           # LLM-as-judge quality evaluation
│   └── ...
│
├── data/
│   └── convcodebench/              # Benchmark conversation datasets (JSONL)
│
└── results/                        # JSON output from evaluation runs
```

---

## Evaluation

The `evaluation/` directory contains scripts to reproduce all benchmark results.

**Setup**: evaluation scripts expect benchmark repos cloned locally. Set `CHATGIT_REPO_BASE` to their parent directory:

```bash
export CHATGIT_REPO_BASE=/path/to/bench_repos
# The scripts expect: /path/to/bench_repos/flask_bench, requests_bench, etc.
```

**Run the full suite:**

```bash
# ConvCodeBench — multi-turn conversation evaluation (10 repos)
python -m evaluation.run_convcodebench

# Full benchmark (retrieval + generation + faithfulness + latency)
python -m evaluation.run_benchmark

# Ablation study — each novelty independently
python -m evaluation.ablation

# LLM-as-judge quality scores
python -m evaluation.llm_judge_eval

# Cross-repo generalization
python -m evaluation.generate_cross_repo_bench
python -m evaluation.run_full_eval
```

**Retrain the intent classifier:**

```bash
python -m evaluation.train_intent_classifier
# Output: chatgit/core/intent_clf.pkl
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `MODEL_NAME` | `llama-3.1-8b-instant` | Groq model ID |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `WORKSPACE_PATH` | `~/Documents/github_repos` | Repo clone directory |
| `CHROMA_DIR` | `~/.chatgit_cache/chroma_db` | ChromaDB persistence path |
| `CHATGIT_REPO_BASE` | `/tmp` | Parent dir for bench repos |
| `CHATGIT_REPO_<NAME>` | — | Per-repo path override (FLASK, REQUESTS, FASTAPI, ...) |
| `CHATGIT_CONVS_PATH` | — | Override path to conversations JSONL |

---

## Supported Languages

| Language | Extensions |
|----------|-----------|
| Python | `.py` |
| JavaScript | `.js`, `.jsx` |
| TypeScript | `.ts`, `.tsx` |
| Java | `.java` |
| Swift | `.swift` |
| C / C++ | `.c`, `.cpp`, `.h`, `.hpp` |

---

## Acknowledgments

- [Groq](https://groq.com/) for ultra-fast Llama 3.1 inference
- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [HuggingFace](https://huggingface.co/) for BGE embeddings and the cross-encoder reranker
- [NetworkX](https://networkx.org/) for graph analysis
- [vis-network](https://visjs.github.io/vis-network/) for call graph visualization
