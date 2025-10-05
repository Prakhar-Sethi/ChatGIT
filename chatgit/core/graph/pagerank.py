
import ast
import os
import re
import networkx as nx
from pathlib import Path
from collections import defaultdict

class CodePageRankAnalyzer:
    """Analyzes code importance using PageRank algorithm, supporting multiple languages."""
    
    def __init__(self):
        self.file_graph = nx.DiGraph()
        self.function_graph = nx.DiGraph()
        self.import_graph = nx.DiGraph()
        self.file_info = {}
        self.function_info = {}
        
        # NEW: Map function names to their full qualified names
        self.function_name_to_full = defaultdict(list)  # Maps 'func_name' -> ['file1.py::func_name', ...]
        
    def analyze_repository(self, repo_path):
        """Analyze entire repository and build graphs for supported file types."""
        repo_path = Path(repo_path)
        
        # Define supported extensions
        supported_exts = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.swift', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'}
        
        # PASS 1: Collect all function definitions
        print("[PageRank] Pass 1: Collecting function definitions...")
        for file_path in repo_path.rglob("*.*"):
            if any(skip in str(file_path) for skip in ['venv', '__pycache__', '.git', 'node_modules']):
                continue
            ext = file_path.suffix.lower()
            if ext not in supported_exts:
                continue
            relative_path = str(file_path.relative_to(repo_path))
            if ext == '.py':
                self._collect_python_functions(file_path, relative_path)
            else:
                self._collect_generic_functions(file_path, relative_path)
        
        print(f"[PageRank] Found {len(self.function_graph.nodes())} function definitions")
        print(f"[PageRank] Function name mapping has {len(self.function_name_to_full)} unique names")
        
        # PASS 2: Analyze files and build call graph
        print("[PageRank] Pass 2: Building call graph...")
        for file_path in repo_path.rglob("*.*"):
            if any(skip in str(file_path) for skip in ['venv', '__pycache__', '.git', 'node_modules']):
                continue
            ext = file_path.suffix.lower()
            if ext not in supported_exts:
                continue
            relative_path = str(file_path.relative_to(repo_path))
            if ext == '.py':
                self._analyze_file(file_path, relative_path, repo_path)
            else:
                self._analyze_generic_file(file_path, relative_path)
        
        print(f"[PageRank] Built graph with {len(self.function_graph.edges())} function call edges")
    
    def _collect_python_functions(self, file_path, relative_path):
        """First pass: collect all Python function definitions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_full_name = f"{relative_path}::{node.name}"
                    self.function_graph.add_node(func_full_name)
                    self.function_name_to_full[node.name].append(func_full_name)
        except Exception as e:
            print(f"Error collecting functions from {file_path}: {e}")
    
    def _collect_generic_functions(self, file_path, relative_path):
        """First pass: collect all function definitions from non-Python files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            ext = Path(file_path).suffix.lower()
            func_patterns = self._get_function_patterns(ext)
            
            for pat in func_patterns:
                for match in re.finditer(pat, content):
                    func_name = match.group(1)
                    if func_name not in {'if', 'while', 'for', 'switch', 'catch', 'return'}:
                        func_full_name = f"{relative_path}::{func_name}"
                        self.function_graph.add_node(func_full_name)
                        self.function_name_to_full[func_name].append(func_full_name)
        except Exception as e:
            print(f"Error collecting functions from {file_path}: {e}")
    
    def _get_function_patterns(self, ext):
        """Get regex patterns for function definitions"""
        if ext in {'.js', '.jsx', '.ts', '.tsx'}:
            return [
                r"function\s+(\w+)\s*\(", 
                r"const\s+(\w+)\s*=\s*\(.*?\)\s*=>", 
                r"const\s+(\w+)\s*=\s*function"
            ]
        elif ext == '.java':
            return [r"(?:public|protected|private|static|\s)+\w+\s+(\w+)\s*\("]
        elif ext == '.swift':
            return [r"func\s+(\w+)"]
        elif ext in {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'}:
            return [r"\w+\s+(\w+)\s*\("]
        return []
    
    def _analyze_file(self, file_path, relative_path, repo_path):
        """Analyze a single Python file (second pass - build edges)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
            
            # Add file to graph
            self.file_graph.add_node(relative_path)
            self.file_info[relative_path] = {
                'path': relative_path,
                'functions': [],
                'classes': [],
                'imports': [],
                'lines': len(content.split('\n'))
            }
            
            # Analyze imports (file-level connections)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_import_edge(relative_path, alias.name)
                        self.file_info[relative_path]['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_import_edge(relative_path, node.module)
                        self.file_info[relative_path]['imports'].append(node.module)
                
                # Analyze functions
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_full_name = f"{relative_path}::{node.name}"
                    self.function_info[func_full_name] = {
                        'name': node.name,
                        'file': relative_path,
                        'line': node.lineno,
                        'calls': []
                    }
                    self.file_info[relative_path]['functions'].append(node.name)
                    
                    # Find function calls and create edges
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                called_func = child.func.id
                                self._add_function_call_edge(func_full_name, called_func, relative_path)
                                self.function_info[func_full_name]['calls'].append(called_func)
                
                # Analyze classes
                elif isinstance(node, ast.ClassDef):
                    self.file_info[relative_path]['classes'].append(node.name)
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def _analyze_generic_file(self, file_path, relative_path):
        """Analyze non-Python files (second pass - build edges)"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Add file node
            self.file_graph.add_node(relative_path)
            self.file_info[relative_path] = {
                'path': relative_path,
                'functions': [],
                'classes': [],
                'imports': [],
                'lines': len(content.split('\n'))
            }

            # Process imports
            ext = Path(file_path).suffix.lower()
            import_patterns = self._get_import_patterns(ext)
            
            imports = []
            for pat in import_patterns:
                for match in re.findall(pat, content):
                    imports.append(match)
                    self._add_import_edge(relative_path, match)
            self.file_info[relative_path]['imports'] = imports

            # Process functions
            func_patterns = self._get_function_patterns(ext)
            
            functions = []
            defined_funcs_with_pos = []
            for pat in func_patterns:
                for match in re.finditer(pat, content):
                    func_name = match.group(1)
                    if func_name in {'if', 'while', 'for', 'switch', 'catch', 'return'}:
                        continue
                    
                    functions.append(func_name)
                    func_full_name = f"{relative_path}::{func_name}"
                    self.function_info[func_full_name] = {
                        'name': func_name,
                        'file': relative_path,
                        'line': content[:match.start()].count('\n') + 1,
                        'calls': []
                    }
                    defined_funcs_with_pos.append((match.start(), match.end(), func_name))
            
            self.file_info[relative_path]['functions'] = functions
            defined_funcs_with_pos.sort()
            
            # Find function calls
            call_pattern = r"(\w+)\s*\("
            for match in re.finditer(call_pattern, content):
                called_name = match.group(1)
                if called_name in {'if', 'while', 'for', 'switch', 'catch', 'return'}:
                    continue
                
                call_pos = match.start()
                
                # Find which function contains this call
                caller_name = None
                for i, (start, end, name) in enumerate(defined_funcs_with_pos):
                    if call_pos > start:
                        if i + 1 < len(defined_funcs_with_pos):
                            next_start, _, _ = defined_funcs_with_pos[i + 1]
                            if call_pos < next_start:
                                caller_name = name
                                break
                        else:
                            caller_name = name
                            break
                
                if caller_name:
                    caller_full = f"{relative_path}::{caller_name}"
                    self._add_function_call_edge(caller_full, called_name, relative_path)
                    if caller_full in self.function_info:
                        self.function_info[caller_full]['calls'].append(called_name)

            # Process classes
            class_patterns = []
            if ext in {'.js', '.jsx', '.ts', '.tsx', '.java', '.swift', '.cpp', '.cc', '.cxx', '.h', '.hpp'}:
                class_patterns = [r"class\s+(\w+)"]
            
            classes = []
            for pat in class_patterns:
                for match in re.findall(pat, content):
                    classes.append(match)
            self.file_info[relative_path]['classes'] = classes

        except Exception as e:
            print(f"Error analyzing generic file {file_path}: {e}")
    
    def _get_import_patterns(self, ext):
        """Get regex patterns for imports"""
        if ext in {'.js', '.jsx', '.ts', '.tsx'}:
            return [r"import\s+[^'\"]+['\"]([^'\"]+)['\"]", r"require\(['\"]([^'\"]+)['\"]\)"]
        elif ext == '.java':
            return [r"import\s+([\w\.]+);"]
        elif ext == '.swift':
            return [r"import\s+([\w]+)"]
        elif ext in {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'}:
            return [r"#include\s+[\"<]([^\">]+)[\">]"]
        return []
    
    def _add_function_call_edge(self, caller_full_name, called_func_name, current_file):
        """Add edge between caller and callee with pragmatic heuristic resolution"""
        # Strategy 1: Same file (highest confidence)
        same_file_target = f"{current_file}::{called_func_name}"
        if same_file_target in self.function_graph.nodes():
            self.function_graph.add_edge(caller_full_name, same_file_target)
            return
        
        # Strategy 2: Function name mapping with file proximity
        if called_func_name in self.function_name_to_full:
            candidates = self.function_name_to_full[called_func_name]
            
            if len(candidates) == 1:
                # Only one match - high confidence
                self.function_graph.add_edge(caller_full_name, candidates[0])
                return
            
            # Multiple candidates - prefer functions in nearby files (same directory)
            caller_dir = str(Path(current_file).parent)
            for candidate in candidates:
                candidate_file = candidate.split('::')[0]
                candidate_dir = str(Path(candidate_file).parent)
                
                if candidate_dir == caller_dir:
                    # Same directory - higher confidence
                    self.function_graph.add_edge(caller_full_name, candidate)
                    return
            
            # No same-directory match - add edges to top 3 candidates with lower weight
            # This helps PageRank but acknowledges uncertainty
            for candidate in candidates[:3]:
                self.function_graph.add_edge(caller_full_name, candidate, weight=0.3)
    
    def _add_import_edge(self, from_file, to_module):
        """Add edge for import relationship"""
        # Add to import graph (tracks ALL imports)
        self.import_graph.add_node(from_file)
        self.import_graph.add_node(to_module)
        self.import_graph.add_edge(from_file, to_module)
        
        # Only add to file_graph if it's a LOCAL import
        is_local = False
        
        # Check if relative import
        if to_module.startswith('.'):
            is_local = True
        
        # Check if module corresponds to a file in our repo
        possible_files = [
            f"{to_module}.py",
            f"{to_module.replace('.', '/')}.py",
            to_module if to_module.endswith('.py') else None
        ]
        
        for pf in possible_files:
            if pf and pf in self.file_graph.nodes():
                is_local = True
                self.file_graph.add_edge(from_file, pf)
                return
        
        # Exclude known external libraries
        external_libs = {
            'os', 'sys', 're', 'json', 'ast', 'pathlib', 'collections',
            'numpy', 'pandas', 'nltk', 'pickle', 'torch', 'tensorflow',
            'sklearn', 'matplotlib', 'seaborn', 'requests', 'flask',
            'django', 'fastapi', 'typing', 'datetime', 'time', 'asyncio',
            'networkx', 'llama_index', 'langchain', 'groq', 'pydantic',
            'subprocess', 'shutil', 'dotenv', 'contextlib', 'abc'
        }
        
        module_root = to_module.split('.')[0]
        if module_root not in external_libs and is_local:
            self.file_graph.add_edge(from_file, to_module)
    
    def get_file_pagerank(self):
        """Calculate PageRank for files - returns ONLY actual code files, never modules"""
        if len(self.file_graph.nodes()) == 0:
            return []
        
        try:
            pagerank = nx.pagerank(self.file_graph, alpha=0.85)
            
            # STRICT filtering - only actual code files
            CODE_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', 
                              '.cpp', '.c', '.h', '.hpp', '.swift'}
            
            file_pagerank = {
                node: score for node, score in pagerank.items()
                if any(node.endswith(ext) for ext in CODE_EXTENSIONS)
            }
            
            # If no results, something is wrong with the graph
            if not file_pagerank:
                print(f"[PageRank] Warning: No code files found in file_graph. Graph has {len(self.file_graph.nodes())} nodes.")
                print(f"[PageRank] Sample nodes: {list(self.file_graph.nodes())[:10]}")
                return []
            
            return sorted(file_pagerank.items(), key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"[PageRank] Error calculating file pagerank: {e}")
            return []
    
    def get_function_pagerank(self):
        """Calculate PageRank for functions"""
        if len(self.function_graph.nodes()) == 0:
            return []
        
        # Check if graph has edges
        if len(self.function_graph.edges()) == 0:
            print("[PageRank] WARNING: Function graph has no edges! All functions will have equal PageRank.")
            # Return nodes with equal scores
            n = len(self.function_graph.nodes())
            return [(node, 1.0/n) for node in self.function_graph.nodes()]
        
        try:
            pagerank = nx.pagerank(self.function_graph, alpha=0.85)
            return sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        except:
            return []
    
    def get_import_pagerank(self):
        """Calculate PageRank for imports/modules"""
        if len(self.import_graph.nodes()) == 0:
            return []
        
        try:
            pagerank = nx.pagerank(self.import_graph, alpha=0.85)
            return sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        except:
            return []
    
    def get_hub_files(self, top_n=10):
        """Get files that import many other files (hubs)"""
        if len(self.file_graph.nodes()) == 0:
            return []
        
        file_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.md', '.txt'}
        out_degree = dict(self.file_graph.out_degree())
        
        # Filter to files only
        filtered_degree = {
            node: degree for node, degree in out_degree.items()
            if any(node.endswith(ext) for ext in file_extensions) or '/' in node
        }
        
        # If no results after filtering, use unfiltered but exclude known modules
        if not filtered_degree:
            known_modules = {'os', 'sys', 're', 'json', 'ast', 'pathlib', 'collections', 'numpy', 'pandas'}
            filtered_degree = {
                node: degree for node, degree in out_degree.items()
                if node not in known_modules
            }
            print(f"[PageRank] Warning: No hub files matched filter")
        
        # Only return files with degree > 0
        results = [(node, degree) for node, degree in filtered_degree.items() if degree > 0]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_authority_files(self, top_n=10):
        """Get files that are imported by many other files (authorities)"""
        if len(self.file_graph.nodes()) == 0:
            return []
        
        file_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.md', '.txt'}
        in_degree = dict(self.file_graph.in_degree())
        
        # Filter to files only
        filtered_degree = {
            node: degree for node, degree in in_degree.items()
            if any(node.endswith(ext) for ext in file_extensions) or '/' in node
        }
        
        # If no results after filtering, use unfiltered but exclude known modules
        if not filtered_degree:
            known_modules = {'os', 'sys', 're', 'json', 'ast', 'pathlib', 'collections', 'numpy', 'pandas'}
            filtered_degree = {
                node: degree for node, degree in in_degree.items()
                if node not in known_modules
            }
            print(f"[PageRank] Warning: No authority files matched filter")
        
        # Only return files with degree > 0
        results = [(node, degree) for node, degree in filtered_degree.items() if degree > 0]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_central_functions(self, top_n=10):
        """Get most central functions using betweenness centrality"""
        if len(self.function_graph.nodes()) == 0:
            return []
        
        if len(self.function_graph.edges()) == 0:
            print("[PageRank] WARNING: No edges in function graph for centrality calculation")
            return []
        
        try:
            centrality = nx.betweenness_centrality(self.function_graph)
            return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except:
            return []

    def get_hits_scores(self, top_n: int = 10):
        """
        Run the HITS (Hyperlink-Induced Topic Search) algorithm on the
        function call graph.

        Hubs    = functions that call many important functions
                  (e.g. orchestrators, controllers, main entry points)
        Authorities = functions called by many important hubs
                      (e.g. core utilities, shared helpers, critical business logic)

        Returns dict with keys 'hubs' and 'authorities', each a list of
        (qualified_name, score) tuples sorted descending.
        """
        if (len(self.function_graph.nodes()) == 0
                or len(self.function_graph.edges()) == 0):
            return {"hubs": [], "authorities": []}
        try:
            hubs, authorities = nx.hits(
                self.function_graph, max_iter=200, normalized=True
            )
            top_hubs  = sorted(hubs.items(),        key=lambda x: x[1], reverse=True)[:top_n]
            top_auths = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return {"hubs": top_hubs, "authorities": top_auths}
        except Exception as exc:
            print(f"[HITS] Function graph error: {exc}")
            return {"hubs": [], "authorities": []}

    def get_file_hits_scores(self, top_n: int = 10):
        """
        Run HITS on the file import/dependency graph.

        File Hubs       = files that import many widely-used modules
                          (entry points, CLI runners, app factories)
        File Authorities = files imported by many entry points
                           (core libraries, shared models, base classes)
        """
        if (len(self.file_graph.nodes()) == 0
                or len(self.file_graph.edges()) == 0):
            return {"hubs": [], "authorities": []}
        try:
            hubs, authorities = nx.hits(
                self.file_graph, max_iter=200, normalized=True
            )
            _exts = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h'}
            def _is_file(n):
                return any(n.endswith(e) for e in _exts) or '/' in n or '\\' in n

            file_hubs  = {k: v for k, v in hubs.items()        if _is_file(k)}
            file_auths = {k: v for k, v in authorities.items() if _is_file(k)}

            top_hubs  = sorted(file_hubs.items(),  key=lambda x: x[1], reverse=True)[:top_n]
            top_auths = sorted(file_auths.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return {"hubs": top_hubs, "authorities": top_auths}
        except Exception as exc:
            print(f"[HITS] File graph error: {exc}")
            return {"hubs": [], "authorities": []}

    def get_file_metrics(self, file_path):
        """Get all metrics for a specific file"""
        if file_path not in self.file_info:
            return None
        
        info = self.file_info[file_path].copy()
        
        # Add PageRank
        file_pr = dict(self.get_file_pagerank())
        info['pagerank'] = file_pr.get(file_path, 0)
        
        # Add degree metrics
        info['imports_count'] = self.file_graph.out_degree(file_path) if file_path in self.file_graph else 0
        info['imported_by_count'] = self.file_graph.in_degree(file_path) if file_path in self.file_graph else 0
        
        return info
    
    def get_summary_stats(self):
        """Get summary statistics"""
        return {
            'total_files': len([n for n in self.file_graph.nodes() if any(n.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.cpp'])]),
            'total_functions': len(self.function_graph.nodes()),
            'total_function_calls': len(self.function_graph.edges()),
            'total_imports': len(self.import_graph.edges()),
            'avg_file_connections': sum(dict(self.file_graph.degree()).values()) / max(len(self.file_graph.nodes()), 1),
            'avg_function_calls': sum(dict(self.function_graph.degree()).values()) / max(len(self.function_graph.nodes()), 1)
        }


        