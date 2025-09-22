"""AST parser for multi-language code analysis"""

import os
import ast
import json
import re


def _parse_javascript_typescript(content, file_path):
    """Parse JavaScript/TypeScript files for functions and classes"""
    functions = []
    classes = []
    imports = []
    
    # Match function declarations
    func_patterns = [
        r'function\s+(\w+)\s*\(',  # function name()
        r'const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',  # const name = () =>
        r'let\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',  # let name = () =>
        r'(\w+)\s*:\s*(?:async\s+)?\([^)]*\)\s*=>',  # name: () =>
    ]
    
    for pattern in func_patterns:
        #scans the file and matches the pattern
        matches = re.finditer(pattern, content, re.MULTILINE) 
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            functions.append({
                'name': match.group(1),
                'file': file_path,
                'line': line_num,
                'language': 'javascript'
            })
    
    # Match class declarations
    class_matches = re.finditer(r'class\s+(\w+)', content, re.MULTILINE)
    for match in class_matches:
        line_num = content[:match.start()].count('\n') + 1
        classes.append({
            'name': match.group(1),
            'file': file_path,
            'line': line_num,
            'language': 'javascript'
        })
    
    # Match imports
    import_patterns = [
        r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',  # import x from 'module'
        r'require\([\'"]([^\'"]+)[\'"]\)',  # require('module')
    ]
    
    for pattern in import_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            imports.append({
                'module': match.group(1),
                'file': file_path,
                'language': 'javascript'
            })
    
    return functions, classes, imports

def _parse_java(content, file_path):
    """Parse Java files for functions and classes"""
    functions = []
    classes = []
    imports = []
    
    # Match method declarations (simplified)
    method_pattern = r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{'
    matches = re.finditer(method_pattern, content, re.MULTILINE)
    for match in matches:
        line_num = content[:match.start()].count('\n') + 1
        functions.append({
            'name': match.group(1),
            'file': file_path,
            'line': line_num,
            'language': 'java'
        })
    
    # Match class declarations
    class_matches = re.finditer(r'(?:public|private|protected)?\s*class\s+(\w+)', content, re.MULTILINE)
    for match in class_matches:
        line_num = content[:match.start()].count('\n') + 1
        classes.append({
            'name': match.group(1),
            'file': file_path,
            'line': line_num,
            'language': 'java'
        })
    
    # Match imports
    import_matches = re.finditer(r'import\s+([\w.]+);', content, re.MULTILINE)
    for match in import_matches:
        imports.append({
            'module': match.group(1),
            'file': file_path,
            'language': 'java'
        })
    
    return functions, classes, imports

def _parse_swift(content, file_path):
    """Parse Swift files for functions and classes"""
    functions = []
    classes = []
    imports = []
    
    # Match function declarations
    func_matches = re.finditer(r'func\s+(\w+)\s*\(', content, re.MULTILINE)
    for match in func_matches:
        line_num = content[:match.start()].count('\n') + 1
        functions.append({
            'name': match.group(1),
            'file': file_path,
            'line': line_num,
            'language': 'swift'
        })
    
    # Match class/struct declarations
    class_patterns = [
        r'class\s+(\w+)',
        r'struct\s+(\w+)',
        r'enum\s+(\w+)',
    ]
    
    for pattern in class_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            classes.append({
                'name': match.group(1),
                'file': file_path,
                'line': line_num,
                'language': 'swift'
            })
    
    # Match imports
    import_matches = re.finditer(r'import\s+(\w+)', content, re.MULTILINE)
    for match in import_matches:
        imports.append({
            'module': match.group(1),
            'file': file_path,
            'language': 'swift'
        })
    
    return functions, classes, imports

def _parse_cpp(content, file_path):
    """Parse C++ files for functions and classes"""
    functions = []
    classes = []
    imports = []
    
    # Match function declarations (simplified)
    func_pattern = r'(?:\w+(?:\s*\*|\s*&)?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{'
    matches = re.finditer(func_pattern, content, re.MULTILINE)
    for match in matches:
        line_num = content[:match.start()].count('\n') + 1
        if match.group(1) not in ['if', 'while', 'for', 'switch']:  # Filter out keywords
            functions.append({
                'name': match.group(1),
                'file': file_path,
                'line': line_num,
                'language': 'cpp'
            })
    
    # Match class declarations
    class_matches = re.finditer(r'class\s+(\w+)', content, re.MULTILINE)
    for match in class_matches:
        line_num = content[:match.start()].count('\n') + 1
        classes.append({
            'name': match.group(1),
            'file': file_path,
            'line': line_num,
            'language': 'cpp'
        })
    
    # Match includes
    include_matches = re.finditer(r'#include\s+[<"]([^>"]+)[>"]', content, re.MULTILINE)
    for match in include_matches:
        imports.append({
            'module': match.group(1),
            'file': file_path,
            'language': 'cpp'
        })
    
    return functions, classes, imports

def generate_repo_ast(repo_path):
    """Generate detailed AST of repository with multi-language support"""
    repo_summary = {
        'files': {},
        'functions': [],
        'classes': [],
        'imports': []
    }
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules', '.venv', '.idea', '.vscode']]
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_path).replace('\\', '/')
            
            try:
                # Determine file type
                ext = os.path.splitext(file)[1].lower()
                
                if ext == '.py':
                    # Python files - use AST
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content, filename=relative_path)
                    
                    file_info = {
                        'path': relative_path,
                        'functions': [],
                        'classes': [],
                        'imports': [],
                        'type': 'python'
                    }
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            func_info = {
                                'name': node.name,
                                'file': relative_path,
                                'line': node.lineno,
                                'args': [arg.arg for arg in node.args.args],
                                'docstring': ast.get_docstring(node) or "",
                                'is_async': isinstance(node, ast.AsyncFunctionDef),
                                'language': 'python'
                            }
                            file_info['functions'].append(func_info)
                            repo_summary['functions'].append(func_info)
                        
                        elif isinstance(node, ast.ClassDef):
                            class_methods = []
                            for item in node.body:
                                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    class_methods.append({
                                        'name': item.name,
                                        'line': item.lineno
                                    })
                            
                            class_info = {
                                'name': node.name,
                                'file': relative_path,
                                'line': node.lineno,
                                'methods': class_methods,
                                'docstring': ast.get_docstring(node) or "",
                                'language': 'python'
                            }
                            file_info['classes'].append(class_info)
                            repo_summary['classes'].append(class_info)
                        
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                import_info = {
                                    'module': alias.name,
                                    'alias': alias.asname,
                                    'file': relative_path,
                                    'language': 'python'
                                }
                                file_info['imports'].append(import_info)
                                repo_summary['imports'].append(import_info)
                        
                        elif isinstance(node, ast.ImportFrom):
                            for alias in node.names:
                                import_info = {
                                    'module': f"{node.module}.{alias.name}" if node.module else alias.name,
                                    'alias': alias.asname,
                                    'file': relative_path,
                                    'language': 'python'
                                }
                                file_info['imports'].append(import_info)
                                repo_summary['imports'].append(import_info)
                    
                    repo_summary['files'][relative_path] = file_info
                
                elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                    # JavaScript/TypeScript files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    funcs, classes, imports = _parse_javascript_typescript(content, relative_path)
                    
                    file_info = {
                        'path': relative_path,
                        'functions': funcs,
                        'classes': classes,
                        'imports': imports,
                        'type': 'javascript'
                    }
                    
                    repo_summary['functions'].extend(funcs)
                    repo_summary['classes'].extend(classes)
                    repo_summary['imports'].extend(imports)
                    repo_summary['files'][relative_path] = file_info
                
                elif ext == '.java':
                    # Java files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    funcs, classes, imports = _parse_java(content, relative_path)
                    
                    file_info = {
                        'path': relative_path,
                        'functions': funcs,
                        'classes': classes,
                        'imports': imports,
                        'type': 'java'
                    }
                    
                    repo_summary['functions'].extend(funcs)
                    repo_summary['classes'].extend(classes)
                    repo_summary['imports'].extend(imports)
                    repo_summary['files'][relative_path] = file_info
                
                elif ext == '.swift':
                    # Swift files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    funcs, classes, imports = _parse_swift(content, relative_path)
                    
                    file_info = {
                        'path': relative_path,
                        'functions': funcs,
                        'classes': classes,
                        'imports': imports,
                        'type': 'swift'
                    }
                    
                    repo_summary['functions'].extend(funcs)
                    repo_summary['classes'].extend(classes)
                    repo_summary['imports'].extend(imports)
                    repo_summary['files'][relative_path] = file_info
                
                elif ext in ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']:
                    # C/C++ files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    funcs, classes, imports = _parse_cpp(content, relative_path)
                    
                    file_info = {
                        'path': relative_path,
                        'functions': funcs,
                        'classes': classes,
                        'imports': imports,
                        'type': 'cpp'
                    }
                    
                    repo_summary['functions'].extend(funcs)
                    repo_summary['classes'].extend(classes)
                    repo_summary['imports'].extend(imports)
                    repo_summary['files'][relative_path] = file_info
                
                else:
                    # Other files - just track them
                    repo_summary['files'][relative_path] = {
                        'path': relative_path,
                        'type': ext[1:] if ext else 'unknown'
                    }
            
            except Exception as e:
                # Error parsing file
                repo_summary['files'][relative_path] = {
                    'path': relative_path,
                    'error': str(e),
                    'type': os.path.splitext(file)[1][1:] or 'unknown'
                }
    
    # Only count actual code files, not documentation or error files
    CODE_TYPES = {'python', 'javascript', 'java', 'cpp', 'swift'}
    code_files = [f for f in repo_summary['files'].values() 
                  if f.get('type') in CODE_TYPES]
    
    # Count unique imported packages (not individual import statements)
    # Example: "from numpy import a, b, c" counts as 1 package "numpy"
    unique_packages = set()
    for imp in repo_summary['imports']:
        module = imp.get('module', '')
        # Extract base module name (before the first dot)
        base_module = module.split('.')[0] if module else ''
        if base_module:
            unique_packages.add(base_module)
    
    repo_summary['stats'] = {
        'total_files': len(code_files),
        'total_functions': len(repo_summary['functions']),
        'total_classes': len(repo_summary['classes']),
        'total_packages': len(unique_packages)  # Changed from total_imports
    }
    
    # Note: total_packages counts unique top-level packages/libraries used,
    # not the number of import statements
    
    return repo_summary