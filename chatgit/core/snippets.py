import re
from pathlib import Path
from difflib import SequenceMatcher

class ImprovedCodeSnippetExtractor:
    """Enhanced code snippet extractor with better accuracy"""
    
    # Configurable thresholds
    DEFAULT_SIMILARITY = 0.80  # General matching threshold
    HIGH_CONFIDENCE = 0.90     # High confidence threshold
    HINT_FILE_SIMILARITY = 0.75  # Lower threshold for hint files
    MIN_SNIPPET_LINES = 2      # Minimum lines for matching
    
    def __init__(self, repo_path, similarity_threshold=None):
        self.repo_path = Path(repo_path)
        self.file_cache = {}
        self.similarity_threshold = similarity_threshold or self.DEFAULT_SIMILARITY
    
    def load_file(self, file_path):
        """Load and cache file contents"""
        if file_path not in self.file_cache:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_cache[file_path] = f.readlines()
            except:
                self.file_cache[file_path] = []
        return self.file_cache[file_path]
    
    def normalize_code(self, code):
        """Normalize code for better matching"""
        # Remove line numbers
        code = re.sub(r'^\s*\d+\s*\|\s*', '', code, flags=re.MULTILINE)
        # Normalize whitespace but preserve structure
        lines = [line.rstrip() for line in code.split('\n')]
        # Remove empty lines at start/end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return lines
    
    def calculate_similarity(self, snippet_lines, file_lines):
        """Calculate similarity between code snippets"""
        # Join lines for sequence matching
        snippet_text = '\n'.join(snippet_lines)
        file_text = '\n'.join(file_lines)
        
        # Use SequenceMatcher for accurate similarity
        similarity = SequenceMatcher(None, snippet_text, file_text).ratio()
        return similarity
    
    def find_code_location(self, code_snippet, file_hint=None, file_extensions=None):
        """
        Find where a code snippet appears in the repository
        
        Args:
            code_snippet: The code to find
            file_hint: Optional filename hint from context
            file_extensions: File extensions to search
        
        Returns:
            Best match with location info or None
        """
        snippet_lines = self.normalize_code(code_snippet)
        
        if not snippet_lines or len(snippet_lines) < 2:
            return None
        
        # Determine file extensions
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h']
        elif isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        
        # Collect all potential matches
        candidates = []
        
        # If file hint provided, search that file first
        search_order = []
        if file_hint:
            hint_path = self.repo_path / file_hint
            if hint_path.exists():
                search_order.append(hint_path)
        
        # Add all other matching files
        for ext in file_extensions:
            for file_path in self.repo_path.rglob(f"*{ext}"):
                if any(skip in str(file_path) for skip in ['venv', '__pycache__', '.git', 'node_modules', 'build', 'dist']):
                    continue
                if file_path not in search_order:
                    search_order.append(file_path)
        
        # Search through files
        for file_path in search_order:
            file_lines = self.load_file(file_path)
            if not file_lines:
                continue
            
            # Try to find snippet with sliding window
            for i in range(len(file_lines) - len(snippet_lines) + 1):
                window_lines = [line.rstrip() for line in file_lines[i:i+len(snippet_lines)]]
                
                # Calculate similarity
                similarity = self.calculate_similarity(snippet_lines, window_lines)
                
                # Adaptive threshold: lower for hint files, use configured threshold otherwise
                is_hint_file = file_hint and str(file_path.relative_to(self.repo_path)) == file_hint
                threshold = self.HINT_FILE_SIMILARITY if is_hint_file else self.similarity_threshold
                
                # Only consider matches above threshold
                if similarity >= threshold:
                    candidates.append({
                        'file': str(file_path.relative_to(self.repo_path)),
                        'start_line': i + 1,
                        'end_line': i + len(snippet_lines),
                        'confidence': similarity,
                        'is_hint_file': is_hint_file
                    })
        
        # Sort candidates: prioritize hint file, then by confidence
        if candidates:
            candidates.sort(key=lambda x: (x['is_hint_file'], x['confidence']), reverse=True)
            return candidates[0]
        
        return None
    
    def extract_code_with_context(self, file_path, start_line, end_line, context_lines=2):
        """Extract code with surrounding context"""
        file_lines = self.load_file(self.repo_path / file_path)
        
        # Calculate range with context
        context_start = max(0, start_line - 1 - context_lines)
        context_end = min(len(file_lines), end_line + context_lines)
        
        # Extract lines
        code_lines = file_lines[context_start:context_end]
        
        return {
            'code': ''.join(code_lines),
            'start_line': context_start + 1,
            'end_line': context_end,
            'highlight_start': start_line,
            'highlight_end': end_line
        }
    
    def add_line_numbers(self, code, start_line=1, highlight_start=None, highlight_end=None):
        """Add line numbers with optional highlighting"""
        lines = code.split('\n')
        result = []
        
        for i, line in enumerate(lines, start=start_line):
            # Add marker for highlighted lines
            marker = "►" if highlight_start and highlight_start <= i <= highlight_end else " "
            result.append(f"{marker}{i:4d} | {line}")
        
        return '\n'.join(result)
    
    def enhance_response(self, response, repo_path, context_metadata=None):
        """
        Enhance LLM response with accurate code locations
        
        Args:
            response: The LLM response text
            repo_path: Repository root path
            context_metadata: Dict mapping snippets to their source files
        """
        code_pattern = r'```(\w+)?\n(.*?)```'
        
        enhanced_response = response
        processed_blocks = set()
        
        for match in re.finditer(code_pattern, response, re.DOTALL):
            lang = match.group(1) or ''
            code = match.group(2)
            
            # Skip very short or already processed snippets
            if len(code.strip()) < 20 or code in processed_blocks:
                continue
            
            processed_blocks.add(code)
            
            # Try to get file hint from context metadata
            file_hint = None
            if context_metadata:
                # Extract potential file references from nearby text
                snippet_context = response[max(0, match.start()-200):match.start()]
                for filename in context_metadata.keys():
                    if filename in snippet_context:
                        file_hint = filename
                        break
            
            # Determine extensions
            ext_map = {
                'python': ['.py'], 'py': ['.py'],
                'javascript': ['.js', '.jsx'], 'js': ['.js', '.jsx'],
                'typescript': ['.ts', '.tsx'], 'ts': ['.ts', '.tsx'],
                'java': ['.java'],
                'cpp': ['.cpp', '.cc', '.cxx'], 'c': ['.c', '.h'],
            }
            
            extensions = ext_map.get(lang.lower(), None)
            location = self.find_code_location(code, file_hint, extensions)
            
            if location and location['confidence'] >= 0.85:
                # Extract with context
                context_code = self.extract_code_with_context(
                    location['file'],
                    location['start_line'],
                    location['end_line'],
                    context_lines=2
                )
                
                # Format metadata
                confidence_emoji = "✓" if location['confidence'] > 0.95 else "~"
                file_info = (
                    f"**{confidence_emoji} {location['file']}** "
                    f"· Lines {context_code['highlight_start']}-{context_code['highlight_end']} "
                    f"(confidence: {location['confidence']:.0%})"
                )
                
                # Add line numbers with highlighting
                numbered_code = self.add_line_numbers(
                    context_code['code'],
                    context_code['start_line'],
                    context_code['highlight_start'],
                    context_code['highlight_end']
                )
                
                # Create enhanced block
                old_block = match.group(0)
                new_block = f"\n{file_info}\n```{lang}\n{numbered_code}\n```"
                
                enhanced_response = enhanced_response.replace(old_block, new_block, 1)
        
        return enhanced_response