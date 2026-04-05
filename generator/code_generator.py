"""Code generator - orchestrate 3-phase LLM generation pipeline."""
import json
import re
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from generator.llm_interface import call_llm
from generator.prompt_templates import prompt_understand, prompt_extract_patterns, prompt_generate
from config import LLM_MODEL_UNDERSTAND, LLM_MODEL_PATTERN, LLM_MODEL_GENERATE


@dataclass
class GeneratedFile:
    path: str
    action: str          # "create" or "modify"
    description: str
    content: str = ""    # For create: full file content
    search: str = ""     # For modify: string to find
    replace: str = ""    # For modify: replacement


@dataclass
class GenerationResult:
    instance_id: str = ""
    success: bool = False
    files: List[GeneratedFile] = field(default_factory=list)
    understanding: Optional[Dict] = None
    patterns: Optional[Dict] = None
    error: str = ""
    raw_llm_output: str = ""


class CodeGenerator:
    """Orchestrate the 3-phase LLM generation pipeline."""

    def __init__(self, context_builder):
        self.context_builder = context_builder

    def generate(self, ctx) -> GenerationResult:
        result = GenerationResult(instance_id=ctx.instance_id)
        try:
            formatted_context = self.context_builder.format_context_for_llm(ctx)
            component_names = [c.name for c in ctx.components]
            target_files = list({c.target_file for c in ctx.components})

            # Phase 1: Understanding
            print("  [Phase 1/3] Understanding...")
            understanding = self._phase_understand(formatted_context, ctx.request, component_names)
            if not understanding:
                result.error = "Phase 1 (understand) failed"
                return result
            result.understanding = understanding

            # Phase 2: Pattern extraction
            print("  [Phase 2/3] Extracting patterns...")
            patterns = self._phase_extract_patterns(formatted_context, ctx.request, understanding)
            if not patterns:
                result.error = "Phase 2 (patterns) failed"
                return result
            result.patterns = patterns

            # Phase 3: Code generation
            print("  [Phase 3/3] Generating code...")
            generated = self._phase_generate(
                formatted_context, ctx.request, understanding, patterns,
                component_names, target_files
            )
            if not generated:
                result.error = "Phase 3 (generate) failed"
                return result

            result.files = generated
            result.success = True
        except Exception as e:
            result.error = str(e)
            import traceback
            traceback.print_exc()
        return result

    def _phase_understand(self, context: str, request: str,
                          component_names: list) -> Optional[Dict]:
        prompt = prompt_understand(request, context, component_names)
        response = call_llm(
            prompt,
            system="You are a Python code analyst. Output valid JSON only.",
            model=LLM_MODEL_UNDERSTAND,
            json_mode=True,
        )
        return self._parse_json_response(response)

    def _phase_extract_patterns(self, context: str, request: str,
                                 understanding: Dict) -> Optional[Dict]:
        prompt = prompt_extract_patterns(request, context, understanding)
        response = call_llm(
            prompt,
            system="You are a Python code pattern expert. Output valid JSON only.",
            model=LLM_MODEL_PATTERN,
            json_mode=True,
        )
        return self._parse_json_response(response)

    def _phase_generate(self, context: str, request: str,
                        understanding: Dict, patterns: Dict,
                        component_names: list, target_files: list) -> Optional[List[GeneratedFile]]:
        prompt = prompt_generate(
            request, context, understanding, patterns,
            component_names, target_files
        )
        response = call_llm(
            prompt,
            system="You are a Python code generator. Generate complete, correct code. Output valid JSON only.",
            model=LLM_MODEL_GENERATE,
            max_tokens=8192,
        )
        parsed = self._parse_json_response(response)
        if not parsed:
            return None
        return self._parse_generated_files(parsed)

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        brace_count = 0
        start = None
        for i, c in enumerate(response):
            if c == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    try:
                        return json.loads(response[start:i+1])
                    except json.JSONDecodeError:
                        start = None
        return None

    def _parse_generated_files(self, parsed: Dict) -> List[GeneratedFile]:
        files = []
        for f in parsed.get("files", []):
            files.append(GeneratedFile(
                path=f.get("path", ""),
                action=f.get("action", "create"),
                description=f.get("description", ""),
                content=f.get("content", ""),
                search=f.get("search", ""),
                replace=f.get("replace", ""),
            ))
        return files

    # ========================================================================
    # Robust patch application with fuzzy matching
    # ========================================================================

    def apply_generation(self, result: GenerationResult, repo_dir: Path,
                         dry_run: bool = False) -> Dict[str, str]:
        """Apply generated files with robust search/replace matching.

        Matching strategy (cascade):
        1. Exact match
        2. Normalized whitespace match (strip trailing spaces, normalize line endings)
        3. Fuzzy match using difflib (find best matching region in file)
        4. Line-based match (ignore blank lines and comments)
        """
        outcomes = {}
        for gf in result.files:
            target_path = repo_dir / gf.path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if gf.action == "create":
                    if not dry_run:
                        target_path.write_text(gf.content)
                    outcomes[gf.path] = "created"
                elif gf.action == "modify":
                    outcome = self._apply_modify(gf, target_path, dry_run)
                    outcomes[gf.path] = outcome
                else:
                    outcomes[gf.path] = f"error: unknown action {gf.action}"
            except Exception as e:
                outcomes[gf.path] = f"error: {e}"
        return outcomes

    def _apply_modify(self, gf: GeneratedFile, target_path: Path,
                      dry_run: bool) -> str:
        """Apply a modify action with fuzzy matching fallback."""
        if not target_path.exists():
            return "error: file not found"

        current = target_path.read_text()

        if not gf.search:
            return "error: empty search string"

        # Strategy 1: Exact match
        if gf.search in current:
            new_content = current.replace(gf.search, gf.replace, 1)
            if not dry_run:
                target_path.write_text(new_content)
            return "modified"

        # Strategy 2: Normalized match (strip trailing whitespace, normalize line endings)
        norm_current = self._normalize(current)
        norm_search = self._normalize(gf.search)
        if norm_search in norm_current:
            # Find the position in normalized text, map back to original
            new_content = self._apply_normalized_match(current, norm_current, norm_search, gf.replace)
            if new_content is not None:
                if not dry_run:
                    target_path.write_text(new_content)
                return "modified (normalized)"

        # Strategy 3: Fuzzy block match - find the best matching region
        match_result = self._fuzzy_match_block(current, gf.search)
        if match_result is not None:
            start_char, end_char, ratio = match_result
            if ratio >= 0.6:  # At least 60% similar
                original_block = current[start_char:end_char]
                new_content = current[:start_char] + gf.replace + current[end_char:]
                if not dry_run:
                    target_path.write_text(new_content)
                return f"modified (fuzzy {ratio:.0%})"

        # Strategy 4: Line-based match (strip blank lines, compare content lines only)
        line_result = self._line_based_match(current, gf.search, gf.replace)
        if line_result is not None:
            if not dry_run:
                target_path.write_text(line_result)
            return "modified (line-based)"

        # Strategy 5: First-line anchor match
        anchor_result = self._anchor_match(current, gf.search, gf.replace)
        if anchor_result is not None:
            if not dry_run:
                target_path.write_text(anchor_result)
            return "modified (anchor)"

        return "error: search string not found (tried exact/normalized/fuzzy/line/anchor)"

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for fuzzy comparison: strip trailing spaces, unify line endings."""
        lines = text.split('\n')
        # Strip trailing whitespace from each line, remove blank trailing lines
        lines = [l.rstrip() for l in lines]
        while lines and lines[-1] == '':
            lines.pop()
        return '\n'.join(lines)

    def _apply_normalized_match(self, original: str, norm_original: str,
                                 norm_search: str, replace_text: str) -> Optional[str]:
        """Apply match using normalized positions, mapped back to original."""
        idx = norm_original.find(norm_search)
        if idx == -1:
            return None

        # Count characters up to the match in normalized text to estimate position
        # Use line-based mapping for accuracy
        norm_lines = norm_original[:idx].split('\n')
        match_start_line = len(norm_lines) - 1

        norm_search_lines = norm_search.split('\n')
        match_end_line = match_start_line + len(norm_search_lines)

        # Replace the corresponding lines in original
        orig_lines = original.split('\n')
        new_lines = orig_lines[:match_start_line] + [replace_text] + orig_lines[match_end_line:]
        return '\n'.join(new_lines)

    def _fuzzy_match_block(self, content: str, search: str) -> Optional[Tuple[int, int, float]]:
        """Find the best matching block for search within content.

        Uses a sliding window approach with difflib.SequenceMatcher.
        Returns (start_char, end_char, similarity_ratio) or None.
        """
        search_lines = search.split('\n')
        content_lines = content.split('\n')
        n_search = len(search_lines)

        if n_search < 2 or n_search > len(content_lines):
            return None

        best_ratio = 0.0
        best_start = None
        best_end = None

        # Sliding window: try each possible starting position
        step = max(1, n_search // 4)
        for i in range(0, len(content_lines) - n_search + 1, step):
            window = '\n'.join(content_lines[i:i + n_search])
            ratio = difflib.SequenceMatcher(None, window, search).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                # Calculate char positions
                start_char = sum(len(l) + 1 for l in content_lines[:i])
                end_char = start_char + len(window)
                best_start = start_char
                best_end = end_char

        # Also try slightly different window sizes
        for delta in [-2, -1, 1, 2, 3]:
            adj_n = n_search + delta
            if adj_n < 2 or adj_n > len(content_lines):
                continue
            for i in range(0, len(content_lines) - adj_n + 1, step):
                window = '\n'.join(content_lines[i:i + adj_n])
                ratio = difflib.SequenceMatcher(None, window, search).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    start_char = sum(len(l) + 1 for l in content_lines[:i])
                    end_char = start_char + len(window)
                    best_start = start_char
                    best_end = end_char

        if best_start is not None and best_ratio >= 0.5:
            return (best_start, best_end, best_ratio)
        return None

    def _line_based_match(self, content: str, search: str,
                          replace: str) -> Optional[str]:
        """Match by comparing non-empty, non-comment lines.

        Ignores blank lines and pure comment lines for matching.
        """
        def _content_lines(text):
            """Extract significant lines (non-blank, non-pure-comment)."""
            result = []
            for line in text.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    result.append(stripped)
            return result

        search_sig = _content_lines(search)
        if len(search_sig) < 2:
            return None

        content_lines = content.split('\n')
        content_sig = _content_lines(content)

        # Find the first significant line of search in content
        if not search_sig:
            return None

        first_sig = search_sig[0]
        for i, cl in enumerate(content_sig):
            if cl == first_sig or difflib.SequenceMatcher(None, cl, first_sig).ratio() > 0.8:
                # Check if subsequent lines also match
                match_count = 0
                for j, sl in enumerate(search_sig):
                    if i + j < len(content_sig):
                        if content_sig[i + j] == sl or difflib.SequenceMatcher(
                            None, content_sig[i + j], sl
                        ).ratio() > 0.8:
                            match_count += 1

                if match_count >= len(search_sig) * 0.7:
                    # Found the region - need to map back to original line numbers
                    # Find which original lines correspond to these content_sig lines
                    orig_line_idx = 0
                    sig_count = 0
                    start_orig = None
                    end_orig = None

                    for li, line in enumerate(content_lines):
                        stripped = line.strip()
                        if stripped and not stripped.startswith('#'):
                            if sig_count == i:
                                start_orig = li
                            if sig_count == i + match_count - 1:
                                end_orig = li + 1
                                break
                            sig_count += 1

                    if start_orig is not None and end_orig is not None:
                        new_lines = (
                            content_lines[:start_orig]
                            + [replace]
                            + content_lines[end_orig:]
                        )
                        return '\n'.join(new_lines)

        return None

    def _anchor_match(self, content: str, search: str,
                      replace: str) -> Optional[str]:
        """Match using the first meaningful line as anchor, expand context."""
        search_lines = search.split('\n')
        content_lines = content.split('\n')

        # Find the first non-empty, non-comment line in search as anchor
        anchor = None
        anchor_idx = 0
        for si, sl in enumerate(search_lines):
            stripped = sl.strip()
            if stripped and not stripped.startswith('#'):
                anchor = stripped
                anchor_idx = si
                break

        if not anchor:
            return None

        # Find anchor in content (exact or very close match)
        for ci, cl in enumerate(content_lines):
            if cl.strip() == anchor or difflib.SequenceMatcher(
                None, cl.strip(), anchor
            ).ratio() > 0.85:
                # Found anchor - expand to match the search region
                # The search might start a few lines before the anchor
                start = max(0, ci - anchor_idx)
                n_lines = len(search_lines)

                # Try replacing n_lines from start
                end = min(start + n_lines, len(content_lines))
                region = '\n'.join(content_lines[start:end])

                # Check if this region is reasonably similar
                ratio = difflib.SequenceMatcher(None, region, search).ratio()
                if ratio >= 0.5:
                    new_lines = (
                        content_lines[:start]
                        + [replace]
                        + content_lines[end:]
                    )
                    return '\n'.join(new_lines)

        return None
