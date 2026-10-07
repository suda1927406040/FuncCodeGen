"""Commit tracer - discover co-modified files via git history analysis.

Implements the FeatureSpace Generation step from the paper (Section 3.2):
given an anchor function (chiral pair of the target), trace its git introduction
commit and identify files that were co-modified in that commit. These files
represent the full FeatureSpace context needed to implement similar functionality.

This is a deterministic exploration step (0 LLM calls).
"""
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


@dataclass
class FeatureSpaceFile:
    """A file discovered via FeatureSpace expansion of an anchor's commit history."""
    file: str               # relative path
    anchor_commit: str      # ancestral reference commit (introduction or related change)
    diff_summary: str       # short description of what changed
    diff_content: str       # actual diff for this file in that commit
    reason: str             # why this file is relevant to the new feature
    key_entities: List[str] = field(default_factory=list)  # key variables/functions in diff
    current_symbols: List[str] = field(default_factory=list)
    commit_subject: str = ""
    roles: List[str] = field(default_factory=list)


class CommitTracer:
    """Trace git history to expand FeatureSpaces beyond AST-visible scope."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = str(repo_dir)
        self._intro_cache = {}

    def trace_anchor_co_modified_files(
        self,
        anchor_method_name: str,
        anchor_file: str,
        base_commit: str,
        exclude_files: Optional[Set[str]] = None,
        max_files: int = 3,
        max_diff_chars: int = 2000,
        new_component_name: str = "",
    ) -> List[FeatureSpaceFile]:
        """Find files co-modified when an anchor method was first introduced.

        Args:
            anchor_method_name: e.g. "_find_existing_compatible_binaries"
            anchor_file: e.g. "conans/client/graph/graph_binaries.py"
            base_commit: only consider commits before this (the system's view boundary)
            exclude_files: files to skip (already in target components)
            max_files: max FeatureSpace files to return
            max_diff_chars: truncate diff per file
            new_component_name: name of the new method (for keyword extraction)

        Returns:
            List of FeatureSpaceFile for co-modified files, sorted by relevance.
        """
        exclude_files = exclude_files or set()

        # Extract feature keywords from anchor and new component names
        feature_keywords = self._extract_feature_keywords(anchor_method_name, new_component_name)

        # Step 1: Find the commit that introduced the anchor method
        intro_commit = self._find_introduction_commit(anchor_method_name, anchor_file, base_commit)
        if not intro_commit:
            return []

        # Step 2: Check commit quality — skip large refactoring commits
        commit_subject = self._get_commit_subject(intro_commit)
        co_modified = self._get_co_modified_files(intro_commit)
        if not co_modified:
            return []

        # Heuristic: if the commit touches >8 Python files and has "refactor"/"remove"/"cleanup"
        # in the subject, it's likely a large-scale refactoring with low FeatureSpace signal
        py_files = [f for f in co_modified if f.endswith('.py')]
        is_mega_refactor = (
            len(py_files) > 8
            and any(kw in commit_subject.lower() for kw in ('refactor', 'remove', 'cleanup', 'reorg'))
        )
        if is_mega_refactor:
            # Try the next commit that introduced this method (e.g., a feature-specific one)
            alt_commit = self._find_next_introduction_commit(
                anchor_method_name, anchor_file, base_commit, exclude={intro_commit}
            )
            if alt_commit:
                intro_commit = alt_commit
                commit_subject = self._get_commit_subject(intro_commit)
                co_modified = self._get_co_modified_files(intro_commit)

        # Step 3: Filter and rank by keyword relevance
        anchor_package = self._get_package_prefix(anchor_file)
        candidates = []

        for filepath in co_modified:
            if not (Path(self.repo_dir) / filepath).is_file():
                continue  # Historical paths deleted or renamed before base are not edit candidates.
            # Skip the anchor file itself
            if filepath == anchor_file:
                continue
            # Skip already-targeted files
            if filepath in exclude_files:
                continue
            # Skip non-Python files
            if not filepath.endswith('.py'):
                continue
            # Skip test files
            if self._is_test_file(filepath):
                continue
            # Relevance: same package neighborhood gets priority
            file_package = self._get_package_prefix(filepath)
            if not self._is_related_package(anchor_package, file_package):
                continue

            # Get the diff for this specific file in the introduction commit
            diff = self._get_file_diff(intro_commit, filepath)
            if not diff:
                continue

            # Keyword relevance scoring: diff must contain at least one feature keyword
            diff_text = diff.lower()
            relevance_score = self._score_diff_relevance(diff, feature_keywords)
            if relevance_score <= 0:
                continue  # Skip files with no keyword overlap

            summary = self._summarize_diff(diff)
            key_entities = self._extract_key_entities(diff)
            reason = (
                f"Co-modified with anchor `{anchor_method_name}` in commit {intro_commit[:8]}. "
                f"Changes: {summary}"
            )

            candidates.append((relevance_score, FeatureSpaceFile(
                file=filepath,
                anchor_commit=intro_commit,
                diff_summary=summary,
                diff_content=diff[:max_diff_chars],
                reason=reason,
                key_entities=key_entities,
            )))

        # Sort by relevance score (descending) and take top max_files
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [c[1] for c in candidates[:max_files]]
        return results

    def _find_introduction_commit(self, method_name: str, filepath: str,
                                  base_commit: str) -> Optional[str]:
        """Find the earliest commit before base_commit that introduced `def {method_name}`."""
        cache_key = (method_name, filepath, base_commit)
        if cache_key in self._intro_cache:
            return self._intro_cache[cache_key]
        self._intro_cache[cache_key] = None
        search_pattern = f"def {method_name}"
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", base_commit,
                 f"-S{search_pattern}", "--", filepath],
                cwd=self.repo_dir,
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if result.returncode != 0 or not result.stdout.strip():
            return None

        # Parse output: each line is "hash subject"
        commits = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(None, 1)
                if parts:
                    commits.append(parts[0])

        if not commits:
            return None

        # The last commit in the list (from -S search) is the earliest introduction
        # But we need to verify it's an ancestor of base_commit
        for commit_hash in reversed(commits):
            if self._is_ancestor(commit_hash, base_commit):
                self._intro_cache[cache_key] = commit_hash
                return commit_hash

        return None

    def _get_co_modified_files(self, commit_hash: str) -> List[str]:
        """Get list of files modified in a commit."""
        try:
            result = subprocess.run(
                ["git", "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit_hash],
                cwd=self.repo_dir,
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        if result.returncode != 0:
            return []

        return [line for line in result.stdout.splitlines() if line]

    def _get_file_diff(self, commit_hash: str, filepath: str) -> str:
        """Get the diff for a specific file in a commit."""
        try:
            result = subprocess.run(
                ["git", "show", commit_hash, "--", filepath],
                cwd=self.repo_dir,
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

        if result.returncode != 0:
            return ""

        return result.stdout

    def _is_ancestor(self, commit_hash: str, base_commit: str) -> bool:
        """Check if commit_hash is an ancestor of base_commit."""
        try:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit_hash, base_commit],
                cwd=self.repo_dir,
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def _get_package_prefix(filepath: str) -> str:
        """Get the package prefix of a file path.

        e.g. "conans/client/graph/graph_binaries.py" → "conans/client/graph"
        """
        parts = Path(filepath).parent.parts
        return str(Path(*parts)) if parts else ""

    @staticmethod
    def _is_test_file(filepath: str) -> bool:
        """Check if a file is a test file."""
        lower = filepath.lower()
        return any(p in lower for p in ('/test/', '/tests/', '_test.py', 'test_', '/testing/'))

    @staticmethod
    def _is_related_package(anchor_pkg: str, file_pkg: str) -> bool:
        """Check if two packages are in a related neighborhood.

        Related means: same package, or one is a parent/child of the other.
        e.g. "conans/client/graph" and "conans/client/graph" → True
        e.g. "conans/client/graph" and "conans/client" → True
        e.g. "conans/client/graph" and "conans/model" → False
        """
        if not anchor_pkg or not file_pkg:
            return False
        # Same package
        if anchor_pkg == file_pkg:
            return True
        # Parent-child relationship
        if anchor_pkg.startswith(file_pkg) or file_pkg.startswith(anchor_pkg):
            return True
        # Share parent up to 2 levels: conans/client/graph and conans/client/api
        anchor_parts = anchor_pkg.split('/')
        file_parts = file_pkg.split('/')
        shared_depth = 0
        for a, f in zip(anchor_parts, file_parts):
            if a == f:
                shared_depth += 1
            else:
                break
        return shared_depth >= 2

    @staticmethod
    def _extract_feature_keywords(anchor_name: str, new_component_name: str) -> Set[str]:
        """Extract domain keywords from method names for relevance filtering.

        e.g. "_find_existing_compatible_binaries" + "_find_build_compatible_binary"
        → {"compat", "build", "binari", "find", "exist"}

        Common stop-words are excluded (get, set, is, has, etc.)
        """
        stopwords = {'get', 'set', 'is', 'has', 'can', 'the', 'a', 'an',
                      'self', 'cls', 'new', 'old', 'init', 'call', 'make'}
        keywords = set()
        for name in [anchor_name, new_component_name]:
            words = re.findall(r'[a-z]+', name.lower())
            for w in words:
                if len(w) >= 3 and w not in stopwords:
                    keywords.add(w)
        return keywords

    def _score_diff_relevance(self, diff: str, feature_keywords: Set[str]) -> float:
        """Score how relevant a diff is to the feature being implemented.

        Returns 0 if the diff has no keyword overlap (likely irrelevant).
        Returns a positive score based on keyword density.
        """
        if not feature_keywords:
            return 1.0  # No keywords to filter on, accept all

        diff_lower = diff.lower()
        hit_count = 0
        total_weight = 0
        for kw in feature_keywords:
            weight = min(len(kw), 6) / 3.0  # Longer keywords worth more
            total_weight += weight
            if kw in diff_lower:
                hit_count += weight

        if hit_count == 0:
            return 0.0

        return hit_count / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _extract_key_entities(diff: str) -> List[str]:
        """Extract key variable/function names from a diff for semantic guidance."""
        names = set()
        for line in diff.split('\n'):
            if line.startswith('+++') or line.startswith('---'):
                continue
            stripped = line.lstrip('+- ')
            # Look for attribute accesses: self.xxx, node.xxx, conanfile.xxx
            for m in re.finditer(r'(?:self\.|conanfile\.|node\.|info\.)([a-zA-Z_]\w*)', stripped):
                names.add(m.group(1))
            # Look for function calls: xxx( or xxx = or BINARY_XXX
            for m in re.finditer(r'\b([A-Z][A-Z_]{2,})\b', stripped):
                names.add(m.group(1))
        return sorted(names)[:10]

    def _get_commit_subject(self, commit_hash: str) -> str:
        """Get the subject line of a commit."""
        try:
            result = subprocess.run(
                ["git", "log", "--format=%s", "-1", commit_hash],
                cwd=self.repo_dir,
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def _find_next_introduction_commit(self, method_name: str, filepath: str,
                                        base_commit: str,
                                        exclude: Set[str]) -> Optional[str]:
        """Find the second-earliest commit that introduced `def {method_name}`.

        Used when the earliest commit is a large refactoring (low signal).
        """
        search_pattern = f"def {method_name}"
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", base_commit,
                 f"-S{search_pattern}", "--", filepath],
                cwd=self.repo_dir,
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if result.returncode != 0 or not result.stdout.strip():
            return None

        commits = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(None, 1)
                if parts:
                    commits.append(parts[0])

        for commit_hash in reversed(commits):
            if commit_hash in exclude:
                continue
            if self._is_ancestor(commit_hash, base_commit):
                return commit_hash

        return None

    @staticmethod
    def _summarize_diff(diff: str) -> str:
        """Create a short summary of what a diff changes."""
        added = []
        removed = []
        for line in diff.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added.append(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                removed.append(line[1:].strip())

        parts = []
        if removed:
            parts.append(f"removed {len(removed)} line(s)")
        if added:
            parts.append(f"added {len(added)} line(s)")

        names = set()
        for line in added + removed:
            for m in re.finditer(r'(?:self\.|conanfile\.|node\.)([a-zA-Z_]\w*)', line):
                names.add(m.group(1))

        summary = ', '.join(parts) if parts else 'modified'
        if names:
            summary += f" (key names: {', '.join(sorted(names)[:5])})"
        return summary
