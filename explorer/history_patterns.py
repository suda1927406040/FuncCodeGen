"""Retrieve coherent, base-ancestor commits as implementation references.

Only requirement text, base symbols and Git history are inputs. No evaluation
patch or tests are consumed. Historical snippets are evidence, not edit targets.
"""
import ast
from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import subprocess
import time
from typing import List

from explorer.commit_tracer import FeatureSpaceFile
from explorer.feature_retriever import identifier_terms, is_source_file

RESOURCE_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".j2", ".jinja", ".jinja2", ".pyi"}


@dataclass
class HistoricalChange:
    file: str
    roles: List[str]
    definitions: List[str]
    calls: List[str]
    fields_written: List[str]
    imports: List[str]
    current_symbols: List[str]
    diff: str
    score: float = 0.0
    anchor: bool = False


@dataclass
class CommitPattern:
    commit: str
    parent: str
    subject: str
    message: str
    anchors: List[str]
    origins: List[str]
    score: float
    reasons: List[str]
    changes: List[HistoricalChange]
    relationships: List[str] = field(default_factory=list)


def changed_lines(diff):
    return [line[1:] for line in diff.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]


def focused_diff(diff, terms, budget):
    """Keep whole hunks; oversized hunks become explicitly labelled observations."""
    pieces = re.split(r"(?=^@@ )", diff, flags=re.MULTILINE)
    header = pieces[0]
    remaining = max(0, budget - len(header) - 100)
    ranked = sorted(enumerate(pieces[1:]), key=lambda item: (-len(terms & identifier_terms(item[1])), item[0]))
    selected = []
    for position, hunk in ranked:
        if len(hunk) <= remaining:
            selected.append((position, hunk))
            remaining -= len(hunk)
    if selected:
        return header + "".join(hunk for _, hunk in sorted(selected)) + "# Other historical hunks omitted.\n"
    observations = []
    remaining = max(0, budget - 100)
    for line in changed_lines(diff):
        if terms & identifier_terms(line) and len(line) + 1 <= remaining:
            observations.append(line)
            remaining -= len(line) + 1
    return "# Selected historical changed lines; this is not an applicable patch.\n" + "\n".join(observations)


class CommitPatternRetriever:
    def __init__(self, repo_dir, index, max_commits=2, time_budget=12):
        self.repo_dir = Path(repo_dir)
        self.index = index
        self.max_commits = max_commits
        self.deadline = time.monotonic() + time_budget
        self.discovery_deadline = time.monotonic() + time_budget * 0.6
        self.discovering = False
        self.cache = {}
        self.diagnostics = {"available": False, "timed_out": False, "queries": 0, "candidate_commits": 0}

    def _git(self, *args, timeout=6):
        if args in self.cache:
            return self.cache[args]
        remaining = (self.discovery_deadline if self.discovering else self.deadline) - time.monotonic()
        if remaining <= 0:
            self.diagnostics["timed_out"] = True
            return None
        self.diagnostics["queries"] += 1
        try:
            result = subprocess.run(["git", "--literal-pathspecs", *args], cwd=self.repo_dir,
                                    capture_output=True, text=True, errors="replace",
                                    timeout=min(timeout, remaining))
            value = result.stdout if result.returncode == 0 else None
        except subprocess.TimeoutExpired:
            self.diagnostics["timed_out"] = True
            value = None
        except FileNotFoundError:
            value = None
        self.cache[args] = value
        return value

    def _metadata(self, commit):
        raw = self._git("show", "-s", "--format=%H%x00%P%x00%s%x00%b", commit)
        if raw is None:
            return None
        sha, parents, subject, body = raw.strip().split("\0", 3)
        return sha, parents.split()[0] if parents else "", subject, body[:1600]

    def _diff(self, commit, parent, file):
        if parent:
            return self._git("diff", "--no-ext-diff", "--no-textconv", "--unified=3", parent, commit, "--", file) or ""
        return self._git("show", "--format=", "--root", "--no-ext-diff", "--no-textconv", "--unified=3", commit, "--", file) or ""

    def _has_symbol(self, commit, file, qualified):
        source = self._git("show", commit + ":" + file)
        if source is None:
            return False
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        def collect(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    if collect(node.body, prefix + node.name + "."):
                        return True
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and prefix + node.name == qualified:
                    return True
            return False
        return collect(tree.body)

    def _introduction(self, anchor, base):
        file, qualified = anchor
        name = qualified.split(".")[-1]
        # Count changes alone also include deletions and same-name methods in
        # other classes. Verify the qualified symbol before/after the commit.
        raw = self._git("log", "--format=%H", "--reverse", base, "-Sdef " + name + "(", "--", file)
        for commit in (raw or "").splitlines()[:6]:
            metadata = self._metadata(commit)
            if not metadata:
                continue
            _, parent, _, _ = metadata
            if self._has_symbol(commit, file, qualified) and (not parent or not self._has_symbol(parent, file, qualified)):
                return commit
        return ""

    def _change(self, file, diff, terms, anchor_files):
        lines = changed_lines(diff)
        text = "\n".join(lines)
        definitions = set(re.findall(r"\b(?:async\s+)?def\s+(\w+)\s*\(", text))
        definitions.update(re.findall(r"\bclass\s+(\w+)", text))
        calls = set(re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", text))
        calls = {name.split(".")[-1] for name in calls} - definitions
        fields = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\.([A-Za-z_]\w*)\s*=(?!=)", text))
        imports = set()
        for line in lines:
            if line.strip().startswith(("import ", "from ")):
                try:
                    for node in ast.walk(ast.parse(line.strip())):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            imports.update(alias.name for alias in node.names)
                except SyntaxError:
                    pass
        roles = []
        if definitions:
            roles.append("definition/signature changes")
        if calls:
            roles.append("call-site changes")
        if fields:
            roles.append("state writes: " + ", ".join(sorted(fields)[:5]))
        if imports:
            roles.append("import wiring")
        if Path(file).name == "__init__.py" and (imports or "__all__" in text):
            roles.append("public exports")
        if Path(file).suffix in RESOURCE_SUFFIXES:
            roles.append("configuration/template/interface data")
        if any(word in text for word in ["except ", "raise ", " if ", "elif "]):
            roles.append("guard/error-path changes")
        if not roles:
            roles.append("implementation changes")
        current = []
        headers = "\n".join(line for line in diff.splitlines() if line.startswith("@@"))
        contextual_definitions = set(re.findall(r"\b(?:async\s+)?def\s+(\w+)\s*\(", diff))
        for symbol in self.index.by_file.get(file, []):
            leaf = symbol.name.split(".")[-1]
            score = (5 if leaf in definitions else 0) + (2 if re.search(r"\b" + re.escape(leaf) + r"\b", headers) else 0)
            score += 2 if leaf in contextual_definitions else 0
            score += min(2, len((symbol.reads | symbol.writes) & {name.lstrip('_') for name in fields}))
            if score:
                current.append((score, symbol.name))
        current_symbols = [name for _, name in sorted(current, key=lambda item: (-item[0], item[1]))[:3]]
        score = 2 * len(terms & identifier_terms(text)) + (3 if imports or fields else 0)
        return HistoricalChange(file, roles, sorted(definitions), sorted(calls), sorted(fields), sorted(imports),
                                current_symbols, focused_diff(diff, terms, 2400), score, file in anchor_files)

    def retrieve(self, ctx, anchor_keys):
        base = getattr(ctx, "base_commit", "")
        if not base or self._git("cat-file", "-e", base + "^{commit}") is None:
            return []
        self.diagnostics["available"] = True
        frequency = Counter()
        for symbol in self.index.symbols.values():
            frequency.update(identifier_terms(symbol.name))
        title = ctx.request.splitlines()[0] if ctx.request else ""
        title_terms = identifier_terms(title)
        terms = set(title_terms)
        for comp in ctx.components:
            terms |= identifier_terms(comp.name)
        generic = {"add", "support", "implement", "change", "method", "function", "class", "enh", "fix",
                   "remove", "create", "use", "allow", "file", "path", "default", "target", "existing"}
        terms -= generic
        title_terms -= generic
        terms = set(sorted(terms, key=lambda word: (-len(word) / math.sqrt(1 + frequency[word]), word))[:12])
        anchors = []
        # Exact requirement references can live outside declared target files.
        references = getattr(ctx, "request_reference_keys", [])
        # Avoid using prose-only matches (e.g. "intersection" in a Fourier
        # request) as anchors for unrelated historical features.
        explicit_names = {value.split('.')[-1] for value in re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", ctx.request)
                          if '.' in value or '_' in value or '`' + value + '`' in ctx.request}
        strong_references = [key for key in references if self.index.symbols[key].name.split('.')[-1] in explicit_names]
        target_anchors = [key for values in anchor_keys.values() for key in values[:2]]
        for key in target_anchors + strong_references[:3]:
            symbol = self.index.symbols.get(key)
            if symbol and (symbol.file, symbol.name) not in anchors:
                anchors.append((symbol.file, symbol.name))
        anchors = anchors[:6]
        pool = {}
        def add(commit, origin, anchor=None):
            if len(pool) >= 28 and commit not in pool:
                return
            item = pool.setdefault(commit, {"origins": set(), "anchors": set()})
            item["origins"].add(origin)
            if anchor:
                item["anchors"].add(anchor)

        # Message retrieval covers relevant edits that kept the function name;
        # introductions alone miss such maintenance/integration changes.
        query_terms = sorted(terms, key=lambda word: (-len(word) / math.sqrt(1 + frequency[word]), word))[:6]
        self.discovering = True
        if query_terms:
            raw = self._git("log", "-n", "10", "--format=%H", "--regexp-ignore-case", "--extended-regexp",
                            "--grep=" + "|".join(re.escape(word) for word in query_terms), base)
            for commit in (raw or "").splitlines():
                add(commit, "requirement/message similarity")
        for anchor in anchors:
            commit = self._introduction(anchor, base)
            if commit:
                add(commit, "verified symbol introduction", anchor)
            pattern = r"(^|[^[:alnum:]_])" + re.escape(anchor[1].split('.')[-1]) + r"([^[:alnum:]_]|$)"
            raw = self._git("log", "-n", "3", "--format=%H", base, "-G" + pattern, "--", anchor[0])
            for commit in (raw or "").splitlines():
                add(commit, "symbol change", anchor)
        self.discovering = False
        self.diagnostics["candidate_commits"] = len(pool)
        ranked = []
        targets = {comp.target_file for comp in ctx.components}
        for commit, item in pool.items():
            metadata = self._metadata(commit)
            if not metadata:
                continue
            sha, parent, subject, body = metadata
            # All searches start at base, never HEAD/--all. This check also
            # protects later refactors that introduce new candidate sources.
            if self._git("merge-base", "--is-ancestor", sha, base) is None:
                continue
            args = ([parent, sha] if parent else ["--root", sha])
            raw = self._git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", *args)
            paths = (raw or "").split("\0")
            paths = [file for file in paths if file and (is_source_file(file) or
                     (Path(file).suffix in RESOURCE_SUFFIXES and not set(Path(file).parts) & {"tests", "test", "docs", "doc"}))]
            # Avoid treating mass reformatting/reorganization as a feature recipe.
            if not paths or len(paths) > 24:
                continue
            active = set(self.index.all_files)
            paths = [file for file in paths if file in active and (self.repo_dir / file).is_file()]
            message_hits = terms & identifier_terms(subject + " " + body)
            anchor_files = {file for file, _ in item["anchors"]} | targets
            if not message_hits and not item["anchors"]:
                continue
            intent_hits = message_hits & title_terms
            specific_intent = {word for word in intent_hits if len(word) >= 5 and frequency[word] <= 3}
            if not item['anchors'] and not (set(paths) & targets) and not (len(intent_hits) >= 2 or specific_intent):
                continue
            changes = []
            for file in sorted(paths, key=lambda file: (file not in anchor_files, file))[:12]:
                diff = self._diff(sha, parent, file)
                if diff:
                    changes.append(self._change(file, diff, terms, anchor_files))
            if not changes:
                continue
            changes.sort(key=lambda change: (not change.anchor, -change.score, change.file))
            changes = changes[:6]
            anchored_target = any(file in targets for file, _ in item['anchors'])
            score = 4 * len(message_hits) + (12 if anchored_target else 5 if item["anchors"] else 0)
            score += min(8, sum(c.score for c in changes) / 4)
            if len(changes) > 1:
                score += 3
            if any(word in subject.lower() for word in ["refactor", "cleanup", "reformat"]):
                score -= 5
            reasons = ["requirement terms in message: " + ", ".join(sorted(message_hits))] if message_hits else []
            reasons += sorted(item["origins"])
            relationships = []
            present = {change.file for change in changes}
            for change in changes:
                for name in change.current_symbols:
                    source_key = change.file + ':' + name
                    for target_key in sorted(self.index.forward.get(source_key, [])):
                        target = self.index.symbols[target_key]
                        if target.file in present and target.file != change.file:
                            relationships.append('resolved static base call: ' + source_key + ' -> ' + target_key)
            for source in changes:
                for dest in changes:
                    if source is dest:
                        continue
                    shared = set(source.definitions) & (set(dest.calls) | set(dest.imports))
                    for name in sorted(shared):
                        if len(name) >= 4 and name not in {"__init__", "get", "set", "str", "list", "dict"}:
                            relationships.append(source.file + " defines/changes " + name + "; " + dest.file + " changes its call/import")
            ranked.append(CommitPattern(sha, parent, subject, body, [file + ":" + name for file, name in sorted(item["anchors"])],
                                        sorted(item["origins"]), score, reasons, changes, list(dict.fromkeys(relationships))[:6]))
        ranked.sort(key=lambda pattern: (-pattern.score, pattern.commit))
        selected = []
        # Prefer coverage of distinct files/roles over near-duplicate recipes.
        covered = set()
        while ranked and len(selected) < self.max_commits:
            pattern = max(ranked, key=lambda value: value.score + min(3, len({c.file for c in value.changes} - covered)))
            selected.append(pattern)
            covered.update(change.file for change in pattern.changes)
            ranked.remove(pattern)
        return selected


def companion_files(patterns, targets, max_files=6):
    candidates = {}
    for pattern in patterns:
        for change in pattern.changes:
            if change.file in targets:
                continue
            score = pattern.score + change.score
            if "public exports" in change.roles and any(Path(change.file).parent == Path(target).parent for target in targets):
                score += 10  # Preserve the target's public integration boundary.
            record = FeatureSpaceFile(change.file, pattern.commit, "; ".join(change.roles), change.diff,
                                      "coordinated change in %s: %s" % (pattern.commit[:8], pattern.subject),
                                      sorted(set(change.definitions + change.fields_written))[:10],
                                      change.current_symbols, pattern.subject, change.roles)
            if change.file not in candidates or score > candidates[change.file][0]:
                candidates[change.file] = (score, record)
    return [record for _, record in sorted(candidates.values(), key=lambda value: (-value[0], value[1].file))[:max_files]]


def render_commit_patterns(patterns, budget=8000):
    if not patterns:
        return ""
    intro = ("\nCOORDINATED HISTORICAL REFERENCES (commits at/before this instance's base):\n"
             "Read each commit as one change across responsibilities. Roles and name links below are static observations, "
             "not proof of runtime behavior. For each relevant historical role, check the current base implementation "
             "in <code>, decide whether the requirement needs an analogous edit, and keep existing behavior intact. "
             "Historical code may have been renamed or superseded; copy edit-before snippets only from current <code>. "
             "Before the edit blocks, give a brief integration plan mapping the required behavior to current entry points, "
             "helpers, state/guard paths, and exports/configuration. Explain which historical roles apply. "
             "Then emit all edits needed for that plan, preserving old API signatures and behavior unless the requirement changes them.\n")
    groups = []
    remaining = budget - len(intro)
    for position, pattern in enumerate(patterns):
        allocation = remaining // (len(patterns) - position)
        mapping = ["\nCOMMIT %s — %s\n" % (pattern.commit[:12], pattern.subject),
                   "Retrieved via: %s.\n" % "; ".join(pattern.reasons),
                   "Commit message: %s\n" % pattern.message[:600]]
        for change in pattern.changes:
            mapping.append("- %s: %s; current symbols: %s\n" % (change.file, "; ".join(change.roles),
                           ", ".join(change.current_symbols) or "inspect current file"))
        for relationship in pattern.relationships:
            mapping.append("  Observed name link: " + relationship + "\n")
        group = "".join(mapping)
        if len(group) > allocation:
            continue  # Never cut off a commit's responsibility map halfway.
        for change in pattern.changes:
            allowance = min(1600 if change.anchor else 1000, allocation - len(group) - len(change.file) - 70)
            if allowance < 250:
                continue
            diff = focused_diff(change.diff, identifier_terms(pattern.subject), allowance)
            fragment = "\nHistorical hunks for %s:\n```diff\n%s\n```\n" % (change.file, diff)
            if len(group) + len(fragment) <= allocation:
                group += fragment
        groups.append(group)
        remaining -= len(group)
    return intro + "".join(groups) if groups else ""
