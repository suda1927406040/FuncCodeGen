"""Retrieve integration locations from base-tree symbols, calls and state flow.

No tests or reference patches are read. History is an optional signal; retrieval
also works on source archives with no Git history.
"""
import ast
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import os
import math
from pathlib import Path
import re
import subprocess
from typing import Dict, List, Set


EXCLUDED_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
                 "tests", "test", "docs", "doc"}
STOP_WORDS = {"get", "set", "find", "self", "cls", "init", "new", "the", "this",
              "that", "with", "from", "for", "and", "def", "allow", "feature",
              "changelog", "docs", "omit", "close", "not", "there", "current"}
COMMON_FIELDS = {"name", "value", "data", "items", "args", "kwargs", "logger",
                 "output", "append", "get", "set", "keys", "values", "__class__"}


def identifier_terms(text: str) -> Set[str]:
    compounds = {word.lower() for word in re.findall(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b", text)}
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    terms = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) < 3 or word in STOP_WORDS:
            continue
        if word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word.startswith("compatib"):
            word = "compatib"
        terms.add(word)
    return terms | {word for word in compounds if len(word) >= 3}


def dotted_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = dotted_name(node.value)
        return root + "." + node.attr if root else node.attr
    return ""


def is_source_file(path: str) -> bool:
    p = Path(path)
    return (p.suffix == ".py" and not set(p.parts) & EXCLUDED_DIRS
            and not p.name.startswith("test_") and not p.name.endswith("_test.py"))


@dataclass
class Symbol:
    file: str
    name: str
    class_name: str
    source: str
    line: int
    calls: Set[str] = field(default_factory=set)
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    state_updates: List[Dict] = field(default_factory=list)

    @property
    def key(self):
        return self.file + ":" + self.name


@dataclass
class RecallCandidate:
    file: str
    score: float
    reasons: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    snippets: List[str] = field(default_factory=list)
    locations: List[Dict] = field(default_factory=list)


class RepositoryIndex:
    """One AST pass per production file; explicit calls are resolved conservatively."""

    def __init__(self, repo_dir: Path, base_commit: str = ""):
        self.repo_dir = Path(repo_dir)
        self.base_commit = base_commit
        self.symbols: Dict[str, Symbol] = {}
        self.by_file = defaultdict(list)
        self.by_name = defaultdict(list)
        self.class_bases = {}
        self.bindings = {}
        self.imports = {}
        self.modules = {}
        self.forward = defaultdict(set)
        self.reverse = defaultdict(set)
        self.parse_failures = []
        self.string_literals = {}
        paths = self._source_paths()
        for path in paths:
            parts = Path(path).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            self.modules[".".join(parts)] = path
            if parts and parts[0] in {"src", "lib", "python"}:
                self.modules[".".join(parts[1:])] = path
        for path in paths:
            self._index_file(path)
        self._resolve_calls()

    def _source_paths(self):
        command = (["git", "ls-tree", "-rz", "--name-only", self.base_commit]
                   if self.base_commit else ["git", "ls-files", "-z"])
        proc = subprocess.run(command,
                              cwd=self.repo_dir, capture_output=True, timeout=30)
        if proc.returncode == 0:
            paths = proc.stdout.decode().split("\0")
        else:
            paths = []
            for directory, dirs, files in os.walk(self.repo_dir):
                dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
                paths.extend(str((Path(directory) / f).relative_to(self.repo_dir)) for f in sorted(files))
        self.all_files = sorted(p for p in paths if p and not set(Path(p).parts) & EXCLUDED_DIRS)
        return sorted(p for p in paths if p and is_source_file(p)
                      and (self.repo_dir / p).is_file())

    def _index_file(self, path):
        source = (self.repo_dir / path).read_text(errors="replace")
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            self.parse_failures.append(path)
            return
        imports = {}
        # Python imports at module or conditional scope remain useful evidence.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = list(Path(path).parent.parts)
                    module = ".".join(package[:len(package) - node.level + 1] + (module.split(".") if module else []))
                for alias in node.names:
                    imports[alias.asname or alias.name] = (module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = (alias.name, "")
        self.imports[path] = imports
        self.string_literals[path] = [node.value for node in ast.walk(tree)
                                      if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        lines = source.splitlines()
        encoded_lines = source.splitlines(keepends=True)
        segment_cache = {}

        def segment(node):
            if node is None:
                return ""
            key = (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
            if key not in segment_cache:
                start, left, end, right = key
                if start == end:
                    value = encoded_lines[start - 1].encode("utf-8")[left:right].decode("utf-8")
                else:
                    value = (encoded_lines[start - 1].encode("utf-8")[left:].decode("utf-8")
                             + "".join(encoded_lines[start:end - 1])
                             + encoded_lines[end - 1].encode("utf-8")[:right].decode("utf-8"))
                segment_cache[key] = value
            return segment_cache[key]

        def updates(node, guards=()):
            result = []
            if isinstance(node, ast.If):
                condition = (segment(node.test) or "condition")[:160]
                for child in node.body:
                    result.extend(updates(child, guards + ("if " + condition,)))
                for child in node.orelse:
                    result.extend(updates(child, guards + ("else of " + condition,)))
                return result
            if isinstance(node, ast.ExceptHandler):
                exception = segment(node.type) if node.type else "any exception"
                for child in node.body:
                    result.extend(updates(child, guards + ("except " + exception,)))
                return result
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = (segment(node.value) or "")[:160] if node.value else ""
                for target in targets:
                    for child in ast.walk(target):
                        if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store):
                            result.append({"field": child.attr.lstrip("_"), "receiver": dotted_name(child.value),
                                           "line": child.lineno, "value": value, "guards": list(guards),
                                           "statement": (segment(node) or "").splitlines()[0][:200]})
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    result.extend(updates(child, guards))
            return result

        def collect(nodes, prefix="", class_name=""):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    qualified = prefix + node.name
                    self.class_bases[path + ":" + qualified] = [dotted_name(base) for base in node.bases]
                    collect(node.body, qualified + ".", qualified)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    symbol = Symbol(path, prefix + node.name, class_name,
                                    "\n".join(lines[start - 1:node.end_lineno]), start)
                    for statement in node.body:
                        symbol.state_updates.extend(updates(statement))
                    for child in ast.walk(node):
                        if isinstance(child, (ast.Assign, ast.AnnAssign)):
                            value = child.value
                            assigned = child.targets if isinstance(child, ast.Assign) else [child.target]
                            if isinstance(value, ast.Call):
                                for target in assigned:
                                    variable = dotted_name(target)
                                    if variable:
                                        self.bindings[(path, class_name, variable)] = dotted_name(value.func)
                        if isinstance(child, ast.Call):
                            name = dotted_name(child.func)
                            if name:
                                symbol.calls.add(name)
                        elif isinstance(child, ast.Attribute):
                            name = child.attr.lstrip("_")
                            if name and name not in COMMON_FIELDS:
                                (symbol.writes if isinstance(child.ctx, ast.Store)
                                 else symbol.reads).add(name)
                    self.symbols[symbol.key] = symbol
                    self.by_file[path].append(symbol)
                    self.by_name[symbol.name.split(".")[-1]].append(symbol.key)
        collect(tree.body)

    def _resolve_class(self, file, name, seen=None):
        seen = set() if seen is None else seen
        key = file + ":" + name
        if key in self.class_bases:
            return key
        if key in seen or len(seen) >= 5:
            return ""
        seen.add(key)
        root, _, rest = name.partition(".")
        imported = self.imports.get(file, {}).get(root)
        if imported:
            module, original = imported
            target_file = self.modules.get(module, "")
            target_name = ".".join(p for p in [original, rest] if p)
            return self._resolve_class(target_file, target_name, seen)
        return ""

    def _resolve_method(self, class_key, method, seen=None):
        seen = set() if seen is None else seen
        key = class_key + "." + method
        if key in self.symbols:
            return key
        if class_key in seen or len(seen) >= 8:
            return ""
        seen.add(class_key)
        file = class_key.split(":", 1)[0]
        for base in self.class_bases.get(class_key, []):
            parent = self._resolve_class(file, base)
            resolved = self._resolve_method(parent, method, seen) if parent else ""
            if resolved:
                return resolved
        return ""

    def _resolve_calls(self):
        for key, symbol in self.symbols.items():
            for call in symbol.calls:
                target_file, name = symbol.file, call
                if call.startswith(("self.", "cls.")):
                    name = symbol.class_name + "." + call.split(".", 1)[1]
                    parts = call.rsplit(".", 1)
                    bound = self.bindings.get((symbol.file, symbol.class_name, parts[0]))
                    if bound:
                        owner = self._resolve_class(symbol.file, bound)
                        resolved = self._resolve_method(owner, parts[1]) if owner else ""
                    else:
                        resolved = self._resolve_method(symbol.file + ":" + symbol.class_name, call.split(".", 1)[1])
                    if resolved:
                        self.forward[key].add(resolved)
                        self.reverse[resolved].add(key)
                        continue
                else:
                    root, _, rest = call.partition(".")
                    imported = self.imports[symbol.file].get(root)
                    if imported:
                        module, imported_name = imported
                        target_file = self.modules.get(module, "")
                        name = ".".join(p for p in [imported_name, rest] if p)
                        if not imported_name and rest:
                            # Handle a module imported without an alias.
                            full = call.rsplit(".", 1)
                            target_file = self.modules.get(full[0], target_file)
                            name = full[-1]
                target = self._resolve_symbol(target_file, name)
                if target in self.symbols and target != key:
                    self.forward[key].add(target)
                    self.reverse[target].add(key)

    def _resolve_symbol(self, file, name, seen=None):
        seen = set() if seen is None else seen
        key = file + ":" + name
        if key in self.symbols:
            return key
        if key in seen or len(seen) >= 5:
            return ""
        seen.add(key)
        constructor = file + ":" + name + ".__init__"
        if constructor in self.symbols:
            return constructor
        root, _, rest = name.partition(".")
        imported = self.imports.get(file, {}).get(root)
        if imported:
            module, original = imported
            submodule = ".".join(p for p in [module, original] if p)
            if rest and submodule in self.modules:
                return self._resolve_symbol(self.modules[submodule], rest, seen)
            target_file = self.modules.get(module, "")
            target_name = ".".join(p for p in [original, rest] if p)
            return self._resolve_symbol(target_file, target_name, seen)
        return ""


class FeatureRetriever:
    def __init__(self, repo_dir: Path, max_files: int = 12, max_hops: int = 3,
                 base_commit: str = ""):
        self.index = RepositoryIndex(repo_dir, base_commit)
        self.max_files = max_files
        self.max_hops = max_hops
        self.anchor_keys = {}
        self.target_context_symbols = {}
        self.request_references = []
        self._prepared_signature = None

    def _request_symbols(self, request):
        """Resolve explicit API references from the requirement, independent of targets."""
        index = self.index
        references = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", request))
        references.update(re.findall(r"`([A-Za-z_]\w*)`", request))
        references.update(re.findall(r"\b[A-Za-z_]\w*_[A-Za-z_]\w*\b", request))
        matches = set()
        for reference in sorted(references):
            if reference in index.modules:
                continue  # A namespace like conan.tools.files is not a method named files.
            candidates = index.by_name.get(reference.split(".")[-1], [])
            exact = [key for key in candidates if index.symbols[key].name == reference]
            if "." in reference:
                exact.extend(key for key in candidates if '.' in index.symbols[key].name
                             and reference.endswith('.' + index.symbols[key].name))
                for module, file in index.modules.items():
                    if reference.startswith(module + "."):
                        key = index._resolve_symbol(file, reference[len(module) + 1:])
                        if key:
                            exact.append(key)
                # Short module references still need evidence of that module,
                # instead of matching an arbitrary same-named function.
                owner, _, leaf = reference.rpartition('.')
                for module, file in index.modules.items():
                    if module.endswith('.' + owner) or module == owner:
                        key = index._resolve_symbol(file, leaf)
                        if key:
                            exact.append(key)
                selected = exact
            else:
                selected = exact or (candidates if len(candidates) <= 3 else [])
            matches.update(selected)
        # Requests also name APIs in prose, e.g. "Sympy's satisfiable". Accept
        # distinctive exact identifiers with very few repository definitions;
        # qualified/backticked references retain priority over these matches.
        plain = set()
        for word in set(re.findall(r"\b[A-Za-z_]\w*\b", request)):
            if len(word) < 7 or word.lower() in STOP_WORDS:
                continue
            candidates = index.by_name.get(word, [])
            if len(candidates) <= 2:
                plain.update(candidates)
        return (sorted(matches) + sorted(plain - matches))[:12]

    def prepare_anchors(self, ctx):
        signature = (ctx.request, tuple((c.target_file, c.name) for c in ctx.components))
        if signature == self._prepared_signature:
            return
        index = self.index
        terms = identifier_terms(ctx.request)
        seeds = set()
        for component in ctx.components:
            wanted = identifier_terms(component.name.split(".")[-1])
            terms |= wanted
            matches = []
            for symbol in index.by_file[component.target_file]:
                overlap = wanted & identifier_terms(symbol.name.split(".")[-1])
                if overlap:
                    score = len(overlap) / max(1, len(wanted | identifier_terms(symbol.name.split(".")[-1])))
                    matches.append((score, symbol.key))
            anchors = [key for _, key in sorted(matches, key=lambda v: (-v[0], v[1]))[:3]]
            if not anchors:
                anchors = [s.key for s in index.by_file[component.target_file]
                           if wanted & identifier_terms(s.source)][:3]
            self.anchor_keys[component.name] = anchors
            seeds.update(anchors)
        self.request_references = self._request_symbols(ctx.request)
        self._prepared_terms = terms
        self._prepared_seeds = seeds
        self._prepared_signature = signature

    def retrieve(self, ctx) -> List[RecallCandidate]:
        self.prepare_anchors(ctx)
        index = self.index
        targets = {c.target_file for c in ctx.components}
        candidates = {}
        evidence = defaultdict(set)
        terms = set(self._prepared_terms)
        seeds = set(self._prepared_seeds)

        def add(symbol, weight, channel, reason, fields=()):
            if symbol.file in targets:
                return
            candidate = candidates.setdefault(symbol.file, RecallCandidate(symbol.file, 0.0))
            marker = (channel, reason)
            if marker not in evidence[symbol.file]:
                # Cap repeated evidence within each channel to avoid file-size bias.
                count = sum(1 for kind, _ in evidence[symbol.file] if kind == channel)
                limit = 2 if channel == "state" else 3
                if count < limit:
                    candidate.score += weight
                evidence[symbol.file].add(marker)
                if len(candidate.reasons) < 8:
                    candidate.reasons.append(reason)
            if symbol.name not in candidate.symbols and len(candidate.symbols) < 4:
                candidate.symbols.append(symbol.name)
                candidate.snippets.append(symbol.source)
                candidate.locations.append({"symbol": symbol.name, "line": symbol.line,
                                            "reads": sorted(symbol.reads), "writes": sorted(symbol.writes),
                                            "state_updates": symbol.state_updates, "related_fields": []})
            for location in candidate.locations:
                if location["symbol"] == symbol.name:
                    location["related_fields"] = sorted(set(location["related_fields"]) | set(fields))

        for key in self.request_references:
            add(index.symbols[key], 18.0, "request", "explicit API reference in requirement: " + key)
            seeds.add(key)

        # Historical companions carry current-base symbols, so retrieval can
        # preserve the relevant method instead of only including a file name.
        seen_history = set()
        for component in ctx.components:
            for historical in component.feature_space_files:
                marker = (historical.file, historical.anchor_commit)
                if marker in seen_history:
                    continue
                seen_history.add(marker)
                reason = "base-ancestor history: " + historical.reason
                resolved = [index.symbols[historical.file + ':' + name]
                            for name in getattr(historical, 'current_symbols', [])
                            if historical.file + ':' + name in index.symbols]
                for symbol in resolved:
                    add(symbol, 5.0, "history", reason)
                if not resolved:
                    candidate = candidates.setdefault(historical.file, RecallCandidate(historical.file, 0.0))
                    candidate.score += 5.0
                    candidate.reasons.append(reason)

        queue = deque((key, 0) for key in sorted(seeds))
        visited = set(seeds)
        reached = set(seeds)
        while queue:
            key, depth = queue.popleft()
            if depth >= self.max_hops:
                continue
            for direction, neighbors in [("callee", index.forward[key]), ("caller", index.reverse[key])]:
                for other in sorted(neighbors):
                    symbol = index.symbols[other]
                    add(symbol, 8.0 / (depth + 1), "call",
                        "%s of %s (hop %d)" % (direction, key, depth + 1))
                    reached.add(other)
                    if other not in visited:
                        visited.add(other)
                        queue.append((other, depth + 1))

        # Match producers and consumers, not arbitrary shared field references.
        # Frequency and package filters keep generic state from flooding recall.
        frequency = Counter()
        for symbol in index.symbols.values():
            frequency.update(symbol.reads | symbol.writes)
        roots = {Path(p).parts[0] for p in targets}
        state_seeds = [index.symbols[k] for k in sorted(reached)
                       if index.symbols[k].file in targets]
        self.target_context_symbols = {
            file: sorted(s.name for s in state_seeds if s.file == file)
            for file in targets
        }
        for symbol in index.symbols.values():
            if Path(symbol.file).parts[0] not in roots:
                continue
            for seed in state_seeds:
                fields = (symbol.writes & seed.reads) | (symbol.reads & seed.writes)
                fields = sorted(f for f in fields if frequency[f] <= 30)
                if fields:
                    weight = min(6.0, 2.0 + sum(6.0 / math.sqrt(frequency[f]) for f in fields))
                    produced = sorted(set(fields) & symbol.writes & seed.reads)
                    consumed = sorted(set(fields) & symbol.reads & seed.writes)
                    relation = []
                    if produced:
                        relation.append("writes " + ", ".join(produced[:4]) + " consumed by " + seed.key)
                    if consumed:
                        relation.append("reads " + ", ".join(consumed[:4]) + " produced by " + seed.key)
                    add(symbol, weight, "state",
                        "; ".join(relation), fields=fields)

        for symbol in index.symbols.values():
            if Path(symbol.file).parts[0] not in roots:
                continue
            overlap = terms & identifier_terms(symbol.name)
            if len(overlap) >= 2:
                add(symbol, min(3.0, len(overlap)), "lexical",
                    "symbol matches feature terms: " + ", ".join(sorted(overlap)))

        for component in ctx.components:
            # A new module can require both its new package initializer and the
            # nearest existing public export module. No analogous function exists
            # in the target file yet, so anchor-only retrieval cannot find these.
            target = Path(component.target_file)
            if not (index.repo_dir / target).is_file():
                package_files = []
                existing_package = False
                for parent in target.parents:
                    if str(parent) == ".":
                        break
                    initializer = str(parent / "__init__.py")
                    if (index.repo_dir / initializer).is_file():
                        package_files.append(initializer)
                        existing_package = True
                        break
                    package_files.append(initializer)
                if existing_package:
                    for initializer in package_files:
                        candidate = candidates.setdefault(initializer, RecallCandidate(initializer, 0.0))
                        if not any("package export" in r for r in candidate.reasons):
                            candidate.score += 10.0
                            candidate.reasons.append("package export/initialization path for new module " + str(target))
            # Existing modules can also need export updates at a higher package level.
            # Follow current import/re-export evidence, rather than adding every ancestor.
            for parent in target.parents:
                if str(parent) == ".":
                    break
                initializer = str(parent / "__init__.py")
                for local, (module, original) in index.imports.get(initializer, {}).items():
                    source = index.modules.get(module)
                    nested = index.modules.get(".".join(p for p in [module, original] if p))
                    resolved = index._resolve_symbol(source or "", original)
                    owner_file = resolved.split(":", 1)[0] if resolved else ""
                    if target.as_posix() in {source, nested, owner_file}:
                        candidate = candidates.setdefault(initializer, RecallCandidate(initializer, 0.0))
                        reason = "public import/re-export path for " + str(target)
                        if reason not in candidate.reasons:
                            candidate.score += 10.0
                            candidate.reasons.append(reason)
        # Data-driven dispatch frequently loads JSON/YAML files rather than
        # calling Python functions. Match resource paths observed in target code.
        resource_extensions = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".j2", ".jinja", ".jinja2"}
        for target_file in targets:
            for literal in index.string_literals.get(target_file, []):
                if not literal or len(literal) > 160 or "\n" in literal:
                    continue
                clean = literal.replace("\\", "/").strip("/")
                if not clean or clean.startswith("."):
                    continue
                for resource in index.all_files:
                    path = Path(resource)
                    if path.suffix not in resource_extensions:
                        continue
                    related = (path.name == clean or clean in path.parts
                               or resource.endswith("/" + clean))
                    if not related:
                        continue
                    stem_terms = identifier_terms(path.stem)
                    specific = bool(stem_terms & terms)
                    # A dynamic resource directory needs a feature-specific leaf.
                    if path.name != clean and not specific:
                        continue
                    candidate = candidates.setdefault(resource, RecallCandidate(resource, 0.0))
                    reason = "resource path referenced by %s: %s" % (target_file, clean)
                    if reason not in candidate.reasons:
                        candidate.score += 16.0 if specific else 10.0
                        candidate.reasons.append(reason)
                    candidate.symbols = sorted(set(candidate.symbols) | {
                        value for value in index.string_literals.get(target_file, [])
                        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)})[:40]
        # Embedded source templates can reference target classes without an AST
        # call edge. Limit this channel to template/example paths and exact names.
        target_classes = {c.parent_class for c in ctx.components if c.parent_class}
        for file, literals in index.string_literals.items():
            if not any(word in file.lower() for word in ["template", "example"]):
                continue
            mentioned = {name for name in target_classes for value in literals
                         if re.search(r"\b" + re.escape(name) + r"\b", value)}
            if mentioned and file not in targets:
                candidate = candidates.setdefault(file, RecallCandidate(file, 0.0))
                candidate.score += 12.0
                candidate.reasons.append("embedded template uses target class: " + ", ".join(sorted(mentioned)))
        return sorted(candidates.values(), key=lambda c: (-c.score, c.file))[:self.max_files]
