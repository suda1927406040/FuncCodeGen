"""Find state producers in base source, independently of the current patch's files.

This is repair-time retrieval. No reference patch, test patch, or future history
is an input. Relationships found statically remain hypotheses until observed.
"""
import ast
from collections import Counter, defaultdict
from pathlib import Path
import re
import subprocess


def expand_producers(repo, watch, patch, limit=4):
    from generator.state_feedback import production_path
    repo = Path(repo)
    weights = Counter()
    imports = defaultdict(set)
    trees = {}

    def read(name):
        if name not in trees:
            try:
                path = repo / name
                path.resolve().relative_to(repo.resolve())
                trees[name] = ast.parse(path.read_text(errors='replace'))
            except (OSError, ValueError, SyntaxError):
                trees[name] = None
        return trees[name]

    # Conditions expose the state the edited consumers depend on. Added code
    # can mention producers' fields even when its function does not exist at base.
    for line in patch.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            for field in re.findall(r'\.([a-zA-Z_]\w*)\b(?!\s*\()', line):
                weights[field] += 12 if re.search(r'\b(if|elif|while)\b', line) else 1
    for name, symbols in watch.items():
        tree = read(name)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                target = node.module.replace('.', '/') + '.py'
                if production_path(target) and (repo / target).is_file():
                    imports[target].update(alias.name for alias in node.names)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in symbols:
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.IfExp)):
                        called = {id(x.func) for x in ast.walk(child.test) if isinstance(x, ast.Call)}
                        for value in ast.walk(child.test):
                            if isinstance(value, ast.Attribute) and id(value) not in called:
                                weights[value.attr] += 3
    fields = [f for f, _ in weights.most_common(24) if not f.startswith('__')]
    if not fields:
        return []
    pattern = r'\.(' + '|'.join(re.escape(f) for f in fields) + r')\b'
    try:
        found = subprocess.run(['rg', '-l', '--glob', '*.py', '--', pattern, '.'],
                               cwd=repo, text=True, capture_output=True, timeout=20)
        paths = found.stdout.splitlines() if found.returncode in (0, 1) else []
    except (OSError, subprocess.TimeoutExpired):
        paths = []
    # Imports remain available when rg is absent on an offline evaluation host.
    if not paths:
        paths = [str(p.relative_to(repo)) for p in repo.rglob('*.py')
                 if '.git' not in p.parts and production_path(str(p.relative_to(repo)))][:5000]
    scores, matches, symbols = Counter(), defaultdict(set), defaultdict(set)
    writers, readers, frequencies = {}, {}, Counter()
    for name in sorted(set(p.removeprefix('./') for p in paths) | set(imports)):
        if name in watch or not production_path(name):
            continue
        tree = read(name)
        if tree is None:
            continue
        file_fields = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} & set(fields)
        frequencies.update(file_fields)
        file_writes, file_reads = set(), set()
        for function in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            writes = {n.attr for n in ast.walk(function)
                      if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and n.attr in fields}
            reads = {n.attr for n in ast.walk(function) if isinstance(n, ast.Attribute) and n.attr in fields}
            if writes:
                file_writes.update(writes)
                matches[name].update(writes)
                symbols[name].add(function.name)
            # Include an imported responsibility that prepares/validates objects
            # read by the current consumer, even if assignments use an alias.
            if name in imports and reads:
                file_reads.update(reads)
                matches[name].update(reads)
                symbols[name].add(function.name)
        writers[name], readers[name] = file_writes, file_reads
    # Common attributes such as ref/name/self.info must not crowd out a rare
    # validation field. Count each field once per file, not once per method.
    for name in writers:
        scores[name] = sum(min(weights[f], 8) / max(1, frequencies[f]) for f in writers[name]) * 3
        scores[name] += sum(min(weights[f], 8) / max(1, frequencies[f]) for f in readers[name])
        if scores[name] and name in imports:
            scores[name] *= 4
    scores = {name: score for name, score in scores.items() if score}
    selected = sorted(scores, key=lambda name: (-scores[name], name))[:limit]
    return [{'file': name, 'symbols': sorted(symbols[name])[:16],
             'fields': sorted(matches[name]),
             'basis': 'base-code field access/write and import relationships; execution unverified'}
            for name in selected]
