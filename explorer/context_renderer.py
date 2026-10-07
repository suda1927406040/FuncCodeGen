"""Render complete AST blocks under a character budget, after full-file parsing."""
import ast
import re
from pathlib import Path
from typing import List

from explorer.feature_retriever import identifier_terms


def render_resource(source: str, symbols: List[str], request: str, budget: int) -> str:
    """Keep original resource lines so before snippets still match the real file."""
    if len(source) <= budget:
        return source
    lines = source.splitlines(keepends=True)
    terms = identifier_terms(request)
    wanted = set(symbols)
    windows = []
    for number, line in enumerate(lines):
        key = re.search(r'["\']([^"\']+)["\']\s*:', line)
        score = len(terms & identifier_terms(line))
        if key and key.group(1) in wanted:
            score += 10
        if score:
            windows.append((score, max(0, number - 4), min(len(lines), number + 5)))
    chosen = set()
    remaining = max(0, budget - 150)
    for _, start, end in sorted(windows, key=lambda item: (-item[0], item[1])):
        extra = set(range(start, end)) - chosen
        size = sum(len(lines[n]) for n in extra) + 60
        if size <= remaining:
            chosen.update(extra)
            remaining -= size
    if not chosen:
        for number, line in enumerate(lines):
            if len(line) > remaining:
                break
            chosen.add(number)
            remaining -= len(line)
    parts = ["# Original resource excerpts; omitted lines are not deleted.\n"]
    previous = -2
    for number in sorted(chosen):
        if number > previous + 1:
            parts.append("\n# ... other resource lines omitted ...\n")
        parts.append(lines[number])
        previous = number
    return "".join(parts)


def render_source(source: str, symbols: List[str], request: str, budget: int) -> str:
    if len(source) <= budget:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Never pretend a prefix of an unparseable file is its complete contents.
        return "# File omitted: cannot select complete AST blocks within this budget."
    lines = source.splitlines(keepends=True)
    wanted = set(symbols)
    priorities = {name: 25.0 / (position + 1) for position, name in reversed(list(enumerate(symbols)))}
    terms = identifier_terms(request)
    blocks = []
    headers = {}

    def add(node, name="", class_name=""):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        end = node.end_lineno
        text = "".join(lines[start:end])
        score = 1 + len(terms & identifier_terms(name))
        if name in wanted or name.split(".")[-1] in wanted:
            score += 100 + priorities.get(name, priorities.get(name.split(".")[-1], 0))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            score += 40
        if name.endswith(".__init__"):
            score += 8
        blocks.append((score, start, end, text, class_name))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            # Include the full multiline class declaration, up to its first body statement.
            first = node.body[0]
            body_start = min([first.lineno] + [d.lineno for d in getattr(first, "decorator_list", [])]) - 1
            headers[node.name] = (class_start, "".join(lines[class_start:body_start]))
            for child in node.body:
                member = getattr(child, 'name', '')
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    member = next((target.id for target in targets if isinstance(target, ast.Name)), '')
                add(child, node.name + "." + member, node.name)
        else:
            add(node, getattr(node, "name", ""))
    selected = []
    used_headers = set()
    size = 100  # Explicit omission markers.
    for _, start, end, text, class_name in sorted(blocks, key=lambda b: (-b[0], b[1])):
        extra = len(text) + 45
        if class_name and class_name not in used_headers:
            extra += len(headers[class_name][1])
        if size + extra > budget:
            continue
        selected.append((start, end, text))
        size += extra
        if class_name and class_name not in used_headers:
            header_start, header = headers[class_name]
            selected.append((header_start, header_start + header.count("\n"), header))
            used_headers.add(class_name)
    parts = ["# Selected complete source blocks. Omitted code is not deleted.\n"]
    previous_end = -1
    for start, end, text in sorted(selected):
        if previous_end >= 0 and start > previous_end:
            parts.append("\n# ... other source blocks omitted ...\n")
        parts.append(text)
        previous_end = end
    return "".join(parts) if selected else "# File omitted: no complete block fits the budget."


def render_code_files(repo_dir: Path, files: List[str], ctx, budget: int,
                      file_caps=None, minimum_budgets=None) -> str:
    sections = []
    targets = {c.target_file for c in ctx.components} if ctx else set()
    symbols = {}
    minimum = {file: 1000 for file in files}
    if ctx:
        symbols.update({path: list(names) for path, names in
                        ctx.recall_diagnostics.get("target_context_symbols", {}).items()})
        for component in ctx.components:
            symbols.setdefault(component.target_file, []).extend(
                [component.name] + [n.class_name + "." + n.method_name for n in component.reference_call_chain])
        for candidate in ctx.recalled_files:
            symbols[candidate.file] = candidate.symbols
            if candidate.snippets:
                # Reserve enough room for the primary retrieved function,
                # imports/class header and omission markers, rather than a
                # fixed small prefix that may contain no relevant method.
                minimum[candidate.file] = min(9000, len(candidate.snippets[0]) + 700)
    sources = {}
    for file in files:
        path = repo_dir / file
        source = path.read_text(errors="replace") if path.is_file() else ""
        sources[file] = source
        minimum[file] = min(minimum[file], len(source)) if source else 80
    if minimum_budgets:
        minimum.update(minimum_budgets)
    remaining = budget
    for position, file in enumerate(files):
        path = repo_dir / file
        header, footer = "\n[start of %s]\n" % file, "\n[end of %s]\n" % file
        overhead = len(header) + len(footer)
        # Reserve room for later candidate files rather than letting one file consume everything.
        reserve = min(sum(minimum[later] + len(later) * 2 + 40 for later in files[position + 1:]),
                      max(0, remaining // 2))
        cap = (file_caps or {}).get(file, 22000 if file in targets else 9000)
        allocation = min(cap, remaining - overhead - reserve)
        if allocation < 200:
            continue
        if path.is_file():
            source = sources[file]
            render = render_source if path.suffix in {".py", ".pyi"} else render_resource
            text = render(source, symbols.get(file, []), ctx.request if ctx else "", allocation)
        else:
            text = "(file does not exist yet, needs to be created)"
        section = header + text + footer
        if len(section) <= remaining:
            sections.append(section)
            remaining -= len(section)
    return "".join(sections)
