"""Render complete AST blocks under a character budget, after full-file parsing."""
import ast
from pathlib import Path
from typing import List

from explorer.feature_retriever import identifier_terms


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
    terms = identifier_terms(request)
    blocks = []
    headers = {}

    def add(node, name="", class_name=""):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        end = node.end_lineno
        text = "".join(lines[start:end])
        score = 1 + len(terms & identifier_terms(name))
        if name in wanted or name.split(".")[-1] in wanted:
            score += 100
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            score += 40
        if name.endswith(".__init__"):
            score += 8
        blocks.append((score, start, end, text, class_name))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            # Include the full multiline class declaration, up to its first body statement.
            headers[node.name] = (class_start, "".join(lines[class_start:node.body[0].lineno - 1]))
            for child in node.body:
                add(child, node.name + "." + getattr(child, "name", ""), node.name)
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


def render_code_files(repo_dir: Path, files: List[str], ctx, budget: int) -> str:
    sections = []
    targets = {c.target_file for c in ctx.components} if ctx else set()
    symbols = {}
    if ctx:
        symbols.update({path: list(names) for path, names in
                        ctx.recall_diagnostics.get("target_context_symbols", {}).items()})
        for component in ctx.components:
            symbols.setdefault(component.target_file, []).extend(
                [component.name] + [n.class_name + "." + n.method_name for n in component.reference_call_chain])
        for candidate in ctx.recalled_files:
            symbols[candidate.file] = candidate.symbols
    remaining = budget
    for position, file in enumerate(files):
        path = repo_dir / file
        header, footer = "\n[start of %s]\n" % file, "\n[end of %s]\n" % file
        overhead = len(header) + len(footer)
        # Reserve room for later candidate files rather than letting one file consume everything.
        reserve = min(1800 * (len(files) - position - 1), max(0, remaining // 2))
        allocation = min(22000 if file in targets else 9000, remaining - overhead - reserve)
        if allocation < 200:
            continue
        if path.is_file():
            source = path.read_text(errors="replace")
            text = render_source(source, symbols.get(file, []), ctx.request if ctx else "", allocation)
        else:
            text = "(file does not exist yet, needs to be created)"
        section = header + text + footer
        if len(section) <= remaining:
            sections.append(section)
            remaining -= len(section)
    return "".join(sections)
