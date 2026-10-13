"""Resolve explicit function replacements against the current candidate AST."""
import ast
import json
import textwrap


def current_symbol_source(source, symbol, limit=4000):
    """Bounded current source of a qualified Class.method or function."""
    nodes = []
    def visit(body, prefix=''):
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                if name == symbol:
                    nodes.append(node)
                visit(node.body, name + '.')
    try:
        visit(ast.parse(source).body)
    except (SyntaxError, ValueError):
        return ''
    if len(nodes) != 1:
        return ''
    start = min([nodes[0].lineno] + [d.lineno for d in nodes[0].decorator_list]) - 1
    text = ''.join(source.splitlines(keepends=True)[start:nodes[0].end_lineno])
    return text[:limit]


def parse_current_edits(raw, current):
    from generator.state_feedback import safe_path
    from run_oracle import parse_edits
    try:
        payload = json.loads(raw)
    except ValueError:
        return parse_edits(raw)
    entries = payload.get('edits', [])
    if not any(isinstance(e, dict) and 'symbol' in e for e in entries):
        return parse_edits(raw)
    result = []
    seen = set()
    for entry in entries:
        if 'symbol' in entry and 'replacement' not in entry and 'before' in entry and 'after' in entry:
            # Hybrid form: the model named the symbol but supplied a snippet.
            # Fall back to the snippet edit; it anchors on the same function.
            entry = {k: entry[k] for k in ('file_path', 'before', 'after')}
        if 'symbol' not in entry:
            if not all(isinstance(entry.get(k), str) for k in ['file_path', 'before', 'after']):
                raise ValueError('Incomplete text edit')
            result.append(entry)
            continue
        file = safe_path(entry.get('file_path', ''))
        symbol = entry['symbol']
        source = (current / file).read_text()
        lines = source.splitlines(keepends=True)
        nodes = []
        def visit(body, prefix=''):
            for node in body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + node.name
                    if name == symbol and not isinstance(node, ast.ClassDef):
                        nodes.append(node)
                    visit(node.body, name + '.')
        visit(ast.parse(source).body)
        if len(nodes) != 1 or (file, symbol) in seen:
            raise ValueError('Function replacement needs one unambiguous current symbol: ' + str(symbol))
        seen.add((file, symbol))
        replacement = textwrap.dedent(entry['replacement']).strip('\n') + '\n'
        parsed = ast.parse(replacement)
        named = [node for node in parsed.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == nodes[0].name]
        if len(named) != 1:
            raise ValueError('Replacement must contain the named function exactly once: ' + str(symbol))
        if len(parsed.body) > 1:
            # Appended sibling definitions/helpers are trimmed instead of
            # wasting a repair round: the resolver replaces exactly the named
            # function, and any trimming-induced name errors surface in the
            # compile preflight with a retry message.
            segment = ast.get_source_segment(replacement, named[0])
            if segment is None:
                segment = ast.unparse(named[0])
            replacement = textwrap.dedent(segment).strip('\n') + '\n'
        node = nodes[0]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        indent = lines[node.lineno-1][:node.col_offset]
        # Preserve method binding decorators unless a full decorator list is supplied.
        if node.decorator_list and not named[0].decorator_list:
            replacement = textwrap.dedent(''.join(lines[start:node.lineno-1])) + replacement
        result.append({'file_path': file, 'symbol': symbol,
                       'before': ''.join(lines[start:node.end_lineno]).rstrip('\n'),
                       'after': textwrap.indent(replacement, indent).rstrip('\n')})
    return result
