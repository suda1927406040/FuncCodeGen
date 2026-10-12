"""Resolve explicit function replacements against the current candidate AST."""
import ast
import json
import textwrap


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
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)) or parsed.body[0].name != nodes[0].name:
            raise ValueError('Replacement must contain exactly the named function, not appended sibling definitions')
        node = nodes[0]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        indent = lines[node.lineno-1][:node.col_offset]
        # Preserve method binding decorators unless a full decorator list is supplied.
        if node.decorator_list and not parsed.body[0].decorator_list:
            replacement = textwrap.dedent(''.join(lines[start:node.lineno-1])) + replacement
        result.append({'file_path': file, 'symbol': symbol,
                       'before': ''.join(lines[start:node.end_lineno]).rstrip('\n'),
                       'after': textwrap.indent(replacement, indent).rstrip('\n')})
    return result
