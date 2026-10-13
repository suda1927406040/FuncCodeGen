"""Deterministic edit normalization and feedback from the current base tree."""
import ast
from collections import Counter
import json
from pathlib import Path
import re


OMISSION = re.compile(r'^[ \t]*# \.\.\. other (?:source blocks|resource lines) omitted \.\.\.[ \t]*$', re.MULTILINE)


def normalize_markers(text):
    corrections = []
    markers = re.compile(r'^\[(start|end) of the snippet (before|after) editing in ([^\]\n]+)\]$', re.MULTILINE)
    for block in re.finditer(r'<edit>(.*?)(?:</edit>|(?=<edit>)|$)', text, re.DOTALL):
        found = list(markers.finditer(block.group(1)))
        if len(found) != 4 or len({m[3] for m in found}) != 1:
            continue
        if [(m[1], m[2]) for m in found] == [('start', 'before'), ('end', 'before'),
                                                  ('start', 'after'), ('end', 'before')]:
            m = found[-1]
            corrections.append((block.start(1) + m.start(), block.start(1) + m.end(),
                                m.group().replace('snippet before editing', 'snippet after editing')))
    for start, end, value in reversed(corrections):
        text = text[:start] + value + text[end:]
    return text


def unique_edits(edits):
    seen, result = set(), []
    for edit in edits:
        key = (edit['file_path'], edit['before'], edit['after'])
        if key not in seen:
            result.append(edit)
            seen.add(key)
    return result


def align_indent(original, after):
    """Use shared source anchors; a new sibling can lower the minimum indent."""
    def meaningful(lines):
        return [(line.strip(), len(line) - len(line.lstrip())) for line in lines if line.strip()]
    old, new = meaningful(original), meaningful(after)
    old_counts, new_counts = Counter(x[0] for x in old), Counter(x[0] for x in new)
    new_indent = dict(new)
    offsets = [indent - new_indent[text] for text, indent in old
               if old_counts[text] == new_counts[text] == 1]
    if 0 in offsets:
        return after
    if offsets and len(set(offsets)) == 1:
        delta = offsets[0]
    elif old and new:
        delta = old[0][1] - new[0][1]
    else:
        delta = 0
    return [(' ' * delta + line if delta >= 0 else line[min(-delta, len(line) - len(line.lstrip())):])
            if line.strip() else line for line in after]


def expand_omissions(before, after, source):
    """Preserve hidden source between uniquely ordered islands in both snippets."""
    before_markers, after_markers = list(OMISSION.finditer(before)), list(OMISSION.finditer(after))
    if not before_markers and not after_markers:
        return before, after
    if len(before_markers) != len(after_markers) or not before_markers:
        raise ValueError('Omission placeholders must be preserved in both snippets')
    pieces = [p.strip() for p in OMISSION.split(before)]
    if any(not p for p in pieces):
        raise ValueError('Omission needs nonempty source anchors on both sides')
    candidates = [([], 0)]
    for piece in pieces:
        next_candidates = []
        for spans, position in candidates:
            while True:
                offset = source.find(piece, position)
                if offset < 0:
                    break
                next_candidates.append((spans + [(offset, offset + len(piece))], offset + len(piece)))
                position = offset + 1
                if len(next_candidates) > 32:
                    raise ValueError('Omission anchors are ambiguous')
        candidates = next_candidates
    if len(candidates) != 1:
        raise ValueError('Omission anchors do not identify one region')
    spans = candidates[0][0]
    replacement = after
    for index, marker in reversed(list(enumerate(after_markers))):
        hidden = source[spans[index][1]:spans[index + 1][0]].strip('\n')
        replacement = replacement[:marker.start()] + hidden + replacement[marker.end():]
    start = source.rfind('\n', 0, spans[0][0]) + 1
    end = spans[-1][1]
    return source[start:end], replacement


def new_module_definitions(before, after, source):
    """Recover pure additions when copied surrounding excerpts are not adjacent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    current = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    declarations = list(re.finditer(r'^(?:async\s+def|def|class)\s+(\w+)\b', after, re.MULTILINE))
    old_names = set(re.findall(r'^(?:async\s+def|def|class)\s+(\w+)\b', before, re.MULTILINE))
    additions, spans = [], []
    for i, match in enumerate(declarations):
        if match[1] in old_names or match[1] in current:
            continue
        end = declarations[i + 1].start() if i + 1 < len(declarations) else len(after)
        chunk = after[match.start():end].rstrip() + '\n'
        try:
            body = ast.parse(chunk).body
        except SyntaxError:
            return None
        if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return None
        additions.append(chunk)
        spans.append((match.start(), end))
    if not additions:
        return None
    retained = after
    for start, end in reversed(spans):
        retained = retained[:start] + retained[end:]
    # Only unchanged surrounding code can be ignored, never another requested edit.
    significant = lambda text: [line.rstrip() for line in text.splitlines() if line.strip()]
    if significant(retained) != significant(before):
        return None
    return '\n'.join(additions)


def repair_prompt(instance, repo_dir, edits, error, previous, output_budget=80000,
                  structural_obligations=None):
    """Show contiguous actual source rather than asking the model to reuse elisions."""
    statement = instance.get('problem_statement', '')
    declared = instance.get('new_components') or []
    prefix = ('Fix the previous implementation so its complete edits apply to the CURRENT BASE below.\n'
              'Implement every requested component. Preserve existing behavior. Do not edit tests or documentation.\n'
              'Return one JSON object: {"edits": [{"file_path": "relative/path.py", "before": "exact current code", '
              '"after": "complete replacement code"}]}. For a new file, before is empty and after is the full file.\n'
              'Use short, unique, contiguous before snippets. Copy them exactly, including indentation. '
              'Separate excerpts are not adjacent in the real file. Never include omission placeholders. '
              'Avoid duplicate edits and ensure later edits refer to code after earlier edits. '
              'Return all edits for the implementation, including unchanged edits from the previous response.\n'
              '<request_and_definitions>\n' + statement + '\n' + json.dumps(declared, ensure_ascii=False)
              + '\n</request_and_definitions>\nValidation failure: ' + str(error)[:2000] + '\n')
    if structural_obligations:
        prefix += ('<accepted_structural_obligations>\n'
                   + json.dumps(structural_obligations, ensure_ascii=False, indent=2)
                   + '\n</accepted_structural_obligations>\n'
                     'These obligations came from the checked repository lifecycle. Modify each named role symbol '
                     'so the generated source satisfies the mapping; a call from a different symbol does not satisfy '
                     'the dispatcher role.\n')
    history = '\n<previous_response>\n' + previous[:16000] + '\n</previous_response>\n'
    remaining = output_budget - len(prefix) - len(history) - 100
    if remaining < 1000:
        history = '\n<previous_response>\n' + previous[:2000] + '\n</previous_response>\n'
        remaining = output_budget - len(prefix) - len(history) - 100
    files = list(dict.fromkeys([e['file_path'] for e in edits] + [g['file'] for g in declared]))
    sections = []
    root = Path(repo_dir).resolve()
    for file in files:
        path = (root / file).resolve()
        if not path.is_relative_to(root) or '.git' in Path(file).parts:
            continue
        header = '\nFILE ' + file + '\n'
        cap = min(22000, max(1000, remaining // max(1, len(files) - len(sections))))
        if not path.is_file():
            body = '(new file; write its full implementation)\n'
        else:
            source = path.read_text(errors='replace')
            lines = source.splitlines(keepends=True)
            if len(source) <= cap:
                body = 'CONTIGUOUS COMPLETE FILE:\n' + source
            else:
                wanted = set()
                for edit in edits:
                    if edit['file_path'] == file:
                        for match in re.finditer(r'\b(?:def|class)\s+(\w+)', edit['before']):
                            wanted.add(match[1])
                for group in declared:
                    if group['file'] == file:
                        wanted.update(c['name'].split('.')[-1] for c in group['components'])
                windows = [(0, min(45, len(lines)))]
                try:
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in wanted:
                            windows.append((max(0, node.lineno - 8), min(len(lines), node.end_lineno + 7)))
                except SyntaxError:
                    pass
                for edit in edits:
                    if edit['file_path'] != file:
                        continue
                    anchors = [x.strip() for x in edit['before'].splitlines() if x.strip() and not x.lstrip().startswith('#')]
                    for anchor in anchors[:1] + anchors[-1:]:
                        for number, line in enumerate(lines):
                            if line.strip() == anchor:
                                windows.append((max(0, number - 30), min(len(lines), number + 70)))
                excerpts, used = [], set()
                size = 0
                for start, end in windows:
                    if (start, end) in used:
                        continue
                    used.add((start, end))
                    chunk = ''.join(lines[start:end])
                    if size + len(chunk) + 100 > cap:
                        continue
                    excerpts.append(f'CONTIGUOUS EXCERPT lines {start + 1}-{end}:\n' + chunk)
                    size += len(chunk) + 100
                body = '\nEND EXCERPT\n\n'.join(excerpts)
        section = header + body
        if len(section) <= remaining:
            sections.append(section)
            remaining -= len(section)
    prompt = prefix + '\n<current_base>\n' + ''.join(sections) + '\n</current_base>\n' + history
    if len(prompt) > output_budget:
        raise ValueError('Repair instructions exceed the context budget')
    return prompt
