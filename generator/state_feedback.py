"""Runtime evidence, bounded repair prompts, and one sequential repair workflow.

Expected values come from test failures or the request, never a reference patch.
Line events describe the state BEFORE that line executes.
"""
import ast
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile

STATE_CONTRACT = """\
Trace feature state from its producer through cloning/configuration changes to each consumer guard.
A default field value is not evidence that validation ran for that object and configuration.
Keep separate validity/permission states separate, and keep identity and selected configuration consistent.
Evaluate a guard after its required state update; preserve existing dispatch/cache/fallback behavior.
Use ancestral coordinated changes to locate producer and consumer responsibilities, not to copy obsolete code.
"""


def safe_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in {'..', '.git'} for p in path.parts):
        raise ValueError('Unsafe source path')
    return str(path)


def production_path(name):
    parts = PurePosixPath(name).parts
    return not (set(parts) & {'test', 'tests', 'doc', 'docs'} or
                (parts and (parts[-1].startswith('test_') or parts[-1].endswith('_test.py'))))


def patch_paths(patch):
    names = []
    for a, b in re.findall(r'^diff --git a/(.*?) b/(.*?)$', patch, re.M):
        names.extend([safe_path(a), safe_path(b)])
    return list(dict.fromkeys(names))


def watch_config(repo, instance, patch, recall=None, output=''):
    """Choose current-code functions; include state writers in recalled modules."""
    paths = patch_paths(patch)
    symbols = defaultdict(list)
    declared_symbols = defaultdict(list)
    for group in instance.get('new_components') or []:
        paths.append(group['file'])
        symbols[group['file']].extend(c['name'].split('.')[-1] for c in group['components'])
        declared_symbols[group['file']].extend(c['name'].split('.')[-1] for c in group['components'])
    for item in (recall or {}).get('candidates', []):
        paths.append(item['file'])
        symbols[item['file']].extend(n.split('.')[-1] for n in item.get('symbols', []))
    for name in re.findall(r'(?:/testbed/|File ["\'])([^"\'\s:]+\.py)', output):
        paths.append(name)
    changed_by_file = defaultdict(list)
    for section in patch.split('diff --git ')[1:]:
        first = section.splitlines()[0]
        match = re.match(r'a/(.*?) b/(.*)', first)
        if match:
            changed_by_file[match.group(2)].extend(re.findall(r'^\+[^+]*?\bdef (\w+)', section, re.M))
    watch = {}
    terms = set(re.findall(r'[a-zA-Z_]{4,}', instance['problem_statement'].lower()))
    for name in dict.fromkeys(paths):
        try:
            name = safe_path(name)
            if not production_path(name) or not name.endswith('.py'):
                continue
            path = Path(repo) / name
            path.resolve().relative_to(Path(repo).resolve())
            source = path.read_text(errors='replace') if path.is_file() else ''
            tree = ast.parse(source)
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            changed = set(changed_by_file[name])
            ordered = sorted(functions, key=lambda n: (
                n.name not in symbols[name] and n.name not in changed,
                -len(terms & set(re.findall(r'[a-zA-Z_]+', n.name.lower()))), n.lineno))
            watch[name] = list(dict.fromkeys(changed_by_file[name] + symbols[name] + [n.name for n in ordered[:20]]))
        except (ValueError, OSError, SyntaxError):
            continue
        if len(watch) >= 24:
            break
    if instance['repo'] == 'conan-io/conan':
        from generator.state_profiles.conan_probe import WATCH
        focused = {}
        for name in watch:
            base = Path(name).name
            if base in WATCH:
                focused[name] = list(dict.fromkeys(sorted(WATCH[base]) + changed_by_file[name] + declared_symbols[name]))
            elif changed_by_file[name] or declared_symbols[name]:
                focused[name] = list(dict.fromkeys(changed_by_file[name] + declared_symbols[name]))
        watch = focused
    return {'schema_version': 1, 'watch': watch, 'repo': instance['repo'],
            'event_limit_per_process': 12000, 'event_limit_per_test': 1600,
            'line_timing': 'before_execution', 'reference_patch_used': False}


def extract_packet(events, report=None, trace_meta=None):
    """Keep actual transitions, binding identities and last guards near failure."""
    by_test = defaultdict(list)
    errors, transitions, guards = [], [], []
    previous = {}
    for event in events:
        test = event.get('test', 'unscoped')
        by_test[test].append(event)
        key = (event.get('process'), test, event.get('frame'), event.get('function'))
        values = event.get('values', {})
        prior = previous.get(key)
        if prior is not None and values != prior:
            changes = {k: {'before': prior.get(k), 'after': v} for k, v in values.items() if prior.get(k) != v}
            if changes:
                transitions.append(dict(event, changes=changes))
        previous[key] = values
        if event.get('event') == 'exception':
            errors.append(event)
        if event.get('statement', '').startswith(('if ', 'elif ', 'while ')):
            guards.append(event)
    status = (report or {}).get('tests_status', {})
    failures = list(dict.fromkeys(t for group in status.values() for t in group.get('failure', [])))
    relevant = lambda e: not failures or e.get('test') in failures or e.get('test') in {'unscoped', 'collection'}
    tails = {t: es[-12:] for t, es in by_test.items() if t in failures}
    if not tails and events:
        tails = {'unscoped': events[-24:]}
    packet = {'schema_version': 1, 'reference_patch_or_execution_used': False,
              'line_timing': 'before_execution', 'trace_metadata': trace_meta or {},
              'failed_tests': failures, 'grade': report or {},
              'exceptions': [e for e in errors if relevant(e)][-10:],
              'state_transitions': [e for e in transitions if relevant(e)][-20:],
              'consumer_guards': [e for e in guards if relevant(e)][-12:],
              'failure_trace_tails': dict(list(tails.items())[:8]),
              'limits': ['Observed values are not invented expected values.',
                         'A guard event records values before evaluating the condition.',
                         'Tracing is bounded and may not cover every producer.']}
    # Conan adapter uses receiver/configuration facts; no instance-specific fix.
    def scoped_receivers(value, process):
        if isinstance(value, dict):
            return {k: (str(process)+':'+str(v) if k == 'receiver' and v is not None else scoped_receivers(v, process))
                    for k,v in value.items()}
        if isinstance(value, list):
            return [scoped_receivers(v, process) for v in value]
        return value
    conan = [dict(e, state=scoped_receivers(e['state'], e.get('process'))) for e in events if 'state' in e]
    if conan:
        from generator.state_profiles.conan import extract
        packet['conan_state_relations'] = extract(conan)
    return packet


def failure_excerpt(output, budget=16000):
    lines = output.splitlines(keepends=True)
    regions = set(range(max(0, len(lines) - 45), len(lines)))
    for i, line in enumerate(lines):
        if re.search(r'FAILED|ERROR|AssertionError|Traceback|^E\s|Error:|FAIL:', line):
            regions.update(range(max(0, i - 8), min(len(lines), i + 12)))
    text = ''.join(lines[i] for i in sorted(regions))
    return text[:budget // 2] + ('\n... diagnostic output omitted ...\n' + text[-budget // 2:] if len(text) > budget else text[budget // 2:])


def repair_prompt(instance, current, patch, packet, output, watch, prior=None, budget=80000):
    from explorer.context_renderer import render_source
    header = ('Repair the CURRENT implementation using actual failing-test evidence.\n'
              + instance['problem_statement'] + '\n' + STATE_CONTRACT
              + '\nFind the earliest broken producer/configuration/consumer relation supported by evidence. '
                'Verify import bindings, object identity, configuration and state read/write order. '
                'Use assertion/expected values only from test output or the request. '
                'Preserve passing behavior and do not modify tests or hard-code test inputs.\n'
              + '\n<failure_output>\n' + failure_excerpt(output) + '\n</failure_output>\n')
    compact = dict(packet)
    for key in ['state_transitions', 'consumer_guards', 'exceptions']:
        compact[key] = packet.get(key, [])[-8:]
    facts = json.dumps(compact, ensure_ascii=False, indent=1)
    # Remove whole observations if needed; never truncate JSON into invalid text.
    while len(facts) > 24000:
        longest = max(['state_transitions', 'consumer_guards', 'exceptions', 'failure_trace_tails', 'conan_state_relations'],
                      key=lambda k: len(json.dumps(compact.get(k, {}))))
        value = compact.get(longest)
        if isinstance(value, list) and value:
            compact[longest] = value[len(value)//2 + 1:]
        elif isinstance(value, dict) and value:
            compact[longest] = dict(list(value.items())[len(value)//2 + 1:])
        else:
            break
        facts = json.dumps(compact, ensure_ascii=False, indent=1)
    header += '\n<runtime_state_packet>\n' + facts + '\n</runtime_state_packet>\n'
    if prior:
        header += '\nPrevious repair failed preflight; it was NOT applied:\n' + prior[:2000] + '\n'
    tail = ('\nReturn only JSON: {"edits": [{"file_path": "relative/path.py", '
            '"before": "exact contiguous snippet from CURRENT source", "after": "complete replacement"}]}. '
            'Include every required production edit and imports. Keep snippets concise.\n')
    priority_symbols = defaultdict(list)
    primary = []
    for e in packet.get('exceptions', []) + packet.get('state_transitions', []) + packet.get('consumer_guards', []):
        if e.get('file'):
            primary.append(e['file'])
            priority_symbols[e['file']].append(e.get('function', '').split('.')[-1])
    files = list(dict.fromkeys(primary + patch_paths(patch) + list(watch.get('watch', {}))))
    files = [f for f in files if f and production_path(f)]
    remaining = budget - len(header) - len(tail) - 50
    if remaining < 1000:
        raise ValueError('Feedback instructions leave no source budget')
    sections = []
    for index, name in enumerate(files[:24]):
        path = Path(current) / safe_path(name)
        path.resolve().relative_to(Path(current).resolve())
        if not path.is_file():
            continue
        source = path.read_text(errors='replace')
        names = list(dict.fromkeys(priority_symbols[name] + watch.get('watch', {}).get(name, [])))
        minimum = 1200
        try:
            functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            lines = source.splitlines(keepends=True)
            # Reserve the actual implicated function's body, including imports/header.
            implicated = priority_symbols[name][:2]
            minimum = max([1200]+[len(''.join(lines[n.lineno-1:n.end_lineno]))+700 for n in functions if n.name in implicated])
        except SyntaxError:
            pass
        cap = min(remaining, max(minimum, remaining // max(1, len(files[:24]) - index)))
        block = render_source(source, names, instance['problem_statement'], max(0, cap-90))
        section = '\n[start of %s]\n%s\n[end of %s]\n' % (name, block, name)
        if len(section) <= remaining:
            sections.append(section)
            remaining -= len(section)
    return header + '\n<current_code>\n' + ''.join(sections) + '\n</current_code>\n' + tail


def repair_once(instance, repo, patch, packet, output, watch, directory, previous_error=None, call=None):
    """Apply edits to a disposable source tree and export a combined base diff."""
    from generator.llm_interface import call_llm_with_usage, LLMCallError
    from run_oracle import parse_edits, edits_to_patch
    call = call or call_llm_with_usage
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    names = list(dict.fromkeys(patch_paths(patch) + list(watch.get('watch', {}))))
    with tempfile.TemporaryDirectory(prefix='fea-state-repair-') as temp:
        current = Path(temp)
        for name in names:
            source = Path(repo) / safe_path(name)
            source.resolve().relative_to(Path(repo).resolve())
            if source.is_file():
                dest = current / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        def git(*args, **kwargs):
            return subprocess.run(['git', *args], cwd=current, text=True, capture_output=True, check=True, **kwargs)
        git('init', '-q')
        git('add', '.')
        git('-c', 'user.name=FEA state feedback', '-c', 'user.email=state-feedback@example.invalid',
            'commit', '--allow-empty', '-qm', 'isolated base source')
        if patch.strip():
            git('apply', '--whitespace=nowarn', '-', input=patch)
        prompt = repair_prompt(instance, current, patch, packet, output, watch, previous_error)
        (directory / 'prompt.md').write_text(prompt)
        try:
            response = call(prompt, system='Repair from failing-test runtime evidence and current source. Return only JSON edits.',
                            max_tokens=16384, temperature=0, json_mode=True)
        except LLMCallError as error:
            (directory / 'llm_usage.json').write_text(json.dumps(error.metrics, indent=2))
            raise
        (directory / 'llm_usage.json').write_text(json.dumps(response.metrics(), indent=2))
        (directory / 'response.json').write_text(response.text)
        if response.finish_reason in {'length', 'max_tokens'}:
            raise ValueError('Feedback response truncated')
        edits = parse_edits(response.text)
        if not edits or any(not production_path(safe_path(e['file_path'])) for e in edits):
            raise ValueError('Repair needs complete production edits')
        # The model may only modify files supplied in its current-source context.
        for edit in edits:
            if ('[start of %s]' % edit['file_path']) not in prompt:
                raise ValueError('Repair requested a file outside current-source evidence')
        incremental = edits_to_patch(edits, current)
        if not incremental.strip():
            raise ValueError('Repair made no changes')
        git('apply', '--check', '-', input=incremental)
        git('apply', '-', input=incremental)
        (directory / 'incremental.patch').write_text(incremental)
        for file in patch_paths(incremental):
            if file.endswith('.py') and (current / file).is_file():
                compile((current / file).read_text(), file, 'exec')
        git('add', '-N', '.')
        combined = git('diff', 'HEAD', '--binary', '--no-ext-diff').stdout
        checked = subprocess.run(['git', 'apply', '--check', '-'], cwd=repo, input=combined,
                                 text=True, capture_output=True, timeout=30)
        if checked.returncode:
            raise ValueError('Combined repair does not apply to base: ' + checked.stderr[:2000])
        (directory / 'combined.patch').write_text(combined)
        return combined
