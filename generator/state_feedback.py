"""Runtime evidence, bounded repair prompts, and one sequential repair workflow.

Expected values come from test failures or the request, never a reference patch.
Line events describe the state BEFORE that line executes.
"""
import ast
from collections import defaultdict
import json
import hashlib
from pathlib import Path, PurePosixPath
from generator.feedback_evidence import failure_excerpt, evidence_bundle, grade_summary
from generator.feature_memory import (seed_featurespace, attach_observations, memory_files,
                                      memory_view, associated_evidence, merge_update,
                                      save_memory, focused_elements, history_view, UPDATE_SCHEMA)
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


def source_context(instance, current, patch, packet, watch, budget, memory=None):
    from explorer.context_renderer import render_source
    priority_symbols = defaultdict(list)
    primary = []
    for event in packet.get('exceptions', []) + packet.get('state_transitions', []) + packet.get('consumer_guards', []):
        if event.get('file') and production_path(event['file']):
            primary.append(event['file'])
            priority_symbols[event['file']].append(event.get('function', '').split('.')[-1])
    for entry in (memory or {}).get('elements', {}).values():
        if entry.get('status') == 'active':
            priority_symbols[entry['file']].append(entry.get('symbol', '').split('.')[-1])
    focused = []
    if memory:
        for key in sorted(focused_elements(memory)):
            entry = memory.get('elements', {}).get(key)
            if entry:
                focused.append(entry['file'])
    files = list(dict.fromkeys(focused + primary + patch_paths(patch) + memory_files(memory or {}) + list(watch.get('watch', {}))))
    files = [f for f in files if f and production_path(f) and (Path(current) / safe_path(f)).is_file()][:24]
    remaining = budget
    sections, included = [], []
    for index, name in enumerate(files):
        path = Path(current) / safe_path(name)
        path.resolve().relative_to(Path(current).resolve())
        source = path.read_text(errors='replace')
        names = list(dict.fromkeys(priority_symbols[name] + watch.get('watch', {}).get(name, [])))
        minimum = 1200
        try:
            functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            lines = source.splitlines(keepends=True)
            implicated = priority_symbols[name][:2]
            minimum = max([1200] + [len(''.join(lines[n.lineno-1:n.end_lineno]))+700 for n in functions if n.name in implicated])
        except SyntaxError:
            pass
        cap = min(remaining, max(minimum, remaining // max(1, len(files) - index)))
        if cap < 300:
            continue
        block = render_source(source, names, instance['problem_statement'], max(0, cap-120))
        section = '\n[start of %s]\nEvidence ID: code:%s\n%s\n[end of %s]\n' % (name, name, block, name)
        if len(section) <= remaining:
            sections.append(section)
            included.append(name)
            remaining -= len(section)
    return '\n<current_code>\n' + ''.join(sections) + '\n</current_code>\n', included


def repair_prompt(instance, current, patch, packet, output, watch, prior=None, budget=80000,
                  memory=None, history=None, mode='featurespace'):
    header = ('Implement a coordinated repair of the CURRENT source.\n'
              '<requirement>\n' + instance['problem_statement'] + '\n</requirement>\n' + STATE_CONTRACT +
              '\nPreserve passing behavior. Do not modify tests or hard-code test inputs. '
              'Historical observations are patch-specific, not proof about the current candidate.\n')
    if mode == 'featurespace' and memory:
        header += ('\nUse the following updated FeatureSpace to identify the functional groups, dependencies '
                   'and conditions that must be repaired together. Initial-plan conditions are hypotheses; '
                   'retired entries must not constrain the repair. Check the cited evidence against current source.\n'
                   '<featurespace>\n' + json.dumps(memory_view(memory), ensure_ascii=False) + '\n</featurespace>\n'
                   '<featurespace_evidence>\n' + json.dumps(associated_evidence(memory), ensure_ascii=False) + '\n</featurespace_evidence>\n')
    else:
        if mode == 'basic':
            facts = {'grade': grade_summary(packet.get('grade')), 'test_output': failure_excerpt(output)}
        else:
            facts = evidence_bundle(packet, output)
        header += '\n<execution_feedback>\n' + json.dumps(facts, ensure_ascii=False) + '\n</execution_feedback>\n'
        header += '\n<previous_attempts>\n' + json.dumps(history_view(history or []), ensure_ascii=False) + '\n</previous_attempts>\n'
    if prior:
        header += '\nPrevious repair was rejected and NOT applied:\n' + prior[:2000] + '\n'
    tail = ('\nReturn only JSON: {"edits": [{"file_path": "relative/path.py", '
            '"before": "exact contiguous snippet from CURRENT source", "after": "complete replacement"}]}. '
            'Include every required production edit and imports. Keep snippets concise. '
            'Do not add duplicate method definitions or repeat a rejected/ineffective candidate.\n')
    remaining = budget - len(header) - len(tail) - 100
    if remaining < 1000:
        raise ValueError('Feedback instructions leave no source budget')
    code, _ = source_context(instance, current, patch, packet, watch, remaining, memory if mode == 'featurespace' else None)
    return header + code + tail


def diagnosis_prompt(instance, current, patch, packet, watch, memory, bundle, budget=80000):
    header = ('Update a repository-specific FeatureSpace before generating a repair. Do not return code edits.\n'
              '<requirement evidence_id="requirement">\n' + instance['problem_statement'] + '\n</requirement>\n'
              + STATE_CONTRACT + '\n'
              'Explain the first supported break between state production, configuration changes and consuming branches. '
              'Distinguish observed facts from inferred explanations. A default value is not proof of validation. '
              'Missing events in bounded traces are uncertainty, not proof of non-execution. '
              'Explain why related files must change together; revise functional groups, dependencies and conditions. '
              'Initial plan constraints can be wrong: explicitly retire or correct contradicted hypotheses. '
              'Use prior attempt/test deltas to avoid repeating ineffective diagnoses. Preserve passing behavior. '
              'Do not infer unseen expected values, use reference solutions, or introduce test-specific constants. '
              'Cite only supplied ev: IDs, code:<file> IDs, or requirement. Code citations cover only displayed snippets. '
              'Use existing entry IDs to revise claims. New IDs start E:/G:/L:/C:. '
              'Keep the delta small; all inferred claims remain hypotheses until evaluated.\n'
              '<featurespace>\n' + json.dumps(memory_view(memory, 18000), ensure_ascii=False) + '\n</featurespace>\n'
              '<current_execution_evidence>\n' + json.dumps(bundle, ensure_ascii=False) + '\n</current_execution_evidence>\n')
    tail = '\nReturn only a JSON object using this schema; omit unused entries, never invent evidence IDs:\n' + UPDATE_SCHEMA
    remaining = budget - len(header) - len(tail) - 100
    if remaining < 1000:
        raise ValueError('FeatureSpace diagnosis leaves no source budget')
    code, files = source_context(instance, current, patch, packet, watch, remaining, memory)
    return header + code + tail, files


def _duplicate_methods(source):
    counts = defaultdict(int)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # Unedited recalled code can target a different Python language version.
        return counts
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        local = defaultdict(int)
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.decorator_list:
                local[(cls.name, node.name, ast.dump(node, include_attributes=False))] += 1
        # Two independent factory-local classes may legitimately have the same
        # name/methods. Count repetition inside a class, not across those scopes.
        for key, count in local.items():
            counts[key] = max(counts[key], count)
    return counts


def behavior_signature(current, repo, names):
    """Fingerprint changed ASTs/other bytes, independent of diff hunks or comments."""
    changes = {}
    for name in sorted(set(names)):
        def value(root):
            file = Path(root) / safe_path(name)
            if not file.is_file():
                return None
            raw = file.read_bytes()
            if name.endswith('.py'):
                try:
                    return ast.dump(ast.parse(raw), include_attributes=False)
                except (SyntaxError, ValueError):
                    pass
            return hashlib.sha256(raw).hexdigest()
        base, edited = value(repo), value(current)
        if base != edited:
            changes[name] = edited
    return hashlib.sha256(json.dumps(changes, sort_keys=True).encode()).hexdigest()


def repair_once(instance, repo, patch, packet, output, watch, directory, previous_error=None, call=None,
                memory=None, history=None, plan=None, recall=None, round_number=1, mode='featurespace'):
    """Reflect, merge FeatureSpace, then edit a disposable source tree.

    Completed model responses can be replayed after interruption. Invalid
    reflection falls back to the existing memory plus observed evidence; it does
    not prevent the code-generation stage from running.
    """
    from generator.llm_interface import call_llm_with_usage, LLMCallError
    from run_oracle import parse_edits, edits_to_patch
    if mode not in {'featurespace', 'state', 'basic'}:
        raise ValueError('Unknown feedback mode')
    call = call or call_llm_with_usage
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    history = history or []
    patch_hash = hashlib.sha256(patch.encode()).hexdigest()
    binding = {'instance_id': instance['instance_id'], 'patch_sha256': patch_hash, 'mode': mode, 'round': round_number}
    binding_file = directory / 'input.json'
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError('Repair checkpoint belongs to a different input')
    save_memory(binding_file, binding)
    bundle = evidence_bundle(packet, output)
    save_memory(directory / 'evidence.json', bundle)
    memory = memory or seed_featurespace(instance, plan, recall)
    if memory['instance_id'] != instance['instance_id'] or memory.get('base_commit') != instance.get('base_commit'):
        raise ValueError('FeatureSpace belongs to a different instance/base')
    memory = attach_observations(memory, bundle, patch_hash, round_number)
    memory['attempts'] = history
    save_memory(directory / 'featurespace_before.json', memory)
    names = list(dict.fromkeys(patch_paths(patch) + list(watch.get('watch', {})) + memory_files(memory)))
    for event in packet.get('exceptions', []) + packet.get('state_transitions', []) + packet.get('consumer_guards', []):
        if event.get('file') and production_path(event['file']):
            names.append(safe_path(event['file']))
    names = list(dict.fromkeys(names))
    def model_stage(stage, prompt, max_tokens):
        response_file = directory / (stage + '_response.json')
        metrics_file = directory / (stage + '_usage.json')
        (directory / (stage + '_prompt.md')).write_text(prompt)
        if response_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            if metrics.get('finish_reason') in {'length', 'max_tokens'}:
                raise ValueError(stage + ' response truncated')
            return response_file.read_text()
        ledger_path = directory / 'model_calls.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
        ledger.append({'stage': stage, 'status': 'started'})
        save_memory(ledger_path, ledger)
        try:
            response = call(prompt, system=('Update FeatureSpace from execution evidence. Return only JSON.' if stage == 'diagnosis'
                                          else 'Repair current source using the supplied context. Return only JSON edits.'),
                            max_tokens=max_tokens, temperature=0, json_mode=True)
        except LLMCallError as error:
            ledger[-1].update(status='error', metrics=error.metrics)
            save_memory(ledger_path, ledger)
            save_memory(metrics_file, error.metrics)
            raise
        ledger[-1].update(status='completed', metrics=response.metrics())
        save_memory(ledger_path, ledger)
        save_memory(metrics_file, response.metrics())
        temporary = response_file.with_suffix('.json.tmp')
        temporary.write_text(response.text)
        temporary.replace(response_file)
        if response.finish_reason in {'length', 'max_tokens'}:
            raise ValueError(stage + ' response truncated')
        return response.text
    try:
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
            before_signature = behavior_signature(current, repo, names)
            save_memory(directory / 'candidate_before.json', {'patch_sha256': patch_hash, 'behavior_sha256': before_signature})
            if mode == 'featurespace':
                try:
                    prompt, files = diagnosis_prompt(instance, current, patch, packet, watch, memory, bundle)
                    raw = model_stage('diagnosis', prompt, 6144)
                    delta = json.loads(raw)
                    memory = merge_update(memory, delta, files)
                    save_memory(directory / 'featurespace_delta.json', delta)
                except (ValueError, TypeError, KeyError, LLMCallError) as error:
                    # A bad reflection must not make a formerly repairable case empty.
                    save_memory(directory / 'diagnosis_error.json', {'error': type(error).__name__ + ': ' + str(error)[:1500]})
                    memory['diagnosis_status'] = 'fallback_previous_memory_with_current_observations'
            save_memory(directory / 'featurespace_after.json', memory)
            prompt = repair_prompt(instance, current, patch, packet, output, watch, previous_error,
                                   memory=memory, history=history, mode=mode)
            (directory / 'prompt.md').write_text(prompt)
            raw = model_stage('repair', prompt, 16384)
            (directory / 'response.json').write_text(raw)
            edits = parse_edits(raw)
            if not edits or any(not production_path(safe_path(e['file_path'])) for e in edits):
                raise ValueError('Repair needs complete production edits')
            for edit in edits:
                if ('[start of %s]' % edit['file_path']) not in prompt:
                    raise ValueError('Repair requested a file outside current-source evidence')
            previous_duplicates = {name: _duplicate_methods((current / name).read_bytes())
                                   for name in names if name.endswith('.py') and (current / name).is_file()}
            incremental = edits_to_patch(edits, current)
            if not incremental.strip():
                raise ValueError('Repair made no changes')
            git('apply', '--check', '-', input=incremental)
            git('apply', '-', input=incremental)
            (directory / 'incremental.patch').write_text(incremental)
            for file in patch_paths(incremental):
                if file.endswith('.py') and (current / file).is_file():
                    source = (current / file).read_bytes()
                    compile(source, file, 'exec')
                    duplicates = _duplicate_methods(source)
                    if any(count > max(1, previous_duplicates.get(file, {}).get(key, 0)) for key, count in duplicates.items()):
                        raise ValueError('Repair introduces an identical duplicate method')
            after_signature = behavior_signature(current, repo, names + patch_paths(incremental))
            if after_signature == before_signature:
                raise ValueError('Repair changes no executable structure (same AST/content)')
            seen = {h.get(k) for h in history for k in ['behavior_before', 'behavior_after'] if h.get(k)}
            if after_signature in seen:
                raise ValueError('Repair repeats an earlier candidate (cycle)')
            git('add', '-N', '.')
            combined = git('diff', 'HEAD', '--binary', '--no-ext-diff').stdout
            checked = subprocess.run(['git', 'apply', '--check', '-'], cwd=repo, input=combined,
                                     text=True, capture_output=True, timeout=30)
            if checked.returncode:
                raise ValueError('Combined repair does not apply to base: ' + checked.stderr[:2000])
            save_memory(directory / 'candidate_after.json', {
                'patch_sha256': hashlib.sha256(combined.encode()).hexdigest(), 'behavior_sha256': after_signature,
                'changed_files': patch_paths(incremental)})
            # Publish the completed patch last; this is the controller's resume marker.
            complete = directory / 'combined.patch.tmp'
            complete.write_text(combined)
    finally:
        ledger_path = directory / 'model_calls.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
        metrics = [entry.get('metrics', {}) for entry in ledger]
        save_memory(directory / 'llm_usage.json', {
            'logical_calls': len(metrics), 'provider_attempts': sum(m.get('attempts', 1) for m in metrics),
            'reported_total_tokens': sum(m.get('total_tokens') or 0 for m in metrics),
            'usage_unknown_calls': sum(m.get('total_tokens') is None for m in metrics),
            'stages': {p.stem: json.loads(p.read_text()) for p in directory.glob('*_usage.json') if p.name != 'llm_usage.json'}})
    complete.replace(directory / 'combined.patch')
    return combined
