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
                                      save_memory, focused_elements, history_view,
                                      execution_obligations, preserve_runtime_observations,
                                      UPDATE_SCHEMA)
import re
import shutil
import subprocess
import tempfile
import textwrap

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
        declared_symbols[group['file']].extend(c['name'] for c in group['components'])
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
                declared_leaves = [symbol.split('.')[-1] for symbol in declared_symbols[name]]
                focused[name] = list(dict.fromkeys(sorted(WATCH[base]) + changed_by_file[name] + declared_leaves))
            elif changed_by_file[name] or declared_symbols[name]:
                focused[name] = list(dict.fromkeys(changed_by_file[name]
                                                   + [symbol.split('.')[-1] for symbol in declared_symbols[name]]))
        watch = focused
    from generator.feedback_context import expand_producers
    expansion = expand_producers(repo, watch, patch)
    for item in expansion:
        watch[item['file']] = item['symbols']
    return {'schema_version': 1, 'watch': watch, 'repo': instance['repo'],
            'source_expansion': expansion,
            'declared_interfaces': dict(declared_symbols),
            'event_limit_per_process': 40000, 'event_limit_per_test': 5000,
            'line_timing': 'before_execution', 'reference_patch_used': False}


def extract_packet(events, report=None, trace_meta=None, watch=None):
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
    # Branch outcomes record which suite a guard actually took; the observer
    # derives them from the next executed line in the same frame, never by
    # re-evaluating the condition.
    branches = [e for e in events if e.get('branch') and relevant(e)]
    branch_outcomes = [{
        'test': e['test'], 'file': e.get('file'), 'function': e.get('function'), 'line': e.get('line'),
        'command': e.get('command'), 'invocation': e.get('invocation'),
        'condition': e['branch'].get('condition'), 'entered_body': e['branch'].get('outcome'),
        'next_line': e['branch'].get('next_line'), 'body_calls': e['branch'].get('body_calls', []),
        'values_before': e['branch'].get('values_before'), 'state_before': e['branch'].get('state_before'),
        'basis': e['branch'].get('basis'), 'evidence': {'seq': e.get('seq'), 'process': e.get('process')}}
        for e in branches[-24:]]
    # Only a complete bounded trace can confirm a watched function was never
    # called in a failed test; otherwise absence stays explicitly unknown.
    meta = trace_meta or {}
    reasons = []
    if meta.get('trace_status') != 'observed':
        reasons.append('trace_status=' + str(meta.get('trace_status')))
    if meta.get('extraction_truncated'):
        reasons.append('extraction_truncated')
    if meta.get('timed_out'):
        reasons.append('diagnosis_timed_out')
    summaries = meta.get('observer_summaries') or []
    observed_processes = {e.get('process') for e in events}
    summarized_processes = {s.get('process') for s in summaries if s.get('finished')}
    if not summaries or observed_processes - summarized_processes:
        reasons.append('missing_complete_process_summary')
    for summary in summaries:
        if 'truncated_tests' not in summary:
            reasons.append('missing_per_test_limit_status')
        if summary.get('truncated') or summary.get('truncated_tests'):
            reasons.append('process_event_limit=' + str(summary.get('process')))
        if summary.get('observer_errors'):
            reasons.append('observer_errors=' + str(summary.get('observer_errors')))
    truncated_tests = sorted({t for s in summaries for t in s.get('truncated_tests', [])})
    coverage = {'status': 'partial' if reasons else 'complete', 'reasons': reasons[:8],
                'truncated_tests': truncated_tests[:12]}
    coverage_records = []
    if watch:
        complete = coverage['status'] == 'complete'
        for file, names in (watch.get('watch') or {}).items():
            called = defaultdict(set)
            for event in events:
                if event.get('event') == 'call' and event.get('file') == file:
                    leaf = event.get('function', '').split('.')[-1]
                    if leaf in names:
                        called[leaf].add(event.get('test', 'unscoped'))
            for name in names:
                called_failed = sorted(t for t in called.get(name, ()) if not failures or t in failures)
                not_called = [t for t in failures if t not in called.get(name, ())]
                if not called_failed and not not_called:
                    continue
                per_test = [{'test': t,
                             'status': 'not_called_confirmed' if complete and t in by_test and t not in truncated_tests
                                       else 'not_observed_unknown'} for t in not_called[:6]]
                coverage_records.append({'file': file, 'function': name,
                                         'called_in_failed_tests': called_failed[:6],
                                         'not_called': per_test})
        # Functions absent from failed tests are the most diagnostic; list them first.
        coverage_records.sort(key=lambda record: not record['not_called'])
        coverage_records = coverage_records[:60]
    # Successful examples are observations of this candidate, not rules inferred
    # from test names. Preserve a few contrasting paths alongside failure evidence.
    passing = [t for group in status.values() for t in group.get('success', [])]
    successful = []
    for test in passing:
        observed = by_test.get(test, [])
        returns = [e for e in observed if e.get('event') == 'return']
        if returns:
            successful.append({'test': test, 'outcome': 'passed', 'events': [
                {k: e[k] for k in ['file', 'function', 'command', 'invocation', 'state', 'return_value'] if k in e}
                for e in returns[-3:]]})
    invocations = []
    scopes = defaultdict(list)
    for e in events:
        scopes[(e.get('test'), e.get('process'), e.get('invocation'), e.get('command'))].append(e)
    for (test, process, invocation, command), scoped in scopes.items():
        if not relevant(scoped[-1]):
            continue
        calls = defaultdict(int)
        for e in scoped:
            if e.get('event') == 'call':
                calls[e.get('file', '') + ':' + e.get('function', '')] += 1
        invocations.append({'test': test, 'process': process, 'invocation': invocation, 'command': command,
                            'calls': dict(calls), 'coverage': coverage['status']})
    tails = {t: es[-12:] for t, es in by_test.items() if t in failures}
    if not tails and events:
        tails = {'unscoped': events[-24:]}
    packet = {'schema_version': 1, 'reference_patch_or_execution_used': False,
              'line_timing': 'before_execution', 'trace_metadata': trace_meta or {},
              'failed_tests': failures, 'grade': report or {},
              'exceptions': [e for e in errors if relevant(e)][-10:],
              'state_transitions': [e for e in transitions if relevant(e)][-20:],
              'consumer_guards': [e for e in guards if relevant(e)][-12:],
              'branch_outcomes': branch_outcomes,
              'invocation_calls': invocations[-16:],
              'successful_execution': successful[:6],
              'trace_coverage': coverage,
              'watched_call_coverage': coverage_records if watch is not None else [],
              'failure_trace_tails': dict(list(tails.items())[:8]),
              'limits': ['Observed values are not invented expected values.',
                         'A guard event records values before evaluating the condition.',
                         'Tracing is bounded and may not cover every producer.',
                         'Branch outcomes state which suite a guard took, derived from the next executed line in the same frame.',
                         'watched_call_coverage: not_called_confirmed holds only for failed tests under complete coverage; not_observed_unknown cannot decide.']}
    routes = []
    declared = (watch or {}).get('declared_interfaces', {})
    target_names = {name.split('.')[-1] for names in declared.values() for name in names}
    for test in failures:
        samples = by_test.get(test, [])
        if not samples:
            continue
        # One pytest test may execute multiple CLI commands. The last observed
        # command is kept separate from earlier expected failures in that test.
        last = samples[-1]
        samples = [e for e in samples if (e.get('process'), e.get('invocation')) == (last.get('process'), last.get('invocation'))]
        targets = []
        for file, names in declared.items():
            for name in names:
                leaf = name.split('.')[-1]
                calls = [e for e in samples if e.get('event') == 'call' and e.get('file') == file
                         and e.get('function','').split('.')[-1] == leaf]
                targets.append({'file': file, 'function': name, 'calls': len(calls),
                                'status': 'called' if calls else 'not_called_confirmed' if coverage['status'] == 'complete' else 'unknown'})
        blocked = []
        for event in samples:
            branch = event.get('branch', {})
            if branch.get('outcome') is False and target_names.intersection(branch.get('body_calls', [])):
                state = branch.get('state_before') or {}
                blocked.append({'file': event.get('file'), 'function': event.get('function'),
                                'line': branch['guard_line'], 'condition': branch['condition'],
                                'skipped_target_calls': sorted(target_names.intersection(branch['body_calls'])),
                                'node': state.get('node'), 'current_info': state.get('current_info'),
                                'seq': event.get('seq')})
        routes.append({'test': test, 'command': last.get('command'), 'invocation': last.get('invocation'),
                       'process': last.get('process'), 'target_calls': targets, 'blocked_guards': blocked[-3:]})
    packet['failure_routes'] = routes
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
        packet['conan_state_relations'] = extract(conan, declared=(watch or {}).get('declared_interfaces', {}))
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
    expanded = [item['file'] for item in watch.get('source_expansion', [])]
    for item in watch.get('source_expansion', []):
        priority_symbols[item['file']].extend(item['symbols'])
    files = list(dict.fromkeys(focused + primary + patch_paths(patch) + expanded
                              + memory_files(memory or {}) + list(watch.get('watch', {}))))
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
                  memory=None, history=None, mode='featurespace', required_edit_targets=None):
    required_targets = []
    completion_scope = required_edit_targets is not None
    header = ('Implement a coordinated repair of the CURRENT source.\n'
              '<requirement>\n' + instance['problem_statement'] + '\n</requirement>\n' + STATE_CONTRACT +
              '\nPreserve passing behavior. Do not modify tests or hard-code test inputs. '
              'Historical observations are patch-specific, not proof about the current candidate.\n')
    if mode == 'featurespace' and memory:
        obligations = execution_obligations(memory)
        all_required_targets = list(dict.fromkeys(
            (target['file'], target['symbol'])
            for obligation in obligations
            for target in obligation.get('required_repair_targets', [])
            if target.get('file') and target.get('symbol')))
        required_targets = (list(dict.fromkeys(
            (target['file'], target['symbol'])
            for target in required_edit_targets
            if target.get('file') and target.get('symbol')))
            if completion_scope else all_required_targets)
        target_instruction = (
            'A previous proposal already supplies edits for the other required targets. '
            'Return only the executable edits needed to complete every target in the '
            'required_edit_checklist near the output contract. '
            if completion_scope else
            'Every required_repair_target must be changed as part of that coordinated repair; ')
        header += ('\nUse the following updated FeatureSpace to identify the functional groups, dependencies '
                   'and conditions that must be repaired together. Initial-plan conditions are hypotheses; '
                   'retired entries must not constrain the repair. Check the cited evidence against current source. '
                   'Entries with origin=execution_flow are responsibilities discovered by the self-evolution cycle. '
                   'Repair every focused broken flow as a producer-to-consumer chain: represent the input state, '
                   'refresh derived state after object or configuration changes, and only then repair its consumer. '
                   'For candidate-selection flows, preserve the candidate producer and make the selector continue '
                   'past rejected alternatives before concluding that selection failed. '
                   + target_instruction + 'Other listed '
                   'responsibilities must be reviewed against the invariant and changed only when current source violates it. '
                   'A downstream function not being called is a symptom and does not by itself justify widening its guard. '
                   'Each check states a hypothesis and the expected_change this repair must realize; '
                   'diagnosis_check obligations bind those expectations to the named FeatureSpace responsibilities. '
                   'Do not repeat a change whose expected change a check review already refuted. '
                   'Preserve the supported behaviors in verified_candidates and successful_execution evidence. '
                   'When a callback is optional, distinguish absence, validation rejection, and an execution bug. '
                   'Repair the lifecycle or dispatch condition instead of discarding a responsibility that other '
                   'passing executions require. Reuse existing producer/validation routines when appropriate.\n'
                   '<execution_obligations>\n' + json.dumps(obligations, ensure_ascii=False) + '\n</execution_obligations>\n'
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
    checklist = ''
    if required_targets:
        checklist = (
            '\n<required_edit_checklist>\n'
            + json.dumps([
                {'file_path': file, 'symbol': symbol}
                for file, symbol in required_targets
            ], ensure_ascii=False)
            + '\n</required_edit_checklist>\n'
            'The checklist is an executable completion contract derived from the current '
            'FeatureSpace. Your JSON must contain a real executable change for EVERY exact '
            'file_path and symbol in it. Review the producer-to-consumer invariant before '
            'editing each function. Copying its current body is not a change and will be rejected.\n'
            + ('Only complete these missing responsibilities; the controller retains the '
               'other edits from the previous proposal.\n' if completion_scope else ''))
    tail = (checklist + '\nReturn only JSON: {"edits": [{"file_path": "relative/path.py", '
            '"symbol": "Class.method", "replacement": "def method(...):\\n    complete implementation"}]}. '
            'Replace existing functions by their exact qualified symbol and provide ONE complete function '
            'including its decorators. These interfaces already exist in the current candidate; do not append '
            'duplicate definitions. For imports or changes outside functions use '
            '{"file_path":"relative/path.py","before":"exact CURRENT snippet","after":"replacement"}. '
            'Include every required production edit and imports. Keep snippets concise. '
            'Do not add duplicate method definitions or repeat a rejected/ineffective candidate.\n')
    remaining = budget - len(header) - len(tail) - 100
    if remaining < 1000:
        raise ValueError('Feedback instructions leave no source budget')
    code, _ = source_context(instance, current, patch, packet, watch, remaining, memory if mode == 'featurespace' else None)
    return header + code + tail


def diagnosis_prompt(instance, current, patch, packet, watch, memory, bundle, budget=80000):
    historical = associated_evidence(memory, budget=9000)
    historical['observations'] = [e for e in historical['observations'] if e['id'] not in memory.get('current_evidence', [])]
    view = memory_view(memory, 18000)
    # An evidence ID displayed on a FeatureSpace entry is supplied to the
    # diagnosis even when the bounded evidence payload was omitted. Keep the
    # prompt contract and transaction validator aligned.
    visible_ids = []
    for kind in ('elements', 'groups', 'links', 'conditions'):
        for entry in view.get(kind, []):
            visible_ids.extend(entry.get('evidence', []))
            visible_ids.extend(entry.get('retirement', {}).get('evidence', []))
    for check in view.get('checks', []):
        visible_ids.extend(check.get('evidence', []))
    memory['diagnosis_evidence_ids'] = list(dict.fromkeys(
        list(memory.get('current_evidence', []))
        + [e['id'] for e in historical['observations']]
        + [e for e in visible_ids if isinstance(e, str) and e.startswith('ev:')]))
    header = ('Update a repository-specific FeatureSpace before generating a repair. Do not return code edits.\n'
              '<requirement evidence_id="requirement">\n' + instance['problem_statement'] + '\n</requirement>\n'
              + STATE_CONTRACT + '\n'
              'Begin with the execution_flow entries already incorporated into FeatureSpace and the causal_flow '
              'observations. Walk them in causal order: request input to represented state, object/configuration '
              'change to recomputed derived state, and derived state to its consuming branch. Locate the earliest '
              'broken responsibility, then retain every downstream relation needed for a coordinated repair. '
              'Use failure_routes only as downstream symptoms: a call in a different command does not prove this '
              'failing command reached it, and a skipped call does not justify widening a guard while its input '
              'state is absent or stale. Check claimed missing code against CURRENT code before repeating a '
              'previous diagnosis: already present restoration or validation is not a missing implementation. '
              'Source expansion provides previously unseen state producers and related responsibilities. '
              'Trace the observed field back to its writer and check whether that writer runs after each '
              'configuration/object change. Do not assume the initial plan contains every required file. '
              'When a producer returns multiple alternatives, compare each alternative\'s observed eligibility '
              'state with the selector\'s iteration and return point. Distinguish failure of one candidate from '
              'exhaustion of the candidate set. '
              'Compare failing paths with successful_execution and verified_candidates: identify the '
              'conditions under which each behavior is required. Test identifiers locate evidence, never '
              'become code conditions. A different output identity can result from selecting a different '
              'object/configuration; do not assume the identity calculation itself is wrong. '
              'First review the previous round checks: compare each expected_change with the current '
              'branch_outcomes, watched_call_coverage and the attempt test deltas, and record the outcome '
              'in check_reviews. If an expected change did not occur, re-examine that repair and the '
              'judgement itself (retire or correct it); if it occurred while tests still fail, locate the '
              'next break in execution order. '
              'Treat tests listed as new_failures in the latest attempt as behavior that the current candidate '
              'broke. Identify the changed responsibility on their execution path and retain a preservation '
              'condition in FeatureSpace; do not trade those behaviors for the requested feature. '
              'When execution shows a state-dependent callback under a one-time owner but not under a shared '
              'configuration refresh, revise the responsibility placement: reuse the existing callback at the '
              'shared boundary and avoid duplicating object state or lifecycle work. '
              'Explain the earliest supported break between input representation, state production, configuration '
              'changes, state recomputation and consuming branches. Add the observed responsibilities and their '
              'relations to FeatureSpace; do not reduce the diagnosis to a failing line or function list. '
              'Distinguish observed facts from inferred explanations. A default value is not proof of validation. '
              'Missing events in bounded traces are uncertainty, not proof of non-execution; only '
              'watched_call_coverage entries marked not_called_confirmed establish that a watched function '
              'was not called. Branch outcomes show which suite a guard actually took. '
              'Explain why related files must change together; revise functional groups, dependencies and conditions. '
              'Initial plan constraints can be wrong: explicitly retire or correct contradicted hypotheses. '
              'Use prior attempt/test deltas to avoid repeating ineffective diagnoses. Preserve passing behavior. '
              'Do not infer unseen expected values, use reference solutions, or introduce test-specific constants. '
              'Cite only ev: IDs from the current execution evidence, ev: IDs already stored in the FeatureSpace '
              '(they are stamped with the earlier patch and round they belong to, and never describe the current '
              'candidate), code:<file> IDs, or requirement. Code citations cover only displayed snippets. '
              'Use existing entry IDs to revise claims. New IDs start E:/G:/L:/C:. '
              'Entries with origin=execution_flow are current observations: do not overwrite or retire them. '
              'Add or revise hypotheses around those observed relationships. '
              'Every change must come with a check whose expected_change is observable next round. '
              'Keep the delta small; all inferred claims remain hypotheses until evaluated.\n'
              '<featurespace>\n' + json.dumps(view, ensure_ascii=False) + '\n</featurespace>\n'
              '<current_execution_evidence>\n' + json.dumps(bundle, ensure_ascii=False) + '\n</current_execution_evidence>\n'
              '<historical_execution_evidence>\n' + json.dumps(historical, ensure_ascii=False) + '\n</historical_execution_evidence>\n')
    tail = '\nReturn only a JSON object using this schema; omit unused entries, never invent evidence IDs:\n' + UPDATE_SCHEMA
    remaining = budget - len(header) - len(tail) - 100
    if remaining < 1000:
        raise ValueError('FeatureSpace diagnosis leaves no source budget')
    code, files = source_context(instance, current, patch, packet, watch, remaining, memory)
    return header + code + tail, files


def _duplicate_methods(source):
    """Count same-name method definitions per class body.

    In Python a redefined method silently shadows the earlier one whatever its
    body, so counting by name catches overrides, not just identical copies.
    """
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
                local[node.name] += 1
        # Two independent factory-local classes may legitimately have the same
        # name/methods. Count repetition inside a class, not across those scopes.
        for name, count in local.items():
            counts[(cls.name, name)] = max(counts[(cls.name, name)], count)
    return counts


def _edited_responsibilities(edits, current):
    """Map exact snippet edits back to their containing current function."""
    def executable_change(edit):
        before, after = edit.get('before', ''), edit.get('after', '')
        if before == after:
            return False
        try:
            old = ast.dump(ast.parse(textwrap.dedent(before)), include_attributes=False)
            new = ast.dump(ast.parse(textwrap.dedent(after)), include_attributes=False)
            return old != new
        except (SyntaxError, ValueError):
            return True
    result = {(edit.get('file_path'), edit.get('symbol'))
              for edit in edits if edit.get('symbol') and executable_change(edit)}
    for edit in edits:
        if edit.get('symbol') or not edit.get('before') or not executable_change(edit):
            continue
        file = edit.get('file_path')
        path = Path(current) / safe_path(file)
        if not path.is_file():
            continue
        source = path.read_text(errors='replace')
        before = edit['before']
        if source.count(before) != 1:
            continue
        offset = source.index(before)
        first = source.count('\n', 0, offset) + 1
        last = first + before.count('\n')
        candidates = []
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        def visit(body, prefix=''):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + '.')
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
                    if start <= first and node.end_lineno >= last:
                        candidates.append((node.end_lineno - start, prefix + node.name))
                    visit(node.body, prefix + node.name + '.')
        visit(tree.body)
        if candidates:
            result.add((file, min(candidates)[1]))
    return result


def _merge_repair_edits(existing, completion):
    """Overlay a focused completion on a previously parsed repair proposal."""
    merged = []
    positions = {}
    for edit in list(existing) + list(completion):
        if edit.get('symbol'):
            key = ('symbol', edit.get('file_path'), edit.get('symbol'))
        else:
            key = ('snippet', edit.get('file_path'), edit.get('before'))
        if key in positions:
            merged[positions[key]] = edit
        else:
            positions[key] = len(merged)
            merged.append(edit)
    return merged


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
    memory = attach_observations(memory, bundle, patch_hash, round_number, history)
    memory['attempts'] = history
    memory['source_expansion'] = watch.get('source_expansion', [])
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
                    try:
                        memory = merge_update(memory, delta, files)
                        accepted_delta = delta
                    except ValueError:
                        filtered, protected = preserve_runtime_observations(memory, delta)
                        if not protected:
                            raise
                        memory = merge_update(memory, filtered, files)
                        accepted_delta = filtered
                        save_memory(directory / 'diagnosis_warning.json', {
                            'warning': 'Ignored attempted mutations of immutable current execution observations',
                            'ignored': protected})
                    save_memory(directory / 'featurespace_delta.json', accepted_delta)
                except (ValueError, TypeError, KeyError, LLMCallError) as error:
                    # A bad reflection must not make a formerly repairable case empty.
                    save_memory(directory / 'diagnosis_error.json', {'error': type(error).__name__ + ': ' + str(error)[:1500]})
                    memory['diagnosis_status'] = 'fallback_previous_memory_with_current_observations'
            save_memory(directory / 'featurespace_after.json', memory)
            prompt = repair_prompt(instance, current, patch, packet, output, watch, previous_error,
                                   memory=memory, history=history, mode=mode)
            (directory / 'prompt.md').write_text(prompt)
            raw = model_stage('repair', prompt, 16384)
            from generator.feedback_edits import parse_current_edits
            obligations = execution_obligations(memory) if mode == 'featurespace' else []
            retained_edits = []
            for repair_attempt in range(3):
                proposed_edits = parse_current_edits(raw, current)
                edits = _merge_repair_edits(retained_edits, proposed_edits)
                if not edits or any(not production_path(safe_path(e['file_path'])) for e in edits):
                    raise ValueError('Repair needs complete production edits')
                edited_symbols = _edited_responsibilities(edits, current)
                omitted = []
                for obligation in obligations:
                    targets = obligation.get('required_repair_targets', [])
                    missing = [target for target in targets
                               if (target['file'], target['symbol']) not in edited_symbols]
                    if missing:
                        omitted.append({'stage': obligation.get('stage'), 'targets': missing})
                if not omitted:
                    break
                message = ('Repair omits or only echoes runtime-discovered coordinated responsibilities: '
                           + json.dumps(omitted, ensure_ascii=False))
                if repair_attempt == 2:
                    raise ValueError(message)
                if repair_attempt == 0:
                    retry_prompt = repair_prompt(
                        instance, current, patch, packet, output, watch,
                        '\n'.join(value for value in [previous_error, message] if value),
                        memory=memory, history=history, mode=mode)
                    raw = model_stage('repair_retry', retry_prompt, 16384)
                    continue
                retained_edits = edits
                missing_targets = list({
                    (target['file'], target['symbol']): target
                    for item in omitted for target in item['targets']
                }.values())
                completion_prompt = repair_prompt(
                    instance, current, patch, packet, output, watch,
                    '\n'.join(value for value in [
                        previous_error, message,
                        'The non-missing edits from the latest proposal are retained. '
                        'Complete only the exact missing responsibilities.'
                    ] if value),
                    memory=memory, history=history, mode=mode,
                    required_edit_targets=missing_targets)
                raw = model_stage('repair_completion', completion_prompt, 8192)
            (directory / 'response.json').write_text(json.dumps(
                {'edits': edits}, ensure_ascii=False))
            for edit in edits:
                if ('[start of %s]' % edit['file_path']) not in prompt:
                    raise ValueError('Repair requested a file outside current-source evidence')
            previous_duplicates = {name: _duplicate_methods((current / name).read_bytes())
                                   for name in names if name.endswith('.py') and (current / name).is_file()}
            incremental = edits_to_patch(edits, current)
            if not incremental.strip():
                # The model can agree with a diagnosis yet return the exact
                # current implementation. Give it the concrete failed
                # transaction in the same semantic round so an otherwise
                # useful execution/diagnosis cycle is not discarded.
                noop_message = (
                    'The proposed edits exactly match the CURRENT source and make no executable change. '
                    'The observed tests still fail, so do not repeat that implementation. Re-read the '
                    'current execution obligations and return a different executable repair that changes '
                    'the broken producer-to-consumer behavior.')
                noop_prompt = repair_prompt(
                    instance, current, patch, packet, output, watch,
                    '\n'.join(value for value in [previous_error, noop_message] if value),
                    memory=memory, history=history, mode=mode)
                raw = model_stage('repair_noop_retry', noop_prompt, 16384)
                edits = parse_current_edits(raw, current)
                if not edits or any(not production_path(safe_path(e['file_path'])) for e in edits):
                    raise ValueError('No-op repair retry needs complete production edits')
                edited_symbols = _edited_responsibilities(edits, current)
                omitted = []
                for obligation in obligations:
                    missing = [target for target in obligation.get('required_repair_targets', [])
                               if (target['file'], target['symbol']) not in edited_symbols]
                    if missing:
                        omitted.append({'stage': obligation.get('stage'), 'targets': missing})
                if omitted:
                    raise ValueError('No-op repair retry omits runtime-discovered responsibilities: '
                                     + json.dumps(omitted, ensure_ascii=False))
                for edit in edits:
                    if ('[start of %s]' % edit['file_path']) not in noop_prompt:
                        raise ValueError('No-op repair retry requested a file outside current-source evidence')
                (directory / 'response.json').write_text(json.dumps(
                    {'edits': edits}, ensure_ascii=False))
                incremental = edits_to_patch(edits, current)
                if not incremental.strip():
                    raise ValueError('Repair made no changes after explicit no-op retry')
            git('apply', '--check', '-', input=incremental)
            git('apply', '-', input=incremental)
            (directory / 'incremental.patch').write_text(incremental)
            for file in patch_paths(incremental):
                if file.endswith('.py') and (current / file).is_file():
                    source = (current / file).read_bytes()
                    compile(source, file, 'exec')
                    duplicates = _duplicate_methods(source)
                    # Only same-name repetition INSIDE one class body shadows an
                    # earlier definition; equal names across different classes are legal.
                    if any(count > 1 and count > previous_duplicates.get(file, {}).get(key, 0)
                           for key, count in duplicates.items()):
                        raise ValueError('Repair adds a duplicate method name inside a class '
                                         '(redefinition shadows the existing method)')
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
