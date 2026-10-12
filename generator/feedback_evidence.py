"""Select execution evidence without mistaking harness source dumps for failures.

Selection is balanced across observation categories. The full logs and packets
remain on disk; omitted observations are counted rather than silently discarded.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re


def test_output_body(output):
    output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', output)
    # SWE harness scripts can print an entire git snapshot before running tests.
    starts = list(re.finditer(r'^=+\s*test session starts\s*=+.*$', output, re.M))
    if not starts:
        starts = list(re.finditer(r'^=+\s*(?:FAILURES|ERRORS)\s*=+.*$', output, re.M))
    return output[starts[0].start():] if starts else output


def _clip(text, limit):
    if len(text) <= limit:
        return text
    marker = '\n... omitted ...\n'
    left = (limit - len(marker)) // 2
    return text[:left] + marker + text[-(limit - len(marker) - left):]


def failure_excerpt(output, budget=16000):
    body = test_output_body(output)
    # Pytest separates individual failures this way. Allocate a share to each
    # failure instead of spending the whole budget on the first traceback.
    starts = [m.start() for m in re.finditer(r'^(?:_{3,} .+ _{3,}|FAIL: .+|ERROR: .+)$', body, re.M)]
    if starts:
        blocks = [body[a:b] for a, b in zip(starts, starts[1:] + [len(body)])]
        blocks = blocks[:12]
        tail = body[-min(2000, budget // 5):]
        share = max(120, (budget - len(tail) - 100) // len(blocks))
        return _clip('\n'.join(_clip(b, share) for b in blocks) + '\n' + tail, budget)
    lines = body.splitlines(keepends=True)
    regions = set(range(max(0, len(lines) - 45), len(lines)))
    for i, line in enumerate(lines):
        # Ignore diff additions/removals and shell echo text, not real pytest E lines.
        if not line.startswith(('+', '-', 'diff --git', '+++', '---')) and re.search(
                r'^(?:FAILED |ERROR |FAIL:|ERROR:|E\s|Traceback|AssertionError)|\b\w+Error:', line):
            regions.update(range(max(0, i - 6), min(len(lines), i + 10)))
    return _clip(''.join(lines[i] for i in sorted(regions)), budget)


def read_failure_output(evaluated, diagnosed=None):
    """Prefer formal test output; recover from old head-truncated archives."""
    candidates = [(Path(evaluated) / 'test_output.txt', 'evaluation')]
    if diagnosed:
        candidates.append((Path(diagnosed) / 'diagnostic_output.txt', 'diagnosis'))
    candidates.append((Path(evaluated) / 'run_instance.log', 'harness'))
    found = []
    for path, source in candidates:
        if path.is_file():
            value = path.read_text(errors='replace')
            body = test_output_body(value)
            score = 3 if body != value else 0
            score += bool(re.search(r'^(?:FAILED |FAIL:|E\s+.*(?:Error|assert)|Traceback)', body, re.M))
            found.append((score, source, value))
    if not found:
        return 'No usable test output; consult the report and trace limitations.', 'unavailable'
    _, source, value = max(found, key=lambda item: item[0])
    return value, source


def grade_summary(report):
    status = (report or {}).get('tests_status', {})
    return {'resolved': bool((report or {}).get('resolved')),
            'has_test_results': bool(status),
            'failed': sorted({t for group in status.values() for t in group.get('failure', [])}),
            'passed': sorted({t for group in status.values() for t in group.get('success', [])})}


def test_delta(before, after):
    before, after = before or {}, after or {}
    return {'fixed': sorted(set(before.get('failed', [])) & set(after.get('passed', []))),
            'new_failures': sorted(set(before.get('passed', [])) & set(after.get('failed', []))),
            'still_failing': sorted(set(before.get('failed', [])) & set(after.get('failed', []))),
            'not_observed': sorted((set(before.get('failed', [])) | set(before.get('passed', [])))
                                   - set(after.get('failed', [])) - set(after.get('passed', [])))}


def _bounded(value, depth=0):
    """Bound individual values, retaining object/configuration keys and timing."""
    if isinstance(value, str):
        return _clip(value, 900)
    if depth > 6:
        return _clip(json.dumps(value, ensure_ascii=False), 500)
    if isinstance(value, dict):
        return {k: _bounded(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        result = [_bounded(v, depth + 1) for v in value[:8]]
        if len(value) > 8:
            result.append({'omitted_items': len(value) - 8})
        return result
    return value


def _diverse(items):
    """One observation per test before spending space on repeated observations."""
    groups = OrderedDict()
    for item in items:
        groups.setdefault(item.get('test', 'unscoped') if isinstance(item, dict) else 'unscoped', []).append(item)
    while groups:
        for key in list(groups):
            yield groups[key].pop(0)
            if not groups[key]:
                del groups[key]


def _trace_summary(metadata):
    """Keep execution integrity facts without spending evidence budget on watch dumps."""
    metadata = metadata or {}
    result = {key: metadata.get(key) for key in [
        'instance_id', 'network_mode', 'image_id', 'base_commit', 'fresh_initial_container',
        'prediction_sha256', 'baseline_verified', 'timed_out', 'seconds', 'trace_events',
        'extraction_bytes', 'trace_status', 'extraction_truncated'] if key in metadata}
    summaries = metadata.get('observer_summaries') or []
    if summaries:
        result['observer_summary'] = {
            'processes': len(summaries),
            'finished_processes': sum(bool(item.get('finished')) for item in summaries),
            'observer_errors': sum(item.get('observer_errors', 0) for item in summaries),
            'truncated_processes': sum(bool(item.get('truncated')) for item in summaries),
            'truncated_tests': sorted({test for item in summaries
                                       for test in item.get('truncated_tests', [])})[:20]}
    return result


def evidence_bundle(packet, output, budget=26000):
    """Produce stable, citeable observations; keep every nonempty category visible."""
    categories = OrderedDict()
    # Execution-discovered responsibility breaks come first. Split stages so
    # one frequent flow cannot displace a different upstream relation.
    for key, value in packet.items():
        if key.endswith('_state_relations') and isinstance(value, dict):
            for name, records in value.items():
                if isinstance(records, list) and name not in {'limits', 'identity_scope'}:
                    if name == 'causal_flow':
                        stages = OrderedDict()
                        for record in records:
                            stages.setdefault(record.get('stage', 'unknown'), []).append(record)
                        for stage, items in stages.items():
                            categories[key + '.causal_flow.' + stage] = items
                    else:
                        categories[key + '.' + name] = records
    for key in ['successful_execution', 'failure_routes', 'invocation_calls', 'branch_outcomes']:
        categories[key] = packet.get(key, [])
    # Distilled execution facts (branch outcomes, watch coverage) come before
    # raw snapshots so they survive tight budgets.
    if packet.get('trace_coverage'):
        categories['trace_coverage'] = [packet['trace_coverage']]
    if packet.get('watched_call_coverage'):
        categories['watched_call_coverage'] = list(packet['watched_call_coverage'])
    for key in ['branch_outcomes', 'exceptions', 'state_transitions', 'consumer_guards']:
        categories[key] = list(reversed(packet.get(key, [])))
    for test, records in packet.get('failure_trace_tails', {}).items():
        categories['trace_tail:' + test] = records[-3:]
    result = {'grade': grade_summary(packet.get('grade')),
              'output_source': packet.get('output_source', 'provided_test_output'),
              'trace_metadata': _trace_summary(packet.get('trace_metadata', {})),
              'line_timing': 'before_execution',
              'limits': list(packet.get('limits', [])) + [
                  'Absence from a bounded trace is a coverage gap, not proof a producer never ran.',
                  'Adapter relations are diagnostic hypotheses; verify against current code and the request.'],
              'categories': {k: {'total': len(v), 'selected': 0} for k, v in categories.items()},
              'observations': []}
    for key, cap in [('failed', 24), ('passed', 12)]:
        values = result['grade'][key]
        result['grade'][key + '_total'] = len(values)
        result['grade'][key] = [_clip(v, 300) for v in values[:cap]]
    excerpt = failure_excerpt(output, min(10000, budget // 3))
    records = [('test_output', {'text': excerpt})] if excerpt else []
    queues = {key: list(_diverse(values)) for key, values in categories.items()}
    while any(queues.values()):
        for key, queue in queues.items():
            if queue:
                records.append((key, queue.pop(0)))
    for category, value in records:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
        candidate = packet.get('trace_metadata', {}).get('prediction_sha256', '')
        identity = 'ev:' + hashlib.sha256((candidate + category + raw).encode()).hexdigest()[:16]
        item = {'id': identity, 'category': category, 'observation': _bounded(value)}
        # test_output already has a deliberate per-failure budget.
        if category == 'test_output':
            item['observation'] = value
        elif category == 'failure_routes' or '.causal_flow.' in category:
            # Compact, command-specific facts must survive without losing the
            # broken relation in the middle of an arbitrary string excerpt.
            item['observation'] = value
        else:
            cap = min(2200, max(600, (budget - len(excerpt) - 4000) // max(1, sum(bool(v) for v in categories.values()))))
            if len(json.dumps(item, ensure_ascii=False)) > cap:
                item['observation'] = {'excerpt': _clip(raw, cap - 200), 'observation_truncated': True}
        result['observations'].append(item)
        if len(json.dumps(result, ensure_ascii=False)) > budget:
            result['observations'].pop()
            continue
        if category in result['categories']:
            result['categories'][category]['selected'] += 1
    return result
