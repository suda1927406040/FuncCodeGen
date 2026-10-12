"""An execution-updated FeatureSpace, scoped to one instance and base revision.

The initial planner's claims are hypotheses, not immutable requirements. Model
updates cite supplied execution/current-code evidence and are merged locally;
an invalid update leaves the previous memory intact. No reference patch is read.
"""
import copy
import hashlib
import json


COLLECTIONS = ('elements', 'groups', 'links', 'conditions')


def _id(prefix, text):
    return prefix + ':' + hashlib.sha256(text.encode()).hexdigest()[:12]


def seed_featurespace(instance, plan=None, recall=None):
    from generator.state_feedback import production_path, safe_path
    memory = {'schema_version': 1, 'instance_id': instance['instance_id'],
              'base_commit': instance.get('base_commit'), 'revision': 0,
              'elements': {}, 'groups': {}, 'links': {}, 'conditions': {},
              'evidence': {}, 'attempts': [], 'focus': [], 'checks': [], 'check_reviews': [],
              'reference_patch_or_execution_used': False}
    def element(file, symbol, origin):
        file = safe_path(file)
        if not production_path(file):
            return None
        key = _id('E', file + ':' + symbol)
        memory['elements'][key] = {'id': key, 'file': file, 'symbol': symbol,
                                   'text': symbol or file, 'status': 'active',
                                   'certainty': 'hypothesis', 'origin': origin, 'evidence': []}
        return key
    for group in instance.get('new_components') or []:
        for component in group.get('components', []):
            element(group['file'], component['name'], 'declared_interface')
    for candidate in (recall or {}).get('candidates', [])[:32]:
        for symbol in candidate.get('symbols', [])[:12] or ['']:
            element(candidate['file'], symbol, 'initial_recall')
    plan = plan or {}
    for context in plan.get('context', [])[:24]:
        if context.get('file'):
            for symbol in context.get('symbols', [])[:12]:
                element(context['file'], symbol, 'initial_plan_context')
    step_groups = {}
    for step in plan.get('steps', [])[:32]:
        member = element(step['file'], step.get('symbol', ''), 'initial_plan')
        if member is None:
            continue
        key = _id('G', step['id'])
        step_groups[step['id']] = key
        memory['groups'][key] = {'id': key, 'text': step.get('purpose', step['id']),
                                'members': [member], 'status': 'active',
                                'certainty': 'hypothesis', 'origin': 'initial_plan', 'evidence': []}
        memory['groups'][key]['initial_references'] = [
            {k: item[k] for k in ['source', 'file', 'commit', 'symbol', 'symbols'] if k in item}
            for item in step.get('evidence', [])[:6] if isinstance(item, dict)]
        for text in step.get('constraints', []):
            cid = _id('C', step['id'] + text)
            memory['conditions'][cid] = {'id': cid, 'text': text, 'applies_to': [member],
                                         'status': 'active', 'certainty': 'hypothesis',
                                         'origin': 'initial_plan', 'evidence': []}
    for step in plan.get('steps', []):
        for dependency in step.get('depends_on', []):
            if step['id'] in step_groups and dependency in step_groups:
                key = _id('L', dependency + '->' + step['id'])
                memory['links'][key] = {'id': key, 'from': step_groups[dependency], 'to': step_groups[step['id']],
                                       'kind': 'planned_dependency', 'text': dependency + ' precedes ' + step['id'],
                                       'status': 'active', 'certainty': 'hypothesis',
                                       'origin': 'initial_plan', 'evidence': []}
    for text in plan.get('contracts', [])[:24]:
        cid = _id('C', text)
        memory['conditions'][cid] = {'id': cid, 'text': text, 'applies_to': [], 'status': 'active',
                                     'certainty': 'hypothesis', 'origin': 'initial_plan', 'evidence': []}
    return memory


def memory_files(memory):
    focus = focused_elements(memory)
    entries = memory.get('elements', {})
    return list(dict.fromkeys(e['file'] for e in sorted(entries.values(), key=lambda e: e['id'] not in focus)
                              if e.get('status') == 'active'))


def focused_elements(memory):
    """Resolve a focused role, dependency or condition to its program elements."""
    entries = {key: entry for kind in COLLECTIONS for key, entry in memory.get(kind, {}).items()}
    pending, visited, elements = list(memory.get('focus', [])), set(), set()
    while pending:
        key = pending.pop()
        if key in visited or key not in entries:
            continue
        visited.add(key)
        entry = entries[key]
        if 'file' in entry:
            elements.add(key)
        pending.extend(entry.get('members', []) + entry.get('applies_to', []))
        pending.extend(entry[k] for k in ['from', 'to'] if k in entry)
    return elements


def attach_observations(memory, bundle, patch_hash, round_number):
    memory = copy.deepcopy(memory)
    # Current observations are separate from inferred functional relationships.
    for item in bundle['observations']:
        memory['evidence'][item['id']] = dict(item, patch_sha256=patch_hash, round=round_number)
    memory['current_evidence'] = [item['id'] for item in bundle['observations']]
    memory['current_grade'] = bundle['grade']
    memory['observation_limits'] = bundle['limits']
    memory['current_patch_sha256'] = patch_hash
    for kind in COLLECTIONS:
        for entry in memory[kind].values():
            if entry.get('patch_sha256') and entry['patch_sha256'] != patch_hash:
                entry['needs_recheck'] = True
    return memory


def _brief(value):
    if isinstance(value, str):
        return value if len(value) <= 900 else value[:880] + ' ... [omitted]'
    if isinstance(value, dict):
        return {k: _brief(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_brief(v) for v in value[:12]] + ([{'omitted_items': len(value)-12}] if len(value) > 12 else [])
    return value


def history_view(history):
    """Keep outcomes and changed assumptions without repeating entire test suites."""
    result = []
    for attempt in history[-4:]:
        item = {key: attempt[key] for key in ['round', 'patch_updated', 'outcome', 'behavior_before', 'behavior_after'] if key in attempt}
        for key in ['diagnosis', 'error']:
            if attempt.get(key):
                item[key] = attempt[key][:350]
        item['changed_files'] = [f[:140] for f in attempt.get('changed_files', [])[:4]]
        delta = attempt.get('test_delta', {})
        item['test_delta'] = {k: [test[:160] for test in tests[:2]] for k, tests in delta.items()}
        item['test_delta_counts'] = {k: len(tests) for k, tests in delta.items()}
        result.append(item)
    return result


def memory_view(memory, budget=24000):
    """Serialize whole entries with explicit omissions; never slice JSON or keys."""
    view = {key: _brief(copy.deepcopy(memory.get(key))) for key in [
        'instance_id', 'base_commit', 'revision', 'current_patch_sha256',
        'current_grade', 'observation_limits', 'focus', 'checks', 'diagnosis', 'diagnosis_status']}
    view['attempts'] = history_view(memory.get('attempts', []))
    if view.get('current_grade'):
        for key in ['failed', 'passed']:
            view['current_grade'][key] = [t[:200] for t in memory['current_grade'].get(key, [])[:8]]
    view['checks'] = [{'hypothesis': c['hypothesis'][:500], 'expected_change': c['expected_change'][:500],
                       'evidence': c['evidence'][:4], 'basis': c.get('basis', '')[:500],
                       'made_for_patch_sha256': c.get('patch_sha256')}
                      for c in memory.get('checks', [])[:6]
                      if isinstance(c, dict) and c.get('hypothesis') and c.get('expected_change')]
    view['check_reviews'] = [{'check': r['check'][:400], 'observed': r['observed'],
                              'conclusion': r.get('conclusion', '')[:400], 'evidence': r['evidence'][:4]}
                             for r in memory.get('check_reviews', [])[-6:] if isinstance(r, dict)]
    view.update({key: [] for key in COLLECTIONS})
    view['omitted_entries'] = {key: 0 for key in COLLECTIONS}
    focus = set(memory.get('focus', []))
    # Include refuted hypotheses too, so the writer does not revive them.
    focus |= focused_elements(memory)
    queues = {kind: sorted(memory.get(kind, {}).values(), key=lambda entry: (
        entry['id'] not in focus, entry.get('status') != 'retired', -entry.get('updated_revision', 0)))
        for kind in COLLECTIONS}
    # Round robin prevents a large initial recall from displacing every condition.
    entries = []
    while any(queues.values()):
        for kind in ['conditions', 'groups', 'links', 'elements']:
            if queues[kind]:
                entries.append((kind, queues[kind].pop(0)))
    for kind, entry in entries:
        view[kind].append(entry)
        if len(json.dumps(view, ensure_ascii=False)) > budget:
            view[kind].pop()
            view['omitted_entries'][kind] += 1
    return view


def associated_evidence(memory, budget=14000):
    """Evidence belongs to the memory, including a bounded fallback if diagnosis fails."""
    referenced = []
    for kind in COLLECTIONS:
        for item in memory.get(kind, {}).values():
            referenced.extend(item.get('evidence', []))
            referenced.extend(item.get('retirement', {}).get('evidence', []))
    for check in memory.get('checks', []):
        referenced.extend(check.get('evidence', []))
    # Current evidence first: historical observations are explicitly patch-scoped.
    current = memory.get('current_evidence', [])
    order = list(dict.fromkeys([e for e in current if e in referenced] + current + referenced))
    result = {'observations': [], 'omitted': 0}
    for key in order:
        if key.startswith('code:') or key not in memory['evidence']:
            continue
        result['observations'].append(memory['evidence'][key])
        if len(json.dumps(result, ensure_ascii=False)) > budget:
            result['observations'].pop()
            result['omitted'] += 1
    return result


UPDATE_SCHEMA = '''{
  "summary": "Earliest supported failure and why the current functional understanding needs revision",
  "check_reviews": [{"check": "expected_change reviewed from the previous round", "observed": "expected_change_observed|expected_change_not_observed|not_observable", "evidence": ["ev:..."], "conclusion": "whether the repair or the judgement was wrong, and what follows"}],
  "upsert": {
    "elements": [{"id":"E:new", "file":"src/module.py", "symbol":"Class.method", "text":"responsibility", "evidence":["code:src/module.py"]}],
    "groups": [{"id":"G:new", "text":"shared functional role", "members":["E:new"], "evidence":["ev:..."]}],
    "links": [{"id":"L:new", "from":"E:new", "to":"G:new", "kind":"state/control/call", "text":"producer/configuration/consumer dependency", "evidence":["ev:...","code:src/module.py"]}],
    "conditions": [{"id":"C:new", "text":"condition justified by request, code or test evidence", "applies_to":["E:new"], "evidence":["ev:..."]}]
  },
  "retire": [{"id":"C:old", "reason":"evidence contradicts this initial hypothesis", "evidence":["ev:..."]}],
  "focus": ["E:new"],
  "checks": [{"hypothesis":"which functional relation is wrong or missing", "expected_change":"which call, branch or test outcome must change after this repair", "evidence":["ev:...","code:src/module.py"], "basis":"source condition and execution record supporting the judgement"}]
}'''


def merge_update(memory, delta, code_files):
    """Validate a complete delta transaction before changing any previous entry."""
    from generator.state_feedback import production_path, safe_path
    if not isinstance(delta, dict):
        raise ValueError('FeatureSpace update must be an object')
    if not isinstance(delta.get('summary'), str) or not delta['summary'].strip() or len(delta['summary']) > 3000:
        raise ValueError('FeatureSpace update needs a bounded diagnosis')
    result = copy.deepcopy(memory)
    revision = memory['revision'] + 1
    current_ids = set(memory.get('current_evidence', []))
    # Historical observations stay citable: each is stamped with the patch and
    # round it belongs to. They describe earlier candidates, never the current one.
    stored_ids = set(memory.get('evidence', {}))
    valid_evidence = current_ids | stored_ids | {'code:' + f for f in code_files} | {'requirement'}
    def historical(ids):
        return sorted({e for e in ids if e in stored_ids and e not in current_ids})
    def evidence(value):
        if not isinstance(value, list) or not value or len(value) > 12 or any(
                not isinstance(e, str) or e not in valid_evidence for e in value):
            raise ValueError('FeatureSpace update cites unavailable evidence')
        return value
    def text(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 1800:
            raise ValueError('FeatureSpace entry needs bounded explanatory text')
        return value
    def optional(value):
        if value is None:
            return ''
        if not isinstance(value, str) or len(value) > 1800:
            raise ValueError('FeatureSpace entry needs bounded explanatory text')
        return value
    upsert = delta.get('upsert', {})
    if not isinstance(upsert, dict) or set(upsert) - set(COLLECTIONS):
        raise ValueError('Unknown FeatureSpace collection')
    all_ids = {key: kind for kind in COLLECTIONS for key in memory[kind]}
    new_ids = set()
    for kind, entries in upsert.items():
        if not isinstance(entries, list) or len(entries) > 24:
            raise ValueError('Too many FeatureSpace updates')
        for entry in entries:
            key = entry.get('id') if isinstance(entry, dict) else None
            if not isinstance(key, str) or not key.startswith(kind[0].upper() + ':') or len(key) > 100:
                raise ValueError('Invalid FeatureSpace entry ID')
            if key in new_ids or (key in all_ids and all_ids[key] != kind):
                raise ValueError('Conflicting FeatureSpace IDs')
            new_ids.add(key)
            all_ids[key] = kind
    def references(value, kinds):
        if not isinstance(value, list) or len(value) > 32 or any(
                not isinstance(v, str) or all_ids.get(v) not in kinds for v in value):
            raise ValueError('FeatureSpace contains dangling or mistyped references')
        return value
    for kind, entries in upsert.items():
        for entry in entries:
            key = entry['id']
            clean = {'id': key, 'text': text(entry.get('text')), 'evidence': evidence(entry.get('evidence')),
                     'status': 'active', 'certainty': 'hypothesis', 'origin': 'execution_reflection',
                     'updated_revision': revision, 'patch_sha256': memory['current_patch_sha256']}
            stale = historical(clean['evidence'])
            if stale:
                clean['cites_historical_evidence'] = stale
            if kind == 'elements':
                file = safe_path(entry.get('file', ''))
                if file not in code_files or not production_path(file):
                    raise ValueError('FeatureSpace element is outside current source evidence')
                clean.update(file=file, symbol=text(entry.get('symbol') or file))
            elif kind == 'groups':
                clean['members'] = references(entry.get('members'), {'elements'})
                if not clean['members']:
                    raise ValueError('Functional group needs program elements')
            elif kind == 'links':
                references([entry.get('from'), entry.get('to')], {'elements', 'groups'})
                clean.update({'from': entry['from'], 'to': entry['to'], 'kind': text(entry.get('kind'))})
            else:
                clean['applies_to'] = references(entry.get('applies_to', []), {'elements', 'groups', 'links'})
            old = result[kind].get(key)
            if old:
                clean['previous'] = {k: old[k] for k in ['text', 'status', 'evidence']}
            result[kind][key] = clean
    retirements = delta.get('retire', [])
    if not isinstance(retirements, list) or len(retirements) > 32:
        raise ValueError('Invalid FeatureSpace retirements')
    for item in retirements:
        key = item.get('id') if isinstance(item, dict) else None
        if key not in all_ids or key in new_ids:
            raise ValueError('Cannot retire unknown or simultaneously updated entry')
        entry = result[all_ids[key]][key]
        entry.update(status='retired', updated_revision=revision,
                     retirement={'reason': text(item.get('reason')), 'evidence': evidence(item.get('evidence'))})
        stale = historical(entry['retirement']['evidence'])
        if stale:
            entry['retirement']['cites_historical_evidence'] = stale
    result['focus'] = references(delta.get('focus', []), {'elements', 'groups', 'links', 'conditions'})
    reviews = delta.get('check_reviews', [])
    if not isinstance(reviews, list) or len(reviews) > 12:
        raise ValueError('Invalid check reviews')
    cleaned_reviews = []
    for review in reviews:
        if not isinstance(review, dict) or review.get('observed') not in {
                'expected_change_observed', 'expected_change_not_observed', 'not_observable'}:
            raise ValueError('Check review needs a supported observed outcome')
        clean = {'check': text(review.get('check')), 'observed': review['observed'],
                 'evidence': evidence(review.get('evidence')), 'conclusion': optional(review.get('conclusion'))}
        stale = historical(clean['evidence'])
        if stale:
            clean['cites_historical_evidence'] = stale
        cleaned_reviews.append(clean)
    result['check_reviews'] = cleaned_reviews
    checks = delta.get('checks', [])
    if not isinstance(checks, list) or len(checks) > 12:
        raise ValueError('Invalid verification checks')
    changed = any(upsert.get(kind) for kind in COLLECTIONS) or bool(retirements)
    # Every accepted revision carries its own verifiable expectation; a change
    # without an expected execution difference cannot be reviewed next round.
    if changed and not checks:
        raise ValueError('FeatureSpace changes need a check with expected_change')
    cleaned_checks = []
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError('Invalid verification check')
        clean = {'hypothesis': text(check.get('hypothesis')), 'expected_change': text(check.get('expected_change')),
                 'evidence': evidence(check.get('evidence')), 'basis': optional(check.get('basis')),
                 'patch_sha256': memory['current_patch_sha256']}
        stale = historical(clean['evidence'])
        if stale:
            clean['cites_historical_evidence'] = stale
        cleaned_checks.append(clean)
    result['checks'] = cleaned_checks
    result.update(revision=revision, diagnosis=delta['summary'], diagnosis_status='updated')
    if sum(len(result[k]) for k in COLLECTIONS) > 800:
        raise ValueError('FeatureSpace memory limit reached')
    return result


def save_memory(path, memory):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(memory, ensure_ascii=False, indent=2))
    temp.replace(path)
