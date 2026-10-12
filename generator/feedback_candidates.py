"""Retain evaluated, patch-bound witnesses without generating extra candidates."""
import hashlib
import json
from pathlib import Path


def quality(record):
    return (-len(record['regression_failures']), len(record['new_passes']),
            len(record['regression_passes']), -record['round'])


def record_candidate(root, iid, patch, grade, memory, number):
    from generator.feature_memory import save_memory
    directory = Path(root) / 'candidates' / iid
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.is_file() else []
    statuses = grade.get('tests_status', {})
    if not statuses or not grade.get('patch_successfully_applied'):
        return index
    sha = hashlib.sha256(patch.encode()).hexdigest()
    if not any(r['patch_sha256'] == sha for r in index):
        (directory / (sha + '.patch')).write_text(patch)
        save_memory(directory / (sha + '.memory.json'), memory)
        save_memory(directory / (sha + '.report.json'), grade)
        index.append({'patch_sha256': sha, 'round': number,
                      'new_passes': statuses.get('FAIL_TO_PASS', {}).get('success', []),
                      'new_failures': statuses.get('FAIL_TO_PASS', {}).get('failure', []),
                      'regression_passes': statuses.get('PASS_TO_PASS', {}).get('success', []),
                      'regression_failures': statuses.get('PASS_TO_PASS', {}).get('failure', []),
                      'resolved': bool(grade.get('resolved'))})
        save_memory(index_path, index)
    return index


def candidate_view(root, iid, records, current_patch):
    if not records:
        return []
    first = max(records, key=quality)
    selected = [first]
    others = [r for r in records if r is not first]
    if others:
        # Preserve complementary supported behavior, not just two scalar scores.
        selected.append(max(others, key=lambda r: (
            len(set(r['new_passes']) - set(first['new_passes'])), quality(r))))
    result = []
    current_sha = hashlib.sha256(current_patch.encode()).hexdigest()
    for record in selected:
        item = dict(record, is_current=record['patch_sha256'] == current_sha)
        if not item['is_current']:
            patch = (Path(root) / 'candidates' / iid / (record['patch_sha256'] + '.patch')).read_text()
            item['earlier_patch_reference'] = patch[:5000]
            item['reference_truncated'] = len(patch) > 5000
            item['reference_scope'] = 'Earlier candidate, not current source. Do not copy edits blindly.'
        result.append(item)
    return result


def select_candidate(root, iid):
    directory = Path(root) / 'candidates' / iid
    if not (directory / 'index.json').is_file():
        return None
    records = json.loads((directory / 'index.json').read_text())
    if not records:
        return None
    selected = max(records, key=quality)
    sha = selected['patch_sha256']
    return selected, (directory / (sha + '.patch')).read_text(), json.loads((directory / (sha + '.report.json')).read_text())
