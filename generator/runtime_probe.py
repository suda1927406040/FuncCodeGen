"""Bounded observer loaded via sitecustomize; works with pytest and unittest.

No user-defined repr/equality/properties are evaluated by the generic snapshot.
One observer file per process; identities are only meaningful within that file.
"""
import atexit
import json
import linecache
import os
import sys
import threading

_config = {}
_stream = None
_seq = 0
_test = 'collection'
_counts = {}
_ids = {}
_held = []
_last = {}
_lock = threading.RLock()
_conan = None
_errors = 0
_command = None
_invocation = 0


def identity(value):
    key = id(value)
    if key not in _ids:
        _ids[key] = 'object_%d' % (len(_ids) + 1)
        _held.append(value)
    return _ids[key]


def value_state(value, depth=0):
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        return value[:200]
    if type(value) in (list, tuple):
        return {'type': type(value).__name__, 'length': len(value),
                'items': [value_state(v, depth+1) for v in value[:6]] if depth < 2 else []}
    if type(value) is dict:
        pairs = list(value.items())[:10]
        return {'type': 'dict', 'length': len(value),
                'items': {k[:80]: value_state(v, depth+1) for k, v in pairs if type(k) is str} if depth < 2 else {}}
    result = {'receiver': identity(value), 'type': type(value).__name__}
    if depth < 2:
        try:
            attrs = object.__getattribute__(value, '__dict__')
        except (AttributeError, TypeError):
            attrs = {}
        if type(attrs) is dict:
            names = sorted(attrs, key=lambda n: (n not in {'binary', 'invalid', 'cant_build', 'settings', 'info', 'options', '_package_id', 'state', 'status', 'value'}, n))[:10]
            result['fields'] = {n: value_state(attrs[n], depth+1) for n in names
                                if type(n) is str and not n.startswith('__')}
    return result


def scoped_test(frame):
    if _test not in {'collection', 'unscoped'}:
        return _test
    current = frame
    for _ in range(20):
        if current is None:
            break
        name = current.f_code.co_name
        if name.startswith('test_'):
            path = current.f_code.co_filename.replace('/testbed/', '')
            owner = current.f_locals.get('self')
            return path + '::' + (type(owner).__name__ + '::' if owner is not None else '') + name
        current = current.f_back
    return _test


def selected_test(test):
    failures = _config.get('failed_tests', [])
    return not failures or test in failures or any(
        test.endswith('::'+name) or name.endswith('::'+test) for name in failures)


def trace(frame, event, arg):
    global _seq, _errors, _command, _invocation
    if _stream is None or _seq >= _config.get('event_limit_per_process', 6000):
        return None
    filename, name = frame.f_code.co_filename, frame.f_code.co_name
    rel = filename[9:] if filename.startswith('/testbed/') else ''
    if _conan is not None and filename.endswith('/conan/test/utils/tools.py') and name == '_run_cli' and event == 'call':
        _command = frame.f_locals.get('command_line')
        _invocation += 1
    watched = name in _config.get('watch', {}).get(rel, [])
    recipe_hook = _conan is not None and filename.endswith('/conanfile.py') and name in {'validate', 'validate_build'}
    if not watched and not recipe_hook:
        return None
    if event not in {'call', 'line', 'return', 'exception'}:
        return trace
    try:
        with _lock:
            test = scoped_test(frame)
            if not selected_test(test):
                return trace
            if _counts.get(test, 0) >= _config.get('event_limit_per_test', 800):
                return trace
            statement = linecache.getline(filename, frame.f_lineno).strip()[:250]
            names = sorted(frame.f_locals, key=lambda n: (n not in {'node', 'self', 'conanfile', 'info', 'compatible', 'result', 'settings', 'options'}, n))[:8]
            values = {n: value_state(frame.f_locals[n]) for n in names if not n.startswith('__')}
            state = _conan.snapshot(frame, event, arg) if _conan else None
            if state:
                # The adapter already selects receiver/configuration fields. Avoid
                # emitting unrelated logger/cache changes on every source line.
                values = {'observed_state': state}
            marker = json.dumps([values, state], sort_keys=True)
            key = (threading.current_thread().ident, id(frame))
            guard = statement.startswith(('if ', 'elif ', 'while ', 'for '))
            if event == 'line' and marker == _last.get(key) and not guard:
                return trace
            _last[key] = marker
            _seq += 1
            _counts[test] = _counts.get(test, 0) + 1
            owner = frame.f_locals.get('self')
            item = {'seq': _seq, 'process': os.getpid(), 'test': test, 'event': event,
                    'frame': str(key), 'file': rel or 'recipe/conanfile.py',
                    'function': (type(owner).__name__ + '.' if owner is not None else '') + name,
                    'line': frame.f_lineno, 'statement': statement, 'values': values}
            if _conan is not None:
                item.update(command=_command, invocation=_invocation)
            if state is not None:
                item['state'] = state
            if event == 'exception':
                item['exception'] = {'type': arg[0].__name__, 'message': str(arg[1])[:300]}
            if event == 'return':
                item['return_value'] = value_state(arg)
                _last.pop(key, None)
            _stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception:
        _errors += 1
    return trace


def finish():
    global _stream
    if _stream is not None:
        sys.settrace(None)
        threading.settrace(None)
        _stream.close()
        _stream = None
        path = os.path.join(os.environ['FEA_STATE_OUTPUT'], 'summary-%d.json' % os.getpid())
        with open(path, 'w') as stream:
            json.dump({'events': _seq, 'observer_errors': _errors, 'tests': _counts,
                       'truncated': _seq >= _config.get('event_limit_per_process', 6000),
                       'process': os.getpid(), 'line_timing': 'before_execution'}, stream)


def install():
    global _config, _stream, _conan, _test
    if _stream is not None or not os.environ.get('FEA_STATE_CONFIG'):
        return
    with open(os.environ['FEA_STATE_CONFIG']) as stream:
        _config = json.load(stream)
    _test = os.environ.get('FEA_STATE_TEST', 'collection')
    if _config.get('repo') == 'conan-io/conan':
        import conan_probe
        _conan = conan_probe
    dest = os.environ['FEA_STATE_OUTPUT']
    os.makedirs(dest, exist_ok=True)
    _stream = open(os.path.join(dest, 'trace-%d.jsonl' % os.getpid()), 'a', buffering=1)
    atexit.register(finish)
    sys.settrace(trace)
    threading.settrace(trace)


def pytest_runtest_setup(item):
    global _test
    _test = item.nodeid
    os.environ['FEA_STATE_TEST'] = _test


def pytest_runtest_teardown(item):
    global _test
    _test = 'unscoped'
    os.environ.pop('FEA_STATE_TEST', None)
