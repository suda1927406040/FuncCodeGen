"""Pytest diagnostic plugin: observe bounded Conan state and receiver bindings."""
import json
import linecache
import os
from pathlib import Path
import sys
import threading



WATCH = {
    'compute_pid.py': {'compute_package_id', 'run_validate_package_id'},
    'compatibility.py': {'compatibles', '_compatible_infos'},
    'graph_binaries.py': {'_evaluate_node', '_process_node', '_evaluate_build', '_compatible_found',
                          '_find_build_compatible_binary', '_find_existing_compatible_binaries'},
    'build_mode.py': {'__init__', 'allowed', 'allowed_compatible', 'forced', 'should_build_missing'},
    'info.py': {'clone'},
}
OUTPUT = Path(os.environ.get('STATE_TRACE_PATH', '/tmp/state_trace.jsonl'))
SUMMARY = Path(os.environ.get('STATE_SUMMARY_PATH', '/tmp/state_summary.json'))
_test = ''
_ids = {}
_objects = []
_last = {}
_stream = None
_sequence = 0
_reports = []
_command = None
_invocation = 0


def scalar(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value[:180] if isinstance(value, str) else value
    return type(value).__name__


def tag(value):
    if value is None:
        return None
    key = id(value)
    if key not in _ids:
        _ids[key] = 'object_%d' % (len(_ids) + 1)
        _objects.append(value)  # Prevent address reuse within this test.
    return _ids[key]


def settings_value(settings, key):
    try:
        return scalar(settings.get_safe(key)) if settings is not None else None
    except Exception:
        return None


def info_state(info):
    if info is None:
        return None
    return {'receiver': tag(info), 'cppstd': settings_value(getattr(info, 'settings', None), 'compiler.cppstd'),
            'invalid': scalar(getattr(info, 'invalid', None)), 'cant_build': scalar(getattr(info, 'cant_build', None))}


def snapshot(frame, event, arg):
    local = frame.f_locals
    node = local.get('node')
    conanfile = local.get('conanfile', local.get('conan_file'))
    if conanfile is None and node is not None:
        conanfile = getattr(node, 'conanfile', None)
    if conanfile is None and frame.f_code.co_name in {'validate', 'validate_build'}:
        conanfile = local.get('self')
    result = {}
    if conanfile is not None:
        result['recipe'] = tag(conanfile)
        result['has_build_validator'] = callable(getattr(conanfile, 'validate_build', None))
        result['recipe_cppstd'] = settings_value(getattr(conanfile, 'settings', None), 'compiler.cppstd')
        result['current_info'] = info_state(getattr(conanfile, 'info', None))
        result['original_info'] = info_state(getattr(conanfile, 'original_info', None))
    for name in ['c', 'compatible', 'compatible_package', 'compatible_pkg', 'compat_info', 'result']:
        value = local.get(name)
        if value is not None and hasattr(value, 'cant_build'):
            result[name] = info_state(value)
    if node is not None:
        result['node'] = {'receiver': tag(node), 'binary': scalar(getattr(node, 'binary', None)),
                          'package_id': scalar(getattr(node, '_package_id', None)),
                          'should_build': scalar(getattr(node, 'should_build', None)),
                          'build_allowed': scalar(getattr(node, 'build_allowed', None))}
    owner = local.get('self')
    mode = local.get('build_mode')
    if owner is not None and type(owner).__name__ == 'BuildMode':
        mode = owner
    if mode is not None:
        result['build_mode'] = {name: scalar(getattr(mode, name, None)) for name in ['missing', 'never', 'cascade']}
        for name in ['patterns', 'build_missing_patterns', '_build_compatible_patterns', '_build_compatible_excluded']:
            result['build_mode'][name] = getattr(mode, name, None)
    if event == 'return':
        if frame.f_code.co_name == 'compatibles' and isinstance(arg, dict):
            result['returned_candidates'] = [{'package_id': k, 'info': info_state(v)} for k, v in arg.items()]
        elif arg is None or isinstance(arg, (bool, str, int)):
            result['return_value'] = scalar(arg)
    if event == 'exception':
        result['exception'] = {'type': arg[0].__name__, 'message': scalar(str(arg[1]))}
    return result

