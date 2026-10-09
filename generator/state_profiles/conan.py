"""Extract receiver- and configuration-specific feedback from runtime events.

This module reads failing-run events only. Reference executions are not inputs.
"""
from collections import defaultdict
import json


def extract(events, field='cant_build', validator='validate_build', context='cppstd'):
    validation_calls = defaultdict(list)
    test_has_validator = set()
    candidates = {}
    unvalidated, mismatches, mode_errors, producer_errors, dispatch_errors, stale_guards = [], [], [], [], [], []
    seen_unvalidated, seen_mismatch, seen_mode = set(), set(), set()
    last_invalid_guard = {}
    for event in events:
        test, seq, state = event['test'], event['seq'], event['state']
        info = state.get('current_info') or {}
        command = event.get('command') or ''
        node = state.get('node') or {}
        scope = (test, event.get('invocation'), node.get('receiver'))
        if (event['event'] == 'call' and event['function'].endswith('._find_build_compatible_binary')
                and '--build=missing' in command and '--build=compatible' not in command):
            dispatch_errors.append({'test': test, 'invocation': event.get('invocation'), 'command': command,
                                    'observed_consumer': event['function'], 'node_binary': node.get('binary'),
                                    'expected_role': 'ordinary missing-binary cache/remote search',
                                    'evidence': {'seq': seq, 'file': event['file'], 'line': event['line']}})
        if event['function'].endswith('._evaluate_node'):
            statement = event.get('statement', '')
            if event['event'] == 'line' and 'if node.binary == BINARY_INVALID' in statement:
                last_invalid_guard[scope] = {'seq': seq, 'line': event['line'], 'binary': node.get('binary'), 'statement': statement}
            if event['event'] == 'return' and '--build=compatible' in command and node.get('binary') == 'Invalid':
                guard = last_invalid_guard.get(scope)
                if guard and guard['binary'] != 'Invalid':
                    stale_guards.append({'test': test, 'invocation': event.get('invocation'), 'command': command,
                                         'receiver': node.get('receiver'), 'field': 'node.binary',
                                         'guard_observation': guard, 'later_value': node['binary'],
                                         'status': 'guard_becomes_true_after_its_last_evaluation',
                                         'evidence': {'seq': seq, 'file': event['file'], 'line': event['line']}})
        error = state.get('exception') or {}
        if error.get('type') in {'NameError', 'AttributeError'}:
            producer_errors.append({'test': test, 'invocation': event.get('invocation'), 'command': event.get('command'),
                                    'receiver': info.get('receiver'), 'configuration': {context: info.get(context)},
                                    'error': error, 'evidence': {'seq': seq, 'file': event['file'],
                                                               'function': event['function'], 'line': event['line']}})
        if event['event'] == 'call' and event['function'].split('.')[-1] == validator:
            key = (test, info.get('receiver'), info.get(context))
            validation_calls[key].append(seq)
            test_has_validator.add(test)
        for candidate in state.get('returned_candidates', []):
            value = candidate['info']
            candidates[(test, candidate['package_id'])] = value
            key = (test, value.get('receiver'), value.get(context))
            validated = any(s < seq for s in validation_calls[key])
            has_validator = state.get('has_build_validator', test in test_has_validator)
            if has_validator and not validated and key not in seen_unvalidated:
                seen_unvalidated.add(key)
                unvalidated.append({'test': test, 'receiver': value.get('receiver'),
                                    'configuration': {context: value.get(context)}, 'field': field,
                                    'observed_value': value.get(field),
                                    'invocation': event.get('invocation'), 'command': event.get('command'),
                                    'status': 'validation_not_observed_for_this_receiver_and_configuration',
                                    'evidence': {'seq': seq, 'file': event['file'], 'function': event['function'], 'line': event['line']}})
        node = state.get('node') or {}
        if event['event'] == 'return' and event['function'].endswith('._evaluate_node'):
            candidate = candidates.get((test, node.get('package_id')))
            if candidate and candidate.get(context) is not None and info.get(context) is not None and candidate[context] != info[context]:
                key = (test, node.get('receiver'), node.get('package_id'), info.get('receiver'))
                if key not in seen_mismatch:
                    seen_mismatch.add(key)
                    mismatches.append({'test': test, 'node_receiver': node.get('receiver'), 'node_binary': node.get('binary'),
                                       'package_id_configuration': {context: candidate[context]},
                                       'current_info_configuration': {context: info[context]},
                                       'current_info_receiver': info.get('receiver'),
                                       'invocation': event.get('invocation'), 'command': event.get('command'),
                                       'evidence': {'seq': seq, 'file': event['file'], 'function': event['function'], 'line': event['line']}})
            mode = state.get('build_mode') or {}
            tokens = [x for x in mode.get('patterns', []) if isinstance(x, str) and x.startswith('compatible')]
            if tokens and mode.get('_build_compatible_patterns') is None and test not in seen_mode:
                seen_mode.add(test)
                mode_errors.append({'test': test, 'requested_tokens': tokens, 'observed_storage': 'build_mode.patterns',
                                    'dedicated_compatible_state': None, 'node_binary': node.get('binary'),
                                    'invocation': event.get('invocation'), 'command': event.get('command'),
                                    'evidence': {'seq': seq, 'file': event['file'], 'function': event['function'], 'line': event['line']}})
    return {'schema_version': 1, 'identity_scope': ['test', 'receiver', context],
            'field': field, 'validator': validator,
            'unvalidated_candidates': unvalidated, 'identity_mismatches': mismatches,
            'compatible_mode_observations': mode_errors,
            'producer_execution_errors': producer_errors,
            'dispatch_contract_observations': dispatch_errors,
            'stale_guard_observations': stale_guards,
            'reference_patch_or_execution_used': False,
            'limits': ['Missing validation is an observed coverage gap, not an invented expected field value.',
                       'The context fingerprint used here is cppstd; broader repositories need their own watched context fields.']}


def render_prompt(packet, source_evidence):
    compact = dict(packet)
    compact['unvalidated_candidates'] = packet['unvalidated_candidates'][:12]
    compact['identity_mismatches'] = packet['identity_mismatches'][:4]
    compact['compatible_mode_observations'] = packet['compatible_mode_observations'][:4]
    return ('Use the following failing-test state observations to repair the implementation.\n'
            'Trace the first broken producer-to-consumer relation; do not patch only the final exception.\n'
            'Default False is not proof of successful validation. Match every validation and read to the SAME '
            'receiver and configuration after cloning, rebinding or changing settings.\n'
            'Keep consumption validity and buildability separate; failure to build must not automatically mark '
            'an otherwise consumable configuration invalid. Keep package identity and selected info/settings consistent.\n'
            'Preserve the existing missing-build path while implementing the requested compatible-build path.\n'
            'For each change, identify its state producer, consumer guard, required invariant and a test to verify it. '
            'Treat the observations as evidence, and mark unsupported hypotheses instead of inventing state values.\n'
            '<runtime_state_feedback>\n' + json.dumps(compact, ensure_ascii=False, indent=2)
            + '\n</runtime_state_feedback>\n<current_source_evidence>\n' + source_evidence
            + '\n</current_source_evidence>\n')
