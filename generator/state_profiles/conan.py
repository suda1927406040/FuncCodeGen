"""Extract receiver- and configuration-specific feedback from runtime events.

This module reads failing-run events only. Reference executions are not inputs.
"""
from collections import defaultdict
import json


def extract(events, field='cant_build', validator='validate_build', context='cppstd', declared=None):
    validation_calls = defaultdict(list)
    refresh_calls = defaultdict(list)
    field_reads = defaultdict(list)
    test_has_validator = set()
    candidates = {}
    unvalidated, mismatches, mode_errors, producer_errors, dispatch_errors, stale_guards = [], [], [], [], [], []
    seen_unvalidated, seen_mismatch, seen_mode = set(), set(), set()
    last_invalid_guard = {}
    mode_initializations, policy_decisions = [], []

    def actor(event, role):
        return {'file': event.get('file'), 'symbol': event.get('function'), 'role': role,
                'line': event.get('line')}

    for event in events:
        test, seq, state = event['test'], event['seq'], event['state']
        info = state.get('current_info') or {}
        command = event.get('command') or ''
        node = state.get('node') or {}
        scope = (test, event.get('invocation'), node.get('receiver'))
        if (event['event'] == 'call' and event['function'].endswith('._find_build_compatible_binary')
                and '--build=missing' in command and '--build=compatible' not in command):
            caller = (event.get('watched_callers') or [None])[0]
            dispatch_errors.append({'test': test, 'invocation': event.get('invocation'), 'command': command,
                                    'observed_consumer': event['function'], 'node_binary': node.get('binary'),
                                    'expected_role': 'ordinary missing-binary cache/remote search',
                                    'controller': caller,
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
            key = (test, event.get('invocation'), info.get('receiver'), info.get(context))
            validation_calls[key].append(seq)
            test_has_validator.add(test)
        if event['event'] == 'call' and event['function'].split('.')[-1] == 'run_validate_package_id':
            key = (test, event.get('invocation'), info.get('receiver'), info.get(context))
            refresh_calls[key].append(event)
        if field in event.get('statement', '') and info.get('receiver') is not None:
            key = (test, event.get('invocation'), info.get('receiver'), info.get(context))
            field_reads[key].append(event)
        mode = state.get('build_mode') or {}
        mode_input = state.get('build_mode_input') or {}
        if event['event'] == 'return' and event['function'].endswith('BuildMode.__init__'):
            mode_initializations.append((event, list(mode_input.get('params') or []), mode))
        leaf = event.get('function', '').split('.')[-1]
        if event['event'] == 'return' and leaf.startswith('allowed_'):
            policy_decisions.append((event, leaf[len('allowed_'):],
                                     state.get('return_value', event.get('return_value')),
                                     mode.get('receiver')))
        for candidate in state.get('returned_candidates', []):
            value = candidate['info']
            candidates[(test, event.get('invocation'), candidate['package_id'])] = value
            key = (test, event.get('invocation'), value.get('receiver'), value.get(context))
            validated = any(s < seq for s in validation_calls[key])
            has_validator = state.get('has_build_validator', test in test_has_validator)
            if has_validator and not validated and key not in seen_unvalidated:
                seen_unvalidated.add(key)
                unvalidated.append({'test': test, 'receiver': value.get('receiver'),
                                    'configuration': {context: value.get(context)}, 'field': field,
                                    'observed_value': value.get(field),
                                    'invocation': event.get('invocation'), 'command': event.get('command'),
                                    'status': 'validation_not_observed_for_this_receiver_and_configuration',
                                    'producer': actor(event, 'candidate_state_producer'),
                                    'evidence': {'seq': seq, 'file': event['file'], 'function': event['function'], 'line': event['line']}})
        node = state.get('node') or {}
        if event['event'] == 'return' and event['function'].endswith('._evaluate_node'):
            candidate = candidates.get((test, event.get('invocation'), node.get('package_id')))
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
    policy_concepts = {concept for _, concept, _, _ in policy_decisions}
    for event, params, mode in mode_initializations:
        for token in params:
            if not isinstance(token, str):
                continue
            concept = token.split(':', 1)[0]
            if concept not in policy_concepts:
                continue
            matching = [(decision, outcome) for decision, name, outcome, receiver in policy_decisions
                        if name == concept and decision.get('test') == event.get('test')
                        and decision.get('invocation') == event.get('invocation')
                        and receiver == mode.get('receiver')]
            # If the policy consumer was never reached, the observed break is
            # downstream dispatch, not evidence that the input state producer
            # represented the request incorrectly.
            if not matching:
                continue
            if matching and any(outcome is True for _, outcome in matching):
                continue
            key = (event.get('test'), event.get('invocation'), mode.get('receiver'), concept)
            if key in seen_mode:
                continue
            seen_mode.add(key)
            mode_errors.append({'test': event.get('test'), 'requested_tokens': [token],
                                'resulting_policy_state': {k: v for k, v in mode.items() if k != 'receiver'},
                                'permission_decision': (matching[-1][1] if matching else 'not_observed'),
                                'invocation': event.get('invocation'), 'command': event.get('command'),
                                'producer': actor(event, 'policy_state_producer'),
                                'consumer': actor(matching[-1][0], 'policy_state_consumer') if matching else None,
                                'evidence': {'seq': event.get('seq'), 'file': event.get('file'),
                                             'function': event.get('function'), 'line': event.get('line')}})
    causal_flow = []
    for record in mode_errors:
        actors = [record['producer']]
        if record.get('consumer'):
            actors.append(record['consumer'])
        causal_flow.append({'stage': 'input_to_state', 'status': 'broken', 'test': record['test'],
                            'invocation': record['invocation'], 'command': record.get('command'),
                            'actors': actors,
                            'relation': 'A request-specific policy input was observed, but its permission decision was false or was not reached.',
                            'required_invariant': 'A request-specific policy input must produce state that its permission decision consumes.',
                            'observation': {'input': record['requested_tokens'],
                                            'resulting_state': record['resulting_policy_state'],
                                            'permission_decision': record['permission_decision']},
                            'evidence': record['evidence']})
    for record in unvalidated:
        key = (record['test'], record.get('invocation'), record.get('receiver'),
               record.get('configuration', {}).get(context))
        refresh = refresh_calls.get(key, [])
        reads = [event for event in field_reads.get(key, []) if event.get('seq', 0) > record['evidence']['seq']]
        actors = []
        if refresh:
            actors.append(actor(refresh[-1], 'configuration_state_refresh'))
        else:
            actors.append(record['producer'])
        if reads:
            actors.append(actor(reads[0], 'derived_state_consumer'))
        causal_flow.append({'stage': 'configuration_to_state', 'status': 'broken', 'test': record['test'],
                            'invocation': record.get('invocation'), 'command': record.get('command'),
                            'actors': actors,
                            'relation': 'A configuration-specific candidate reached a state-refresh boundary without the derived field being validated for the same receiver and configuration.',
                            'required_invariant': 'After an object or configuration changes, every derived field used downstream must be recomputed for that same receiver and configuration.',
                            'observation': {'receiver': record['receiver'],
                                            'configuration': record['configuration'],
                                            'field': field, 'observed_value': record['observed_value'],
                                            'validator': validator, 'consumer_observed': bool(reads)},
                            'evidence': record['evidence']})
        # A callback can be correct for the initial object yet stale for cloned
        # configurations when it lives above the shared refresh boundary.
        owners = []
        for event in events:
            if (event.get('test') == record['test'] and event.get('invocation') == record.get('invocation')
                    and event.get('event') == 'call'
                    and event.get('function', '').split('.')[-1] == validator):
                owners.extend(event.get('watched_callers') or [])
        if refresh and owners:
            owner = owners[0]
            refresh_actor = actor(refresh[-1], 'shared_configuration_refresh')
            owner_actor = {'file': owner.get('file'), 'symbol': owner.get('function'),
                           'role': 'current_one_time_state_owner', 'line': owner.get('line')}
            placement_actors = [owner_actor, refresh_actor]
            if reads:
                placement_actors.append(actor(reads[0], 'derived_state_consumer'))
            causal_flow.append({
                'stage': 'responsibility_placement', 'status': 'broken', 'test': record['test'],
                'invocation': record.get('invocation'), 'command': record.get('command'),
                'actors': placement_actors,
                'relation': 'A configuration-dependent callback ran under a one-time state owner, but not when the shared refresh boundary was reused for another receiver or configuration.',
                'required_invariant': 'A configuration-dependent responsibility must run exactly once inside the shared refresh boundary used for every receiver and configuration.',
                'observation': {'validator': validator, 'current_owner': owner,
                                'shared_refresh': refresh_actor,
                                'unrefreshed_receiver': record['receiver'],
                                'configuration': record['configuration']},
                'evidence': record['evidence']})

    for record in dispatch_errors:
        decisions = [(event, outcome) for event, name, outcome, _ in policy_decisions
                     if name == 'compatible' and event.get('test') == record['test']
                     and event.get('invocation') == record.get('invocation')]
        actors = []
        controller = record.get('controller')
        if controller:
            actors.append({'file': controller.get('file'), 'symbol': controller.get('function'),
                           'role': 'mode_dispatch_controller', 'line': controller.get('line')})
        if decisions:
            actors.append(actor(decisions[-1][0], 'feature_policy_consumer'))
        actors.append({'file': record['evidence'].get('file'),
                       'symbol': record.get('observed_consumer'),
                       'role': 'feature_specific_responsibility',
                       'line': record['evidence'].get('line')})
        causal_flow.append({
            'stage': 'policy_to_dispatch', 'status': 'broken', 'test': record['test'],
            'invocation': record.get('invocation'), 'command': record.get('command'),
            'actors': actors,
            'relation': 'An ordinary missing-build request entered a feature-specific compatible-build responsibility without that feature policy in the request.',
            'required_invariant': 'Each build policy must dispatch to its own responsibility; ordinary missing-binary search and feature-specific compatible building remain separate paths.',
            'observation': {'request_policy': 'missing',
                            'feature_permission_decision': decisions[-1][1] if decisions else 'not_observed',
                            'entered_responsibility': record.get('observed_consumer'),
                            'node_binary': record.get('node_binary')},
            'evidence': record['evidence']})

    decision_observations = []
    for event, concept, outcome, receiver in policy_decisions:
        decision_observations.append({
            'test': event.get('test'), 'invocation': event.get('invocation'),
            'command': event.get('command'), 'policy': concept, 'decision': outcome,
            'receiver': receiver,
            'policy_state': {k: v for k, v in (event.get('state', {}).get('build_mode') or {}).items()
                             if k != 'receiver'},
            'evidence': {'seq': event.get('seq'), 'file': event.get('file'),
                         'function': event.get('function'), 'line': event.get('line')}})

    declared = declared or {}
    declared_targets = [(file, symbol) for file, symbols in declared.items() for symbol in symbols]
    if declared_targets:
        policy_concepts = {symbol.split('.')[-1][len('allowed_'):]
                           for _, symbol in declared_targets
                           if symbol.split('.')[-1].startswith('allowed_')}
        requested_scopes = set()
        for event, params, _ in mode_initializations:
            if any(isinstance(token, str) and token.split(':', 1)[0] in policy_concepts
                   for token in params):
                requested_scopes.add((event.get('test'), event.get('invocation')))
        calls = defaultdict(set)
        for event in events:
            if event.get('event') != 'call':
                continue
            scope = (event.get('test'), event.get('invocation'))
            leaf = event.get('function', '').split('.')[-1]
            for file, symbol in declared_targets:
                if event.get('file') == file and leaf == symbol.split('.')[-1]:
                    calls[scope].add((file, symbol))
        seen_dispatch = set()
        for event in events:
            if not (event.get('event') == 'return'
                    and event.get('function', '').endswith('._evaluate_node')):
                continue
            state = event.get('state') or {}
            node = state.get('node') or {}
            if node.get('binary') not in {'Invalid', 'Missing'}:
                continue
            scope = (event.get('test'), event.get('invocation'))
            if scope not in requested_scopes:
                continue
            missing = [(file, symbol) for file, symbol in declared_targets
                       if (file, symbol) not in calls.get(scope, set())]
            key = scope + tuple(missing)
            if not missing or key in seen_dispatch:
                continue
            seen_dispatch.add(key)
            actors = [actor(event, 'feature_dispatch_controller')]
            actors.extend({'file': file, 'symbol': symbol,
                           'role': 'declared_feature_responsibility_not_reached'}
                          for file, symbol in missing)
            causal_flow.append({
                'stage': 'state_to_feature_dispatch', 'status': 'broken',
                'test': event.get('test'), 'invocation': event.get('invocation'),
                'command': event.get('command'), 'actors': actors,
                'relation': 'Execution ended in a terminal binary state without reaching one or more declared feature responsibilities in the same command.',
                'required_invariant': 'After state production, dispatch must reach the feature responsibilities under their policy conditions while preserving existing command paths.',
                'observation': {'terminal_binary': node.get('binary'),
                                'unreached_responsibilities': [symbol for _, symbol in missing]},
                'evidence': {'seq': event.get('seq'), 'file': event.get('file'),
                             'function': event.get('function'), 'line': event.get('line')}})
    return {'schema_version': 1, 'identity_scope': ['test', 'invocation', 'receiver', context],
            'field': field, 'validator': validator,
            'causal_flow': causal_flow,
            'unvalidated_candidates': unvalidated, 'identity_mismatches': mismatches,
            'compatible_mode_observations': mode_errors,
            'policy_decision_observations': decision_observations,
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
