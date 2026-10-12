import unittest

from generator.state_profiles.conan import extract


def event(seq, function, state, kind='return', test='case'):
    return {'seq': seq, 'function': function, 'state': state, 'event': kind,
            'test': test, 'file': 'pkg/current.py', 'line': seq}


class StatePacketTests(unittest.TestCase):
    def test_policy_input_without_runtime_state_becomes_an_upstream_causal_break(self):
        init = event(1, 'BuildMode.__init__', {
            'build_mode_input': {'params': ['compatible:&']},
            'build_mode': {'receiver': 'mode', 'patterns': ['compatible:&'],
                           'build_missing_patterns': []}})
        allowed = event(2, 'BuildMode.allowed_compatible', {
            'build_mode': {'receiver': 'mode'}, 'return_value': False})
        for item in [init, allowed]:
            item.update(invocation=7, command='create . --build=compatible:&', file='conans/client/graph/build_mode.py')
        packet = extract([init, allowed])
        flow = packet['causal_flow'][0]
        self.assertEqual(flow['stage'], 'input_to_state')
        self.assertEqual([a['role'] for a in flow['actors']],
                         ['policy_state_producer', 'policy_state_consumer'])
        self.assertEqual(flow['observation']['input'], ['compatible:&'])
        self.assertFalse(flow['observation']['permission_decision'])

    def test_candidate_field_without_same_configuration_validation_links_refresh_to_consumer(self):
        refresh = event(1, 'run_validate_package_id', {
            'current_info': {'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}, 'call')
        produced = event(2, 'BinaryCompatibility.compatibles', {'has_build_validator': True,
            'returned_candidates': [{'package_id': 'p14', 'info': {
                'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}]})
        consumed = event(3, 'GraphBinariesAnalyzer._find_build_compatible_binary', {
            'current_info': {'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}, 'line')
        consumed['statement'] = 'if compatible_package.cant_build:'
        for item, file in [(refresh, 'conans/client/graph/compute_pid.py'),
                           (produced, 'conans/client/graph/compatibility.py'),
                           (consumed, 'conans/client/graph/graph_binaries.py')]:
            item.update(invocation=4, command='create . --build=compatible', file=file)
        flow = extract([refresh, produced, consumed])['causal_flow'][0]
        self.assertEqual(flow['stage'], 'configuration_to_state')
        self.assertEqual([a['role'] for a in flow['actors']],
                         ['configuration_state_refresh', 'derived_state_consumer'])
        self.assertTrue(flow['observation']['consumer_observed'])

    def test_one_time_callback_owner_is_linked_to_shared_refresh_boundary(self):
        validation = event(1, 'Pkg.validate_build', {
            'current_info': {'receiver': 'main', 'cppstd': '11', 'cant_build': False}}, 'call')
        validation['watched_callers'] = [{'file': 'conans/client/graph/compute_pid.py',
                                          'function': 'compute_package_id', 'line': 60}]
        refresh = event(2, 'run_validate_package_id', {
            'current_info': {'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}, 'call')
        produced = event(3, 'BinaryCompatibility.compatibles', {'has_build_validator': True,
            'returned_candidates': [{'package_id': 'p14', 'info': {
                'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}]})
        consumed = event(4, 'GraphBinariesAnalyzer._find_build_compatible_binary', {
            'current_info': {'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}, 'line')
        consumed['statement'] = 'if not compatible.cant_build:'
        for item, file in [(validation, 'recipe/conanfile.py'),
                           (refresh, 'conans/client/graph/compute_pid.py'),
                           (produced, 'conans/client/graph/compatibility.py'),
                           (consumed, 'conans/client/graph/graph_binaries.py')]:
            item.update(invocation=4, command='create . --build=compatible', file=file)
        flows = extract([validation, refresh, produced, consumed])['causal_flow']
        placement = next(flow for flow in flows if flow['stage'] == 'responsibility_placement')
        self.assertEqual([actor['role'] for actor in placement['actors'][:2]],
                         ['current_one_time_state_owner', 'shared_configuration_refresh'])
        self.assertEqual(placement['actors'][0]['symbol'], 'compute_package_id')

    def test_terminal_state_without_declared_feature_functions_is_dispatch_flow(self):
        init = event(1, 'BuildMode.__init__', {
            'build_mode_input': {'params': ['compatible']},
            'build_mode': {'receiver': 'mode'}})
        evaluated = event(2, 'GraphBinariesAnalyzer._evaluate_node', {
            'node': {'receiver': 'node', 'binary': 'Invalid', 'package_id': 'p'}}, 'return')
        for item, file in [(init, 'conans/client/graph/build_mode.py'),
                           (evaluated, 'conans/client/graph/graph_binaries.py')]:
            item.update(invocation=9, command='create . --build=compatible', file=file)
        declared = {'conans/client/graph/build_mode.py': ['BuildMode.allowed_compatible'],
                    'conans/client/graph/graph_binaries.py': [
                        'GraphBinariesAnalyzer._find_build_compatible_binary']}
        flows = extract([init, evaluated], declared=declared)['causal_flow']
        dispatch = next(flow for flow in flows if flow['stage'] == 'state_to_feature_dispatch')
        self.assertEqual(dispatch['observation']['terminal_binary'], 'Invalid')
        self.assertEqual(len(dispatch['actors']), 3)
        self.assertEqual(dispatch['actors'][1]['symbol'], 'BuildMode.allowed_compatible')

    def test_missing_policy_entering_feature_handler_is_a_causal_boundary_break(self):
        init = event(1, 'BuildMode.__init__', {
            'build_mode_input': {'params': ['missing']},
            'build_mode': {'receiver': 'mode', 'missing': True}})
        entered = event(2, 'GraphBinariesAnalyzer._find_build_compatible_binary', {
            'node': {'receiver': 'node', 'binary': 'Missing'}}, 'call')
        entered['watched_callers'] = [{
            'file': 'conans/client/graph/graph_binaries.py',
            'function': 'GraphBinariesAnalyzer._evaluate_node', 'line': 210}]
        decision = event(3, 'BuildMode.allowed_compatible', {
            'build_mode': {'receiver': 'mode', 'missing': True}, 'return_value': True})
        for item, file in [(init, 'conans/client/graph/build_mode.py'),
                           (entered, 'conans/client/graph/graph_binaries.py'),
                           (decision, 'conans/client/graph/build_mode.py')]:
            item.update(invocation=2, command='install . --build=missing', file=file)
        packet = extract([init, entered, decision])
        flow = next(item for item in packet['causal_flow']
                    if item['stage'] == 'policy_to_dispatch')
        self.assertEqual(flow['observation']['request_policy'], 'missing')
        self.assertTrue(flow['observation']['feature_permission_decision'])
        self.assertEqual([actor['role'] for actor in flow['actors']],
                         ['mode_dispatch_controller', 'feature_policy_consumer',
                          'feature_specific_responsibility'])
        self.assertTrue(packet['policy_decision_observations'][0]['decision'])

    def test_validation_of_main_info_does_not_validate_a_different_candidate(self):
        trace = [event(1, 'Pkg.validate_build', {'current_info': {'receiver': 'main', 'cppstd': '11'}}, 'call'),
                 event(2, 'Compatibility.compatibles', {'returned_candidates': [
                     {'package_id': 'p14', 'info': {'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}]})]
        packet = extract(trace)
        self.assertEqual(packet['unvalidated_candidates'][0]['receiver'], 'candidate')
        self.assertIs(packet['unvalidated_candidates'][0]['observed_value'], False)

    def test_validation_becomes_stale_when_same_receiver_configuration_changes(self):
        trace = [event(1, 'Pkg.validate_build', {'current_info': {'receiver': 'same', 'cppstd': '11'}}, 'call'),
                 event(2, 'Compatibility.compatibles', {'returned_candidates': [
                     {'package_id': 'p14', 'info': {'receiver': 'same', 'cppstd': '14', 'cant_build': False}}]})]
        self.assertEqual(len(extract(trace)['unvalidated_candidates']), 1)

    def test_candidate_validation_in_same_configuration_is_recognized(self):
        trace = [event(1, 'Pkg.validate_build', {'current_info': {'receiver': 'candidate', 'cppstd': '14'}}, 'call'),
                 event(2, 'Compatibility.compatibles', {'returned_candidates': [
                     {'package_id': 'p14', 'info': {'receiver': 'candidate', 'cppstd': '14', 'cant_build': 'needs 17'}}]})]
        self.assertEqual(extract(trace)['unvalidated_candidates'], [])

    def test_callback_free_recipe_does_not_get_a_spurious_missing_validation_warning(self):
        trace = [event(1, 'Compatibility.compatibles', {'returned_candidates': [
                     {'package_id': 'p14', 'info': {'receiver': 'candidate', 'cppstd': '14', 'cant_build': False}}]})]
        self.assertEqual(extract(trace)['unvalidated_candidates'], [])

    def test_node_package_identity_must_match_selected_info_configuration(self):
        trace = [event(1, 'Compatibility.compatibles', {'returned_candidates': [
                     {'package_id': 'p23', 'info': {'receiver': 'c23', 'cppstd': 'gnu23', 'cant_build': False}}]}),
                 event(2, 'Graph._evaluate_node', {'node': {'receiver': 'node', 'binary': 'Invalid', 'package_id': 'p23'},
                                                   'current_info': {'receiver': 'main', 'cppstd': '11'}})]
        packet = extract(trace)
        self.assertEqual(packet['identity_mismatches'][0]['package_id_configuration']['cppstd'], 'gnu23')
        self.assertEqual(packet['identity_mismatches'][0]['current_info_configuration']['cppstd'], '11')

    def test_other_recipe_validator_does_not_require_validation_of_a_callback_free_recipe(self):
        trace = [event(1, 'Pkg.validate_build', {'current_info': {'receiver': 'main', 'cppstd': '11'}}, 'call'),
                 event(2, 'Compatibility.compatibles', {'has_build_validator': False, 'returned_candidates': [
                     {'package_id': 'p14', 'info': {'receiver': 'other-recipe', 'cppstd': '14', 'cant_build': False}}]})]
        self.assertEqual(extract(trace)['unvalidated_candidates'], [])

    def test_missing_import_is_reported_at_the_candidate_producer_with_invocation(self):
        trace = [event(1, 'Compatibility.compatibles', {'current_info': {'receiver': 'c14', 'cppstd': '14'},
                     'exception': {'type': 'NameError', 'message': 'missing_name'}}, 'exception')]
        trace[0].update(invocation=2, command='create . --build=compatible')
        errors = extract(trace)['producer_execution_errors']
        self.assertEqual(errors[0]['receiver'], 'c14')
        self.assertEqual(errors[0]['invocation'], 2)

    def test_guard_is_not_rechecked_after_later_state_update(self):
        trace = [event(1, 'Graph._evaluate_node', {'node': {'receiver': 'n', 'binary': 'Missing'}}, 'line'),
                 event(2, 'Graph._evaluate_node', {'node': {'receiver': 'n', 'binary': 'Invalid'}})]
        trace[0]['statement'] = 'if node.binary == BINARY_INVALID and mode.allowed_compatible():'
        for e in trace:e.update(invocation=3, command='create . --build=compatible')
        stale = extract(trace)['stale_guard_observations']
        self.assertEqual(stale[0]['guard_observation']['binary'], 'Missing')
        self.assertEqual(stale[0]['later_value'], 'Invalid')

    def test_guard_rechecked_with_updated_state_is_not_stale(self):
        trace = [event(1, 'Graph._evaluate_node', {'node': {'receiver': 'n', 'binary': 'Missing'}}, 'line'),
                 event(2, 'Graph._evaluate_node', {'node': {'receiver': 'n', 'binary': 'Invalid'}}, 'line'),
                 event(3, 'Graph._evaluate_node', {'node': {'receiver': 'n', 'binary': 'Invalid'}})]
        for e in trace:
            e.update(invocation=3, command='create . --build=compatible')
            if e['event']=='line':e['statement']='if node.binary == BINARY_INVALID:'
        self.assertEqual(extract(trace)['stale_guard_observations'], [])


if __name__ == '__main__':
    unittest.main()
