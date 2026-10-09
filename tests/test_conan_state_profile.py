import unittest

from generator.state_profiles.conan import extract


def event(seq, function, state, kind='return', test='case'):
    return {'seq': seq, 'function': function, 'state': state, 'event': kind,
            'test': test, 'file': 'pkg/current.py', 'line': seq}


class StatePacketTests(unittest.TestCase):
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
