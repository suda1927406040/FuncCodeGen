import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from generator.feedback_evidence import evidence_bundle, failure_excerpt, read_failure_output, test_delta
from generator.feature_memory import (attach_observations, execution_obligations, merge_update,
                                      preserve_runtime_observations, seed_featurespace, memory_view)
from generator.llm_interface import LLMResponse
from generator.state_feedback import (_edited_responsibilities, diagnosis_prompt,
                                      repair_once, repair_prompt, watch_config)


def response(value):
    return LLMResponse(json.dumps(value), 'openai', 'test', 'stop', 10, 20, 30)


def section(prompt, name):
    return json.loads(prompt.split('<' + name + '>\n', 1)[1].split('\n</' + name + '>', 1)[0])


class EvidenceTests(unittest.TestCase):
    def test_causal_stages_survive_as_separate_evidence_and_enter_featurespace(self):
        actors = lambda left, right: [
            {'file': 'policy.py', 'symbol': left, 'role': 'producer'},
            {'file': 'select.py', 'symbol': right, 'role': 'consumer'}]
        packet = {'repo_state_relations': {'causal_flow': [
            {'stage': 'input_to_state', 'status': 'broken', 'test': 'a',
             'actors': actors('parse', 'allow'), 'relation': 'policy state is absent',
             'required_invariant': 'policy input reaches its consumer'},
            {'stage': 'configuration_to_state', 'status': 'broken', 'test': 'a',
             'actors': actors('refresh', 'select'), 'relation': 'derived state is stale',
             'required_invariant': 'derived state is refreshed after configuration changes'}]}}
        bundle = evidence_bundle(packet, 'E   AssertionError: wrong state')
        self.assertGreater(bundle['categories']['repo_state_relations.causal_flow.input_to_state']['selected'], 0)
        self.assertGreater(bundle['categories']['repo_state_relations.causal_flow.configuration_to_state']['selected'], 0)
        instance = {'instance_id': 'flow', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': []}
        memory = attach_observations(seed_featurespace(instance), bundle, 'patch', 1)
        observed = [entry for entry in memory['groups'].values()
                    if entry.get('origin') == 'execution_flow']
        self.assertEqual({entry['flow_stage'] for entry in observed},
                         {'input_to_state', 'configuration_to_state'})
        self.assertTrue(memory['focus'])
        view = memory_view(memory)
        self.assertTrue(any(entry.get('flow_status') == 'broken' for entry in view['conditions']))
        obligations = execution_obligations(memory)
        self.assertEqual({item['stage'] for item in obligations},
                         {'input_to_state', 'configuration_to_state'})
        targets = {(target['file'], target['symbol']) for item in obligations
                   for target in item['required_repair_targets']}
        self.assertEqual(targets, {('policy.py', 'parse'), ('select.py', 'allow'),
                                   ('policy.py', 'refresh'), ('select.py', 'select')})

        observed_group = next(entry for entry in memory['groups'].values()
                              if entry.get('origin') == 'execution_flow')
        delta = {'summary': 'try to erase observation', 'retire': [{
            'id': observed_group['id'], 'reason': 'ignore runtime',
            'evidence': observed_group['evidence']}], 'focus': [], 'checks': []}
        with self.assertRaisesRegex(ValueError, 'cannot be retired'):
            merge_update(memory, delta, ['policy.py', 'select.py'])

    def test_invalid_runtime_overwrite_can_be_filtered_without_losing_diagnosis(self):
        instance = {'instance_id': 'salvage', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': []}
        packet = {'repo_state_relations': {'causal_flow': [{
            'stage': 'policy_to_dispatch', 'status': 'broken', 'test': 'missing',
            'actors': [{'file': 'mode.py', 'symbol': 'Mode.allowed',
                        'role': 'feature_policy_consumer_wrong_decision'}],
            'relation': 'The policy returned the wrong decision.',
            'required_invariant': 'The feature policy rejects ordinary requests.'}]}}
        memory = attach_observations(
            seed_featurespace(instance), evidence_bundle(packet, 'failed'), 'candidate', 1)
        group = next(entry for entry in memory['groups'].values()
                     if entry.get('origin') == 'execution_flow')
        evidence = group['evidence']
        delta = {
            'summary': 'The policy state and ordinary request are conflated.',
            'upsert': {'groups': [{
                'id': group['id'], 'text': 'attempted overwrite',
                'members': group['members'], 'evidence': evidence}]},
            'focus': [group['id']],
            'checks': [{
                'hypothesis': 'The policy grants an ordinary request.',
                'expected_change': 'The permission decision becomes false.',
                'evidence': evidence, 'basis': 'Observed decision.'}]}
        filtered, removed = preserve_runtime_observations(memory, delta)
        self.assertEqual(removed[0]['id'], group['id'])
        merged = merge_update(memory, filtered, ['mode.py'])
        self.assertEqual(merged['groups'][group['id']]['text'], group['text'])
        self.assertEqual(merged['diagnosis'], delta['summary'])
        self.assertEqual(merged['checks'][0]['hypothesis'],
                         'The policy grants an ordinary request.')

    def test_new_regression_becomes_a_featurespace_preservation_flow(self):
        instance = {'instance_id': 'regression', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': []}
        memory = seed_featurespace(instance, recall={'candidates': [
            {'file': 'policy.py', 'symbols': ['parse']} ]})
        packet = {'failure_routes': [{'test': 'test_old_behavior', 'target_calls': [
            {'file': 'policy.py', 'function': 'parse', 'calls': 1, 'status': 'called'}]}]}
        bundle = evidence_bundle(packet, 'E   AssertionError: regression')
        history = [{'patch_after': 'candidate', 'changed_files': ['policy.py'],
                    'test_delta': {'new_failures': ['test_old_behavior']}}]
        memory = attach_observations(memory, bundle, 'candidate', 2, history)
        groups = [entry for entry in memory['groups'].values()
                  if entry.get('origin') == 'regression_flow']
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['test'], 'test_old_behavior')
        obligation = next(item for item in execution_obligations(memory)
                          if item['stage'] == 'behavior_preservation')
        self.assertEqual(obligation['previously_passing_test'], 'test_old_behavior')
        self.assertEqual(obligation['required_repair_targets'], [])

    def test_terminal_dispatch_flow_keeps_context_without_forcing_one_edit_location(self):
        instance = {'instance_id': 'dispatch', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': []}
        packet = {'repo_state_relations': {'causal_flow': [{
            'stage': 'state_to_feature_dispatch', 'status': 'broken', 'test': 'feature',
            'actors': [
                {'file': 'graph.py', 'symbol': 'Graph.evaluate', 'role': 'feature_dispatch_controller'},
                {'file': 'mode.py', 'symbol': 'Mode.allowed', 'role': 'declared_feature_responsibility_not_reached'}],
            'relation': 'A terminal state prevented feature dispatch.',
            'required_invariant': 'Restore dispatch after state production.'}]}}
        memory = attach_observations(
            seed_featurespace(instance), evidence_bundle(packet, 'failed'), 'candidate', 1)
        obligation = next(item for item in execution_obligations(memory)
                          if item['stage'] == 'state_to_feature_dispatch')
        self.assertEqual(len(obligation['responsibilities']), 2)
        self.assertEqual(obligation['required_repair_targets'], [])

    def test_wrong_policy_dispatch_requires_only_the_observed_permission_consumer(self):
        instance = {'instance_id': 'policy', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': []}
        packet = {'repo_state_relations': {'causal_flow': [{
            'stage': 'policy_to_dispatch', 'status': 'broken', 'test': 'missing',
            'actors': [
                {'file': 'graph.py', 'symbol': 'Graph.evaluate', 'role': 'mode_dispatch_controller'},
                {'file': 'mode.py', 'symbol': 'Mode.allowed_feature', 'role': 'feature_policy_consumer_wrong_decision'},
                {'file': 'graph.py', 'symbol': 'Graph.feature_handler', 'role': 'feature_specific_responsibility'}],
            'relation': 'An ordinary request entered the feature-specific handler.',
            'required_invariant': 'Each policy dispatches only to its own responsibility.'}]}}
        memory = attach_observations(
            seed_featurespace(instance), evidence_bundle(packet, 'failed'), 'candidate', 1)
        obligation = next(item for item in execution_obligations(memory)
                          if item['stage'] == 'policy_to_dispatch')
        self.assertEqual(obligation['required_repair_targets'], [
            {'file': 'mode.py', 'symbol': 'Mode.allowed_feature'}])

    def test_late_rejected_policy_dispatch_requires_the_controller(self):
        instance = {'instance_id': 'late-policy', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': []}
        packet = {'repo_state_relations': {'causal_flow': [{
            'stage': 'policy_to_dispatch', 'status': 'broken', 'test': 'missing',
            'actors': [
                {'file': 'graph.py', 'symbol': 'Graph.evaluate', 'role': 'mode_dispatch_controller'},
                {'file': 'mode.py', 'symbol': 'Mode.allowed_feature', 'role': 'feature_policy_consumer_after_dispatch'},
                {'file': 'graph.py', 'symbol': 'Graph.feature_handler', 'role': 'feature_specific_responsibility'}],
            'relation': 'The handler ran before the policy rejected the request.',
            'required_invariant': 'Evaluate permission before entering the handler.'}]}}
        memory = attach_observations(
            seed_featurespace(instance), evidence_bundle(packet, 'failed'), 'candidate', 1)
        obligation = next(item for item in execution_obligations(memory)
                          if item['stage'] == 'policy_to_dispatch')
        self.assertEqual(obligation['required_repair_targets'], [
            {'file': 'graph.py', 'symbol': 'Graph.evaluate'}])

    def test_current_flow_roles_override_historical_roles_on_same_elements(self):
        instance = {'instance_id': 'role-history', 'repo': 'test/repo',
                    'base_commit': 'base', 'new_components': []}
        def packet(policy_role, relation):
            return {'repo_state_relations': {'causal_flow': [{
                'stage': 'policy_to_dispatch', 'status': 'broken', 'test': 'missing',
                'actors': [
                    {'file': 'graph.py', 'symbol': 'Graph.evaluate',
                     'role': 'mode_dispatch_controller'},
                    {'file': 'mode.py', 'symbol': 'Mode.allowed_feature',
                     'role': policy_role},
                    {'file': 'graph.py', 'symbol': 'Graph.feature_handler',
                     'role': 'feature_specific_responsibility'}],
                'relation': relation,
                'required_invariant': 'Dispatch at the correct policy boundary.'}]}}
        memory = attach_observations(
            seed_featurespace(instance),
            evidence_bundle(packet('feature_policy_consumer_wrong_decision', 'wrong decision'), 'failed'),
            'patch-one', 1)
        memory = attach_observations(
            memory,
            evidence_bundle(packet('feature_policy_consumer_after_dispatch', 'late gate'), 'failed'),
            'patch-two', 2)
        policy = next(entry for entry in memory['elements'].values()
                      if entry['symbol'] == 'Mode.allowed_feature')
        self.assertIn('feature_policy_consumer_wrong_decision', policy['execution_roles'])
        obligation = next(item for item in execution_obligations(memory)
                          if item['stage'] == 'policy_to_dispatch')
        self.assertEqual(obligation['required_repair_targets'], [
            {'file': 'graph.py', 'symbol': 'Graph.evaluate'}])
        policy_responsibility = next(
            item for item in obligation['responsibilities']
            if item['symbol'] == 'Mode.allowed_feature')
        self.assertEqual(policy_responsibility['roles'],
                         ['feature_policy_consumer_after_dispatch'])

    def test_diagnosis_checks_bind_named_functions_as_verification_responsibilities(self):
        instance = {'instance_id': 'checks', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': [{'file': 'mode.py', 'components': [
                        {'name': 'Mode.allowed_feature'}]}]}
        memory = seed_featurespace(instance, recall={'candidates': [
            {'file': 'refresh.py', 'symbols': ['refresh_state']} ]})
        memory['current_patch_sha256'] = 'candidate'
        memory['checks'] = [{
            'hypothesis': 'Mode.allowed_feature still rejects the represented policy state.',
            'expected_change': 'allowed_feature returns true after refresh_state recomputes it.',
            'evidence': ['requirement'], 'patch_sha256': 'candidate'}]
        obligation = next(item for item in execution_obligations(memory)
                          if item['stage'] == 'diagnosis_check')
        self.assertEqual({(target['file'], target['symbol'])
                          for target in obligation['responsibilities']},
                         {('mode.py', 'Mode.allowed_feature'),
                          ('refresh.py', 'refresh_state')})
        self.assertEqual(obligation['required_repair_targets'], [])

    def test_qualified_constructor_check_does_not_bind_other_constructors(self):
        instance = {'instance_id': 'constructors', 'repo': 'test/repo', 'base_commit': 'base',
                    'new_components': [
                        {'file': 'mode.py', 'components': [{'name': 'BuildMode.__init__'}]},
                        {'file': 'node.py', 'components': [{'name': 'Node.__init__'}]}]}
        memory = seed_featurespace(instance)
        memory['current_patch_sha256'] = 'candidate'
        memory['checks'] = [{
            'hypothesis': 'BuildMode.__init__ fails to represent the requested policy.',
            'expected_change': 'BuildMode.__init__ stores the policy state.',
            'evidence': ['requirement'], 'patch_sha256': 'candidate'}]
        obligation = next(item for item in execution_obligations(memory)
                          if item['stage'] == 'diagnosis_check')
        self.assertEqual(obligation['responsibilities'], [
            {'file': 'mode.py', 'symbol': 'BuildMode.__init__'}])
        self.assertEqual(obligation['required_repair_targets'], [])

    def test_trace_metadata_is_summarized_before_evidence_budgeting(self):
        packet = {'trace_metadata': {'trace_status': 'observed', 'trace_events': 50,
                  'observer_summaries': [{'process': 1, 'finished': True, 'observer_errors': 0,
                                          'truncated': False, 'truncated_tests': [],
                                          'watch': {'large.py': ['name'] * 1000}}]}}
        bundle = evidence_bundle(packet, 'failed')
        self.assertEqual(bundle['trace_metadata']['observer_summary']['finished_processes'], 1)
        self.assertNotIn('watch', json.dumps(bundle['trace_metadata']))

    def test_real_failures_survive_large_harness_source_dump(self):
        prefix = '+    raise AssertionError("not an actual failure")\n' * 100000
        output = prefix + '''============================= test session starts ==============================
collected 2 items
=================================== FAILURES ===================================
________________________ TestBuild.test_compatible ________________________
E   AssertionError: expected Build, got Invalid
________________________ TestBuild.test_missing ________________________
E   AssertionError: missing must retain its configuration
=========================== short test summary info ============================
FAILED test_build.py::TestBuild::test_compatible
FAILED test_build.py::TestBuild::test_missing
'''
        excerpt = failure_excerpt(output)
        self.assertIn('expected Build, got Invalid', excerpt)
        self.assertIn('missing must retain its configuration', excerpt)
        self.assertNotIn('not an actual failure', excerpt)
        self.assertLessEqual(len(excerpt), 16000)

    def test_old_truncated_evaluation_falls_back_to_diagnostic_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evaluated, diagnosed = root/'evaluation', root/'diagnosis'
            evaluated.mkdir(); diagnosed.mkdir()
            (evaluated/'test_output.txt').write_text('+ raise AssertionError("source dump")\n')
            (diagnosed/'diagnostic_output.txt').write_text('==== FAILURES ====\nE   AssertionError: actual failure\n')
            output, source = read_failure_output(evaluated, diagnosed)
            self.assertEqual(source, 'diagnosis')
            self.assertIn('actual failure', output)

    def test_all_relation_categories_survive_large_packets(self):
        categories = ['unvalidated_candidates', 'compatible_mode_observations', 'dispatch_contract_observations']
        packet = {'conan_state_relations': {key: [{'test': 'case'+str(i), 'receiver': str(i),
                                                  'configuration': {'mode': 'B'}, 'details': 'x'*5000} for i in range(30)]
                                          for key in categories},
                  'state_transitions': [{'test': str(i), 'values': {'a': 'v'*10000}} for i in range(30)]}
        bundle = evidence_bundle(packet, 'E   AssertionError: expected state', budget=18000)
        self.assertLessEqual(len(json.dumps(bundle, ensure_ascii=False)), 18000)
        for key in categories:
            counts = bundle['categories']['conan_state_relations.'+key]
            self.assertEqual(counts['total'], 30)
            self.assertGreaterEqual(counts['selected'], 1)
        self.assertIn('coverage gap', ' '.join(bundle['limits']))

    def test_delta_does_not_call_unobserved_tests_fixed(self):
        delta = test_delta({'failed': ['a', 'b'], 'passed': ['c']}, {'failed': ['c'], 'passed': ['a']})
        self.assertEqual(delta['fixed'], ['a'])
        self.assertEqual(delta['new_failures'], ['c'])
        self.assertEqual(delta['not_observed'], ['b'])


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root/'repo'; self.repo.mkdir()
        (self.repo/'state.py').write_text('def select(node):\n    return "Missing"\n')
        for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base')]:
            subprocess.run(['git', *args], cwd=self.repo, check=True, capture_output=True)
        self.instance = {'instance_id': 'case', 'repo': 'test/repo', 'base_commit': 'base',
                         'problem_statement': 'Select a buildable candidate while retaining existing modes.',
                         'new_components': [{'file': 'state.py', 'components': [{'name': 'select'}]}],
                         'patch': 'GOLD_POISON', 'test_patch': 'TEST_PATCH_POISON'}
        self.initial = ('diff --git a/state.py b/state.py\n--- a/state.py\n+++ b/state.py\n@@ -1,2 +1,2 @@\n'
                        ' def select(node):\n-    return "Missing"\n+    return "Invalid"\n')
        self.plan = {'steps': [{'id': 'S1', 'file': 'state.py', 'symbol': 'select', 'purpose': 'Choose a candidate',
                                'constraints': ['Reuse the missing permission for all modes'], 'depends_on': []}]}
        self.watch = watch_config(self.repo, self.instance, self.initial)
        self.packet = {'grade': {'tests_status': {'FAIL_TO_PASS': {'failure': ['build'], 'success': []},
                                                'PASS_TO_PASS': {'success': ['existing'], 'failure': []}}}}
        self.memory = attach_observations(seed_featurespace(self.instance, self.plan),
                                           evidence_bundle(self.packet, 'E   AssertionError: invalid candidate'), 'inputhash', 1)

    def tearDown(self):
        self.temp.cleanup()

    def delta(self, memory, evidence_ids, reviews=None):
        eid = next(iter(memory['elements']))
        cid = next(iter(memory['conditions']))
        value = {'summary': 'Permission and candidate validity require separate conditions.',
                 'upsert': {'conditions': [{'id': 'C:separate', 'text': 'Validate each candidate in its own configuration.',
                                           'applies_to': [eid], 'evidence': evidence_ids}]},
                 'retire': [{'id': cid, 'reason': 'Current execution contradicts shared permission.', 'evidence': evidence_ids}],
                 'focus': [eid],
                 'checks': [{'hypothesis': 'Selection reuses one permission state for every candidate.',
                             'expected_change': 'Each candidate is validated in its own configuration.',
                             'evidence': evidence_ids, 'basis': 'Shared permission guard in current source.'}]}
        if reviews is not None:
            value['check_reviews'] = reviews
        return value

    def packet_with_select_flow(self):
        packet = copy.deepcopy(self.packet)
        packet['repo_state_relations'] = {'causal_flow': [{
            'stage': 'input_to_state', 'status': 'broken', 'test': 'build',
            'actors': [{'file': 'state.py', 'symbol': 'select',
                        'role': 'policy_state_producer'}],
            'relation': 'The requested state is not represented.',
            'required_invariant': 'The request must produce state consumed by select.'
        }]}
        return packet

    def model(self, prompt, **kwargs):
        self.assertNotIn('GOLD_POISON', prompt)
        self.assertNotIn('TEST_PATCH_POISON', prompt)
        self.assertLessEqual(len(prompt), 80000)
        if 'Do not return code edits' in prompt:
            view = section(prompt, 'featurespace')
            memory = {k: {e['id']: e for e in view[k]} for k in ['elements', 'conditions']}
            evidence = section(prompt, 'current_execution_evidence')['observations'][0]['id']
            return response(self.delta(memory, [evidence]))
        view = section(prompt, 'featurespace')
        self.assertTrue(any(c['status'] == 'retired' for c in view['conditions']))
        self.assertTrue(any(c['id'] == 'C:separate' for c in view['conditions']))
        self.assertNotIn('<failure_output>', prompt)
        self.assertNotIn('<runtime_state_packet>', prompt)
        return response({'edits': [{'file_path': 'state.py', 'before': '    return "Invalid"', 'after': '    return "Build"'}]})

    def test_two_stage_evolution_retires_wrong_hypothesis_and_counts_both_calls(self):
        directory = self.root/'round1'
        with patch('generator.llm_interface.call_llm_with_usage', side_effect=self.model) as llm:
            result = repair_once(self.instance, self.repo, self.initial, self.packet, 'E   AssertionError: invalid',
                                 self.watch, directory, plan=self.plan)
        self.assertEqual(llm.call_count, 2)
        self.assertIn('+    return "Build"', result)
        memory = json.loads((directory/'featurespace_after.json').read_text())
        self.assertEqual(memory['revision'], 1)
        self.assertEqual(memory['conditions']['C:separate']['certainty'], 'hypothesis')
        usage = json.loads((directory/'llm_usage.json').read_text())
        self.assertEqual(usage['logical_calls'], 2)
        self.assertEqual(usage['reported_total_tokens'], 60)
        self.assertIn('Missing', (self.repo/'state.py').read_text())

    def test_omitted_diagnosed_responsibility_gets_one_in_round_retry(self):
        eid = next(iter(self.memory['elements']))
        calls = []
        def model(prompt, **kwargs):
            calls.append(prompt)
            if 'Do not return code edits' in prompt:
                evidence = section(prompt, 'current_execution_evidence')['observations'][0]['id']
                return response({'summary': 'select still consumes the wrong state.',
                                 'focus': [eid], 'checks': [{
                                     'hypothesis': 'select returns the invalid state.',
                                     'expected_change': 'select returns the buildable state.',
                                     'evidence': [evidence], 'basis': 'Current function output.'}]})
            replacement = ('def select(node):\n    return "Build"'
                           if 'Previous repair was rejected' in prompt
                           else 'def select(node):\n    return "Invalid"')
            return response({'edits': [{'file_path': 'state.py', 'symbol': 'select',
                                        'replacement': replacement}]})
        result = repair_once(self.instance, self.repo, self.initial,
                             self.packet_with_select_flow(), 'failed',
                             self.watch, self.root/'retry', call=model, memory=self.memory)
        self.assertEqual(len(calls), 3)
        self.assertIn('+    return "Build"', result)
        self.assertTrue((self.root/'retry/repair_retry_response.json').is_file())

    def test_required_edit_checklist_is_adjacent_to_output_contract(self):
        packet = self.packet_with_select_flow()
        memory = attach_observations(
            self.memory, evidence_bundle(packet, 'failed'),
            self.memory['current_patch_sha256'], 2)
        prompt = repair_prompt(
            self.instance, self.repo, self.initial, packet, 'failed',
            self.watch, memory=memory)
        checklist = section(prompt, 'required_edit_checklist')
        self.assertEqual(checklist, [{'file_path': 'state.py', 'symbol': 'select'}])
        tail = prompt[prompt.index('<required_edit_checklist>'):]
        self.assertIn('real executable change for EVERY exact', tail)
        self.assertIn('Return only JSON', tail)
        self.assertLess(tail.index('Return only JSON') - tail.index('select'), 1000)

    def test_second_omission_keeps_prior_edits_and_gets_focused_completion(self):
        eid = next(iter(self.memory['elements']))
        calls = []
        def model(prompt, **kwargs):
            calls.append(prompt)
            if 'Do not return code edits' in prompt:
                evidence = section(prompt, 'current_execution_evidence')['observations'][0]['id']
                return response({'summary': 'select still consumes the wrong state.',
                                 'focus': [eid], 'checks': [{
                                     'hypothesis': 'select returns the invalid state.',
                                     'expected_change': 'select returns the buildable state.',
                                     'evidence': [evidence], 'basis': 'Current function output.'}]})
            replacement = ('def select(node):\n    return "Build"'
                           if 'Only complete these missing responsibilities' in prompt
                           else 'def select(node):\n    return "Invalid"')
            return response({'edits': [{'file_path': 'state.py', 'symbol': 'select',
                                        'replacement': replacement}]})
        result = repair_once(
            self.instance, self.repo, self.initial,
            self.packet_with_select_flow(), 'failed',
            self.watch, self.root/'completion', call=model, memory=self.memory)
        self.assertEqual(len(calls), 4)
        self.assertIn('+    return "Build"', result)
        self.assertTrue((self.root/'completion/repair_completion_response.json').is_file())
        completion = calls[-1][calls[-1].index('<required_edit_checklist>'):]
        self.assertIn('Only complete these missing responsibilities', completion)

    def test_invalid_reflection_does_not_block_code_generation(self):
        replies = [response({'broken': 'no valid delta'}), response({'edits': [
            {'file_path': 'state.py', 'before': '    return "Invalid"', 'after': '    return "Build"'}]})]
        directory = self.root/'bad_diagnosis'
        with patch('generator.llm_interface.call_llm_with_usage', side_effect=replies) as llm:
            result = repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', self.watch, directory)
        self.assertEqual(llm.call_count, 2)
        self.assertIn('+    return "Build"', result)
        self.assertTrue((directory/'diagnosis_error.json').is_file())
        self.assertEqual(json.loads((directory/'featurespace_after.json').read_text())['revision'], 0)

    def test_resume_replays_completed_model_stages_without_new_calls(self):
        directory = self.root/'resumed'
        first = repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', self.watch, directory,
                            call=self.model, plan=self.plan)
        (directory/'combined.patch').unlink()  # interruption after response, before publishing candidate
        with patch('generator.llm_interface.call_llm_with_usage') as llm:
            second = repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', self.watch, directory, plan=self.plan)
        llm.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(json.loads((directory/'llm_usage.json').read_text())['logical_calls'], 2)

    def test_unknown_evidence_rejected_atomically(self):
        before = copy.deepcopy(self.memory)
        delta = self.delta(self.memory, ['ev:invented'])
        with self.assertRaisesRegex(ValueError, 'unavailable evidence'):
            merge_update(self.memory, delta, ['state.py'])
        self.assertEqual(self.memory, before)

    def test_new_element_must_have_current_source_evidence(self):
        delta = {'summary': 'unknown source', 'upsert': {'elements': [{'id': 'E:outside', 'file': '../secret.py',
                   'symbol': 'private', 'text': 'unavailable', 'evidence': ['requirement']}]}}
        with self.assertRaises(ValueError):
            merge_update(self.memory, delta, ['state.py'])

    def test_historical_evidence_is_citable_but_flagged_as_stale(self):
        first = evidence_bundle({'grade': {'tests_status': {'FAIL_TO_PASS': {'failure': ['build']}}}},
                                'E   AssertionError: first candidate failure')
        memory = attach_observations(seed_featurespace(self.instance, self.plan), first, 'patch_one', 1)
        old_id = first['observations'][0]['id']
        second = evidence_bundle({'grade': {'tests_status': {'FAIL_TO_PASS': {'failure': ['build']}}}},
                                 'E   AssertionError: second candidate failure')
        memory = attach_observations(memory, second, 'patch_two', 2)
        current_id = memory['current_evidence'][0]
        self.assertIn(old_id, memory['evidence'])
        self.assertNotIn(old_id, memory['current_evidence'])
        eid = next(iter(memory['elements']))
        delta = {'summary': 'The same break appeared in the earlier candidate.',
                 'upsert': {'conditions': [{'id': 'C:history', 'text': 'The break predates this round.',
                                           'applies_to': [eid], 'evidence': [old_id, current_id]}]},
                 'focus': [eid],
                 'checks': [{'hypothesis': 'Entry dispatch depends on a stale state.',
                             'expected_change': 'The entry guard consumes the updated state.',
                             'evidence': [current_id]}]}
        merged = merge_update(memory, delta, ['state.py'])
        entry = merged['conditions']['C:history']
        self.assertEqual(entry['cites_historical_evidence'], [old_id])
        # Stored observations stay patch- and round-scoped for the reader.
        self.assertEqual(merged['evidence'][old_id]['patch_sha256'], 'patch_one')
        self.assertEqual(merged['evidence'][old_id]['round'], 1)

    def test_diagnosis_accepts_evidence_ids_visible_on_featurespace_entries(self):
        first = evidence_bundle({'failure_routes': [{'test': 'old', 'target_calls': []}]}, 'old failure')
        memory = attach_observations(seed_featurespace(self.instance, self.plan), first, 'old_patch', 1)
        old_id = first['observations'][0]['id']
        condition = next(iter(memory['conditions'].values()))
        condition['evidence'] = [old_id]
        second = evidence_bundle(self.packet, 'new failure')
        memory = attach_observations(memory, second, 'new_patch', 2)
        diagnosis_prompt(self.instance, self.repo, '', self.packet, self.watch, memory, second)
        self.assertIn(old_id, memory['diagnosis_evidence_ids'])

    def test_change_without_verifiable_check_is_rejected(self):
        evidence_id = self.memory['current_evidence'][0]
        eid = next(iter(self.memory['elements']))
        delta = {'summary': 'no verifiable expectation',
                 'upsert': {'conditions': [{'id': 'C:blind', 'text': 'Change without expectation.',
                                           'applies_to': [eid], 'evidence': [evidence_id]}]},
                 'checks': []}
        before = copy.deepcopy(self.memory)
        with self.assertRaisesRegex(ValueError, 'need a check'):
            merge_update(self.memory, delta, ['state.py'])
        self.assertEqual(self.memory, before)

    def test_check_reviews_are_validated_and_stored(self):
        evidence_id = self.memory['current_evidence'][0]
        reviews = [{'check': 'Each candidate is validated in its own configuration.',
                    'observed': 'expected_change_not_observed', 'evidence': [evidence_id],
                    'conclusion': 'The repair kept the shared guard; revise the entry condition.'}]
        merged = merge_update(self.memory, self.delta(self.memory, [evidence_id], reviews=reviews), ['state.py'])
        self.assertEqual(merged['check_reviews'][0]['observed'], 'expected_change_not_observed')
        self.assertEqual(merged['checks'][0]['patch_sha256'], merged['current_patch_sha256'])
        bad = [dict(reviews[0], observed='kind_of')]
        with self.assertRaisesRegex(ValueError, 'supported observed outcome'):
            merge_update(self.memory, self.delta(self.memory, [evidence_id], reviews=bad), ['state.py'])

    def test_duplicate_method_repair_is_rejected_before_evaluation(self):
        source = 'class Mode:\n    def allowed(self):\n        return False\n'
        (self.repo/'mode.py').write_text(source)
        subprocess.run(['git', 'add', '.'], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'mode'],
                       cwd=self.repo, check=True, capture_output=True)
        watch = {'watch': {'state.py': ['select'], 'mode.py': ['allowed']}}
        reply = response({'edits': [{'file_path': 'mode.py', 'before': source,
                                    'after': source + '\n    def allowed(self):\n        return False\n'}]})
        with self.assertRaisesRegex(ValueError, 'duplicate method'):
            repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', watch, self.root/'duplicate',
                        call=lambda *a, **k: reply, mode='basic')
        self.assertFalse((self.root/'duplicate/combined.patch').exists())
        self.assertEqual((self.repo/'mode.py').read_text(), source)

    def test_same_name_method_override_with_new_body_is_rejected(self):
        # Round 1 of the Conan run slipped through the old identical-body check:
        # a redefinition with a different body still shadows the original method.
        source = 'class Mode:\n    def allowed(self):\n        return False\n'
        (self.repo/'mode.py').write_text(source)
        subprocess.run(['git', 'add', '.'], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'mode'],
                       cwd=self.repo, check=True, capture_output=True)
        watch = {'watch': {'state.py': ['select'], 'mode.py': ['allowed']}}
        override = response({'edits': [{'file_path': 'mode.py', 'before': source,
                                        'after': source + '\n    def allowed(self):\n        return True\n'}]})
        with self.assertRaisesRegex(ValueError, 'duplicate method'):
            repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', watch, self.root/'override',
                        call=lambda *a, **k: override, mode='basic')
        self.assertFalse((self.root/'override/combined.patch').exists())
        self.assertEqual((self.repo/'mode.py').read_text(), source)

    def test_comment_only_repair_is_rejected(self):
        reply = response({'edits': [{'file_path': 'state.py', 'before': '    return "Invalid"',
                                    'after': '    return "Invalid"  # still invalid'}]})
        with self.assertRaisesRegex(ValueError, 'same AST'):
            repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', self.watch, self.root/'noop',
                        call=lambda *a, **k: reply, mode='basic')

    def test_unchanged_function_echo_does_not_cover_a_repair_obligation(self):
        current = self.root/'responsibility'; current.mkdir()
        (current/'state.py').write_text(
            'def select():\n    return "Invalid"\n\ndef refresh():\n    return False\n')
        edits = [
            {'file_path': 'state.py', 'symbol': 'select',
             'before': 'def select():\n    return "Invalid"',
             'after': 'def select():\n    return "Invalid"'},
            {'file_path': 'state.py', 'symbol': 'refresh',
             'before': 'def refresh():\n    return False',
             'after': 'def refresh():\n    return True'}]
        covered = _edited_responsibilities(edits, current)
        self.assertNotIn(('state.py', 'select'), covered)
        self.assertIn(('state.py', 'refresh'), covered)

    def test_same_named_methods_in_independent_classes_are_allowed(self):
        def factory(name):
            return '\ndef '+name+'():\n    class Mode:\n        def allowed(self):\n            return False\n    return Mode\n'
        code = 'def select(node):\n    return "Missing"\n' + factory('one') + factory('two')
        (self.repo/'state.py').write_text(code)
        reply = response({'edits': [{'file_path': 'state.py', 'before': factory('two'),
                                    'after': factory('two') + factory('three')}]})
        result = repair_once(self.instance, self.repo, '', self.packet, 'failed', self.watch, self.root/'factories',
                             call=lambda *a, **k: reply, mode='basic')
        self.assertIn('+def three():', result)

    def test_return_to_previous_candidate_is_rejected(self):
        reply = response({'edits': [{'file_path': 'state.py', 'before': '    return "Invalid"', 'after': '    return "Build"'}]})
        first = repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', self.watch, self.root/'first',
                            call=lambda *a, **k: reply, mode='basic')
        before = json.loads((self.root/'first/candidate_before.json').read_text())
        after = json.loads((self.root/'first/candidate_after.json').read_text())
        history = [{'behavior_before': before['behavior_sha256'], 'behavior_after': after['behavior_sha256'],
                    'outcome': 'no_test_improvement'}]
        reverse = response({'edits': [{'file_path': 'state.py', 'before': '    return "Build"', 'after': '    return "Invalid"'}]})
        with self.assertRaisesRegex(ValueError, 'cycle'):
            repair_once(self.instance, self.repo, first, self.packet, 'failed', self.watch, self.root/'second',
                        call=lambda *a, **k: reverse, history=history, mode='basic')


if __name__ == '__main__':
    unittest.main()
