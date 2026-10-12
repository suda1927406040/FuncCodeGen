import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from generator.feedback_evidence import evidence_bundle, failure_excerpt, read_failure_output, test_delta
from generator.feature_memory import attach_observations, merge_update, seed_featurespace, memory_view
from generator.llm_interface import LLMResponse
from generator.state_feedback import repair_once, watch_config


def response(value):
    return LLMResponse(json.dumps(value), 'openai', 'test', 'stop', 10, 20, 30)


def section(prompt, name):
    return json.loads(prompt.split('<' + name + '>\n', 1)[1].split('\n</' + name + '>', 1)[0])


class EvidenceTests(unittest.TestCase):
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

    def delta(self, memory, evidence_ids):
        eid = next(iter(memory['elements']))
        cid = next(iter(memory['conditions']))
        return {'summary': 'Permission and candidate validity require separate conditions.',
                'upsert': {'conditions': [{'id': 'C:separate', 'text': 'Validate each candidate in its own configuration.',
                                          'applies_to': [eid], 'evidence': evidence_ids}]},
                'retire': [{'id': cid, 'reason': 'Current execution contradicts shared permission.', 'evidence': evidence_ids}],
                'focus': [eid], 'checks': [{'description': 'Check selected candidate and retained modes.', 'evidence': evidence_ids}]}

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

    def test_comment_only_repair_is_rejected(self):
        reply = response({'edits': [{'file_path': 'state.py', 'before': '    return "Invalid"',
                                    'after': '    return "Invalid"  # still invalid'}]})
        with self.assertRaisesRegex(ValueError, 'same AST'):
            repair_once(self.instance, self.repo, self.initial, self.packet, 'failed', self.watch, self.root/'noop',
                        call=lambda *a, **k: reply, mode='basic')

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
