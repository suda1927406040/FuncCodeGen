"""Offline controller integration: fixed initial candidates, two repairs, resume."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from generator.llm_interface import LLMResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'scripts'))
import run_state_feedback as controller
import state_feedback_transport as transport


class FeedbackControllerTests(unittest.TestCase):
    def test_fixed_initial_candidates_evolution_outcomes_resume_and_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root/'repos/case-0'; repo.mkdir(parents=True)
            (repo/'state.py').write_text('def select():\n    return "Missing"\n')
            for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base')]:
                subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)
            base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
            rows = [{'instance_id': 'case-'+str(i), 'repo': 'sample/repo', 'base_commit': base,
                     'problem_statement': 'Select a buildable candidate.',
                     'new_components': [{'file': 'state.py', 'components': [{'name': 'select'}]}]} for i in range(200)]
            inputs = root/'inputs.json'; inputs.write_text(json.dumps(rows))
            old = root/'old'; old.mkdir(); (old/'generation_inputs.json').write_text(json.dumps(rows))
            initial = old/'initial/outputs/case-0'; initial.mkdir(parents=True)
            initial_patch = ('diff --git a/state.py b/state.py\n--- a/state.py\n+++ b/state.py\n@@ -1,2 +1,2 @@\n'
                             ' def select():\n-    return "Missing"\n+    return "Invalid"\n')
            (initial/'generated.patch').write_text(initial_patch)
            (initial/'result.json').write_text(json.dumps({'instance_id': 'case-0', 'success': True}))
            (initial/'worker_complete.json').write_text('{"exit_code":0}')
            (initial/'llm_usage.json').write_text('{"logical_calls":2,"provider_attempts":2,"reported_total_tokens":100}')
            (initial/'plan.json').write_text(json.dumps({'steps': [{'id': 'S1', 'file': 'state.py', 'symbol': 'select',
                'purpose': 'select', 'constraints': ['Original assumption'], 'depends_on': []}]}))
            args = SimpleNamespace(run_root=root/'new', inputs=inputs, prepared_repos=root/'repos',
                credentials_file=root/'unused-credentials', host='unused', bundle=str(root/'bundle'), workers=1,
                repair_rounds=2, worker=None, resume=False, same_host=True, proxy=None, feedback_mode='featurespace',
                reuse_initial_from=old, instance_id=['case-0'])
            grades = [
                {'resolved': False, 'patch_successfully_applied': True, 'tests_status': {'FAIL_TO_PASS': {'failure': ['a', 'b'], 'success': []}, 'PASS_TO_PASS': {'failure': [], 'success': ['old']}}},
                {'resolved': False, 'patch_successfully_applied': True, 'tests_status': {'FAIL_TO_PASS': {'failure': ['b'], 'success': ['a']}, 'PASS_TO_PASS': {'failure': [], 'success': ['old']}}},
                {'resolved': True, 'patch_successfully_applied': True, 'tests_status': {'FAIL_TO_PASS': {'failure': [], 'success': ['a', 'b']}, 'PASS_TO_PASS': {'failure': [], 'success': ['old']}}},
            ]
            stages = []
            def offline_stage(args, source, mode, number, ids, predictions, write_json):
                stages.append((mode, number))
                self.assertEqual(ids, ['case-0'])
                directory = root/'evaluations'/str(number)/mode
                instance = directory/'case-0'; instance.mkdir(parents=True, exist_ok=True)
                predictions(directory/'predictions.jsonl', ids)
                if mode == 'evaluate':
                    (instance/'report.json').write_text(json.dumps(grades[number]))
                    (instance/'test_output.txt').write_text('==== FAILURES ====\nE   AssertionError: wrong selected state\n')
                else:
                    (instance/'trace_metadata.json').write_text('{"fresh_initial_container":true,"baseline_verified":true,"trace_status":"no_events"}')
                    (instance/'state_trace.jsonl').write_text('')
                return directory
            prompts = []
            def llm(prompt, **kwargs):
                prompts.append(prompt)
                if 'Do not return code edits' in prompt:
                    view = json.loads(prompt.split('<featurespace>\n')[1].split('\n</featurespace>')[0])
                    eid = view['elements'][0]['id']
                    if view['attempts']:
                        self.assertEqual(view['attempts'][-1]['test_delta']['fixed'], ['a'])
                        self.assertEqual(view['attempts'][-1]['outcome'], 'improved')
                    payload = {'summary': 'Separate producer state from consumer permission.',
                               'upsert': {'conditions': [{'id': 'C:validated', 'text': 'Validate before selection.',
                                          'applies_to': [eid], 'evidence': ['code:state.py']}]},
                               'focus': [eid],
                               'checks': [{'hypothesis': 'Selection consumes an unvalidated state.',
                                           'expected_change': 'The invalid state branch stops being selected.',
                                           'evidence': ['code:state.py'], 'basis': 'Current selection path.'}]}
                else:
                    partial = '    return "Partial"' in prompt
                    payload = {'edits': [{'file_path': 'state.py', 'before': '    return "Partial"' if partial else '    return "Invalid"',
                                          'after': '    return "Build"' if partial else '    return "Partial"'}]}
                return LLMResponse(json.dumps(payload), 'openai', 'stub', 'stop', 10, 20, 30)
            with patch.object(controller, 'parse_args', return_value=args), patch.object(controller, 'credentials'), \
                 patch.object(transport, 'same_host_stage', side_effect=offline_stage), \
                 patch('generator.llm_interface.call_llm_with_usage', side_effect=llm), contextlib.redirect_stdout(io.StringIO()):
                controller.main()
            self.assertEqual(len(prompts), 4)
            self.assertEqual(stages, [('evaluate', 0), ('trace', 0), ('evaluate', 1), ('trace', 1), ('evaluate', 2)])
            history = json.loads((args.run_root/'repair_history.json').read_text())['case-0']
            self.assertEqual(history[0]['test_delta']['fixed'], ['a'])
            self.assertEqual(history[1]['outcome'], 'resolved')
            memory = json.loads((args.run_root/'featurespaces/case-0.json').read_text())
            self.assertEqual(memory['revision'], 2)
            self.assertEqual(len(memory['attempts']), 2)
            summary = json.loads((args.run_root/'summary.json').read_text())
            self.assertEqual(summary['new_run_logical_calls'], 4)
            self.assertEqual(summary['logical_calls'], 6)
            self.assertEqual(summary['new_run_reported_tokens'], 120)
            self.assertEqual(summary['execution_counts'], {'evaluation_cases': 3, 'diagnostic_cases': 2})
            self.assertEqual((initial/'generated.patch').read_text(), initial_patch)
            self.assertIn('Missing', (repo/'state.py').read_text())
            final = (args.run_root/'predictions_final.jsonl').read_text()
            args.resume = True
            with patch.object(controller, 'parse_args', return_value=args), patch.object(controller, 'credentials'), \
                 patch.object(transport, 'same_host_stage', side_effect=offline_stage), \
                 patch('generator.llm_interface.call_llm_with_usage') as calls, contextlib.redirect_stdout(io.StringIO()):
                controller.main()
            calls.assert_not_called()
            self.assertEqual((args.run_root/'predictions_final.jsonl').read_text(), final)
            self.assertEqual(json.loads((args.run_root/'repair_history.json').read_text())['case-0'], history)


if __name__ == '__main__':
    unittest.main()
