import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from explorer.context_builder import ContextBuilder
from generator.llm_interface import LLMCallError, LLMResponse
from generator.planned_generation import PlanContext, PlanError
from run_oracle import OUTPUT_FORMAT_INSTRUCTION, run_single


def edit(file, before, after):
    return ('<edit>\n[start of the snippet before editing in %s]\n%s\n'
            '[end of the snippet before editing in %s]\n'
            '[start of the snippet after editing in %s]\n%s\n'
            '[end of the snippet after editing in %s]\n</edit>') % (file, before, file, file, after, file)


class PlannedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        self.output = Path(self.temp.name) / 'out'
        self.write('engine/select.py', 'def select_profile(node):\n    return node.profile\n')
        self.write('model/state.py', 'def initialize_hidden(node):\n    node.profile = 1\n')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD')
        self.instance = {'instance_id': 'case', 'repo': 'test/repo', 'base_commit': self.base,
                         'problem_statement': '<request>Support compatible profile selection</request>',
                         'new_components': [{'file': 'engine/select.py', 'components': [{'name': 'select_profile', 'type': 'function'}]}],
                         'patch': 'GOLD_PATCH_POISON', 'test_patch': 'HIDDEN_TEST_POISON'}

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True).strip()

    def write(self, file, source):
        path = self.root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    def context(self):
        builder = ContextBuilder(self.root, history_mode='off')
        ctx = builder.build_context('case', self.instance['problem_statement'], [],
                                    new_components_raw=self.instance['new_components'], base_commit=self.instance['base_commit'])
        return PlanContext(self.root, ctx, builder.repository_index)

    def plan(self):
        return {'contracts': ['Support compatible selection while preserving normal profiles'],
                'steps': [{'id': 'S1', 'file': 'engine/select.py', 'symbol': 'select_profile', 'action': 'modify',
                           'purpose': 'Extend profile selection', 'depends_on': [], 'constraints': ['Keep normal profiles'],
                           'evidence': [{'source': 'current', 'file': 'engine/select.py', 'symbol': 'select_profile'}]}],
                'missing_context': [{'file': '', 'symbol': 'initialize_hidden', 'reason': 'Inspect the state producer'}]}

    def reply(self, text, tokens=10, finish='stop'):
        return LLMResponse(text, 'openai', 'test-model', finish, 100, tokens, 100 + tokens,
                           {'prompt_tokens': 100, 'completion_tokens': tokens}, seconds=0.1)

    def test_two_calls_supplement_unseen_source_and_export_one_patch(self):
        planned = json.dumps(self.plan())
        generated = edit('engine/select.py', 'def select_profile(node):\n    return node.profile',
                         'def select_profile(node):\n    return node.profile if node.profile is not None else 0')
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(planned), self.reply(generated)]) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertTrue(result['success'])
        self.assertEqual(llm.call_count, 2)
        planning_prompt, code_prompt = [call.args[0] for call in llm.call_args_list]
        self.assertNotIn('[start of model/state.py]', planning_prompt)
        self.assertIn('[start of model/state.py]', code_prompt)
        self.assertIn('def initialize_hidden(node):', code_prompt)
        self.assertNotIn('MUST add an import', code_prompt)
        for prompt in [planning_prompt, code_prompt]:
            self.assertNotIn('GOLD_PATCH_POISON', prompt)
            self.assertNotIn('HIDDEN_TEST_POISON', prompt)
            self.assertLessEqual(len(prompt), 80000)
        out = self.output / 'case'
        self.assertTrue((out / 'generated.patch').is_file())
        self.assertEqual(result['pipeline_attempts'], 1)
        self.assertEqual(result['submitted_patch_count'], 1)
        self.assertEqual(result['llm_usage']['logical_calls'], 2)
        self.assertEqual(result['llm_usage']['total_tokens'], 220)
        self.assertEqual((self.root / 'engine/select.py').read_text(), 'def select_profile(node):\n    return node.profile\n')
        self.assertTrue((out / 'plan_validation.json').is_file())

    def test_invalid_plan_falls_back_without_reusing_previous_patch(self):
        out = self.output / 'case'
        out.mkdir(parents=True)
        (out / 'generated.patch').write_text('PREVIOUS_SUCCESS')
        generated = edit('engine/select.py', 'return node.profile', 'return 2')
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply('{"steps":['), self.reply(generated)]) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertTrue(result['success'])
        self.assertEqual(llm.call_count, 2)
        self.assertTrue(result['planning_fallback'])
        self.assertNotIn('PREVIOUS_SUCCESS', (out / 'generated.patch').read_text())
        self.assertEqual(result['submitted_patch_count'], 1)
        self.assertFalse(json.loads((out / 'plan_validation.json').read_text())['valid'])

    def test_truncated_plan_uses_direct_codegen(self):
        generated = edit('engine/select.py', 'return node.profile', 'return 2')
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(json.dumps(self.plan()), finish='length'), self.reply(generated)]) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertTrue(result['success'])
        self.assertEqual(llm.call_count, 2)
        self.assertTrue(result['planning_fallback'])

    def test_single_mode_still_makes_one_code_call(self):
        generated = edit('engine/select.py', 'return node.profile', 'return 2')
        with patch('run_oracle.call_llm_with_usage', return_value=self.reply(generated)) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, generation_mode='single', history_mode='off')
        self.assertTrue(result['success'])
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(result['llm_calls'], 1)

    def test_recall_only_previews_planning_without_model_calls(self):
        with patch('run_oracle.call_llm_with_usage') as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, recall_only=True, history_mode='off')
        self.assertTrue(result['success'])
        llm.assert_not_called()
        self.assertTrue((self.output / 'case/planning_prompt.md').is_file())
        self.assertEqual(result['llm_calls'], 0)

    def test_invalid_patch_keeps_usage_and_does_not_add_repair_call(self):
        generated = edit('engine/select.py', 'return node.profile', 'return (')
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(json.dumps(self.plan())), self.reply(generated)]) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off', max_repair_attempts=0)
        self.assertFalse(result['success'])
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result['llm_usage']['total_tokens'], 220)
        self.assertFalse((self.output / 'case/generated.patch').exists())

    def test_generation_api_failure_keeps_both_stage_metrics(self):
        error = LLMCallError('openai', 'test-model', 3, 0.2, RuntimeError('server'))
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(json.dumps(self.plan())), error]) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertFalse(result['success'])
        self.assertEqual(result['stage'], 'generation')
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result['llm_usage']['provider_attempts'], 4)
        self.assertIsNone(result['llm_usage']['total_tokens'])
        self.assertEqual(result['llm_usage']['reported_total_tokens'], 110)

    def test_cycles_unknown_symbols_and_escaping_paths_are_rejected(self):
        context = self.context()
        for mutate in ['cycle', 'symbol', 'path']:
            plan = self.plan()
            if mutate == 'cycle': plan['steps'][0]['depends_on'] = ['S1']
            if mutate == 'symbol': plan['steps'][0]['symbol'] = 'made_up_function'
            if mutate == 'path': plan['steps'][0]['file'] = '../outside.py'
            with self.assertRaises(PlanError, msg=mutate):
                context.parse_validate(json.dumps(plan))

    def test_optional_missing_context_warning_is_explicit(self):
        context = self.context()
        plan = self.plan()
        plan['missing_context'].append({'file': '', 'symbol': 'does_not_exist', 'reason': 'guess'})
        checked, report = context.parse_validate(json.dumps(plan))
        self.assertTrue(report['warnings'])
        self.assertEqual(checked['context'][0]['file'], 'model/state.py')

    def test_fully_qualified_module_reference_maps_to_current_symbol(self):
        self.assertEqual(self.context().resolve('engine/select.py', 'engine.select.select_profile'),
                         ('engine/select.py', ['select_profile']))

    def test_unprovided_history_evidence_is_rejected(self):
        plan = self.plan()
        plan['steps'][0]['evidence'] = [{'source': 'history', 'file': 'engine/select.py', 'commit': 'f' * 40}]
        with self.assertRaises(PlanError):
            self.context().parse_validate(json.dumps(plan))

    def test_inherited_method_and_common_fields_resolve_current_owners(self):
        self.write('engine/base.py', 'class Parent:\n    def __init__(self):\n        self.value = 0\n'
                   '    def normalize(self, value):\n        return value\n')
        self.write('engine/child.py', 'from engine.base import Parent\nclass Child(Parent):\n'
                   '    slots = ("data",)\n    def __init__(self):\n        self.data = 1\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'classes')
        self.instance['base_commit'] = self.git('rev-parse', 'HEAD')
        context = self.context()
        self.assertEqual(context.resolve('engine/child.py', 'Child.normalize'), ('engine/base.py', ['Parent.normalize']))
        self.assertEqual(context.resolve('engine/child.py', 'Child.data'), ('engine/child.py', ['Child.__init__']))
        self.assertEqual(context.resolve('engine/child.py', 'Child.slots'), ('engine/child.py', ['Child.slots']))
        self.assertEqual(context.resolve('engine/child.py', 'Child.value'), ('engine/base.py', ['Parent.__init__']))

    def test_unfittable_required_context_uses_direct_codegen(self):
        self.write('model/state.py', 'def initialize_hidden(node):\n' + '    node.profile = 1\n' * 5000)
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'oversized')
        self.instance['base_commit'] = self.git('rev-parse', 'HEAD')
        generated = edit('engine/select.py', 'return node.profile', 'return 2')
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(json.dumps(self.plan())), self.reply(generated)]) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertTrue(result['success'])
        self.assertTrue(result['planning_fallback'])
        self.assertEqual(llm.call_count, 2)
        self.assertTrue((self.output / 'case/generated.patch').exists())

    def test_invalid_code_is_repaired_from_base_and_only_final_patch_is_exported(self):
        generated = edit('engine/select.py', 'return node.profile', 'return (')
        repaired = json.dumps({'edits': [{'file_path': 'engine/select.py', 'before': 'return node.profile', 'after': 'return 2'}]})
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(json.dumps(self.plan())), self.reply(generated), self.reply(repaired)]) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertTrue(result['success'])
        self.assertEqual(llm.call_count, 3)
        self.assertEqual(result['repair_attempts'], 1)
        self.assertIn('def select_profile(node):', llm.call_args_list[-1].args[0])
        self.assertNotIn('GOLD_PATCH_POISON', llm.call_args_list[-1].args[0])
        self.assertTrue(llm.call_args_list[-1].kwargs['json_mode'])
        self.assertEqual(result['submitted_patch_count'], 1)
        self.assertEqual(result['llm_usage']['total_tokens'], 330)

    def test_truncated_code_does_not_submit_a_partial_candidate(self):
        generated = edit('engine/select.py', 'return node.profile', 'return 2')
        repaired = json.dumps({'edits': [{'file_path': 'engine/select.py', 'before': 'return node.profile', 'after': 'return 3'}]})
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(json.dumps(self.plan())), self.reply(generated, finish='length'), self.reply(repaired)]):
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertTrue(result['success'])
        self.assertIn('+    return 3', (self.output / 'case/generated.patch').read_text())
        self.assertEqual(result['repair_attempts'], 1)

    def test_repair_limit_keeps_usage_and_never_submits_invalid_code(self):
        generated = edit('engine/select.py', 'return node.profile', 'return (')
        with patch('run_oracle.call_llm_with_usage', side_effect=[self.reply(json.dumps(self.plan()))] + [self.reply(generated)] * 3) as llm:
            result = run_single(self.instance, self.output, repo_dir_override=self.root, history_mode='off')
        self.assertFalse(result['success'])
        self.assertEqual(llm.call_count, 4)
        self.assertEqual(result['repair_attempts'], 2)
        self.assertEqual(result['submitted_patch_count'], 0)
        self.assertFalse((self.output / 'case/generated.patch').exists())

    def test_new_file_can_be_defined_in_dependency_order(self):
        self.instance['new_components'] = [{'file': 'engine/new.py', 'components': [{'name': 'feature', 'type': 'function'}]}]
        plan = self.plan()
        plan['steps'] = [
            {'id': 'S1', 'file': 'engine/new.py', 'symbol': '', 'action': 'create', 'purpose': 'Create module', 'depends_on': [],
             'evidence': [{'source': 'requirement', 'file': 'engine/new.py', 'symbol': 'feature'}]},
            {'id': 'S2', 'file': 'engine/new.py', 'symbol': 'feature', 'action': 'create_symbol', 'purpose': 'Define feature', 'depends_on': ['S1'],
             'evidence': [{'source': 'requirement', 'file': 'engine/new.py', 'symbol': 'feature'}]},
        ]
        checked, _ = self.context().parse_validate(json.dumps(plan))
        self.assertEqual([s['id'] for s in checked['steps']], ['S1', 'S2'])
        plan['steps'][1]['depends_on'] = []
        with self.assertRaises(PlanError):
            self.context().parse_validate(json.dumps(plan))

    def test_module_bindings_are_existing_symbols_and_rendered(self):
        self.write('engine/select.py', 'LIMITS = {"normal": 1}\n__all__ = ["select_profile"]\n'
                   'def select_profile(node):\n    return node.profile\n')
        context = self.context()
        self.assertEqual(context.resolve('engine/select.py', 'LIMITS'), ('engine/select.py', ['LIMITS']))
        self.assertEqual(context.resolve('engine/select.py', '__all__'), ('engine/select.py', ['__all__']))
        plan = self.plan()
        plan['steps'][0]['symbol'] = 'LIMITS'
        plan['steps'][0]['evidence'][0]['symbol'] = 'LIMITS'
        checked, _ = context.parse_validate(json.dumps(plan))
        _, recall = context.generation_prompt(self.instance, checked, OUTPUT_FORMAT_INSTRUCTION)
        self.assertIn('engine/select.py:LIMITS', recall['visible_symbols'])

    def test_nonproduction_steps_are_ignored_without_discarding_production_plan(self):
        plan = self.plan()
        plan['steps'].append({'id': 'S2', 'file': 'tests/test_select.py', 'symbol': '', 'action': 'create',
                              'purpose': 'Test feature', 'depends_on': ['S1'], 'evidence': []})
        checked, report = self.context().parse_validate(json.dumps(plan))
        self.assertEqual([s['id'] for s in checked['steps']], ['S1'])
        self.assertEqual(len(report['ignored_steps']), 1)

    def test_closed_policy_lifecycle_is_a_candidate_not_a_primary_template(self):
        self.write('engine/mode.py', 'class Mode:\n'
                   '    def __init__(self, params):\n'
                   '        self.build_missing_patterns = []\n'
                   '        self._build_missing_excluded = []\n'
                   '        for param in params:\n'
                   '            if param == "missing":\n'
                   '                pass\n'
                   '            elif param.startswith("missing:"):\n'
                   '                value = param[len("missing:"):]\n'
                   '                if value.startswith("!"):\n'
                   '                    self._build_missing_excluded.append(value[1:])\n'
                   '                else:\n'
                   '                    self.build_missing_patterns.append(value)\n'
                   '    def allowed(self, item):\n'
                   '        return self.should_build_missing(item)\n'
                   '    def should_build_missing(self, item):\n'
                   '        if self._build_missing_excluded:\n'
                   '            return item not in self._build_missing_excluded\n'
                   '        return item in self.build_missing_patterns\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'mode policy')
        self.instance['base_commit'] = self.git('rev-parse', 'HEAD')
        self.instance['problem_statement'] = (
            '<request>Support an independent compatible build policy</request>'
            '<definitions><definition file="engine/mode.py">'
            'class Mode:\n    def allowed_compatible(self, item): ...\n'
            '</definition></definitions>')
        self.instance['new_components'] = [
            {'file': 'engine/mode.py', 'components': [
                {'name': 'Mode.allowed_compatible', 'type': 'function'}]}]
        context = self.context()
        prompt = context.planning_prompt(self.instance)
        self.assertEqual(len(context.structural_analogies), 1)
        analogy = context.structural_analogies[0]
        self.assertEqual(analogy['source_concept'], 'missing')
        self.assertEqual(analogy['target_concept'], 'compatible')
        self.assertEqual(analogy['observed_chain'][0]['input_forms'], ['missing', 'missing:'])
        self.assertIn('Mode.__init__', prompt)
        self.assertIn('candidate_to_verify', prompt)
        self.assertNotIn('PRIMARY template', prompt)

        plan = self.plan()
        plan['analogy_decisions'] = [{'component': 'Mode.allowed_compatible',
                                      'decision': 'apply',
                                      'reason': 'The requested policy needs the observed input-state-query-dispatch lifecycle.'}]
        plan['steps'] = [
            {'id': 'S1', 'file': 'engine/mode.py', 'symbol': 'Mode.allowed_compatible',
             'action': 'create_symbol', 'purpose': 'Create the target policy query',
             'depends_on': [], 'constraints': [],
             'evidence': [{'source': 'requirement', 'file': 'engine/mode.py',
                           'symbol': 'Mode.allowed_compatible'}]},
            {'id': 'S2', 'file': 'engine/mode.py', 'symbol': 'Mode.__init__',
             'action': 'modify', 'purpose': 'Own target input state',
             'depends_on': [], 'constraints': [],
             'evidence': [{'source': 'current', 'file': 'engine/mode.py',
                           'symbol': 'Mode.__init__'}]},
            {'id': 'S3', 'file': 'engine/mode.py', 'symbol': 'Mode.allowed',
             'action': 'modify', 'purpose': 'Connect the target query at the observed dispatcher',
             'depends_on': ['S1'], 'constraints': [],
             'evidence': [{'source': 'current', 'file': 'engine/mode.py',
                           'symbol': 'Mode.allowed'}]},
        ]
        checked, _ = context.parse_validate(json.dumps(plan))
        self.assertEqual(checked['analogy_decisions'][0]['role_mapping']['dispatcher'], 'Mode.allowed')
        generation, _ = context.generation_prompt(self.instance, checked, OUTPUT_FORMAT_INSTRUCTION)
        for symbol in ['Mode.__init__', 'Mode.allowed', 'Mode.should_build_missing']:
            self.assertIn(symbol, generation)
        self.assertIn('not mandatory templates', generation)

        plan['steps'] = plan['steps'][:2]
        completed, report = context.parse_validate(json.dumps(plan))
        added = next(step for step in completed['steps'] if step['symbol'] == 'Mode.allowed')
        self.assertEqual(added['action'], 'modify')
        self.assertTrue(any('dispatcher:Mode.allowed' in warning for warning in report['warnings']))

    def test_name_similarity_without_owned_state_is_not_an_analogy(self):
        self.write('engine/mode.py', 'class Engine:\n'
                   '    def find_existing(self, node):\n'
                   '        return self.download(node)\n'
                   '    def download(self, node):\n'
                   '        return node\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'selection helpers')
        self.instance['base_commit'] = self.git('rev-parse', 'HEAD')
        self.instance['problem_statement'] = '<request>Find a build compatible</request>'
        self.instance['new_components'] = [
            {'file': 'engine/mode.py', 'components': [
                {'name': 'Engine.find_build_compatible', 'type': 'function'}]}]
        self.assertEqual(self.context().structural_analogies, [])

    def test_required_large_function_is_visible_or_fails_before_codegen(self):
        self.write('model/state.py', 'def initialize_hidden(node):\n' + '    node.profile = 1\n' * 700)
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'large function')
        self.instance['base_commit'] = self.git('rev-parse', 'HEAD')
        context = self.context()
        checked, _ = context.parse_validate(json.dumps(self.plan()))
        prompt, recall = context.generation_prompt(self.instance, checked, OUTPUT_FORMAT_INSTRUCTION)
        self.assertIn('model/state.py:initialize_hidden', recall['visible_symbols'])
        self.assertLessEqual(len(prompt), 80000)


if __name__ == '__main__':
    unittest.main()
