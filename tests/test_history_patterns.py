from dataclasses import asdict
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from explorer.context_builder import ContextBuilder
from explorer.feature_retriever import RepositoryIndex
from explorer.history_patterns import CommitPatternRetriever, companion_files, focused_diff, render_commit_patterns
from run_oracle import build_prompt


class HistoryPatternTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'History Test')
        self.git('config', 'user.email', 'test@example.invalid')

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True).strip()

    def write(self, file, source):
        path = self.root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    def commit(self, message):
        self.git('add', '.')
        self.git('commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def integration_history(self):
        self.write('engine/select.py', 'def select_profile(node):\n    return node.profile\n')
        self.write('model/state.py', 'def initialize_flags(node):\n    node.active = False\n')
        self.write('api/__init__.py', 'from engine.select import select_profile\n')
        self.write('config/choices.json', '{"modes": ["default"]}\n')
        self.commit('Initial library layout')
        self.write('engine/select.py', 'from model.state import initialize_flags\n'
                   'def select_profile(node):\n    initialize_flags(node)\n    return node.profile\n')
        # Companion contains no profile/compatibility keyword and is in a
        # different top-level package, but participates in the same operation.
        self.write('model/state.py', 'def initialize_flags(node):\n'
                   '    node.active = True\n    node.failure = None\n')
        self.write('api/__init__.py', 'from engine.select import select_profile\n__all__ = ["select_profile"]\n')
        self.write('config/choices.json', '{"modes": ["default", "alternate"]}\n')
        self.write('tests/test_selection.py', 'REFERENCE = "this test must not enter the historical context"\n')
        return self.commit('Support compatible profile selection\n\nInitialize state before selecting a profile and expose the public entry.')

    def retrieve(self, base, name='select_profile', file='engine/select.py'):
        index = RepositoryIndex(self.root, base)
        ctx = SimpleNamespace(base_commit=base, request='Support compatible profile selection',
                              components=[SimpleNamespace(name='select_compatible_profile', target_file=file)],
                              request_reference_keys=[])
        retriever = CommitPatternRetriever(self.root, index)
        patterns = retriever.retrieve(ctx, {'new_feature': [file + ':' + name]})
        return retriever, patterns

    def test_group_keeps_cross_package_state_exports_and_configuration(self):
        base = self.integration_history()
        _, patterns = self.retrieve(base)
        pattern = next(value for value in patterns if value.commit == base)
        by_file = {change.file: change for change in pattern.changes}
        self.assertTrue({'engine/select.py', 'model/state.py', 'api/__init__.py', 'config/choices.json'} <= set(by_file))
        self.assertNotIn('tests/test_selection.py', by_file)
        self.assertIn('public exports', by_file['api/__init__.py'].roles)
        self.assertIn('initialize_flags', by_file['model/state.py'].current_symbols)
        self.assertTrue(any('resolved static base call' in link and 'model/state.py' in link for link in pattern.relationships))
        records = companion_files([pattern], {'engine/select.py'})
        self.assertTrue(all(record.anchor_commit == base for record in records))

    def test_verified_introduction_distinguishes_same_name_in_other_class(self):
        self.write('engine/select.py', 'class Other:\n    def select_profile(self):\n        return 1\n')
        original = self.commit('Initial unrelated class')
        self.write('engine/select.py', 'class Other:\n    def select_profile(self):\n        return 1\n'
                   'class Engine:\n    def select_profile(self):\n        return 2\n')
        base = self.commit('Add profile selection to Engine')
        retriever = CommitPatternRetriever(self.root, RepositoryIndex(self.root, base))
        self.assertEqual(retriever._introduction(('engine/select.py', 'Engine.select_profile'), base), base)
        self.assertNotEqual(base, original)

    def test_future_commit_and_untracked_gold_like_file_are_excluded(self):
        base = self.integration_history()
        self.write('engine/select.py', 'def select_profile(node):\n    return "FUTURE_IMPLEMENTATION"\n')
        future = self.commit('Support compatible profile selection with future implementation')
        self.git('checkout', '-q', base)
        self.write('engine/untracked_reference.py', 'FUTURE = "not a base file"\n')
        _, patterns = self.retrieve(base)
        self.assertNotIn(future, [pattern.commit for pattern in patterns])
        self.assertNotIn('FUTURE_IMPLEMENTATION', str([asdict(pattern) for pattern in patterns]))
        self.assertFalse(any(change.file == 'engine/untracked_reference.py' for pattern in patterns for change in pattern.changes))
        for pattern in patterns:
            self.git('merge-base', '--is-ancestor', pattern.commit, base)

    def test_message_retrieval_finds_existing_function_change(self):
        self.write('engine/select.py', 'def select_profile(node):\n    return node.profile\n')
        self.commit('Initial definition')
        self.write('engine/select.py', 'def select_profile(node):\n'
                   '    if node.active:\n        return node.profile\n    return None\n')
        base = self.commit('Support compatible profile selection guards')
        _, patterns = self.retrieve(base)
        self.assertIn(base, [pattern.commit for pattern in patterns])
        self.assertIn('requirement/message similarity', next(p for p in patterns if p.commit == base).origins)

    def test_large_refactor_is_not_a_feature_recipe(self):
        self.write('engine/select.py', 'def select_profile(node):\n    return node.profile\n')
        self.commit('Initial profile selection')
        for n in range(30):
            self.write('model/part_%d.py' % n, 'def helper_%d():\n    return 1\n' % n)
        large = self.commit('Refactor compatible profile selection package')
        _, patterns = self.retrieve(large)
        self.assertNotIn(large, [pattern.commit for pattern in patterns])

    def test_prose_only_api_match_does_not_seed_an_unrelated_commit(self):
        self.write('engine/select.py', 'def select_profile(node):\n    return node.profile\n')
        self.write('math/ideals.py', 'def intersection(values):\n    return values\n')
        self.commit('Initial library')
        self.write('math/ideals.py', 'def intersection(values):\n    return sorted(values)\n')
        unrelated = self.commit('Improve ideal intersection algorithm')
        index = RepositoryIndex(self.root, unrelated)
        ctx = SimpleNamespace(base_commit=unrelated,
                              request='Support compatible profile selection\nDocumentation also mentions intersection.',
                              components=[SimpleNamespace(name='select_compatible_profile', target_file='engine/select.py')],
                              request_reference_keys=['math/ideals.py:intersection'])
        patterns = CommitPatternRetriever(self.root, index).retrieve(ctx, {'new': ['engine/select.py:select_profile']})
        self.assertNotIn(unrelated, [pattern.commit for pattern in patterns])

    def test_namespace_reference_does_not_match_same_named_properties(self):
        self.write('pkg/tools/files/__init__.py', '"""Public file tools namespace."""\n')
        self.write('pkg/locks.py', 'class Lock:\n    @property\n    def files(self):\n        return []\n')
        base = self.commit('Initial namespaces and locks')
        statement = '<request>Add helpers to the pkg.tools.files namespace.</request>'
        raw = [{'file': 'pkg/tools/files/new_feature.py', 'components': [{'name': 'new_helper', 'type': 'function'}]}]
        ctx = ContextBuilder(self.root).build_context('case', statement, [], new_components_raw=raw, base_commit=base)
        self.assertNotIn('pkg/locks.py:Lock.files', ctx.request_reference_keys)

    def test_history_prompt_keeps_atomic_group_and_current_source(self):
        base = self.integration_history()
        statement = '<request>Support compatible profile selection</request>'
        raw = [{'file': 'engine/select.py', 'components': [{'name': 'select_compatible_profile', 'type': 'function'}]}]
        ctx = ContextBuilder(self.root).build_context('case', statement, [], new_components_raw=raw, base_commit=base)
        prompt = build_prompt({'problem_statement': statement, 'new_components': raw, 'base_commit': base}, self.root, ctx)
        self.assertLessEqual(len(prompt), 80000)
        self.assertIn('COMMIT ' + base[:12], prompt)
        self.assertIn('model/state.py', prompt)
        self.assertIn('def initialize_flags(node):', prompt.split('\n<code>\n')[1])
        self.assertNotIn('REFERENCE COMMIT DIFF', prompt)
        self.assertEqual(prompt.count('```') % 2, 0)
        small = render_commit_patterns(ctx.history_patterns, budget=1500)
        self.assertLessEqual(len(small), 1500)
        self.assertEqual(small.count('```') % 2, 0)

    def test_off_mode_does_not_run_legacy_history_during_prompt_build(self):
        base = self.integration_history()
        statement = '<request>Support compatible profile selection</request>'
        raw = [{'file': 'engine/select.py', 'components': [{'name': 'select_compatible_profile', 'type': 'function'}]}]
        ctx = ContextBuilder(self.root, history_mode='off').build_context('case', statement, [], new_components_raw=raw, base_commit=base)
        prompt = build_prompt({'problem_statement': statement, 'new_components': raw, 'base_commit': base}, self.root, ctx)
        self.assertFalse(ctx.history_patterns)
        self.assertNotIn('COMMIT ' + base[:12], prompt)
        self.assertNotIn('CALL CHAIN HISTORY', prompt)

    def test_no_git_archive_reports_unavailable(self):
        # A directory outside this test Git tree is needed, or Git would walk
        # upward and silently discover the parent repository.
        with tempfile.TemporaryDirectory() as directory:
            index = RepositoryIndex(Path(directory))
            ctx = SimpleNamespace(base_commit='f' * 40)
            retriever = CommitPatternRetriever(directory, index)
            self.assertEqual(retriever.retrieve(ctx, {}), [])
            self.assertFalse(retriever.diagnostics['available'])

    def test_oversized_diff_is_labelled_and_not_cut_mid_hunk(self):
        diff = 'diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1,100 @@\n' + '+    node.profile = 1\n' * 100
        focused = focused_diff(diff, {'profile'}, 300)
        self.assertLessEqual(len(focused), 300)
        self.assertIn('not an applicable patch', focused)
        self.assertNotIn('@@', focused)


if __name__ == '__main__':
    unittest.main()
