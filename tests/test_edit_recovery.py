import tempfile
from pathlib import Path
import unittest

from generator.edit_recovery import expand_omissions
from run_oracle import _align_indent, edits_to_patch, parse_edits


class EditRecoveryTests(unittest.TestCase):
    def test_mislabeled_closer_is_recovered_only_for_same_path_complete_sections(self):
        response = ('<edit>\n[start of the snippet before editing in file.py]\nVALUE = 1\n'
                    '[end of the snippet before editing in file.py]\n'
                    '[start of the snippet after editing in file.py]\nVALUE = 2\n'
                    '[end of the snippet before editing in file.py]\n</edit>')
        self.assertEqual(parse_edits(response)[0]['after'], 'VALUE = 2')
        self.assertEqual(parse_edits(response.replace('after editing in file.py', 'after editing in other.py')), [])

    def test_new_sibling_does_not_reindent_existing_method_body(self):
        original = ['        return value']
        after = ['        return value', '', '    def sibling(self):', '        return 2']
        self.assertEqual(_align_indent(original, after), after)

    def test_new_top_level_function_preserves_existing_source_indentation(self):
        original = ['    return value']
        after = ['    return value', '', 'def feature():', '    return 2']
        self.assertEqual(_align_indent(original, after), after)

    def test_omission_is_expanded_without_deleting_hidden_source(self):
        source = 'import os\n\ndef existing():\n    return 42\n\ndef other():\n    return 1\n'
        before = 'import os\n\n# ... other source blocks omitted ...\n\ndef other():\n    return 1'
        after = 'import os\nimport sys\n\n# ... other source blocks omitted ...\n\ndef other():\n    return 2'
        expanded_before, expanded_after = expand_omissions(before, after, source)
        self.assertIn('def existing():\n    return 42', expanded_before)
        self.assertIn('def existing():\n    return 42', expanded_after)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'file.py').write_text(source)
            patch = edits_to_patch([{'file_path': 'file.py', 'before': before, 'after': after}], root)
        self.assertIn('+import sys', patch)
        self.assertNotIn('-    return 42', patch)

    def test_ambiguous_omission_is_rejected(self):
        before = 'start\n# ... other source blocks omitted ...\nend'
        with self.assertRaises(ValueError):
            expand_omissions(before, before, 'start\nvalue\nend\nstart\nother\nend')

    def test_identical_duplicate_edits_are_applied_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'file.py').write_text('VALUE = 1\n')
            edit = {'file_path': 'file.py', 'before': 'VALUE = 1', 'after': 'VALUE = 2'}
            patch = edits_to_patch([edit, dict(edit)], root)
        self.assertEqual(patch.count('+VALUE = 2'), 1)

    def test_partial_signature_context_does_not_delete_the_real_line_suffix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'file.py').write_text('def one():\n    return 1\n\ndef following(a, b=None):\n    return b\n')
            before = '    return 1\n\ndef following(a,'
            after = '    return 1\n\ndef feature():\n    return 2\n\ndef following(a,'
            patch = edits_to_patch([{'file_path': 'file.py', 'before': before, 'after': after}], root)
        self.assertIn(' def following(a, b=None):', patch)
        self.assertIn('+def feature():', patch)

    def test_existing_empty_file_accepts_an_empty_before(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / '__init__.py').write_text('')
            patch = edits_to_patch([{'file_path': '__init__.py', 'before': '', 'after': 'from .feature import feature'}], root)
        self.assertIn('+from .feature import feature', patch)

    def test_blank_line_difference_and_partial_signature_keep_following_function(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'file.py').write_text('def one():\n    result = 1\n\n    return result\n\n\n\ndef following(a, b=None):\n    return b\n')
            before = '    result = 1\n\n    return result\n\n\ndef following(a,'
            after = '    result = 1\n\n    return result\n\n\ndef feature():\n    return 2\n\n\ndef following(a,'
            patch = edits_to_patch([{'file_path': 'file.py', 'before': before, 'after': after}], root)
        self.assertIn('def following(a, b=None):', patch)
        self.assertIn('+def feature():', patch)

    def test_new_global_function_can_be_added_without_assuming_excerpts_are_adjacent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'file.py').write_text('def first():\n    return 1\n\ndef hidden():\n    return 42\n\ndef following(a, b=None):\n    return b\n')
            before = '    return 1\n\ndef following(a,'
            after = '    return 1\n\ndef feature():\n    return 2\n\ndef following(a,'
            patch = edits_to_patch([{'file_path': 'file.py', 'before': before, 'after': after}], root)
        self.assertIn('+def feature():', patch)
        self.assertNotIn('-def hidden', patch)
        self.assertNotIn('-def following', patch)


if __name__ == '__main__':
    unittest.main()
