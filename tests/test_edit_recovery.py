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


if __name__ == '__main__':
    unittest.main()
