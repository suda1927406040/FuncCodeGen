import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from data_loader import parse_problem_statement
from explorer.ast_analyzer import ASTAnalyzer
from explorer.context_builder import ContextBuilder
from explorer.feature_retriever import RepositoryIndex
from run_oracle import _align_indent, apply_edits_to_repo, build_prompt, edits_to_patch, parse_edits


class FeatureRecallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def context(self, name="Engine.select_compatible", file="pkg/engine.py"):
        statement = "<request>Support compatible profile selection</request>"
        raw = [{"file": file, "components": [{"name": name, "type": "function"}]}]
        return ContextBuilder(self.root).build_context("case", statement, [], new_components_raw=raw)

    def test_large_file_symbols_are_not_lost(self):
        self.write("pkg/engine.py", "# filler\n" * 4000 +
                   "class Engine:\n    def select(self, profile):\n        return profile\n")
        info = ASTAnalyzer(max_file_chars=100).parse_file(self.root, "pkg/engine.py")
        self.assertIn("select", info.classes["Engine"].methods)

    def test_cross_file_alias_and_reexport_are_resolved(self):
        self.write("pkg/engine.py", "from pkg.api import prepare as prep\n"
                   "class Engine:\n    def select(self, node):\n        return prep(node)\n")
        self.write("pkg/api.py", "from pkg.producer import prepare\n")
        self.write("pkg/producer.py", "def prepare(node):\n    return node.profile\n")
        context = self.context()
        self.assertIn("pkg/producer.py", [c.file for c in context.recalled_files])
        candidate = next(c for c in context.recalled_files if c.file == "pkg/producer.py")
        self.assertTrue(any("callee" in r for r in candidate.reasons))

    def test_state_producer_is_retrieved_without_call_or_history(self):
        self.write("pkg/engine.py", "class Engine:\n    def select(self, node):\n"
                   "        return node.compatible_profile\n")
        self.write("pkg/producer.py", "def initialize(node, profile):\n"
                   "    node.compatible_profile = profile\n")
        self.write("tests/test_reference.py", "def initialize(node):\n"
                   "    node.compatible_profile = 'reference'\n")
        context = self.context()
        files = [c.file for c in context.recalled_files]
        self.assertIn("pkg/producer.py", files)
        self.assertNotIn("tests/test_reference.py", files)
        candidate = next(c for c in context.recalled_files if c.file == "pkg/producer.py")
        self.assertEqual(candidate.locations[0]["state_updates"][0]["field"], "compatible_profile")

    def test_state_update_keeps_validation_guard_and_exception_type(self):
        self.write("pkg/engine.py", "class Engine:\n    def select(self, node):\n"
                   "        return node.cannot_build\n")
        self.write("pkg/producer.py", "def validate(node):\n    if node.check_build:\n"
                   "        try:\n            node.validate()\n        except ValueError as error:\n"
                   "            node.cannot_build = str(error)\n")
        candidate = next(c for c in self.context().recalled_files if c.file == "pkg/producer.py")
        update = candidate.locations[0]["state_updates"][0]
        self.assertEqual(update["guards"], ["if node.check_build", "except ValueError"])

    def test_module_level_functions_are_anchors(self):
        self.write("pkg/functions.py", "from pkg.helper import normalize\n"
                   "def select_profile(profile):\n    return normalize(profile)\n")
        self.write("pkg/helper.py", "def normalize(profile):\n    return profile.strip()\n")
        context = self.context("select_compatible_profile", "pkg/functions.py")
        self.assertIn("pkg/helper.py", [c.file for c in context.recalled_files])
        self.assertEqual(context.components[0].similar_functions[0]["class"], "")

    def test_new_package_recalls_existing_export_and_missing_initializer(self):
        self.write("pkg/tools/__init__.py", "from .existing import existing\n")
        self.write("pkg/tools/existing.py", "def existing():\n    return 1\n")
        context = self.context("new_feature", "pkg/tools/new_package/feature.py")
        files = {c.file for c in context.recalled_files}
        self.assertTrue({"pkg/tools/__init__.py", "pkg/tools/new_package/__init__.py"}.issubset(files))

    def test_multiple_anchors_cover_different_dependencies(self):
        self.write("pkg/engine.py", "from pkg.first import primary\nfrom pkg.second import secondary\n"
                   "class Engine:\n    def select_local(self, node):\n        return primary(node)\n"
                   "    def select_remote(self, node):\n        return secondary(node)\n")
        self.write("pkg/first.py", "def primary(node):\n    return node.local\n")
        self.write("pkg/second.py", "def secondary(node):\n    return node.remote\n")
        context = self.context()
        files = {c.file for c in context.recalled_files}
        self.assertTrue({"pkg/first.py", "pkg/second.py"}.issubset(files))

    def test_prompt_budget_keeps_retrieved_producer_visible(self):
        self.write("pkg/engine.py", "# filler\n" * 10000 +
                   "class Engine:\n    def select(self, node):\n        return node.compatible_profile\n")
        self.write("pkg/producer.py", "def initialize(node, profile):\n"
                   "    node.compatible_profile = profile\n")
        instance = {"problem_statement": "<request>Support compatible profile selection</request>",
                    "new_components": [{"file": "pkg/engine.py", "components": [
                        {"name": "Engine.select_compatible", "type": "function"}]}]}
        context = self.context()
        prompt = build_prompt(instance, self.root, context)
        self.assertLessEqual(len(prompt), 80000)
        self.assertIn("[start of pkg/producer.py]", prompt)
        self.assertIn("node.compatible_profile = profile", prompt)

    def test_missing_outer_closer_recovers_complete_edits_only(self):
        complete = ("<edit>\n[start of the snippet before editing in file.py]\nold\n"
                    "[end of the snippet before editing in file.py]\n"
                    "[start of the snippet after editing in file.py]\nnew\n"
                    "[end of the snippet after editing in file.py]\n")
        incomplete = ("<edit>\n[start of the snippet before editing in other.py]\nold\n"
                      "[end of the snippet before editing in other.py]\n"
                      "[start of the snippet after editing in other.py]\nunfinished")
        self.assertEqual([e["file_path"] for e in parse_edits(complete + incomplete)], ["file.py"])
        self.assertEqual(parse_edits(incomplete), [])

    def test_empty_before_is_a_valid_new_file_edit(self):
        response = ("<edit>\n[start of the snippet before editing in new.py]\n"
                    "[end of the snippet before editing in new.py]\n"
                    "[start of the snippet after editing in new.py]\ndef f():\n    return 1\n"
                    "[end of the snippet after editing in new.py]\n</edit>")
        self.assertEqual(parse_edits(response)[0]["before"], "")

    def test_mismatched_edit_paths_are_rejected(self):
        response = ("<edit>\n[start of the snippet before editing in a.py]\nold\n"
                    "[end of the snippet before editing in a.py]\n"
                    "[start of the snippet after editing in b.py]\nnew\n"
                    "[end of the snippet after editing in b.py]\n</edit>")
        self.assertEqual(parse_edits(response), [])

    def test_indent_alignment_can_remove_excess_indentation(self):
        self.assertEqual(_align_indent(["def f():", "    pass"],
                                      ["    def f():", "        return 1"]),
                         ["def f():", "    return 1"])

    def test_export_and_apply_agree_including_new_files_and_missing_eof_newline(self):
        self.write("pkg/existing.py", "VALUE = 1")
        edits = [{"file_path": "pkg/existing.py", "before": "VALUE = 1", "after": "VALUE = 2"},
                 {"file_path": "pkg/new.py", "before": "", "after": "def feature():\n    return 42"}]
        patch = edits_to_patch(edits, self.root)
        self.assertIn("No newline at end of file", patch)
        self.assertIn("new file mode", patch)
        apply_edits_to_repo(edits, self.root)
        self.assertEqual((self.root / "pkg/existing.py").read_text(), "VALUE = 2")
        self.assertTrue((self.root / "pkg/new.py").is_file())

    def test_empty_new_package_initializer_can_be_applied(self):
        edits = [{"file_path": "pkg/__init__.py", "before": "", "after": ""}]
        result = apply_edits_to_repo(edits, self.root)
        self.assertEqual(result["pkg/__init__.py"], "created")
        self.assertEqual((self.root / "pkg/__init__.py").read_text(), "")

    def test_unlocatable_or_ambiguous_edits_do_not_partially_apply(self):
        self.write("pkg/a.py", "VALUE = 1\n")
        self.write("pkg/b.py", "def one():\n    return 1\ndef two():\n    return 1\n")
        edits = [{"file_path": "pkg/a.py", "before": "VALUE = 1", "after": "VALUE = 2"},
                 {"file_path": "pkg/b.py", "before": "return 1", "after": "return 2"}]
        with self.assertRaises(ValueError):
            apply_edits_to_repo(edits, self.root)
        self.assertEqual((self.root / "pkg/a.py").read_text(), "VALUE = 1\n")

    def test_syntax_invalid_patch_is_rejected_before_application(self):
        self.write("pkg/a.py", "VALUE = 1\n")
        edits = [{"file_path": "pkg/a.py", "before": "VALUE = 1", "after": "VALUE = ("}]
        with self.assertRaises(SyntaxError):
            apply_edits_to_repo(edits, self.root)
        self.assertEqual((self.root / "pkg/a.py").read_text(), "VALUE = 1\n")

    def test_index_uses_requested_base_tree_not_future_files(self):
        self.write("pkg/base.py", "def base():\n    return 1\n")
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "base"], cwd=self.root, check=True)
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        self.write("pkg/future.py", "def future():\n    return 2\n")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        index = RepositoryIndex(self.root, base)
        self.assertIn("pkg/base.py", index.imports)
        self.assertNotIn("pkg/future.py", index.imports)


if __name__ == "__main__":
    unittest.main()
