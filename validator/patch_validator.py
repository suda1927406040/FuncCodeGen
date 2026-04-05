"""Patch validator - compare generated code against gold patch."""
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class FileDiff:
    """Represents a change to a single file in a patch."""
    file_path: str
    is_new_file: bool
    hunks: List[Dict[str, Any]] = field(default_factory=list)  # [{added: [lines], removed: [lines]}]


@dataclass
class PatchAnalysis:
    """Structured analysis of a git patch."""
    files_changed: List[FileDiff] = field(default_factory=list)
    new_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    total_lines_added: int = 0
    total_lines_removed: int = 0


@dataclass
class ValidationResult:
    """Result of comparing generated output against gold patch."""
    instance_id: str = ""
    structural_match: float = 0.0    # 0-1, how many gold files were covered
    file_matches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    test_result: Optional[Dict[str, Any]] = None  # test execution results if run


def parse_patch(patch_text: str) -> PatchAnalysis:
    """Parse a unified diff patch into structured format."""
    analysis = PatchAnalysis()
    if not patch_text or not patch_text.strip():
        return analysis

    current_diff = None
    current_hunk = None

    for line in patch_text.split("\n"):
        if line.startswith("diff --git"):
            # New file diff
            if current_diff:
                analysis.files_changed.append(current_diff)
            match = re.match(r'diff --git a/(.*?) b/(.*?)$', line)
            if match:
                file_path = match.group(2)
                current_diff = FileDiff(file_path=file_path, is_new_file=False)
                current_hunk = None

        elif line.startswith("new file"):
            if current_diff:
                current_diff.is_new_file = True
            if current_diff and current_diff.file_path not in analysis.new_files:
                analysis.new_files.append(current_diff.file_path)

        elif line.startswith("@@"):
            # New hunk
            if current_diff and current_hunk:
                current_diff.hunks.append(current_hunk)
            current_hunk = {"added": [], "removed": []}

        elif current_hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk["added"].append(line[1:])
                analysis.total_lines_added += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk["removed"].append(line[1:])
                analysis.total_lines_removed += 1

    # Don't forget the last diff/hunk
    if current_diff:
        if current_hunk:
            current_diff.hunks.append(current_hunk)
        analysis.files_changed.append(current_diff)

    # Build modified files list
    for fd in analysis.files_changed:
        if not fd.is_new_file and fd.file_path not in analysis.modified_files:
            analysis.modified_files.append(fd.file_path)

    return analysis


def validate_structure(generated_files: Dict[str, str],
                       gold_patch: str) -> ValidationResult:
    """Validate generated output against gold patch at structural level.

    Args:
        generated_files: dict of file_path -> action ("created"|"modified"|error)
        gold_patch: the gold standard patch text

    Returns:
        ValidationResult with structural match score
    """
    gold = parse_patch(gold_patch)
    result = ValidationResult(instance_id="")

    # Check each gold file
    covered = 0
    for gold_file in gold.files_changed:
        expected_action = "create" if gold_file.is_new_file else "modify"
        actual = generated_files.get(gold_file.file_path, "missing")

        match_info = {
            "expected_action": expected_action,
            "actual_action": actual,
            "gold_hunks": len(gold_file.hunks),
            "is_new_file": gold_file.is_new_file,
        }

        if actual == expected_action or actual == "created" or actual == "modified":
            covered += 1
            match_info["covered"] = True
        else:
            match_info["covered"] = False

        result.file_matches[gold_file.file_path] = match_info

    if gold.files_changed:
        result.structural_match = covered / len(gold.files_changed)
    else:
        result.structural_match = 0.0

    return result


def run_tests(repo_dir: Path, test_commands: List[str],
              timeout: int = 120) -> Dict[str, Any]:
    """Run test commands and collect results.

    Args:
        repo_dir: Path to the repo
        test_commands: List of test commands to run
        timeout: Timeout per command in seconds

    Returns:
        {"passed": int, "failed": int, "errors": int, "output": str}
    """
    all_output = []
    passed = failed = errors = 0

    for cmd in test_commands:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=str(repo_dir), timeout=timeout,
            )
            all_output.append(f"$ {cmd}\n{result.stdout}\n{result.stderr}")

            # Parse pytest output
            if "passed" in result.stdout.lower():
                import re
                match = re.search(r'(\d+) passed', result.stdout)
                if match:
                    passed += int(match.group(1))
                match = re.search(r'(\d+) failed', result.stdout)
                if match:
                    failed += int(match.group(1))
                match = re.search(r'(\d+) error', result.stdout)
                if match:
                    errors += int(match.group(1))

            if result.returncode != 0:
                failed += 1

        except subprocess.TimeoutExpired:
            all_output.append(f"$ {cmd}\nTIMEOUT after {timeout}s")
            errors += 1
        except Exception as e:
            all_output.append(f"$ {cmd}\nERROR: {e}")
            errors += 1

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "output": "\n".join(all_output),
    }
