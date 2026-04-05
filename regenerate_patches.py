#!/usr/bin/env python3
"""Regenerate patches from existing edits.json files and produce result.jsonl.

Uses the fixed edits_to_patch logic to produce well-formed git diffs.
"""
import json
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional
import difflib

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_instance
from config import REPOS_DIR, OUTPUT_DIR


def _quick_ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _apply_edit_to_text(text: str, before: str, after: str) -> Optional[str]:
    if before in text:
        return text.replace(before, after, 1)
    norm_before = before.strip()
    lines = text.splitlines()
    before_lines = norm_before.splitlines()
    n = len(before_lines)
    if n == 0:
        return None
    best_idx = -1
    best_ratio = 0.0
    for i in range(max(0, len(lines) - n + 1)):
        window = "\n".join(lines[i:i + n])
        ratio = _quick_ratio(window, norm_before)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_idx >= 0 and best_ratio > 0.5:
        new_lines = lines[:best_idx] + after.splitlines() + lines[best_idx + n:]
        return "\n".join(new_lines)
    return None


def edits_to_patch(edits: List[Dict], repo_dir: Path) -> str:
    """Convert edit blocks to a single well-formed git diff per file."""
    file_edits = OrderedDict()
    for edit in edits:
        fp = edit["file_path"]
        file_edits.setdefault(fp, []).append(edit)

    patch_parts = []

    for fp, fp_edits in file_edits.items():
        full_path = repo_dir / fp

        if not full_path.exists():
            combined_after = "\n".join(e["after"] for e in fp_edits)
            after_lines = combined_after.splitlines()
            patch_parts.append(
                f"diff --git a/{fp} b/{fp}\n"
                f"new file mode 100644\n"
                f"--- /dev/null\n"
                f"+++ b/{fp}\n"
                f"@@ -0,0 +1,{len(after_lines)} @@\n"
                + "\n".join(f"+{line}" for line in after_lines)
                + "\n"
            )
            continue

        original = full_path.read_text()
        modified = original
        any_applied = False
        for edit in fp_edits:
            before = edit["before"]
            after = edit["after"]
            if not before:
                continue
            result = _apply_edit_to_text(modified, before, after)
            if result is not None:
                modified = result
                any_applied = True

        if not any_applied:
            continue

        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            orig_lines, mod_lines,
            fromfile=f"a/{fp}", tofile=f"b/{fp}",
            n=3,
        ))
        if diff_lines:
            patch_parts.append(
                f"diff --git a/{fp} b/{fp}\n"
                + "".join(diff_lines)
            )

    return "\n".join(patch_parts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Base output directory with per-instance folders")
    parser.add_argument("--result-jsonl", default=None,
                        help="Output JSONL path (default: <output-dir>/result.jsonl)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    result_path = Path(args.result_jsonl) if args.result_jsonl else output_dir / "result.jsonl"

    rows = []
    skipped = 0
    regenerated = 0

    for case_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        edits_file = case_dir / "edits.json"
        if not edits_file.exists():
            skipped += 1
            continue

        # Load instance_id from result.json if available, else dir name
        iid = case_dir.name
        result_file = case_dir / "result.json"
        if result_file.exists():
            try:
                data = json.loads(result_file.read_text())
                if data.get("instance_id"):
                    iid = data["instance_id"]
            except (json.JSONDecodeError, OSError):
                pass

        edits = json.loads(edits_file.read_text())
        if not edits:
            skipped += 1
            continue

        # Determine repo dir
        repo_dir = REPOS_DIR / iid

        # Generate patch
        patch = edits_to_patch(edits, repo_dir)

        # Save regenerated patch
        (case_dir / "generated.patch").write_text(patch)

        rows.append({
            "instance_id": iid,
            "model_patch": patch,
            "model_name_or_path": "deepseek-chat",
        })
        regenerated += 1

    # Write JSONL
    with open(result_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Regenerated: {regenerated}, Skipped: {skipped}")
    print(f"Output: {result_path}")


if __name__ == "__main__":
    main()
