"""
Run code generation with Oracle-style natural-detailed prompt format.
Generates before/after edit blocks, converts to git patch, saves everything to output/{instance_id}/.

Usage:
    conda run -n py38 python run_oracle.py --instance joke2k__faker-738
    conda run -n py38 python run_oracle.py --limit 5
"""
import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import REPOS_DIR, OUTPUT_DIR
from data_loader import load_dataset, load_instance, parse_problem_statement
from repo_manager import prepare_repo, get_file_content
from explorer.context_builder import ContextBuilder
from generator.llm_interface import call_llm

# ============================================================================
# Output format template (extracted from Oracle natural-detailed)
# ============================================================================
OUTPUT_FORMAT_INSTRUCTION = """\
For each edit, output the changes in the following format:

```
<edit>
[start of the snippet before editing in <file_path>]
<code_before_edit>
[end of the snippet before editing in <file_path>]

[start of the snippet after editing in <file_path>]
<code_after_edit>
[end of the snippet after editing in <file_path>]
</edit>
```

Notes:
- The <file_path> is the relative path of the file being edited.
- The <code_before_edit> snippet should include several lines before and after the modified region, unless the file was originally empty.
- If a file was originally empty, leave <code_before_edit> blank but ensure <code_after_edit> includes the new content.
- Ensure the edits are sequential and address all necessary changes to achieve the requested feature.

Here is an example of the output format:

```
<edit>
[start of the snippet before editing in src/code.py]
def factorial(a):
    res = 1
    while a >= 0:
        res *= a
    return res
[end of the snippet before editing in src/code.py]

[start of the snippet after editing in src/code.py]
def factorial(a):
    assert type(a) == int and a >= 0
    res = 1
    while a >= 2:
        res *= a
        a -= 1
    return res
[end of the snippet after editing in src/code.py]
</edit>

<edit>
[start of the snippet before editing in src/demo.py]
[end of the snippet before editing in src/demo.py]

[start of the snippet after editing in src/demo.py]
from code import factorial
print(factorial(5))
[end of the snippet after editing in src/demo.py]
</edit>
```

I need you to solve the feature request with a series of edits in the format shown above.
Respond below:"""


def build_code_section(repo_dir: Path, instance: dict, extra_files: set = None) -> str:
    """Build the <code> section with all relevant file contents.

    Includes: target files from new_components + their __init__.py files
    + any extra files discovered by the explorer (base classes, etc.).
    """
    parts = ["\nAll the involved code files are listed below with their contents.\n"]
    seen_files = set()

    # Collect files from new_components
    target_files = set()
    for comp_item in (instance.get("new_components") or []):
        target_files.add(comp_item["file"])

    # Also add files mentioned in definitions from problem_statement
    parsed = parse_problem_statement(instance["problem_statement"])
    for defn in parsed.get("definitions", []):
        target_files.add(defn["file"])

    # Merge extra files from explorer (base classes, sibling modules, etc.)
    if extra_files:
        target_files |= extra_files

    # Read and include each file
    for fpath in sorted(target_files):
        if fpath in seen_files:
            continue
        seen_files.add(fpath)

        content = get_file_content(repo_dir, fpath)
        if content:
            parts.append(f"\n[start of {fpath}]")
            parts.append(content)
            parts.append(f"[end of {fpath}]\n")
        else:
            parts.append(f"\n[start of {fpath}]")
            parts.append("(file does not exist yet, needs to be created)")
            parts.append(f"[end of {fpath}]\n")

        # Also include parent __init__.py
        parent = str(Path(fpath).parent)
        init_file = f"{parent}/__init__.py"
        if init_file not in seen_files and not fpath.endswith("__init__.py"):
            seen_files.add(init_file)
            init_content = get_file_content(repo_dir, init_file)
            if init_content:
                parts.append(f"\n[start of {init_file}]")
                parts.append(init_content)
                parts.append(f"[end of {init_file}]\n")

    return "\n".join(parts)


def _detect_structural_sibling_pattern(comp, init_method=None):
    """Detect dispatcher→worker pattern in call chain where worker is the better template.

    Pattern example: allowed → should_build_missing.
    New method allowed_compatible should follow should_build_missing's structure, not allowed's.

    Returns dict with primary_template, dispatcher, source_keyword, target_keyword,
    init_parsing_block; or None if no pattern detected.
    """
    if not comp.reference_call_chain or len(comp.reference_call_chain) < 2:
        return None

    new_name = comp.name.split(".")[-1]
    new_words = set(re.findall(r'[a-z]+', new_name))

    # First node is the dispatcher (matched by name similarity)
    dispatcher = comp.reference_call_chain[0]
    dispatcher_words = set(re.findall(r'[a-z]+', dispatcher.method_name))

    # Words unique to new method (not in dispatcher)
    new_unique = new_words - dispatcher_words
    if not new_unique:
        return None

    # Look at subsequent nodes (workers called by dispatcher)
    for node in comp.reference_call_chain[1:]:
        node_words = set(re.findall(r'[a-z]+', node.method_name))
        worker_unique = node_words - dispatcher_words

        if not worker_unique:
            continue

        # Found a worker with unique keywords — this is the structural sibling
        source_kw = max(worker_unique, key=len)
        target_kw = max(new_unique, key=len)

        # Extract __init__ parsing pattern for source keyword
        init_block = None
        if init_method:
            init_block = _extract_init_parsing_pattern(init_method.body_source, source_kw)

        return {
            'primary_template': node,
            'dispatcher': dispatcher,
            'source_keyword': source_kw,
            'target_keyword': target_kw,
            'init_parsing_block': init_block,
        }

    return None


def _extract_init_parsing_pattern(init_source, keyword):
    """Extract the __init__ code block that parses 'keyword:' params.

    Returns the contiguous block of lines around the keyword mention,
    with a few lines of context before and after.
    """
    lines = init_source.split('\n')
    keyword_indices = [i for i, line in enumerate(lines) if keyword in line]

    if not keyword_indices:
        return None

    # Find contiguous blocks (allowing small gaps)
    blocks = []
    current = [keyword_indices[0]]
    for idx in keyword_indices[1:]:
        if idx - current[-1] <= 2:
            current.append(idx)
        else:
            blocks.append(current)
            current = [idx]
    blocks.append(current)

    # Take the largest block
    best = max(blocks, key=len)
    start = max(0, best[0] - 2)
    end = min(len(lines), best[-1] + 2)

    return '\n'.join(lines[start:end])


def build_analysis_section(ctx) -> str:
    """Build the <analysis> section from ContextBuilder's deterministic analysis.

    This is the key section that injects:
    - Calling patterns / call chains of sibling methods
    - Class __init__ variables and class-level variables
    - Inter-component dependencies (e.g. unix_partition calls unix_device)
    - __init__.py analysis (what needs exporting)
    - Similar existing methods to follow as reference
    - Base class methods
    """
    parts = ["\nHere is a detailed analysis of the codebase to help you implement correctly:\n"]

    for comp in ctx.components:
        # Detect dispatcher→worker structural sibling pattern early
        structural_pattern = None
        if (comp.sibling_methods and "__init__" in comp.sibling_methods
                and comp.reference_call_chain and len(comp.reference_call_chain) >= 2):
            structural_pattern = _detect_structural_sibling_pattern(
                comp, comp.sibling_methods["__init__"])

        parts.append(f"\n--- Analysis for {comp.name} ({comp.comp_type}) in {comp.target_file} ---\n")

        # Parent class
        if comp.parent_class:
            parts.append(f"**This is a method of class `{comp.parent_class}`"
                        + (f" (nested class `{comp.nested_class}`)" if comp.nested_class else "")
                        + ".**\n")

        # Class-level variables (TYPE_MAPPING, prefixes, etc.)
        if comp.class_vars:
            parts.append("**Existing class-level variables** (you may need to add similar ones):")
            for var_name, var_val in comp.class_vars.items():
                parts.append(f"  {var_name}: {var_val}")
            parts.append("")

        # Instance variables from __init__
        if comp.class_init_vars:
            parts.append("**Instance variables set in __init__** (your new method may need these):")
            for var, val in comp.class_init_vars.items():
                parts.append(f"  self.{var} = {val}")
            parts.append("")

        # CRITICAL: Full __init__ method body — this contains parameter parsing,
        # initialization patterns, and config logic that new methods MUST follow.
        # e.g. if __init__ parses "missing:pattern" params, a new "compatible" feature
        # needs the same parsing style.
        if comp.sibling_methods and "__init__" in comp.sibling_methods:
            init_method = comp.sibling_methods["__init__"]
            parts.append("**__init__ method full source** (you may need to add similar initialization logic):")
            init_body = init_method.body_source
            if len(init_body) > 4000:
                init_body = init_body[:4000] + "\n    # ... (truncated)"
            parts.append(init_body)
            parts.append("")

            # Trace: which __init__ variables are used by the reference call chain?
            # This tells the LLM "the reference method uses self.X which is set in __init__ line Y,
            # so your new method may need a similar self.Y initialized in __init__"
            if comp.reference_call_chain:
                used_vars = set()
                for node in comp.reference_call_chain:
                    import re as _re
                    # Find all self.xxx references in the call chain methods
                    for match in _re.finditer(r'self\.([a-zA-Z_]\w*)', node.body_source):
                        used_vars.add(match.group(1))
                # Also check the new component's similar functions
                for sim in comp.similar_functions[:1]:
                    for match in _re.finditer(r'self\.([a-zA-Z_]\w*)', sim.get("body", "")):
                        used_vars.add(match.group(1))

                if used_vars:
                    parts.append("**Variables from __init__ used by reference methods** "
                                "(your new method likely needs similar variables added to __init__):")
                    for var in sorted(used_vars):
                        if var in comp.class_init_vars:
                            parts.append(f"  self.{var} = {comp.class_init_vars[var]}  ← USED by reference")
                        # else: variable might come from base class
                    parts.append("")

        # === Structural Sibling Pattern: __init__ must be modified ===
        if structural_pattern:
            sp = structural_pattern
            src_kw = sp['source_keyword']
            tgt_kw = sp['target_keyword']
            worker = sp['primary_template']

            if sp['init_parsing_block']:
                parts.append(f"**CRITICAL `__init__` PARSING PATTERN TO COPY**:")
                parts.append(
                    f"Method `{worker.method_name}()` uses `self.build_{src_kw}_patterns` and "
                    f"`self._build_{src_kw}_excluded`, which are populated in `__init__` by parsing "
                    f"`{src_kw}:` parameters. You MUST add analogous parsing for `{tgt_kw}:` parameters.\n"
                )
                parts.append(f"Extracted `{src_kw}:` parsing from `__init__` (mirror this for `{tgt_kw}:`):")
                parts.append(f"```python")
                parts.append(sp['init_parsing_block'])
                parts.append(f"```\n")

            parts.append(f"**WHY YOU MUST MODIFY `__init__`**:")
            parts.append(
                f"- `{worker.method_name}()` works because `__init__` parses `{src_kw}:` params → "
                f"`self.build_{src_kw}_patterns` / `self._build_{src_kw}_excluded`."
            )
            parts.append(
                f"- Your new method `{comp.name.split('.')[-1]}()` needs analogous variables: "
                f"`self.build_{tgt_kw}_patterns` / `self._build_{tgt_kw}_excluded`."
            )
            parts.append(
                f"- Therefore, you MUST add `{tgt_kw}:` parsing logic in `__init__`, "
                f"mirroring the `{src_kw}:` pattern above."
            )
            parts.append(
                f"- You should also modify `{sp['dispatcher'].method_name}()` to call your new method, "
                f"just as it already calls `{worker.method_name}()`.\n"
            )

        # Sibling methods (LLM needs to see these to follow patterns)
        if comp.sibling_methods:
            comp_short = comp.name.split(".")[-1]
            parts.append(f"**Existing methods in the same class** ({len(comp.sibling_methods)} total):")
            for mname, method in comp.sibling_methods.items():
                if mname == comp_short:
                    continue
                # Show signature + docstring + body
                parts.append(f"\nMethod `{mname}`:")
                if method.docstring:
                    parts.append(f'  Docstring: """{method.docstring}"""')
                # Include full method source — this is critical for pattern learning
                body = method.body_source
                if len(body) > 3000:
                    body = body[:3000] + "\n    # ... (truncated)"
                parts.append(body)
            parts.append("")

        # Call graph of siblings
        if comp.calling_patterns:
            parts.append("**Method call graph** (which self.xxx() each method calls):")
            for caller, callees in comp.calling_patterns.items():
                parts.append(f"  {caller} → calls: {', '.join(callees)}")
            parts.append("")

        # REFERENCE CALL CHAIN — the most important signal
        if comp.reference_call_chain:
            if structural_pattern:
                sp = structural_pattern
                worker = sp['primary_template']
                dispatcher = sp['dispatcher']

                parts.append("**REFERENCE IMPLEMENTATION** — complete call chain context:")
                parts.append(
                    f"`{dispatcher.method_name}()` (dispatcher) calls `{worker.method_name}()` (worker). "
                    f"Your new method `{comp.name.split('.')[-1]}` should follow "
                    f"`{worker.method_name}()` as the **PRIMARY template**.\n"
                )

                # Show worker first (PRIMARY)
                parts.append(
                    f"Method `{worker.class_name}.{worker.method_name}()` "
                    f"**← PRIMARY TEMPLATE, follow this closely**:"
                )
                if worker.callees:
                    parts.append(f"  (internally calls: {', '.join(worker.callees)})")
                body = worker.body_source
                if len(body) > 3000:
                    body = body[:3000] + "\n    # ... (truncated)"
                parts.append(body)
                parts.append("")

                # Show dispatcher next
                parts.append(
                    f"Method `{dispatcher.class_name}.{dispatcher.method_name}()` "
                    f"(dispatcher — should also call your new method):"
                )
                if dispatcher.callees:
                    parts.append(f"  (internally calls: {', '.join(dispatcher.callees)})")
                body = dispatcher.body_source
                if len(body) > 3000:
                    body = body[:3000] + "\n    # ... (truncated)"
                parts.append(body)
                parts.append("")

                # Show any other chain nodes
                for node in comp.reference_call_chain:
                    if (node.method_name != worker.method_name
                            and node.method_name != dispatcher.method_name):
                        parts.append(f"Method `{node.class_name}.{node.method_name}()`:")
                        body = node.body_source
                        if len(body) > 3000:
                            body = body[:3000] + "\n    # ... (truncated)"
                        parts.append(body)
                        parts.append("")
            else:
                parts.append("**REFERENCE IMPLEMENTATION** — follow this pattern closely:")
                parts.append("The most similar existing method and its full call chain are shown below.")
                parts.append("Your new method should follow the SAME structural pattern.\n")
                for node in comp.reference_call_chain:
                    parts.append(f"Method `{node.class_name}.{node.method_name}()`:")
                    if node.callees:
                        parts.append(f"  (internally calls: {', '.join(node.callees)})")
                    body = node.body_source
                    if len(body) > 3000:
                        body = body[:3000] + "\n    # ... (truncated)"
                    parts.append(body)
                    parts.append("")

        # Inter-component dependencies
        if comp.depends_on_components:
            parts.append(f"**INTER-COMPONENT DEPENDENCY**: This component CALLS other new components: {comp.depends_on_components}")
            parts.append(f"You MUST make `{comp.name}` call `self.{comp.depends_on_components[0]}()` (or similar) — do NOT reimplement the logic.\n")

        # Base classes
        if comp.base_classes:
            parts.append("**Base class definitions**:")
            for base_name, (base_file, base_module) in comp.base_classes.items():
                base_cls_name = base_name.split(".")[-1]
                if base_cls_name in base_module.classes:
                    base_cls = base_module.classes[base_cls_name]
                    parts.append(f"  {base_name} (from {base_file}):")
                    parts.append(f"    Bases: {', '.join(base_cls.bases)}")
                    parts.append(f"    Methods: {', '.join(base_cls.methods.keys())}")
                    # Show relevant base methods
                    comp_words = set(re.findall(r'[a-z]+', comp.name.split(".")[-1]))
                    for mname, method in base_cls.methods.items():
                        m_words = set(re.findall(r'[a-z]+', mname))
                        if comp_words & m_words and len(comp_words) > 0:
                            body = method.body_source[:2000]
                            parts.append(f"\n  Relevant base method `{mname}`:")
                            parts.append(f"  {body}")
                    parts.append("")

        # Similar functions
        if comp.similar_functions:
            if structural_pattern:
                parts.append(
                    f"**Similar existing functions** (by name pattern — but note: "
                    f"`{structural_pattern['primary_template'].method_name}()` shown above is the PRIMARY template):"
                )
            else:
                parts.append("**Similar existing functions** (by name pattern, follow their style):")
            for sim in comp.similar_functions[:3]:
                parts.append(f"\n  {sim['class']}.{sim['method']} (similarity: {sim['similarity']:.1%}):")
                parts.append(f"  {sim['body'][:2000]}")
            parts.append("")

        # __init__.py analysis
        if comp.init_py_exports:
            init = comp.init_py_exports
            parts.append(f"**__init__.py analysis** (`{init.file}`):")
            if init.has_all:
                parts.append(f"  Has __all__ = {init.all_names[:20]}")
            if init.imports_our_module:
                parts.append(f"  Already imports our module ✓")
            else:
                parts.append(f"  Does NOT import our module — you MUST add an import and update __all__.")
            if init.sibling_export_pattern:
                parts.append(f"  Export pattern to follow: {init.sibling_export_pattern}")
            parts.append("")

    # Extra changes warning
    if ctx.has_extra_changes:
        parts.append("**IMPORTANT**: In addition to the new components, other code changes may be needed:")
        parts.append("- Adding imports (e.g. `import string`) to existing files")
        parts.append("- Adding class-level variables (e.g. `unix_device_prefixes = (...)`)")
        parts.append("- Registering the new component in existing lookup tables or dispatch dicts")
        parts.append("- Modifying __init__.py to export the new components")
        parts.append("")

    return "\n".join(parts)


def build_prompt(instance: dict, repo_dir: Path, ctx=None) -> str:
    """Build the full Oracle-style natural-detailed prompt WITH analysis."""
    ps = instance["problem_statement"]

    # Extract <request> section
    request_text = ""
    if "<request>" in ps and "</request>" in ps:
        request_text = ps.split("<request>")[1].split("</request>")[0].strip()
    else:
        request_text = ps.split("<<NEW FEATURE REQUEST>>")[1] if "<<" in ps else ps
        if "<request>" in request_text:
            request_text = request_text.split("<request>")[1].split("</request>")[0].strip()

    # Extract <definitions> section
    defs_text = ""
    if "<definitions>" in ps and "</definitions>" in ps:
        defs_text = ps.split("<definitions>")[1].split("</definitions>")[0].strip()

    # Collect extra files from explorer context (base classes, etc.)
    extra_files = set()
    if ctx:
        for comp in ctx.components:
            for base_name, (base_file, base_module) in comp.base_classes.items():
                extra_files.add(base_file)

    # Build analysis section from explorer
    analysis_section = ""
    if ctx:
        analysis_section = build_analysis_section(ctx)

    # Build full prompt
    prompt = f"""You will be provided with code files and a feature request which requires new code to add to the codebase.
<request>
{request_text}
</request>

There are several new functions or classes that need to be implemented, using the definitions below:
<definitions>
{defs_text}
</definitions>

{analysis_section}

<code>
{build_code_section(repo_dir, instance, extra_files)}
</code>

{OUTPUT_FORMAT_INSTRUCTION}"""

    return prompt


def parse_edits(response: str) -> List[Dict]:
    """Parse LLM response into structured edit list.

    Supports both bracket styles:
    - [start of the snippet before editing in file.py]   (Oracle standard)
    - <start of the snippet before editing in file.py>   (LLM variation)

    Returns list of dicts: [{file_path, before, after}]
    """
    edits = []

    # Find all <edit>...</edit> blocks
    edit_blocks = re.findall(r'<edit>(.*?)</edit>', response, re.DOTALL)

    for block in edit_blocks:
        # Try both bracket styles for "start/end of the snippet before/after editing in <file>"
        # Pattern: [start of the snippet before editing in <file>] or <start of the snippet before editing in <file>>
        before_header_pattern = r'[\[<]start of the snippet before editing in (.*?)[\]>]'
        after_header_pattern = r'[\[<]start of the snippet after editing in (.*?)[\]>]'
        before_footer_pattern = r'[\[<]end of the snippet before editing in .*?[\]>]'
        after_footer_pattern = r'[\[<]end of the snippet after editing in .*?[\]>]'

        # Find before section
        before_match = re.search(
            before_header_pattern + r'\s*\n(.*?)\n' + before_footer_pattern,
            block, re.DOTALL,
        )
        after_match = re.search(
            after_header_pattern + r'\s*\n(.*?)\n' + after_footer_pattern,
            block, re.DOTALL,
        )

        if before_match and after_match:
            file_path = before_match.group(1).strip()
            before = before_match.group(2)
            after = after_match.group(2)
            edits.append({
                "file_path": file_path,
                "before": before,
                "after": after,
            })

    # Fallback: try without the "the snippet" phrase
    if not edits:
        edit_blocks = re.findall(r'<edit>(.*?)</edit>', response, re.DOTALL)
        for block in edit_blocks:
            before_files = re.findall(r'[\[<]start.*?before.*?in (.*?)[\]>]', block)
            after_files = re.findall(r'[\[<]start.*?after.*?in (.*?)[\]>]', block)
            if before_files and after_files:
                fp = before_files[0].strip()
                # Extract code between headers and footers
                before_match = re.search(
                    r'[\[<]start.*?before.*?in .*?[\]>]\s*\n(.*?)\n[\[<]end.*?before',
                    block, re.DOTALL,
                )
                after_match = re.search(
                    r'[\[<]start.*?after.*?in .*?[\]>]\s*\n(.*?)\n[\[<]end.*?after',
                    block, re.DOTALL,
                )
                edits.append({
                    "file_path": fp,
                    "before": before_match.group(1) if before_match else "",
                    "after": after_match.group(1) if after_match else "",
                })

    return edits


def edits_to_patch(edits: List[Dict], repo_dir: Path) -> str:
    """Convert edit blocks to a unified diff patch.

    Groups edits by file, applies all edits for each file sequentially,
    then generates a single proper ``diff --git`` block per file.
    This avoids the malformed-patch problem where multiple ``--- a/file``
    headers appear without ``diff --git`` separators.
    """
    import difflib

    # Group edits by file path (preserve order of first appearance)
    from collections import OrderedDict
    file_edits = OrderedDict()
    for edit in edits:
        fp = edit["file_path"]
        file_edits.setdefault(fp, []).append(edit)

    patch_parts = []

    for fp, fp_edits in file_edits.items():
        full_path = repo_dir / fp

        if not full_path.exists():
            # New file — concatenate all "after" content
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

        # Read current file content
        original = full_path.read_text()

        # Apply all edits for this file sequentially to produce modified content
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
            # No edits could be applied — skip this file rather than emit garbage
            continue

        # Generate a single unified diff for this file
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


def _apply_edit_to_text(text: str, before: str, after: str) -> Optional[str]:
    """Try to apply a single before→after edit to *text*.

    Returns the modified text, or None if no match was found.
    """
    # Exact match
    if before in text:
        return text.replace(before, after, 1)

    # Normalised fuzzy match
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


def _quick_ratio(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def apply_edits_to_repo(edits: List[Dict], repo_dir: Path, dry_run: bool = False) -> Dict[str, str]:
    """Apply parsed edits directly to the repo files."""
    outcomes = {}
    for edit in edits:
        fp = edit["file_path"]
        target = repo_dir / fp
        target.parent.mkdir(parents=True, exist_ok=True)

        if not target.exists():
            # New file
            if not dry_run:
                target.write_text(edit["after"] + "\n")
            outcomes[fp] = "created"
            continue

        current = target.read_text()
        before = edit["before"]

        if before and before in current:
            new_content = current.replace(before, edit["after"], 1)
            if not dry_run:
                target.write_text(new_content)
            outcomes[fp] = "modified (exact)"
        else:
            # Fuzzy: find best match and replace
            norm_before = before.strip()
            lines = current.splitlines()
            before_lines = norm_before.splitlines()
            n = len(before_lines)

            best_idx = -1
            best_ratio = 0
            for i in range(max(0, len(lines) - n + 1)):
                window = "\n".join(lines[i:i+n])
                ratio = _quick_ratio(window, norm_before)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i

            if best_idx >= 0 and best_ratio > 0.5:
                new_lines = lines[:best_idx] + edit["after"].splitlines() + lines[best_idx + n:]
                if not dry_run:
                    target.write_text("\n".join(new_lines))
                outcomes[fp] = f"modified (fuzzy {best_ratio:.0%})"
            else:
                # Append mode
                if not dry_run:
                    target.write_text(current.rstrip("\n") + "\n\n" + edit["after"] + "\n")
                outcomes[fp] = "appended (no match)"

    return outcomes


def run_single(instance: dict, output_base: Path, dry_run: bool = False,
               apply: bool = False) -> dict:
    """Run oracle-style generation for one instance."""
    iid = instance["instance_id"]
    out_dir = output_base / iid
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "instance_id": iid,
        "repo": instance["repo"],
        "success": False,
        "n_edits": 0,
        "files": [],
        "timing": {},
    }

    import time
    t_start = time.time()

    # Step 1: Checkout repo
    print(f"[1/4] Preparing repo...")
    try:
        repo_dir = prepare_repo(iid, instance["base_commit"], instance["repo"])
    except Exception:
        import subprocess
        subprocess.run(["git", "fetch", "--unshallow"], cwd=str(REPOS_DIR/iid),
                       capture_output=True, timeout=120)
        subprocess.run(["git", "fetch", "--all"], cwd=str(REPOS_DIR/iid),
                       capture_output=True, timeout=120)
        repo_dir = prepare_repo(iid, instance["base_commit"], instance["repo"])

    result["timing"]["checkout"] = time.time() - t_start

    # Step 2: Run deterministic analysis + Build prompt
    print(f"[2/4] Running AST analysis & building prompt...")
    t0 = time.time()
    parsed = parse_problem_statement(instance["problem_statement"])
    builder = ContextBuilder(repo_dir)
    ctx = builder.build_context(
        instance_id=iid,
        problem_statement=instance["problem_statement"],
        definitions=parsed["definitions"],
        has_extra_changes=parsed["has_extra_changes"],
        new_components_raw=instance.get("new_components"),
    )
    prompt = build_prompt(instance, repo_dir, ctx=ctx)
    (out_dir / "prompt.md").write_text(prompt)
    # Also save the formatted context from explorer for debugging
    (out_dir / "explorer_context.md").write_text(builder.format_context_for_llm(ctx))
    result["timing"]["prompt_build"] = time.time() - t0
    result["n_components"] = len(ctx.components)
    result["analysis"] = {
        comp.name: {
            "parent_class": comp.parent_class,
            "siblings": len(comp.sibling_methods),
            "init_vars": len(comp.class_init_vars),
            "call_chain": [n.method_name for n in comp.reference_call_chain],
            "depends_on": comp.depends_on_components,
            "init_py_needed": (not comp.init_py_exports.imports_our_module) if comp.init_py_exports else None,
        }
        for comp in ctx.components
    }
    print(f"  Components analyzed: {len(ctx.components)}")
    for comp in ctx.components:
        chain = " → ".join(n.method_name for n in comp.reference_call_chain) if comp.reference_call_chain else "none"
        deps = f", depends_on={comp.depends_on_components}" if comp.depends_on_components else ""
        init_need = " [INIT PY NEEDS UPDATE]" if comp.init_py_exports and not comp.init_py_exports.imports_our_module else ""
        print(f"    {comp.name}: chain={chain}{deps}{init_need}")
    print(f"  Prompt: {len(prompt)} chars")

    # Step 3: Call LLM
    print(f"[3/4] Calling LLM...")
    t0 = time.time()
    response = call_llm(
        prompt,
        system="You are an expert Python developer. Output exact code edits in the specified format.",
        model="deepseek-chat",
        max_tokens=8192,
        temperature=0.0,
    )
    result["timing"]["llm_call"] = time.time() - t0
    print(f"  Response: {len(response)} chars")

    # Save raw response
    (out_dir / "response.md").write_text(response)

    # Step 4: Parse edits → generate patch
    print(f"[4/4] Parsing edits...")
    t0 = time.time()
    edits = parse_edits(response)
    result["n_edits"] = len(edits)
    result["files"] = list({e["file_path"] for e in edits})

    if edits:
        # Save parsed edits as JSON
        (out_dir / "edits.json").write_text(json.dumps(edits, indent=2))

        # Generate patch
        patch = edits_to_patch(edits, repo_dir)
        (out_dir / "generated.patch").write_text(patch)

        # Save each edit as before/after files for easy inspection
        for i, edit in enumerate(edits):
            fp_safe = edit["file_path"].replace("/", "_")
            (out_dir / f"edit_{i}_{fp_safe}").write_text(
                f"# FILE: {edit['file_path']}\n"
                f"# ===== BEFORE =====\n{edit['before']}\n\n"
                f"# ===== AFTER =====\n{edit['after']}\n"
            )

        # Apply edits to repo if requested
        if apply:
            outcomes = apply_edits_to_repo(edits, repo_dir, dry_run=dry_run)
            (out_dir / "apply_results.json").write_text(json.dumps(outcomes, indent=2))
            for fp, status in outcomes.items():
                print(f"  [{status}] {fp}")

        result["success"] = True
    else:
        print(f"  WARNING: No edits parsed from response!")
        (out_dir / "parse_failure.txt").write_text(
            f"Could not parse any <edit> blocks from LLM response.\n\n"
            f"Response:\n{response[:3000]}"
        )

    result["timing"]["parse"] = time.time() - t0
    result["timing"]["total"] = time.time() - t_start

    # Save gold patch for comparison
    if instance.get("patch"):
        (out_dir / "gold.patch").write_text(instance["patch"])

    # Save summary
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"  Done: {result['n_edits']} edits in {result['timing']['total']:.1f}s")
    print(f"  Output: {out_dir}/")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", help="Single instance ID")
    parser.add_argument("--limit", type=int, help="Run first N instances")
    parser.add_argument("--repo", help="Filter by repo")
    parser.add_argument("--output", default="/data2/ymz/TOSEM2026/output", help="Output base dir")
    parser.add_argument("--apply", action="store_true", help="Apply edits to repo")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to repo")
    args = parser.parse_args()

    # Ensure env vars
    os.environ.setdefault("OPENAI_API_KEY", "sk-95fe62163fc44c258255f82e47837b5c")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    instances = load_dataset()
    print(f"Loaded {len(instances)} instances")

    if args.instance:
        instances = [i for i in instances if i["instance_id"] == args.instance]
    if args.repo:
        instances = [i for i in instances if i["repo"] == args.repo]
    if args.limit:
        instances = instances[:args.limit]

    print(f"Running {len(instances)} instances")

    results = []
    for idx, inst in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(instances)}] {inst['instance_id']}")
        try:
            r = run_single(inst, output_base, dry_run=args.dry_run, apply=args.apply)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"instance_id": inst["instance_id"], "success": False, "error": str(e)})

    # Summary
    ok = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"SUMMARY: {ok}/{len(results)} successful")
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        n = r.get("n_edits", 0)
        files = r.get("files", [])
        print(f"  [{status}] {r['instance_id']}: {n} edits, files={files}")

    (output_base / "summary.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
