"""
Run code generation with Oracle-style natural-detailed prompt format.
Generates before/after edit blocks, converts to git patch, saves everything to output/{instance_id}/.

Usage:
    conda run -n py38 python run_oracle.py --instance joke2k__faker-738
    conda run -n py38 python run_oracle.py --limit 5
"""
import os
import ast
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import REPOS_DIR, OUTPUT_DIR, MAX_CONTEXT_CHARS, LLM_MODEL_GENERATE, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_PLAN_MAX_TOKENS
from explorer.context_renderer import render_code_files
from explorer.history_patterns import render_commit_patterns
from dataclasses import asdict
from data_loader import load_dataset, load_instance, parse_problem_statement
from repo_manager import prepare_repo, get_file_content
from explorer.context_builder import ContextBuilder
from explorer.structural_analogies import discover_structural_analogies
from generator.llm_interface import call_llm_with_usage, LLMCallError
from generator.planned_generation import PlanContext, PlanError
from generator.state_feedback import STATE_CONTRACT
from generator.edit_recovery import normalize_markers, unique_edits, align_indent, expand_omissions, new_module_definitions, repair_prompt

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
- CRITICAL: The <code_before_edit> MUST be copied EXACTLY from the code shown in the <code> section above. Do NOT paraphrase, reformat, or rewrite the code. Copy it character-for-character including all whitespace and indentation. Any mismatch will cause the edit to fail.
- Never copy source omission placeholders into an edit. Choose one contiguous source excerpt.
- Preserve absolute indentation. New sibling methods must stay at the class level.
- Do not repeat an identical edit or emit unchanged before/after pairs.
- Keep <code_before_edit> snippets concise: include ONLY the lines being changed plus 5-10 lines of context before and after. Target 10-30 lines total. Do NOT include large blocks of unchanged code.
- Your edits must be COMPLETE. Include ALL necessary changes: new imports, class-level variables (__slots__ if applicable), __init__ modifications, new methods, bug fixes to existing methods, __init__.py exports, and .pyi type stub updates.
- Double-check that all method/function names used across your edits are consistent. If you define a method in one edit, use the exact same name when calling it in other edits.

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


def build_code_section(repo_dir: Path, instance: dict, extra_files: set = None,
                       ctx=None, budget: int = MAX_CONTEXT_CHARS) -> str:
    target_files = {item["file"] for item in instance.get("new_components") or []}
    target_files.update(d["file"] for d in parse_problem_statement(
        instance["problem_statement"]).get("definitions", []))
    ranked = [c.file for c in ctx.recalled_files] if ctx else []
    target_companions = []
    for file in sorted(target_files):
        for companion in [str(Path(file).parent / "__init__.py"), file + "i"]:
            if (repo_dir / companion).is_file():
                target_companions.append(companion)
    files = list(dict.fromkeys(sorted(target_files) + target_companions + ranked + sorted(extra_files or set())))
    companions = []
    for file in files:
        for companion in [str(Path(file).parent / "__init__.py"), file + "i"]:
            if (repo_dir / companion).is_file():
                companions.append(companion)
    files = list(dict.fromkeys(files + companions))
    return render_code_files(repo_dir, files, ctx, budget)


def _detect_structural_sibling_pattern(comp, init_method=None):
    """Adapt a closed, data-derived lifecycle candidate for the legacy prompt."""
    holder = type('ComponentHolder', (), {'components': [comp]})()
    candidates = discover_structural_analogies(holder, limit=1)
    if not candidates:
        return None
    candidate = candidates[0]
    symbols = {item['role']: item['symbol'].split('.')[-1]
               for item in candidate['observed_chain']}
    by_name = {node.method_name: node for node in comp.reference_call_chain}
    dispatcher = by_name.get(symbols.get('dispatcher'))
    worker = by_name.get(symbols.get('policy_query'))
    if dispatcher is None or worker is None:
        return None
    return {
        'candidate': candidate,
        'analogy_worker': worker,
        'dispatcher': dispatcher,
        'source_keyword': candidate['source_concept'],
        'target_keyword': candidate['target_concept'],
        'init_parsing_block': (_extract_init_parsing_pattern(
            init_method.body_source, candidate['source_concept']) if init_method else None),
    }


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


def _analyze_featurespace_diff_semantics(fsf, anchor_name: str, comp) -> str:
    """Analyze FeatureSpace diff to provide semantic guidance to the LLM.

    Instead of just showing the raw diff, we reason about:
    - What variables/attributes were moved or renamed
    - What the migration pattern means for the new feature
    - What specific action the LLM should take in this file
    """
    diff = fsf.diff_content
    entities = fsf.key_entities

    # Extract before→after patterns from the diff
    # Pattern: "- old.expr" / "+ new.expr" means a migration
    removed_lines = []
    added_lines = []
    for line in diff.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            removed_lines.append(line[1:].strip())
        elif line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:].strip())

    insights = []

    # Detect attribute migration: node.xxx → yyy.xxx
    migrations = {}
    for r_line in removed_lines:
        for m in re.finditer(r'(\w+\.)([a-zA-Z_]\w*)\s*=', r_line):
            old_prefix, attr = m.group(1), m.group(2)
            for a_line in added_lines:
                # Same attribute name but different prefix?
                if attr in a_line and old_prefix not in a_line:
                    for m2 in re.finditer(r'(\w+\.)([a-zA-Z_]\w*)\s*=', a_line):
                        if m2.group(2) == attr:
                            new_prefix = m2.group(1)
                            migrations[attr] = (old_prefix, new_prefix)

    if migrations:
        for attr, (old_p, new_p) in migrations.items():
            insights.append(
                f"Attribute `{attr}` was migrated from `{old_p}` to `{new_p}`. "
                f"Your new feature may need to use `{new_p}{attr}` instead of `{old_p}{attr}`."
            )

    # Detect variable renaming patterns
    renames = {}
    for r_line in removed_lines:
        for m in re.finditer(r'(?:self\.|_)([a-zA-Z_]\w*)\s*=', r_line):
            old_name = m.group(1)
            for a_line in added_lines:
                for m2 in re.finditer(r'(?:self\.|_)([a-zA-Z_]\w*)\s*=', a_line):
                    new_name = m2.group(1)
                    # Check if names share most words but differ (e.g., build_xxx vs _build_xxx)
                    old_words = set(re.findall(r'[a-z]+', old_name))
                    new_words = set(re.findall(r'[a-z]+', new_name))
                    if old_words and new_words and old_words == new_words and old_name != new_name:
                        renames[old_name] = new_name

    if renames:
        for old_n, new_n in renames.items():
            insights.append(
                f"Variable `{old_n}` was renamed to `{new_n}`. "
                f"Follow the new naming convention `{new_n}` in your implementation."
            )

    # Detect function call additions (new calls added in the anchor commit)
    new_calls = set()
    for a_line in added_lines:
        for m in re.finditer(r'(?:self\.|\.)([a-zA-Z_]\w*)\s*\(', a_line):
            new_calls.add(m.group(1))
    if new_calls and entities:
        overlap = new_calls & set(entities)
        if overlap:
            insights.append(
                f"New function calls introduced: {', '.join(overlap)}. "
                f"Your new method may need to call these as well."
            )

    if not insights:
        # Generic guidance based on entities
        if entities:
            return (
                f"This diff involves entities: {', '.join(entities[:6])}. "
                f"Check how these entities are used in the current code and ensure "
                f"your new feature interacts with them correctly."
            )
        return ""

    return " | ".join(insights)


def _trace_call_chain_introduction(repo_dir: Path,
                                    call_chain: list = None,
                                    base_commit: str = "",
                                    target_file: str = "",
                                    sibling_methods: dict = None) -> str:
    """Find the full commit diffs that introduced each method in the call chain.

    For each method in the reference call chain, finds its introduction commit
    and shows the FULL commit diff (all files). This reveals the complete
    "dynamic introduction process": what __init__ variables were added,
    how the method was structured, and how the dispatcher was modified
    to call it.

    When sibling methods have decorators (e.g. @property), also searches
    for commits that added those decorator patterns to the target file,
    which may be more recent and more informative than the introduction commit.
    """
    if not repo_dir or not base_commit or not call_chain:
        return ""

    import subprocess as _sp
    parts = []
    seen_commits = set()

    # Detect decorator patterns from siblings
    extra_searches = []
    if sibling_methods:
        for mname, method in sibling_methods.items():
            if getattr(method, 'is_property', False):
                extra_searches.append('@property')
                break  # One search for @property is enough

    for node in call_chain:
        mname = getattr(node, "method_name", "")
        if not mname:
            continue

        try:
            # Collect unique commits: introduction + decorator commits
            all_commits = []

            # 1) Find commits that introduced this method's definition
            result = _sp.run(
                ["git", "log", "--oneline", base_commit,
                 "-S", f"def {mname}", "--", target_file or "."],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=15,
            )
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    h = line.split()[0]
                    if h not in all_commits:
                        all_commits.append(h)

            # 2) Also search for decorator commits if applicable
            for kw in extra_searches:
                r2 = _sp.run(
                    ["git", "log", "--oneline", base_commit,
                     "-S", kw, "--", target_file or "."],
                    cwd=str(repo_dir), capture_output=True, text=True, timeout=15,
                )
                for line in r2.stdout.strip().split('\n'):
                    if line.strip():
                        h = line.split()[0]
                        if h not in all_commits:
                            all_commits.append(h)

            if not all_commits:
                continue

            # Take the MOST RECENT commit (first in list = newest)
            # This gives us the latest version including any decorator additions
            intro_commit = all_commits[0]
            if intro_commit in seen_commits:
                continue
            seen_commits.add(intro_commit)

            # Get body
            body = _sp.run(
                ["git", "log", "--format=%B", "-1", intro_commit],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
            ).stdout.strip()

            # Get author date
            date = _sp.run(
                ["git", "log", "--format=%ai", "-1", intro_commit],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
            ).stdout.strip()

            # Get diff for the TARGET FILE only (less noise)
            diff = _sp.run(
                ["git", "diff", f"{intro_commit}^..{intro_commit}",
                 "--", target_file or "."],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
            ).stdout

            diff_lines = diff.count('\n')
            if not diff or diff_lines >= 200:
                continue

            subject = body.split('\n')[0] if body else ""
            parts.append(
                f"  **`{mname}()`** definition-count change in `{intro_commit[:8]}` "
                f"({date[:10]}): {subject[:80]}"
            )
            parts.append(f"  Target file diff:")
            if len(diff) > 3000:
                diff = diff[:3000] + "\n  # ... (truncated)"
            for dline in diff.split('\n'):
                parts.append(f"  {dline}")
            parts.append("")

        except Exception:
            continue

    if not parts:
        return ""

    return (
        "**CALL CHAIN HISTORY** — selected historical changes to reference definitions "
        "(target-file diffs):\n"
        + "\n".join(parts)
    )

    if parts:
        intro = f"**HISTORICAL COMMITS** modifying `{target_file}` — how similar features were added:\n"
        return intro + "\n".join(parts)
    return ""


def _analyze_branch_conditions(comp) -> str:
    """Analyze branch conditions in the reference code and FeatureSpace diffs to guide LLM.

    Strategy:
    1. Scan all sibling methods + call chain for `else BINARY_XXX` transition patterns
    2. The transition condition (e.g., BINARY_INVALID from `else BINARY_INVALID`) is where
       the new method should be triggered
    3. Also scan for `== BINARY_XXX` comparisons to find where this condition is checked
    """
    comp_name = comp.name.split(".")[-1]

    NOISE_WORDS = {'TODO', 'FIXME', 'HACK', 'XXX', 'NOTE', 'WARN', 'ERROR',
                   'TRUE', 'FALSE', 'NONE', 'NULL', 'SELF', 'CLS', 'PASS',
                   'RAISE', 'RETURN', 'BREAK', 'CONTINUE', 'IMPORT', 'FROM',
                   'WITH', 'ASSERT', 'YIELD', 'DEL', 'GLOBAL', 'NONLOCAL',
                   'CONAN', 'CONANFILE', 'CONANINFO', 'CONAN_MANIFEST'}

    # Step 1: Scan ALL code we have (call chain + sibling methods) for:
    # a) `else BINARY_XXX` transition patterns (where the "failure" branch leads)
    # b) `== BINARY_XXX` comparison patterns (where conditions are checked)
    all_sources = {}  # method_name → body_source

    # From call chain
    for node in comp.reference_call_chain:
        all_sources[node.method_name] = node.body_source

    # From sibling methods (includes _evaluate_node which is NOT in the call chain)
    for mname, method in comp.sibling_methods.items():
        if mname not in all_sources:
            all_sources[mname] = method.body_source

    # Collect transition conditions (else BINARY_XXX patterns)
    transition_conditions = set()
    # Collect comparison conditions (== BINARY_XXX patterns)
    comparison_conditions = {}  # const → [(method_name, line)]

    for method_name, source in all_sources.items():
        for line in source.split('\n'):
            stripped = line.strip()

            # Transition: `else BINARY_XXX` (failure/alternative path)
            for m in re.finditer(r'else\s+([A-Z][A-Z_]{2,})\b', stripped):
                const_name = m.group(1)
                if const_name not in NOISE_WORDS:
                    transition_conditions.add(const_name)

            # Comparison: `== BINARY_XXX` (condition check)
            for m in re.finditer(r'==\s*([A-Z][A-Z_]{2,})', stripped):
                const_name = m.group(1)
                if const_name not in NOISE_WORDS:
                    if const_name not in comparison_conditions:
                        comparison_conditions[const_name] = []
                    comparison_conditions[const_name].append((method_name, stripped))

    if not transition_conditions:
        return ""

    # Step 2: Also scan FeatureSpace diffs
    for fsf in comp.feature_space_files:
        for line in fsf.diff_content.split('\n'):
            stripped = line.lstrip('+- ')
            for m in re.finditer(r'else\s+([A-Z][A-Z_]{2,})\b', stripped):
                const_name = m.group(1)
                if const_name not in NOISE_WORDS:
                    transition_conditions.add(const_name)

    # Step 3: Target = transition conditions
    # Note: A transition condition may NOT yet have a == comparison in the code
    # (because the gold patch ADDS the first == comparison for it).
    # So we use ALL transition conditions, not just those with existing comparisons.
    target_conditions = transition_conditions

    if not target_conditions:
        return ""

    # Build hints
    hints = []
    for const_name in sorted(target_conditions):
        usages = comparison_conditions.get(const_name, [])

        hint_lines = [
            f"Your new method `{comp_name}` should be triggered when `node.binary == {const_name}`.",
            f"The `{const_name}` state represents a case that your new method is designed to handle. "
            f"Add a new branch to check for this condition.",
        ]
        hint_lines.append(f"**Where `{const_name}` is set**:")
        for method_name, source in all_sources.items():
            for line in source.split('\n'):
                stripped = line.strip()
                if f'else {const_name}' in stripped:
                    hint_lines.append(f"  In `{method_name}()`: {stripped}")

        if usages:
            hint_lines.append(f"**Where `{const_name}` is checked**:")
            for method_name, line in usages[:3]:
                hint_lines.append(f"  In `{method_name}()`: {line}")

        # Determine the dispatcher method (the method that calls the anchor)
        dispatcher_method = ""
        if comp.reference_call_chain:
            anchor_name = comp.reference_call_chain[0].method_name
            # Find which sibling method calls the anchor
            for mname, callees in comp.calling_patterns.items():
                if anchor_name in callees:
                    dispatcher_method = mname
                    break
            # Fallback: the method with most condition branches is likely the dispatcher
            if not dispatcher_method:
                for mname in all_sources:
                    if mname != anchor_name and 'binary' in all_sources[mname].lower():
                        dispatcher_method = mname
                        break

        # Determine existing condition branches (what NOT to use)
        existing_branches = sorted(comparison_conditions.keys() - {const_name})

        if dispatcher_method:
            hint_lines.append(
                f"\n**Where to add**: In `{dispatcher_method}()`, add a NEW `if` block "
                f"for `{const_name}`"
            )
        else:
            hint_lines.append(
                f"\n**Where to add**: Add a NEW `if` block for `{const_name}` "
                f"in the method that orchestrates the binary evaluation flow"
            )

        if existing_branches:
            hint_lines.append(
                f"after the existing `{'`, `'.join(existing_branches[:3])}` branches."
            )
        hint_lines.append(f"```python")
        hint_lines.append(f"if node.binary == {const_name} and <condition_for_your_method>:")
        hint_lines.append(f"    # call your new method here")
        hint_lines.append(f"```")

        # Warn about similar conditions
        for existing_const in comparison_conditions:
            if existing_const != const_name:
                existing_words = set(re.findall(r'[a-z]+', existing_const.lower()))
                target_words = set(re.findall(r'[a-z]+', const_name.lower()))
                if existing_words & target_words:
                    hint_lines.append(
                        f"\n**WARNING: Do NOT use `{existing_const}`** — that branch already "
                        f"has its own logic. Your new method is for when the node is `{const_name}` "
                        f"(cannot build), and should try to BUILD a compatible instead."
                    )

        hints.append("\n".join(hint_lines))

    return "\n\n".join(hints) if hints else ""


def build_analysis_section(ctx, base_commit: str = "") -> str:
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
    history_mode = ctx.recall_diagnostics.get("history_mode", "legacy")
    grouped_history = render_commit_patterns(getattr(ctx, "history_patterns", []),
                                            budget=min(8000, MAX_CONTEXT_CHARS // 10))

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

            # Check if this class uses __slots__ - warn the LLM
            if init_method and "__slots__" in init_method.body_source:
                parts.append(
                    "**CRITICAL WARNING**: This class uses `__slots__`. If your new feature "
                    "adds any new instance attributes (e.g. `self.new_attr`), you MUST:\n"
                    "  1. Add the attribute name to the `__slots__` list/tuple\n"
                    "  2. Initialize it in `__init__`\n"
                    "  Without both steps, the attribute will not be storable on instances."
                )
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

        # A closed repository lifecycle is evidence to verify, never a mandatory template.
        if structural_pattern:
            sp = structural_pattern
            src_kw = sp['source_keyword']
            tgt_kw = sp['target_keyword']
            worker = sp['analogy_worker']
            candidate = sp['candidate']

            if sp['init_parsing_block']:
                parts.append("**Repository lifecycle analogy candidate (verify before use)**:")
                parts.append(
                    f"Current code connects `{src_kw}` input forms to state, "
                    f"`{worker.method_name}()`, and `{sp['dispatcher'].method_name}()`. "
                    f"The requested component introduces the distinct token `{tgt_kw}`.\n"
                )
                parts.append("Observed chain: " + json.dumps(
                    candidate['observed_chain'], ensure_ascii=False))
                parts.append(f"Relevant `{src_kw}` parsing excerpt:")
                parts.append(f"```python")
                parts.append(sp['init_parsing_block'])
                parts.append(f"```\n")

            parts.append(
                "Use this analogy only if the request needs the target concept to participate in the same "
                "lifecycle. If it applies, map every role while preserving distinct state; adapt names and "
                "representation to current conventions. Do not copy source-specific conditions or force this "
                "analogy onto other components.\n"
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
                # Prepend decorators to body source for accurate display
                decorator_set = set(getattr(method, 'decorators', []))
                if method.is_property:
                    decorator_set.add('property')
                if method.is_staticmethod:
                    decorator_set.add('staticmethod')
                if method.is_classmethod:
                    decorator_set.add('classmethod')
                decorator_lines = '\n'.join(f'    @{d}' for d in decorator_set)
                body = method.body_source
                if decorator_lines:
                    body = decorator_lines + '\n' + body
                if len(body) > 3000:
                    body = body[:3000] + "\n    # ... (truncated)"
                parts.append(body)
            parts.append("")

        # Compact API reference: all callable methods in this class with signatures
        if comp.sibling_methods:
            parts.append("**Available methods in this class** (call these via `self.xxx()` if they provide needed functionality):")
            for mname, method in comp.sibling_methods.items():
                if mname == comp_short:
                    continue
                # Extract signature from def line
                sig = method.body_source.split('\n')[0] if method.body_source else f"def {mname}(self)"
                # Prepend decorator annotations
                decorator_set = set(getattr(method, 'decorators', []))
                if method.is_property:
                    decorator_set.add('property')
                if method.is_staticmethod:
                    decorator_set.add('staticmethod')
                if method.is_classmethod:
                    decorator_set.add('classmethod')
                decorator_prefix = ' '.join(f'@{d}' for d in decorator_set)
                sig_display = f"{decorator_prefix} {sig.strip()}" if decorator_prefix else sig.strip()
                # Extract return annotation if present
                ret = ""
                if "->" in sig:
                    ret = sig.split("->")[-1].strip().rstrip(':')
                # Docstring one-liner
                doc_first = ""
                if method.docstring:
                    doc_first = method.docstring.split('\n')[0].strip()[:80]
                parts.append(f"  `{sig_display}`")
                if doc_first:
                    parts.append(f"    {doc_first}")
                if ret:
                    parts.append(f"    Returns: {ret}")
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
                worker = sp['analogy_worker']
                dispatcher = sp['dispatcher']

                parts.append("**POTENTIAL STRUCTURAL ANALOGY** — complete source lifecycle:")
                parts.append(
                    f"`{dispatcher.method_name}()` (dispatcher) calls `{worker.method_name}()` (worker). "
                    f"Use this chain as evidence to evaluate `{comp.name.split('.')[-1]}`, not as a body to copy.\n"
                )

                parts.append(
                    f"Method `{worker.class_name}.{worker.method_name}()` "
                    f"(observed source policy query):"
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

        # Branch condition analysis: help LLM distinguish similar conditions
        if history_mode == "legacy" and comp.reference_call_chain and comp.feature_space_files:
            branch_hints = _analyze_branch_conditions(comp)
            if branch_hints:
                parts.append("**CRITICAL: Branch condition guidance**:")
                parts.append(branch_hints)
                parts.append("")

        # Inter-component dependencies
        if comp.depends_on_components:
            parts.append(f"**INTER-COMPONENT DEPENDENCY**: This component CALLS other new components: {comp.depends_on_components}")
            parts.append(f"You MUST make `{comp.name}` call `self.{comp.depends_on_components[0]}()` (or similar) — do NOT reimplement the logic.\n")

        # Reference commit: show the focused diff that introduced the anchor method
        if history_mode == "legacy" and comp.feature_space_files and comp.reference_call_chain:
            anchor_name = comp.reference_call_chain[0].method_name
            anchor_commit = comp.feature_space_files[0].anchor_commit
            # Get clean diff for the anchor file (like a gold patch)
            try:
                import subprocess as _sp
                diff = _sp.run(
                    ["git", "diff", f"{anchor_commit}^..{anchor_commit}", "--", comp.target_file],
                    cwd=str(ctx.repo_dir), capture_output=True, text=True, timeout=10,
                ).stdout
                # Only show if focused (gold-patch-like: < 80 lines, multiple small edits OK)
                diff_lines = diff.count('\n')
                if diff and diff_lines < 120:
                    parts.append(
                        f"**REFERENCE COMMIT DIFF** — historical target-file changes retrieved using `{anchor_name}` "
                        f"(commit `{anchor_commit[:8]}`):"
                    )
                    parts.append(
                        f"Compare this historical pattern with the current requirement and base code. "
                        f"Use only the responsibilities that apply to the new feature.\n"
                    )
                    parts.append(f"```diff\n{diff}\n```\n")
            except Exception:
                pass

        # FeatureSpace: co-modified files discovered via anchor's git history
        if history_mode == "legacy" and comp.feature_space_files:
            anchor_name = comp.reference_call_chain[0].method_name if comp.reference_call_chain else "anchor"
            parts.append(f"**FEATURESPACE — Candidate integration files from base-ancestor history**:")
            parts.append(
                f"Your reference anchor `{anchor_name}` was introduced in commit "
                f"`{comp.feature_space_files[0].anchor_commit[:8]}`. "
                f"That commit ALSO modified the files listed below. Since your new method follows "
                f"`{anchor_name}`'s pattern, these files likely need analogous edits. "
                f"Check whether the current feature requires changes there; historical co-modification alone does not require an edit.\n"
            )
            for fsf in comp.feature_space_files:
                parts.append(f"- `{fsf.file}`: {fsf.diff_summary}")
                # Semantic analysis of the diff
                semantic = _analyze_featurespace_diff_semantics(fsf, anchor_name, comp)
                if semantic:
                    parts.append(f"  **What this diff means**: {semantic}")
                parts.append(f"  Historical diff (what changed when anchor was introduced):")
                parts.append(f"  ```")
                for diff_line in fsf.diff_content.split('\n'):
                    parts.append(f"  {diff_line}")
                parts.append(f"  ```\n")
            fs_files = [fsf.file for fsf in comp.feature_space_files]
            parts.append(
                "**ACTION REQUIRED**: Review the current content of these files in the <code> section. "
                "Determine which files the current feature actually requires changing, then generate `<edit>` blocks "
                f"for: {', '.join(fs_files)}.\n"
            )

        # Call chain introductions: show how each reference method was added (full commit)
        if history_mode == "legacy" and ctx and ctx.repo_dir and base_commit:
            try:
                if comp.reference_call_chain:
                    hist = _trace_call_chain_introduction(
                        ctx.repo_dir,
                        call_chain=comp.reference_call_chain,
                        base_commit=base_commit,
                        target_file=comp.target_file,
                        sibling_methods=comp.sibling_methods,
                    )
                elif comp.sibling_methods:
                    # Fallback: trace how each sibling was introduced
                    hist = _trace_call_chain_introduction(
                        ctx.repo_dir,
                        call_chain=[type('Node', (), {'method_name': m})()
                                    for m in comp.sibling_methods if m != comp.name.split('.')[-1]],
                        base_commit=base_commit,
                        target_file=comp.target_file,
                        sibling_methods=comp.sibling_methods,
                    )
                if hist:
                    parts.append(hist + "\n")
            except Exception:
                pass

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
                    f"**Similar existing functions** (name similarity proposes candidates only; "
                    f"the closed lifecycle above supplies the stronger evidence):"
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

    # Domain-specific hints based on request content
    request_lower = ""
    for comp in ctx.components:
        request_lower += " " + comp.name.lower()

    if "metadata" in request_lower and "routing" in request_lower:
        parts.append("**METADATA ROUTING PATTERN**: This feature requires implementing scikit-learn's metadata routing. The correct pattern is:")
        parts.append("1. Import from `utils._metadata_requests`: `MetadataRouter`, `MethodMapping`, `process_routing`, `_routing_enabled`")
        parts.append("2. In `get_metadata_routing()`: return `MetadataRouter(owner=self.__class__.__name__).add(method=method, router=self._get_regressor().get_metadata_routing())`")
        parts.append("3. In `fit()`: add `routed_params = process_routing(self, \"fit\", **kwargs)` and pass routed params to internal estimator")
        parts.append("4. In `predict()`: similarly use `process_routing(self, \"predict\", **kwargs)`")
        parts.append("5. Use `MethodMapping().add(caller=\"fit\", callee=\"fit\").add(caller=\"predict\", callee=\"predict\")` for method mapping")
        parts.append("")

    selected = []
    remaining = max(1000, MAX_CONTEXT_CHARS // 4) - len(grouped_history)
    for part in parts:
        if len(part) + 1 <= remaining:
            selected.append(part)
            remaining -= len(part) + 1
    return grouped_history + "\n".join(selected)





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

    # Collect extra files from explorer context (base classes, FeatureSpace, etc.)
    extra_files = set()
    if ctx:
        for comp in ctx.components:
            for base_name, (base_file, base_module) in comp.base_classes.items():
                extra_files.add(base_file)
            # FeatureSpace: files discovered via anchor's git commit history
            for fsf in comp.feature_space_files:
                extra_files.add(fsf.file)

    if ctx:
        extra_files.update(candidate.file for candidate in ctx.recalled_files)

    # Build analysis section from explorer
    analysis_section = ""
    if ctx:
        base_commit = instance.get("base_commit", "")
        analysis_section = build_analysis_section(ctx, base_commit=base_commit)

    # Build FeatureSpace additions to definitions
    fs_defs_addition = ""
    if ctx:
        fs_files_all = set()
        for comp in ctx.components:
            for fsf in comp.feature_space_files:
                fs_files_all.add(fsf.file)
        if fs_files_all:
            fs_defs_addition = "\n\n**Additional candidate files to inspect (discovered via FeatureSpace analysis):**\n"
            for f in sorted(fs_files_all):
                fs_defs_addition += f"- {f}\n"

    recall_section = ""
    if ctx and ctx.recalled_files:
        recall_section = "\nRetrieved integration candidates (inspect evidence before editing):\n"
        for candidate in ctx.recalled_files:
            recall_section += "- %s: %s\n" % (candidate.file, "; ".join(candidate.reasons[:2]))
            for location in candidate.locations:
                updates = [u for u in location["state_updates"] if u["field"] in location["related_fields"]]
                for update in updates[:2]:
                    recall_section += "  %s:%d in %s: %s [%s]\n" % (
                        candidate.file, update["line"], location["symbol"], update["statement"],
                        "; ".join(update["guards"]) or "unconditional")
        recall_section += (
            "When introducing alternate configurations or fallback paths, inspect whether producer state "
            "must be reset or recomputed before these consumers run. Keep distinct validation/error flags "
            "and their branch conditions distinct; do not merge flags just because their names are similar.\n")
    # Render code last, using the remaining budget of the complete prompt.
    prompt = f"""You will be provided with code files and a feature request which requires new code to add to the codebase.
<request>
{request_text}
</request>

There are several new functions or classes that need to be implemented, using the definitions below:
<definitions>
{defs_text}
</definitions>
{fs_defs_addition}
{analysis_section}
{recall_section}

<code>
__FUNCGEN_CODE_SECTION__
</code>

{STATE_CONTRACT}
{OUTPUT_FORMAT_INSTRUCTION}"""

    marker = "__FUNCGEN_CODE_SECTION__"
    budget = MAX_CONTEXT_CHARS - (len(prompt) - len(marker))
    if budget < 1000:
        raise ValueError("Feature request and analysis exceed the configured context budget")
    code = build_code_section(repo_dir, instance, extra_files, ctx=ctx, budget=budget)
    return prompt.replace(marker, code)


def visible_code_files(prompt: str) -> List[str]:
    code = prompt.split("\n<code>\n", 1)[1].split("\n</code>", 1)[0]
    return re.findall(r"^\[start of ([^\n\]]+)\]$", code, re.MULTILINE)


def parse_edits(response: str) -> List[Dict]:
    """Recover fully delimited snippets even when the outer edit closer is absent."""
    response = normalize_markers(response)
    raw = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    if raw.startswith('{'):
        try:
            parsed = json.loads(raw)
            entries = parsed.get('edits', [])
            if isinstance(entries, list) and entries and all(isinstance(e, dict) and
                    all(isinstance(e.get(k), str) for k in ['file_path', 'before', 'after']) for e in entries):
                return unique_edits([{k: e[k] for k in ['file_path', 'before', 'after']} for e in entries])
        except (ValueError, AttributeError):
            pass
    edits = []
    blocks = re.findall(r"<edit>(.*?)(?:</edit>|(?=<edit>)|$)", response, re.DOTALL)

    def section(block, phase):
        pattern = (r"[\[<]start[^\n]*?\b" + phase +
                   r"\b[^\n]*?\bin ([^\]>\n]+)[\]>][ \t]*\r?\n(.*?)"
                   r"[\[<]end[^\n]*?\b" + phase +
                   r"\b[^\n]*?\bin ([^\]>\n]+)[\]>]")
        match = re.search(pattern, block, re.DOTALL)
        if not match or match.group(1).strip() != match.group(3).strip():
            return None
        body = match.group(2)
        if body.endswith("\n"):
            body = body[:-1]
        return match.group(1).strip(), body

    for block in blocks:
        before, after = section(block, "before"), section(block, "after")
        if before and after and before[0] == after[0]:
            edits.append({"file_path": before[0], "before": before[1], "after": after[1]})
            continue
        # Alternate XML markers still require complete before and after sections.
        file = re.search(r"<file_path>(.*?)</file_path>", block, re.DOTALL)
        before = re.search(r"<start_of_snippet_before_editing>[ \t]*\n(.*?)(?:<end_of_snippet_before_editing>|</start_of_snippet_before_editing>)", block, re.DOTALL)
        after = re.search(r"<start_of_snippet_after_editing>[ \t]*\n(.*?)(?:<end_of_snippet_after_editing>|</start_of_snippet_after_editing>)", block, re.DOTALL)
        if file and before and after:
            edits.append({"file_path": file.group(1).strip(),
                          "before": before.group(1).rstrip("\n"),
                          "after": after.group(1).rstrip("\n")})
    return edits


def _fix_edit_name_consistency(edits: List[Dict]) -> List[Dict]:
    """Fix cross-edit method name inconsistencies.

    Detects cases where one edit defines a method (e.g. get_languages_for_item)
    but another edit calls it by a different name (e.g. _languages_for_item).
    Common pattern: LLM uses _method in one edit and get_method in another.
    """
    # Collect all method names defined in 'after' sections
    defined_names = {}  # name -> (edit_index, file_path)
    for i, edit in enumerate(edits):
        for match in re.finditer(r'def (\w+)\s*\(', edit.get("after", "")):
            defined_names[match.group(1)] = (i, edit["file_path"])

    # Check for mismatches in method calls across edits
    corrections = {}  # wrong_name -> correct_name
    for i, edit in enumerate(edits):
        after_text = edit.get("after", "")
        for match in re.finditer(r'(?:self\.|cls\.)(\w+)\s*\(', after_text):
            called = match.group(1)
            if called not in defined_names:
                # Look for similar defined names
                for name in defined_names:
                    # Pattern: _method vs get_method
                    if called.lstrip('_') == name.replace('get_', '').lstrip('_'):
                        # Only correct if defined in a different edit
                        def_idx, _ = defined_names[name]
                        if def_idx != i:
                            corrections[called] = name
                            break

    # Apply corrections
    if corrections:
        for edit in edits:
            for wrong, correct in corrections.items():
                edit['after'] = edit['after'].replace(f'self.{wrong}(', f'self.{correct}(')
                edit['after'] = edit['after'].replace(f'cls.{wrong}(', f'cls.{correct}(')

    return edits


def _deduplicate_edit_lines(edits: List[Dict]) -> List[Dict]:
    """Remove consecutive duplicate lines from edit 'after' sections.

    LLMs frequently produce duplicate lines like:
      return True
      return True
    or:
      raise ConanException("...")
      raise ConanException("...")
    This post-processing removes them.
    """
    for edit in edits:
        lines = edit['after'].split('\n')
        deduped = []
        for i, line in enumerate(lines):
            # Skip if this line is identical to the previous line
            if deduped and line.strip() == deduped[-1].strip() and line.strip():
                continue
            deduped.append(line)
        edit['after'] = '\n'.join(deduped)
    return edits


def edits_to_patch(edits: List[Dict], repo_dir: Path) -> str:
    """Convert edit blocks to a unified diff patch.

    For each edit: read the actual file from repo, use LLM 'before' only as a
    positional hint to locate the region, extract the REAL content, replace with
    LLM 'after' (with indent alignment), then generate unified diff.
    """
    import difflib
    from collections import OrderedDict

    file_edits = OrderedDict()
    for edit in unique_edits(edits):
        fp = edit["file_path"]
        if Path(fp).is_absolute() or '..' in Path(fp).parts or '.git' in Path(fp).parts:
            raise ValueError('Invalid edit path: ' + fp)
        file_edits.setdefault(fp, []).append(edit)

    patch_parts = []

    for fp, fp_edits in file_edits.items():
        full_path = repo_dir / fp
        try:
            full_path.resolve().relative_to(repo_dir.resolve())
        except ValueError:
            raise ValueError("Edit path leaves the repository: %s" % fp)

        if not full_path.exists():
            combined_after = "\n".join(e["after"] for e in fp_edits)
            if fp.endswith(".py"):
                compile(combined_after, fp, "exec")
            after_lines = combined_after.splitlines()
            if not after_lines:
                patch_parts.append(f"diff --git a/{fp} b/{fp}\nnew file mode 100644\nindex 0000000..e69de29\n")
                continue
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
        orig_lines = original.splitlines()
        mod_lines = list(orig_lines)
        any_applied = False

        for edit in fp_edits:
            current_text = "\n".join(mod_lines)
            if fp.endswith('.py') and edit['before'].strip() and edit['before'].strip() not in current_text:
                additions = new_module_definitions(edit['before'], edit['after'], current_text)
                if additions:
                    mod_lines += ['', ''] + additions.splitlines()
                    any_applied = True
                    continue
            before_text, after_text = expand_omissions(edit['before'], edit['after'], '\n'.join(mod_lines))
            before_hint = before_text.strip()

            # Skip no-op edits (before == after after normalizing whitespace)
            if before_hint and before_hint == after_text.strip():
                continue

            if not before_hint:
                if not original and len(fp_edits) == 1:
                    mod_lines = after_text.splitlines()
                    any_applied = True
                    continue
                raise ValueError("Existing-file edit needs a before snippet: %s" % fp)

            # Find the region in the current (modified) text
            region = _locate_region(mod_lines, before_hint)
            if region is None:
                raise ValueError("Edit cannot be uniquely located in %s" % fp)

            start, end = region
            # Align indentation of after to match original region
            after_aligned = _align_indent(mod_lines[start:end], after_text.splitlines())
            last_hint = before_hint.splitlines()[-1].strip()
            last_source = mod_lines[end - 1].strip()
            signature_completed = False
            if (re.match(r'(?:async\s+def|def|class)\s+', last_hint) and last_source.startswith(last_hint)
                    and len(last_source) > len(last_hint) and after_aligned
                    and after_aligned[-1].rstrip().endswith(last_hint)):
                after_aligned[-1] += last_source[len(last_hint):]
                signature_completed = True
            # A short unchanged context anchor may stop inside the final line.
            # Preserve its physical suffix instead of deleting a signature's arguments.
            current_text = "\n".join(mod_lines)
            if current_text.count(before_hint) == 1:
                match_end = current_text.index(before_hint) + len(before_hint)
                line_end = current_text.find("\n", match_end)
                suffix = current_text[match_end:line_end if line_end >= 0 else len(current_text)]
                if suffix and not signature_completed:
                    if after_aligned and after_aligned[-1].rstrip().endswith(last_hint):
                        after_aligned[-1] += suffix
                    else:
                        raise ValueError("Before snippet ends inside a source line; include the entire final line")
            mod_lines = mod_lines[:start] + after_aligned + mod_lines[end:]
            any_applied = True

        if not any_applied:
            continue

        modified = "\n".join(mod_lines) + ("\n" if original.endswith("\n") and mod_lines else "")
        if fp.endswith(".py"):
            try:
                compile(original, fp, "exec")
            except SyntaxError:
                pass  # Preserve compatibility with repositories using older grammars.
            else:
                compile(modified, fp, "exec")
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True), modified.splitlines(keepends=True),
            fromfile=f"a/{fp}", tofile=f"b/{fp}",
        ))
        if len(diff_lines) > 2:
            body = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                           for line in diff_lines)
            patch_parts.append(f"diff --git a/{fp} b/{fp}\n" + body)

    return "\n".join(patch_parts)


def _min_indent(lines: list) -> int:
    mins = []
    for l in lines:
        s = l.lstrip()
        if s:
            mins.append(len(l) - len(s))
    return min(mins) if mins else 0


def _align_indent(orig_lines: list, after_lines: list) -> list:
    return align_indent(orig_lines, after_lines)


def _locate_region(lines: list, hint_stripped: str) -> Optional[Tuple[int, int]]:
    """Locate the region in lines that best matches hint_stripped.

    Returns (start_line, end_line) or None.
    """
    import difflib as _dl

    # Exact substring match first
    text = "\n".join(lines)
    hint = hint_stripped
    if hint in text:
        if text.count(hint) != 1:
            return None
        pos = text.index(hint)
        start = text[:pos].count("\n")
        end = start + hint.count("\n") + 1  # +1: n+1 lines have n newlines
        return start, end

    # Fuzzy: sliding window
    hint_lines = hint_stripped.splitlines()
    significant_hint = [line.strip() for line in hint_lines if line.strip()]
    significant_source = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]
    canonical = []
    for i in range(len(significant_source) - len(significant_hint) + 1):
        window = significant_source[i:i + len(significant_hint)]
        if not significant_hint or any(actual != wanted for (_, actual), wanted in zip(window[:-1], significant_hint[:-1])):
            continue
        last, wanted = window[-1][1], significant_hint[-1]
        partial_signature = bool(re.match(r'(?:async\s+def|def|class)\s+', wanted)) and last.startswith(wanted)
        if last == wanted or partial_signature:
            canonical.append((window[0][0], window[-1][0] + 1))
    if len(canonical) == 1:
        return canonical[0]
    if len(canonical) > 1:
        return None
    n = len(hint_lines)
    if n == 0:
        return None

    best_idx = -1
    best_ratio = 0.0
    step = 1 if n < 50 else max(1, n // 10)

    for i in range(0, len(lines) - n + 1, step):
        window = "\n".join(lines[i:i + n])
        ratio = _dl.SequenceMatcher(None, window, hint_stripped).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_idx >= 0 and best_ratio >= 0.85:
        return best_idx, best_idx + n

    # Bookend matching for long hints (>20 lines)
    # Use first 10 and last 10 lines as anchors
    if n > 20:
        head = '\n'.join(hint_lines[:10])
        tail = '\n'.join(hint_lines[-10:])

        candidates = []
        for i in range(len(lines) - n + 1):
            window_head = '\n'.join(lines[i:i+10])
            head_ratio = _dl.SequenceMatcher(None, window_head, head).ratio()
            if head_ratio > 0.8:
                window_tail = '\n'.join(lines[i+n-10:i+n])
                tail_ratio = _dl.SequenceMatcher(None, window_tail, tail).ratio()
                if tail_ratio > 0.7:
                    candidates.append((i, head_ratio + tail_ratio))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[0] + n

    return None


def apply_edits_to_repo(edits: List[Dict], repo_dir: Path, dry_run: bool = False) -> Dict[str, str]:
    patch = edits_to_patch(edits, repo_dir)
    if not patch.strip():
        raise ValueError("Parsed edits produced an empty patch")
    check = subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", "-"],
                           input=patch, text=True, capture_output=True, cwd=repo_dir, timeout=30)
    if check.returncode:
        raise ValueError("Generated patch does not apply: " + check.stderr)
    outcomes = {e["file_path"]: "modified (patch)" if (repo_dir / e["file_path"]).exists()
                else "created" for e in edits if not (repo_dir / e["file_path"]).exists() or e["before"].strip() != e["after"].strip()}
    if not dry_run:
        subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=patch,
                       text=True, capture_output=True, cwd=repo_dir, timeout=30, check=True)
    return outcomes


def run_single(instance: dict, output_base: Path, dry_run: bool = False,
               apply: bool = False, repo_dir_override: Optional[Path] = None,
               recall_only: bool = False, recall_mode: str = "expanded",
               max_recalled_files: int = 12, history_mode: str = "patterns",
               generation_mode: str = "plan-edit", plan_max_tokens: int = LLM_PLAN_MAX_TOKENS,
               planner_model: Optional[str] = None, planner_json_mode: str = "api",
               max_repair_attempts: int = 2, repair_max_tokens: int = 16384) -> dict:
    """Run oracle-style generation for one instance."""
    iid = instance["instance_id"]
    out_dir = output_base / iid
    out_dir.mkdir(parents=True, exist_ok=True)
    if generation_mode not in {'single', 'plan-edit'} or planner_json_mode not in {'api', 'prompt'}:
        raise ValueError('Invalid generation or planner JSON mode')
    if plan_max_tokens <= 0 or repair_max_tokens <= 0 or max_repair_attempts < 0:
        raise ValueError('Token budgets must be positive and repair limit nonnegative')
    # Re-running an instance must never submit a successful previous patch
    # after the new plan or generation failed.
    for stale in ['generation_fallback.json', 'repair_history.json', 'generated.patch', 'edits.json', 'response.md', 'plan.json', 'plan_response.md',
                  'plan_validation.json', 'planning_prompt.md', 'generation_prompt.md', 'generation_recall.json',
                  'llm_usage.json', 'parse_failure.txt', 'apply_results.json']:
        (out_dir / stale).unlink(missing_ok=True)

    for pattern in ['response_attempt_*.md', 'repair_prompt_*.md']:
        for stale in out_dir.glob(pattern):
            stale.unlink()

    result = {
        "instance_id": iid,
        "repo": instance["repo"],
        "success": False,
        "n_edits": 0,
        "files": [],
        "timing": {},
        "generation_mode": generation_mode,
        "pipeline_attempts": 1,
        "max_final_candidates": 1,
        "submitted_patch_count": 0,
        "stage": "analysis",
        "max_repair_attempts": max_repair_attempts,
        "repair_attempts": 0,
        "planning_fallback": False,
    }
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2))

    import time
    t_start = time.time()

    # Explicit source archives enable read-only recall experiments without Git checkout.
    print("[1/4] Preparing repo...")
    repo_dir = Path(repo_dir_override) if repo_dir_override else prepare_repo(
        iid, instance["base_commit"], instance["repo"])
    if not repo_dir.is_dir():
        raise ValueError("Repository directory does not exist: %s" % repo_dir)

    result["timing"]["checkout"] = time.time() - t_start

    # Step 2: Run deterministic analysis + Build prompt
    print(f"[2/4] Running AST analysis & building prompt...")
    t0 = time.time()
    parsed = parse_problem_statement(instance["problem_statement"])
    builder = ContextBuilder(repo_dir, recall_mode=recall_mode, max_recalled_files=max_recalled_files,
                             history_mode=history_mode)
    ctx = builder.build_context(
        instance_id=iid,
        problem_statement=instance["problem_statement"],
        definitions=parsed["definitions"],
        has_extra_changes=parsed["has_extra_changes"],
        new_components_raw=instance.get("new_components"),
        base_commit=instance.get("base_commit", ""),
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
        fs_files = [fsf.file for fsf in comp.feature_space_files]
        fs_info = f", featurespace={fs_files}" if fs_files else ""
        print(f"    {comp.name}: chain={chain}{deps}{init_need}{fs_info}")
    print(f"  Prompt: {len(prompt)} chars")

    recall = {
        "instance_id": iid,
        "mode": recall_mode,
        "target_files": sorted({c.target_file for c in ctx.components}),
        "candidates": [asdict(candidate) for candidate in ctx.recalled_files],
        "history_files": sorted({f.file for c in ctx.components for f in c.feature_space_files}),
        "diagnostics": ctx.recall_diagnostics,
        "history_patterns": [asdict(pattern) for pattern in ctx.history_patterns],
        "visible_history_commits": re.findall(r"^COMMIT ([0-9a-f]{12}) —", prompt, re.MULTILINE),
        "visible_code_files": visible_code_files(prompt),
        "prompt_chars": len(prompt),
    }
    plan_context = None
    if generation_mode == 'plan-edit':
        plan_context = PlanContext(repo_dir, ctx, builder.repository_index)
        try:
            planning_prompt = plan_context.planning_prompt(instance)
            (out_dir / 'planning_prompt.md').write_text(planning_prompt)
        except Exception as error:
            result['planning_fallback'] = True
            (out_dir / 'generation_fallback.json').write_text(json.dumps({'stage': 'planning_context', 'reason': str(error)}))
            plan_context = None
    (out_dir / "recall.json").write_text(json.dumps(recall, indent=2))
    if recall_only:
        result.update(success=True, mode="recall_only", recall=recall, stage='recall_complete', llm_calls=0)
        result["timing"]["total"] = time.time() - t_start
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    # Step 3: One workflow, optionally a planning call plus one code call.
    print(f"[3/4] Calling LLM...")
    calls = []
    def save_usage():
        usage = {'pipeline_attempts': 1, 'max_final_candidates': 1,
                 'logical_calls': len(calls), 'provider_attempts': sum(c['attempts'] for c in calls),
                 'calls': calls}
        for field in ['input_tokens', 'output_tokens', 'total_tokens']:
            values = [c.get(field) for c in calls]
            usage[field] = sum(values) if values and all(v is not None for v in values) else None
            usage['reported_' + field] = sum(v for v in values if v is not None)
        usage['seconds'] = round(sum(c['seconds'] for c in calls), 6)
        result['llm_calls'] = len(calls)
        result['llm_usage'] = usage
        (out_dir / 'llm_usage.json').write_text(json.dumps(usage, indent=2))
    def invoke(stage, text, **kwargs):
        result['stage'] = stage
        try:
            reply = call_llm_with_usage(text, **kwargs)
        except LLMCallError as error:
            calls.append(dict(error.metrics, stage=stage, prompt_chars=len(text), max_tokens=kwargs['max_tokens']))
            save_usage()
            raise
        calls.append(dict(reply.metrics(), stage=stage, prompt_chars=len(text), max_tokens=kwargs['max_tokens']))
        save_usage()
        return reply
    t0 = time.time()
    direct_prompt = prompt
    try:
        if plan_context:
            try:
                planned = invoke('planning', planning_prompt,
                                 system='You plan repository changes from supplied evidence. Return only the requested JSON object, no code.',
                                 model=planner_model or LLM_MODEL_GENERATE, max_tokens=plan_max_tokens,
                                 temperature=LLM_TEMPERATURE, json_mode=planner_json_mode == 'api')
                (out_dir / 'plan_response.md').write_text(planned.text)
                result['timing']['planning'] = planned.seconds
                result['stage'] = 'plan_validation'
                if planned.finish_reason in {'length', 'max_tokens'}:
                    raise PlanError('Planner output was truncated')
                plan, validation = plan_context.parse_validate(planned.text)
                (out_dir / 'plan.json').write_text(json.dumps(plan, indent=2))
                (out_dir / 'plan_validation.json').write_text(json.dumps(validation, indent=2))
                result['stage'] = 'generation_context'
                prompt, generation_recall = plan_context.generation_prompt(instance, plan, OUTPUT_FORMAT_INSTRUCTION)
                (out_dir / 'generation_recall.json').write_text(json.dumps(generation_recall, indent=2))
            except Exception as error:
                reason = {'stage': result['stage'], 'reason': str(error), 'error_type': type(error).__name__}
                result['planning_fallback'] = True
                if isinstance(error, PlanError):
                    (out_dir / 'plan_validation.json').write_text(json.dumps(error.report, indent=2))
                (out_dir / 'generation_fallback.json').write_text(json.dumps(reason, indent=2))
                prompt = direct_prompt
        (out_dir / 'generation_prompt.md').write_text(prompt)
        (out_dir / 'prompt.md').write_text(prompt)
        reply = invoke('generation', prompt,
                       system="You are an expert Python developer. Output exact code edits in the specified format.",
                       model=LLM_MODEL_GENERATE, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
        history = []
        for attempt in range(max_repair_attempts + 1):
            response = reply.text
            (out_dir / 'response.md').write_text(response)
            (out_dir / ('response_attempt_%d.md' % attempt)).write_text(response)
            result['stage'] = 'patch'
            edits = unique_edits(_fix_edit_name_consistency(parse_edits(response)))
            result['n_edits'] = len(edits)
            result['files'] = sorted({e['file_path'] for e in edits})
            try:
                if reply.finish_reason in {'length', 'max_tokens'}:
                    raise ValueError('Code response was truncated; regenerate complete concise edits')
                if not edits:
                    raise ValueError('No complete edits parsed from the code response')
                if '<edit>' in response and len(re.findall(r'<edit>', response)) > len(parse_edits(response)):
                    raise ValueError('Code response contains incomplete or invalid edit blocks')
                patch = edits_to_patch(edits, repo_dir)
                if not patch.strip():
                    raise ValueError('Parsed edits produced an empty patch')
                # Check the exact unified diff without changing the base tree.
                check = subprocess.run(['git', 'apply', '--check', '--whitespace=nowarn', '-'],
                                       cwd=repo_dir, input=patch, text=True, capture_output=True)
                if check.returncode:
                    raise ValueError('Patch does not apply: ' + check.stderr[:2000])
                break
            except (ValueError, SyntaxError) as error:
                history.append({'attempt': attempt, 'error_type': type(error).__name__, 'error': str(error)})
                (out_dir / 'repair_history.json').write_text(json.dumps(history, indent=2))
                if attempt >= max_repair_attempts:
                    raise
                feedback = repair_prompt(instance, repo_dir, edits, error, response, MAX_CONTEXT_CHARS)
                (out_dir / ('repair_prompt_%d.md' % (attempt + 1))).write_text(feedback)
                result['repair_attempts'] = attempt + 1
                reply = invoke('generation_repair_%d' % (attempt + 1), feedback,
                               system='You repair a repository implementation using validation errors and actual base code. Return only the requested JSON object.',
                               model=LLM_MODEL_GENERATE, max_tokens=repair_max_tokens,
                               temperature=LLM_TEMPERATURE, json_mode=True)
        (out_dir / 'edits.json').write_text(json.dumps(edits, indent=2))
        (out_dir / 'generated.patch').write_text(patch)
        result['submitted_patch_count'] = 1
        result['timing']['llm_call'] = time.time() - t0
    except Exception as error:
        result.update(success=False, error=str(error), error_type=type(error).__name__)
        result['timing']['total'] = time.time() - t_start
        save_usage()
        (out_dir / 'result.json').write_text(json.dumps(result, indent=2))
        return result

    if edits:
        for i, edit in enumerate(edits):
            fp_safe = edit['file_path'].replace('/', '_')
            (out_dir / ('edit_%d_%s' % (i, fp_safe))).write_text(
                '# FILE: %s\n# ===== BEFORE =====\n%s\n\n# ===== AFTER =====\n%s\n' %
                (edit['file_path'], edit['before'], edit['after']))

        # Apply edits to repo if requested
        if apply:
            result['stage'] = 'apply'
            try:
                outcomes = apply_edits_to_repo(edits, repo_dir, dry_run=dry_run)
            except Exception as error:
                result.update(error=str(error), error_type=type(error).__name__)
                result['timing']['total'] = time.time() - t_start
                (out_dir / 'result.json').write_text(json.dumps(result, indent=2))
                return result
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
    result['stage'] = 'complete' if result['success'] else 'parse_failure'

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
    parser.add_argument("--instance", nargs="+", help="Instance ID(s)")
    parser.add_argument("--limit", type=int, help="Run first N instances")
    parser.add_argument("--repo", help="Filter by repo")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output base dir")
    parser.add_argument("--apply", action="store_true", help="Apply edits to repo")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to repo")
    parser.add_argument("--dataset-path", help="Path to dataset arrow file")
    parser.add_argument("--repo-dir", type=Path, help="Use an existing base-tree directory without checkout")
    parser.add_argument("--recall-only", action="store_true", help="Save retrieval and prompt without calling a model")
    parser.add_argument("--recall-mode", choices=["v5", "expanded"], default="expanded")
    parser.add_argument("--recall-max-files", type=int, default=12)
    parser.add_argument("--history-mode", choices=["patterns", "legacy", "off"], default="patterns")
    parser.add_argument("--generation-mode", choices=['single', 'plan-edit'], default='plan-edit')
    parser.add_argument("--plan-max-tokens", type=int, default=LLM_PLAN_MAX_TOKENS)
    parser.add_argument("--planner-model", help='Defaults to the generation model')
    parser.add_argument("--planner-json-mode", choices=['api', 'prompt'], default='api')
    parser.add_argument('--max-repair-attempts', type=int, default=2)
    parser.add_argument('--repair-max-tokens', type=int, default=16384)
    args = parser.parse_args()
    if args.repo_dir and (not args.instance or len(args.instance) != 1):
        parser.error("--repo-dir requires exactly one --instance")
    if args.repo_dir and args.apply:
        parser.error("--repo-dir is read-only; omit --apply")
    if not 1 <= args.recall_max_files <= 32:
        parser.error("--recall-max-files must be between 1 and 32")
    if not 1 <= args.plan_max_tokens <= 32768:
        parser.error('--plan-max-tokens must be between 1 and 32768')

    # Ensure env vars
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    ds_path = Path(args.dataset_path) if args.dataset_path else None
    print(f"Loading dataset{f' from {ds_path}' if ds_path else ''}...")
    instances = load_dataset(ds_path)
    print(f"Loaded {len(instances)} instances")

    if args.instance:
        ids = set(args.instance if isinstance(args.instance, list) else [args.instance])
        instances = [i for i in instances if i["instance_id"] in ids]
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
            r = run_single(inst, output_base, dry_run=args.dry_run, apply=args.apply,
                           repo_dir_override=args.repo_dir, recall_only=args.recall_only,
                           recall_mode=args.recall_mode, max_recalled_files=args.recall_max_files,
                           history_mode=args.history_mode, generation_mode=args.generation_mode,
                           plan_max_tokens=args.plan_max_tokens, planner_model=args.planner_model,
                           planner_json_mode=args.planner_json_mode, max_repair_attempts=args.max_repair_attempts,
                           repair_max_tokens=args.repair_max_tokens)
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
    if ok < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
