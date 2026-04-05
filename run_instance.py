"""Run a single instance for debugging and testing.

Usage:
    conda run -n py38 python run_instance.py tobymao__sqlglot-4579
    conda run -n py38 python run_instance.py tobymao__sqlglot-4579 --dry-run
    conda run -n py38 python run_instance.py tobymao__sqlglot-4579 --validate
    conda run -n py38 python run_instance.py tobymao__sqlglot-4579 --show-context
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_instance, parse_problem_statement
from repo_manager import prepare_repo
from explorer.context_builder import ContextBuilder
from generator.code_generator import CodeGenerator
from validator.patch_validator import validate_structure


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_instance.py <instance_id> [--dry-run] [--validate] [--show-context]")
        sys.exit(1)

    instance_id = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    validate_flag = "--validate" in sys.argv
    show_context = "--show-context" in sys.argv

    # Load
    print(f"Loading instance: {instance_id}")
    instance = load_instance(instance_id)
    if not instance:
        print(f"Instance not found: {instance_id}")
        sys.exit(1)

    print(f"Repo: {instance['repo']}")
    print(f"Base commit: {instance['base_commit']}")

    # Prepare repo
    print("Preparing repo...")
    repo_dir = prepare_repo(instance_id, instance["base_commit"], instance["repo"])
    print(f"Repo at: {repo_dir}")

    # Parse
    parsed = parse_problem_statement(instance["problem_statement"])
    print(f"\nRequest:\n{parsed['request'][:300]}")
    print(f"\nDefinitions:")
    for d in parsed["definitions"]:
        for c in d.get("components", []):
            print(f"  {d['file']} -> {c['name']} ({c['type']})")
            print(f"    Signature: {c['signature']}")

    # Build context
    print("\nBuilding context...")
    builder = ContextBuilder(repo_dir)
    ctx = builder.build_context(
        instance_id=instance_id,
        problem_statement=instance["problem_statement"],
        definitions=parsed["definitions"],
        has_extra_changes=parsed["has_extra_changes"],
        new_components_raw=instance.get("new_components"),
    )
    print(f"Context size: {ctx.total_context_chars} chars")
    print(f"Components: {len(ctx.components)}")

    for comp in ctx.components:
        print(f"\n  Component: {comp.name}")
        print(f"    File: {comp.target_file}")
        print(f"    Parent class: {comp.parent_class}")
        if comp.nested_class:
            print(f"    Nested in: {comp.nested_class}")
        print(f"    Sibling methods: {len(comp.sibling_methods)}")
        print(f"    Init vars: {len(comp.class_init_vars)}")
        print(f"    Base classes: {list(comp.base_classes.keys())}")
        print(f"    Similar functions: {len(comp.similar_functions)}")
        print(f"    Init.py: {'needs update' if comp.init_py_exports and not comp.init_py_exports.get('imports_our_module') else 'ok/unknown'}")

    if show_context:
        formatted = builder.format_context_for_llm(ctx)
        print(f"\n{'='*60}")
        print("FORMATTED CONTEXT FOR LLM:")
        print(f"{'='*60}")
        print(formatted[:5000])
        print(f"\n... (total {len(formatted)} chars)")
        return

    # Generate
    print("\nGenerating code...")
    generator = CodeGenerator(builder)
    result = generator.generate(ctx)

    if not result.success:
        print(f"Generation failed: {result.error}")
        sys.exit(1)

    print(f"\nGenerated {len(result.files)} file changes:")
    for f in result.files:
        print(f"\n  [{f.action}] {f.path}")
        print(f"  Description: {f.description}")
        if f.action == "create":
            print(f"  Content preview:\n{f.content[:500]}")
        elif f.action == "modify":
            print(f"  Search:\n{f.search[:200]}")
            print(f"  Replace:\n{f.replace[:200]}")

    # Apply
    if not dry_run:
        print("\nApplying changes...")
        outcomes = generator.apply_generation(result, repo_dir)
        for path, status in outcomes.items():
            print(f"  [{status}] {path}")

    # Validate
    if validate_flag and instance.get("patch"):
        print("\nValidating against gold patch...")
        gen_dict = {f.path: f.action for f in result.files}
        val = validate_structure(gen_dict, instance["patch"])
        print(f"Structural match: {val.structural_match:.1%}")
        for fp, info in val.file_matches.items():
            status = "OK" if info["covered"] else "MISS"
            print(f"  [{status}] {fp}")


if __name__ == "__main__":
    main()
