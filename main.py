"""Main pipeline - orchestrate the full code generation framework.

Usage:
    # Run a single instance
    conda run -n py38 python main.py --instance tobymao__sqlglot-4579

    # Run first N instances
    conda run -n py38 python main.py --limit 10

    # Run specific repo's instances
    conda run -n py38 python main.py --repo tobymao/sqlglot --limit 5

    # Dry run (no file changes, just generate)
    conda run -n py38 python main.py --instance tobymao__sqlglot-4579 --dry-run

    # Run with validation against gold patch
    conda run -n py38 python main.py --instance tobymao__sqlglot-4579 --validate

    # Run with test execution
    conda run -n py38 python main.py --instance tobymao__sqlglot-4579 --run-tests
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, REPOS_DIR
from data_loader import load_dataset, parse_problem_statement
from repo_manager import prepare_repo, get_file_content
from explorer.context_builder import ContextBuilder
from generator.code_generator import CodeGenerator, GenerationResult
from validator.patch_validator import parse_patch, validate_structure, run_tests


def run_instance(instance: dict, dry_run: bool = False,
                 validate: bool = False, run_tests_flag: bool = False,
                 output_dir: Optional[Path] = None) -> dict:
    """Run the full pipeline for a single instance.

    Returns a result dict with:
        instance_id, success, error, files_generated,
        structural_match (if validate), test_results (if run_tests),
        timing info
    """
    start_time = time.time()
    instance_id = instance["instance_id"]
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "instance_id": instance_id,
        "success": False,
        "error": "",
        "files_generated": [],
        "timing": {},
    }

    print(f"\n{'='*60}")
    print(f"Instance: {instance_id}")
    print(f"Repo: {instance['repo']}")
    print(f"{'='*60}")

    # Step 1: Prepare repo
    print("[1/6] Preparing repo...")
    t0 = time.time()
    try:
        repo_dir = prepare_repo(
            instance_id, instance["base_commit"], instance["repo"]
        )
    except Exception as e:
        result["error"] = f"Repo preparation failed: {e}"
        result["timing"]["repo_prep"] = time.time() - t0
        return result
    result["timing"]["repo_prep"] = time.time() - t0
    print(f"  Repo ready at {repo_dir}")

    # Step 2: Parse problem statement
    print("[2/6] Parsing problem statement...")
    t0 = time.time()
    parsed = parse_problem_statement(instance["problem_statement"])
    if not parsed["definitions"]:
        # Fall back to new_components from the dataset
        if instance.get("new_components"):
            from explorer.context_builder import ContextBuilder
            defs = ContextBuilder._raw_components_to_defs(None, instance["new_components"])
            parsed["definitions"] = defs
    result["timing"]["parsing"] = time.time() - t0
    print(f"  Request: {parsed['request'][:100]}...")
    print(f"  Components: {sum(len(d.get('components', [])) for d in parsed['definitions'])}")

    # Step 3: Build context (deterministic)
    print("[3/6] Building context (deterministic AST analysis)...")
    t0 = time.time()
    builder = ContextBuilder(repo_dir)
    ctx = builder.build_context(
        instance_id=instance_id,
        problem_statement=instance["problem_statement"],
        definitions=parsed["definitions"],
        has_extra_changes=parsed["has_extra_changes"],
        new_components_raw=instance.get("new_components"),
    )
    result["timing"]["context_building"] = time.time() - t0
    print(f"  Context size: {ctx.total_context_chars} chars")
    print(f"  Components analyzed: {len(ctx.components)}")

    # Save formatted context for debugging
    ctx_file = output_dir / f"{instance_id}_context.md"
    ctx_file.write_text(builder.format_context_for_llm(ctx))

    # Step 4: Generate code (3 LLM calls)
    print("[4/6] Generating code (3-phase LLM pipeline)...")
    t0 = time.time()
    generator = CodeGenerator(builder)
    gen_result = generator.generate(ctx)
    result["timing"]["generation"] = time.time() - t0

    if not gen_result.success:
        result["error"] = gen_result.error
        # Save error details
        err_file = output_dir / f"{instance_id}_error.json"
        err_file.write_text(json.dumps({
            "instance_id": instance_id,
            "error": gen_result.error,
            "understanding": gen_result.understanding,
        }, indent=2, default=str))
        return result

    print(f"  Generated {len(gen_result.files)} file changes")

    # Step 5: Apply or preview
    print("[5/6] Applying changes...")
    t0 = time.time()
    if dry_run:
        for gf in gen_result.files:
            print(f"  [{gf.action}] {gf.path}: {gf.description[:80]}")
            result["files_generated"].append({
                "path": gf.path,
                "action": gf.action,
                "description": gf.description,
            })
    else:
        outcomes = generator.apply_generation(gen_result, repo_dir)
        for path, status in outcomes.items():
            print(f"  [{status}] {path}")
            result["files_generated"].append({
                "path": path,
                "action": status,
            })
    result["timing"]["apply"] = time.time() - t0

    # Step 6: Validate (optional)
    if validate and instance.get("patch"):
        print("[6/6] Validating against gold patch...")
        t0 = time.time()
        gen_dict = {f["path"]: f["action"] for f in result["files_generated"]}
        val_result = validate_structure(gen_dict, instance["patch"])
        result["structural_match"] = val_result.structural_match
        result["file_matches"] = val_result.file_matches
        result["timing"]["validation"] = time.time() - t0
        print(f"  Structural match: {val_result.structural_match:.1%}")
        for fp, info in val_result.file_matches.items():
            status = "OK" if info["covered"] else "MISS"
            print(f"    [{status}] {fp} (expected: {info['expected_action']}, got: {info['actual_action']})")

    if run_tests_flag and instance.get("FAIL_TO_PASS"):
        print("[6/6] Running tests...")
        t0 = time.time()
        test_cmds = [f"python -m pytest {t} -x -v --tb=short" for t in instance["FAIL_TO_PASS"]]
        test_result = run_tests(repo_dir, test_cmds)
        result["test_results"] = test_result
        result["timing"]["tests"] = time.time() - t0
        print(f"  Passed: {test_result['passed']}, Failed: {test_result['failed']}, Errors: {test_result['errors']}")

    # Save generation details
    result["success"] = True
    detail_file = output_dir / f"{instance_id}_result.json"
    detail_file.write_text(json.dumps({
        "instance_id": instance_id,
        "success": True,
        "files": [{"path": f.path, "action": f.action, "description": f.description}
                  for f in gen_result.files],
        "understanding": gen_result.understanding,
        "patterns": gen_result.patterns,
        "timing": result["timing"],
    }, indent=2, default=str))

    total = time.time() - start_time
    result["timing"]["total"] = total
    print(f"\n  Total time: {total:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="Symbolic Code Generation Framework")
    parser.add_argument("--instance", type=str, help="Run a specific instance ID")
    parser.add_argument("--limit", type=int, help="Run first N instances")
    parser.add_argument("--repo", type=str, help="Filter by repo name (e.g. tobymao/sqlglot)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just generate")
    parser.add_argument("--validate", action="store_true", help="Validate against gold patch")
    parser.add_argument("--run-tests", action="store_true", help="Run FAIL_TO_PASS tests")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    # Load dataset
    print("Loading FEA-bench dataset...")
    instances = load_dataset()
    print(f"Loaded {len(instances)} instances")

    # Filter
    if args.instance:
        instances = [i for i in instances if i["instance_id"] == args.instance]
        if not instances:
            print(f"Instance '{args.instance}' not found!")
            return
    if args.repo:
        instances = [i for i in instances if i["repo"] == args.repo]
    if args.limit:
        instances = instances[:args.limit]

    print(f"Running {len(instances)} instances")

    # Run
    results = []
    for i, inst in enumerate(instances):
        print(f"\n[{i+1}/{len(instances)}]")
        r = run_instance(
            inst,
            dry_run=args.dry_run,
            validate=args.validate,
            run_tests_flag=args.run_tests,
            output_dir=Path(args.output) if args.output else None,
        )
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successes = sum(1 for r in results if r["success"])
    print(f"Success: {successes}/{len(results)}")

    if args.validate:
        matches = [r.get("structural_match", 0) for r in results if "structural_match" in r]
        if matches:
            print(f"Avg structural match: {sum(matches)/len(matches):.1%}")

    # Save summary
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
