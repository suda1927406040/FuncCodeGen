"""Evaluate DevEval results using definition-based replacement.

Instead of body_position replacement, we use AST to find the function's
full range and replace the entire function with the LLM-generated one.
This avoids indentation issues.

Usage:
    conda run -n py38 python eval_deveval_def.py --limit 10
    conda run -n py38 python eval_deveval_def.py --workers 8
"""
import os, sys, json, ast, shutil, argparse, textwrap, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DEVEVAL_DATA = "/data2/ymz/baseline/DevEval/data.jsonl"
DEVEVAL_SOURCE = "/data2/ymz/baseline/DevEval/Source_Code"


def load_data():
    by_ns = {}
    with open(DEVEVAL_DATA) as f:
        for line in f:
            d = json.loads(line)
            by_ns[d["namespace"]] = d
    return by_ns


def find_func_range(filepath, func_name, comp_type):
    """Use AST to find exact [start_line, end_line] of a function/method."""
    with open(filepath) as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target = func_name.split(".")[-1]

    def _walk(node):
        if isinstance(node, ast.FunctionDef) and node.name == target:
            return node
        if isinstance(node, ast.AsyncFunctionDef) and node.name == target:
            return node
        for child in ast.iter_child_nodes(node):
            r = _walk(child)
            if r:
                return r
        return None

    func_node = _walk(tree)
    if func_node is None:
        return None

    return func_node.lineno, func_node.end_lineno  # 1-indexed


def eval_one(ns, data, gen_code, source_root):
    """Evaluate: replace function with generated code, run pytest, restore."""
    project_path = os.path.join(source_root, data["project_path"])
    completion_path = os.path.join(source_root, data["completion_path"])
    func_name = ns.split(".")[-1]

    if not os.path.exists(completion_path):
        return ns, "Error", "file not found"

    # Find function range via AST
    func_range = find_func_range(completion_path, func_name, data["type"])
    if func_range is None:
        return ns, "Error", "AST parse failed"

    line_start, line_end = func_range  # 1-indexed

    # Backup
    bak = completion_path + ".bak"
    shutil.copy2(completion_path, bak)

    try:
        with open(completion_path) as f:
            file_lines = f.readlines()

        # Ensure generated code ends with newline
        if not gen_code.endswith('\n'):
            gen_code += '\n'

        # Normalize indentation to match original function
        orig_line = file_lines[line_start - 1]
        orig_indent = len(orig_line) - len(orig_line.lstrip())
        # Find def line indent in generated code, then shift everything
        gen_lines = gen_code.splitlines(True)
        def_line = next((l for l in gen_lines if l.strip().startswith('def ')), None)
        if def_line:
            gen_def_indent = len(def_line) - len(def_line.lstrip())
            shift = orig_indent - gen_def_indent
            if shift != 0:
                gen_lines = [
                    (' ' * shift + l if l.strip() else l)
                    for l in gen_lines
                ]
            gen_code = ''.join(gen_lines)

        # Replace entire function: lines [line_start-1 : line_end]
        new_lines = file_lines[:line_start - 1] + [gen_code] + file_lines[line_end:]

        with open(completion_path, "w") as f:
            f.writelines(new_lines)

        # Run pytest in DevEval conda environment
        proc = subprocess.run(
            ["conda", "run", "--name", "DevEval", "--no-capture-output",
             "python", "-m", "pytest", "-x"] + data["tests"],
            cwd=project_path,
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0:
            return ns, "Pass", ""
        else:
            detail = (proc.stderr or proc.stdout)[-200:]
            return ns, "Fail", detail

    except subprocess.TimeoutExpired:
        return ns, "Error", "timeout"
    except Exception as e:
        return ns, "Error", str(e)
    finally:
        if os.path.exists(bak):
            shutil.move(bak, completion_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="/data2/ymz/TOSEM2026/output_deveval_v3")
    parser.add_argument("--source_code_root", default="/data2/ymz/baseline/DevEval/Source_Code_eval")
    args = parser.parse_args()

    print("Loading data...")
    data_by_ns = load_data()
    print(f"Loaded {len(data_by_ns)} instances")

    # Load generated completions
    output_jsonl = Path(args.output) / "deveval_output.jsonl"
    if not output_jsonl.exists():
        print(f"ERROR: {output_jsonl} not found")
        return

    tasks = []
    with open(output_jsonl) as f:
        for line in f:
            js = json.loads(line)
            ns = js["namespace"]
            if ns in data_by_ns:
                tasks.append((ns, data_by_ns[ns], js["completion"]))

    print(f"Found {len(tasks)} generated results")
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"Limited to {len(tasks)}")

    # Prepare isolated copies
    source_root = args.source_code_root
    print(f"Using source root: {source_root}")

    print(f"Evaluating {len(tasks)} instances (single-threaded)...")
    results = {}
    done = 0

    for ns, data, gen in tasks:
        done += 1
        try:
            ns_out, result, detail = eval_one(ns, data, gen, source_root)
        except Exception as e:
            ns_out, result, detail = ns, "Error", str(e)

        results[ns_out] = {"result": result, "detail": detail}
        status = "PASS" if result == "Pass" else result[0]
        print(f"  [{done}/{len(tasks)}] [{status:4s}] {ns[:55]}")

    passed = sum(1 for v in results.values() if v["result"] == "Pass")
    failed = sum(1 for v in results.values() if v["result"] == "Fail")
    errors = sum(1 for v in results.values() if v["result"] == "Error")

    print(f"\n{'='*60}")
    print(f"Evaluated: {len(results)}")
    print(f"Pass:  {passed} ({passed/len(results)*100:.1f}%)")
    print(f"Fail:  {failed}")
    print(f"Error: {errors}")

    (Path(args.output) / "eval_summary.json").write_text(json.dumps({
        "total": len(results), "passed": passed, "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / len(results) * 100, 2) if results else 0,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
