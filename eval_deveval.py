"""Evaluate DevEval results: paste generated code into source files, run pytest.

Uses AST to find exact function body range (ignores body_position from data.jsonl
which may be inaccurate). Each worker gets an isolated copy of the project.

Usage:
    conda run -n py38 python eval_deveval.py --workers 8
    conda run -n py38 python eval_deveval.py --limit 10
"""
import os, sys, json, ast, shutil, subprocess, argparse, textwrap
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DEVEVAL_DATA = "/data2/ymz/baseline/DevEval/data.jsonl"
DEVEVAL_SOURCE = "/data2/ymz/baseline/DevEval/Source_Code"
OUTPUT_BASE = Path("/data2/ymz/TOSEM2026/output_deveval")
EVAL_WORK_DIR = Path("/data2/ymz/TOSEM2026/eval_deveval_work")


def load_data():
    by_ns = {}
    with open(DEVEVAL_DATA) as f:
        for line in f:
            d = json.loads(line)
            by_ns[d["namespace"]] = d
    return by_ns


def find_func_body_range(filepath, func_name):
    """Use AST to find exact [start_line, end_line] of a function/method body.

    Returns (signature_end, body_end) — 1-indexed line numbers.
    signature_end = last line of signature+docstring (body starts after this)
    body_end = last line of the function
    """
    with open(filepath) as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines()

    # Walk AST to find function with matching name
    # For methods like Class.method, search inside classes
    parts = func_name.split(".")
    target = parts[-1]

    def _walk(node):
        if isinstance(node, ast.FunctionDef) and node.name == target:
            return node
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_" + target):
            return node
        for child in ast.iter_child_nodes(node):
            r = _walk(child)
            if r:
                return r
        return None

    func_node = _walk(tree)
    if func_node is None:
        return None

    # The function body starts after decorators + signature
    # end_lineno gives the last line of the function
    func_end = func_node.end_lineno  # 1-indexed

    # Find where the body starts (first statement after docstring)
    body_start = func_node.lineno  # def line (1-indexed)
    if func_node.body:
        first_stmt = func_node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant) and isinstance(first_stmt.value.value, str):
            # Has docstring - body starts after docstring
            body_start = first_stmt.end_lineno + 1
        else:
            body_start = first_stmt.lineno

    return body_start, func_end


def eval_one(ns, data, gen_code, source_root):
    """Evaluate: paste generated code, run pytest, restore."""
    project_path = os.path.join(source_root, data["project_path"])
    completion_path = os.path.join(source_root, data["completion_path"])
    func_name = ns.split(".")[-1]

    if not os.path.exists(completion_path):
        return ns, "Error", "file not found"

    # Find exact body range via AST
    body_range = find_func_body_range(completion_path, func_name)
    if body_range is None:
        return ns, "Error", "AST parse failed"

    body_start, func_end = body_range  # 1-indexed
    indent = data.get("indent", 4)

    # Backup
    bak = completion_path + ".bak"
    shutil.copy2(completion_path, bak)

    try:
        with open(completion_path) as f:
            file_lines = f.readlines()

        # Replace body: keep lines before body_start, insert generated, skip to after func_end
        dedented = textwrap.dedent(gen_code)
        adjusted = textwrap.indent(dedented, " " * indent)

        new_lines = file_lines[:body_start - 1] + [adjusted + "\n"] + file_lines[func_end:]

        with open(completion_path, "w") as f:
            f.writelines(new_lines)

        # Run pytest
        proc = subprocess.run(
            ["python", "-m", "pytest", "-x"] + data["tests"],
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
    args = parser.parse_args()

    print("Loading data...")
    data_by_ns = load_data()
    print(f"Loaded {len(data_by_ns)} instances")

    # Collect tasks
    tasks = []
    for ns, data in data_by_ns.items():
        safe = ns.replace(".", "_").replace("/", "_")
        gen_file = OUTPUT_BASE / safe / "generated.py"
        if not gen_file.exists():
            continue
        tasks.append((ns, data, gen_file.read_text()))

    print(f"Found {len(tasks)} generated results")
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"Limited to {len(tasks)}")

    # Prepare isolated copies
    projects_needed = list(set(t[1]["project_path"] for t in tasks))
    print(f"Preparing {len(projects_needed)} project copies for {args.workers} workers...")
    EVAL_WORK_DIR.mkdir(parents=True, exist_ok=True)
    work_copies = []
    for i in range(args.workers):
        wd = EVAL_WORK_DIR / f"w{i}"
        wd.mkdir(parents=True, exist_ok=True)
        for proj in projects_needed:
            src = Path(DEVEVAL_SOURCE) / proj
            dst = wd / proj
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        work_copies.append(str(wd))

    # Distribute tasks
    worker_tasks = [[] for _ in range(args.workers)]
    for i, t in enumerate(tasks):
        worker_tasks[i % args.workers].append(t)

    print(f"Evaluating {len(tasks)} instances...")
    results = {}
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for wid, wtasks in enumerate(worker_tasks):
            src_root = work_copies[wid]
            for ns, data, gen in wtasks:
                f = executor.submit(eval_one, ns, data, gen, src_root)
                futures[f] = ns

        for future in as_completed(futures):
            done += 1
            try:
                ns, result, detail = future.result()
            except Exception as e:
                ns, result, detail = futures[future], "Error", str(e)

            results[ns] = {"result": result, "detail": detail}
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

    (OUTPUT_BASE / "eval_summary.json").write_text(json.dumps({
        "total": len(results), "passed": passed, "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / len(results) * 100, 2) if results else 0,
        "results": results,
    }, indent=2))

    print("Cleaning up...")
    shutil.rmtree(EVAL_WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
