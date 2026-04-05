"""Evaluate already-generated CoderEval results by execution.

Reads generated.py from each output dir, inserts into test file, runs it.

Usage:
    conda run -n py38 python eval_coder_eval.py
    conda run -n py38 python eval_coder_eval.py --limit 10
"""
import os
import sys
import ast
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CODEREVAL_BASE = Path("/data2/ymz/codereval_data/CoderEval-Docker/home/travis/builds")
CODEREVAL_JSON = CODEREVAL_BASE / "CoderEval4Python.json"
OUTPUT_BASE = Path("/data2/ymz/TOSEM2026/output_coder_eval")


def load_coder_eval_json():
    with open(CODEREVAL_JSON) as f:
        return json.load(f)


def find_test_file(record, ce_json):
    """Find the test file path for a CoderEval record."""
    _id = record["_id"]
    file_path = record["file_path"]
    project = record.get("project", "")
    name = record["name"]
    repo_base = CODEREVAL_BASE / "repos"

    exceptions = {
        str(repo_base / "standalone/neo4j-_meta-deprecated.py"):
            str(repo_base / "neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py"),
        str(repo_base / "standalone/neo4j-work-query-unit_of_work.py"):
            str(repo_base / "neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py"),
        str(repo_base / "standalone/krake-krake-controller-kubernetes-hooks-on.py"):
            str(repo_base / "rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py"),
    }

    # Standalone
    standalone = str(repo_base / "standalone" /
                     (file_path.replace(".py", "").replace("/", "-") + "-" + name + ".py"))
    if standalone in exceptions and os.path.exists(exceptions[standalone]):
        return exceptions[standalone]
    if os.path.exists(standalone):
        return standalone

    # In-repo
    repo_dir = str(repo_base / project.replace("/", "---"))
    if repo_dir == str(repo_base / "neo4j---neo4j-python-driver"):
        repo_file = os.path.join(repo_dir, "src", file_path).replace(
            ".py", "_" + name + "_passk_validte.py")
    else:
        repo_file = os.path.join(repo_dir, file_path).replace(
            ".py", "_" + name + "_passk_validte.py")

    if "mozilla---relman-auto-nag" in repo_file:
        repo_file = repo_file.replace("/mozilla---relman-auto-nag/auto-ang",
                                      "/mozilla---relman-auto-nag/bugbot")
    if repo_file in exceptions and os.path.exists(exceptions[repo_file]):
        return exceptions[repo_file]
    if os.path.exists(repo_file):
        return repo_file

    return None


def eval_one(generated_code, test_file, func_name):
    """Insert generated code into test file and execute."""
    with open(test_file) as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"passed": False, "error": "test file syntax error"}

    lines = content.split("\n")
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == func_name or node.name == "_" + func_name:
                target = node
                break

    if target is None:
        return {"passed": False, "error": f"func '{func_name}' not found"}

    start_line = target.lineno
    end_line = target.end_lineno
    first_line = lines[start_line - 1]
    start_indent = len(first_line) - len(first_line.lstrip())

    # Normalize generated code indent
    code_lines = generated_code.strip().split("\n")
    code_indent = len(code_lines[0]) - len(code_lines[0].lstrip()) if code_lines[0].strip() else 0
    if code_indent < start_indent:
        pad = " " * (start_indent - code_indent)
        code_lines = [pad + cl if cl.strip() else cl for cl in code_lines]

    new_code = "\n".join(code_lines)
    patched = lines[:start_line - 1] + [new_code] + lines[end_line:]
    patched_content = "\n".join(patched)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     dir=os.path.dirname(test_file)) as tmp:
        tmp.write(patched_content)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(test_file))
        return {
            "passed": proc.returncode == 0,
            "return_code": proc.returncode,
            "stderr": (proc.stderr or "")[:300],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout"}
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default=str(OUTPUT_BASE))
    args = parser.parse_args()

    output_base = Path(args.output)
    ce_json = load_coder_eval_json()

    # Build id -> record map
    record_map = {}
    for r in ce_json["RECORDS"]:
        record_map[r["_id"]] = r

    # Find all instance dirs that have generated.py
    instance_dirs = sorted([d for d in output_base.iterdir() if d.is_dir() and (d / "generated.py").exists()])
    if args.limit:
        instance_dirs = instance_dirs[:args.limit]

    print(f"Evaluating {len(instance_dirs)} instances...\n")

    results = []
    for i, d in enumerate(instance_dirs):
        rid = d.name
        generated = (d / "generated.py").read_text()
        record = record_map.get(rid)
        if not record:
            print(f"[{i+1}/{len(instance_dirs)}] {rid[:12]}... SKIP (not in CoderEval4Python.json)")
            continue

        func_name = record["name"]
        level = record["level"]
        test_file = find_test_file(record, ce_json)

        if not test_file:
            print(f"[{i+1}/{len(instance_dirs)}] {rid[:12]}... {func_name} ({level}) SKIP (no test file)")
            results.append({"id": rid, "level": level, "passed": False, "error": "no test file"})
            continue

        ev = eval_one(generated, test_file, func_name)
        status = "PASS" if ev["passed"] else "FAIL"
        print(f"[{i+1}/{len(instance_dirs)}] {status} {rid[:12]}... {func_name} ({level})")
        if not ev["passed"] and ev.get("stderr"):
            print(f"    stderr: {ev['stderr'][:120]}")

        results.append({"id": rid, "level": level, "passed": ev["passed"],
                        "detail": ev})

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*50}")
    print(f"PASS: {passed}/{total} ({passed/total*100:.1f}%)" if total else "No results")

    levels = {}
    for r in results:
        lvl = r["level"]
        levels.setdefault(lvl, {"count": 0, "passed": 0})
        levels[lvl]["count"] += 1
        if r["passed"]:
            levels[lvl]["passed"] += 1

    print("\nPer-level:")
    for lvl, s in sorted(levels.items()):
        print(f"  {lvl:25s}: {s['passed']:2d}/{s['count']:2d} ({s['passed']/s['count']*100:.0f}%)")

    (output_base / "eval_summary.json").write_text(json.dumps({
        "total": total, "passed": passed,
        "pass_rate": round(passed/total*100, 2) if total else 0,
        "per_level": levels, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
