"""Evaluate CoderEval generated code using the native PythonExec.py logic
with CoderEval's own Python 3.10 (which has all project dependencies).

Usage:
    python3 eval_coder_exec.py [--output-dir output_coder_eval]

This reads generated.py from each instance dir, inserts into the test file
(using the same AST logic as PythonExec.py), and runs with CoderEval Python.
"""
import ast
import json
import os
import subprocess
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# CoderEval Docker filesystem root
CODEREVAL_ROOT = Path("/data2/ymz/codereval_data/CoderEval-Docker")
CODEREVAL_PYTHON = CODEREVAL_ROOT / "usr/local/bin/python3.10"
CODEREVAL_SITE = CODEREVAL_ROOT / "usr/local/lib/python3.10/site-packages"

# CoderEval paths (matching PythonExec.py)
CODEREVAL_BASE = CODEREVAL_ROOT / "home/travis/builds"
CODEREVAL_JSON = CODEREVAL_BASE / "CoderEval4Python.json"
CODEREVAL_REPO_BASE = CODEREVAL_BASE / "repos"

# dict_std_nonestd from PythonExec.py
dict_std_nonestd = {
    str(CODEREVAL_REPO_BASE / "standalone/neo4j-_meta-deprecated.py"):
        str(CODEREVAL_REPO_BASE / "neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py"),
    str(CODEREVAL_REPO_BASE / "standalone/neo4j-work-query-unit_of_work.py"):
        str(CODEREVAL_REPO_BASE / "neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py"),
    str(CODEREVAL_REPO_BASE / "standalone/krake-krake-controller-kubernetes-hooks-on.py"):
        str(CODEREVAL_REPO_BASE / "rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py"),
}


def load_coder_eval_json():
    with open(CODEREVAL_JSON) as f:
        return json.load(f)


def resolve_test_file(record):
    """Resolve test file path using same logic as PythonExec.py."""
    _id = record["_id"]
    file_path = record["file_path"]
    project = record.get("project", "")
    name = record["name"]

    # Try standalone
    standalone = (str(CODEREVAL_REPO_BASE) + "/standalone/" +
                  file_path.replace(".py", "").replace("/", "-") + "-" + name + ".py")
    if standalone in dict_std_nonestd:
        return dict_std_nonestd[standalone]
    if os.path.exists(standalone):
        return standalone

    # Try in-repo
    repo_dir = str(CODEREVAL_REPO_BASE) + "/" + project.replace("/", "---")
    if repo_dir == str(CODEREVAL_REPO_BASE) + "/neo4j---neo4j-python-driver":
        save_data = os.path.join(repo_dir + "/src", file_path).replace(
            ".py", "_" + name + "_passk_validte.py")
    else:
        save_data = os.path.join(repo_dir, file_path).replace(
            ".py", "_" + name + "_passk_validte.py")

    # Fix mozilla path
    save_data = save_data.replace(
        "/mozilla---relman-auto-nag/auto-ang",
        "/mozilla---relman-auto-nag/bugbot")

    if save_data in dict_std_nonestd:
        return dict_std_nonestd[save_data]
    if os.path.exists(save_data):
        return save_data

    return None


def eval_one(test_file, generated_code, func_name):
    """Evaluate one instance using CoderEval's PythonExec logic."""
    with open(test_file) as f:
        file_content = f.read()

    file_lines = file_content.split("\n")

    # AST find target function
    tree = ast.parse(file_content)
    start_line = end_line = start_indent = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == func_name or node.name == "_" + func_name:
                start_line = node.lineno
                end_line = node.end_lineno
                indent_s = file_lines[start_line - 1]
                tttt = indent_s.lstrip(" ")
                start_indent = len(indent_s) - len(tttt)
                break

    if start_line is None:
        return {"is_pass": False, "error": f"Function '{func_name}' not found in test file"}

    # Build template: everything before function + placeholder + everything after
    new_data = ""
    for i in range(0, start_line - 1):
        new_data += file_lines[i] + "\n"
    new_data += "<insert generated code here>\n"
    for i in range(end_line, len(file_lines)):
        new_data += file_lines[i] + "\n"

    # Adjust generated code indent
    code_lines = generated_code.split("\n")
    tttt = code_lines[0].lstrip(" ")
    code_indent = len(code_lines[0]) - len(tttt)

    if start_indent > code_indent:
        pad = " " * (start_indent - code_indent)
        new_code = "\n".join(pad + line for line in code_lines) + "\n"
    else:
        new_code = generated_code

    out_data = new_data.replace("<insert generated code here>", new_code)

    # Write temp file
    tmp_path = test_file.replace(".py", "_eval_tmp.py")
    with open(tmp_path, 'w') as f:
        f.write(out_data)

    try:
        proc = subprocess.run(
            [str(CODEREVAL_PYTHON), tmp_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(tmp_path),
        )
        return {
            "is_pass": proc.returncode == 0,
            "return_code": proc.returncode,
            "stdout": proc.stdout[:500],
            "stderr": proc.stderr[:500] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"is_pass": False, "error": "timeout (30s)"}
    except Exception as e:
        return {"is_pass": False, "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
                        default="/data2/ymz/TOSEM2026/output_coder_eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load CoderEval metadata
    coder_eval_json = load_coder_eval_json()
    collection = {r["_id"]: r for r in coder_eval_json["RECORDS"]}
    print(f"Loaded {len(collection)} records from CoderEval4Python.json")

    # Collect results from output dir
    results = []
    for d in sorted(os.listdir(output_dir)):
        rf = output_dir / d / "result.json"
        gf = output_dir / d / "generated.py"
        if not rf.is_file():
            continue
        r = json.load(open(rf))
        if not r.get("success"):
            continue
        rid = r["id"]
        if rid not in collection:
            continue

        # Read generated code
        gen_code = open(gf).read() if gf.exists() else r.get("completion", "")
        if not gen_code.strip():
            continue

        meta = collection[rid]
        func_name = meta["name"]
        level = meta.get("level", "unknown")

        # Resolve test file
        test_file = resolve_test_file(meta)
        if not test_file:
            results.append({
                "_id": rid, "level": level, "name": func_name,
                "is_pass": False, "error": "no test file"
            })
            continue

        # Evaluate
        eval_result = eval_one(test_file, gen_code, func_name)
        results.append({
            "_id": rid,
            "level": level,
            "name": func_name,
            "test_file": test_file,
            **eval_result,
        })

        status = "PASS" if eval_result["is_pass"] else "FAIL"
        print(f"  [{status}] {rid[:12]} {func_name} ({level})")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.get("is_pass"))
    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} passed ({passed/total*100:.1f}%)")

    # Per-level
    levels = defaultdict(lambda: {"count": 0, "passed": 0})
    for r in results:
        levels[r["level"]]["count"] += 1
        if r.get("is_pass"):
            levels[r["level"]]["passed"] += 1

    print("\nPer-level:")
    for lvl in ["self_contained", "slib_runnable", "plib_runnable",
                "class_runnable", "file_runnable", "project_runnable"]:
        s = levels[lvl]
        if s["count"]:
            print(f"  {lvl:25s}: {s['passed']:2d}/{s['count']:2d} ({s['passed']/s['count']*100:.0f}%)")

    # Save
    (output_dir / "exec_results.json").write_text(json.dumps({
        "total": total, "passed": passed,
        "pass_rate": round(passed/total*100, 2),
        "per_level": dict(levels),
        "results": results,
    }, indent=2, default=str))
    print(f"\nSaved to {output_dir / 'exec_results.json'}")


if __name__ == "__main__":
    main()
