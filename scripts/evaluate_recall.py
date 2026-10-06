"""Compare file recall on base-tree snapshots without calling a model.

Reference patches are used ONLY after retrieval, to score its output.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import load_dataset, parse_problem_statement
from explorer.context_builder import ContextBuilder
from explorer.feature_retriever import is_source_file
from run_oracle import build_prompt, visible_code_files


def retrieve(instance, repo_dir, mode, max_files):
    # Keep held-out tests and reference patches out of the retrieval interface.
    request = {key: instance.get(key) for key in
               ["instance_id", "repo", "base_commit", "problem_statement", "new_components"]}
    ctx = ContextBuilder(repo_dir, recall_mode=mode, max_recalled_files=max_files).build_context(
        request["instance_id"], request["problem_statement"],
        parse_problem_statement(request["problem_statement"])["definitions"],
        new_components_raw=request["new_components"], base_commit=request["base_commit"])
    prompt = build_prompt(request, repo_dir, ctx)
    return {
        "files": sorted(set(visible_code_files(prompt))),
        "candidates": [{"file": c.file, "score": c.score, "reasons": c.reasons,
                        "locations": c.locations}
                       for c in ctx.recalled_files],
        "prompt_chars": len(prompt),
        "parse_failures": ctx.recall_diagnostics.get("parse_failures", []),
    }


def legacy_retrieve(instance, repo_dir, baseline_root):
    request = {key: instance.get(key) for key in
               ["instance_id", "repo", "base_commit", "problem_statement", "new_components"]}
    code = '''import json,sys,re
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from data_loader import parse_problem_statement
from explorer.context_builder import ContextBuilder
from run_oracle import build_prompt
i=json.load(sys.stdin);r=Path(sys.argv[2])
c=ContextBuilder(r).build_context(i["instance_id"],i["problem_statement"],
    parse_problem_statement(i["problem_statement"])["definitions"],new_components_raw=i["new_components"])
p=build_prompt(i,r,c)
code=p.split("\\n<code>\\n",1)[1].split("\\n</code>",1)[0]
print(json.dumps({"files":sorted(set(re.findall(r"^\\[start of ([^\\n\\]]+)\\]$",code,re.MULTILINE))),"prompt_chars":len(p)}))
'''
    process = subprocess.run([sys.executable, "-c", code, str(baseline_root), str(repo_dir)],
                             input=json.dumps(request), capture_output=True, text=True, check=True)
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--instance", nargs="+")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--modes", nargs="+", choices=["v1", "v5", "expanded"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_files <= 32:
        parser.error("--max-files must be between 1 and 32")
    instances = [i for i in load_dataset(args.dataset_path)
                 if (args.repo_root / i["instance_id"]).is_dir()
                 and (not args.instance or i["instance_id"] in args.instance)]
    modes = args.modes or (["v1"] if args.baseline_root else []) + ["v5", "expanded"]
    if "v1" in modes and not args.baseline_root:
        parser.error("v1 mode requires --baseline-root")
    records = []
    for instance in instances:
        repo_dir = args.repo_root / instance["instance_id"]
        targets = {i["file"] for i in instance.get("new_components") or []}
        results = {}
        for mode in modes:
            start = time.monotonic()
            result = (legacy_retrieve(instance, repo_dir, args.baseline_root) if mode == "v1"
                      else retrieve(instance, repo_dir, mode, args.max_files))
            result["seconds"] = round(time.monotonic() - start, 3)
            results[mode] = result
        # Scoring begins only after all retrieval modes finish.
        expected = {p for p in re.findall(r"^diff --git a/(.*?) b/", instance.get("patch", ""), re.MULTILINE)
                    if is_source_file(p)}
        hidden = expected - targets
        for result in results.values():
            found = set(result["files"])
            result.update(gold_hits=len(expected & found), gold_total=len(expected),
                          hidden_hits=len(hidden & found), hidden_total=len(hidden),
                          missing=sorted(expected - found),
                          hidden_missing=sorted(hidden - found))
        records.append({"instance_id": instance["instance_id"], "expected_files": sorted(expected),
                        "hidden_files": sorted(hidden), "results": results})
        print(instance["instance_id"], {mode: "%d/%d hidden" %
              (result["hidden_hits"], result["hidden_total"]) for mode, result in results.items()}, flush=True)
    summary = {}
    for mode in modes:
        values = [r["results"][mode] for r in records]
        total = sum(v["gold_total"] for v in values)
        hidden_total = sum(v["hidden_total"] for v in values)
        hits = sum(v["gold_hits"] for v in values)
        hidden_hits = sum(v["hidden_hits"] for v in values)
        summary[mode] = {"instances": len(values), "gold_hits": hits, "gold_total": total,
                         "file_recall": hits / total if total else None,
                         "hidden_hits": hidden_hits, "hidden_total": hidden_total,
                         "hidden_recall": hidden_hits / hidden_total if hidden_total else None,
                         "mean_files": sum(len(v["files"]) for v in values) / len(values) if values else None,
                         "mean_prompt_chars": sum(v["prompt_chars"] for v in values) / len(values) if values else None,
                         "seconds": round(sum(v["seconds"] for v in values), 3)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
