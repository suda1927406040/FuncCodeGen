"""Summarize saved unresolved diagnostics; never feeds labels into retrieval."""
import argparse
import json
from pathlib import Path


def summarize(root):
    manifest = json.loads((root / "original_unresolved.json").read_text())
    rows = manifest["records"]
    unresolved = set(manifest["baseline"]["unresolved_ids"])
    if {row["instance_id"] for row in rows} != unresolved or len(rows) != len(unresolved):
        raise ValueError("Manifest must cover the exact unresolved set without duplicates")
    source = lambda path: Path(path).suffix in {".py", ".pyi"}
    totals = {"cases": len(rows), "repair_files": 0, "visible_repair_files": 0,
              "extra_repair_files": 0, "visible_extra_repair_files": 0,
              "cases_with_missing_source_file": 0,
              "p2p_regression_cases": sum(row["p2p_fail"] > 0 for row in rows),
              "f2p_any_pass_cases": sum(row["f2p_pass"] > 0 for row in rows),
              "partial_f2p_cases": sum(row["f2p_pass"] > 0 and row["f2p_fail"] > 0 for row in rows)}
    missing_cases = []
    resources = []
    for row in rows:
        expected = {file for file in row["gold_files"] if source(file)}
        hidden = {file for file in row["hidden_files"] if source(file)}
        found = set(row["original_visible_files"])
        totals["repair_files"] += len(expected)
        totals["visible_repair_files"] += len(expected & found)
        totals["extra_repair_files"] += len(hidden)
        totals["visible_extra_repair_files"] += len(hidden & found)
        if expected - found:
            totals["cases_with_missing_source_file"] += 1
            missing_cases.append({"instance_id": row["instance_id"], "missing": sorted(expected - found)})
        runtime_missing = [file for file in row["original_missing_files"]
                           if Path(file).suffix in {".json", ".toml", ".cfg", ".ini"}
                           and not file.startswith((".changes/", "releasenotes/"))]
        if runtime_missing:
            resources.append({"instance_id": row["instance_id"], "missing": runtime_missing})
    reports = json.loads((root / "existing_location_reports.json").read_text())
    location_metrics = {}
    for name, records in reports.items():
        if {row["instance_id"] for row in records} != unresolved:
            raise ValueError("Historical location report has different case coverage: " + name)
        channels = {}
        for channel in ["reference_chain_entities", "selective_local_entities", "class_scope_entities",
                        "featurespace_entities", "repository_wide_lexical_entities"]:
            hit = sum(len(set(row["gold_existing_locations"]) & set(row[channel])) for row in records)
            total = sum(len(set(row["gold_existing_locations"])) for row in records)
            predicted = sum(len(set(row[channel])) for row in records)
            channels[channel] = {"hits": hit, "total": total, "predicted": predicted,
                                 "recall": hit / total if total else None}
        location_metrics[name] = {"eligible_cases": sum(bool(row["gold_existing_locations"]) for row in records),
                                  "channels": channels}
    structural = json.loads((root / "structural_audit.json").read_text())["counts"]
    regression = json.loads((root / "available_base_regression.json").read_text())
    available_unresolved = [row for row in regression["records"] if row["instance_id"] in unresolved]
    subset = {"evaluated_unresolved": len(available_unresolved), "requested_unresolved": len(unresolved),
              "ids": [row["instance_id"] for row in available_unresolved]}
    for field in ["gold_hits", "gold_total", "hidden_hits", "hidden_total", "location_hits", "location_total"]:
        subset[field] = sum(row["results"]["expanded"].get(field, 0) for row in available_unresolved)
    return {"baseline": {key: manifest["baseline"][key] for key in
                         ["total_instances", "resolved_instances", "unresolved_instances",
                          "empty_patch_instances", "error_instances"]},
            "original_prompt_source_recall": totals, "missing_source_cases": missing_cases,
            "missing_resource_or_config_cases": resources,
            "historical_location_metrics": location_metrics,
            "structural_warning_counts": structural,
            "local_regression_unresolved_subset": subset,
            "limitations": ["Historical location reports are precomputed server artifacts, not results of this branch.",
                            "Structural warnings can include valid helper relocation or inherited methods.",
                            "File and symbol recall do not establish that the generated patch solves the case.",
                            "No new model generation or SWE-bench evaluation was run."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[1] / "evaluation/unresolved")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.report_dir)
    output = args.output or args.report_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["original_prompt_source_recall"], indent=2))


if __name__ == "__main__":
    main()
