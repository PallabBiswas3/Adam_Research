"""Verify preservation and the frozen actual pilot without rerunning training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"strong-baseline-20261012"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inventory(records, base, actual_paths):
    expected = {r["path"] for r in records}
    assert expected == actual_paths, ("changed file set", expected ^ actual_paths)
    for record in records:
        path = base/record["path"]
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], path
    return len(records)


if __name__ == "__main__":
    before = json.loads((OUT/"raw"/"preservation_before.json").read_text())
    trust = ROOT.parent/"trustgeoopt"
    trust_count = check_inventory(before["trustgeoopt"], ROOT.parent,
        {p.relative_to(ROOT.parent).as_posix() for p in trust.rglob("*") if p.is_file()})
    old = {p.relative_to(ROOT).as_posix() for p in (ROOT/"results").rglob("*") if p.is_file() and OUT not in p.parents}
    prior_count = check_inventory(before["prior_results"], ROOT, old)
    freeze = json.loads((OUT/"pilot"/"freeze.json").read_text())
    assert digest(ROOT/"configs"/"strong_baseline_pilot.json") == freeze["config_sha256"]
    for item in freeze["source_files"]:
        assert digest(ROOT/item["path"]) == item["sha256"], item["path"]
    dataset = json.loads((OUT/"dataset_provenance.json").read_text())
    for item in dataset["artifacts"]:
        assert digest(ROOT/item["path"]) == item["sha256"], item["path"]
    audit = json.loads((OUT/"source_audit.json").read_text())
    divide = next(x for x in audit if x["method"] == "dividemix")
    for name in ["LICENSE", "PreResNet.py"]:
        record = next(x for x in divide["assets"] if x["path"] == name)
        assert digest(ROOT/"src"/"robust_learning"/"strong"/"vendor_dividemix"/name) == record["sha256"]
    result = json.loads((OUT/"pilot"/"results.json").read_text())
    assert result["config"] == freeze["config"]
    assert not result["oracle_opened_by_pilot"] and not result["full_campaign_executed"]
    records = result["executed_method_records"]
    assert [r["method"] for r in records] == freeze["config"]["methods"]
    for record in records:
        assert json.loads((OUT/"pilot"/(record["method"]+".json")).read_text()) == record
    files = list((ROOT/"scripts").glob("*.py"))+list((ROOT/"src"/"robust_learning"/"strong").rglob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"))
    reports = ["strong_baseline_protocol", "dss_fidelity", "dividemix_fidelity", "benchmark_feasibility", "strong_baseline_results", "next_research_decision"]
    assert all((ROOT/"research"/(name+".md")).exists() for name in reports)
    verification = {"passed": True, "trustgeoopt_files_unchanged": trust_count,
        "prior_result_files_unchanged": prior_count, "pilot_frozen_files_checked": len(freeze["source_files"]),
        "dataset_files_checked": len(dataset["artifacts"]), "vendored_files_match_MIT_source": True,
        "all_attempts_retained": True, "quality_evaluation_executed": False,
        "preservation_inventory_sha256": digest(OUT/"raw"/"preservation_before.json")}
    (OUT/"verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification))
