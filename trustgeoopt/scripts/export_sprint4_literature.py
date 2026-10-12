"""Append verified Sprint 4 metadata while preserving earlier evidence fields."""
import csv
import json
from pathlib import Path

import yaml

project = Path(__file__).resolve().parents[1]
research = project/"research"
sources = json.loads((research/"sprint4_sources.json").read_text())
database = research/"paper_database.csv"
with database.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fields, rows = reader.fieldnames, list(reader)
ids = {row["id"] for row in rows}
for source in sources["new_records"]:
    if source["id"] in ids:
        continue
    row = dict.fromkeys(fields, "")
    for field in ["id", "title", "authors", "year", "venue", "url", "full_text"]:
        row[field] = source[field]
    row.update(central_method=source["mechanism"], theoretical_assumptions=source["assumptions"],
        overhead=source["cost"], limitations=source["failure"], extraction_level=source["location"],
        verified_on=sources["date"], relevance="Sprint 4 noise/curvature identification",
        overlap="Established mechanism; no estimator or optimizer novelty claimed")
    rows.append(row)
for update in sources["updated_records"]:
    row = next(r for r in rows if r["id"] == update["id"])
    row["extraction_level"] = update["location"]
    row["theoretical_assumptions"] = update["assumptions"]
    row["verified_on"] = sources["date"]
    primary_text = {"kfac2015": "https://arxiv.org/html/1503.05671",
        "ekfac2018": "https://arxiv.org/pdf/1806.03884",
        "seng2020": "https://arxiv.org/pdf/2006.05924",
        "purifying2025": "https://arxiv.org/pdf/2506.03595",
        "foam2026": "https://arxiv.org/html/2606.02365v1"}
    row["full_text"] = primary_text[update["id"]]
with database.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fields)
    writer.writeheader()
    writer.writerows(rows)
# Keep the YAML export source in sync without discarding original field types.
yaml_path = research/"papers.yaml"
catalog = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
paper_list = catalog["papers"] if isinstance(catalog, dict) else catalog
by_id = {r["id"]: r for r in paper_list}
new_ids = {r["id"] for r in sources["new_records"]}
updated_ids = {r["id"] for r in sources["updated_records"]}
for row in rows:
    if row["id"] in new_ids and row["id"] not in by_id:
        paper_list.append(row)
    elif row["id"] in updated_ids:
        for field in ["full_text", "theoretical_assumptions", "extraction_level", "verified_on"]:
            by_id[row["id"]][field] = row[field]
yaml_path.write_text(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True), encoding="utf-8")
log_path = research/"search_log.json"
log = json.loads(log_path.read_text())
log["sprint4"] = {k: v for k, v in sources.items() if k not in {"new_records", "updated_records"}}
log["sprint4"].update(new_records=[r["id"] for r in sources["new_records"]],
    upgraded_records=[r["id"] for r in sources["updated_records"]], total_primary_records=len(rows),
    extraction_table="sprint4_literature_audit.md", source_metadata="sprint4_sources.json")
log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
bib = project/"paper"/"references.bib"
existing = bib.read_text(encoding="utf-8")
additions = []
for source in sources["new_records"]:
    if "{"+source["id"]+"," in existing:
        continue
    additions.append("@misc{"+source["id"]+",\n  title = {"+source["title"]+"},\n  author = {"+
        source["authors"].replace("; ", " and ").replace(" and colleagues", " and others")+
        "},\n  year = {"+str(source["year"])+"},\n  url = {"+source["url"]+"},\n  note = {"+source["venue"]+"; Sprint 4 mechanism audit}\n}\n")
if additions:
    bib.write_text(existing+"\n% Sprint 4 verified mechanism sources\n"+"\n".join(additions), encoding="utf-8")
print(f"{len(rows)} records; {len(additions)} new BibTeX entries")
