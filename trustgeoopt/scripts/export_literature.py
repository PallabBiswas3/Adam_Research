"""Build CSV and BibTeX from human-verified primary records; does not verify URLs."""
import csv
from pathlib import Path

import yaml


def export(root):
    (root/"paper").mkdir(parents=True, exist_ok=True)
    papers = yaml.safe_load((root/"research"/"papers.yaml").read_text(encoding="utf-8"))
    fields = ["id", "title", "authors", "year", "first_arxiv_year", "venue", "url", "full_text",
              "central_method", "mathematical_contribution", "theoretical_assumptions", "experiments",
              "overhead", "limitations", "relevance", "overlap", "extraction_level", "verified_on"]
    with (root/"research"/"paper_database.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for paper in papers:
            row = {field: paper.get(field, "Not extracted / not applicable; consult primary full text") for field in fields}
            row["verified_on"] = "2026-10-12"
            writer.writerow(row)
    entries = []
    for paper in papers:
        if paper["id"] == "rmsprop_notes":
            # Unverified date is not promoted to a verified publication year.
            year = "n.d."
        else:
            year = str(paper["year"])
        author = paper["authors"].replace("; ", " and ")
        entries.append("@misc{"+paper["id"]+",\n  title = {{"+paper["title"]+"}},\n  author = {"+author+
            "},\n  year = {"+year+"},\n  howpublished = {"+paper["venue"]+"},\n  url = {"+paper["url"]+"}\n}\n")
    (root/"paper"/"references.bib").write_text("\n".join(entries), encoding="utf-8")
    print(f"Exported {len(papers)} primary records (including one teaching-note source)")


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[1])
