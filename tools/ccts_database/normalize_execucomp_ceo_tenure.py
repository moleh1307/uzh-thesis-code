#!/usr/bin/env python3
"""Normalize a WRDS Compustat ExecuComp annual CSV into CEO tenure rows.

The script preserves the licensed raw export unchanged. It filters annual rows
to CEOANN=CEO, splits each GVKEY/EXECID history into contiguous annual CEO
episodes, and emits the source-independent schema consumed by
validate_ceo_turnovers_external.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


OUTPUT_COLUMNS = [
    "source_name", "source_record_id", "source_url_or_locator", "company_name", "isin", "cusip", "gvkey",
    "person_name", "person_name_key", "ceo_title", "ceo_start_date", "ceo_end_date", "evidence_type", "notes",
    "execid", "tenure_episode", "first_ceoann_year", "last_ceoann_year", "annual_ceo_rows", "start_date_source", "end_date_source",
    "normalization_status",
]
AUDIT_COLUMNS = [
    "gvkey", "execid", "tenure_episode", "person_name", "annual_ceo_rows", "first_ceoann_year", "last_ceoann_year",
    "becameceo_values", "leftofc_values", "cusip_values", "normalization_status", "audit_notes",
]
ALIASES = {
    "gvkey": ["gvkey"],
    "execid": ["execid", "exec_id"],
    "person_name": ["exec_fullname", "execfullname", "exec_name", "executive_name"],
    "first_name": ["exec_fname", "exec_firstname", "first_name"],
    "last_name": ["exec_lname", "exec_lastname", "last_name"],
    "year": ["year", "fyear"],
    "ceoann": ["ceoann", "ceo_ann"],
    "titleann": ["titleann", "title_ann", "title"],
    "becameceo": ["becameceo", "became_ceo"],
    "leftofc": ["leftofc", "left_ofc", "left_office"],
    "company_name": ["coname", "company_name"],
    "cusip": ["cusip", "cusip_full"],
    "ticker": ["ticker", "tic"],
}


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()


def resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    by_lower = {name.lower(): name for name in fieldnames}
    return {
        target: next((by_lower[alias] for alias in aliases if alias in by_lower), "")
        for target, aliases in ALIASES.items()
    }


def parse_date(value: str) -> str:
    value = clean(value)
    if not value:
        return ""
    for fmt in (None, "%m/%d/%Y", "%Y%m%d", "%d%b%Y", "%d-%b-%Y"):
        try:
            parsed = datetime.fromisoformat(value) if fmt is None else datetime.strptime(value, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return ""


def as_year(value: str) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", clean(value))
    return int(match.group()) if match else None


def values(rows: list[dict[str, str]], column: str) -> list[str]:
    return sorted({clean(row.get(column, "")) for row in rows if clean(row.get(column, ""))}) if column else []


def latest_value(rows: list[dict[str, str]], column: str, year_column: str) -> str:
    if not column:
        return ""
    ordered = sorted(rows, key=lambda row: as_year(row.get(year_column, "")) or -1, reverse=True)
    return next((clean(row.get(column, "")) for row in ordered if clean(row.get(column, ""))), "")


def person_name(row: Mapping[str, str], columns: Mapping[str, str]) -> str:
    if columns["person_name"]:
        return clean(row.get(columns["person_name"], ""))
    return clean(f"{row.get(columns['first_name'], '')} {row.get(columns['last_name'], '')}")


def contiguous_episodes(rows: list[dict[str, str]], year_column: str) -> list[list[dict[str, str]]]:
    ordered = sorted(rows, key=lambda row: as_year(row.get(year_column, "")) or -1)
    episodes: list[list[dict[str, str]]] = []
    previous_year: int | None = None
    for row in ordered:
        year = as_year(row.get(year_column, ""))
        if year is None:
            continue
        if previous_year is None or year > previous_year + 1:
            episodes.append([])
        episodes[-1].append(row)
        previous_year = year
    return episodes


def write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw WRDS ExecuComp annual CSV export.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-locator", default="WRDS Compustat ExecuComp Annual Compensation export")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    input_handle = gzip.open(args.input, "rt", encoding="utf-8-sig", newline="") if args.input.suffix.lower() == ".gz" else args.input.open("r", encoding="utf-8-sig", newline="")
    with input_handle as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("ExecuComp input has no header.")
        columns = resolve_columns(reader.fieldnames)
        required = [name for name in ("gvkey", "execid", "year", "ceoann") if not columns[name]]
        if required:
            raise SystemExit(f"ExecuComp input missing required columns: {', '.join(required)}")
        raw_rows = list(reader)

    ceo_rows = [row for row in raw_rows if normalize(row.get(columns["ceoann"], "")) == "ceo"]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ceo_rows:
        groups[(clean(row[columns["gvkey"]]), clean(row[columns["execid"]]))].append(row)

    episode_rows: list[dict[str, Any]] = []
    for (gvkey, execid), rows in sorted(groups.items()):
        all_became = [parsed for raw in values(rows, columns["becameceo"]) if (parsed := parse_date(raw))]
        all_left = [parsed for raw in values(rows, columns["leftofc"]) if (parsed := parse_date(raw))]
        episodes = contiguous_episodes(rows, columns["year"])
        for episode_number, episode in enumerate(episodes, start=1):
            years = sorted(year for row in episode if (year := as_year(row.get(columns["year"], ""))) is not None)
            first_year, last_year = min(years), max(years)
            start_candidates = [value for value in all_became if episode_number == 1 and int(value[:4]) <= first_year or first_year - 1 <= int(value[:4]) <= first_year + 1]
            end_candidates = [value for value in all_left if last_year - 1 <= int(value[:4]) <= last_year + 1]
            start = min(start_candidates) if start_candidates else f"{first_year}-01-01"
            end = max(end_candidates) if end_candidates else f"{last_year}-12-31"
            start_source = "becameceo_exact" if start_candidates else "ceoann_year_lower_bound"
            end_source = "leftofc_exact" if end_candidates else "ceoann_year_upper_bound"
            sample = episode[-1]
            name = person_name(sample, columns)
            episode_rows.append({
                "source_name": "Compustat ExecuComp via WRDS", "source_record_id": f"gvkey={gvkey};execid={execid};episode={episode_number}",
                "source_url_or_locator": args.source_locator, "company_name": latest_value(episode, columns["company_name"], columns["year"]),
                "isin": "", "cusip": latest_value(episode, columns["cusip"], columns["year"]), "gvkey": gvkey,
                "person_name": name, "person_name_key": normalize(name), "ceo_title": latest_value(episode, columns["titleann"], columns["year"]),
                "ceo_start_date": start, "ceo_end_date": end, "evidence_type": "licensed_execucomp_annual_ceoann",
                "execid": execid, "tenure_episode": episode_number, "first_ceoann_year": first_year,
                "last_ceoann_year": last_year, "annual_ceo_rows": len(episode),
                "start_date_source": start_source, "end_date_source": end_source,
                "becameceo_values": ";".join(values(episode, columns["becameceo"])),
                "leftofc_values": ";".join(values(episode, columns["leftofc"])),
                "cusip_values": ";".join(values(episode, columns["cusip"])),
            })

    by_firm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        by_firm[row["gvkey"]].append(row)
    for rows in by_firm.values():
        rows.sort(key=lambda row: (row["first_ceoann_year"], row["ceo_start_date"], row["execid"]))
        for current, following in zip(rows, rows[1:]):
            if (
                current["end_date_source"] == "ceoann_year_upper_bound"
                and following["start_date_source"] == "becameceo_exact"
                and following["first_ceoann_year"] <= current["last_ceoann_year"] + 1
                and following["ceo_start_date"] >= current["ceo_start_date"]
            ):
                current["ceo_end_date"] = following["ceo_start_date"]
                current["end_date_source"] = "next_ceo_becameceo_exact"

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in sorted(episode_rows, key=lambda item: (item["gvkey"], item["first_ceoann_year"], item["execid"])):
        exact_start = row["start_date_source"] == "becameceo_exact"
        exact_end = row["end_date_source"] in {"leftofc_exact", "next_ceo_becameceo_exact"}
        invalid_date_order = date.fromisoformat(row["ceo_start_date"]) > date.fromisoformat(row["ceo_end_date"])
        if invalid_date_order:
            status = "invalid_date_order_review"
            notes = "ExecuComp exact/fiscal-year evidence yields a start date after the end date; preserve source values and exclude from automatic confirmation."
        elif exact_start and exact_end:
            status = "exact_tenure_dates"
            notes = "Exact CEO episode boundaries from ExecuComp fields or the next CEO's exact start."
        else:
            status = "annual_flag_date_bounds"
            notes = "At least one episode boundary is inferred from annual CEOANN coverage; retain for review."
        row.update({"normalization_status": status, "notes": notes})
        output_rows.append(row)
        audit_rows.append({
            "gvkey": row["gvkey"], "execid": row["execid"], "tenure_episode": row["tenure_episode"],
            "person_name": row["person_name"], "annual_ceo_rows": row["annual_ceo_rows"],
            "first_ceoann_year": row["first_ceoann_year"], "last_ceoann_year": row["last_ceoann_year"],
            "becameceo_values": row["becameceo_values"], "leftofc_values": row["leftofc_values"],
            "cusip_values": row["cusip_values"], "normalization_status": status, "audit_notes": notes,
        })

    output_path = args.output_dir / "execucomp_ceo_tenure_normalized.csv"
    audit_path = args.output_dir / "execucomp_ceo_tenure_normalization_audit.csv"
    write_csv(output_path, OUTPUT_COLUMNS, output_rows)
    write_csv(audit_path, AUDIT_COLUMNS, audit_rows)
    status_counts = Counter(row["normalization_status"] for row in output_rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "raw_rows": len(raw_rows),
        "ceoann_rows": len(ceo_rows), "normalized_tenures": len(output_rows), "return_episode_splits": len(output_rows) - len(groups),
        "invalid_date_order_rows": status_counts.get("invalid_date_order_review", 0), "status_counts": dict(status_counts),
        "normalized_csv": str(output_path), "audit_csv": str(audit_path),
        "caveat": "Annual CEOANN bounds are not exact turnover dates. Prefer BECAMECEO/LEFTOFC and externally review consequential transitions.",
    }
    (args.output_dir / "execucomp_normalization_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
