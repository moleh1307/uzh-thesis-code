"""Derive label-free scoring metadata from frozen requests and their source key."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv
configure_csv()
from run_local_specificity import read_jsonl, validate_request
from aggregate_local_specificity import index_unique, read_csv_rows, sha256_path
from specificity_validation import strict_json

FIELDS = ["custom_id", "unit_type", "event_id", "start_date", "unit_word_count",
          "question_word_count", "period_bin", "source_unit_id", "source_content_sha256"]


def derive(requests, keys):
    ids = [request["custom_id"] for request in requests]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(keys):
        raise ValueError("request/source-key identity coverage mismatch")
    rows = []
    for request in requests:
        validate_request(request)
        key = keys[request["custom_id"]]
        target = strict_json(request["messages"][1]["content"])
        kind = target["unit_type"]
        text = target["ceo_presentation_segment"] if kind == "pre" else target["ceo_answer"]
        question = "" if kind == "pre" else target["analyst_question"]
        content = text if kind == "pre" else question + "\n" + text
        if (key["unit_type"] != kind or int(key["target_word_count"]) != len(text.split())
                or int(key["question_word_count"] or ("0" if kind == "pre" else "")) != len(question.split())
                or key["source_content_sha256"] != hashlib.sha256(content.encode()).hexdigest()):
            raise ValueError("request text differs from source-key identity")
        if not key["event_id"] or not key["start_date"] or not key["period_bin"]:
            raise ValueError("missing event/date/period identity")
        rows.append({**{field: key[field] for field in ("unit_type", "event_id", "start_date", "period_bin",
                     "source_unit_id", "source_content_sha256")}, "custom_id": request["custom_id"],
                     "unit_word_count": len(text.split()), "question_word_count": len(question.split())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--source-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = derive(read_jsonl(args.input_jsonl), index_unique(read_csv_rows(args.source_key), "audit_id", "source key"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "scoring_unit_metadata.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"status": "derived_label_free_metadata", "units": len(rows),
        "request_sha256": sha256_path(args.input_jsonl), "source_key_sha256": sha256_path(args.source_key),
        "metadata_sha256": sha256_path(path), "human_labels_included": False}
    (args.output_dir / "metadata_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
