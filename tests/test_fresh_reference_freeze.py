import copy
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tools/llm_measurement/freeze_fresh_specificity_reference.py"
spec = importlib.util.spec_from_file_location("fresh_reference_freezer", path)
freezer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freezer)


class FreshReferenceFreezeTests(unittest.TestCase):
    def setUp(self):
        self.coded, self.blinded, self.requests = [], [], []
        self.scores = {"scores_in_audit_id_order": []}
        self.classes = {"classes_in_audit_id_order": []}
        for i in range(1, 101):
            unit_type = "pre" if i <= 40 else "qa"
            row = {"audit_id": f"FRESHV1_{i:04d}", "audit_split": "fresh_evaluation",
                   "unit_type": unit_type, "ceo_presentation_segment": "Business detail" if unit_type == "pre" else "",
                   "analyst_question": "Question" if unit_type == "qa" else "",
                   "ceo_answer": "Answer" if unit_type == "qa" else "",
                   "human_content_class": "substantive", "human_ok": "1",
                   "human_specificity": "4", "human_notes": ""}
            score, label = 4, "substantive"
            if i == 14:
                row.update(human_content_class="procedural_only", human_ok="0", human_specificity="0")
                score, label = 0, "procedural_only"
            if i == 53:
                row.update(human_content_class="uncertain", human_ok="", human_specificity="", human_notes="Uninterpretable source")
                score, label = None, "uncertain"
            self.coded.append(row)
            source = dict(row)
            source.update({k: "" for k in freezer.HUMAN_FIELDS})
            self.blinded.append(source)
            target = {"unit_type": unit_type}
            if unit_type == "pre":
                target["ceo_presentation_segment"] = row["ceo_presentation_segment"]
            else:
                target.update(analyst_question=row["analyst_question"], ceo_answer=row["ceo_answer"])
            self.requests.append({"custom_id": row["audit_id"], "messages": [{"role": "user", "content": json.dumps(target)}]})
            self.scores["scores_in_audit_id_order"].append(score)
            self.classes["classes_in_audit_id_order"].append(label)
        self.revisions = {"revised_classes": {}}
        self.confirmation = {"audit_id": "FRESHV1_0008", "confirmed_class": "substantive"}

    def validate(self):
        return freezer.validate_reference(self.coded, self.blinded, self.requests, self.scores,
                                          self.classes, self.revisions, self.confirmation)

    def test_missing_is_not_procedural_zero(self):
        ledger = self.validate()
        self.assertEqual(len(ledger), 100)
        self.assertEqual(sum(r["include_primary_numeric"] for r in ledger), 98)
        self.assertEqual(sum(r["include_primary_eligibility"] for r in ledger), 99)
        self.assertEqual(ledger[13]["reference_state"], "procedural_only")
        self.assertEqual(ledger[52]["human_specificity"], "")
        self.assertEqual(ledger[52]["retain_model_output_diagnostic"], 1)

    def test_missing_imputation_rejected(self):
        self.coded[52].update(human_ok="0", human_specificity="0")
        with self.assertRaises(ValueError):
            self.validate()

    def test_unknown_blank_rejected(self):
        self.coded[0].update(human_ok="", human_specificity="")
        with self.assertRaises(ValueError):
            self.validate()

    def test_score_or_class_override_rejected(self):
        original = copy.deepcopy(self.coded)
        self.coded[0]["human_specificity"] = "5"
        with self.assertRaises(ValueError):
            self.validate()
        self.coded = original
        self.coded[0]["human_content_class"] = "mixed"
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_or_missing_id_rejected(self):
        self.coded[-1]["audit_id"] = self.coded[0]["audit_id"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_source_or_model_input_change_rejected(self):
        self.coded[0]["ceo_presentation_segment"] += " altered"
        with self.assertRaises(ValueError):
            self.validate()
        self.coded[0]["ceo_presentation_segment"] = self.blinded[0]["ceo_presentation_segment"]
        self.requests[0]["messages"][0]["content"] = '{}'
        with self.assertRaises(ValueError):
            self.validate()

    def test_malformed_csv_rejected(self):
        for data in [b"audit_id,audit_id\nx,y\n", b"audit_id,score\nx\n", b"audit_id,score\nx,4,5\n"]:
            with self.assertRaises(ValueError):
                freezer.csv_rows(data)

    def test_dashboard_export_preserves_frozen_missingness_contract(self):
        import test_dashboard_persistence as persistence
        dashboard = persistence.dashboard
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = dashboard.CONTENT_CLASS_AUDIT_COLUMNS
            original = self.coded
            self.blinded = [{k: row.get(k, "") for k in fields} for row in self.blinded]
            source = root / "source.csv"
            with source.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.blinded)
            store = dashboard.AuditStore(source, root / "progress.json", root / "coded.csv")
            for row in original:
                payload = {k: row[k] for k in freezer.HUMAN_FIELDS}
                payload["audit_id"] = row["audit_id"]
                for k in ("human_ok", "human_specificity"):
                    payload[k] = int(payload[k]) if payload[k] else ""
                store.save(payload)
            self.coded = list(csv.DictReader(io.StringIO(store.export_csv_text())))
            ledger = self.validate()
            self.assertEqual(ledger[52]["reference_state"], "source_uninterpretable")
            self.assertEqual(sum(r["include_primary_numeric"] for r in ledger), 98)


if __name__ == "__main__":
    unittest.main()
