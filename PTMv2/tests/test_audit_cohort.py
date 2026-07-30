"""Unit tests for the cohort-audit data integrity checks."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scr"))
from audit_cohort import summarise_alignment  # noqa: E402


class CohortAuditTest(unittest.TestCase):
    def test_summary_counts_complete_binary_labels(self) -> None:
        feature_index = pd.Index(["P1", "P2", "P3"], name="Patient_ID")
        labels = pd.DataFrame(
            {
                "sample_id": ["P1", "P2"],
                "patient_id": ["P1", "P2"],
                "raw_label": ["G2", "G3"],
                "class_name": ["moderate", "poor"],
                "target": [0, 1],
            }
        )
        summary, class_counts = summarise_alignment(feature_index, labels)
        values = summary.set_index("metric")["value"]
        self.assertEqual(values["labelled_patients_present_in_matrix"], 2)
        self.assertEqual(values["matrix_samples_without_grade_label"], 1)
        self.assertEqual(class_counts["patients"].sum(), 2)

    def test_duplicate_patient_is_rejected(self) -> None:
        labels = pd.DataFrame(
            {
                "sample_id": ["P1", "P1"],
                "patient_id": ["P1", "P1"],
                "raw_label": ["G2", "G3"],
                "class_name": ["moderate", "poor"],
                "target": [0, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summarise_alignment(pd.Index(["P1"]), labels)


if __name__ == "__main__":
    unittest.main()
