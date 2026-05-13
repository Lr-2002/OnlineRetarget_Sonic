import unittest

from online_retarget.data.row_sampling import (
    sampling_run_tag,
    scan_sampling_report,
    select_rows_for_scan,
)


class RowSamplingTests(unittest.TestCase):
    def test_default_selection_preserves_first_n_behavior(self):
        rows = [{"row_index": str(index), "category": category} for index, category in enumerate(["A", "A", "B"])]

        selected = select_rows_for_scan(rows, limit=2)

        self.assertEqual([row["row_index"] for row in selected], ["0", "1"])

    def test_stratified_selection_round_robins_sorted_groups(self):
        rows = [
            {"row_index": "0", "category": "Walk", "split": "train"},
            {"row_index": "1", "category": "Walk", "split": "train"},
            {"row_index": "2", "category": "Jump", "split": "train"},
            {"row_index": "3", "category": "Jump", "split": "train"},
            {"row_index": "4", "category": "Idle", "split": "val"},
        ]

        selected = select_rows_for_scan(rows, limit=4, sample_by=("category",))

        self.assertEqual([row["row_index"] for row in selected], ["4", "2", "0", "3"])

    def test_sampling_report_counts_candidate_and_selected_groups(self):
        rows = [
            {"row_index": "0", "category": "Walk"},
            {"row_index": "1", "category": "Walk"},
            {"row_index": "2", "category": "Jump"},
        ]
        selected = select_rows_for_scan(rows, limit=2, sample_by=("category",))

        report = scan_sampling_report(rows, selected, limit=2, sample_by=("category",))

        self.assertEqual(report["mode"], "stratified_round_robin")
        self.assertEqual(report["candidate_group_counts"]["category=Walk"], 2)
        self.assertEqual(report["selected_group_counts"]["category=Jump"], 1)
        self.assertEqual(report["selected_group_counts"]["category=Walk"], 1)

    def test_sampling_run_tag_marks_stratified_limits(self):
        self.assertEqual(sampling_run_tag(8), "limit8")
        self.assertEqual(sampling_run_tag(None, ("category",)), "full")
        self.assertEqual(sampling_run_tag(8, ("category", "split")), "limit8_by-category-split")


if __name__ == "__main__":
    unittest.main()
