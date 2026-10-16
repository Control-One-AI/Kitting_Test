from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from estimate_camera_offset import calculate_offsets
from make_cross_camera_debug_video import (
    load_cross_camera_csv,
    load_tracking_csv,
)


class OffsetTests(unittest.TestCase):
    def test_reference_event_normalizes_to_same_time(self) -> None:
        cam0_offset, cam1_offset = calculate_offsets(12.30, 17.80)
        self.assertAlmostEqual(cam0_offset, 0.0)
        self.assertAlmostEqual(cam1_offset, -5.5)
        self.assertAlmostEqual(12.30 + cam0_offset, 17.80 + cam1_offset)


class CsvTests(unittest.TestCase):
    def test_loads_current_tracking_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "frame",
                        "track_id",
                        "global_id",
                        "journey_direction",
                        "lifecycle_state",
                        "actual_bin_id",
                        "cx",
                        "cy",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "frame": "52",
                        "track_id": "17",
                        "global_id": "3",
                        "journey_direction": "A_TO_B",
                        "lifecycle_state": "ACTIVE",
                        "actual_bin_id": "BIN_3",
                        "cx": "101.5",
                        "cy": "202.5",
                    }
                )
            records = load_tracking_csv(path)
            self.assertEqual(records[52][0].local_id, "17")
            self.assertEqual(records[52][0].global_id, "3")
            self.assertEqual(records[52][0].actual_bin, "BIN_3")

    def test_cross_camera_ids_and_events_are_read_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "factory_roll_id",
                        "cam0_global_id",
                        "cam1_global_id",
                        "cam0_exit_timestamp",
                        "cam1_entry_timestamp",
                        "handoff_gap_seconds",
                        "match_score",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "factory_roll_id": "R007",
                        "cam0_global_id": "G3",
                        "cam1_global_id": "G1",
                        "cam0_exit_timestamp": "20.0",
                        "cam1_entry_timestamp": "25.5",
                        "handoff_gap_seconds": "1.43",
                        "match_score": "0.87",
                    }
                )
            data = load_cross_camera_csv(path)
            self.assertEqual(data.factory_id("cam0", "3"), "R007")
            self.assertEqual(data.factory_id("cam1", "G1"), "R007")
            self.assertEqual(len(data.events), 2)
            self.assertIn("EXIT", data.events[0].display_text())


if __name__ == "__main__":
    unittest.main()
