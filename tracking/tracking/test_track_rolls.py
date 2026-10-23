#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from global_identity import (
    CORRECT,
    PLACEMENT_PENDING,
    VISIBLE,
    WRONG_BIN,
    GlobalRollState,
)

from track_rolls import (
    TrackState,
    RollTracker,
    centroid_from_corners,
    centroid_in_roi,
    load_roi,
    load_zones,
    validate_zones_for_frame,
    output_log_paths,
    output_global_event_path,
    output_summary_path,
    tracker_type_from_config,
)


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeObb:
    def __init__(self) -> None:
        self.xyxyxyxy = FakeTensor(
            np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=float)
        )
        self.conf = FakeTensor(np.array([0.75]))
        self.cls = FakeTensor(np.array([1.0]))
        self.id = None

    def __len__(self) -> int:
        return 1


class TrackingLogicTests(unittest.TestCase):
    def make_state(self) -> TrackState:
        return TrackState(
            track_id=7,
            first_seen_frame=0,
            last_seen_frame=0,
            last_centroid=(0.0, 0.0),
            centroid_history=deque(maxlen=30),
            confidence_history=deque(maxlen=30),
        )

    @staticmethod
    def make_global() -> GlobalRollState:
        return GlobalRollState(
            global_id=3,
            current_local_id=8,
            state=VISIBLE,
            first_seen_frame=0,
            last_seen_frame=0,
            last_centroid=(100.0, 100.0),
            velocity=(0.0, 0.0),
            last_box_size=(80.0, 40.0),
        )

    def test_clean_overlay_uses_only_operator_statuses(self) -> None:
        state = self.make_global()
        self.assertEqual(RollTracker._production_status(state), "TRACKING")
        self.assertNotIn("Local", RollTracker._production_label(state))
        self.assertNotIn("Candidate", RollTracker._production_label(state))

        state.state = PLACEMENT_PENDING
        state.placement_status = "PROVISIONAL"
        self.assertEqual(
            RollTracker._production_status(state), "PLACEMENT PENDING"
        )

        state.bin_confirmed = True
        state.actual_bin_id = "BIN_3"
        state.verification_result = "UNVERIFIED"
        self.assertEqual(RollTracker._production_status(state), "UNRESOLVED")
        self.assertIn("Actual: BIN_3", RollTracker._production_label(state))

        state.expected_bin_id = "BIN_3"
        state.verification_result = CORRECT
        self.assertEqual(RollTracker._production_status(state), "CORRECT")

        state.expected_bin_id = "BIN_4"
        state.verification_result = WRONG_BIN
        self.assertEqual(RollTracker._production_status(state), "WRONG BIN")
        self.assertTrue(
            RollTracker._production_label(state).startswith(
                "WARNING: WRONG BIN"
            )
        )

    def test_obb_centroid_is_mean_of_four_corners(self) -> None:
        corners = np.array([[10, 10], [30, 10], [30, 20], [10, 20]])
        self.assertEqual(centroid_from_corners(corners), (20.0, 15.0))

    def test_motion_threshold_and_hold_hysteresis(self) -> None:
        state = self.make_state()
        for frame in range(9):
            transition = state.record_detection(
                frame, (frame * 3.0, 0.0), 0.9, 8, 20.0, 3
            )
        self.assertEqual(transition, "BECAME_ACTIVE")
        self.assertTrue(state.is_active)
        self.assertIsNone(state.record_detection(9, (24.0, 0.0), 0.9, 8, 20.0, 3))
        self.assertIsNone(state.record_detection(10, (24.0, 0.0), 0.9, 8, 20.0, 3))
        self.assertIsNone(state.record_detection(11, (24.0, 0.0), 0.9, 8, 20.0, 3))
        self.assertEqual(
            state.record_detection(12, (24.0, 0.0), 0.9, 8, 20.0, 3),
            "BECAME_STATIC",
        )
        self.assertFalse(state.is_active)

    def test_sustained_motion_metrics_and_direction(self) -> None:
        tracker = RollTracker.__new__(RollTracker)
        tracker.args = SimpleNamespace(journey_direction_window=10)
        tracker.zones = {
            "zone_a": np.array(
                [[0, 0], [20, 0], [20, 20], [0, 20]], np.float32
            ),
            "zone_b": np.array(
                [[180, 0], [200, 0], [200, 20], [180, 20]], np.float32
            ),
        }
        state = self.make_state()
        state.pending_since_frame = 0
        state.pending_active_frames = 6
        state.centroid_history.extend(
            (frame, x, 10.0)
            for frame, x in enumerate((160, 150, 140, 130, 120, 110))
        )
        active_frames, displacement, consistency = (
            tracker._pending_motion_metrics(state)
        )
        self.assertEqual(active_frames, 6)
        self.assertAlmostEqual(displacement, 50.0)
        self.assertAlmostEqual(consistency, 1.0)
        self.assertEqual(
            tracker._infer_journey_origin(state, allow_best=False), "B"
        )

    def test_roi_inclusion_includes_boundary(self) -> None:
        roi = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
        self.assertTrue(centroid_in_roi((50.0, 50.0), roi))
        self.assertTrue(centroid_in_roi((0.0, 50.0), roi))
        self.assertFalse(centroid_in_roi((101.0, 50.0), roi))

    def test_roi_loader_and_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roi_path = Path(directory) / "roi.json"
            roi_path.write_text(
                '{"tracking_region": [[0, 0], [100, 0], [100, 100]]}'
            )
            self.assertEqual(load_roi(roi_path).shape, (3, 2))
        tracking, events = output_log_paths(Path("outputs/cam2_tracking.mp4"))
        self.assertEqual(tracking, Path("outputs/cam2_tracking.csv"))
        self.assertEqual(events, Path("outputs/cam2_events.csv"))
        self.assertEqual(
            output_summary_path(Path("outputs/cam2_tracking.mp4")),
            Path("outputs/cam2_tracking_summary.txt"),
        )
        self.assertEqual(
            output_global_event_path(Path("outputs/cam2_tracking.mp4")),
            Path("outputs/cam2_tracking_global_events.csv"),
        )
        self.assertEqual(
            output_global_event_path(Path("outputs/cam2_tracking_global.mp4")),
            Path("outputs/cam2_global_events.csv"),
        )
        self.assertEqual(
            output_global_event_path(
                Path("outputs/cam2_tracking_global_shortgap.mp4")
            ),
            Path("outputs/cam2_global_events_shortgap.csv"),
        )
        tracking, events = output_log_paths(
            Path("outputs/cam2_tracking_botsort.mp4")
        )
        self.assertEqual(
            tracking, Path("outputs/cam2_tracking_botsort.csv")
        )
        self.assertEqual(events, Path("outputs/cam2_events_botsort.csv"))
        _, events = output_log_paths(Path("outputs/cam2_tracking_tuned.mp4"))
        self.assertEqual(
            events, Path("outputs/cam2_tracking_tuned_events.csv")
        )

    def test_zone_loader_accepts_empty_and_configured_polygons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty_path = Path(directory) / "empty.json"
            empty_path.write_text(
                '{"zone_a": [], "zone_b": [], "tracking_zone": []}'
            )
            self.assertEqual(load_zones(empty_path), {})

            zones_path = Path(directory) / "zones.json"
            zones_path.write_text(
                '{"zone_a": [[0,0],[10,0],[10,10]], '
                '"zone_b": [], "tracking_zone": []}'
            )
            zones = load_zones(zones_path)
            self.assertEqual(zones["zone_a"].shape, (3, 2))
            self.assertFalse(validate_zones_for_frame(zones, 1280, 720))

            complete_path = Path(directory) / "complete.json"
            complete_path.write_text(
                '{"zone_a": [[0,0],[10,0],[10,10]], '
                '"tracking_zone": [[0,0],[100,0],[100,100]], '
                '"zone_b": [[80,80],[100,80],[100,100]]}'
            )
            self.assertTrue(
                validate_zones_for_frame(load_zones(complete_path), 1280, 720)
            )

            legacy_path = Path(directory) / "legacy.json"
            legacy_path.write_text(
                '{"pickup_zone": [[0,0],[10,0],[10,10]], '
                '"tracking_zone": [[0,0],[100,0],[100,100]], '
                '"handoff_zone": [[80,80],[100,80],[100,100]]}'
            )
            legacy = load_zones(legacy_path)
            self.assertEqual(set(legacy), {"zone_a", "tracking_zone", "zone_b"})

        self.assertEqual(load_zones(Path(directory) / "missing.json"), {})

    def test_tracker_type_from_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "custom.yaml"
            config_path.write_text(
                "# test config\ntracker_type: botsort # supported tracker\n"
            )
            self.assertEqual(tracker_type_from_config(str(config_path)), "botsort")
        self.assertEqual(tracker_type_from_config("bytetrack.yaml"), "bytetrack")

    def test_unconfirmed_obb_detection_is_not_fatal(self) -> None:
        tracker = RollTracker.__new__(RollTracker)
        tracker.roi = None
        observations = tracker._extract_observations(SimpleNamespace(obb=FakeObb()))
        self.assertEqual(len(observations), 1)
        self.assertIsNone(observations[0].track_id)
        self.assertEqual(observations[0].centroid, (5.0, 5.0))


if __name__ == "__main__":
    unittest.main()
