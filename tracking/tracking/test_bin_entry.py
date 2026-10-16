#!/usr/bin/env python3

from __future__ import annotations

import csv
import io
import unittest

import numpy as np

from bin_geometry import BinGeometry
from bin_placement_dwell import BinEntryManager
from global_identity import ACTIVE, OCCLUDED, GlobalRollState


EVENT_FIELDS = [
    "frame", "timestamp_seconds", "global_id", "roll_id", "local_id",
    "physical_bin_id", "expected_bin_id", "actual_bin_id",
    "placement_confirmed", "placement_confirmation_method", "placement_frame",
    "placement_status", "placement_evidence", "placement_confidence",
    "entered_bin_mouth", "entry_frame", "entry_timestamp", "entry_bin_id",
    "entry_direction_score", "entry_trajectory_consistency",
    "last_visible_centroid",
    "verification_result", "previous_bin_id", "candidate_state",
    "candidate_start_frame", "candidate_start_time", "current_overlap",
    "max_overlap", "accumulated_dwell_seconds", "current_speed_px_per_sec",
    "last_valid_overlap_time", "gap_seconds", "last_visible_frame",
    "last_overlap", "last_centroid_x", "last_centroid_y", "motion_dx",
    "motion_dy", "first_distance_to_bin_center",
    "last_distance_to_bin_center", "centroid_inside", "overlap_increased",
    "moved_deeper", "provisional_elapsed_seconds", "grace_duration_seconds",
    "confirmation_method", "reason", "event",
]


class BinPlacementDwellTests(unittest.TestCase):
    def make_manager(
        self,
        polygons: dict[str, np.ndarray] | None = None,
        *,
        entry_zones: dict[str, np.ndarray] | None = None,
        dwell_seconds: float = 1.0,
        gap_tolerance_seconds: float = 0.30,
        settle_speed: float = 20.0,
        placement_grace_seconds: float = 1.0,
    ) -> tuple[BinEntryManager, io.StringIO]:
        geometry = BinGeometry(
            physical_bins=("BIN_1", "BIN_2", "BIN_3", "BIN_4"),
            cameras={
                "cam0": polygons or {
                    "BIN_1": np.array(
                        [[0, 0], [100, 0], [100, 100], [0, 100]], np.float32
                    ),
                    "BIN_2": np.array(
                        [[200, 0], [300, 0], [300, 100], [200, 100]], np.float32
                    ),
                }
            },
            entry_zones={"cam0": entry_zones or {}},
        )
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        manager = BinEntryManager(
            geometry=geometry,
            camera_id="cam0",
            overlap_threshold=0.35,
            visible_dwell_seconds=dwell_seconds,
            gap_tolerance_seconds=gap_tolerance_seconds,
            placement_grace_seconds=placement_grace_seconds,
            settle_speed_px_per_sec=settle_speed,
            detector_miss_distance_px=80.0,
            static_handoff_seconds=2.0,
            static_handoff_distance_px=120.0,
            ambiguity_margin=0.10,
            allow_reassignment=False,
            fps=10.0,
            event_writer=writer,
            debug=False,
        )
        return manager, stream

    @staticmethod
    def global_state(*, entry_armed: bool = True) -> GlobalRollState:
        return GlobalRollState(
            global_id=3,
            current_local_id=10,
            state=ACTIVE,
            first_seen_frame=0,
            last_seen_frame=0,
            last_centroid=(50.0, 50.0),
            velocity=(0.0, 0.0),
            last_box_size=(40.0, 40.0),
            journey_started=True,
            bin_entry_initialized=entry_armed,
            bin_entry_armed=entry_armed,
        )

    @staticmethod
    def box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)

    @staticmethod
    def events(stream: io.StringIO) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(stream.getvalue())))

    def test_elapsed_dwell_confirms_across_local_id_change(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        for frame in range(11):
            local_id = 10 if frame < 5 else 12
            state.current_local_id = local_id
            manager.process_observation(
                state, local_id, self.box(10, 10, 90, 90), frame
            )
        self.assertTrue(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "BIN_1")
        self.assertEqual(state.bin_entry_frame, 10)
        self.assertEqual(manager.placement_counts["BIN_1"], 1)
        confirmed = next(
            row for row in self.events(stream)
            if row["event"] == "BIN_PLACEMENT_CONFIRMED"
        )
        self.assertEqual(confirmed["global_id"], "3")
        self.assertEqual(confirmed["local_id"], "12")
        self.assertEqual(state.placement_status, "CONFIRMED")
        self.assertEqual(state.placement_evidence, "DIRECT_VISIBLE")
        self.assertEqual(state.placement_confidence, 1.0)

    def test_expected_bin_match_emits_correct_per_roll_result(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        manager.assign_expected_bin(state, "BIN_1")
        for frame in range(11):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        self.assertTrue(state.placement_confirmed)
        self.assertEqual(state.placement_frame, 10)
        self.assertEqual(state.verification_result, "CORRECT")
        verified = next(
            row for row in self.events(stream)
            if row["event"] == "ROLL_PLACEMENT_CORRECT"
        )
        self.assertEqual(verified["expected_bin_id"], "BIN_1")
        self.assertEqual(verified["actual_bin_id"], "BIN_1")
        confirmed = next(
            row for row in self.events(stream)
            if row["event"] == "BIN_PLACEMENT_CONFIRMED"
        )
        self.assertEqual(confirmed["verification_result"], "CORRECT")

    def test_expected_bin_mismatch_emits_wrong_bin_per_roll_result(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        manager.assign_expected_bin(state, "BIN_2")
        for frame in range(11):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        self.assertEqual(state.verification_result, "WRONG_BIN")
        self.assertIn(
            "ROLL_PLACEMENT_WRONG_BIN",
            [row["event"] for row in self.events(stream)],
        )

    def test_confirmed_without_expected_bin_is_unverified(self) -> None:
        manager, _ = self.make_manager()
        state = self.global_state()
        for frame in range(11):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        self.assertEqual(state.verification_result, "UNVERIFIED")

    def test_late_expected_bin_assignment_verifies_locked_actual_bin(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        for frame in range(11):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        result = manager.assign_expected_bin(state, "BIN_2", frame_index=12)
        self.assertEqual(result, "WRONG_BIN")
        self.assertEqual(state.actual_bin_id, "BIN_1")
        self.assertIn(
            "ROLL_PLACEMENT_WRONG_BIN",
            [row["event"] for row in self.events(stream)],
        )

    def test_brief_touch_does_not_confirm(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        for frame in range(4):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        for frame in range(4, 8):
            manager.process_observation(
                state, 10, self.box(120, 120, 150, 150), frame
            )
        self.assertFalse(state.bin_confirmed)
        self.assertEqual(state.candidate_bin_id, "")
        cancelled = [
            row for row in self.events(stream)
            if row["event"] == "BIN_CANDIDATE_CANCELLED"
        ]
        self.assertEqual(cancelled[-1]["reason"], "OVERLAP_LOST")

    def test_short_detection_gap_keeps_candidate(self) -> None:
        manager, _ = self.make_manager()
        state = self.global_state()
        for frame in range(5):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        manager.finish_frame([state], set(), 5)
        manager.finish_frame([state], set(), 6)
        for frame in range(7, 11):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        self.assertTrue(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "BIN_1")

    def test_disappearance_without_mouth_entry_is_unresolved(self) -> None:
        manager, stream = self.make_manager(placement_grace_seconds=0.5)
        state = self.global_state()
        for frame in range(4):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        state.state = OCCLUDED
        manager.finish_frame([state], set(), 4)
        self.assertNotEqual(state.bin_placement_state, "PROVISIONAL_PLACEMENT")
        self.assertFalse(state.bin_confirmed)
        for frame in range(5, 10):
            manager.finish_frame([state], set(), frame)
        self.assertFalse(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "")
        self.assertEqual(state.placement_status, "UNRESOLVED")
        self.assertEqual(state.placement_evidence, "NONE")
        self.assertEqual(state.verification_result, "UNRESOLVED")
        self.assertNotIn(
            "BIN_DISAPPEAR_CONFIRMED",
            [row["event"] for row in self.events(stream)],
        )

    def test_candidate_switch_restarts_timer(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        for frame in range(6):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        for frame in range(6, 17):
            manager.process_observation(
                state, 10, self.box(210, 10, 290, 90), frame
            )
        self.assertTrue(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "BIN_2")
        self.assertEqual(state.bin_entry_frame, 16)
        switched = [
            row for row in self.events(stream)
            if row["event"] == "BIN_CANDIDATE_SWITCHED"
        ]
        self.assertEqual(switched[0]["previous_bin_id"], "BIN_1")
        self.assertEqual(manager.placement_counts["BIN_1"], 0)
        self.assertEqual(manager.placement_counts["BIN_2"], 1)

    def test_high_motion_waits_until_roll_settles(self) -> None:
        manager, _ = self.make_manager(settle_speed=20.0)
        state = self.global_state()
        for frame in range(11):
            x = 10 + frame * 3
            manager.process_observation(state, 10, self.box(x, 10, x + 40, 90), frame)
        self.assertFalse(state.bin_confirmed)
        for frame in range(11, 18):
            manager.process_observation(
                state, 10, self.box(40, 10, 80, 90), frame
            )
        self.assertTrue(state.bin_confirmed)

    def test_starting_bin_not_counted_until_exit_and_reentry(self) -> None:
        manager, _ = self.make_manager()
        state = self.global_state(entry_armed=False)
        for frame in range(3):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        self.assertEqual(state.initial_bin_id, "BIN_1")
        self.assertFalse(state.bin_entry_armed)
        manager.process_observation(state, 10, self.box(120, 120, 150, 150), 3)
        for frame in range(4, 15):
            manager.process_observation(
                state, 10, self.box(210, 10, 290, 90), frame
            )
        self.assertTrue(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "BIN_2")

    def test_provisional_reappearance_outside_cancels_without_count(self) -> None:
        entry = {
            "BIN_1": np.array(
                [[100, 0], [175, 0], [175, 100], [100, 100]], np.float32
            )
        }
        manager, stream = self.make_manager(
            entry_zones=entry, placement_grace_seconds=1.0
        )
        state = self.global_state()
        manager.process_observation(state, 10, self.box(176, 10, 196, 50), 0)
        manager.process_observation(state, 10, self.box(140, 10, 180, 50), 1)
        manager.process_observation(state, 10, self.box(110, 10, 150, 50), 2)
        state.state = OCCLUDED
        manager.finish_frame([state], set(), 3)
        self.assertEqual(state.placement_status, "PROVISIONAL")
        state.state = ACTIVE
        manager.process_observation(
            state, 12, self.box(180, 120, 220, 150), 4
        )
        self.assertFalse(state.bin_confirmed)
        self.assertTrue(state.reappeared_outside_candidate)
        self.assertEqual(manager.placement_counts["BIN_1"], 0)
        self.assertIn(
            "BIN_PROVISIONAL_CANCELLED",
            [row["event"] for row in self.events(stream)],
        )
        self.assertEqual(state.placement_status, "TRACKING")
        self.assertEqual(state.placement_evidence, "NONE")

    def test_provisional_switches_to_second_bin_without_counting_first(self) -> None:
        entry = {
            "BIN_1": np.array(
                [[100, 0], [175, 0], [175, 100], [100, 100]], np.float32
            )
        }
        manager, _ = self.make_manager(
            entry_zones=entry, placement_grace_seconds=1.0
        )
        state = self.global_state()
        manager.process_observation(state, 10, self.box(176, 10, 196, 50), 0)
        manager.process_observation(state, 10, self.box(140, 10, 180, 50), 1)
        manager.process_observation(state, 10, self.box(110, 10, 150, 50), 2)
        state.state = OCCLUDED
        manager.finish_frame([state], set(), 3)
        state.state = ACTIVE
        for frame in range(4, 15):
            manager.process_observation(
                state, 12, self.box(210, 10, 290, 90), frame
            )
        self.assertTrue(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "BIN_2")
        self.assertEqual(manager.placement_counts["BIN_1"], 0)
        self.assertEqual(manager.placement_counts["BIN_2"], 1)

    def test_possible_detector_miss_is_reported_without_bin_candidate(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        manager.process_observation(state, 10, self.box(125, 10, 165, 50), 0)
        manager.process_observation(state, 10, self.box(115, 10, 155, 50), 1)
        manager.process_observation(state, 10, self.box(105, 10, 145, 50), 2)
        state.state = OCCLUDED
        manager.finish_frame([state], set(), 3)
        self.assertEqual(len(manager.possible_detector_misses), 1)
        self.assertEqual(manager.possible_detector_misses[0]["bin_id"], "BIN_1")
        self.assertIn(
            "POSSIBLE_DETECTOR_MISS_AT_BIN_ENTRY",
            [row["event"] for row in self.events(stream)],
        )
        self.assertEqual(state.placement_status, "UNRESOLVED")
        self.assertEqual(state.verification_result, "UNRESOLVED")

    def test_active_candidate_can_finish_through_static_local_handoff(self) -> None:
        manager, stream = self.make_manager()
        state = self.global_state()
        for frame in range(5):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        for frame in range(5, 12):
            ownership = manager.process_static_observation(
                [state], 27, self.box(10, 10, 90, 90), frame
            )
            self.assertIsNotNone(ownership)
        self.assertTrue(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "BIN_1")
        self.assertEqual(state.placement_owner_static_local_id, 27)
        self.assertEqual(
            state.placement_confirmation_method,
            "ACTIVE_TO_STATIC_HANDOFF",
        )
        self.assertTrue(state.factory_placement_completed)
        self.assertEqual(state.placement_status, "CONFIRMED")
        self.assertEqual(state.placement_evidence, "ACTIVE_TO_STATIC")
        self.assertEqual(state.placement_confidence, 0.95)
        self.assertIn(
            "BIN_STATIC_HANDOFF_LINKED",
            [row["event"] for row in self.events(stream)],
        )

    def test_entry_zone_crossing_can_start_provisional_without_occupancy_overlap(self) -> None:
        entry = {
            "BIN_1": np.array(
                [[100, 0], [175, 0], [175, 100], [100, 100]], np.float32
            )
        }
        manager, _ = self.make_manager(
            entry_zones=entry,
            placement_grace_seconds=0.5,
        )
        state = self.global_state()
        manager.process_observation(state, 10, self.box(176, 10, 196, 50), 0)
        manager.process_observation(state, 10, self.box(140, 10, 180, 50), 1)
        manager.process_observation(state, 10, self.box(110, 10, 150, 50), 2)
        state.state = OCCLUDED
        manager.finish_frame([state], set(), 3)
        self.assertEqual(state.bin_placement_state, "PROVISIONAL_PLACEMENT")
        for frame in range(4, 9):
            manager.finish_frame([state], set(), frame)
        self.assertTrue(state.bin_confirmed)
        self.assertEqual(state.actual_bin_id, "BIN_1")
        self.assertEqual(state.placement_confirmation_method, "DISAPPEAR_IN_BIN")
        self.assertTrue(state.entered_bin_mouth)
        self.assertEqual(state.entry_bin_id, "BIN_1")
        self.assertGreaterEqual(state.entry_direction_score, 0.70)
        self.assertGreaterEqual(state.entry_trajectory_consistency, 0.70)
        self.assertEqual(state.placement_status, "CONFIRMED")
        self.assertEqual(state.placement_evidence, "ENTRY_THEN_OCCLUDED")
        self.assertGreater(state.placement_confidence, 0.0)

    def test_stale_entry_crossing_does_not_confirm_late_disappearance(self) -> None:
        entry = {
            "BIN_1": np.array(
                [[80, 0], [175, 0], [175, 100], [80, 100]], np.float32
            )
        }
        manager, _ = self.make_manager(
            entry_zones=entry,
            settle_speed=1.0,
            placement_grace_seconds=0.5,
        )
        state = self.global_state()
        manager.process_observation(state, 10, self.box(176, 10, 196, 50), 0)
        manager.process_observation(state, 10, self.box(140, 10, 180, 50), 1)
        manager.process_observation(state, 10, self.box(90, 10, 130, 50), 2)
        for frame in range(3, 12):
            x = 75 - frame * 5
            manager.process_observation(
                state, 10, self.box(x, 10, x + 40, 50), frame
            )
        state.state = OCCLUDED
        manager.finish_frame([state], set(), 12)
        for frame in range(13, 20):
            manager.finish_frame([state], set(), frame)
        self.assertFalse(state.placement_confirmed)
        self.assertEqual(state.placement_status, "UNRESOLVED")
        self.assertEqual(state.verification_result, "UNRESOLVED")

    def test_finalize_reports_unfinished_candidate(self) -> None:
        manager, _ = self.make_manager()
        state = self.global_state()
        for frame in range(4):
            manager.process_observation(
                state, 10, self.box(10, 10, 90, 90), frame
            )
        manager.finalize([state], 3)
        diagnostic = manager.candidate_history[3][-1]
        self.assertEqual(diagnostic.reason, "DWELL_NOT_LONG_ENOUGH")
        self.assertFalse(diagnostic.confirmed)
        self.assertEqual(state.placement_status, "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
