#!/usr/bin/env python3

from __future__ import annotations

import csv
import io
import unittest

import numpy as np

from global_identity import (
    ACTIVE,
    COMPLETED,
    ENTERING_BIN,
    IdentityObservation,
    GlobalIdentityManager,
    LEAVING_BIN,
    LOST,
    OCCLUDED,
    PLACED,
    PLACEMENT_PENDING,
    VISIBLE,
)


EVENT_FIELDS = [
    "frame",
    "timestamp_seconds",
    "global_id",
    "local_id",
    "old_local_id",
    "new_local_id",
    "event",
    "lifecycle_state",
    "zone",
    "origin_zone",
    "destination_zone",
    "journey_direction",
    "activation_reason",
    "gap_frames",
    "last_seen_distance",
    "predicted_distance",
    "association_distance",
    "recent_speed",
    "dynamic_allowed_distance",
    "was_active_before_occlusion",
    "previous_speed",
    "direction_score",
    "active_bonus",
    "active_allowed_distance",
    "size_ratio",
    "position_score",
    "time_score",
    "motion_score",
    "size_score",
    "reassociation_score",
    "decision_reason",
]


class GlobalIdentityTests(unittest.TestCase):
    def make_manager(
        self,
        occlusion_frames: int = 90,
        zones: dict[str, np.ndarray] | None = None,
        handoff_hold_frames: int = 10,
    ) -> tuple[GlobalIdentityManager, io.StringIO]:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        manager = GlobalIdentityManager(
            fps=30.0,
            frame_width=1280,
            frame_height=720,
            occlusion_frames=occlusion_frames,
            active_occlusion_frames=90,
            active_reassoc_max_distance=300.0,
            max_distance_fraction=0.18,
            min_score=0.65,
            short_gap_frames=30,
            short_gap_max_distance=150.0,
            motion_tolerance=1.5,
            max_dynamic_distance=300.0,
            size_ratio_min=0.25,
            size_ratio_max=4.0,
            hard_size_ratio_min=0.10,
            hard_size_ratio_max=8.0,
            zones=zones,
            lifecycle_enabled=bool(zones),
            handoff_hold_frames=handoff_hold_frames,
            endpoint_margin_px=50.0,
            destination_proximity_px=60.0,
            event_writer=writer,
            debug=False,
            debug_verbose=False,
        )
        return manager, stream

    @staticmethod
    def observation(local_id: int, x: float, y: float = 100.0) -> IdentityObservation:
        return IdentityObservation(local_id, (x, y), (80.0, 40.0))

    def test_new_local_id_reassociates_to_occluded_global(self) -> None:
        manager, stream = self.make_manager()
        self.assertEqual(manager.process_frame([self.observation(1, 100)], 0), {1: 1})
        self.assertEqual(manager.process_frame([self.observation(1, 110)], 1), {1: 1})
        manager.process_frame([], 2)
        self.assertEqual(manager.global_rolls[1].state, OCCLUDED)
        self.assertEqual(manager.process_frame([self.observation(8, 130)], 3), {8: 1})
        self.assertEqual(manager.global_rolls[1].state, VISIBLE)
        self.assertEqual(manager.global_rolls[1].current_local_id, 8)
        self.assertNotIn(1, manager.local_to_global)
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        reassociation = [
            event for event in events if event["event"] == "GLOBAL_REASSOCIATED"
        ]
        self.assertEqual(len(reassociation), 1)
        self.assertEqual(reassociation[0]["old_local_id"], "1")
        self.assertEqual(reassociation[0]["new_local_id"], "8")

    def test_motion_state_never_changes_identity(self) -> None:
        manager, _ = self.make_manager()
        manager.process_frame([self.observation(1, 100)], 0)
        manager.update_motion_state(1, False, 0)
        manager.update_motion_state(1, True, 1)
        manager.update_motion_state(1, False, 2)
        self.assertEqual(manager.local_to_global, {1: 1})
        self.assertEqual(len(manager.global_rolls), 1)

    def test_placement_seals_global_and_later_pickup_gets_new_id(self) -> None:
        manager, stream = self.make_manager()
        manager.process_frame([self.observation(1, 100)], 0)
        state = manager.global_rolls[1]
        state.actual_bin_id = "BIN_3"
        state.bin_confirmed = True
        manager.seal_by_placement(1, 1)
        self.assertEqual(state.state, PLACED)
        self.assertFalse(state.reassociation_allowed)
        self.assertNotIn(1, manager.local_to_global)

        mapping = manager.process_frame([self.observation(8, 105)], 2)
        self.assertEqual(mapping, {8: 2})
        self.assertEqual(state.state, PLACED)
        self.assertEqual(state.local_attachment_history, [(0, 1)])
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertIn("GLOBAL_PLACED", [row["event"] for row in events])
        self.assertIn(
            "REASSOCIATION_REJECTED_TERMINAL_GLOBAL",
            [row["event"] for row in events],
        )

    def test_pending_placement_accepts_near_same_bin_fragment(self) -> None:
        manager, _ = self.make_manager()
        manager.process_frame([self.observation(1, 100)], 0)
        manager.process_frame([], 1)
        state = manager.global_rolls[1]
        state.state = PLACEMENT_PENDING
        state.related_bin_id = "BIN_3"
        state.bin_motion_relation = ENTERING_BIN
        fragment = IdentityObservation(
            8,
            (110.0, 100.0),
            (80.0, 40.0),
            related_bin_id="BIN_3",
            bin_motion_relation=ENTERING_BIN,
        )
        self.assertEqual(manager.process_frame([fragment], 2), {8: 1})
        self.assertEqual(state.current_local_id, 8)

    def test_pending_placement_rejects_opposite_bin_flow(self) -> None:
        manager, stream = self.make_manager()
        manager.process_frame([self.observation(1, 100)], 0)
        manager.process_frame([], 1)
        state = manager.global_rolls[1]
        state.state = PLACEMENT_PENDING
        state.related_bin_id = "BIN_3"
        state.bin_motion_relation = ENTERING_BIN
        pickup = IdentityObservation(
            8,
            (105.0, 100.0),
            (80.0, 40.0),
            centroid_history=((1, 100.0, 100.0), (2, 105.0, 100.0)),
            related_bin_id="BIN_3",
            bin_motion_relation=LEAVING_BIN,
            is_active=True,
            became_active=True,
        )
        self.assertEqual(manager.process_frame([pickup], 2), {8: 2})
        self.assertEqual(state.state, PLACEMENT_PENDING)
        self.assertEqual(
            manager.event_counts["REASSOCIATION_REJECTED_OPPOSITE_BIN_FLOW"],
            1,
        )
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertIn(
            "REASSOCIATION_REJECTED_OPPOSITE_BIN_FLOW",
            [row["event"] for row in events],
        )

    def test_edge_on_size_change_does_not_veto_strong_motion_match(self) -> None:
        manager, _ = self.make_manager()
        manager.process_frame([self.observation(1, 100)], 0)
        manager.process_frame([self.observation(1, 110)], 1)
        manager.process_frame([], 2)
        edge_on = IdentityObservation(8, (130.0, 100.0), (80.0, 20.0))
        self.assertEqual(manager.process_frame([edge_on], 3), {8: 1})

    def test_real_short_gap_sequence_reuses_global_id(self) -> None:
        manager, stream = self.make_manager()
        manager.process_frame([self.observation(1, 100)], 69)
        manager.process_frame([], 70)
        reappeared = IdentityObservation(2, (112.4, 100.0), (72.0, 40.0))
        self.assertEqual(manager.process_frame([reappeared], 86), {2: 1})
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        reassociation = next(
            event for event in events if event["event"] == "GLOBAL_REASSOCIATED"
        )
        self.assertEqual(reassociation["gap_frames"], "16")
        self.assertEqual(reassociation["decision_reason"], "short_gap_spatial_match")
        self.assertAlmostEqual(float(reassociation["association_distance"]), 12.4)

    def test_short_gap_accepts_ratio_just_below_old_minimum(self) -> None:
        manager, _ = self.make_manager()
        manager.process_frame([self.observation(12, 100)], 678)
        manager.process_frame([], 679)
        changed = IdentityObservation(
            17,
            (219.3, 100.0),
            (80.0, 40.0 * 0.493),
        )
        self.assertEqual(manager.process_frame([changed], 692), {17: 1})

    def test_short_gap_accepts_ratio_above_old_maximum(self) -> None:
        manager, _ = self.make_manager()
        manager.process_frame([self.observation(25, 100)], 886)
        manager.process_frame([], 887)
        changed = IdentityObservation(
            26,
            (168.5, 100.0),
            (80.0, 40.0 * 2.488),
        )
        self.assertEqual(manager.process_frame([changed], 898), {26: 1})

    def test_active_roll_reassociates_after_45_missing_frames(self) -> None:
        manager, stream = self.make_manager()
        for frame, x in zip(range(581, 586), (100, 108, 116, 124, 132)):
            manager.process_frame([self.observation(10, x)], frame)
            manager.update_motion_state(1, True, frame)
        manager.process_frame([], 586)
        reappeared = IdentityObservation(
            12,
            (320.0, 100.0),
            (80.0, 40.0 * 2.7),
        )
        self.assertEqual(manager.process_frame([reappeared], 631), {12: 1})
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        reassociation = next(
            event for event in events if event["event"] == "GLOBAL_REASSOCIATED"
        )
        self.assertEqual(reassociation["gap_frames"], "45")
        self.assertEqual(reassociation["old_local_id"], "10")
        self.assertEqual(reassociation["new_local_id"], "12")
        self.assertEqual(
            reassociation["decision_reason"], "active_occlusion_spatial_match"
        )
        self.assertEqual(reassociation["was_active_before_occlusion"], "True")

    def test_static_roll_keeps_conservative_long_gap_rules(self) -> None:
        manager, _ = self.make_manager()
        manager.process_frame([self.observation(10, 132)], 585)
        manager.update_motion_state(1, False, 585)
        manager.process_frame([], 586)
        reappeared = IdentityObservation(
            12,
            (320.0, 100.0),
            (80.0, 40.0 * 2.7),
        )
        self.assertEqual(manager.process_frame([reappeared], 631), {12: 2})

    def test_explicit_static_local_remains_unassigned(self) -> None:
        manager, stream = self.make_manager(zones=self.lifecycle_zones())
        mapping = manager.process_frame(
            [self.observation(5, 50)],
            0,
            eligible_new_local_ids=set(),
            create_global_local_ids=set(),
        )
        self.assertEqual(mapping, {})
        self.assertEqual(manager.global_rolls, {})
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertNotIn("GLOBAL_CREATED", [row["event"] for row in events])

    def test_active_pickup_commits_one_transport_global(self) -> None:
        manager, stream = self.make_manager(zones=self.lifecycle_zones())
        mapping = manager.process_frame(
            [self.observation(7, 50)],
            10,
            eligible_new_local_ids={7},
            create_global_local_ids={7},
        )
        self.assertEqual(mapping, {7: 1})
        self.assertEqual(manager.global_rolls[1].state, ACTIVE)
        self.assertTrue(manager.global_rolls[1].journey_started)
        self.assertEqual(manager.global_rolls[1].journey_direction, "A_TO_B")
        self.assertEqual(manager.global_rolls[1].destination_zone, "B")
        self.assertEqual(manager.static_global_creation_count, 0)
        events = [
            row["event"]
            for row in csv.DictReader(io.StringIO(stream.getvalue()))
        ]
        self.assertEqual(events.count("GLOBAL_CREATED"), 1)
        self.assertEqual(events.count("GLOBAL_PICKED_UP"), 1)

    def test_only_active_fragment_can_reassociate(self) -> None:
        manager, _ = self.make_manager(zones=self.lifecycle_zones())
        manager.process_frame(
            [self.observation(7, 50)],
            0,
            eligible_new_local_ids={7},
            create_global_local_ids={7},
        )
        manager.process_frame([], 1, set(), set())

        static_mapping = manager.process_frame(
            [self.observation(8, 60)],
            2,
            eligible_new_local_ids=set(),
            create_global_local_ids=set(),
        )
        self.assertEqual(static_mapping, {})
        self.assertEqual(manager.global_rolls[1].state, OCCLUDED)

        active_mapping = manager.process_frame(
            [self.observation(8, 60)],
            3,
            eligible_new_local_ids={8},
            create_global_local_ids=set(),
        )
        self.assertEqual(active_mapping, {8: 1})

    def test_pickup_preserves_precommit_motion_for_reassociation(self) -> None:
        manager, _ = self.make_manager()
        pickup = IdentityObservation(
            26,
            (26.0, 304.0),
            (30.0, 10.0),
            centroid_history=(
                (881, 21.0, 317.0),
                (882, 23.0, 314.0),
                (883, 23.0, 312.0),
                (884, 23.0, 307.0),
                (885, 26.0, 304.0),
            ),
        )
        created = manager.process_frame(
            [pickup],
            885,
            eligible_new_local_ids={26},
            create_global_local_ids={26},
            create_global_origins={26: "A"},
        )
        global_id = created[26]
        manager.process_frame([], 886, set(), set())
        continued = manager.process_frame(
            [IdentityObservation(27, (163.0, 218.0), (48.0, 16.0))],
            906,
            eligible_new_local_ids={27},
            create_global_local_ids={27},
        )
        self.assertEqual(continued[27], global_id)
        self.assertEqual(manager.event_counts["GLOBAL_CREATED"], 1)
        self.assertEqual(manager.event_counts["GLOBAL_REASSOCIATED"], 1)

    @staticmethod
    def lifecycle_zones() -> dict[str, np.ndarray]:
        return {
            "zone_a": np.array(
                [[0, 0], [100, 0], [100, 200], [0, 200]], np.float32
            ),
            "tracking_zone": np.array(
                [[0, 0], [450, 0], [450, 200], [0, 200]], np.float32
            ),
            "zone_b": np.array(
                [[400, 0], [500, 0], [500, 200], [400, 200]], np.float32
            ),
        }

    def start_pickup_journey(self, manager: GlobalIdentityManager) -> None:
        manager.process_frame(
            [self.observation(10, 50)],
            0,
            eligible_new_local_ids={10},
            create_global_local_ids={10},
            create_global_origins={10: "A"},
        )
        self.assertEqual(manager.global_rolls[1].state, ACTIVE)

    def test_completed_global_is_terminal_and_new_roll_gets_new_id(self) -> None:
        manager, stream = self.make_manager(zones=self.lifecycle_zones())
        self.start_pickup_journey(manager)
        manager.process_frame([self.observation(10, 420)], 1)
        self.assertTrue(manager.global_rolls[1].entered_destination)
        manager.process_frame([], 2)
        self.assertEqual(manager.global_rolls[1].state, COMPLETED)

        mapping = manager.process_frame(
            [self.observation(12, 425)],
            3,
            eligible_new_local_ids={12},
            create_global_local_ids={12},
            create_global_origins={12: "B"},
        )
        self.assertEqual(mapping, {12: 2})
        self.assertEqual(manager.global_rolls[1].state, COMPLETED)
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertIn("GLOBAL_PICKED_UP", [row["event"] for row in events])
        self.assertIn(
            "GLOBAL_ENTERED_DESTINATION", [row["event"] for row in events]
        )
        completed = next(
            row for row in events if row["event"] == "GLOBAL_COMPLETED"
        )
        self.assertEqual(
            completed["decision_reason"], "disappeared_after_destination"
        )

    def test_handoff_hold_completes_visible_roll(self) -> None:
        manager, _ = self.make_manager(
            zones=self.lifecycle_zones(), handoff_hold_frames=3
        )
        self.start_pickup_journey(manager)
        for frame in range(1, 4):
            manager.process_frame([self.observation(10, 420)], frame)
            manager.update_motion_state(1, True, frame)
        self.assertEqual(manager.global_rolls[1].state, COMPLETED)
        self.assertEqual(
            manager.global_rolls[1].completion_reason,
            "destination_hold_reached",
        )

    def test_b_to_a_journey_completes_only_at_zone_a(self) -> None:
        manager, stream = self.make_manager(
            zones=self.lifecycle_zones(), handoff_hold_frames=2
        )
        manager.process_frame(
            [self.observation(20, 450)],
            0,
            eligible_new_local_ids={20},
            create_global_local_ids={20},
            create_global_origins={20: "B"},
        )
        state = manager.global_rolls[1]
        self.assertEqual(state.journey_direction, "B_TO_A")
        self.assertEqual(state.origin_zone, "B")
        self.assertEqual(state.destination_zone, "A")
        manager.process_frame([self.observation(20, 450)], 1)
        self.assertFalse(state.entered_destination)
        self.assertNotEqual(state.state, COMPLETED)
        manager.process_frame([self.observation(20, 50)], 2)
        manager.process_frame([self.observation(20, 50)], 3)
        self.assertEqual(state.state, COMPLETED)
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        entered = next(
            row
            for row in events
            if row["event"] == "GLOBAL_ENTERED_DESTINATION"
        )
        self.assertEqual(entered["origin_zone"], "B")
        self.assertEqual(entered["destination_zone"], "A")
        self.assertEqual(entered["journey_direction"], "B_TO_A")

    def test_b_to_a_near_destination_completes_only_after_disappearance(self) -> None:
        manager, stream = self.make_manager(zones=self.lifecycle_zones())
        manager.process_frame(
            [self.observation(20, 450)],
            0,
            eligible_new_local_ids={20},
            create_global_local_ids={20},
            create_global_origins={20: "B"},
        )
        manager.process_frame([self.observation(20, 250)], 1)
        manager.process_frame([self.observation(20, 150)], 2)
        state = manager.global_rolls[1]
        self.assertTrue(state.approaching_destination)
        self.assertAlmostEqual(state.last_distance_to_destination, 50.0)
        self.assertFalse(state.entered_destination_zone)
        self.assertNotEqual(state.state, COMPLETED)

        manager.process_frame([], 3)

        self.assertEqual(state.state, COMPLETED)
        self.assertTrue(state.entered_destination)
        self.assertFalse(state.entered_destination_zone)
        self.assertEqual(
            state.completion_reason, "disappeared_at_destination_exit"
        )
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        entered = next(
            row for row in events if row["event"] == "GLOBAL_ENTERED_DESTINATION"
        )
        self.assertEqual(entered["decision_reason"], "destination_exit_proximity")

    def test_destination_proximity_alone_does_not_complete(self) -> None:
        manager, _ = self.make_manager(zones=self.lifecycle_zones())
        manager.process_frame(
            [self.observation(20, 450)],
            0,
            eligible_new_local_ids={20},
            create_global_local_ids={20},
            create_global_origins={20: "B"},
        )
        manager.process_frame([self.observation(20, 250)], 1)
        manager.process_frame([self.observation(20, 150)], 2)
        state = manager.global_rolls[1]
        self.assertTrue(state.approaching_destination)
        self.assertNotEqual(state.state, COMPLETED)
        self.assertFalse(state.entered_destination)

    def test_moving_away_from_near_destination_does_not_complete(self) -> None:
        manager, _ = self.make_manager(zones=self.lifecycle_zones())
        manager.process_frame(
            [self.observation(20, 450)],
            0,
            eligible_new_local_ids={20},
            create_global_local_ids={20},
            create_global_origins={20: "B"},
        )
        manager.process_frame([self.observation(20, 250)], 1)
        manager.process_frame([self.observation(20, 150)], 2)
        manager.process_frame([self.observation(20, 170)], 3)
        manager.process_frame([], 4)
        state = manager.global_rolls[1]
        self.assertEqual(state.state, OCCLUDED)
        self.assertFalse(state.entered_destination)

    def test_ambiguous_candidate_creates_new_global(self) -> None:
        manager, stream = self.make_manager()
        manager.process_frame(
            [self.observation(1, 100), self.observation(2, 105)], 0
        )
        manager.process_frame([], 1)
        mapping = manager.process_frame([self.observation(8, 103)], 2)
        self.assertEqual(mapping, {8: 3})
        self.assertEqual(manager.global_rolls[1].state, OCCLUDED)
        self.assertEqual(manager.global_rolls[2].state, OCCLUDED)
        events = list(csv.DictReader(io.StringIO(stream.getvalue())))
        ambiguous = [
            event
            for event in events
            if event["event"] == "AMBIGUOUS_REASSOCIATION"
        ]
        self.assertEqual(len(ambiguous), 2)

    def test_global_identity_expires_only_after_window(self) -> None:
        manager, _ = self.make_manager(occlusion_frames=2)
        manager.process_frame([self.observation(1, 100)], 0)
        manager.process_frame([], 1)
        manager.process_frame([], 2)
        self.assertEqual(manager.global_rolls[1].state, OCCLUDED)
        manager.process_frame([], 3)
        self.assertEqual(manager.global_rolls[1].state, LOST)


if __name__ == "__main__":
    unittest.main()
