"""Application-level roll identity above temporary local tracker IDs."""

from __future__ import annotations

import csv
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Sequence

import numpy as np
import cv2


NEW = "NEW"
VISIBLE = "VISIBLE"
ACTIVE = "ACTIVE"
OCCLUDED = "OCCLUDED"
PLACEMENT_PENDING = "PLACEMENT_PENDING"
PLACED = "PLACED"
COMPLETED = "COMPLETED"
LOST = "LOST"
VISIBLE_STATES = frozenset({NEW, VISIBLE, ACTIVE})
TERMINAL_STATES = frozenset({PLACED, COMPLETED})

ENTERING_BIN = "ENTERING_BIN"
LEAVING_BIN = "LEAVING_BIN"
NEUTRAL = "NEUTRAL"
PLACEMENT_PENDING_DISTANCE_FRACTION = 0.40
PLACEMENT_PENDING_MAX_SPEED_PX_PER_FRAME = 5.0

# Per-roll placement verification is deliberately independent of tracker and
# camera lifecycle state.  UNRESOLVED means no final physical placement has
# been confirmed yet; UNVERIFIED means placement is known but the upstream
# expected-bin assignment is not.
UNRESOLVED = "UNRESOLVED"
UNVERIFIED = "UNVERIFIED"
CORRECT = "CORRECT"
WRONG_BIN = "WRONG_BIN"

# Geometry-only priorities for same-camera reassociation.
POSITION_WEIGHT = 0.45
TIME_WEIGHT = 0.25
MOTION_PREDICTION_WEIGHT = 0.20
SIZE_WEIGHT = 0.10
SHORT_GAP_SCORE_BONUS = 0.10
ACTIVE_POSITION_WEIGHT = 0.35
ACTIVE_TIME_WEIGHT = 0.20
ACTIVE_DIRECTION_WEIGHT = 0.25
ACTIVE_SIZE_WEIGHT = 0.10
ACTIVE_PRIOR_BONUS = 0.10
VELOCITY_HISTORY_POINTS = 5
DESTINATION_APPROACH_MIN_FRAMES = 2
A_TO_B = "A_TO_B"
B_TO_A = "B_TO_A"


@dataclass(frozen=True)
class IdentityObservation:
    local_id: int
    centroid: tuple[float, float]
    box_size: tuple[float, float]
    # Local tracks can exist for several STATIC/PENDING frames before pickup.
    # Preserve that evidence so a new transport identity has a real velocity.
    centroid_history: tuple[tuple[int, float, float], ...] = ()
    related_bin_id: str = ""
    bin_motion_relation: str = NEUTRAL
    is_active: bool = False
    became_active: bool = False


@dataclass(frozen=True)
class ReassociationAssessment:
    global_id: int
    gap_frames: int
    last_seen_distance: float
    predicted_distance: float
    association_distance: float
    recent_speed: float
    dynamic_allowed_distance: float
    was_active_before_occlusion: bool
    previous_speed: float
    direction_score: float
    active_bonus: float
    active_allowed_distance: float
    size_ratio: float
    position_score: float
    time_score: float
    motion_score: float
    size_score: float
    score: float
    valid: bool
    reason: str


@dataclass
class GlobalRollState:
    global_id: int
    current_local_id: int
    state: str
    first_seen_frame: int
    last_seen_frame: int
    last_centroid: tuple[float, float]
    velocity: tuple[float, float]
    last_box_size: tuple[float, float]
    missing_frames: int = 0
    centroid_history: Deque[tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=VELOCITY_HISTORY_POINTS)
    )
    last_motion_active: bool | None = None
    reassociation_count: int = 0
    was_active_before_occlusion: bool = False
    last_velocity: tuple[float, float] = (0.0, 0.0)
    last_speed: float = 0.0
    last_motion_direction: tuple[float, float] = (0.0, 0.0)
    last_seen_centroid: tuple[float, float] | None = None
    last_seen_box_size: tuple[float, float] | None = None
    occlusion_start_frame: int | None = None
    journey_started: bool = False
    origin_zone: str = ""
    destination_zone: str = ""
    journey_direction: str = ""
    activation_reason: str = ""
    entered_destination: bool = False
    entered_destination_zone: bool = False
    destination_entered_frame: int | None = None
    destination_consecutive_frames: int = 0
    last_distance_to_destination: float | None = None
    min_distance_to_destination: float | None = None
    approaching_destination: bool = False
    destination_approach_frames: int = 0
    last_zone: str = "UNCONFIGURED"
    completion_reason: str = ""
    in_zone_a: bool = False
    in_tracking_zone: bool = False
    in_zone_b: bool = False
    ever_entered_zone_a: bool = False
    ever_entered_zone_b: bool = False
    candidate_bin_id: str = ""
    candidate_bin_frames: int = 0
    candidate_bin_max_overlap: float = 0.0
    candidate_bin_last_overlap: float = 0.0
    bin_placement_state: str = "NONE"
    candidate_start_frame: int | None = None
    candidate_start_time: float | None = None
    candidate_last_valid_frame: int | None = None
    candidate_last_valid_time: float | None = None
    candidate_dwell_seconds: float = 0.0
    candidate_valid_overlap_seconds: float = 0.0
    candidate_current_speed: float = 0.0
    candidate_bins_seen: list[str] = field(default_factory=list)
    last_candidate_cancel_reason: str = ""
    actual_bin_id: str = ""
    bin_confirmed: bool = False
    bin_entry_frame: int | None = None
    bin_max_overlap: float = 0.0
    placement_counted: bool = False
    placement_dwell_seconds: float = 0.0
    placement_settled_speed: float = 0.0
    placement_reason: str = ""
    disappeared_inside_candidate: bool = False
    provisional_placement_seen: bool = False
    provisional_start_frame: int | None = None
    provisional_start_time: float | None = None
    reappeared_outside_candidate: bool = False
    placement_confirmation_method: str = ""
    placement_owner_static_local_id: int | None = None
    factory_placement_completed: bool = False
    possible_detector_miss_bin_id: str = ""
    bin_entry_initialized: bool = False
    bin_entry_armed: bool = False
    initial_bin_id: str = ""
    factory_roll_id: str | None = None
    expected_bin_id: str | None = None
    verification_result: str = UNRESOLVED
    placement_status: str = "TRACKING"
    placement_evidence: str = "NONE"
    placement_confidence: float = 0.0
    entered_bin_mouth: bool = False
    entry_frame: int | None = None
    entry_timestamp: float | None = None
    entry_bin_id: str = ""
    entry_direction_score: float = 0.0
    entry_trajectory_consistency: float = 0.0
    last_visible_centroid: tuple[float, float] | None = None
    reassociation_allowed: bool = True
    placement_confirmed_frame: int | None = None
    bin_motion_relation: str = NEUTRAL
    related_bin_id: str = ""
    local_attachment_history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def roll_id(self) -> str:
        """Return the best available application identity without inventing one."""

        return self.factory_roll_id or f"G{self.global_id}"

    @property
    def placement_confirmed(self) -> bool:
        """Business-facing alias for the legacy bin-confirmation field."""

        return self.bin_confirmed

    @property
    def placement_frame(self) -> int | None:
        """Business-facing alias for the legacy bin-entry frame field."""

        return self.bin_entry_frame

    def observe(self, observation: IdentityObservation, frame_index: int) -> None:
        self.last_seen_frame = frame_index
        self.last_centroid = observation.centroid
        self.last_box_size = observation.box_size
        self.missing_frames = 0
        if self.state not in TERMINAL_STATES | {PLACEMENT_PENDING}:
            self.state = ACTIVE if self.journey_started else VISIBLE
        self.centroid_history.append(
            (frame_index, observation.centroid[0], observation.centroid[1])
        )
        self.velocity = estimate_velocity(self.centroid_history)

    def begin_occlusion(self, frame_index: int) -> None:
        """Freeze the motion evidence from the last visible detection."""

        if self.state in TERMINAL_STATES:
            return
        self.state = OCCLUDED
        self.occlusion_start_frame = frame_index
        self.was_active_before_occlusion = bool(self.last_motion_active)
        self.last_velocity = self.velocity
        self.last_speed = math.hypot(*self.last_velocity)
        if self.last_speed > 1e-6:
            self.last_motion_direction = (
                self.last_velocity[0] / self.last_speed,
                self.last_velocity[1] / self.last_speed,
            )
        else:
            self.last_motion_direction = (0.0, 0.0)
        self.last_seen_centroid = self.last_centroid
        self.last_seen_box_size = self.last_box_size


def box_size_from_corners(corners: np.ndarray) -> tuple[float, float]:
    """Return rotation-independent long and short OBB side lengths."""

    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected four OBB corners, received shape {points.shape}")
    side_lengths = [
        float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
        for index in range(4)
    ]
    long_side = max(side_lengths)
    short_side = min(side_lengths)
    return long_side, short_side


def estimate_velocity(
    history: Sequence[tuple[int, float, float]],
) -> tuple[float, float]:
    """Estimate pixels-per-frame velocity from recent global observations."""

    if len(history) < 2:
        return 0.0, 0.0
    first = history[0]
    last = history[-1]
    frame_delta = last[0] - first[0]
    if frame_delta <= 0:
        return 0.0, 0.0
    return (
        (last[1] - first[1]) / frame_delta,
        (last[2] - first[2]) / frame_delta,
    )


class GlobalIdentityManager:
    """Conservatively reconnect new local IDs to occluded global rolls."""

    def __init__(
        self,
        *,
        fps: float,
        frame_width: int,
        frame_height: int,
        occlusion_frames: int,
        active_occlusion_frames: int,
        active_reassoc_max_distance: float,
        max_distance_fraction: float,
        min_score: float,
        short_gap_frames: int,
        short_gap_max_distance: float,
        motion_tolerance: float,
        max_dynamic_distance: float,
        size_ratio_min: float,
        size_ratio_max: float,
        hard_size_ratio_min: float,
        hard_size_ratio_max: float,
        zones: dict[str, np.ndarray] | None,
        lifecycle_enabled: bool,
        handoff_hold_frames: int,
        endpoint_margin_px: float,
        destination_proximity_px: float,
        event_writer: csv.DictWriter,
        debug: bool,
        debug_verbose: bool = False,
    ) -> None:
        self.fps = fps
        self.frame_diagonal = math.hypot(frame_width, frame_height)
        self.occlusion_frames = occlusion_frames
        self.active_occlusion_frames = active_occlusion_frames
        self.active_reassoc_max_distance = active_reassoc_max_distance
        self.max_distance_fraction = max_distance_fraction
        self.min_score = min_score
        self.short_gap_frames = short_gap_frames
        self.short_gap_max_distance = short_gap_max_distance
        self.motion_tolerance = motion_tolerance
        self.max_dynamic_distance = max_dynamic_distance
        self.size_ratio_min = size_ratio_min
        self.size_ratio_max = size_ratio_max
        self.hard_size_ratio_min = hard_size_ratio_min
        self.hard_size_ratio_max = hard_size_ratio_max
        self.zones = zones or {}
        self.lifecycle_enabled = lifecycle_enabled
        self.handoff_hold_frames = handoff_hold_frames
        self.endpoint_margin_px = endpoint_margin_px
        self.destination_proximity_px = destination_proximity_px
        self.event_writer = event_writer
        self.debug = debug
        self.debug_verbose = debug_verbose
        self.next_global_id = 1
        self.global_rolls: dict[int, GlobalRollState] = {}
        self.local_to_global: dict[int, int] = {}
        self.event_counts: Counter[str] = Counter()
        self.frame_transition_events: list[tuple[int, str]] = []
        self.static_global_creation_count = 0
        self.possible_false_global_reuse_events: list[
            tuple[int, int, int, str]
        ] = []

    def seal_by_placement(
        self, global_id: int, frame_index: int
    ) -> tuple[int, ...]:
        """Make a placed transport journey terminal and detach local tracks."""

        global_state = self.global_rolls[global_id]
        if global_state.state == PLACED:
            return ()
        global_state.state = PLACED
        global_state.reassociation_allowed = False
        global_state.placement_confirmed_frame = frame_index
        global_state.missing_frames = 0
        detached = tuple(
            local_id
            for local_id, mapped_global_id in list(self.local_to_global.items())
            if mapped_global_id == global_id
        )
        for local_id in detached:
            self.local_to_global.pop(local_id, None)
        self._log_event(
            frame_index,
            global_id,
            global_state.current_local_id,
            None,
            "GLOBAL_PLACED",
            decision_reason=(
                f"placement_confirmed:{global_state.actual_bin_id or 'UNKNOWN'}"
            ),
        )
        self.frame_transition_events.append((global_id, "GLOBAL_PLACED"))
        if self.debug:
            print(
                f"[frame {frame_index}] GLOBAL G{global_id} -> PLACED "
                f"in {global_state.actual_bin_id}; reassociation disabled"
            )
        return detached

    def process_frame(
        self,
        observations: Sequence[IdentityObservation],
        frame_index: int,
        eligible_new_local_ids: set[int] | None = None,
        create_global_local_ids: set[int] | None = None,
        create_global_origins: dict[int, str] | None = None,
        create_global_reasons: dict[int, str] | None = None,
    ) -> dict[int, int]:
        """Update identity state and return local-to-global IDs for this frame."""

        self.frame_transition_events = []
        by_local = {observation.local_id: observation for observation in observations}
        legacy_mode = (
            eligible_new_local_ids is None and create_global_local_ids is None
        )
        eligible_ids = (
            set(by_local) if eligible_new_local_ids is None else eligible_new_local_ids
        )
        create_ids = (
            set(by_local) if create_global_local_ids is None else create_global_local_ids
        )
        origin_by_local = create_global_origins or {}
        reason_by_local = create_global_reasons or {}
        transport_start_ids = set() if legacy_mode else create_ids
        seen_globals: set[int] = set()
        new_observations: list[IdentityObservation] = []

        # First update local IDs which are already mapped to a live global roll.
        for local_id, observation in by_local.items():
            global_id = self.local_to_global.get(local_id)
            global_state = self.global_rolls.get(global_id) if global_id else None
            if global_state is None or global_state.state == LOST:
                self.local_to_global.pop(local_id, None)
                if local_id in eligible_ids:
                    new_observations.append(observation)
                continue
            if global_state.state == PLACED:
                self.local_to_global.pop(local_id, None)
                self._log_terminal_rejection(
                    global_state, observation, frame_index
                )
                if local_id in eligible_ids:
                    new_observations.append(observation)
                continue
            if global_state.state == COMPLETED:
                # Keep the final local track attached while it is still visible,
                # but never update or reopen the terminal global identity.
                seen_globals.add(global_state.global_id)
                continue
            if global_state.state == PLACEMENT_PENDING and not self._placement_pending_compatible(
                global_state, observation
            ):
                self.local_to_global.pop(local_id, None)
                event = (
                    "REASSOCIATION_REJECTED_OPPOSITE_BIN_FLOW"
                    if self._has_opposite_bin_flow(global_state, observation)
                    else "REASSOCIATION_REJECTED_PLACEMENT_PENDING"
                )
                self._log_event(
                    frame_index,
                    global_state.global_id,
                    global_state.current_local_id,
                    local_id,
                    event,
                    decision_reason="placement_pending_incompatible_fragment",
                )
                self.possible_false_global_reuse_events.append(
                    (frame_index, global_state.global_id, local_id, event)
                )
                if local_id in eligible_ids:
                    new_observations.append(observation)
                continue
            was_occluded = global_state.state in {OCCLUDED, PLACEMENT_PENDING}
            global_state.observe(observation, frame_index)
            if observation.related_bin_id:
                global_state.related_bin_id = observation.related_bin_id
            if observation.bin_motion_relation != NEUTRAL:
                global_state.bin_motion_relation = observation.bin_motion_relation
            seen_globals.add(global_state.global_id)
            if was_occluded:
                self._log_event(
                    frame_index,
                    global_state.global_id,
                    local_id,
                    local_id,
                    "GLOBAL_VISIBLE",
                )
                if self.debug:
                    print(
                        f"[frame {frame_index}] GLOBAL G{global_state.global_id} "
                        f"VISIBLE again on local ID {local_id}"
                    )
            self._update_zone_state(global_state, observation, frame_index)

        # Make disappeared visible rolls available to same-frame new local IDs.
        for global_state in self.global_rolls.values():
            if (
                global_state.state not in VISIBLE_STATES
                or global_state.global_id in seen_globals
            ):
                continue
            if global_state.journey_started and global_state.entered_destination:
                if self.debug:
                    print(
                        f"[frame {frame_index}] G{global_state.global_id} "
                        "disappeared after entering destination zone"
                    )
                self._complete_global(
                    global_state,
                    frame_index,
                    "disappeared_after_destination",
                )
            elif self._can_complete_at_destination_exit(global_state):
                if self.debug:
                    distance = global_state.last_distance_to_destination
                    print(
                        f"[frame {frame_index}] G{global_state.global_id} "
                        "disappeared at destination exit "
                        f"({distance:.1f} px from ZONE A)"
                    )
                self._mark_destination_entered(
                    global_state,
                    frame_index,
                    "destination_exit_proximity",
                )
                self._complete_global(
                    global_state,
                    frame_index,
                    "disappeared_at_destination_exit",
                )
            else:
                global_state.begin_occlusion(frame_index)
                global_state.missing_frames = (
                    frame_index - global_state.last_seen_frame
                )
                self._log_event(
                    frame_index,
                    global_state.global_id,
                    global_state.current_local_id,
                    None,
                    "GLOBAL_OCCLUDED",
                )
                if self.debug:
                    print(
                        f"[frame {frame_index}] GLOBAL G{global_state.global_id} "
                        f"OCCLUDED, last local ID {global_state.current_local_id}"
                    )

        # A terminal global keeps its last local mapping only while that exact
        # local track remains visible. This prevents later local-ID reuse.
        for global_state in self.global_rolls.values():
            if global_state.state not in TERMINAL_STATES:
                continue
            local_id = global_state.current_local_id
            if local_id not in by_local and self.local_to_global.get(local_id) == global_state.global_id:
                self.local_to_global.pop(local_id, None)

        # Local IDs unknown to the identity layer may match an occluded global.
        for local_id, observation in by_local.items():
            if (
                local_id in eligible_ids
                and local_id not in self.local_to_global
                and observation not in new_observations
            ):
                new_observations.append(observation)
        self._associate_new_observations(
            new_observations,
            frame_index,
            create_ids,
            transport_start_ids,
            origin_by_local,
            reason_by_local,
        )

        # Expire only after the configured application-level occlusion window.
        for global_state in self.global_rolls.values():
            if global_state.state != OCCLUDED:
                continue
            global_state.missing_frames = frame_index - global_state.last_seen_frame
            retention_window = self._retention_window(global_state)
            if global_state.missing_frames <= retention_window:
                continue
            global_state.state = LOST
            old_local_id = global_state.current_local_id
            if self.local_to_global.get(old_local_id) == global_state.global_id:
                self.local_to_global.pop(old_local_id, None)
            self._log_event(
                frame_index,
                global_state.global_id,
                old_local_id,
                None,
                "GLOBAL_LOST",
            )
            if self.debug:
                print(
                    f"[frame {frame_index}] GLOBAL G{global_state.global_id} LOST"
                )

        return {
            local_id: self.local_to_global[local_id]
            for local_id in by_local
            if local_id in self.local_to_global
        }

    def update_motion_state(
        self, global_id: int, is_active: bool, frame_index: int
    ) -> None:
        """Update motion state without changing an existing journey identity."""

        global_state = self.global_rolls[global_id]
        if global_state.state in TERMINAL_STATES | {LOST}:
            return
        previous = global_state.last_motion_active
        global_state.last_motion_active = is_active
        if previous is not None and previous != is_active and self.debug:
            old_label = "ACTIVE" if previous else "STATIC"
            new_label = "ACTIVE" if is_active else "STATIC"
            print(
                f"[frame {frame_index}] GLOBAL G{global_id} "
                f"{old_label} -> {new_label}"
            )

    def _update_zone_state(
        self,
        global_state: GlobalRollState,
        observation: IdentityObservation,
        frame_index: int,
    ) -> None:
        zone_a = self.zones.get("zone_a")
        zone_b = self.zones.get("zone_b")
        tracking = self.zones.get("tracking_zone")
        point = observation.centroid

        previous_membership = {
            "zone_a": global_state.in_zone_a,
            "tracking_zone": global_state.in_tracking_zone,
            "zone_b": global_state.in_zone_b,
        }

        in_zone_a = (
            zone_a is not None
            and cv2.pointPolygonTest(zone_a, point, False) >= 0
        )
        in_zone_b = (
            zone_b is not None
            and cv2.pointPolygonTest(zone_b, point, False) >= 0
        )
        in_tracking = (
            tracking is None
            or cv2.pointPolygonTest(tracking, point, False) >= 0
        )

        global_state.in_zone_a = in_zone_a
        global_state.in_tracking_zone = in_tracking and tracking is not None
        global_state.in_zone_b = in_zone_b
        global_state.ever_entered_zone_a |= in_zone_a
        global_state.ever_entered_zone_b |= in_zone_b

        current_membership = {
            "zone_a": global_state.in_zone_a,
            "tracking_zone": global_state.in_tracking_zone,
            "zone_b": global_state.in_zone_b,
        }
        if self.debug:
            for zone_name, inside in current_membership.items():
                if self.zones.get(zone_name) is None:
                    continue
                was_inside = previous_membership[zone_name]
                if inside == was_inside:
                    continue
                transition = "ENTERED" if inside else "LEFT"
                print(
                    f"[frame {frame_index}] G{global_state.global_id} "
                    f"{transition} {zone_name}"
                )

        if in_zone_a:
            global_state.last_zone = "ZONE_A"
        elif in_zone_b:
            global_state.last_zone = "ZONE_B"
        elif tracking is not None and in_tracking:
            global_state.last_zone = "TRACKING"
        else:
            global_state.last_zone = "OUTSIDE"

        if not global_state.journey_started:
            global_state.destination_consecutive_frames = 0
            return

        destination_polygon = (
            zone_b if global_state.destination_zone == "B" else zone_a
        )
        destination_distance = self._distance_to_polygon(
            destination_polygon, point
        )
        previous_distance = global_state.last_distance_to_destination
        if destination_distance is not None:
            global_state.last_distance_to_destination = destination_distance
            if global_state.min_distance_to_destination is None:
                global_state.min_distance_to_destination = destination_distance
            else:
                global_state.min_distance_to_destination = min(
                    global_state.min_distance_to_destination,
                    destination_distance,
                )
            moving_toward = (
                previous_distance is not None
                and destination_distance < previous_distance - 0.5
                and self._velocity_points_toward_polygon(
                    global_state.velocity,
                    point,
                    destination_polygon,
                )
            )
            if moving_toward:
                global_state.destination_approach_frames += 1
            elif (
                previous_distance is not None
                and destination_distance > previous_distance + 0.5
            ):
                global_state.destination_approach_frames = 0
            global_state.approaching_destination = (
                global_state.destination_approach_frames
                >= DESTINATION_APPROACH_MIN_FRAMES
            )
            if self.debug_verbose:
                print(
                    f"[frame {frame_index}] G{global_state.global_id} "
                    f"distance to destination = {destination_distance:.1f} px"
                )

        in_destination = (
            in_zone_b
            if global_state.destination_zone == "B"
            else in_zone_a
        )
        if in_destination:
            global_state.destination_consecutive_frames += 1
            if not global_state.entered_destination:
                global_state.entered_destination_zone = True
                self._mark_destination_entered(
                    global_state,
                    frame_index,
                    "centroid_inside_destination_zone",
                )
        else:
            global_state.destination_consecutive_frames = 0

        if (
            global_state.entered_destination
            and tracking is not None
            and not in_tracking
        ):
            self._complete_global(
                global_state,
                frame_index,
                "left_tracking_after_destination",
            )
        elif (
            in_destination
            and global_state.destination_consecutive_frames
            >= self.handoff_hold_frames
        ):
            self._complete_global(
                global_state,
                frame_index,
                "destination_hold_reached",
            )

    @staticmethod
    def _distance_to_polygon(
        polygon: np.ndarray | None, point: tuple[float, float]
    ) -> float | None:
        """Return zero inside a polygon, otherwise distance to its boundary."""

        if polygon is None:
            return None
        signed_distance = cv2.pointPolygonTest(polygon, point, True)
        return max(0.0, -float(signed_distance))

    @staticmethod
    def _closest_polygon_point(
        polygon: np.ndarray, point: tuple[float, float]
    ) -> np.ndarray:
        target = np.asarray(point, dtype=np.float64)
        points = np.asarray(polygon, dtype=np.float64)
        closest = points[0]
        closest_distance = float("inf")
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            segment = end - start
            length_squared = float(np.dot(segment, segment))
            if length_squared <= 1e-9:
                candidate = start
            else:
                fraction = float(np.dot(target - start, segment) / length_squared)
                candidate = start + min(1.0, max(0.0, fraction)) * segment
            distance = float(np.linalg.norm(candidate - target))
            if distance < closest_distance:
                closest = candidate
                closest_distance = distance
        return closest

    def _velocity_points_toward_polygon(
        self,
        velocity: tuple[float, float],
        point: tuple[float, float],
        polygon: np.ndarray | None,
    ) -> bool:
        if polygon is None:
            return False
        velocity_vector = np.asarray(velocity, dtype=np.float64)
        if float(np.linalg.norm(velocity_vector)) <= 1e-6:
            return False
        point_vector = np.asarray(point, dtype=np.float64)
        toward = self._closest_polygon_point(polygon, point) - point_vector
        if float(np.linalg.norm(toward)) <= 1e-6:
            return True
        return float(np.dot(velocity_vector, toward)) > 0.0

    def _can_complete_at_destination_exit(
        self, global_state: GlobalRollState
    ) -> bool:
        """Use proximity only when a B->A journey then disappears at its exit."""

        distance = global_state.last_distance_to_destination
        return (
            global_state.journey_started
            and global_state.journey_direction == B_TO_A
            and distance is not None
            and distance <= self.destination_proximity_px
            and global_state.approaching_destination
            and bool(global_state.last_motion_active)
        )

    def _mark_destination_entered(
        self,
        global_state: GlobalRollState,
        frame_index: int,
        reason: str,
    ) -> None:
        if global_state.entered_destination:
            return
        global_state.entered_destination = True
        global_state.destination_entered_frame = frame_index
        self._log_event(
            frame_index,
            global_state.global_id,
            global_state.current_local_id,
            global_state.current_local_id,
            "GLOBAL_ENTERED_DESTINATION",
            decision_reason=reason,
        )
        if self.debug:
            if global_state.entered_destination_zone:
                detail = f"entered destination zone {global_state.destination_zone}"
            else:
                detail = "reached configured destination exit corridor"
            print(f"[frame {frame_index}] G{global_state.global_id} {detail}")

    def _complete_global(
        self,
        global_state: GlobalRollState,
        frame_index: int,
        reason: str,
    ) -> None:
        if global_state.state in TERMINAL_STATES:
            return
        global_state.state = COMPLETED
        global_state.reassociation_allowed = False
        global_state.completion_reason = reason
        global_state.missing_frames = 0
        self._log_event(
            frame_index,
            global_state.global_id,
            global_state.current_local_id,
            None,
            "GLOBAL_COMPLETED",
            decision_reason=reason,
        )
        if self.debug:
            print(
                f"[frame {frame_index}] G{global_state.global_id} -> "
                f"COMPLETED ({reason})"
            )

    def _retention_window(self, global_state: GlobalRollState) -> int:
        if global_state.was_active_before_occlusion:
            return self.active_occlusion_frames
        return self.occlusion_frames

    def _associate_new_observations(
        self,
        observations: Sequence[IdentityObservation],
        frame_index: int,
        create_global_local_ids: set[int],
        transport_start_local_ids: set[int],
        origin_by_local: dict[int, str],
        reason_by_local: dict[int, str],
    ) -> None:
        if not observations:
            return
        occluded = [
            state
            for state in self.global_rolls.values()
            if state.state in {OCCLUDED, PLACEMENT_PENDING}
            and state.reassociation_allowed
            and frame_index - state.last_seen_frame - 1
            <= self._retention_window(state)
        ]
        if self.debug and not occluded:
            print("No OCCLUDED candidate available")
        assessments: dict[tuple[int, int], ReassociationAssessment] = {}
        local_candidates: dict[int, list[int]] = {}
        global_candidates: dict[int, list[int]] = {}
        for observation in observations:
            for terminal_state in self.global_rolls.values():
                if terminal_state.state in TERMINAL_STATES:
                    self._log_terminal_rejection(
                        terminal_state, observation, frame_index
                    )
            if self.debug:
                print(f"[frame {frame_index}] New local track {observation.local_id}")
                for state in self.global_rolls.values():
                    if state.state in TERMINAL_STATES:
                        print(
                            f"  G{state.global_id} excluded because {state.state}"
                        )
            for global_state in occluded:
                assessment = self._assess_reassociation(
                    observation, global_state, frame_index
                )
                assessments[(observation.local_id, global_state.global_id)] = assessment
                if self.debug:
                    self._print_candidate(global_state, assessment)
                if not assessment.valid:
                    if assessment.reason == "opposite_bin_flow":
                        event = "REASSOCIATION_REJECTED_OPPOSITE_BIN_FLOW"
                        self._log_event(
                            frame_index,
                            global_state.global_id,
                            global_state.current_local_id,
                            observation.local_id,
                            event,
                            assessment=assessment,
                            decision_reason=assessment.reason,
                        )
                        self.possible_false_global_reuse_events.append(
                            (
                                frame_index,
                                global_state.global_id,
                                observation.local_id,
                                event,
                            )
                        )
                    continue
                local_candidates.setdefault(observation.local_id, []).append(
                    global_state.global_id
                )
                global_candidates.setdefault(global_state.global_id, []).append(
                    observation.local_id
                )

        ambiguous_locals = {
            local_id
            for local_id, candidates in local_candidates.items()
            if len(candidates) > 1
        }
        for candidate_locals in global_candidates.values():
            if len(candidate_locals) > 1:
                ambiguous_locals.update(candidate_locals)

        for local_id in sorted(ambiguous_locals):
            for global_id in local_candidates.get(local_id, []):
                global_state = self.global_rolls[global_id]
                assessment = assessments[(local_id, global_id)]
                self._log_event(
                    frame_index,
                    global_id,
                    global_state.current_local_id,
                    local_id,
                    "AMBIGUOUS_REASSOCIATION",
                    assessment=assessment,
                    decision_reason="multiple_plausible_candidates",
                )
            if self.debug:
                candidates = local_candidates.get(local_id, [])
                formatted = ", ".join(f"G{value}" for value in candidates)
                print(
                    f"[frame {frame_index}] local ID {local_id} ambiguous "
                    f"against [{formatted}]; creating a new global identity"
                )

        used_globals: set[int] = set()
        for observation in observations:
            local_id = observation.local_id
            if local_id in ambiguous_locals:
                if local_id in create_global_local_ids:
                    self._create_global(
                        observation,
                        frame_index,
                        decision_reason="ambiguous_reassociation",
                        transport_started=(
                            local_id in transport_start_local_ids
                        ),
                        origin_zone=origin_by_local.get(local_id),
                        activation_reason=reason_by_local.get(local_id, "endpoint"),
                    )
                elif self.debug:
                    print(
                        f"[frame {frame_index}] local ID {local_id} remains "
                        "UNASSIGNED after ambiguous reassociation"
                    )
                continue
            candidates = local_candidates.get(local_id, [])
            if len(candidates) == 1 and candidates[0] not in used_globals:
                global_id = candidates[0]
                used_globals.add(global_id)
                self._reassociate(
                    observation,
                    self.global_rolls[global_id],
                    frame_index,
                    assessments[(local_id, global_id)],
                )
            else:
                attempted = [
                    assessment
                    for (candidate_local, _), assessment in assessments.items()
                    if candidate_local == local_id
                ]
                best = max(attempted, key=lambda item: item.score, default=None)
                reason = (
                    f"no_valid_reassociation:{best.reason}"
                    if best is not None
                    else "no_occluded_global"
                )
                if local_id not in create_global_local_ids:
                    if self.debug:
                        print(
                            f"Decision: rejected reassociation ({reason}); "
                            f"local ID {local_id} remains UNASSIGNED"
                        )
                    continue
                if self.debug:
                    print(
                        f"Decision: rejected reassociation ({reason}); "
                        f"GLOBAL_CREATED for local ID {local_id}"
                    )
                self._create_global(
                    observation,
                    frame_index,
                    decision_reason=reason,
                    assessment=best,
                    transport_started=(local_id in transport_start_local_ids),
                    origin_zone=origin_by_local.get(local_id),
                    activation_reason=reason_by_local.get(local_id, "endpoint"),
                )

    def _assess_reassociation(
        self,
        observation: IdentityObservation,
        global_state: GlobalRollState,
        frame_index: int,
    ) -> ReassociationAssessment:
        # Missing frames exclude both the last detected frame and the current
        # reappearance frame: last=69, new=86 means frames 70..85 = 16.
        gap = max(0, frame_index - global_state.last_seen_frame - 1)
        last_centroid = global_state.last_seen_centroid or global_state.last_centroid
        last_box_size = global_state.last_seen_box_size or global_state.last_box_size
        previous_velocity = global_state.last_velocity
        previous_speed = global_state.last_speed
        predicted = (
            last_centroid[0] + previous_velocity[0] * (gap + 1),
            last_centroid[1] + previous_velocity[1] * (gap + 1),
        )
        predicted_distance = math.dist(predicted, observation.centroid)
        last_distance = math.dist(last_centroid, observation.centroid)
        association_distance = min(last_distance, predicted_distance)
        recent_speed = previous_speed
        dynamic_allowed_distance = min(
            self.max_dynamic_distance,
            max(
                self.short_gap_max_distance,
                recent_speed * gap * self.motion_tolerance,
            ),
        )
        active_allowed_distance = min(
            self.active_reassoc_max_distance,
            max(
                self.short_gap_max_distance,
                previous_speed * gap * self.motion_tolerance,
            ),
        )
        size_ratio = self._area_ratio(last_box_size, observation.box_size)
        max_distance = self.max_distance_fraction * self.frame_diagonal
        position_score = max(0.0, 1.0 - association_distance / max_distance)
        time_score = max(0.0, 1.0 - gap / self.occlusion_frames)
        size_score = self._size_ratio_score(size_ratio)
        motion_score = max(0.0, 1.0 - predicted_distance / max_distance)
        direction_score = self._direction_score(
            global_state.last_motion_direction,
            last_centroid,
            observation.centroid,
        )
        active_bonus = (
            ACTIVE_PRIOR_BONUS if global_state.was_active_before_occlusion else 0.0
        )
        short_gap_spatial_match = (
            gap <= self.short_gap_frames
            and association_distance <= dynamic_allowed_distance
            and self.hard_size_ratio_min
            <= size_ratio
            <= self.hard_size_ratio_max
        )
        score = (
            POSITION_WEIGHT * position_score
            + TIME_WEIGHT * time_score
            + MOTION_PREDICTION_WEIGHT * motion_score
            + SIZE_WEIGHT * size_score
        )
        if short_gap_spatial_match:
            score = min(1.0, score + SHORT_GAP_SCORE_BONUS)

        active_long_gap = (
            global_state.was_active_before_occlusion
            and self.short_gap_frames < gap <= self.active_occlusion_frames
        )
        if active_long_gap:
            position_score = max(
                0.0, 1.0 - association_distance / active_allowed_distance
            )
            time_score = max(
                0.0, 1.0 - gap / self.active_occlusion_frames
            )
            score = (
                ACTIVE_POSITION_WEIGHT * position_score
                + ACTIVE_TIME_WEIGHT * time_score
                + ACTIVE_DIRECTION_WEIGHT * direction_score
                + ACTIVE_SIZE_WEIGHT * size_score
                + active_bonus
            )

        if global_state.state in TERMINAL_STATES or not global_state.reassociation_allowed:
            valid, reason = False, "terminal_global"
        elif self._has_opposite_bin_flow(global_state, observation):
            valid, reason = False, "opposite_bin_flow"
        elif (
            global_state.state == PLACEMENT_PENDING
            and not self._placement_pending_compatible(global_state, observation)
        ):
            valid, reason = False, "placement_pending_incompatible"
        elif gap > self._retention_window(global_state):
            valid, reason = False, "gap_too_large"
        elif not self.hard_size_ratio_min <= size_ratio <= self.hard_size_ratio_max:
            valid, reason = False, "size_ratio_beyond_emergency_limit"
        elif short_gap_spatial_match:
            valid, reason = True, "short_gap_spatial_match"
        elif active_long_gap and association_distance <= active_allowed_distance:
            # Candidate uniqueness is enforced after all local/global pairs are
            # evaluated. Direction affects ranking/debugging but never vetoes.
            valid, reason = True, "active_occlusion_spatial_match"
        elif active_long_gap:
            valid, reason = False, "active_distance_too_large"
        elif association_distance > max_distance:
            valid, reason = False, "distance_too_large"
        elif score >= self.min_score:
            valid, reason = True, "weighted_score_match"
        else:
            valid, reason = False, "score_below_threshold"
        return ReassociationAssessment(
            global_id=global_state.global_id,
            gap_frames=gap,
            last_seen_distance=last_distance,
            predicted_distance=predicted_distance,
            association_distance=association_distance,
            recent_speed=recent_speed,
            dynamic_allowed_distance=dynamic_allowed_distance,
            was_active_before_occlusion=global_state.was_active_before_occlusion,
            previous_speed=previous_speed,
            direction_score=direction_score,
            active_bonus=active_bonus,
            active_allowed_distance=active_allowed_distance,
            size_ratio=size_ratio,
            position_score=position_score,
            time_score=time_score,
            motion_score=motion_score,
            size_score=size_score,
            score=score,
            valid=valid,
            reason=reason,
        )

    @staticmethod
    def _observation_speed(observation: IdentityObservation) -> float:
        history = observation.centroid_history
        if len(history) < 2:
            return 0.0
        first = history[0]
        last = history[-1]
        frame_delta = last[0] - first[0]
        if frame_delta <= 0:
            return 0.0
        return math.dist((first[1], first[2]), (last[1], last[2])) / frame_delta

    @staticmethod
    def _has_opposite_bin_flow(
        global_state: GlobalRollState, observation: IdentityObservation
    ) -> bool:
        candidate_bin = (
            global_state.related_bin_id
            or global_state.candidate_bin_id
            or global_state.entry_bin_id
            or global_state.actual_bin_id
        )
        return bool(
            candidate_bin
            and observation.related_bin_id == candidate_bin
            and global_state.bin_motion_relation == ENTERING_BIN
            and observation.bin_motion_relation == LEAVING_BIN
        )

    def _placement_pending_compatible(
        self,
        global_state: GlobalRollState,
        observation: IdentityObservation,
    ) -> bool:
        """Allow only a nearby, same-flow fragment to continue pending placement."""

        if global_state.state != PLACEMENT_PENDING:
            return True
        if self._has_opposite_bin_flow(global_state, observation):
            return False
        candidate_bin = (
            global_state.related_bin_id
            or global_state.candidate_bin_id
            or global_state.entry_bin_id
        )
        if candidate_bin and observation.related_bin_id != candidate_bin:
            return False
        allowed_distance = max(
            20.0,
            self.short_gap_max_distance * PLACEMENT_PENDING_DISTANCE_FRACTION,
        )
        if math.dist(global_state.last_centroid, observation.centroid) > allowed_distance:
            return False
        if observation.is_active and (
            observation.became_active
            or self._observation_speed(observation)
            > PLACEMENT_PENDING_MAX_SPEED_PX_PER_FRAME
        ):
            return False
        return observation.bin_motion_relation != LEAVING_BIN

    def _log_terminal_rejection(
        self,
        global_state: GlobalRollState,
        observation: IdentityObservation,
        frame_index: int,
    ) -> None:
        """Audit plausible attempts without ever reopening a sealed journey."""

        if global_state.state not in TERMINAL_STATES:
            return
        distance = math.dist(global_state.last_centroid, observation.centroid)
        if distance > self.active_reassoc_max_distance:
            return
        event = "REASSOCIATION_REJECTED_TERMINAL_GLOBAL"
        self._log_event(
            frame_index,
            global_state.global_id,
            global_state.current_local_id,
            observation.local_id,
            event,
            decision_reason=f"{global_state.state.lower()}_is_terminal",
        )
        self.possible_false_global_reuse_events.append(
            (frame_index, global_state.global_id, observation.local_id, event)
        )

    @staticmethod
    def _direction_score(
        previous_direction: tuple[float, float],
        previous_centroid: tuple[float, float],
        current_centroid: tuple[float, float],
    ) -> float:
        displacement = (
            current_centroid[0] - previous_centroid[0],
            current_centroid[1] - previous_centroid[1],
        )
        displacement_length = math.hypot(*displacement)
        direction_length = math.hypot(*previous_direction)
        if displacement_length <= 1e-6 or direction_length <= 1e-6:
            return 0.5
        cosine = (
            previous_direction[0] * displacement[0]
            + previous_direction[1] * displacement[1]
        ) / (direction_length * displacement_length)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _size_ratio_score(self, size_ratio: float) -> float:
        # Log-space decay treats reciprocal changes symmetrically and continues
        # decreasing gradually beyond the soft range instead of hard-rejecting.
        if size_ratio <= 0:
            return 0.0
        scale = (
            abs(math.log(self.size_ratio_min))
            if size_ratio < 1.0
            else abs(math.log(self.size_ratio_max))
        )
        return math.exp(-abs(math.log(size_ratio)) / max(scale, 1e-6))

    @staticmethod
    def _area_ratio(
        previous: tuple[float, float], current: tuple[float, float]
    ) -> float:
        old_area = max(previous[0] * previous[1], 1e-6)
        new_area = max(current[0] * current[1], 1e-6)
        return new_area / old_area

    @staticmethod
    def _print_candidate(
        global_state: GlobalRollState,
        assessment: ReassociationAssessment,
    ) -> None:
        print(
            f"Candidate GLOBAL G{global_state.global_id}:\n"
            f"  gap = {assessment.gap_frames} frames\n"
            f"  last_seen_distance = {assessment.last_seen_distance:.1f} px\n"
            f"  predicted_distance = {assessment.predicted_distance:.1f} px\n"
            f"  association_distance = {assessment.association_distance:.1f} px\n"
            f"  recent_speed = {assessment.recent_speed:.2f} px/frame\n"
            f"  dynamic_allowed_distance = {assessment.dynamic_allowed_distance:.1f} px\n"
            f"  was_active_before_occlusion = {assessment.was_active_before_occlusion}\n"
            f"  previous_speed = {assessment.previous_speed:.2f} px/frame\n"
            f"  direction_score = {assessment.direction_score:.3f}\n"
            f"  active_allowed_distance = {assessment.active_allowed_distance:.1f} px\n"
            f"  active_bonus = {assessment.active_bonus:.3f}\n"
            f"  size_ratio = {assessment.size_ratio:.3f}\n"
            f"  position_score = {assessment.position_score:.3f}\n"
            f"  time_score = {assessment.time_score:.3f}\n"
            f"  motion_score = {assessment.motion_score:.3f}\n"
            f"  size_score = {assessment.size_score:.3f}\n"
            f"  total_score = {assessment.score:.3f}\n"
            f"  decision = {'ACCEPT' if assessment.valid else 'REJECT'} "
            f"({assessment.reason})"
        )

    def _create_global(
        self,
        observation: IdentityObservation,
        frame_index: int,
        *,
        decision_reason: str,
        assessment: ReassociationAssessment | None = None,
        transport_started: bool = False,
        origin_zone: str | None = None,
        activation_reason: str = "",
    ) -> None:
        if transport_started and origin_zone is None:
            origin_zone = self._endpoint_for_point(observation.centroid)
        if transport_started and origin_zone not in {"A", "B"}:
            raise ValueError(
                "A transport Global ID requires origin_zone A or B"
            )
        destination_zone = (
            "B" if origin_zone == "A" else "A" if origin_zone == "B" else ""
        )
        journey_direction = (
            A_TO_B if origin_zone == "A" else B_TO_A if origin_zone == "B" else ""
        )
        global_id = self.next_global_id
        self.next_global_id += 1
        global_state = GlobalRollState(
            global_id=global_id,
            current_local_id=observation.local_id,
            state=NEW,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            last_centroid=observation.centroid,
            velocity=(0.0, 0.0),
            last_box_size=observation.box_size,
            journey_started=transport_started,
            origin_zone=origin_zone or "",
            destination_zone=destination_zone,
            journey_direction=journey_direction,
            activation_reason=(activation_reason if transport_started else ""),
            related_bin_id=observation.related_bin_id,
            bin_motion_relation=observation.bin_motion_relation,
            local_attachment_history=[(frame_index, observation.local_id)],
        )
        for history_point in observation.centroid_history:
            if history_point[0] < frame_index:
                global_state.centroid_history.append(history_point)
        global_state.observe(observation, frame_index)
        if transport_started:
            global_state.last_motion_active = True
        else:
            self.static_global_creation_count += 1
        self.global_rolls[global_id] = global_state
        self.local_to_global[observation.local_id] = global_id
        self._update_zone_state(global_state, observation, frame_index)
        self._log_event(
            frame_index,
            global_id,
            None,
            observation.local_id,
            "GLOBAL_CREATED",
            assessment=assessment,
            decision_reason=decision_reason,
        )
        if transport_started:
            self._log_event(
                frame_index,
                global_id,
                observation.local_id,
                observation.local_id,
                "GLOBAL_PICKED_UP",
            )
        if self.debug:
            print(
                f"[frame {frame_index}] GLOBAL G{global_id} created from "
                f"local ID {observation.local_id}"
            )
            if transport_started:
                print(
                    f"[frame {frame_index}] GLOBAL G{global_id} committed "
                    f"as {journey_direction} transport ID"
                )

    def _endpoint_for_point(self, point: tuple[float, float]) -> str | None:
        """Return a unique endpoint within the configured proximity margin."""

        matches: list[str] = []
        for endpoint, key in (("A", "zone_a"), ("B", "zone_b")):
            polygon = self.zones.get(key)
            if polygon is None:
                continue
            if cv2.pointPolygonTest(polygon, point, True) >= -self.endpoint_margin_px:
                matches.append(endpoint)
        return matches[0] if len(matches) == 1 else None

    def _reassociate(
        self,
        observation: IdentityObservation,
        global_state: GlobalRollState,
        frame_index: int,
        assessment: ReassociationAssessment,
    ) -> None:
        if global_state.state in TERMINAL_STATES or not global_state.reassociation_allowed:
            self._log_terminal_rejection(global_state, observation, frame_index)
            return
        old_local_id = global_state.current_local_id
        if self.local_to_global.get(old_local_id) == global_state.global_id:
            self.local_to_global.pop(old_local_id, None)
        global_state.current_local_id = observation.local_id
        global_state.reassociation_count += 1
        global_state.local_attachment_history.append(
            (frame_index, observation.local_id)
        )
        if observation.related_bin_id:
            global_state.related_bin_id = observation.related_bin_id
        if observation.bin_motion_relation != NEUTRAL:
            global_state.bin_motion_relation = observation.bin_motion_relation
        global_state.observe(observation, frame_index)
        self.local_to_global[observation.local_id] = global_state.global_id
        self._update_zone_state(global_state, observation, frame_index)
        self._log_event(
            frame_index,
            global_state.global_id,
            old_local_id,
            observation.local_id,
            "GLOBAL_REASSOCIATED",
            assessment=assessment,
            decision_reason=assessment.reason,
        )
        if self.debug:
            print(
                f"[frame {frame_index}] local ID {observation.local_id} "
                f"reassociated -> GLOBAL G{global_state.global_id} "
                f"(old local ID {old_local_id}, score {assessment.score:.3f}, "
                f"reason {assessment.reason})"
            )

    def _log_event(
        self,
        frame_index: int,
        global_id: int,
        old_local_id: int | None,
        new_local_id: int | None,
        event: str,
        assessment: ReassociationAssessment | None = None,
        decision_reason: str = "",
    ) -> None:
        global_state = self.global_rolls.get(global_id)
        self.event_writer.writerow(
            {
                "frame": frame_index,
                "timestamp_seconds": f"{frame_index / self.fps:.3f}",
                "global_id": global_id,
                "local_id": (
                    new_local_id
                    if new_local_id is not None
                    else ("" if old_local_id is None else old_local_id)
                ),
                "old_local_id": "" if old_local_id is None else old_local_id,
                "new_local_id": "" if new_local_id is None else new_local_id,
                "event": event,
                "lifecycle_state": (
                    "" if global_state is None else global_state.state
                ),
                "zone": "" if global_state is None else global_state.last_zone,
                "origin_zone": (
                    "" if global_state is None else global_state.origin_zone
                ),
                "destination_zone": (
                    "" if global_state is None else global_state.destination_zone
                ),
                "journey_direction": (
                    "" if global_state is None else global_state.journey_direction
                ),
                "activation_reason": (
                    "" if global_state is None else global_state.activation_reason
                ),
                "gap_frames": (
                    "" if assessment is None else assessment.gap_frames
                ),
                "last_seen_distance": (
                    ""
                    if assessment is None
                    else f"{assessment.last_seen_distance:.3f}"
                ),
                "predicted_distance": (
                    ""
                    if assessment is None
                    else f"{assessment.predicted_distance:.3f}"
                ),
                "association_distance": (
                    ""
                    if assessment is None
                    else f"{assessment.association_distance:.3f}"
                ),
                "recent_speed": (
                    "" if assessment is None else f"{assessment.recent_speed:.6f}"
                ),
                "dynamic_allowed_distance": (
                    ""
                    if assessment is None
                    else f"{assessment.dynamic_allowed_distance:.3f}"
                ),
                "was_active_before_occlusion": (
                    assessment.was_active_before_occlusion
                    if assessment is not None
                    else (
                        global_state.was_active_before_occlusion
                        if global_state is not None
                        else ""
                    )
                ),
                "previous_speed": (
                    f"{assessment.previous_speed:.6f}"
                    if assessment is not None
                    else (
                        f"{global_state.last_speed:.6f}"
                        if global_state is not None
                        else ""
                    )
                ),
                "direction_score": (
                    "" if assessment is None else f"{assessment.direction_score:.6f}"
                ),
                "active_bonus": (
                    "" if assessment is None else f"{assessment.active_bonus:.6f}"
                ),
                "active_allowed_distance": (
                    ""
                    if assessment is None
                    else f"{assessment.active_allowed_distance:.3f}"
                ),
                "size_ratio": (
                    "" if assessment is None else f"{assessment.size_ratio:.6f}"
                ),
                "position_score": (
                    "" if assessment is None else f"{assessment.position_score:.6f}"
                ),
                "time_score": (
                    "" if assessment is None else f"{assessment.time_score:.6f}"
                ),
                "motion_score": (
                    "" if assessment is None else f"{assessment.motion_score:.6f}"
                ),
                "size_score": (
                    "" if assessment is None else f"{assessment.size_score:.6f}"
                ),
                "reassociation_score": (
                    "" if assessment is None else f"{assessment.score:.6f}"
                ),
                "decision_reason": decision_reason,
            }
        )
        self.event_counts[event] += 1
        if event in {
            "GLOBAL_PICKED_UP",
            "GLOBAL_ENTERED_DESTINATION",
            "GLOBAL_COMPLETED",
            "GLOBAL_PLACED",
        }:
            self.frame_transition_events.append((global_id, event))

    def summary_lines(self) -> list[str]:
        state_counts = Counter(state.state for state in self.global_rolls.values())
        direction_counts = Counter(
            state.journey_direction for state in self.global_rolls.values()
        )
        activation_counts = Counter(
            state.activation_reason for state in self.global_rolls.values()
        )
        return [
            "Global identity summary",
            "-----------------------",
            f"Zone config loaded: {'YES' if self.lifecycle_enabled else 'NO'}",
            f"Global occlusion window: {self.occlusion_frames} frames",
            f"ACTIVE occlusion window: {self.active_occlusion_frames} frames",
            f"ACTIVE reassociation maximum distance: {self.active_reassoc_max_distance:.1f} px",
            f"Short-gap window: {self.short_gap_frames} frames",
            f"Short-gap base distance: {self.short_gap_max_distance:.1f} px",
            f"Motion tolerance: {self.motion_tolerance:.3f}",
            f"Maximum dynamic distance: {self.max_dynamic_distance:.1f} px",
            f"Reassociation maximum distance: {self.max_distance_fraction:.3f} of frame diagonal",
            f"Soft size ratio range: {self.size_ratio_min:.3f} to {self.size_ratio_max:.3f}",
            f"Emergency size ratio limits: {self.hard_size_ratio_min:.3f} to {self.hard_size_ratio_max:.3f}",
            f"Minimum weighted score: {self.min_score:.3f}",
            f"Destination proximity fallback: {self.destination_proximity_px:.1f} px",
            f"Total global roll IDs: {len(self.global_rolls)}",
            f"Visible global rolls: {sum(state_counts[value] for value in VISIBLE_STATES)}",
            f"In-transit global rolls: {state_counts[ACTIVE]}",
            f"Occluded global rolls: {state_counts[OCCLUDED]}",
            f"Placement-pending global rolls: {state_counts[PLACEMENT_PENDING]}",
            f"Global IDs sealed by placement: {state_counts[PLACED]}",
            f"Camera journeys closed: {state_counts[COMPLETED]}",
            f"Lost global rolls: {state_counts[LOST]}",
            f"GLOBAL_CREATED events: {self.event_counts['GLOBAL_CREATED']}",
            f"A_TO_B journeys: {direction_counts[A_TO_B]}",
            f"B_TO_A journeys: {direction_counts[B_TO_A]}",
            f"Endpoint-created journeys: {activation_counts['endpoint']}",
            "Sustained-motion-created journeys: "
            f"{activation_counts['sustained_motion']}",
            f"GLOBAL_REASSOCIATED events: {self.event_counts['GLOBAL_REASSOCIATED']}",
            "Attempts to reassociate sealed globals: "
            f"{self.event_counts['REASSOCIATION_REJECTED_TERMINAL_GLOBAL']}",
            "Opposite-flow reassociation rejections: "
            f"{self.event_counts['REASSOCIATION_REJECTED_OPPOSITE_BIN_FLOW']}",
            "Possible false global reuse events: "
            f"{len(self.possible_false_global_reuse_events)}",
            f"GLOBAL_PICKED_UP events: {self.event_counts['GLOBAL_PICKED_UP']}",
            "GLOBAL_ENTERED_DESTINATION events: "
            f"{self.event_counts['GLOBAL_ENTERED_DESTINATION']}",
            f"GLOBAL_COMPLETED events: {self.event_counts['GLOBAL_COMPLETED']}",
            f"Static Global IDs incorrectly created: {self.static_global_creation_count}",
            f"GLOBAL_LOST events: {self.event_counts['GLOBAL_LOST']}",
            f"AMBIGUOUS_REASSOCIATION events: {self.event_counts['AMBIGUOUS_REASSOCIATION']}",
            "",
            "Per-global zone lifecycle",
            "-------------------------",
            *[
                line
                for state in sorted(
                    self.global_rolls.values(), key=lambda value: value.global_id
                )
                for line in (
                    f"Global G{state.global_id}:",
                    f"  direction: {state.journey_direction or 'unassigned'}",
                    f"  origin: {state.origin_zone or 'unassigned'}",
                    f"  destination: {state.destination_zone or 'unassigned'}",
                    "  camera journey closed: "
                    f"{'yes' if state.state == COMPLETED else 'no'}",
                    f"  final lifecycle state: {state.state}",
                    "  placement_confirmed_frame: "
                    f"{state.placement_confirmed_frame if state.placement_confirmed_frame is not None else '---'}",
                    f"  lifecycle_after_placement: {state.state if state.placement_confirmed_frame is not None else '---'}",
                    f"  reassociation_allowed: {'yes' if state.reassociation_allowed else 'no'}",
                    "  later_local_ids_attached: "
                    + (
                        ", ".join(
                            str(local_id)
                            for attached_frame, local_id in state.local_attachment_history
                            if state.placement_confirmed_frame is not None
                            and attached_frame > state.placement_confirmed_frame
                        )
                        or "none"
                    ),
                )
            ],
        ]
