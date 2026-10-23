"""Time-based physical-bin placement confirmation keyed by Global Roll ID."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, Sequence

import cv2
import numpy as np

from bin_geometry import (
    BinGeometry,
    BinOverlap,
    obb_bin_overlap,
    point_in_bin_entry_zone,
)
from global_identity import (
    ACTIVE,
    COMPLETED,
    CORRECT,
    ENTERING_BIN,
    GlobalRollState,
    LOST,
    OCCLUDED,
    PLACEMENT_PENDING,
    UNRESOLVED,
    UNVERIFIED,
    VISIBLE,
    WRONG_BIN,
)


NONE = "NONE"
BIN_CANDIDATE = "BIN_CANDIDATE"
PROVISIONAL_PLACEMENT = "PROVISIONAL_PLACEMENT"
PLACEMENT_CONFIRMED = "PLACEMENT_CONFIRMED"
PLACEMENT_CANCELLED = "PLACEMENT_CANCELLED"
SPEED_WINDOW_SECONDS = 0.50
APPROACH_WINDOW_SECONDS = 0.75
ENTRY_APPROACH_MIN_PROGRESS_PX = 10.0
ENTRY_OVERLAP_GROWTH = 0.08
ENTRY_ZONE_LOSS_WINDOW_SECONDS = 0.75
ENTRY_MIN_DIRECTION_SCORE = 0.70
ENTRY_MIN_TRAJECTORY_CONSISTENCY = 0.70

TRACKING = "TRACKING"
PROVISIONAL = "PROVISIONAL"
CONFIRMED = "CONFIRMED"
UNRESOLVED_STATUS = "UNRESOLVED"

EVIDENCE_NONE = "NONE"
DIRECT_VISIBLE = "DIRECT_VISIBLE"
ENTRY_THEN_OCCLUDED = "ENTRY_THEN_OCCLUDED"
ACTIVE_TO_STATIC = "ACTIVE_TO_STATIC"


@dataclass
class BinFrameStatus:
    global_id: int
    best_bin_id: str
    best_overlap: float
    second_bin_id: str
    second_overlap: float
    ambiguous: bool
    candidate_bin_id: str = ""
    placement_state: str = NONE
    dwell_seconds: float = 0.0
    current_speed: float = 0.0
    provisional_elapsed_seconds: float = 0.0


@dataclass
class CandidateDiagnostic:
    global_id: int
    bin_id: str
    start_frame: int
    start_time: float
    last_valid_frame: int
    last_valid_time: float
    max_overlap: float
    current_overlap: float
    dwell_seconds: float = 0.0
    valid_overlap_seconds: float = 0.0
    current_speed: float = 0.0
    visible_frames: int = 1
    first_overlap: float = 0.0
    first_distance_to_bin_center: float = 0.0
    last_distance_to_bin_center: float = 0.0
    last_centroid_x: float = 0.0
    last_centroid_y: float = 0.0
    last_centroid_inside: bool = False
    overlap_increased: bool = False
    moved_deeper: bool = False
    provisional: bool = False
    provisional_start_frame: int | None = None
    provisional_start_time: float | None = None
    disappeared_inside_candidate: bool = False
    reappeared_outside: bool = False
    confirmation_method: str = ""
    end_frame: int | None = None
    reason: str = ""
    confirmed: bool = False
    entered_bin_mouth: bool = False
    entry_frame: int | None = None
    entry_direction_score: float = 0.0
    entry_trajectory_consistency: float = 0.0
    entry_confidence: float = 0.0


@dataclass
class EntryEvidence:
    global_id: int
    bin_id: str
    entry_frame: int
    entry_timestamp: float
    direction_score: float
    trajectory_consistency: float
    center_progress_px: float
    last_visible_frame: int
    last_visible_centroid: tuple[float, float]
    last_center_distance: float
    max_occupancy_overlap: float

    @property
    def confidence(self) -> float:
        """Diagnostic confidence; placement decisions use explicit gates."""

        return min(
            1.0,
            0.50 * self.direction_score
            + 0.35 * self.trajectory_consistency
            + 0.15 * min(1.0, self.center_progress_px / 40.0),
        )


class BinEntryManager:
    """Confirm actual placement only after dwell time and low motion."""

    def __init__(
        self,
        *,
        geometry: BinGeometry | None,
        camera_id: str | None,
        overlap_threshold: float,
        visible_dwell_seconds: float,
        gap_tolerance_seconds: float,
        placement_grace_seconds: float,
        settle_speed_px_per_sec: float,
        detector_miss_distance_px: float,
        static_handoff_seconds: float,
        static_handoff_distance_px: float,
        ambiguity_margin: float,
        allow_reassignment: bool,
        fps: float,
        event_writer: csv.DictWriter,
        debug: bool,
    ) -> None:
        self.geometry = geometry
        self.camera_id = camera_id
        self.overlap_threshold = overlap_threshold
        self.visible_dwell_seconds = visible_dwell_seconds
        self.gap_tolerance_seconds = gap_tolerance_seconds
        self.placement_grace_seconds = placement_grace_seconds
        self.settle_speed_px_per_sec = settle_speed_px_per_sec
        self.detector_miss_distance_px = detector_miss_distance_px
        self.static_handoff_seconds = static_handoff_seconds
        self.static_handoff_distance_px = static_handoff_distance_px
        self.ambiguity_margin = ambiguity_margin
        # Retained for CLI compatibility. A confirmed physical placement is
        # immutable and is counted no more than once per Global Roll journey.
        self.allow_reassignment = allow_reassignment
        self.fps = fps
        self.event_writer = event_writer
        self.debug = debug
        self.ambiguous_global_ids: set[int] = set()
        self._currently_ambiguous: set[int] = set()
        self._active_candidates: dict[int, CandidateDiagnostic] = {}
        self.candidate_history: dict[int, list[CandidateDiagnostic]] = defaultdict(list)
        history_length = max(4, int(math.ceil(fps * (SPEED_WINDOW_SECONDS + 0.25))))
        self._speed_history: dict[int, Deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=history_length)
        )
        approach_length = max(
            4, int(math.ceil(fps * (APPROACH_WINDOW_SECONDS + 0.25)))
        )
        self._approach_history: dict[
            int, dict[str, Deque[tuple[int, float, float]]]
        ] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=approach_length))
        )
        self._had_candidate: set[int] = set()
        self._entry_zone_last_frame: dict[int, dict[str, int]] = defaultdict(dict)
        self._entry_zone_crossings: set[tuple[int, str]] = set()
        self._entry_zone_inside: dict[tuple[int, str], bool] = {}
        self._entry_zone_cross_frame: dict[tuple[int, str], int] = {}
        self._entry_evidence: dict[tuple[int, str], EntryEvidence] = {}
        self.static_local_to_owner: dict[int, int] = {}
        self.owner_to_static_local: dict[int, int] = {}
        self.possible_detector_misses: list[dict[str, object]] = []
        self._possible_detector_miss_ids: set[int] = set()
        # Diagnostic only. Counts never participate in placement, identity,
        # journey, or expected-vs-actual decisions.
        self.debug_placement_counts: Counter[str] = Counter()
        self.placement_counts = self.debug_placement_counts
        self._roll_states: dict[int, GlobalRollState] = {}
        self.candidates_started = 0
        self.candidates_cancelled = 0
        self.candidates_switched = 0
        self.placements_confirmed = 0
        self.provisional_started = 0
        self.provisional_cancelled = 0
        self.event_counts: Counter[str] = Counter()
        self._finalized = False

    @property
    def enabled(self) -> bool:
        return bool(
            self.geometry is not None
            and self.camera_id
            and self.geometry.polygons_for_camera(self.camera_id)
        )

    @property
    def configured_bin_ids(self) -> tuple[str, ...]:
        if not self.enabled or self.geometry is None or not self.camera_id:
            return ()
        return tuple(self.geometry.polygons_for_camera(self.camera_id))

    def process_observation(
        self,
        global_state: GlobalRollState,
        local_id: int,
        obb: Sequence[Sequence[float]] | np.ndarray,
        frame_index: int,
    ) -> BinFrameStatus | None:
        self._roll_states[global_state.global_id] = global_state
        if not self.enabled or not global_state.journey_started:
            return None

        timestamp = frame_index / self.fps
        corners = np.asarray(obb, dtype=np.float32)
        centroid_array = corners.mean(axis=0)
        centroid = (float(centroid_array[0]), float(centroid_array[1]))
        global_state.last_visible_centroid = centroid
        if not global_state.placement_confirmed and global_state.placement_status != PROVISIONAL:
            global_state.placement_status = TRACKING
            global_state.verification_result = UNRESOLVED
        speed = self._record_and_measure_speed(
            global_state.global_id, timestamp, centroid
        )
        overlaps = obb_bin_overlap(self.geometry, self.camera_id or "", corners)
        self._record_bin_approach(
            global_state,
            local_id,
            frame_index,
            centroid,
            overlaps,
        )
        status = self._frame_status(global_state.global_id, overlaps)

        # Starting occupancy is the source, not a new placement. The state
        # machine arms only after the transported roll clears all bins once.
        if not global_state.bin_entry_initialized:
            global_state.bin_entry_initialized = True
            qualifying = [
                overlap
                for overlap in overlaps
                if overlap.overlap_ratio >= self.overlap_threshold
            ]
            if qualifying:
                global_state.initial_bin_id = qualifying[0].physical_bin_id
                global_state.bin_entry_armed = False
                if self.debug:
                    print(
                        f"[frame {frame_index}] G{global_state.global_id} starts "
                        f"in {global_state.initial_bin_id}; waiting for exit"
                    )
            else:
                global_state.bin_entry_armed = True
        if not global_state.bin_entry_armed:
            if any(
                overlap.overlap_ratio >= self.overlap_threshold
                for overlap in overlaps
            ):
                return self._decorate(status, global_state)
            global_state.bin_entry_armed = True
            if self.debug:
                print(
                    f"[frame {frame_index}] G{global_state.global_id} cleared "
                    f"starting bin {global_state.initial_bin_id}; placement armed"
                )
            return self._decorate(status, global_state)

        global_state.bin_max_overlap = max(
            global_state.bin_max_overlap, status.best_overlap
        )
        if global_state.bin_confirmed:
            return self._decorate(status, global_state)

        active = self._active_candidates.get(global_state.global_id)
        overlap_by_bin = {
            overlap.physical_bin_id: overlap.overlap_ratio for overlap in overlaps
        }
        evidence_by_bin = {
            overlap.physical_bin_id: overlap for overlap in overlaps
        }

        # A provisional placement may be invalidated by a reassociated Global
        # Roll. The identity is unchanged; only the physical-placement claim is
        # reconsidered.
        if active is not None and active.provisional:
            candidate_evidence = evidence_by_bin.get(active.bin_id)
            candidate_overlap = overlap_by_bin.get(active.bin_id, 0.0)
            best_is_valid = bool(
                status.best_bin_id
                and status.best_overlap >= self.overlap_threshold
            )
            if best_is_valid and status.best_bin_id != active.bin_id:
                self._cancel_provisional(
                    global_state,
                    local_id,
                    frame_index,
                    candidate_overlap,
                    speed,
                    reason="REAPPEARED_IN_DIFFERENT_BIN",
                )
                self._start_candidate(
                    global_state,
                    local_id,
                    frame_index,
                    status.best_bin_id,
                    status.best_overlap,
                    speed,
                    centroid,
                    evidence_by_bin[status.best_bin_id],
                    event="BIN_CANDIDATE_SWITCHED",
                    previous_bin_id=active.bin_id,
                )
                self.candidates_switched += 1
                return self._decorate(status, global_state, frame_index)
            if (
                candidate_evidence is not None
                and (
                    candidate_overlap >= self.overlap_threshold
                    or candidate_evidence.centroid_inside
                )
            ):
                self._continue_candidate(
                    global_state,
                    local_id,
                    frame_index,
                    candidate_overlap,
                    speed,
                    centroid,
                    candidate_evidence,
                    allow_confirmation=True,
                )
                return self._decorate(status, global_state, frame_index)
            self._cancel_provisional(
                global_state,
                local_id,
                frame_index,
                candidate_overlap,
                speed,
                reason="REAPPEARED_OUTSIDE_CANDIDATE_BIN",
            )
            return self._decorate(status, global_state, frame_index)

        if status.ambiguous:
            self.ambiguous_global_ids.add(global_state.global_id)
            if global_state.global_id not in self._currently_ambiguous:
                self._currently_ambiguous.add(global_state.global_id)
                self._log_event(
                    frame_index=frame_index,
                    global_id=global_state.global_id,
                    local_id=local_id,
                    candidate=active,
                    physical_bin_id=status.best_bin_id,
                    previous_bin_id="",
                    current_overlap=status.best_overlap,
                    speed=speed,
                    event="BIN_ENTRY_AMBIGUOUS",
                    reason="AMBIGUOUS_OVERLAP",
                )
            # Preserve the existing candidate on a qualifying ambiguous frame,
            # but never confirm until one physical bin is clearly dominant.
            if active is not None:
                candidate_overlap = overlap_by_bin.get(active.bin_id, 0.0)
                if candidate_overlap >= self.overlap_threshold:
                    self._continue_candidate(
                        global_state,
                        local_id,
                        frame_index,
                        candidate_overlap,
                        speed,
                        centroid,
                        evidence_by_bin[active.bin_id],
                        allow_confirmation=False,
                    )
                else:
                    self._handle_gap(
                        global_state,
                        local_id,
                        frame_index,
                        speed,
                        "OVERLAP_LOST",
                    )
            return self._decorate(status, global_state, frame_index)
        self._currently_ambiguous.discard(global_state.global_id)

        best_is_valid = bool(
            status.best_bin_id and status.best_overlap >= self.overlap_threshold
        )
        if active is None:
            if best_is_valid:
                self._start_candidate(
                    global_state,
                    local_id,
                    frame_index,
                    status.best_bin_id,
                    status.best_overlap,
                    speed,
                    centroid,
                    evidence_by_bin[status.best_bin_id],
                    event="BIN_CANDIDATE_STARTED",
                )
            return self._decorate(status, global_state, frame_index)

        candidate_overlap = overlap_by_bin.get(active.bin_id, 0.0)
        if best_is_valid and status.best_bin_id != active.bin_id:
            clearly_stronger = (
                candidate_overlap < self.overlap_threshold
                or status.best_overlap - candidate_overlap >= self.ambiguity_margin
            )
            if clearly_stronger:
                old_bin = active.bin_id
                self._switch_candidate(
                    global_state,
                    local_id,
                    frame_index,
                    status.best_bin_id,
                    status.best_overlap,
                    speed,
                    centroid,
                    evidence_by_bin[status.best_bin_id],
                )
                if self.debug:
                    print(
                        f"[frame {frame_index}] G{global_state.global_id} "
                        f"candidate switched {old_bin} -> {status.best_bin_id}"
                    )
                return self._decorate(status, global_state, frame_index)

        if candidate_overlap >= self.overlap_threshold:
            self._continue_candidate(
                global_state,
                local_id,
                frame_index,
                candidate_overlap,
                speed,
                centroid,
                evidence_by_bin[active.bin_id],
                allow_confirmation=True,
            )
        else:
            self._handle_gap(
                global_state,
                local_id,
                frame_index,
                speed,
                "OVERLAP_LOST",
            )
        return self._decorate(status, global_state, frame_index)

    def finish_frame(
        self,
        global_rolls: Sequence[GlobalRollState],
        observed_global_ids: set[int],
        frame_index: int,
    ) -> None:
        """Advance provisional placement or loss diagnostics without fake OBBs."""

        for global_state in global_rolls:
            self._roll_states[global_state.global_id] = global_state
            if global_state.global_id in observed_global_ids or global_state.bin_confirmed:
                continue
            candidate = self._active_candidates.get(global_state.global_id)
            if candidate is None:
                if global_state.state in {OCCLUDED, LOST}:
                    if self._start_entry_zone_provisional(
                        global_state, frame_index
                    ):
                        continue
                    self._record_possible_detector_miss(global_state, frame_index)
                    self._mark_unresolved(global_state)
                continue
            if candidate.provisional:
                elapsed = frame_index / self.fps - (
                    candidate.provisional_start_time or frame_index / self.fps
                )
                if elapsed + 1e-9 >= self.placement_grace_seconds:
                    self._confirm_candidate(
                        global_state,
                        global_state.current_local_id,
                        frame_index,
                        candidate.current_overlap,
                        candidate.current_speed,
                        method="DISAPPEAR_IN_BIN",
                    )
                continue
            gap_seconds = frame_index / self.fps - candidate.last_valid_time
            if (
                global_state.state in {OCCLUDED, LOST}
                and gap_seconds <= self.gap_tolerance_seconds + 1e-9
                and self._qualifies_for_provisional(
                    global_state.global_id, candidate
                )
            ):
                self._start_provisional(
                    global_state,
                    global_state.current_local_id,
                    frame_index,
                )
                continue
            reason = (
                "TRACK_LOST_BEFORE_MEANINGFUL_ENTRY"
                if global_state.state in {OCCLUDED, LOST}
                else "OVERLAP_LOST"
            )
            self._handle_gap(
                global_state,
                global_state.current_local_id,
                frame_index,
                global_state.candidate_current_speed,
                reason,
            )
            if global_state.state in {OCCLUDED, LOST}:
                self._mark_unresolved(global_state)

    def process_static_observation(
        self,
        global_rolls: Sequence[GlobalRollState],
        local_id: int,
        obb: Sequence[Sequence[float]] | np.ndarray,
        frame_index: int,
    ) -> tuple[int, BinFrameStatus] | None:
        """Continue placement evidence without granting a transport Global ID."""

        if not self.enabled:
            return None
        states = {state.global_id: state for state in global_rolls}
        self._roll_states.update(states)
        corners = np.asarray(obb, dtype=np.float32)
        centroid_array = corners.mean(axis=0)
        centroid = (float(centroid_array[0]), float(centroid_array[1]))
        overlaps = obb_bin_overlap(self.geometry, self.camera_id or "", corners)
        status_by_id = {item.physical_bin_id: item for item in overlaps}

        owner_id = self.static_local_to_owner.get(local_id)
        if owner_id is not None:
            owner_state = states.get(owner_id)
            candidate = self._active_candidates.get(owner_id)
            if owner_state is None:
                self.static_local_to_owner.pop(local_id, None)
                return None
            if owner_state.bin_confirmed:
                status = self._frame_status(owner_id, overlaps)
                status.candidate_bin_id = owner_state.actual_bin_id
                status.placement_state = PLACEMENT_CONFIRMED
                return owner_id, status
            if candidate is None:
                self.static_local_to_owner.pop(local_id, None)
                self.owner_to_static_local.pop(owner_id, None)
                return None
        else:
            matches: list[tuple[float, int]] = []
            for global_id, candidate in self._active_candidates.items():
                state = states.get(global_id)
                evidence = status_by_id.get(candidate.bin_id)
                if state is None or evidence is None or state.bin_confirmed:
                    continue
                gap_seconds = (frame_index - candidate.last_valid_frame) / self.fps
                distance = math.dist(
                    (candidate.last_centroid_x, candidate.last_centroid_y),
                    centroid,
                )
                same_bin = (
                    evidence.overlap_ratio >= self.overlap_threshold
                    or evidence.centroid_inside
                )
                entering = (
                    candidate.max_overlap >= self.overlap_threshold
                    and candidate.visible_frames >= 3
                    and (
                        candidate.last_centroid_inside
                        or candidate.overlap_increased
                        or candidate.moved_deeper
                    )
                )
                if (
                    0.0 < gap_seconds <= self.static_handoff_seconds
                    and distance <= self.static_handoff_distance_px
                    and same_bin
                    and entering
                ):
                    matches.append((distance, global_id))
            if not matches:
                return None
            matches.sort()
            if len(matches) > 1 and matches[1][0] - matches[0][0] < 20.0:
                return None
            _distance, owner_id = matches[0]
            owner_state = states[owner_id]
            candidate = self._active_candidates[owner_id]
            self.static_local_to_owner[local_id] = owner_id
            self.owner_to_static_local[owner_id] = local_id
            owner_state.placement_owner_static_local_id = local_id
            self._log_event(
                frame_index=frame_index,
                global_id=owner_id,
                local_id=local_id,
                candidate=candidate,
                physical_bin_id=candidate.bin_id,
                previous_bin_id="",
                current_overlap=status_by_id[candidate.bin_id].overlap_ratio,
                speed=0.0,
                event="BIN_STATIC_HANDOFF_LINKED",
                reason="ACTIVE_TO_STATIC_PLACEMENT_OWNER",
            )

        candidate = self._active_candidates.get(owner_id)
        if candidate is None:
            return None
        evidence = status_by_id.get(candidate.bin_id)
        if evidence is None or not (
            evidence.overlap_ratio >= self.overlap_threshold
            or evidence.centroid_inside
        ):
            return None
        timestamp = frame_index / self.fps
        speed = self._record_and_measure_speed(owner_id, timestamp, centroid)
        self._continue_candidate(
            owner_state,
            local_id,
            frame_index,
            evidence.overlap_ratio,
            speed,
            centroid,
            evidence,
            allow_confirmation=True,
            confirmation_method="ACTIVE_TO_STATIC_HANDOFF",
        )
        return owner_id, self._decorate(status=self._frame_status(owner_id, overlaps), global_state=owner_state, frame_index=frame_index)

    def finalize(
        self, global_rolls: Sequence[GlobalRollState], frame_index: int
    ) -> None:
        """Close every unresolved candidate so every failure has a reason."""

        if self._finalized:
            return
        states = {state.global_id: state for state in global_rolls}
        self._roll_states.update(states)
        for global_id in list(self._active_candidates):
            state = states[global_id]
            candidate = self._active_candidates[global_id]
            candidate.dwell_seconds = max(
                candidate.dwell_seconds,
                frame_index / self.fps - candidate.start_time,
            )
            if candidate.provisional:
                provisional_elapsed = frame_index / self.fps - (
                    candidate.provisional_start_time or frame_index / self.fps
                )
                if provisional_elapsed + 1e-9 >= self.placement_grace_seconds:
                    self._confirm_candidate(
                        state,
                        state.current_local_id,
                        frame_index,
                        candidate.current_overlap,
                        candidate.current_speed,
                        method="DISAPPEAR_IN_BIN",
                    )
                    continue
                reason = "VIDEO_ENDED_DURING_PROVISIONAL"
            elif candidate.dwell_seconds >= self.visible_dwell_seconds:
                reason = "NOT_SETTLED"
            elif state.state in {OCCLUDED, LOST}:
                reason = "TRACK_LOST_BEFORE_MEANINGFUL_ENTRY"
            else:
                reason = "DWELL_NOT_LONG_ENOUGH"
            self._cancel_candidate(
                state,
                state.current_local_id,
                frame_index,
                candidate.current_overlap,
                candidate.current_speed,
                reason,
            )
            self._mark_unresolved(state)
        self._finalized = True

    def _frame_status(
        self, global_id: int, overlaps: tuple[BinOverlap, ...]
    ) -> BinFrameStatus:
        best = overlaps[0] if overlaps else None
        second = overlaps[1] if len(overlaps) > 1 else None
        ambiguous = bool(
            best is not None
            and second is not None
            and best.overlap_ratio >= self.overlap_threshold
            and second.overlap_ratio >= self.overlap_threshold
            and best.overlap_ratio - second.overlap_ratio < self.ambiguity_margin
        )
        return BinFrameStatus(
            global_id=global_id,
            best_bin_id="" if best is None else best.physical_bin_id,
            best_overlap=0.0 if best is None else best.overlap_ratio,
            second_bin_id="" if second is None else second.physical_bin_id,
            second_overlap=0.0 if second is None else second.overlap_ratio,
            ambiguous=ambiguous,
        )

    def _decorate(
        self,
        status: BinFrameStatus,
        global_state: GlobalRollState,
        frame_index: int | None = None,
    ) -> BinFrameStatus:
        status.candidate_bin_id = global_state.candidate_bin_id
        status.placement_state = global_state.bin_placement_state
        status.dwell_seconds = global_state.candidate_dwell_seconds
        status.current_speed = global_state.candidate_current_speed
        if (
            frame_index is not None
            and global_state.provisional_start_time is not None
        ):
            status.provisional_elapsed_seconds = max(
                0.0,
                frame_index / self.fps - global_state.provisional_start_time,
            )
        return status

    def _record_and_measure_speed(
        self,
        global_id: int,
        timestamp: float,
        centroid: tuple[float, float],
    ) -> float:
        history = self._speed_history[global_id]
        if history and abs(history[-1][0] - timestamp) <= 1e-9:
            history[-1] = (timestamp, centroid[0], centroid[1])
        else:
            history.append((timestamp, centroid[0], centroid[1]))
        cutoff = timestamp - SPEED_WINDOW_SECONDS
        while len(history) > 2 and history[1][0] < cutoff:
            history.popleft()
        points = list(history)
        speeds: list[float] = []
        for previous, current in zip(points, points[1:]):
            delta_time = current[0] - previous[0]
            if delta_time <= 1e-9 or current[0] < cutoff:
                continue
            speeds.append(math.dist(previous[1:], current[1:]) / delta_time)
        return float(median(speeds)) if speeds else 0.0

    def _record_bin_approach(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        centroid: tuple[float, float],
        overlaps: tuple[BinOverlap, ...],
    ) -> None:
        global_id = global_state.global_id
        overlap_by_bin = {
            item.physical_bin_id: item.overlap_ratio for item in overlaps
        }
        for bin_id in self.configured_bin_ids:
            history = self._approach_history[global_id][bin_id]
            history.append(
                (
                    frame_index,
                    self._outside_distance(bin_id, centroid),
                    overlap_by_bin.get(bin_id, 0.0),
                )
            )
            if self.geometry is None or self.camera_id is None:
                continue
            entry_polygon = self.geometry.entry_polygons_for_camera(
                self.camera_id
            ).get(bin_id)
            if entry_polygon is None:
                continue
            key = (global_id, bin_id)
            inside_entry = point_in_bin_entry_zone(
                self.geometry, self.camera_id, bin_id, centroid
            )
            was_inside = self._entry_zone_inside.get(key, False)
            self._entry_zone_inside[key] = inside_entry
            if not inside_entry:
                continue
            if not was_inside:
                # A credible crossing must come from an observed point outside
                # the mouth; a roll first detected inside is not enough.
                if len(self._speed_history.get(global_id, ())) < 2:
                    continue
                self._entry_zone_cross_frame[key] = frame_index
            crossing_frame = self._entry_zone_cross_frame.get(key)
            if crossing_frame is None:
                continue
            assessment = self._entry_motion_assessment(bin_id, global_id)
            if assessment is None:
                continue
            direction_score, consistency, progress, last_centroid = assessment
            if (
                direction_score < ENTRY_MIN_DIRECTION_SCORE
                or consistency < ENTRY_MIN_TRAJECTORY_CONSISTENCY
                or progress < ENTRY_APPROACH_MIN_PROGRESS_PX
            ):
                continue
            global_state.bin_motion_relation = ENTERING_BIN
            global_state.related_bin_id = bin_id
            evidence = EntryEvidence(
                global_id=global_id,
                bin_id=bin_id,
                entry_frame=crossing_frame,
                entry_timestamp=crossing_frame / self.fps,
                direction_score=direction_score,
                trajectory_consistency=consistency,
                center_progress_px=progress,
                last_visible_frame=frame_index,
                last_visible_centroid=last_centroid,
                last_center_distance=self._distance_to_bin_center(
                    bin_id, last_centroid
                ),
                max_occupancy_overlap=max(item[2] for item in history),
            )
            previous = self._entry_evidence.get(key)
            self._entry_evidence[key] = evidence
            self._entry_zone_last_frame[global_id][bin_id] = frame_index
            self._entry_zone_crossings.add(key)
            global_state.entered_bin_mouth = True
            global_state.entry_frame = crossing_frame
            global_state.entry_timestamp = crossing_frame / self.fps
            global_state.entry_bin_id = bin_id
            global_state.entry_direction_score = direction_score
            global_state.entry_trajectory_consistency = consistency
            global_state.last_visible_centroid = last_centroid
            active_candidate = self._active_candidates.get(global_id)
            if active_candidate is not None and active_candidate.bin_id == bin_id:
                active_candidate.entered_bin_mouth = True
                active_candidate.entry_frame = crossing_frame
                active_candidate.entry_direction_score = direction_score
                active_candidate.entry_trajectory_consistency = consistency
                active_candidate.entry_confidence = evidence.confidence
            if previous is None or previous.entry_frame != crossing_frame:
                self._log_event(
                    frame_index=frame_index,
                    global_id=global_id,
                    local_id=local_id,
                    candidate=self._active_candidates.get(global_id),
                    physical_bin_id=bin_id,
                    previous_bin_id="",
                    current_overlap=overlap_by_bin.get(bin_id, 0.0),
                    speed=global_state.candidate_current_speed,
                    event="BIN_MOUTH_ENTRY_CONFIRMED",
                    reason=(
                        f"direction={direction_score:.3f};"
                        f"consistency={consistency:.3f};"
                        f"center_progress={progress:.1f}px"
                    ),
                )

    def _entry_motion_assessment(
        self, bin_id: str, global_id: int
    ) -> tuple[float, float, float, tuple[float, float]] | None:
        points = list(self._speed_history.get(global_id, ()))
        if len(points) < 3:
            return None
        centroids = [(item[1], item[2]) for item in points]
        first = centroids[0]
        last = centroids[-1]
        movement = (last[0] - first[0], last[1] - first[1])
        movement_length = math.hypot(*movement)
        if movement_length <= 1e-6:
            return None
        if self.geometry is None or not self.camera_id:
            return None
        polygon = self.geometry.polygons_for_camera(self.camera_id).get(bin_id)
        if polygon is None:
            return None
        center_array = np.asarray(polygon, dtype=np.float32).mean(axis=0)
        center = (float(center_array[0]), float(center_array[1]))
        inward = (center[0] - first[0], center[1] - first[1])
        inward_length = math.hypot(*inward)
        if inward_length <= 1e-6:
            return None
        direction_score = max(
            0.0,
            min(
                1.0,
                (movement[0] * inward[0] + movement[1] * inward[1])
                / (movement_length * inward_length),
            ),
        )
        path_length = sum(
            math.dist(previous, current)
            for previous, current in zip(centroids, centroids[1:])
        )
        consistency = movement_length / path_length if path_length > 1e-6 else 0.0
        progress = math.dist(first, center) - math.dist(last, center)
        return direction_score, consistency, progress, last

    def _outside_distance(
        self, bin_id: str, centroid: tuple[float, float]
    ) -> float:
        if self.geometry is None or not self.camera_id:
            return math.inf
        polygon = self.geometry.polygons_for_camera(self.camera_id).get(bin_id)
        if polygon is None:
            return math.inf
        signed = float(cv2.pointPolygonTest(polygon, centroid, True))
        return max(0.0, -signed)

    def _distance_to_bin_center(
        self, bin_id: str, centroid: tuple[float, float]
    ) -> float:
        if self.geometry is None or not self.camera_id:
            return math.inf
        polygon = self.geometry.polygons_for_camera(self.camera_id).get(bin_id)
        if polygon is None:
            return math.inf
        center = np.asarray(polygon, dtype=np.float32).mean(axis=0)
        return math.dist(centroid, (float(center[0]), float(center[1])))

    def _qualifies_for_provisional(
        self, global_id: int, candidate: CandidateDiagnostic
    ) -> bool:
        """Require a real, manually configured mouth crossing before loss."""

        evidence = self._entry_evidence.get((global_id, candidate.bin_id))
        if evidence is None:
            return False
        loss_gap = (
            candidate.last_valid_frame - evidence.last_visible_frame
        ) / self.fps
        return bool(
            loss_gap >= 0.0
            and loss_gap <= ENTRY_ZONE_LOSS_WINDOW_SECONDS
            and evidence.direction_score >= ENTRY_MIN_DIRECTION_SCORE
            and evidence.trajectory_consistency
            >= ENTRY_MIN_TRAJECTORY_CONSISTENCY
            and evidence.center_progress_px >= ENTRY_APPROACH_MIN_PROGRESS_PX
        )

    def _start_entry_zone_provisional(
        self, global_state: GlobalRollState, frame_index: int
    ) -> bool:
        """Bridge a user-drawn bin mouth crossing into provisional placement."""

        global_id = global_state.global_id
        choices: list[
            tuple[float, float, str, EntryEvidence, list[tuple[int, float, float]]]
        ] = []
        for (evidence_global_id, bin_id), evidence in self._entry_evidence.items():
            if evidence_global_id != global_id:
                continue
            loss_gap = (frame_index - evidence.last_visible_frame) / self.fps
            history = list(self._approach_history[global_id][bin_id])
            if (
                loss_gap < 0
                or loss_gap > ENTRY_ZONE_LOSS_WINDOW_SECONDS
                or len(history) < 3
            ):
                continue
            if (
                evidence.direction_score < ENTRY_MIN_DIRECTION_SCORE
                or evidence.trajectory_consistency
                < ENTRY_MIN_TRAJECTORY_CONSISTENCY
                or evidence.center_progress_px < ENTRY_APPROACH_MIN_PROGRESS_PX
            ):
                continue
            choices.append(
                (
                    -evidence.confidence,
                    evidence.last_center_distance,
                    bin_id,
                    evidence,
                    history,
                )
            )
        if not choices:
            return False
        choices.sort()
        if (
            len(choices) > 1
            and abs(choices[0][0] - choices[1][0]) < self.ambiguity_margin
        ):
            return False
        _negative_confidence, _distance, bin_id, entry_evidence, history = choices[0]
        timestamp = frame_index / self.fps
        speed_points = list(self._speed_history.get(global_id, ()))
        if speed_points:
            last_centroid = (speed_points[-1][1], speed_points[-1][2])
        else:
            last_centroid = global_state.last_centroid
        max_overlap = max(item[2] for item in history)
        candidate = CandidateDiagnostic(
            global_id=global_id,
            bin_id=bin_id,
            start_frame=history[0][0],
            start_time=history[0][0] / self.fps,
            last_valid_frame=history[-1][0],
            last_valid_time=history[-1][0] / self.fps,
            max_overlap=max_overlap,
            current_overlap=history[-1][2],
            dwell_seconds=max(0.0, timestamp - history[0][0] / self.fps),
            current_speed=global_state.last_speed,
            visible_frames=len(history),
            first_overlap=history[0][2],
            first_distance_to_bin_center=self._distance_to_bin_center(
                bin_id, last_centroid
            ),
            last_distance_to_bin_center=self._distance_to_bin_center(
                bin_id, last_centroid
            ),
            last_centroid_x=last_centroid[0],
            last_centroid_y=last_centroid[1],
            overlap_increased=max_overlap - history[0][2] >= ENTRY_OVERLAP_GROWTH,
            moved_deeper=True,
            entered_bin_mouth=True,
            entry_frame=entry_evidence.entry_frame,
            entry_direction_score=entry_evidence.direction_score,
            entry_trajectory_consistency=entry_evidence.trajectory_consistency,
            entry_confidence=entry_evidence.confidence,
        )
        self._active_candidates[global_id] = candidate
        self._had_candidate.add(global_id)
        self.candidate_history[global_id].append(candidate)
        self.candidates_started += 1
        global_state.candidate_bin_id = bin_id
        global_state.candidate_bin_frames = len(history)
        global_state.candidate_bin_max_overlap = max_overlap
        global_state.candidate_bin_last_overlap = history[-1][2]
        global_state.candidate_start_frame = candidate.start_frame
        global_state.candidate_start_time = candidate.start_time
        global_state.candidate_last_valid_frame = candidate.last_valid_frame
        global_state.candidate_last_valid_time = candidate.last_valid_time
        global_state.candidate_dwell_seconds = candidate.dwell_seconds
        global_state.candidate_current_speed = candidate.current_speed
        if bin_id not in global_state.candidate_bins_seen:
            global_state.candidate_bins_seen.append(bin_id)
        self._log_event(
            frame_index=frame_index,
            global_id=global_id,
            local_id=global_state.current_local_id,
            candidate=candidate,
            physical_bin_id=bin_id,
            previous_bin_id="",
            current_overlap=candidate.current_overlap,
            speed=candidate.current_speed,
            event="BIN_ENTRY_ZONE_CROSSED",
            reason="CREDIBLE_ENTRY_THEN_DISAPPEARANCE",
        )
        self._start_provisional(
            global_state, global_state.current_local_id, frame_index
        )
        return True

    def _record_possible_detector_miss(
        self, global_state: GlobalRollState, frame_index: int
    ) -> None:
        global_id = global_state.global_id
        if (
            global_id in self._possible_detector_miss_ids
            or global_id in self._had_candidate
            or not self.enabled
        ):
            return
        candidates: list[tuple[float, float, str, float]] = []
        for bin_id in self.configured_bin_ids:
            history = list(self._approach_history[global_id][bin_id])
            if len(history) < 3:
                continue
            first_distance = history[0][1]
            last_distance = history[-1][1]
            progress = first_distance - last_distance
            max_overlap = max(item[2] for item in history)
            if (
                last_distance <= self.detector_miss_distance_px
                and progress >= ENTRY_APPROACH_MIN_PROGRESS_PX
                and max_overlap < self.overlap_threshold
            ):
                candidates.append((last_distance, -progress, bin_id, max_overlap))
        if not candidates:
            return
        last_distance, negative_progress, bin_id, max_overlap = min(candidates)
        progress = -negative_progress
        self._possible_detector_miss_ids.add(global_id)
        global_state.possible_detector_miss_bin_id = bin_id
        item = {
            "global_id": global_id,
            "bin_id": bin_id,
            "frame": frame_index,
            "last_distance_px": last_distance,
            "approach_progress_px": progress,
            "max_overlap": max_overlap,
        }
        self.possible_detector_misses.append(item)
        self._log_event(
            frame_index=frame_index,
            global_id=global_id,
            local_id=global_state.current_local_id,
            candidate=None,
            physical_bin_id=bin_id,
            previous_bin_id="",
            current_overlap=max_overlap,
            speed=global_state.candidate_current_speed,
            event="POSSIBLE_DETECTOR_MISS_AT_BIN_ENTRY",
            reason="APPROACHED_BIN_WITHOUT_MEANINGFUL_OVERLAP",
        )

    def _start_candidate(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        bin_id: str,
        overlap: float,
        speed: float,
        centroid: tuple[float, float],
        evidence: BinOverlap,
        *,
        event: str,
        previous_bin_id: str = "",
    ) -> None:
        timestamp = frame_index / self.fps
        candidate = CandidateDiagnostic(
            global_id=global_state.global_id,
            bin_id=bin_id,
            start_frame=frame_index,
            start_time=timestamp,
            last_valid_frame=frame_index,
            last_valid_time=timestamp,
            max_overlap=overlap,
            current_overlap=overlap,
            current_speed=speed,
            first_overlap=overlap,
            first_distance_to_bin_center=self._distance_to_bin_center(
                bin_id, centroid
            ),
            last_distance_to_bin_center=self._distance_to_bin_center(
                bin_id, centroid
            ),
            last_centroid_x=centroid[0],
            last_centroid_y=centroid[1],
            last_centroid_inside=evidence.centroid_inside,
        )
        self._active_candidates[global_state.global_id] = candidate
        self._had_candidate.add(global_state.global_id)
        self.candidate_history[global_state.global_id].append(candidate)
        self.candidates_started += 1
        global_state.candidate_bin_id = bin_id
        global_state.candidate_bin_frames = 1
        global_state.candidate_bin_max_overlap = overlap
        global_state.candidate_bin_last_overlap = overlap
        global_state.bin_placement_state = BIN_CANDIDATE
        global_state.candidate_start_frame = frame_index
        global_state.candidate_start_time = timestamp
        global_state.candidate_last_valid_frame = frame_index
        global_state.candidate_last_valid_time = timestamp
        global_state.candidate_dwell_seconds = 0.0
        global_state.candidate_valid_overlap_seconds = 0.0
        global_state.candidate_current_speed = speed
        if bin_id not in global_state.candidate_bins_seen:
            global_state.candidate_bins_seen.append(bin_id)
        self._log_event(
            frame_index=frame_index,
            global_id=global_state.global_id,
            local_id=local_id,
            candidate=candidate,
            physical_bin_id=bin_id,
            previous_bin_id=previous_bin_id,
            current_overlap=overlap,
            speed=speed,
            event=event,
            reason="",
        )
        if self.debug:
            print(
                f"[frame {frame_index}] G{global_state.global_id} candidate "
                f"{bin_id} started; overlap={overlap:.3f}"
            )

    def _continue_candidate(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        overlap: float,
        speed: float,
        centroid: tuple[float, float],
        evidence: BinOverlap,
        *,
        allow_confirmation: bool,
        confirmation_method: str = "VISIBLE_SETTLED",
    ) -> None:
        candidate = self._active_candidates[global_state.global_id]
        timestamp = frame_index / self.fps
        candidate.valid_overlap_seconds += max(
            0.0, timestamp - candidate.last_valid_time
        )
        candidate.last_valid_frame = frame_index
        candidate.last_valid_time = timestamp
        candidate.current_overlap = overlap
        candidate.max_overlap = max(candidate.max_overlap, overlap)
        candidate.dwell_seconds = max(0.0, timestamp - candidate.start_time)
        candidate.current_speed = speed
        candidate.visible_frames += 1
        candidate.overlap_increased = candidate.overlap_increased or (
            overlap - candidate.first_overlap >= ENTRY_OVERLAP_GROWTH
        )
        candidate.last_distance_to_bin_center = self._distance_to_bin_center(
            candidate.bin_id, centroid
        )
        candidate.moved_deeper = candidate.moved_deeper or (
            candidate.first_distance_to_bin_center
            - candidate.last_distance_to_bin_center
            >= ENTRY_APPROACH_MIN_PROGRESS_PX
        )
        candidate.last_centroid_x = centroid[0]
        candidate.last_centroid_y = centroid[1]
        candidate.last_centroid_inside = evidence.centroid_inside
        global_state.candidate_bin_frames += 1
        global_state.candidate_bin_max_overlap = candidate.max_overlap
        global_state.candidate_bin_last_overlap = overlap
        global_state.bin_placement_state = (
            PROVISIONAL_PLACEMENT if candidate.provisional else BIN_CANDIDATE
        )
        global_state.candidate_last_valid_frame = frame_index
        global_state.candidate_last_valid_time = timestamp
        global_state.candidate_dwell_seconds = candidate.dwell_seconds
        global_state.candidate_valid_overlap_seconds = candidate.valid_overlap_seconds
        global_state.candidate_current_speed = speed
        self._log_event(
            frame_index=frame_index,
            global_id=global_state.global_id,
            local_id=local_id,
            candidate=candidate,
            physical_bin_id=candidate.bin_id,
            previous_bin_id="",
            current_overlap=overlap,
            speed=speed,
            event="BIN_CANDIDATE_CONTINUED",
            reason=(
                "NOT_SETTLED"
                if candidate.dwell_seconds >= self.visible_dwell_seconds
                and speed > self.settle_speed_px_per_sec
                else ""
            ),
        )
        if (
            allow_confirmation
            and candidate.dwell_seconds + 1e-9 >= self.visible_dwell_seconds
            and speed <= self.settle_speed_px_per_sec
        ):
            self._confirm_candidate(
                global_state,
                local_id,
                frame_index,
                overlap,
                speed,
                method=confirmation_method,
            )

    def _handle_gap(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        speed: float,
        expiration_reason: str,
    ) -> None:
        candidate = self._active_candidates.get(global_state.global_id)
        if candidate is None:
            return
        timestamp = frame_index / self.fps
        gap_seconds = max(0.0, timestamp - candidate.last_valid_time)
        candidate.dwell_seconds = max(0.0, timestamp - candidate.start_time)
        candidate.current_overlap = 0.0
        candidate.current_speed = speed
        global_state.candidate_bin_last_overlap = 0.0
        global_state.candidate_dwell_seconds = candidate.dwell_seconds
        global_state.candidate_current_speed = speed
        if gap_seconds <= self.gap_tolerance_seconds + 1e-9:
            return
        reason = (
            "NOT_SETTLED"
            if candidate.dwell_seconds >= self.visible_dwell_seconds
            and candidate.current_speed > self.settle_speed_px_per_sec
            else expiration_reason
        )
        self._cancel_candidate(
            global_state, local_id, frame_index, 0.0, speed, reason
        )

    def _switch_candidate(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        new_bin_id: str,
        overlap: float,
        speed: float,
        centroid: tuple[float, float],
        evidence: BinOverlap,
    ) -> None:
        previous = self._active_candidates[global_state.global_id]
        previous.end_frame = frame_index
        previous.reason = "MOVED_TO_DIFFERENT_BIN"
        previous.dwell_seconds = max(
            previous.dwell_seconds, frame_index / self.fps - previous.start_time
        )
        previous_bin_id = previous.bin_id
        self.candidates_cancelled += 1
        self.candidates_switched += 1
        self._clear_candidate_state(global_state)
        self._start_candidate(
            global_state,
            local_id,
            frame_index,
            new_bin_id,
            overlap,
            speed,
            centroid,
            evidence,
            event="BIN_CANDIDATE_SWITCHED",
            previous_bin_id=previous_bin_id,
        )

    def _start_provisional(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
    ) -> None:
        candidate = self._active_candidates[global_state.global_id]
        if candidate.provisional:
            return
        entry_evidence = self._entry_evidence.get(
            (global_state.global_id, candidate.bin_id)
        )
        if entry_evidence is None or not self._qualifies_for_provisional(
            global_state.global_id, candidate
        ):
            self._mark_unresolved(global_state)
            return
        timestamp = frame_index / self.fps
        candidate.provisional = True
        candidate.provisional_start_frame = frame_index
        candidate.provisional_start_time = timestamp
        candidate.disappeared_inside_candidate = True
        global_state.bin_placement_state = PROVISIONAL_PLACEMENT
        global_state.disappeared_inside_candidate = True
        global_state.provisional_placement_seen = True
        global_state.provisional_start_frame = frame_index
        global_state.provisional_start_time = timestamp
        global_state.placement_status = PROVISIONAL
        global_state.placement_evidence = ENTRY_THEN_OCCLUDED
        global_state.placement_confidence = entry_evidence.confidence
        global_state.verification_result = UNRESOLVED
        global_state.state = PLACEMENT_PENDING
        global_state.reassociation_allowed = True
        global_state.bin_motion_relation = ENTERING_BIN
        global_state.related_bin_id = candidate.bin_id
        self.provisional_started += 1
        self._log_event(
            frame_index=frame_index,
            global_id=global_state.global_id,
            local_id=local_id,
            candidate=candidate,
            physical_bin_id=candidate.bin_id,
            previous_bin_id="",
            current_overlap=candidate.current_overlap,
            speed=candidate.current_speed,
            event="BIN_PROVISIONAL_PLACEMENT",
            reason="DISAPPEARED_WITH_STRONG_ENTRY_EVIDENCE",
        )
        if self.debug:
            print(
                f"[frame {frame_index}] G{global_state.global_id} "
                f"PROVISIONAL {candidate.bin_id}; waiting "
                f"{self.placement_grace_seconds:.2f}s"
            )

    def _cancel_provisional(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        overlap: float,
        speed: float,
        *,
        reason: str,
    ) -> None:
        candidate = self._active_candidates.get(global_state.global_id)
        if candidate is None:
            return
        candidate.reappeared_outside = True
        global_state.reappeared_outside_candidate = True
        self._log_event(
            frame_index=frame_index,
            global_id=global_state.global_id,
            local_id=local_id,
            candidate=candidate,
            physical_bin_id=candidate.bin_id,
            previous_bin_id="",
            current_overlap=overlap,
            speed=speed,
            event="BIN_PROVISIONAL_CANCELLED",
            reason=reason,
        )
        self.provisional_cancelled += 1
        self._cancel_candidate(
            global_state,
            local_id,
            frame_index,
            overlap,
            speed,
            reason,
            log_event=False,
        )
        global_state.placement_status = TRACKING
        global_state.placement_evidence = EVIDENCE_NONE
        global_state.placement_confidence = 0.0
        global_state.verification_result = UNRESOLVED
        if global_state.state == PLACEMENT_PENDING:
            global_state.state = (
                ACTIVE if global_state.last_motion_active else VISIBLE
            )

    def _cancel_candidate(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        overlap: float,
        speed: float,
        reason: str,
        *,
        log_event: bool = True,
    ) -> None:
        candidate = self._active_candidates.get(global_state.global_id)
        if candidate is None:
            return
        candidate.end_frame = frame_index
        candidate.reason = reason
        candidate.current_overlap = overlap
        candidate.current_speed = speed
        candidate.dwell_seconds = max(
            candidate.dwell_seconds, frame_index / self.fps - candidate.start_time
        )
        if log_event:
            self._log_event(
                frame_index=frame_index,
                global_id=global_state.global_id,
                local_id=local_id,
                candidate=candidate,
                physical_bin_id=candidate.bin_id,
                previous_bin_id="",
                current_overlap=overlap,
                speed=speed,
                event="BIN_CANDIDATE_CANCELLED",
                reason=reason,
            )
        self.candidates_cancelled += 1
        global_state.last_candidate_cancel_reason = reason
        if self.debug:
            print(
                f"[frame {frame_index}] G{global_state.global_id} candidate "
                f"{candidate.bin_id} cancelled: {reason}"
            )
        self._clear_candidate_state(global_state)
        global_state.bin_placement_state = PLACEMENT_CANCELLED

    @staticmethod
    def _mark_unresolved(global_state: GlobalRollState) -> None:
        if global_state.placement_confirmed:
            return
        global_state.placement_status = UNRESOLVED_STATUS
        global_state.placement_evidence = EVIDENCE_NONE
        global_state.placement_confidence = 0.0
        global_state.verification_result = UNRESOLVED

    def _confirm_candidate(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        overlap: float,
        speed: float,
        *,
        method: str,
    ) -> None:
        candidate = self._active_candidates[global_state.global_id]
        if global_state.placement_counted:
            return
        candidate.end_frame = frame_index
        candidate.reason = "CONFIRMED"
        candidate.confirmed = True
        candidate.confirmation_method = method
        # A physical placement is owned by this roll. Per-bin totals are only
        # retrospective diagnostics and have no role in this decision.
        global_state.actual_bin_id = candidate.bin_id
        global_state.bin_confirmed = True
        global_state.bin_entry_frame = frame_index
        global_state.bin_placement_state = PLACEMENT_CONFIRMED
        global_state.placement_counted = True
        global_state.placement_dwell_seconds = candidate.dwell_seconds
        global_state.placement_settled_speed = speed
        global_state.placement_reason = method
        global_state.placement_confirmation_method = method
        global_state.factory_placement_completed = True
        global_state.placement_status = CONFIRMED
        if method == "DISAPPEAR_IN_BIN":
            global_state.placement_evidence = ENTRY_THEN_OCCLUDED
            entry_evidence = self._entry_evidence.get(
                (global_state.global_id, candidate.bin_id)
            )
            global_state.placement_confidence = (
                candidate.entry_confidence
                if entry_evidence is None
                else entry_evidence.confidence
            )
        elif method == "ACTIVE_TO_STATIC_HANDOFF":
            global_state.placement_evidence = ACTIVE_TO_STATIC
            global_state.placement_confidence = 0.95
        else:
            global_state.placement_evidence = DIRECT_VISIBLE
            global_state.placement_confidence = 1.0
        self.debug_placement_counts[candidate.bin_id] += 1
        self.placements_confirmed += 1
        self._update_verification(
            global_state,
            local_id=local_id,
            frame_index=frame_index,
            candidate=candidate,
            overlap=overlap,
            speed=speed,
            emit_event=False,
        )
        method_event = (
            "BIN_DISAPPEAR_CONFIRMED"
            if method == "DISAPPEAR_IN_BIN"
            else "BIN_VISIBLE_SETTLED_CONFIRMED"
        )
        self._log_event(
            frame_index=frame_index,
            global_id=global_state.global_id,
            local_id=local_id,
            candidate=candidate,
            physical_bin_id=candidate.bin_id,
            previous_bin_id="",
            current_overlap=overlap,
            speed=speed,
            event=method_event,
            reason=method,
        )
        self._log_event(
            frame_index=frame_index,
            global_id=global_state.global_id,
            local_id=local_id,
            candidate=candidate,
            physical_bin_id=candidate.bin_id,
            previous_bin_id="",
            current_overlap=overlap,
            speed=speed,
            event="BIN_PLACEMENT_CONFIRMED",
            reason=method,
        )
        self._update_verification(
            global_state,
            local_id=local_id,
            frame_index=frame_index,
            candidate=candidate,
            overlap=overlap,
            speed=speed,
            emit_event=True,
        )
        self._active_candidates.pop(global_state.global_id, None)
        if self.debug:
            print(
                f"[frame {frame_index}] G{global_state.global_id} PLACED in "
                f"{candidate.bin_id}; dwell={candidate.dwell_seconds:.2f}s "
                f"speed={speed:.1f}px/s"
            )

    def assign_expected_bin(
        self,
        global_state: GlobalRollState,
        physical_bin_id: str,
        *,
        frame_index: int | None = None,
        local_id: int | None = None,
    ) -> str:
        """Assign upstream intent and verify immediately if placement is known."""

        physical_bin_id = physical_bin_id.strip().upper()
        if not physical_bin_id:
            raise ValueError("expected physical bin ID must not be empty")
        if self.geometry is not None and physical_bin_id not in self.geometry.physical_bins:
            configured = ", ".join(self.geometry.physical_bins)
            raise ValueError(
                f"Unknown physical bin {physical_bin_id!r}; configured bins: {configured}"
            )
        self._roll_states[global_state.global_id] = global_state
        global_state.expected_bin_id = physical_bin_id
        if global_state.placement_confirmed:
            self._update_verification(
                global_state,
                local_id=(
                    global_state.current_local_id if local_id is None else local_id
                ),
                frame_index=(
                    global_state.placement_frame
                    if frame_index is None
                    else frame_index
                ),
                candidate=None,
                overlap=global_state.candidate_bin_last_overlap,
                speed=global_state.placement_settled_speed,
                emit_event=True,
            )
        return global_state.verification_result

    def _update_verification(
        self,
        global_state: GlobalRollState,
        *,
        local_id: int,
        frame_index: int | None,
        candidate: CandidateDiagnostic | None,
        overlap: float,
        speed: float,
        emit_event: bool,
    ) -> None:
        """Compare this roll's instructed bin with its locked actual bin."""

        if not global_state.placement_confirmed:
            global_state.verification_result = UNRESOLVED
            return
        if global_state.expected_bin_id is None:
            global_state.verification_result = UNVERIFIED
            return
        if global_state.actual_bin_id == global_state.expected_bin_id:
            global_state.verification_result = CORRECT
            event = "ROLL_PLACEMENT_CORRECT"
        else:
            global_state.verification_result = WRONG_BIN
            event = "ROLL_PLACEMENT_WRONG_BIN"
        if not emit_event:
            return
        resolved_frame = (
            global_state.placement_frame if frame_index is None else frame_index
        )
        self._log_event(
            frame_index=resolved_frame or 0,
            global_id=global_state.global_id,
            local_id=local_id,
            candidate=candidate,
            physical_bin_id=global_state.actual_bin_id,
            previous_bin_id="",
            current_overlap=overlap,
            speed=speed,
            event=event,
            reason=global_state.verification_result,
        )
        if self.debug:
            print(
                f"[frame {resolved_frame or 0}] {global_state.roll_id} placement "
                f"expected={global_state.expected_bin_id} "
                f"actual={global_state.actual_bin_id} -> "
                f"{global_state.verification_result}"
            )

    def _clear_candidate_state(self, global_state: GlobalRollState) -> None:
        self._active_candidates.pop(global_state.global_id, None)
        global_state.candidate_bin_id = ""
        global_state.candidate_bin_frames = 0
        global_state.candidate_bin_max_overlap = 0.0
        global_state.candidate_bin_last_overlap = 0.0
        global_state.bin_placement_state = NONE
        global_state.candidate_start_frame = None
        global_state.candidate_start_time = None
        global_state.candidate_last_valid_frame = None
        global_state.candidate_last_valid_time = None
        global_state.candidate_dwell_seconds = 0.0
        global_state.candidate_valid_overlap_seconds = 0.0
        global_state.candidate_current_speed = 0.0
        global_state.provisional_start_frame = None
        global_state.provisional_start_time = None

    def _log_event(
        self,
        *,
        frame_index: int,
        global_id: int,
        local_id: int,
        candidate: CandidateDiagnostic | None,
        physical_bin_id: str,
        previous_bin_id: str,
        current_overlap: float,
        speed: float,
        event: str,
        reason: str,
    ) -> None:
        timestamp = frame_index / self.fps
        last_valid_time = candidate.last_valid_time if candidate else timestamp
        history = list(self._speed_history.get(global_id, ()))
        if len(history) >= 2:
            motion_dx = history[-1][1] - history[-2][1]
            motion_dy = history[-1][2] - history[-2][2]
        else:
            motion_dx = motion_dy = 0.0
        if event == "BIN_PROVISIONAL_CANCELLED":
            event_state = PLACEMENT_CANCELLED
        elif event in {
            "BIN_PLACEMENT_CONFIRMED",
            "BIN_VISIBLE_SETTLED_CONFIRMED",
            "BIN_DISAPPEAR_CONFIRMED",
            "ROLL_PLACEMENT_CORRECT",
            "ROLL_PLACEMENT_WRONG_BIN",
        }:
            event_state = PLACEMENT_CONFIRMED
        elif candidate is not None and candidate.provisional:
            event_state = PROVISIONAL_PLACEMENT
        elif candidate is not None:
            event_state = BIN_CANDIDATE
        else:
            event_state = NONE
        provisional_elapsed = (
            0.0
            if candidate is None or candidate.provisional_start_time is None
            else max(0.0, timestamp - candidate.provisional_start_time)
        )
        roll_state = self._roll_states.get(global_id)
        self.event_writer.writerow(
            {
                "frame": frame_index,
                "timestamp_seconds": f"{timestamp:.3f}",
                "global_id": global_id,
                "roll_id": (
                    f"G{global_id}" if roll_state is None else roll_state.roll_id
                ),
                "local_id": local_id,
                "physical_bin_id": physical_bin_id,
                "expected_bin_id": (
                    "" if roll_state is None else roll_state.expected_bin_id or ""
                ),
                "actual_bin_id": (
                    "" if roll_state is None else roll_state.actual_bin_id
                ),
                "placement_confirmed": (
                    False if roll_state is None else roll_state.placement_confirmed
                ),
                "placement_confirmation_method": (
                    ""
                    if roll_state is None
                    else roll_state.placement_confirmation_method
                ),
                "placement_frame": (
                    "" if roll_state is None else roll_state.placement_frame
                ),
                "placement_status": (
                    UNRESOLVED_STATUS
                    if roll_state is None
                    else roll_state.placement_status
                ),
                "placement_evidence": (
                    EVIDENCE_NONE
                    if roll_state is None
                    else roll_state.placement_evidence
                ),
                "placement_confidence": (
                    ""
                    if roll_state is None
                    else f"{roll_state.placement_confidence:.6f}"
                ),
                "entered_bin_mouth": (
                    False if roll_state is None else roll_state.entered_bin_mouth
                ),
                "entry_frame": (
                    "" if roll_state is None else roll_state.entry_frame
                ),
                "entry_timestamp": (
                    ""
                    if roll_state is None or roll_state.entry_timestamp is None
                    else f"{roll_state.entry_timestamp:.3f}"
                ),
                "entry_bin_id": (
                    "" if roll_state is None else roll_state.entry_bin_id
                ),
                "entry_direction_score": (
                    ""
                    if roll_state is None
                    else f"{roll_state.entry_direction_score:.6f}"
                ),
                "entry_trajectory_consistency": (
                    ""
                    if roll_state is None
                    else f"{roll_state.entry_trajectory_consistency:.6f}"
                ),
                "last_visible_centroid": (
                    ""
                    if roll_state is None or roll_state.last_visible_centroid is None
                    else (
                        f"{roll_state.last_visible_centroid[0]:.3f},"
                        f"{roll_state.last_visible_centroid[1]:.3f}"
                    )
                ),
                "verification_result": (
                    UNRESOLVED
                    if roll_state is None
                    else roll_state.verification_result
                ),
                "previous_bin_id": previous_bin_id,
                "candidate_state": event_state,
                "candidate_start_frame": "" if candidate is None else candidate.start_frame,
                "candidate_start_time": (
                    "" if candidate is None else f"{candidate.start_time:.3f}"
                ),
                "current_overlap": f"{current_overlap:.6f}",
                "max_overlap": (
                    "" if candidate is None else f"{candidate.max_overlap:.6f}"
                ),
                "accumulated_dwell_seconds": (
                    "" if candidate is None else f"{candidate.dwell_seconds:.3f}"
                ),
                "current_speed_px_per_sec": f"{speed:.3f}",
                "last_valid_overlap_time": f"{last_valid_time:.3f}",
                "gap_seconds": f"{max(0.0, timestamp - last_valid_time):.3f}",
                "last_visible_frame": (
                    "" if candidate is None else candidate.last_valid_frame
                ),
                "last_overlap": (
                    "" if candidate is None else f"{candidate.current_overlap:.6f}"
                ),
                "last_centroid_x": (
                    "" if candidate is None else f"{candidate.last_centroid_x:.3f}"
                ),
                "last_centroid_y": (
                    "" if candidate is None else f"{candidate.last_centroid_y:.3f}"
                ),
                "motion_dx": f"{motion_dx:.3f}",
                "motion_dy": f"{motion_dy:.3f}",
                "first_distance_to_bin_center": (
                    ""
                    if candidate is None
                    else f"{candidate.first_distance_to_bin_center:.3f}"
                ),
                "last_distance_to_bin_center": (
                    ""
                    if candidate is None
                    else f"{candidate.last_distance_to_bin_center:.3f}"
                ),
                "centroid_inside": (
                    "" if candidate is None else candidate.last_centroid_inside
                ),
                "overlap_increased": (
                    "" if candidate is None else candidate.overlap_increased
                ),
                "moved_deeper": (
                    "" if candidate is None else candidate.moved_deeper
                ),
                "provisional_elapsed_seconds": f"{provisional_elapsed:.3f}",
                "grace_duration_seconds": f"{self.placement_grace_seconds:.3f}",
                "confirmation_method": (
                    "" if candidate is None else candidate.confirmation_method
                ),
                "reason": reason,
                "event": event,
            }
        )
        self.event_counts[event] += 1

    def summary_lines(
        self, global_rolls: Sequence[GlobalRollState]
    ) -> list[str]:
        journeys = [state for state in global_rolls if state.journey_started]
        confirmed = [state for state in journeys if state.bin_confirmed]
        lines = [
            "Per-roll placement verification",
            "-------------------------------",
        ]
        if not journeys:
            lines.append("No transport rolls detected.")
        for state in sorted(journeys, key=lambda value: value.global_id):
            result = (
                state.verification_result
                if state.placement_confirmed
                else UNRESOLVED
            )
            lines.append(
                f"{state.roll_id} -> expected "
                f"{state.expected_bin_id or 'NONE'} -> actual "
                f"{state.actual_bin_id or 'UNKNOWN'} -> status "
                f"{state.placement_status} -> evidence "
                f"{state.placement_evidence} -> {result}"
            )
        lines.extend(
            [
                "",
                "Placement diagnostics",
                "---------------------",
                f"Bin placement enabled: {'YES' if self.enabled else 'NO'}",
                f"Bin overlap threshold: {self.overlap_threshold:.3f}",
                f"Visible settled dwell: {self.visible_dwell_seconds:.3f} sec",
                f"Bin gap tolerance: {self.gap_tolerance_seconds:.3f} sec",
                f"Occlusion confirmation grace: {self.placement_grace_seconds:.3f} sec",
                "Bin settle speed threshold: "
                f"{self.settle_speed_px_per_sec:.3f} px/sec",
                f"Speed smoothing window: {SPEED_WINDOW_SECONDS:.3f} sec (median)",
                f"Global journeys detected: {len(journeys)}",
                f"Candidates started: {self.candidates_started}",
                f"Candidates cancelled: {self.candidates_cancelled}",
                f"Candidates switched: {self.candidates_switched}",
                f"Provisional placements started: {self.provisional_started}",
                f"Provisional placements cancelled: {self.provisional_cancelled}",
                f"Placements confirmed: {self.placements_confirmed}",
                f"Journeys with no final placement: {len(journeys) - len(confirmed)}",
                f"Ambiguous bin entries: {len(self.ambiguous_global_ids)}",
                "",
                "Per-roll placement diagnostics",
                "------------------------------",
            ]
        )
        for state in sorted(journeys, key=lambda value: value.global_id):
            diagnostics = self.candidate_history.get(state.global_id, [])
            max_dwell = max((item.dwell_seconds for item in diagnostics), default=0.0)
            max_overlap = max((item.max_overlap for item in diagnostics), default=0.0)
            if state.bin_confirmed:
                reason = state.placement_reason or "UNKNOWN"
                dwell = state.placement_dwell_seconds
                speed = state.placement_settled_speed
            else:
                last = diagnostics[-1] if diagnostics else None
                reason = (
                    last.reason
                    if last is not None and last.reason
                    else state.last_candidate_cancel_reason or "NO_BIN_CANDIDATE"
                )
                dwell = max_dwell
                speed = 0.0 if last is None else last.current_speed
            lines.extend(
                [
                    f"{state.roll_id}:",
                    "  camera_journey_state: "
                    f"{'CLOSED' if state.state == COMPLETED else state.state}",
                    "  candidate_bins: "
                    + (", ".join(state.candidate_bins_seen) or "NONE"),
                    f"  candidate_bin: "
                    f"{(diagnostics[-1].bin_id if diagnostics else 'NONE')}",
                    f"  expected_bin_id: {state.expected_bin_id or 'NONE'}",
                    f"  actual_bin_id: {state.actual_bin_id or 'NONE'}",
                    "  placement_confirmed: "
                    f"{'yes' if state.placement_confirmed else 'no'}",
                    f"  placement_frame: {state.placement_frame if state.placement_frame is not None else 'NONE'}",
                    f"  placement_status: {state.placement_status}",
                    f"  placement_evidence: {state.placement_evidence}",
                    f"  placement_confidence: {state.placement_confidence:.3f}",
                    "  entered_bin_mouth: "
                    f"{'yes' if state.entered_bin_mouth else 'no'}",
                    f"  entry_frame: {state.entry_frame if state.entry_frame is not None else 'NONE'}",
                    "  entry_timestamp: "
                    + (
                        "NONE"
                        if state.entry_timestamp is None
                        else f"{state.entry_timestamp:.3f} sec"
                    ),
                    f"  entry_bin_id: {state.entry_bin_id or 'NONE'}",
                    f"  entry_direction_score: {state.entry_direction_score:.3f}",
                    "  entry_trajectory_consistency: "
                    f"{state.entry_trajectory_consistency:.3f}",
                    "  last_visible_centroid: "
                    + (
                        "NONE"
                        if state.last_visible_centroid is None
                        else (
                            f"({state.last_visible_centroid[0]:.1f}, "
                            f"{state.last_visible_centroid[1]:.1f})"
                        )
                    ),
                    f"  verification_result: "
                    f"{state.verification_result if state.placement_confirmed else UNRESOLVED}",
                    "  placement_owner_static_local_id: "
                    f"{state.placement_owner_static_local_id or 'NONE'}",
                    f"  visible_dwell: {dwell:.3f} sec",
                    f"  max_overlap: {max(max_overlap, state.bin_max_overlap):.3f}",
                    f"  settled_speed: {speed:.3f} px/sec",
                    "  disappeared_inside_candidate: "
                    f"{'yes' if state.disappeared_inside_candidate else 'no'}",
                    "  provisional: "
                    f"{'yes' if state.provisional_placement_seen else 'no'}",
                    "  reappeared_outside: "
                    f"{'yes' if state.reappeared_outside_candidate else 'no'}",
                    "  confirmation_method: "
                    f"{state.placement_confirmation_method or 'NONE'}",
                    "  factory_placement_completed: "
                    f"{'yes' if state.factory_placement_completed else 'no'}",
                    f"  reason: {reason}",
                ]
            )
        bin3_failures = [
            item
            for items in self.candidate_history.values()
            for item in items
            if item.bin_id == "BIN_3" and not item.confirmed
        ]
        lines.extend(["", "BIN_3 candidate failure diagnostics", "-----------------------------------"])
        if not bin3_failures:
            lines.append("None.")
        else:
            for item in bin3_failures:
                lines.append(
                    f"G{item.global_id} BIN_3: start_frame={item.start_frame}, "
                    f"start_time={item.start_time:.3f}s, "
                    f"last_valid_time={item.last_valid_time:.3f}s, "
                    f"dwell={item.dwell_seconds:.3f}s, "
                    f"max_overlap={item.max_overlap:.3f}, "
                    f"speed={item.current_speed:.3f}px/s, "
                    f"reason={item.reason or 'UNRESOLVED'}"
                )
        lines.extend(
            [
                "",
                "POSSIBLE_DETECTOR_MISS_AT_BIN_ENTRY",
                "-----------------------------------",
            ]
        )
        if not self.possible_detector_misses:
            lines.append("None.")
        else:
            for item in self.possible_detector_misses:
                lines.append(
                    f"G{item['global_id']} -> {item['bin_id']}: "
                    f"frame={item['frame']}, "
                    f"last_distance={item['last_distance_px']:.1f}px, "
                    f"approach_progress={item['approach_progress_px']:.1f}px, "
                    f"max_overlap={item['max_overlap']:.3f}, "
                    f"saved_frames={item.get('saved_directory', 'NONE')}"
                )
        lines.extend(
            [
                "",
                "Debug statistics (non-business)",
                "-------------------------------",
                "Physical-bin totals below never affect identity, placement, journey, or verification.",
            ]
        )
        for bin_id in self.configured_bin_ids:
            lines.append(
                f"{bin_id} confirmed placements: "
                f"{self.debug_placement_counts[bin_id]}"
            )
        return lines
