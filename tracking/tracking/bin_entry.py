"""Temporal actual-bin entry confirmation keyed by Global Roll ID."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bin_geometry import BinGeometry, BinOverlap, obb_bin_overlap
from global_identity import GlobalRollState


@dataclass(frozen=True)
class BinFrameStatus:
    global_id: int
    best_bin_id: str
    best_overlap: float
    second_bin_id: str
    second_overlap: float
    ambiguous: bool


class BinEntryManager:
    """Confirm a physical bin after stable OBB overlap across frames."""

    def __init__(
        self,
        *,
        geometry: BinGeometry | None,
        camera_id: str | None,
        overlap_threshold: float,
        confirmation_frames: int,
        ambiguity_margin: float,
        allow_reassignment: bool,
        fps: float,
        event_writer: csv.DictWriter,
        debug: bool,
    ) -> None:
        self.geometry = geometry
        self.camera_id = camera_id
        self.overlap_threshold = overlap_threshold
        self.confirmation_frames = confirmation_frames
        self.ambiguity_margin = ambiguity_margin
        self.allow_reassignment = allow_reassignment
        self.fps = fps
        self.event_writer = event_writer
        self.debug = debug
        self.ambiguous_global_ids: set[int] = set()
        self._currently_ambiguous: set[int] = set()
        self.event_counts: dict[str, int] = {
            "BIN_CANDIDATE_STARTED": 0,
            "BIN_CANDIDATE_RESET": 0,
            "BIN_ENTRY_CONFIRMED": 0,
            "BIN_ENTRY_AMBIGUOUS": 0,
        }

    @property
    def enabled(self) -> bool:
        return bool(
            self.geometry is not None
            and self.camera_id
            and self.geometry.polygons_for_camera(self.camera_id)
        )

    def process_observation(
        self,
        global_state: GlobalRollState,
        local_id: int,
        obb: Sequence[Sequence[float]] | np.ndarray,
        frame_index: int,
    ) -> BinFrameStatus | None:
        if not self.enabled or not global_state.journey_started:
            return None

        overlaps = obb_bin_overlap(
            self.geometry, self.camera_id or "", obb
        )
        status = self._frame_status(global_state.global_id, overlaps)
        # A journey can be created while the roll still overlaps its source
        # bin. That starting occupancy is not an entry. Arm confirmation only
        # after the roll has cleared every qualifying bin polygon once.
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
                        f"[frame {frame_index}] G{global_state.global_id} "
                        f"starts in {global_state.initial_bin_id}; waiting for exit"
                    )
            else:
                global_state.bin_entry_armed = True
        if not global_state.bin_entry_armed:
            still_in_any_bin = any(
                overlap.overlap_ratio >= self.overlap_threshold
                for overlap in overlaps
            )
            if still_in_any_bin:
                return status
            global_state.bin_entry_armed = True
            if self.debug:
                print(
                    f"[frame {frame_index}] G{global_state.global_id} "
                    f"cleared starting bin {global_state.initial_bin_id}; "
                    "actual-bin entry armed"
                )
            return status

        global_state.bin_max_overlap = max(
            global_state.bin_max_overlap, status.best_overlap
        )

        if global_state.bin_confirmed and not self.allow_reassignment:
            return status
        if global_state.bin_confirmed and self.allow_reassignment:
            if (
                not status.ambiguous
                and status.best_bin_id
                and status.best_bin_id != global_state.actual_bin_id
                and status.best_overlap >= self.overlap_threshold
            ):
                previous_bin = global_state.actual_bin_id
                global_state.bin_confirmed = False
                global_state.actual_bin_id = ""
                global_state.bin_entry_frame = None
                self._reset_candidate(
                    global_state,
                    local_id,
                    frame_index,
                    previous_bin,
                    status.best_overlap,
                )
            else:
                return status

        if status.ambiguous:
            self.ambiguous_global_ids.add(global_state.global_id)
            if global_state.global_id not in self._currently_ambiguous:
                self._currently_ambiguous.add(global_state.global_id)
                self._log_event(
                    frame_index,
                    global_state.global_id,
                    local_id,
                    status.best_bin_id,
                    status.best_overlap,
                    global_state.candidate_bin_frames,
                    "BIN_ENTRY_AMBIGUOUS",
                )
                if self.debug:
                    print(
                        f"[frame {frame_index}] G{global_state.global_id} "
                        f"ambiguous bin overlap: {status.best_bin_id} "
                        f"{status.best_overlap:.3f} vs {status.second_bin_id} "
                        f"{status.second_overlap:.3f}"
                    )
            return status
        self._currently_ambiguous.discard(global_state.global_id)

        if (
            not status.best_bin_id
            or status.best_overlap < self.overlap_threshold
        ):
            if global_state.candidate_bin_id:
                self._reset_candidate(
                    global_state,
                    local_id,
                    frame_index,
                    global_state.candidate_bin_id,
                    status.best_overlap,
                )
            return status

        if global_state.candidate_bin_id != status.best_bin_id:
            if global_state.candidate_bin_id:
                self._reset_candidate(
                    global_state,
                    local_id,
                    frame_index,
                    global_state.candidate_bin_id,
                    status.best_overlap,
                )
            global_state.candidate_bin_id = status.best_bin_id
            global_state.candidate_bin_frames = 1
            global_state.candidate_bin_max_overlap = status.best_overlap
            global_state.candidate_bin_last_overlap = status.best_overlap
            self._log_event(
                frame_index,
                global_state.global_id,
                local_id,
                status.best_bin_id,
                status.best_overlap,
                1,
                "BIN_CANDIDATE_STARTED",
            )
            if self.debug:
                print(
                    f"[frame {frame_index}] G{global_state.global_id} "
                    f"candidate {status.best_bin_id} 1/"
                    f"{self.confirmation_frames} overlap={status.best_overlap:.3f}"
                )
        else:
            global_state.candidate_bin_frames += 1
            global_state.candidate_bin_max_overlap = max(
                global_state.candidate_bin_max_overlap,
                status.best_overlap,
            )
            global_state.candidate_bin_last_overlap = status.best_overlap

        if global_state.candidate_bin_frames >= self.confirmation_frames:
            global_state.actual_bin_id = global_state.candidate_bin_id
            global_state.bin_confirmed = True
            global_state.bin_entry_frame = frame_index
            self._log_event(
                frame_index,
                global_state.global_id,
                local_id,
                global_state.actual_bin_id,
                status.best_overlap,
                global_state.candidate_bin_frames,
                "BIN_ENTRY_CONFIRMED",
            )
            if self.debug:
                print(
                    f"[frame {frame_index}] G{global_state.global_id} "
                    f"ACTUAL BIN confirmed: {global_state.actual_bin_id} "
                    f"overlap={status.best_overlap:.3f}"
                )
        return status

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
            and best.overlap_ratio - second.overlap_ratio
            < self.ambiguity_margin
        )
        return BinFrameStatus(
            global_id=global_id,
            best_bin_id="" if best is None else best.physical_bin_id,
            best_overlap=0.0 if best is None else best.overlap_ratio,
            second_bin_id="" if second is None else second.physical_bin_id,
            second_overlap=0.0 if second is None else second.overlap_ratio,
            ambiguous=ambiguous,
        )

    def _reset_candidate(
        self,
        global_state: GlobalRollState,
        local_id: int,
        frame_index: int,
        physical_bin_id: str,
        overlap_ratio: float,
    ) -> None:
        previous_frames = global_state.candidate_bin_frames
        if global_state.candidate_bin_id:
            self._log_event(
                frame_index,
                global_state.global_id,
                local_id,
                physical_bin_id,
                overlap_ratio,
                previous_frames,
                "BIN_CANDIDATE_RESET",
            )
        global_state.candidate_bin_id = ""
        global_state.candidate_bin_frames = 0
        global_state.candidate_bin_max_overlap = 0.0
        global_state.candidate_bin_last_overlap = 0.0

    def _log_event(
        self,
        frame_index: int,
        global_id: int,
        local_id: int,
        physical_bin_id: str,
        overlap_ratio: float,
        confirmation_frames: int,
        event: str,
    ) -> None:
        self.event_writer.writerow(
            {
                "frame": frame_index,
                "timestamp_seconds": f"{frame_index / self.fps:.3f}",
                "global_id": global_id,
                "local_id": local_id,
                "physical_bin_id": physical_bin_id,
                "overlap_ratio": f"{overlap_ratio:.6f}",
                "confirmation_frames": confirmation_frames,
                "event": event,
            }
        )
        self.event_counts[event] += 1

    def summary_lines(
        self, global_rolls: Sequence[GlobalRollState]
    ) -> list[str]:
        journeys = [state for state in global_rolls if state.journey_started]
        confirmed = [state for state in journeys if state.bin_confirmed]
        return [
            "Actual physical-bin entry summary",
            "---------------------------------",
            f"Bin entry enabled: {'YES' if self.enabled else 'NO'}",
            f"Bin overlap threshold: {self.overlap_threshold:.3f}",
            f"Bin confirmation frames: {self.confirmation_frames}",
            f"Bin ambiguity margin: {self.ambiguity_margin:.3f}",
            f"Global journeys detected: {len(journeys)}",
            f"Journeys with confirmed bin: {len(confirmed)}",
            f"Journeys with no confirmed bin: {len(journeys) - len(confirmed)}",
            f"Ambiguous bin entries: {len(self.ambiguous_global_ids)}",
            "",
            "Per-journey actual bin",
            "----------------------",
            *[
                line
                for state in sorted(journeys, key=lambda value: value.global_id)
                for line in (
                    f"G{state.global_id}:",
                    f"  initial_bin: {state.initial_bin_id or 'NONE'}",
                    f"  actual_bin: {state.actual_bin_id or 'NONE'}",
                    "  confirmed_frame: "
                    + (
                        "NONE"
                        if state.bin_entry_frame is None
                        else str(state.bin_entry_frame)
                    ),
                    f"  max_overlap: {state.bin_max_overlap:.3f}",
                )
            ],
        ]
