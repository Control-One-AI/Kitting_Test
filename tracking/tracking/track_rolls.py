#!/usr/bin/env python3
"""Track class-1 sheet rolls with local IDs in a single camera video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Mapping, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from bin_placement_dwell import BinEntryManager, BinFrameStatus
from bin_geometry import BIN_COLORS, BinGeometry, load_bin_config

from global_identity import (
    ACTIVE,
    COMPLETED,
    CORRECT,
    ENTERING_BIN,
    IdentityObservation,
    GlobalIdentityManager,
    GlobalRollState,
    LEAVING_BIN,
    LOST,
    NEUTRAL,
    OCCLUDED,
    PLACED,
    PLACEMENT_PENDING,
    VISIBLE_STATES,
    WRONG_BIN,
    box_size_from_corners,
)


ROLL_CLASS_ID = 1
SCRIPT_DIR = Path(__file__).resolve().parent
Point = tuple[float, float]
HistoryPoint = tuple[int, float, float]
UNASSIGNED_STATIC = "UNASSIGNED_STATIC"
PENDING_PICKUP = "PENDING_PICKUP"
ASSIGNED_TO_GLOBAL = "ASSIGNED_TO_GLOBAL"
BIN_FLOW_MIN_PROGRESS_PX = 10.0
BIN_FLOW_NEAR_DISTANCE_PX = 80.0


class DiscardingCsvWriter:
    """DictWriter-compatible sink used by summary-only runs."""

    def writeheader(self) -> None:
        return None

    def writerow(self, row: Mapping[str, object]) -> None:
        del row


@dataclass
class TrackState:
    """Application state layered on top of an Ultralytics local track ID."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    last_centroid: Point
    centroid_history: Deque[HistoryPoint]
    confidence_history: Deque[float]
    frames_seen: int = 0
    total_displacement: float = 0.0
    is_active: bool = False
    missed_frames: int = 0
    active_until_frame: int = -1
    lost: bool = False
    became_active: bool = False
    first_confidence: float | None = None
    last_confidence: float | None = None
    maximum_recovered_gap: int = 0
    assignment_state: str = UNASSIGNED_STATIC
    ever_assigned_global: bool = False
    pending_since_frame: int | None = None
    pending_active_frames: int = 0
    pending_direction_ambiguous_logged: bool = False
    active_without_global_since: int | None = None
    activation_grace_start_frame: int | None = None
    in_zone_a: bool = False
    near_zone_a: bool = False
    in_tracking_zone: bool = False
    in_zone_b: bool = False
    near_zone_b: bool = False
    placement_owner_global_id: int | None = None
    related_bin_id: str = ""
    bin_motion_relation: str = NEUTRAL

    def record_detection(
        self,
        frame_index: int,
        centroid: Point,
        confidence: float,
        motion_window: int,
        motion_threshold: float,
        active_hold_frames: int,
    ) -> str | None:
        """Update this track and return an activity transition, if any."""

        if self.frames_seen:
            self.total_displacement += math.dist(self.last_centroid, centroid)
        self.last_centroid = centroid
        self.last_seen_frame = frame_index
        self.frames_seen += 1
        self.missed_frames = 0
        self.lost = False
        if self.first_confidence is None:
            self.first_confidence = confidence
        self.last_confidence = confidence
        self.centroid_history.append((frame_index, centroid[0], centroid[1]))
        self.confidence_history.append(confidence)

        moved = self._displacement_over_window(frame_index, motion_window)
        movement_detected = moved is not None and moved > motion_threshold
        if movement_detected:
            self.active_until_frame = max(
                self.active_until_frame, frame_index + active_hold_frames
            )
            if not self.is_active:
                self.is_active = True
                self.became_active = True
                return "BECAME_ACTIVE"
        elif self.is_active and frame_index >= self.active_until_frame:
            self.is_active = False
            return "BECAME_STATIC"
        return None

    def _displacement_over_window(
        self, frame_index: int, motion_window: int
    ) -> float | None:
        target_frame = frame_index - motion_window
        old_point: HistoryPoint | None = None
        for point in self.centroid_history:
            if point[0] <= target_frame:
                old_point = point
            else:
                break
        if old_point is None:
            return None
        return math.dist((old_point[1], old_point[2]), self.last_centroid)


@dataclass(frozen=True)
class TrackObservation:
    track_id: int | None
    class_id: int
    confidence: float
    corners: np.ndarray
    centroid: Point


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track class-1 rolls with local OBB track IDs in one video."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument(
        "--codec",
        default="avc1",
        help="FourCC output codec. Default avc1 writes QuickTime-compatible H.264.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device such as auto, cpu, mps, 0, or 0,1.",
    )
    parser.add_argument("--motion-window", type=positive_int, default=8)
    parser.add_argument("--motion-threshold", type=float, default=20.0)
    parser.add_argument("--active-hold-frames", type=non_negative_int, default=15)
    parser.add_argument("--trajectory-length", type=positive_int, default=30)
    parser.add_argument("--max-missed-frames", type=non_negative_int, default=90)
    parser.add_argument(
        "--global-occlusion-frames",
        type=positive_int,
        default=90,
        help="Frames to retain an invisible global roll before GLOBAL_LOST.",
    )
    parser.add_argument(
        "--active-occlusion-frames",
        type=positive_int,
        default=90,
        help="Long-gap reassociation window for rolls active before occlusion.",
    )
    parser.add_argument(
        "--active-reassoc-max-distance",
        type=float,
        default=300.0,
        help="Maximum pixel distance for long ACTIVE-occlusion reassociation.",
    )
    parser.add_argument(
        "--reassoc-activation-grace-frames",
        type=non_negative_int,
        default=10,
        help="Frames after local-track appearance in which ACTIVE may reassociate.",
    )
    parser.add_argument(
        "--journey-activation-frames",
        type=positive_int,
        default=5,
        help="ACTIVE observations required for sustained-motion promotion.",
    )
    parser.add_argument(
        "--journey-min-displacement",
        type=float,
        default=20.0,
        help="Minimum net centroid displacement for journey promotion.",
    )
    parser.add_argument(
        "--journey-motion-consistency",
        type=float,
        default=0.6,
        help="Minimum net displacement / path length for coherent motion.",
    )
    parser.add_argument(
        "--journey-direction-window",
        type=positive_int,
        default=10,
        help="Recent trajectory observations used to infer A/B direction.",
    )
    parser.add_argument(
        "--pending-max-frames",
        type=positive_int,
        default=30,
        help="Maximum frames an ACTIVE unassigned local may remain pending.",
    )
    parser.add_argument(
        "--reassoc-max-distance",
        type=float,
        default=0.18,
        help="Maximum reassociation distance as a fraction of frame diagonal.",
    )
    parser.add_argument(
        "--reassoc-size-tolerance",
        type=float,
        default=0.60,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reassoc-min-score",
        type=float,
        default=0.65,
        help="Minimum geometry-only reassociation score in [0, 1].",
    )
    parser.add_argument(
        "--short-gap-frames",
        type=positive_int,
        default=30,
        help="Missing-frame limit for prioritized short-gap reassociation.",
    )
    parser.add_argument(
        "--short-gap-max-distance",
        type=float,
        default=150.0,
        help="Base pixel distance for the velocity-aware short-gap fast path.",
    )
    parser.add_argument(
        "--reassoc-motion-tolerance",
        type=float,
        default=1.5,
        help="Multiplier applied to recent speed and missing-frame count.",
    )
    parser.add_argument(
        "--reassoc-max-dynamic-distance",
        type=float,
        default=300.0,
        help="Maximum velocity-aware short-gap distance in pixels.",
    )
    parser.add_argument(
        "--reassoc-size-ratio-min",
        type=float,
        default=0.25,
        help="Soft lower scale for the gradual OBB area-ratio score.",
    )
    parser.add_argument(
        "--reassoc-size-ratio-max",
        type=float,
        default=4.0,
        help="Soft upper scale for the gradual OBB area-ratio score.",
    )
    parser.add_argument(
        "--reassoc-hard-size-ratio-min",
        type=float,
        default=0.10,
        help="Emergency minimum new/old OBB area ratio.",
    )
    parser.add_argument(
        "--reassoc-hard-size-ratio-max",
        type=float,
        default=8.0,
        help="Emergency maximum new/old OBB area ratio.",
    )
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--roi-config", type=Path)
    parser.add_argument(
        "--zones",
        "--zones-config",
        dest="zones_config",
        type=Path,
        default=SCRIPT_DIR / "configs" / "cam2_zones.json",
        help="JSON polygons for zone A, tracking, and zone B lifecycle zones.",
    )
    parser.add_argument(
        "--endpoint-margin-px",
        type=float,
        default=50.0,
        help="Allowed centroid distance outside either endpoint polygon.",
    )
    parser.add_argument(
        "--handoff-hold-frames",
        type=positive_int,
        default=10,
        help="Consecutive destination-zone frames before GLOBAL_COMPLETED.",
    )
    parser.add_argument(
        "--destination-proximity-px",
        type=float,
        default=60.0,
        help=(
            "B_TO_A exit fallback distance from the destination polygon. "
            "Completion still requires subsequent disappearance."
        ),
    )
    parser.add_argument(
        "--draw-zones",
        action="store_true",
        help="Draw configured zone A, tracking, and zone B polygons.",
    )
    parser.add_argument(
        "--bins-config",
        type=Path,
        default=SCRIPT_DIR / "configs" / "bins.json",
        help="Global physical-bin config containing per-camera polygons.",
    )
    parser.add_argument(
        "--camera-id",
        help="Camera key in bins.json, e.g. cam0 or cam1.",
    )
    parser.add_argument(
        "--draw-bins",
        action="store_true",
        help="Draw configured physical bins for --camera-id (visual only).",
    )
    parser.add_argument(
        "--bin-focused",
        action="store_true",
        help=(
            "Show and report physical-bin state instead of zone A/B direction "
            "and destination diagnostics. Requires --camera-id."
        ),
    )
    parser.add_argument(
        "--bin-overlap-threshold",
        type=float,
        default=0.35,
        help="Minimum roll-OBB overlap ratio for a bin candidate.",
    )
    parser.add_argument(
        "--bin-confirm-frames",
        type=positive_int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bin-visible-dwell-seconds",
        "--bin-dwell-seconds",
        dest="bin_visible_dwell_seconds",
        type=float,
        default=0.60,
        help="Visible residence time required for settled placement confirmation.",
    )
    parser.add_argument(
        "--bin-gap-tolerance-seconds",
        type=float,
        default=0.30,
        help="Maximum short overlap/detection dropout retained during a dwell.",
    )
    parser.add_argument(
        "--bin-occlusion-confirm-grace",
        "--bin-placement-grace-seconds",
        dest="bin_placement_grace_seconds",
        type=float,
        default=1.0,
        help=(
            "Grace period before confirming a credible mouth-entry followed "
            "by occlusion."
        ),
    )
    parser.add_argument(
        "--bin-settle-speed-px-per-sec",
        type=float,
        default=70.0,
        help="Maximum median centroid speed allowed when placement confirms.",
    )
    parser.add_argument(
        "--bin-entry-miss-distance-px",
        type=float,
        default=80.0,
        help="Near-bin distance used only for detector-miss diagnostics.",
    )
    parser.add_argument(
        "--placement-static-handoff-seconds",
        type=float,
        default=2.0,
        help="Maximum active-to-static placement ownership time gap.",
    )
    parser.add_argument(
        "--placement-static-handoff-distance-px",
        type=float,
        default=120.0,
        help="Maximum active-to-static placement ownership centroid distance.",
    )
    parser.add_argument(
        "--targeted-annotation-frames",
        type=positive_int,
        default=15,
        help="Pre-loss frames saved for each detector-miss-at-bin-entry case.",
    )
    parser.add_argument(
        "--bin-ambiguity-margin",
        type=float,
        default=0.10,
        help="Wait when the best two qualifying bin overlaps differ by less.",
    )
    parser.add_argument(
        "--allow-bin-reassignment",
        action="store_true",
        help="Debug-only: allow a confirmed actual bin to be reconsidered.",
    )
    parser.add_argument(
        "--save-zone-debug-frames",
        action="store_true",
        help="Save lifecycle transition frames under outputs/zone_debug/.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help=(
            "Show Local IDs, lifecycle/candidate details, trajectories, and "
            "internal counters. Default is the clean production overlay."
        ),
    )
    parser.add_argument(
        "--debug-verbose",
        action="store_true",
        help="Print per-frame destination-distance diagnostics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output artifacts.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only the annotated video and summary TXT; do not create CSVs.",
    )
    parser.add_argument(
        "--bin-events-only",
        action="store_true",
        help=(
            "Write video, summary TXT, and focused bin-event CSV only; suppress "
            "detection, local-event, and global-event CSVs."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be in [0, 1]")
    if args.motion_threshold < 0:
        parser.error("--motion-threshold must be zero or greater")
    if args.endpoint_margin_px < 0:
        parser.error("--endpoint-margin-px must be zero or greater")
    if args.destination_proximity_px < 0:
        parser.error("--destination-proximity-px must be zero or greater")
    if args.draw_bins and not args.camera_id:
        parser.error("--draw-bins requires --camera-id")
    if args.bin_focused and not args.camera_id:
        parser.error("--bin-focused requires --camera-id")
    if args.bin_focused and args.draw_zones:
        parser.error("--bin-focused cannot be combined with --draw-zones")
    if not 0.0 <= args.bin_overlap_threshold <= 1.0:
        parser.error("--bin-overlap-threshold must be in [0, 1]")
    if args.bin_visible_dwell_seconds <= 0:
        parser.error("--bin-visible-dwell-seconds must be greater than zero")
    if args.bin_gap_tolerance_seconds < 0:
        parser.error("--bin-gap-tolerance-seconds must be zero or greater")
    if args.bin_placement_grace_seconds <= 0:
        parser.error("--bin-occlusion-confirm-grace must be greater than zero")
    if args.bin_settle_speed_px_per_sec < 0:
        parser.error("--bin-settle-speed-px-per-sec must be zero or greater")
    if args.bin_entry_miss_distance_px < 0:
        parser.error("--bin-entry-miss-distance-px must be zero or greater")
    if args.placement_static_handoff_seconds <= 0:
        parser.error("--placement-static-handoff-seconds must be greater than zero")
    if args.placement_static_handoff_distance_px <= 0:
        parser.error(
            "--placement-static-handoff-distance-px must be greater than zero"
        )
    if not 0.0 <= args.bin_ambiguity_margin <= 1.0:
        parser.error("--bin-ambiguity-margin must be in [0, 1]")
    if args.allow_bin_reassignment and not args.debug:
        parser.error("--allow-bin-reassignment requires --debug")
    if args.summary_only and args.bin_events_only:
        parser.error("--summary-only and --bin-events-only are mutually exclusive")
    if args.journey_min_displacement < 0:
        parser.error("--journey-min-displacement must be zero or greater")
    if not 0.0 <= args.journey_motion_consistency <= 1.0:
        parser.error("--journey-motion-consistency must be in [0, 1]")
    if not 0.0 < args.reassoc_max_distance <= 1.0:
        parser.error("--reassoc-max-distance must be in (0, 1]")
    if not 0.0 < args.reassoc_size_tolerance <= 1.0:
        parser.error("--reassoc-size-tolerance must be in (0, 1]")
    if not 0.0 <= args.reassoc_min_score <= 1.0:
        parser.error("--reassoc-min-score must be in [0, 1]")
    if args.short_gap_max_distance <= 0:
        parser.error("--short-gap-max-distance must be greater than zero")
    if args.active_reassoc_max_distance < args.short_gap_max_distance:
        parser.error(
            "--active-reassoc-max-distance must be at least "
            "--short-gap-max-distance"
        )
    if args.reassoc_motion_tolerance < 0:
        parser.error("--reassoc-motion-tolerance must be zero or greater")
    if args.reassoc_max_dynamic_distance < args.short_gap_max_distance:
        parser.error(
            "--reassoc-max-dynamic-distance must be at least "
            "--short-gap-max-distance"
        )
    if args.reassoc_size_ratio_min <= 0:
        parser.error("--reassoc-size-ratio-min must be greater than zero")
    if args.reassoc_size_ratio_max < args.reassoc_size_ratio_min:
        parser.error(
            "--reassoc-size-ratio-max must be at least --reassoc-size-ratio-min"
        )
    if args.reassoc_hard_size_ratio_min <= 0:
        parser.error("--reassoc-hard-size-ratio-min must be greater than zero")
    if args.reassoc_hard_size_ratio_min > args.reassoc_size_ratio_min:
        parser.error(
            "--reassoc-hard-size-ratio-min must not exceed "
            "--reassoc-size-ratio-min"
        )
    if args.reassoc_hard_size_ratio_max < args.reassoc_size_ratio_max:
        parser.error(
            "--reassoc-hard-size-ratio-max must be at least "
            "--reassoc-size-ratio-max"
        )
    if len(args.codec) != 4:
        parser.error("--codec must be exactly four characters, for example avc1")
    return args


def load_roi(path: Path | None) -> np.ndarray | None:
    """Load an optional polygon from {\"tracking_region\": [[x, y], ...]}."""

    if path is None:
        return None
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    points = payload.get("tracking_region")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(
            f"{path}: tracking_region must contain at least three [x, y] points"
        )
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
        raise ValueError(f"{path}: tracking_region must be finite [x, y] pairs")
    return polygon


def load_zones(path: Path | None) -> dict[str, np.ndarray]:
    """Load endpoint polygons and normalize legacy pickup/handoff keys."""

    if path is None:
        print("ERROR: no zone config path was provided")
        return {}
    path = path.expanduser().resolve()
    print(f"Loaded zone config: {path}")
    if not path.exists():
        print(f"ERROR: zone config file does not exist: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    aliases = {
        "zone_a": "pickup_zone",
        "tracking_zone": "tracking_zone",
        "zone_b": "handoff_zone",
    }
    zones: dict[str, np.ndarray] = {}
    for name in ("zone_a", "tracking_zone", "zone_b"):
        legacy_name = aliases[name]
        points = payload.get(name)
        if points is None and legacy_name != name:
            points = payload.get(legacy_name)
            if points not in (None, []):
                print(f"Compatibility: normalized {legacy_name} -> {name}")
        if points is None:
            points = []
        if points in (None, []):
            print(f"ERROR: {name} is empty")
            continue
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(
                f"{path}: {name} must be empty or contain at least three [x, y] points"
            )
        polygon = np.asarray(points, dtype=np.float32)
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or not np.isfinite(polygon).all()
        ):
            raise ValueError(f"{path}: {name} must contain finite [x, y] pairs")
        zones[name] = polygon
        print(f"{name}: {polygon.astype(int).tolist()}")
    return zones


def validate_zones_for_frame(
    zones: dict[str, np.ndarray], frame_width: int, frame_height: int
) -> bool:
    """Validate completeness, area, and coordinates against the video frame."""

    valid = True
    required = ("zone_a", "tracking_zone", "zone_b")
    for name in required:
        polygon = zones.get(name)
        if polygon is None:
            valid = False
            continue
        if len(polygon) < 3:
            print(f"ERROR: {name} has fewer than 3 points")
            valid = False
            continue
        if abs(cv2.contourArea(polygon)) < 1.0:
            print(f"ERROR: {name} has zero or negligible polygon area")
            valid = False
        margin_x = frame_width * 0.10
        margin_y = frame_height * 0.10
        for x, y in polygon:
            point = (float(x), float(y))
            inside = 0 <= x < frame_width and 0 <= y < frame_height
            reasonably_near = (
                -margin_x <= x < frame_width + margin_x
                and -margin_y <= y < frame_height + margin_y
            )
            if inside:
                continue
            if reasonably_near:
                print(f"WARNING: {name} point outside frame: {point}")
            else:
                print(
                    f"ERROR: {name} point unreasonably far outside frame: {point}"
                )
                valid = False
    print(f"Lifecycle enabled: {'YES' if valid else 'NO'}")
    return valid


def centroid_from_corners(corners: np.ndarray) -> Point:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected four OBB corners, received shape {points.shape}")
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def centroid_in_roi(centroid: Point, roi: np.ndarray | None) -> bool:
    if roi is None:
        return True
    return cv2.pointPolygonTest(roi, centroid, False) >= 0


def output_log_paths(output_video: Path) -> tuple[Path, Path]:
    tracking_csv = output_video.with_suffix(".csv")
    stem = output_video.stem
    if stem.endswith("_tracking_botsort"):
        camera_stem = stem.removesuffix("_tracking_botsort")
        event_stem = f"{camera_stem}_events_botsort"
    else:
        event_stem = stem.removesuffix("_tracking") + "_events"
    events_csv = output_video.with_name(event_stem + ".csv")
    return tracking_csv, events_csv


def output_summary_path(output_video: Path) -> Path:
    return output_video.with_name(output_video.stem + "_summary.txt")


def output_global_event_path(output_video: Path) -> Path:
    stem = output_video.stem
    if "_tracking_global_" in stem:
        camera_stem, variant = stem.split("_tracking_global_", maxsplit=1)
        stem = f"{camera_stem}_global_events_{variant}"
    elif stem.endswith("_tracking_global"):
        stem = stem.removesuffix("_tracking_global") + "_global_events"
    else:
        stem += "_global_events"
    return output_video.with_name(stem + ".csv")


def output_bin_event_path(output_video: Path) -> Path:
    return output_video.with_name(output_video.stem + "_bin_events.csv")


def tracker_type_from_config(tracker_config: str) -> str:
    """Return tracker_type from a local YAML, or infer it from a built-in name."""

    config_path = Path(tracker_config).expanduser()
    if config_path.is_file():
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if line.startswith("tracker_type:"):
                return line.split(":", maxsplit=1)[1].strip()
    name = config_path.stem.casefold()
    if "botsort" in name:
        return "botsort"
    if "bytetrack" in name:
        return "bytetrack"
    return "unknown"


def ensure_inside_tracking(path: Path, description: str) -> None:
    try:
        path.resolve().relative_to(SCRIPT_DIR)
    except ValueError as exc:
        raise ValueError(
            f"{description} must be inside {SCRIPT_DIR}; received {path.resolve()}"
        ) from exc


class RollTracker:
    """Single-camera local OBB tracker plus application motion state."""

    def __init__(
        self,
        args: argparse.Namespace,
        roi: np.ndarray | None,
        zones: dict[str, np.ndarray],
        bin_polygons: Mapping[str, np.ndarray],
        bin_geometry: BinGeometry | None,
        lifecycle_enabled: bool,
        tracking_writer: csv.DictWriter,
        event_writer: csv.DictWriter,
        global_event_writer: csv.DictWriter,
        bin_event_writer: csv.DictWriter,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.args = args
        self.roi = roi
        self.zones = zones
        self.bin_polygons = bin_polygons
        self.bin_geometry = bin_geometry
        self.lifecycle_enabled = lifecycle_enabled
        self.tracking_writer = tracking_writer
        self.event_writer = event_writer
        self.fps = fps
        self.model = YOLO(str(args.model))
        self.tracks: dict[int, TrackState] = {}
        self.unique_track_ids: set[int] = set()
        self.tracks_that_became_active: set[int] = set()
        self.track_created_event_count = 0
        self.lost_event_count = 0
        self.maximum_simultaneous_tracks = 0
        self.pending_tracks_created = 0
        self.pending_tracks_promoted = 0
        self.pending_tracks_cancelled = 0
        self.pending_direction_ambiguous = 0
        self.endpoint_global_creations = 0
        self.sustained_global_creations = 0
        self.longest_active_without_global = 0
        self.tracker_type = tracker_type_from_config(args.tracker)
        self.bin_frame_statuses: dict[int, BinFrameStatus] = {}
        self.placement_owner_statuses: dict[int, BinFrameStatus] = {}
        self.frame_buffer: Deque[tuple[int, np.ndarray]] = deque(
            maxlen=args.targeted_annotation_frames
        )
        self.saved_detector_miss_count = 0
        self.identity_manager = GlobalIdentityManager(
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            occlusion_frames=args.global_occlusion_frames,
            active_occlusion_frames=args.active_occlusion_frames,
            active_reassoc_max_distance=args.active_reassoc_max_distance,
            max_distance_fraction=args.reassoc_max_distance,
            min_score=args.reassoc_min_score,
            short_gap_frames=args.short_gap_frames,
            short_gap_max_distance=args.short_gap_max_distance,
            motion_tolerance=args.reassoc_motion_tolerance,
            max_dynamic_distance=args.reassoc_max_dynamic_distance,
            size_ratio_min=args.reassoc_size_ratio_min,
            size_ratio_max=args.reassoc_size_ratio_max,
            hard_size_ratio_min=args.reassoc_hard_size_ratio_min,
            hard_size_ratio_max=args.reassoc_hard_size_ratio_max,
            zones=zones,
            lifecycle_enabled=lifecycle_enabled,
            handoff_hold_frames=args.handoff_hold_frames,
            endpoint_margin_px=args.endpoint_margin_px,
            destination_proximity_px=args.destination_proximity_px,
            event_writer=global_event_writer,
            debug=args.debug,
            debug_verbose=args.debug_verbose,
        )
        self.bin_entry_manager = BinEntryManager(
            geometry=bin_geometry,
            camera_id=args.camera_id,
            overlap_threshold=args.bin_overlap_threshold,
            visible_dwell_seconds=args.bin_visible_dwell_seconds,
            gap_tolerance_seconds=args.bin_gap_tolerance_seconds,
            placement_grace_seconds=args.bin_placement_grace_seconds,
            settle_speed_px_per_sec=args.bin_settle_speed_px_per_sec,
            detector_miss_distance_px=args.bin_entry_miss_distance_px,
            static_handoff_seconds=args.placement_static_handoff_seconds,
            static_handoff_distance_px=args.placement_static_handoff_distance_px,
            ambiguity_margin=args.bin_ambiguity_margin,
            allow_reassignment=args.allow_bin_reassignment,
            fps=fps,
            event_writer=bin_event_writer,
            debug=args.debug,
        )

    def process_frame(
        self, frame: np.ndarray, frame_index: int
    ) -> tuple[np.ndarray, list[TrackObservation]]:
        self.bin_frame_statuses = {}
        self.placement_owner_statuses = {}
        self.frame_buffer.append((frame_index, frame.copy()))
        kwargs: dict[str, object] = {
            "source": frame,
            "persist": True,
            "classes": [ROLL_CLASS_ID],
            "tracker": self.args.tracker,
            "conf": self.args.conf,
            "iou": self.args.iou,
            "imgsz": self.args.imgsz,
            "verbose": False,
        }
        if self.args.device != "auto":
            kwargs["device"] = self.args.device
        result = self.model.track(**kwargs)[0]
        observations = self._extract_observations(result)
        transitions = self.update_track_states(observations, frame_index)
        identity_observations = [
            IdentityObservation(
                local_id=observation.track_id,
                centroid=observation.centroid,
                box_size=box_size_from_corners(observation.corners),
                centroid_history=tuple(
                    self.tracks[observation.track_id].centroid_history
                ),
                related_bin_id=self.tracks[observation.track_id].related_bin_id,
                bin_motion_relation=self.tracks[
                    observation.track_id
                ].bin_motion_relation,
                is_active=self.tracks[observation.track_id].is_active,
                became_active=transitions.get(observation.track_id)
                == "BECAME_ACTIVE",
            )
            for observation in observations
            if observation.track_id is not None
        ]
        eligible_new_local_ids: set[int] = set()
        create_global_local_ids: set[int] = set()
        create_global_origins: dict[int, str] = {}
        create_global_reasons: dict[int, str] = {}
        pending_metrics: dict[int, tuple[int, float, float]] = {}
        for observation in observations:
            local_id = observation.track_id
            if local_id is None or local_id in self.identity_manager.local_to_global:
                continue
            state = self.tracks[local_id]
            transition = transitions.get(local_id)
            endpoint_origin = self._endpoint_origin(state)
            if transition == "BECAME_ACTIVE":
                state.active_without_global_since = frame_index
                if endpoint_origin is not None:
                    eligible_new_local_ids.add(local_id)
                    create_global_local_ids.add(local_id)
                    create_global_origins[local_id] = endpoint_origin
                    create_global_reasons[local_id] = (
                        "bin_pickup"
                        if state.bin_motion_relation == LEAVING_BIN
                        else "endpoint"
                    )
                else:
                    self._start_pending(state, frame_index)
            elif transition == "BECAME_STATIC" and state.assignment_state == PENDING_PICKUP:
                self._cancel_pending(state, frame_index, "became_static")

            if state.is_active and state.active_without_global_since is not None:
                self.longest_active_without_global = max(
                    self.longest_active_without_global,
                    frame_index - state.active_without_global_since + 1,
                )

            grace_start = (
                state.activation_grace_start_frame
                if state.activation_grace_start_frame is not None
                else state.first_seen_frame
            )
            age_frames = frame_index - grace_start
            if (
                state.is_active
                and age_frames <= self.args.reassoc_activation_grace_frames
            ):
                eligible_new_local_ids.add(local_id)

            if state.assignment_state != PENDING_PICKUP or not state.is_active:
                continue
            if transition != "BECAME_ACTIVE":
                state.pending_active_frames += 1
            eligible_new_local_ids.add(local_id)
            metrics = self._pending_motion_metrics(state)
            pending_metrics[local_id] = metrics
            active_frames, net_displacement, consistency = metrics
            strong_motion = (
                active_frames >= self.args.journey_activation_frames
                and net_displacement >= self.args.journey_min_displacement
                and consistency >= self.args.journey_motion_consistency
            )
            pending_start = (
                state.pending_since_frame
                if state.pending_since_frame is not None
                else frame_index
            )
            pending_age = frame_index - pending_start + 1
            inferred_origin = (
                self._infer_journey_origin(state, allow_best=False)
                if strong_motion
                else None
            )
            if strong_motion and inferred_origin is not None:
                eligible_new_local_ids.add(local_id)
                create_global_local_ids.add(local_id)
                create_global_origins[local_id] = inferred_origin
                create_global_reasons[local_id] = "sustained_motion"
            elif strong_motion and not state.pending_direction_ambiguous_logged:
                state.pending_direction_ambiguous_logged = True
                self.pending_direction_ambiguous += 1
                self.log_event(
                    frame_index,
                    local_id,
                    "PENDING_DIRECTION_AMBIGUOUS",
                    active_frames=active_frames,
                    net_displacement=net_displacement,
                    motion_consistency=consistency,
                )

            if pending_age >= self.args.pending_max_frames and local_id not in create_global_local_ids:
                best_origin = (
                    self._infer_journey_origin(state, allow_best=True)
                    if strong_motion
                    else None
                )
                if strong_motion and best_origin is not None:
                    create_global_local_ids.add(local_id)
                    create_global_origins[local_id] = best_origin
                    create_global_reasons[local_id] = "sustained_motion"
                else:
                    self._cancel_pending(state, frame_index, "pending_timeout")
                    eligible_new_local_ids.discard(local_id)

        global_ids = self.identity_manager.process_frame(
            identity_observations,
            frame_index,
            eligible_new_local_ids=eligible_new_local_ids,
            create_global_local_ids=create_global_local_ids,
            create_global_origins=create_global_origins,
            create_global_reasons=create_global_reasons,
        )
        bin_processed_globals: set[int] = set()
        unassigned_observations: list[TrackObservation] = []
        for observation in observations:
            if observation.track_id is None:
                continue
            global_id = global_ids.get(observation.track_id)
            state = self.tracks[observation.track_id]
            if global_id is None:
                if not state.is_active:
                    state.assignment_state = UNASSIGNED_STATIC
                    unassigned_observations.append(observation)
                self.log_detection(frame_index, observation, state, None)
                continue
            was_pending = state.assignment_state == PENDING_PICKUP
            global_state = self.identity_manager.global_rolls[global_id]
            if was_pending:
                active_frames, net_displacement, consistency = pending_metrics.get(
                    observation.track_id,
                    self._pending_motion_metrics(state),
                )
                self.pending_tracks_promoted += 1
                self.log_event(
                    frame_index,
                    observation.track_id,
                    "PENDING_PROMOTED",
                    global_id=global_id,
                    active_frames=active_frames,
                    net_displacement=net_displacement,
                    motion_consistency=consistency,
                    journey_direction=global_state.journey_direction,
                    origin_zone=global_state.origin_zone,
                    destination_zone=global_state.destination_zone,
                    activation_reason=create_global_reasons.get(
                        observation.track_id, "reassociation"
                    ),
                )
            if (
                observation.track_id in create_global_local_ids
                and global_state.first_seen_frame == frame_index
            ):
                if global_state.activation_reason == "endpoint":
                    self.endpoint_global_creations += 1
                elif global_state.activation_reason == "sustained_motion":
                    self.sustained_global_creations += 1
            state.assignment_state = ASSIGNED_TO_GLOBAL
            state.ever_assigned_global = True
            state.pending_since_frame = None
            state.pending_active_frames = 0
            state.pending_direction_ambiguous_logged = False
            state.active_without_global_since = None
            self.identity_manager.update_motion_state(
                global_id,
                state.is_active,
                frame_index,
            )
            if global_id not in bin_processed_globals:
                bin_processed_globals.add(global_id)
                bin_status = self.bin_entry_manager.process_observation(
                    global_state,
                    observation.track_id,
                    observation.corners,
                    frame_index,
                )
                if bin_status is not None:
                    self.bin_frame_statuses[global_id] = bin_status
            self.log_detection(frame_index, observation, state, global_id)
        for observation in unassigned_observations:
            if observation.track_id is None:
                continue
            state = self.tracks[observation.track_id]
            ownership = self.bin_entry_manager.process_static_observation(
                list(self.identity_manager.global_rolls.values()),
                observation.track_id,
                observation.corners,
                frame_index,
            )
            if ownership is None:
                if not state.is_active:
                    state.placement_owner_global_id = None
                continue
            owner_global_id, owner_status = ownership
            state.placement_owner_global_id = owner_global_id
            self.placement_owner_statuses[observation.track_id] = owner_status
            self.bin_frame_statuses[owner_global_id] = owner_status
            bin_processed_globals.add(owner_global_id)
        self.bin_entry_manager.finish_frame(
            list(self.identity_manager.global_rolls.values()),
            bin_processed_globals,
            frame_index,
        )
        self._seal_confirmed_placements(frame_index)
        self._save_targeted_annotation_frames()
        annotated = frame.copy()
        self.draw_tracks(annotated, observations, frame_index, global_ids)
        self._save_zone_debug_frames(annotated, frame_index)
        return annotated, observations

    def _seal_confirmed_placements(self, frame_index: int) -> None:
        """Atomically terminate every newly confirmed physical placement."""

        for global_state in self.identity_manager.global_rolls.values():
            if not global_state.placement_confirmed or global_state.state == PLACED:
                continue
            detached_local_ids = self.identity_manager.seal_by_placement(
                global_state.global_id, frame_index
            )
            for local_id in detached_local_ids:
                local_state = self.tracks.get(local_id)
                if local_state is None:
                    continue
                local_state.assignment_state = UNASSIGNED_STATIC
                local_state.active_without_global_since = None

    def assign_expected_bin(
        self,
        global_roll_id: int,
        physical_bin_id: str,
        *,
        frame_index: int | None = None,
    ) -> str:
        """Integration point for future mapping/LED business logic."""

        try:
            global_state = self.identity_manager.global_rolls[global_roll_id]
        except KeyError as exc:
            raise KeyError(f"Unknown Global Roll G{global_roll_id}") from exc
        return self.bin_entry_manager.assign_expected_bin(
            global_state,
            physical_bin_id,
            frame_index=frame_index,
        )

    def _save_targeted_annotation_frames(self) -> None:
        misses = self.bin_entry_manager.possible_detector_misses
        while self.saved_detector_miss_count < len(misses):
            item = misses[self.saved_detector_miss_count]
            directory = (
                self.args.output.parent
                / "targeted_annotations"
                / str(self.args.camera_id or "camera")
                / (
                    f"G{item['global_id']}_{item['bin_id']}_"
                    f"loss_{item['frame']}"
                )
            )
            directory.mkdir(parents=True, exist_ok=True)
            for buffered_frame, image in self.frame_buffer:
                cv2.imwrite(
                    str(directory / f"frame_{buffered_frame:06d}.jpg"),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
            item["saved_directory"] = str(directory)
            self.saved_detector_miss_count += 1

    def _extract_observations(self, result: object) -> list[TrackObservation]:
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return []
        corners = obb.xyxyxyxy.detach().cpu().numpy()
        confidences = obb.conf.detach().cpu().numpy()
        classes = obb.cls.detach().cpu().numpy().astype(int)
        # Ultralytics leaves ``obb.id`` unset when the tracker has not yet
        # confirmed or matched the current detections. This is a normal
        # transient ByteTrack state, not a fatal tracking error.
        track_ids: Sequence[int | None]
        if obb.id is None:
            track_ids = [None] * len(obb)
        else:
            track_ids = obb.id.detach().cpu().numpy().astype(int).tolist()
        observations: list[TrackObservation] = []
        for points, confidence, class_id, track_id in zip(
            corners, confidences, classes, track_ids, strict=True
        ):
            if class_id != ROLL_CLASS_ID:
                continue
            centroid = centroid_from_corners(points)
            if not centroid_in_roi(centroid, self.roi):
                continue
            observations.append(
                TrackObservation(
                    track_id=None if track_id is None else int(track_id),
                    class_id=int(class_id),
                    confidence=float(confidence),
                    corners=points,
                    centroid=centroid,
                )
            )
        return observations

    def update_track_states(
        self,
        observations: Sequence[TrackObservation],
        frame_index: int,
    ) -> dict[int, str | None]:
        seen_ids: set[int] = set()
        transitions: dict[int, str | None] = {}
        history_size = max(
            self.args.trajectory_length, self.args.motion_window + 1
        )
        for observation in observations:
            track_id = observation.track_id
            if track_id is None:
                continue
            seen_ids.add(track_id)
            state = self.tracks.get(track_id)
            if state is None or state.lost:
                state = TrackState(
                    track_id=track_id,
                    first_seen_frame=frame_index,
                    last_seen_frame=frame_index,
                    last_centroid=observation.centroid,
                    centroid_history=deque(maxlen=history_size),
                    confidence_history=deque(maxlen=history_size),
                    activation_grace_start_frame=frame_index,
                )
                self.tracks[track_id] = state
                self.unique_track_ids.add(track_id)
                self.track_created_event_count += 1
                self.log_event(frame_index, track_id, "TRACK_CREATED")
                self.log_event(frame_index, track_id, "LOCAL_UNASSIGNED")

            gap_frames = state.missed_frames
            if (
                gap_frames
                and state.activation_grace_start_frame is not None
                and frame_index - state.activation_grace_start_frame
                <= self.args.reassoc_activation_grace_frames
            ):
                # A just-created local can flicker before BoT-SORT stabilizes.
                # Restart grace only during its initial grace period, never for
                # a mature track that disappears later in the journey.
                state.activation_grace_start_frame = frame_index
            state.maximum_recovered_gap = max(
                state.maximum_recovered_gap, gap_frames
            )
            if gap_frames and self.args.debug:
                print(
                    f"[frame {frame_index}] Track {track_id} recovered after "
                    f"{gap_frames}-frame gap"
                )

            transition = state.record_detection(
                frame_index=frame_index,
                centroid=observation.centroid,
                confidence=observation.confidence,
                motion_window=self.args.motion_window,
                motion_threshold=self.args.motion_threshold,
                active_hold_frames=self.args.active_hold_frames,
            )
            self._update_local_zone_state(state, observation.centroid)
            self._update_local_bin_motion(state)
            transitions[track_id] = transition
            if transition:
                self.log_event(frame_index, track_id, transition)
                if transition == "BECAME_ACTIVE":
                    self.tracks_that_became_active.add(track_id)

        self.maximum_simultaneous_tracks = max(
            self.maximum_simultaneous_tracks, len(seen_ids)
        )
        for track_id, state in self.tracks.items():
            if state.lost or track_id in seen_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames == 1 and self.args.debug:
                print(
                    f"[frame {frame_index}] Track {track_id} temporarily missing "
                    "(1 frame)"
                )
            if state.missed_frames > self.args.max_missed_frames:
                state.is_active = False
                state.lost = True
                self.lost_event_count += 1
                self.log_event(frame_index, track_id, "TRACK_LOST")
            elif state.is_active and frame_index >= state.active_until_frame:
                state.is_active = False
                self.log_event(frame_index, track_id, "BECAME_STATIC")
        return transitions

    def _update_local_bin_motion(self, state: TrackState) -> None:
        """Classify recent motion as entering/leaving one configured bin."""

        state.related_bin_id = ""
        state.bin_motion_relation = NEUTRAL
        history = list(state.centroid_history)
        if not self.bin_polygons or len(history) < 2:
            return
        first = (history[max(0, len(history) - 5)][1], history[max(0, len(history) - 5)][2])
        last = (history[-1][1], history[-1][2])
        candidates: list[tuple[float, str, np.ndarray]] = []
        for bin_id, polygon in self.bin_polygons.items():
            first_distance = max(0.0, -float(cv2.pointPolygonTest(polygon, first, True)))
            last_distance = max(0.0, -float(cv2.pointPolygonTest(polygon, last, True)))
            first_inside = cv2.pointPolygonTest(polygon, first, False) >= 0
            last_inside = cv2.pointPolygonTest(polygon, last, False) >= 0
            if first_inside or last_inside or min(first_distance, last_distance) <= BIN_FLOW_NEAR_DISTANCE_PX:
                candidates.append((min(first_distance, last_distance), bin_id, polygon))
        if not candidates:
            return
        _, bin_id, polygon = min(candidates, key=lambda item: item[0])
        center = tuple(np.asarray(polygon, dtype=np.float32).mean(axis=0))
        first_to_center = math.dist(first, center)
        last_to_center = math.dist(last, center)
        progress = first_to_center - last_to_center
        state.related_bin_id = bin_id
        if progress >= BIN_FLOW_MIN_PROGRESS_PX:
            state.bin_motion_relation = ENTERING_BIN
        elif progress <= -BIN_FLOW_MIN_PROGRESS_PX:
            state.bin_motion_relation = LEAVING_BIN

    def _update_local_zone_state(
        self, state: TrackState, centroid: Point
    ) -> None:
        zone_a = self.zones.get("zone_a")
        tracking = self.zones.get("tracking_zone")
        zone_b = self.zones.get("zone_b")
        if zone_a is not None:
            signed_distance = cv2.pointPolygonTest(zone_a, centroid, True)
            state.in_zone_a = signed_distance >= 0
            state.near_zone_a = (
                signed_distance >= -self.args.endpoint_margin_px
            )
        else:
            state.in_zone_a = False
            state.near_zone_a = False
        state.in_tracking_zone = (
            tracking is not None
            and cv2.pointPolygonTest(tracking, centroid, False) >= 0
        )
        if zone_b is not None:
            signed_distance = cv2.pointPolygonTest(zone_b, centroid, True)
            state.in_zone_b = signed_distance >= 0
            state.near_zone_b = (
                signed_distance >= -self.args.endpoint_margin_px
            )
        else:
            state.in_zone_b = False
            state.near_zone_b = False

    @staticmethod
    def _endpoint_origin(state: TrackState) -> str | None:
        """Return a unique nearby endpoint; overlapping endpoints are ambiguous."""

        near_a = state.in_zone_a or state.near_zone_a
        near_b = state.in_zone_b or state.near_zone_b
        if near_a == near_b:
            return None
        return "A" if near_a else "B"

    def _start_pending(self, state: TrackState, frame_index: int) -> None:
        state.assignment_state = PENDING_PICKUP
        state.pending_since_frame = frame_index
        state.pending_active_frames = 1
        state.pending_direction_ambiguous_logged = False
        self.pending_tracks_created += 1
        self.log_event(frame_index, state.track_id, "PENDING_STARTED")

    def _cancel_pending(
        self, state: TrackState, frame_index: int, reason: str
    ) -> None:
        active_frames, net_displacement, consistency = (
            self._pending_motion_metrics(state)
        )
        self.pending_tracks_cancelled += 1
        self.log_event(
            frame_index,
            state.track_id,
            "PENDING_CANCELLED",
            active_frames=active_frames,
            net_displacement=net_displacement,
            motion_consistency=consistency,
            activation_reason=reason,
        )
        state.assignment_state = UNASSIGNED_STATIC
        state.pending_since_frame = None
        state.pending_active_frames = 0
        state.pending_direction_ambiguous_logged = False
        state.active_without_global_since = None

    def _pending_points(self, state: TrackState) -> list[HistoryPoint]:
        if state.pending_since_frame is None:
            return []
        return [
            point
            for point in state.centroid_history
            if point[0] >= state.pending_since_frame
        ]

    def _pending_motion_metrics(
        self, state: TrackState
    ) -> tuple[int, float, float]:
        points = self._pending_points(state)
        if len(points) < 2:
            return state.pending_active_frames, 0.0, 0.0
        start = (points[0][1], points[0][2])
        end = (points[-1][1], points[-1][2])
        net_displacement = math.dist(start, end)
        path_length = sum(
            math.dist(
                (previous[1], previous[2]),
                (current[1], current[2]),
            )
            for previous, current in zip(points, points[1:])
        )
        consistency = (
            net_displacement / path_length if path_length > 1e-6 else 0.0
        )
        return state.pending_active_frames, net_displacement, consistency

    def _infer_journey_origin(
        self, state: TrackState, *, allow_best: bool
    ) -> str | None:
        """Infer origin from recent travel toward the opposite endpoint."""

        zone_a = self.zones.get("zone_a")
        zone_b = self.zones.get("zone_b")
        points = self._pending_points(state)[-self.args.journey_direction_window :]
        if zone_a is None or zone_b is None or len(points) < 2:
            return None
        before = np.asarray((points[0][1], points[0][2]), dtype=np.float64)
        now = np.asarray((points[-1][1], points[-1][2]), dtype=np.float64)
        movement = now - before
        net = float(np.linalg.norm(movement))
        if net <= 1e-6:
            return None
        center_a = np.mean(zone_a, axis=0, dtype=np.float64)
        center_b = np.mean(zone_b, axis=0, dtype=np.float64)
        toward_a = float(np.linalg.norm(before - center_a) - np.linalg.norm(now - center_a))
        toward_b = float(np.linalg.norm(before - center_b) - np.linalg.norm(now - center_b))
        evidence = toward_b - toward_a
        minimum_evidence = 1.0 if allow_best else max(2.0, 0.20 * net)
        if abs(evidence) < minimum_evidence:
            axis = center_b - center_a
            axis_length = float(np.linalg.norm(axis))
            if axis_length <= 1e-6:
                return None
            evidence = float(np.dot(movement, axis / axis_length))
        if abs(evidence) < minimum_evidence:
            return None
        # Positive evidence means progress toward B, so the origin is A.
        return "A" if evidence > 0 else "B"

    def draw_tracks(
        self,
        frame: np.ndarray,
        observations: Sequence[TrackObservation],
        frame_index: int,
        global_ids: dict[int, int],
    ) -> None:
        if self.args.draw_zones:
            self._draw_zones(frame)
        if self.args.draw_bins:
            self._draw_bins(frame)
        if self.roi is not None and self.args.debug_overlay:
            cv2.polylines(
                frame,
                [np.rint(self.roi).astype(np.int32)],
                True,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        operator_alerts: list[str] = []
        for observation in observations:
            if observation.track_id is None:
                color = (150, 150, 150)
                polygon = np.rint(observation.corners).astype(np.int32)
                cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
                center = tuple(np.rint(observation.centroid).astype(int))
                cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
                label = (
                    f"ROLL | {observation.confidence:.2f} | UNTRACKED"
                    if self.args.debug_overlay
                    else "ROLL --- | Status: TRACKING"
                )
                anchor_x = max(0, int(polygon[:, 0].min()))
                anchor_y = max(24, int(polygon[:, 1].min()) - 8)
                self._draw_label(frame, label, (anchor_x, anchor_y), color)
                continue

            state = self.tracks[observation.track_id]
            global_id = global_ids.get(observation.track_id)
            global_state = (
                self.identity_manager.global_rolls.get(global_id)
                if global_id is not None
                else None
            )
            if self.args.debug_overlay:
                if global_state is not None and global_state.state in {PLACED, COMPLETED}:
                    color = (170, 170, 170)
                else:
                    color = (40, 220, 40) if state.is_active else (255, 170, 40)
            else:
                owner_id = state.placement_owner_global_id
                operator_state = global_state or (
                    self.identity_manager.global_rolls.get(owner_id)
                    if owner_id is not None
                    else None
                )
                operator_status = self._production_status(operator_state)
                color = self._production_color(operator_status, state.is_active)
            polygon = np.rint(observation.corners).astype(np.int32)
            cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)
            center = tuple(np.rint(observation.centroid).astype(int))
            cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)
            if state.is_active and self.args.debug_overlay:
                recent = list(state.centroid_history)[-self.args.trajectory_length :]
                trajectory = np.asarray(
                    [(round(point[1]), round(point[2])) for point in recent],
                    dtype=np.int32,
                )
                if len(trajectory) >= 2:
                    cv2.polylines(
                        frame, [trajectory], False, color, 3, cv2.LINE_AA
                    )

            if not self.args.debug_overlay:
                owner_id = state.placement_owner_global_id
                operator_state = global_state or (
                    self.identity_manager.global_rolls.get(owner_id)
                    if owner_id is not None
                    else None
                )
                label = self._production_label(operator_state)
                operator_status = self._production_status(operator_state)
                if operator_status == "WRONG BIN" and operator_state is not None:
                    operator_alerts.append(
                        "WRONG BIN: "
                        f"{operator_state.roll_id} expected "
                        f"{operator_state.expected_bin_id or '---'}, actual "
                        f"{operator_state.actual_bin_id or '---'}"
                    )
                anchor_x = max(0, int(polygon[:, 0].min()))
                anchor_y = max(24, int(polygon[:, 1].min()) - 8)
                self._draw_label(frame, label, (anchor_x, anchor_y), color)
                continue

            if global_state is not None and global_state.state == PLACED:
                label = (
                    f"ROLL G{global_id} | PLACED | "
                    f"{global_state.actual_bin_id or 'BIN ---'} | SEALED"
                )
            elif global_state is not None and global_state.state == COMPLETED:
                if self.args.bin_focused:
                    label = (
                        f"ROLL G{global_id} | Local {observation.track_id} | "
                        "CAMERA JOURNEY CLOSED"
                    )
                else:
                    direction = global_state.journey_direction.replace("_TO_", "->")
                    label = f"ROLL G{global_id} | CAMERA JOURNEY CLOSED | {direction}"
            elif global_id is None:
                pending = (
                    f"PENDING ({min(state.pending_active_frames, self.args.journey_activation_frames)}/"
                    f"{self.args.journey_activation_frames})"
                    if state.assignment_state == PENDING_PICKUP
                    else ""
                )
                label = (
                    f"ROLL | Local {observation.track_id} | "
                    f"{'ACTIVE' if state.is_active else 'STATIC'} | Global ---"
                    f"{f' | {pending}' if pending else ''}"
                )
                owner_id = state.placement_owner_global_id
                owner_state = (
                    self.identity_manager.global_rolls.get(owner_id)
                    if owner_id is not None
                    else None
                )
                if owner_state is not None:
                    if self.args.bin_focused:
                        status = (
                            owner_state.verification_result.replace("_", " ")
                            if owner_state.placement_confirmed
                            else owner_state.placement_status
                        )
                        evidence = (
                            "ENTRY"
                            if owner_state.placement_status == "PROVISIONAL"
                            else owner_state.placement_evidence.replace(
                                "ENTRY_THEN_OCCLUDED", "ENTRY->OCCLUSION"
                            ).replace("ACTIVE_TO_STATIC", "ACTIVE->STATIC")
                        )
                        label = (
                            f"{owner_state.roll_id} / Local {observation.track_id} | "
                            f"Expected: {owner_state.expected_bin_id or '---'} | "
                            f"Candidate: {owner_state.candidate_bin_id or '---'} | "
                            f"Actual: {owner_state.actual_bin_id or '---'} | "
                            f"Evidence: {evidence} | Status: {status}"
                        )
                    else:
                        label += f" | Placement owner: G{owner_id}"
                        if owner_state.bin_confirmed:
                            label += f" | PLACED: {owner_state.actual_bin_id}"
                        elif owner_state.candidate_bin_id:
                            label += f" | Candidate: {owner_state.candidate_bin_id}"
            else:
                if self.args.bin_focused:
                    label = (
                        f"G{global_id} | Local {observation.track_id} | "
                        f"{'ACTIVE' if state.is_active else 'STATIC'}"
                    )
                    if global_state.initial_bin_id:
                        label += f" | FROM: {global_state.initial_bin_id}"
                else:
                    direction = global_state.journey_direction.replace("_TO_", "->")
                    label = (
                        f"G{global_id} | Local {observation.track_id} | "
                        f"{'ACTIVE' if state.is_active else 'STATIC'} | {direction}"
                    )
                    if global_state.last_distance_to_destination is not None:
                        label += (
                            " | DestDist: "
                            f"{global_state.last_distance_to_destination:.0f} px"
                        )
                    if (
                        global_state.approaching_destination
                        and global_state.last_distance_to_destination is not None
                        and global_state.last_distance_to_destination
                        <= self.args.destination_proximity_px
                    ):
                        label += " | DESTINATION APPROACH"
            if global_state is not None and global_state.state == PLACED:
                label = (
                    f"ROLL G{global_id} | PLACED | "
                    f"{global_state.actual_bin_id or 'BIN ---'} | SEALED"
                )
            elif global_state is not None and self.args.bin_focused:
                expected = global_state.expected_bin_id or "---"
                candidate = global_state.candidate_bin_id or "---"
                actual = global_state.actual_bin_id or "---"
                status = (
                    global_state.verification_result.replace("_", " ")
                    if global_state.placement_confirmed
                    else global_state.placement_status
                )
                evidence = (
                    "ENTRY"
                    if global_state.placement_status == "PROVISIONAL"
                    else global_state.placement_evidence.replace(
                        "ENTRY_THEN_OCCLUDED", "ENTRY->OCCLUSION"
                    ).replace("ACTIVE_TO_STATIC", "ACTIVE->STATIC")
                )
                identity = (
                    f"{global_state.factory_roll_id} / G{global_state.global_id}"
                    if global_state.factory_roll_id
                    else f"G{global_state.global_id}"
                )
                label = (
                    f"{identity} | Expected: {expected} | Candidate: {candidate} | "
                    f"Actual: {actual} | Evidence: {evidence} | Status: {status}"
                )
                if global_state.initial_bin_id:
                    label += f" | PICKUP FROM {global_state.initial_bin_id}"
            elif global_state is not None:
                bin_status = self.bin_frame_statuses.get(global_state.global_id)
                if global_state.bin_confirmed:
                    label += f" | PLACED: {global_state.actual_bin_id}"
                elif bin_status is not None and bin_status.ambiguous:
                    label += (
                        f" | BIN AMBIGUOUS: {bin_status.best_bin_id} "
                        f"{bin_status.best_overlap:.2f} / "
                        f"{bin_status.second_bin_id} "
                        f"{bin_status.second_overlap:.2f}"
                    )
                elif global_state.candidate_bin_id:
                    if global_state.bin_placement_state == "PROVISIONAL_PLACEMENT":
                        provisional_elapsed = (
                            0.0
                            if global_state.provisional_start_time is None
                            else max(
                                0.0,
                                frame_index / self.fps
                                - global_state.provisional_start_time,
                            )
                        )
                        label += (
                            f" | {global_state.candidate_bin_id} PROVISIONAL | "
                            f"lost in bin | grace {provisional_elapsed:.1f}/"
                            f"{self.args.bin_placement_grace_seconds:.1f}s"
                        )
                    else:
                        label += (
                            f" | {global_state.candidate_bin_id} CANDIDATE | "
                            f"dwell {global_state.candidate_dwell_seconds:.2f}/"
                            f"{self.args.bin_visible_dwell_seconds:.2f}s | "
                            f"overlap {global_state.candidate_bin_last_overlap:.2f} | "
                            f"speed {global_state.candidate_current_speed:.0f}px/s"
                        )
            if global_state is not None:
                if global_state.placement_confidence > 0:
                    label += (
                        f" | PlacementConf: {global_state.placement_confidence:.2f}"
                    )
                if global_state.entered_bin_mouth:
                    label += (
                        f" | EntryDir: {global_state.entry_direction_score:.2f}"
                        " | EntryTraj: "
                        f"{global_state.entry_trajectory_consistency:.2f}"
                    )
                if global_state.last_candidate_cancel_reason:
                    label += f" | Reason: {global_state.last_candidate_cancel_reason}"
            anchor_x = max(0, int(polygon[:, 0].min()))
            anchor_y = max(24, int(polygon[:, 1].min()) - 8)
            self._draw_label(frame, label, (anchor_x, anchor_y), color)

        if not self.args.debug_overlay:
            for index, alert in enumerate(dict.fromkeys(operator_alerts)):
                self._draw_label(
                    frame,
                    f"WARNING: {alert}",
                    (18, 42 + index * 34),
                    (30, 30, 255),
                )
            return

        active_ids = sorted(
            state.track_id
            for state in self.tracks.values()
            if state.is_active and not state.lost
        )
        active_global_ids = sorted(
            {
                self.identity_manager.local_to_global[local_id]
                for local_id in active_ids
                if local_id in self.identity_manager.local_to_global
                and self.identity_manager.global_rolls[
                    self.identity_manager.local_to_global[local_id]
                ].state
                in VISIBLE_STATES
            }
        )
        occluded_global_ids = sorted(
            state.global_id
            for state in self.identity_manager.global_rolls.values()
            if state.state == OCCLUDED
        )
        visible_global_count = sum(
            state.state in VISIBLE_STATES
            for state in self.identity_manager.global_rolls.values()
        )
        completed_global_count = sum(
            state.state == COMPLETED
            for state in self.identity_manager.global_rolls.values()
        )
        placed_global_ids = sorted(
            state.global_id
            for state in self.identity_manager.global_rolls.values()
            if state.state == PLACED
        )
        active_camera_journey_count = sum(
            state.journey_started and state.state in VISIBLE_STATES
            for state in self.identity_manager.global_rolls.values()
        )
        confirmed_placement_count = sum(
            state.factory_placement_completed
            for state in self.identity_manager.global_rolls.values()
        )
        lines = [
            f"Frame: {frame_index}",
            f"Detected rolls: {len(observations)}",
            f"Visible globals: {visible_global_count}",
            f"Occluded globals: {occluded_global_ids or '-'}",
            f"Camera journeys closed: {completed_global_count}",
            f"Active camera journeys: {active_camera_journey_count}",
            f"Confirmed placements: {confirmed_placement_count}",
            f"Factory placements completed: {confirmed_placement_count}",
            f"Sealed globals: {placed_global_ids or '-'}",
            "Reassociations: "
            f"{self.identity_manager.event_counts['GLOBAL_REASSOCIATED']}",
        ]
        if self.args.draw_zones and not self.lifecycle_enabled:
            lines.append("WARNING: LIFECYCLE ZONES DISABLED")
        self._draw_overlay(frame, lines)

    @staticmethod
    def _production_status(global_state: GlobalRollState | None) -> str:
        """Translate internal lifecycle fields into operator-facing states."""

        if global_state is None:
            return "TRACKING"
        if global_state.placement_confirmed:
            if global_state.verification_result == CORRECT:
                return "CORRECT"
            if global_state.verification_result == WRONG_BIN:
                return "WRONG BIN"
            return "UNRESOLVED"
        if (
            global_state.state == PLACEMENT_PENDING
            or global_state.placement_status == "PROVISIONAL"
        ):
            return "PLACEMENT PENDING"
        if (
            global_state.placement_status == "UNRESOLVED"
            or global_state.state in {COMPLETED, LOST}
        ):
            return "UNRESOLVED"
        return "TRACKING"

    @classmethod
    def _production_label(cls, global_state: GlobalRollState | None) -> str:
        """Build the compact label used by the default production overlay."""

        if global_state is None:
            return "ROLL --- | Status: TRACKING"
        status = cls._production_status(global_state)
        fields = [
            f"ROLL {global_state.roll_id}",
            f"Expected: {global_state.expected_bin_id or '---'}",
        ]
        if global_state.actual_bin_id:
            fields.append(f"Actual: {global_state.actual_bin_id}")
        fields.append(f"Status: {status}")
        label = " | ".join(fields)
        return f"WARNING: WRONG BIN | {label}" if status == "WRONG BIN" else label

    @staticmethod
    def _production_color(
        status: str, is_active: bool
    ) -> tuple[int, int, int]:
        if status == "WRONG BIN":
            return (30, 30, 255)
        if status == "CORRECT":
            return (40, 220, 40)
        if status == "PLACEMENT PENDING":
            return (0, 200, 255)
        if status == "UNRESOLVED":
            return (0, 150, 255)
        return (40, 220, 40) if is_active else (255, 170, 40)

    def _draw_zones(self, frame: np.ndarray) -> None:
        styles = {
            "zone_a": (80, 220, 80, "ZONE A"),
            "tracking_zone": (255, 210, 50, "TRACKING"),
            "zone_b": (80, 80, 255, "ZONE B"),
        }
        for name, (blue, green, red, label) in styles.items():
            polygon = self.zones.get(name)
            if polygon is None:
                continue
            points = np.rint(polygon).astype(np.int32)
            color = (blue, green, red)
            cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
            anchor = tuple(points[np.argmin(points[:, 1])].tolist())
            cv2.putText(
                frame,
                label,
                (max(0, anchor[0]), max(20, anchor[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_bins(self, frame: np.ndarray) -> None:
        """Draw camera polygons using global physical bin IDs only."""

        for physical_bin_id, polygon in self.bin_polygons.items():
            points = np.rint(polygon).astype(np.int32)
            color = BIN_COLORS.get(physical_bin_id, (255, 255, 255))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.10, frame, 0.90, 0.0, frame)
            cv2.polylines(frame, [points], True, color, 3, cv2.LINE_AA)
            anchor = tuple(points[np.argmin(points[:, 1])].tolist())
            cv2.putText(
                frame,
                physical_bin_id,
                (max(0, anchor[0]), max(20, anchor[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        if self.bin_geometry is not None and self.args.camera_id:
            for physical_bin_id, polygon in self.bin_geometry.entry_polygons_for_camera(
                self.args.camera_id
            ).items():
                points = np.rint(polygon).astype(np.int32)
                color = BIN_COLORS.get(physical_bin_id, (255, 255, 255))
                cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
                anchor = tuple(points[np.argmin(points[:, 1])].tolist())
                cv2.putText(
                    frame,
                    f"{physical_bin_id} ENTRY",
                    (max(0, anchor[0]), max(20, anchor[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def _save_zone_debug_frames(
        self, frame: np.ndarray, frame_index: int
    ) -> None:
        if not self.args.save_zone_debug_frames:
            return
        debug_dir = self.args.output.parent / "zone_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for global_id, event in self.identity_manager.frame_transition_events:
            event_name = event.removeprefix("GLOBAL_")
            path = debug_dir / (
                f"frame_{frame_index:04d}_G{global_id}_{event_name}.jpg"
            )
            if path.exists() and not self.args.overwrite:
                raise FileExistsError(
                    f"Zone debug frame exists; use --overwrite: {path}"
                )
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not save zone debug frame: {path}")

    @staticmethod
    def _draw_label(
        frame: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = anchor
        cv2.rectangle(
            frame,
            (x, max(0, y - height - baseline - 6)),
            (min(frame.shape[1] - 1, x + width + 8), y + baseline),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x + 4, y - 3),
            font,
            scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_overlay(frame: np.ndarray, lines: Sequence[str]) -> None:
        overlay = frame.copy()
        width = 300
        height = 30 + 28 * len(lines)
        cv2.rectangle(overlay, (12, 12), (12 + width, 12 + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (25, 42 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def log_detection(
        self,
        frame_index: int,
        observation: TrackObservation,
        state: TrackState,
        global_id: int | None,
    ) -> None:
        global_state = (
            self.identity_manager.global_rolls.get(global_id)
            if global_id is not None
            else None
        )
        self.tracking_writer.writerow(
            {
                "frame": frame_index,
                "timestamp_seconds": f"{frame_index / self.fps:.3f}",
                "track_id": observation.track_id,
                "global_id": "" if global_id is None else global_id,
                "assignment_state": state.assignment_state,
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
                "candidate_bin_id": (
                    "" if global_state is None else global_state.candidate_bin_id
                ),
                "candidate_bin_frames": (
                    "" if global_state is None else global_state.candidate_bin_frames
                ),
                "candidate_bin_overlap": (
                    ""
                    if global_state is None
                    else f"{global_state.candidate_bin_last_overlap:.6f}"
                ),
                "roll_id": (
                    "" if global_state is None else global_state.roll_id
                ),
                "expected_bin_id": (
                    "" if global_state is None else global_state.expected_bin_id or ""
                ),
                "actual_bin_id": (
                    "" if global_state is None else global_state.actual_bin_id
                ),
                "bin_confirmed": (
                    "" if global_state is None else global_state.bin_confirmed
                ),
                "placement_confirmed": (
                    "" if global_state is None else global_state.placement_confirmed
                ),
                "placement_confirmation_method": (
                    ""
                    if global_state is None
                    else global_state.placement_confirmation_method
                ),
                "placement_frame": (
                    "" if global_state is None else global_state.placement_frame
                ),
                "placement_status": (
                    "" if global_state is None else global_state.placement_status
                ),
                "placement_evidence": (
                    "" if global_state is None else global_state.placement_evidence
                ),
                "placement_confidence": (
                    ""
                    if global_state is None
                    else f"{global_state.placement_confidence:.6f}"
                ),
                "entered_bin_mouth": (
                    "" if global_state is None else global_state.entered_bin_mouth
                ),
                "entry_frame": (
                    "" if global_state is None else global_state.entry_frame
                ),
                "entry_timestamp": (
                    ""
                    if global_state is None or global_state.entry_timestamp is None
                    else f"{global_state.entry_timestamp:.3f}"
                ),
                "entry_bin_id": (
                    "" if global_state is None else global_state.entry_bin_id
                ),
                "entry_direction_score": (
                    ""
                    if global_state is None
                    else f"{global_state.entry_direction_score:.6f}"
                ),
                "entry_trajectory_consistency": (
                    ""
                    if global_state is None
                    else f"{global_state.entry_trajectory_consistency:.6f}"
                ),
                "last_visible_centroid": (
                    ""
                    if global_state is None or global_state.last_visible_centroid is None
                    else (
                        f"{global_state.last_visible_centroid[0]:.3f},"
                        f"{global_state.last_visible_centroid[1]:.3f}"
                    )
                ),
                "verification_result": (
                    "" if global_state is None else global_state.verification_result
                ),
                "class_id": observation.class_id,
                "confidence": f"{observation.confidence:.6f}",
                "cx": f"{observation.centroid[0]:.3f}",
                "cy": f"{observation.centroid[1]:.3f}",
                "is_active": state.is_active,
                "first_seen_frame": state.first_seen_frame,
                "last_seen_frame": state.last_seen_frame,
            }
        )

    def log_event(
        self,
        frame_index: int,
        track_id: int,
        event: str,
        *,
        global_id: int | None = None,
        active_frames: int | None = None,
        net_displacement: float | None = None,
        motion_consistency: float | None = None,
        journey_direction: str = "",
        origin_zone: str = "",
        destination_zone: str = "",
        activation_reason: str = "",
    ) -> None:
        self.event_writer.writerow(
            {
                "frame": frame_index,
                "timestamp_seconds": f"{frame_index / self.fps:.3f}",
                "track_id": track_id,
                "global_id": "" if global_id is None else global_id,
                "event": event,
                "active_frames": "" if active_frames is None else active_frames,
                "net_displacement": (
                    "" if net_displacement is None else f"{net_displacement:.3f}"
                ),
                "motion_consistency": (
                    "" if motion_consistency is None else f"{motion_consistency:.6f}"
                ),
                "journey_direction": journey_direction,
                "origin_zone": origin_zone,
                "destination_zone": destination_zone,
                "activation_reason": activation_reason,
            }
        )
        if self.args.debug:
            readable = {
                "TRACK_CREATED": "CREATED",
                "BECAME_ACTIVE": "ACTIVE",
                "BECAME_STATIC": "STATIC",
                "TRACK_LOST": "LOST",
            }.get(event, event)
            details = ""
            if event == "PENDING_PROMOTED":
                details = (
                    f" -> G{global_id} {journey_direction} "
                    f"({activation_reason}, {net_displacement:.1f}px)"
                )
            print(f"[frame {frame_index}] Track {track_id} {readable}{details}")

    def build_summary(self, processed_frames: int) -> str:
        active_ids = sorted(self.tracks_that_became_active)
        unassigned_local_count = sum(
            not state.ever_assigned_global for state in self.tracks.values()
        )
        if self.args.bin_focused:
            configured_bins = ", ".join(self.bin_polygons) or "none"
            camera_journeys_closed = sum(
                state.state == COMPLETED
                for state in self.identity_manager.global_rolls.values()
            )
            active_camera_journeys = sum(
                state.journey_started and state.state in VISIBLE_STATES
                for state in self.identity_manager.global_rolls.values()
            )
            factory_placements_completed = sum(
                state.factory_placement_completed
                for state in self.identity_manager.global_rolls.values()
            )
            lines = [
                "Physical-bin tracking summary",
                "=============================",
                f"Source: {self.args.source}",
                f"Model: {self.args.model}",
                f"Tracker type: {self.tracker_type}",
                f"Tracker config: {self.args.tracker}",
                f"Confidence threshold: {self.args.conf:.3f}",
                f"Processed frames: {processed_frames}",
                f"Video FPS: {self.fps:.3f}",
                f"Physical-bin camera: {self.args.camera_id}",
                f"Configured physical bins: {configured_bins}",
                "Placement calculation: OBB-to-physical-bin polygon overlap",
                f"Total unique local track IDs: {len(self.unique_track_ids)}",
                "Transport Global IDs created: "
                f"{self.identity_manager.event_counts['GLOBAL_CREATED']}",
                "GLOBAL_REASSOCIATED count: "
                f"{self.identity_manager.event_counts['GLOBAL_REASSOCIATED']}",
                f"Camera journeys closed: {camera_journeys_closed}",
                f"Active camera journeys: {active_camera_journeys}",
                f"Confirmed placements: {factory_placements_completed}",
                "Factory placements completed: "
                f"{factory_placements_completed}",
                "Global IDs sealed by placement: "
                f"{sum(state.state == PLACED for state in self.identity_manager.global_rolls.values())}",
                "Attempts to reassociate sealed globals: "
                f"{self.identity_manager.event_counts['REASSOCIATION_REJECTED_TERMINAL_GLOBAL']}",
                "Opposite-flow reassociation rejections: "
                f"{self.identity_manager.event_counts['REASSOCIATION_REJECTED_OPPOSITE_BIN_FLOW']}",
                "Possible false global reuse events: "
                f"{len(self.identity_manager.possible_false_global_reuse_events)}",
                "",
                "Placement lifecycle audit",
                "-------------------------",
                *[
                    line
                    for state in sorted(
                        self.identity_manager.global_rolls.values(),
                        key=lambda value: value.global_id,
                    )
                    for line in (
                        f"G{state.global_id}:",
                        "  placement_confirmed_frame: "
                        f"{state.placement_confirmed_frame if state.placement_confirmed_frame is not None else 'NONE'}",
                        f"  lifecycle_after_placement: {state.state if state.placement_confirmed_frame is not None else 'N/A'}",
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
                "",
                *self.bin_entry_manager.summary_lines(
                    list(self.identity_manager.global_rolls.values())
                ),
            ]
            return "\n".join(lines).rstrip() + "\n"
        lines = [
            "Tracking continuity summary",
            "===========================",
            f"Source: {self.args.source}",
            f"Model: {self.args.model}",
            f"Tracker type: {self.tracker_type}",
            f"Tracker config: {self.args.tracker}",
            f"Confidence threshold: {self.args.conf:.3f}",
            f"Processed frames: {processed_frames}",
            f"Video FPS: {self.fps:.3f}",
            "",
            f"Total unique local track IDs: {len(self.unique_track_ids)}",
            f"Static unassigned local tracks: {unassigned_local_count}",
            f"Pending tracks created: {self.pending_tracks_created}",
            f"Pending tracks promoted: {self.pending_tracks_promoted}",
            f"Pending tracks cancelled: {self.pending_tracks_cancelled}",
            f"Pending direction ambiguous: {self.pending_direction_ambiguous}",
            "Transport Global IDs created: "
            f"{self.identity_manager.event_counts['GLOBAL_CREATED']}",
            f"Global IDs created via endpoint activation: {self.endpoint_global_creations}",
            "Global IDs created via sustained-motion activation: "
            f"{self.sustained_global_creations}",
            "Longest ACTIVE-without-Global duration: "
            f"{self.longest_active_without_global} frames",
            "A_TO_B journeys: "
            f"{sum(s.journey_direction == 'A_TO_B' for s in self.identity_manager.global_rolls.values())}",
            "B_TO_A journeys: "
            f"{sum(s.journey_direction == 'B_TO_A' for s in self.identity_manager.global_rolls.values())}",
            "GLOBAL_PICKED_UP count: "
            f"{self.identity_manager.event_counts['GLOBAL_PICKED_UP']}",
            "GLOBAL_REASSOCIATED count: "
            f"{self.identity_manager.event_counts['GLOBAL_REASSOCIATED']}",
            "GLOBAL_COMPLETED count: "
            f"{self.identity_manager.event_counts['GLOBAL_COMPLETED']}",
            "Static Global IDs incorrectly created: "
            f"{self.identity_manager.static_global_creation_count}",
            f"Tracks that became ACTIVE: {active_ids}",
            f"Number of TRACK_CREATED events: {self.track_created_event_count}",
            f"Number of TRACK_LOST events: {self.lost_event_count}",
            f"Maximum simultaneous tracks: {self.maximum_simultaneous_tracks}",
            f"active_track_fragments: {len(active_ids)}",
            "",
            "Per-track lifespan",
            "------------------",
        ]
        for track_id in sorted(self.tracks):
            state = self.tracks[track_id]
            duration = (state.last_seen_frame - state.first_seen_frame) / self.fps
            first_confidence = (
                f"{state.first_confidence:.3f}"
                if state.first_confidence is not None
                else "n/a"
            )
            last_confidence = (
                f"{state.last_confidence:.3f}"
                if state.last_confidence is not None
                else "n/a"
            )
            lines.extend(
                [
                    f"Track {track_id}:",
                    f"  first frame: {state.first_seen_frame}",
                    f"  last frame: {state.last_seen_frame}",
                    f"  frames detected: {state.frames_seen}",
                    f"  duration: {duration:.3f} sec",
                    f"  became active: {'yes' if state.became_active else 'no'}",
                    f"  maximum gap before recovery: {state.maximum_recovered_gap} frames",
                    f"  first confidence: {first_confidence}",
                    f"  last confidence: {last_confidence}",
                    f"  final status: {'LOST' if state.lost else 'RETAINED'}",
                    "",
                ]
            )

        active_states = sorted(
            (state for state in self.tracks.values() if state.became_active),
            key=lambda state: (state.first_seen_frame, state.track_id),
        )
        handoffs: list[tuple[TrackState, TrackState, int]] = []
        for old_state, new_state in zip(active_states, active_states[1:]):
            if new_state.first_seen_frame > old_state.last_seen_frame:
                gap = new_state.first_seen_frame - old_state.last_seen_frame - 1
                handoffs.append((old_state, new_state, gap))
        lines.extend(
            [
                "Potential ID handoffs (temporal diagnostic only)",
                "------------------------------------------------",
                "These entries do not merge or reassign tracker IDs.",
            ]
        )
        if not handoffs:
            lines.append("None observed between ACTIVE tracks.")
        else:
            for old_state, new_state, gap in handoffs:
                before_confidence = (
                    f"{old_state.last_confidence:.3f}"
                    if old_state.last_confidence is not None
                    else "n/a"
                )
                after_confidence = (
                    f"{new_state.first_confidence:.3f}"
                    if new_state.first_confidence is not None
                    else "n/a"
                )
                lines.extend(
                    [
                        f"Track {old_state.track_id} -> Track {new_state.track_id}:",
                        f"  last old frame: {old_state.last_seen_frame}",
                        f"  first new frame: {new_state.first_seen_frame}",
                        f"  gap: {gap} frames ({gap / self.fps:.3f} sec)",
                        f"  confidence before gap: {before_confidence}",
                        f"  confidence after gap: {after_confidence}",
                    ]
                )
        lines.extend(["", *self.identity_manager.summary_lines()])
        lines.extend(
            [
                "",
                *self.bin_entry_manager.summary_lines(
                    list(self.identity_manager.global_rolls.values())
                ),
            ]
        )
        return "\n".join(lines).rstrip() + "\n"


def refuse_existing_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n" + formatted
        )


def run(args: argparse.Namespace) -> None:
    args.model = args.model.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.roi_config is not None:
        args.roi_config = args.roi_config.expanduser().resolve()
    if args.zones_config is not None:
        args.zones_config = args.zones_config.expanduser().resolve()
    if args.bins_config is not None:
        args.bins_config = args.bins_config.expanduser().resolve()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.is_file():
        raise FileNotFoundError(f"Source video not found: {args.source}")
    if args.output.suffix.casefold() != ".mp4":
        raise ValueError("--output must use the .mp4 extension")
    ensure_inside_tracking(args.output, "Output video")
    if args.source == args.output:
        raise ValueError("Source and output video paths must be different")

    tracking_csv, events_csv = output_log_paths(args.output)
    summary_path = output_summary_path(args.output)
    global_events_csv = output_global_event_path(args.output)
    bin_events_csv = output_bin_event_path(args.output)
    requested_outputs = [args.output, summary_path]
    if args.bin_events_only:
        requested_outputs.append(bin_events_csv)
    elif not args.summary_only:
        requested_outputs.extend(
            (tracking_csv, events_csv, global_events_csv, bin_events_csv)
        )
    refuse_existing_outputs(requested_outputs, overwrite=args.overwrite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    roi = load_roi(args.roi_config)
    zones = load_zones(args.zones_config)
    bin_geometry: BinGeometry | None = None
    bin_polygons: Mapping[str, np.ndarray] = {}
    if args.camera_id:
        if not args.bins_config.is_file():
            raise FileNotFoundError(f"Bin config not found: {args.bins_config}")
        bin_geometry = load_bin_config(args.bins_config)
        bin_polygons = bin_geometry.polygons_for_camera(args.camera_id)

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open source video: {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source FPS reported by OpenCV: {fps}")
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"No readable frames in source video: {args.source}")
    height, width = first_frame.shape[:2]
    lifecycle_enabled = validate_zones_for_frame(zones, width, height)

    print(f"Video FPS: {fps:.3f}")
    print(f"Resolution: {width}x{height}")
    print(f"Number of frames: {total_frames}")
    print(f"Tracker being used: {args.tracker}")
    print(f"Model path: {args.model}")
    print(f"Confidence threshold: {args.conf:.3f}")
    print(f"Output codec: {args.codec}")
    print(f"ROI: {args.roi_config if args.roi_config else 'whole frame'}")
    print(f"Camera zone config: {args.zones_config}")
    configured_zones = ", ".join(sorted(zones)) or "none"
    print(f"Lifecycle zones: {configured_zones}")
    if args.camera_id:
        configured_bins = ", ".join(bin_polygons) or "none"
        print(f"Physical bin config: {args.bins_config}")
        print(f"Physical bin camera: {args.camera_id}")
        print(f"Configured physical bins: {configured_bins}")
        print(f"Bin overlap threshold: {args.bin_overlap_threshold:.3f}")
        print(
            "Bin visible dwell duration: "
            f"{args.bin_visible_dwell_seconds:.3f} sec"
        )
        print(
            "Bin gap tolerance: "
            f"{args.bin_gap_tolerance_seconds:.3f} sec"
        )
        print(
            "Bin occlusion confirmation grace: "
            f"{args.bin_placement_grace_seconds:.3f} sec"
        )
        print(
            "Bin settle speed threshold: "
            f"{args.bin_settle_speed_px_per_sec:.3f} px/sec"
        )
        print(f"Bin ambiguity margin: {args.bin_ambiguity_margin:.3f}")
        if not bin_polygons:
            print(
                f"WARNING: no physical bin polygons configured for "
                f"{args.camera_id}"
            )
    if args.summary_only:
        output_log_description = "summary TXT only"
    elif args.bin_events_only:
        output_log_description = "focused bin-event CSV + summary TXT"
    else:
        output_log_description = "all CSVs + summary TXT"
    print(f"Output logs: {output_log_description}")
    if args.bin_focused:
        print("Presentation/calculation focus: physical bins")
    if args.draw_zones and not lifecycle_enabled:
        print("WARNING: --draw-zones enabled, but lifecycle zones are invalid")

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"OpenCV could not create output video {args.output} with codec "
            f"{args.codec}. Try --codec avc1 on macOS."
        )

    detection_fields = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "global_id",
        "assignment_state",
        "lifecycle_state",
        "zone",
        "origin_zone",
        "destination_zone",
        "journey_direction",
        "roll_id",
        "expected_bin_id",
        "candidate_bin_id",
        "candidate_bin_frames",
        "candidate_bin_overlap",
        "actual_bin_id",
        "bin_confirmed",
        "placement_confirmed",
        "placement_confirmation_method",
        "placement_frame",
        "placement_status",
        "placement_evidence",
        "placement_confidence",
        "entered_bin_mouth",
        "entry_frame",
        "entry_timestamp",
        "entry_bin_id",
        "entry_direction_score",
        "entry_trajectory_consistency",
        "last_visible_centroid",
        "verification_result",
        "class_id",
        "confidence",
        "cx",
        "cy",
        "is_active",
        "first_seen_frame",
        "last_seen_frame",
    ]
    event_fields = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "global_id",
        "event",
        "active_frames",
        "net_displacement",
        "motion_consistency",
        "journey_direction",
        "origin_zone",
        "destination_zone",
        "activation_reason",
    ]
    global_event_fields = [
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
    bin_event_fields = [
        "frame",
        "timestamp_seconds",
        "global_id",
        "roll_id",
        "local_id",
        "physical_bin_id",
        "expected_bin_id",
        "actual_bin_id",
        "placement_confirmed",
        "placement_confirmation_method",
        "placement_frame",
        "placement_status",
        "placement_evidence",
        "placement_confidence",
        "entered_bin_mouth",
        "entry_frame",
        "entry_timestamp",
        "entry_bin_id",
        "entry_direction_score",
        "entry_trajectory_consistency",
        "last_visible_centroid",
        "verification_result",
        "previous_bin_id",
        "candidate_state",
        "candidate_start_frame",
        "candidate_start_time",
        "current_overlap",
        "max_overlap",
        "accumulated_dwell_seconds",
        "current_speed_px_per_sec",
        "last_valid_overlap_time",
        "gap_seconds",
        "last_visible_frame",
        "last_overlap",
        "last_centroid_x",
        "last_centroid_y",
        "motion_dx",
        "motion_dy",
        "first_distance_to_bin_center",
        "last_distance_to_bin_center",
        "centroid_inside",
        "overlap_increased",
        "moved_deeper",
        "provisional_elapsed_seconds",
        "grace_duration_seconds",
        "confirmation_method",
        "reason",
        "event",
    ]
    processed_frames = 0
    try:
        with ExitStack() as stack:
            if args.summary_only:
                tracking_writer = DiscardingCsvWriter()
                event_writer = DiscardingCsvWriter()
                global_event_writer = DiscardingCsvWriter()
                bin_event_writer = DiscardingCsvWriter()
            elif args.bin_events_only:
                tracking_writer = DiscardingCsvWriter()
                event_writer = DiscardingCsvWriter()
                global_event_writer = DiscardingCsvWriter()
                bin_event_handle = stack.enter_context(
                    bin_events_csv.open("w", newline="", encoding="utf-8")
                )
                bin_event_writer = csv.DictWriter(
                    bin_event_handle, fieldnames=bin_event_fields
                )
            else:
                tracking_handle = stack.enter_context(
                    tracking_csv.open("w", newline="", encoding="utf-8")
                )
                event_handle = stack.enter_context(
                    events_csv.open("w", newline="", encoding="utf-8")
                )
                global_event_handle = stack.enter_context(
                    global_events_csv.open("w", newline="", encoding="utf-8")
                )
                bin_event_handle = stack.enter_context(
                    bin_events_csv.open("w", newline="", encoding="utf-8")
                )
            if not args.summary_only and not args.bin_events_only:
                tracking_writer = csv.DictWriter(
                    tracking_handle, fieldnames=detection_fields
                )
                event_writer = csv.DictWriter(
                    event_handle, fieldnames=event_fields
                )
                global_event_writer = csv.DictWriter(
                    global_event_handle, fieldnames=global_event_fields
                )
                bin_event_writer = csv.DictWriter(
                    bin_event_handle, fieldnames=bin_event_fields
                )
            tracking_writer.writeheader()
            event_writer.writeheader()
            global_event_writer.writeheader()
            bin_event_writer.writeheader()
            tracker = RollTracker(
                args,
                roi,
                zones,
                bin_polygons,
                bin_geometry,
                lifecycle_enabled,
                tracking_writer,
                event_writer,
                global_event_writer,
                bin_event_writer,
                fps,
                width,
                height,
            )

            frame = first_frame
            frame_index = 0
            while True:
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
                annotated, _ = tracker.process_frame(frame, frame_index)
                writer.write(annotated)
                processed_frames += 1
                frame_index += 1
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
            tracker.bin_entry_manager.finalize(
                list(tracker.identity_manager.global_rolls.values()),
                max(0, processed_frames - 1),
            )
            tracker._seal_confirmed_placements(max(0, processed_frames - 1))
    finally:
        writer.release()
        capture.release()

    summary = tracker.build_summary(processed_frames)
    summary_path.write_text(summary, encoding="utf-8")
    print("\nTracking complete\n")
    print(summary, end="")
    print(f"Output video path: {args.output}")
    if not args.summary_only and not args.bin_events_only:
        print(f"CSV path: {tracking_csv}")
        print(f"Event CSV path: {events_csv}")
        print(f"Global event CSV path: {global_events_csv}")
        print(f"Bin event CSV path: {bin_events_csv}")
    elif args.bin_events_only:
        print(f"Bin event CSV path: {bin_events_csv}")
    print(f"Summary path: {summary_path}")


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
