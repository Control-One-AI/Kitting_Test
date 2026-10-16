#!/usr/bin/env python3
"""Create a timestamp-synchronized, side-by-side camera debug video.

This is a visualization utility only.  It reads existing tracking and optional
cross-camera match CSV files; it never performs or changes identity matching.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


TRACK_FIELD_ALIASES = {
    "frame": ("frame", "frame_number", "frame_index"),
    "local_id": ("track_id", "local_id", "local_track_id"),
    "global_id": ("global_id", "camera_global_id", "global_roll_id"),
    "direction": ("journey_direction", "direction"),
    "state": ("lifecycle_state", "state", "assignment_state"),
    "bin": ("actual_bin_id", "actual_bin", "physical_bin_id"),
    "cx": ("cx", "centroid_x", "center_x"),
    "cy": ("cy", "centroid_y", "center_y"),
}

FACTORY_ALIASES = (
    "factory_roll_id",
    "factory_id",
    "factory_global_id",
    "roll_id",
)
CAM0_GLOBAL_ALIASES = (
    "cam0_global_id",
    "cam0_global",
    "cam0_gid",
    "source_global_id",
)
CAM1_GLOBAL_ALIASES = (
    "cam1_global_id",
    "cam1_global",
    "cam1_gid",
    "destination_global_id",
)
CAM0_EXIT_TIME_ALIASES = (
    "cam0_exit_timestamp",
    "cam0_exit_timestamp_seconds",
    "cam0_exit_time",
    "exit_timestamp_seconds",
    "exit_time",
)
CAM1_ENTRY_TIME_ALIASES = (
    "cam1_entry_timestamp",
    "cam1_entry_timestamp_seconds",
    "cam1_entry_time",
    "entry_timestamp_seconds",
    "entry_time",
)
GAP_ALIASES = ("handoff_gap", "handoff_gap_seconds", "gap", "gap_seconds")
SCORE_ALIASES = ("match_score", "score", "reid_score")
EVENT_DISPLAY_SECONDS = 0.75


@dataclass(frozen=True)
class TrackRecord:
    local_id: str
    global_id: str
    direction: str
    state: str
    actual_bin: str
    cx: float | None
    cy: float | None


@dataclass(frozen=True)
class CrossCameraEvent:
    camera: str
    native_timestamp: float
    factory_id: str
    event: str
    handoff_gap: str = ""
    match_score: str = ""

    def display_text(self) -> str:
        parts = [f"{self.factory_id or 'Factory ---'} {self.event}"]
        if self.handoff_gap:
            parts.append(f"gap {format_number(self.handoff_gap)}s")
        if self.match_score:
            parts.append(f"score {format_number(self.match_score)}")
        return " | ".join(parts)


@dataclass
class CrossCameraData:
    factory_by_camera_global: dict[tuple[str, str], str] = field(
        default_factory=dict
    )
    events: list[CrossCameraEvent] = field(default_factory=list)

    def factory_id(self, camera: str, global_id: str) -> str:
        if not global_id:
            return ""
        return self.factory_by_camera_global.get(
            (normalize_camera(camera), normalize_global_id(global_id)), ""
        )

    def visible_events(
        self, camera: str, native_timestamp: float
    ) -> list[CrossCameraEvent]:
        camera = normalize_camera(camera)
        return [
            event
            for event in self.events
            if event.camera == camera
            and abs(event.native_timestamp - native_timestamp)
            <= EVENT_DISPLAY_SECONDS
        ]


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    last_timestamp: float


class TimestampFrameReader:
    """Fetch nearest source frames from a monotonically increasing timeline."""

    def __init__(self, info: VideoInfo) -> None:
        self.info = info
        self.capture = cv2.VideoCapture(str(info.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {info.path}")
        self.last_index: int | None = None
        self.last_frame: np.ndarray | None = None

    def get(self, native_timestamp: float) -> tuple[np.ndarray | None, int | None]:
        epsilon = 0.5 / self.info.fps
        if native_timestamp < -epsilon or native_timestamp > self.info.last_timestamp + epsilon:
            return None, None

        frame_index = int(round(max(0.0, native_timestamp) * self.info.fps))
        frame_index = min(frame_index, self.info.frame_count - 1)
        if frame_index == self.last_index and self.last_frame is not None:
            return self.last_frame.copy(), frame_index

        if self.last_index is None or frame_index != self.last_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None, None
        self.last_index = frame_index
        self.last_frame = frame
        return frame.copy(), frame_index

    def close(self) -> None:
        self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one synchronized CAM0 | CAM1 debug video from existing "
            "videos and tracking CSVs. No identity matching is performed."
        )
    )
    parser.add_argument("--cam0-video", type=Path, required=True)
    parser.add_argument("--cam1-video", type=Path, required=True)
    parser.add_argument("--cam0-csv", type=Path, required=True)
    parser.add_argument("--cam1-csv", type=Path, required=True)
    parser.add_argument(
        "--cross-camera-csv",
        type=Path,
        help="Optional cross-camera matches/events CSV. IDs are never inferred.",
    )
    parser.add_argument("--cam0-offset", type=float, default=0.0)
    parser.add_argument("--cam1-offset", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/cross_camera_debug.mp4")
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Optional normalized start time in seconds.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="Optional normalized end time in seconds.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        help="Output FPS (default: higher of the two source FPS values).",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=540,
        help="Height of each camera panel; aspect ratio is preserved (default: 540).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output video.",
    )
    return parser.parse_args()


def first_value(row: dict[str, str], aliases: Iterable[str]) -> str:
    for name in aliases:
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def optional_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(value: str) -> str:
    number = optional_float(value)
    return f"{number:.2f}" if number is not None else value


def normalize_camera(value: str) -> str:
    compact = value.strip().lower().replace("_", "").replace("-", "")
    if compact in {"0", "camera0", "cam0"}:
        return "cam0"
    if compact in {"1", "camera1", "cam1"}:
        return "cam1"
    return value.strip().lower()


def normalize_global_id(value: str) -> str:
    value = str(value).strip()
    lowered = value.lower()
    for prefix in ("camera_global_", "global_roll_", "camg", "global", "g"):
        if lowered.startswith(prefix):
            return value[len(prefix) :].strip(" _-")
    return value


def display_global_id(value: str) -> str:
    normalized = normalize_global_id(value)
    return f"CamG{normalized}" if normalized else "CamG---"


def inspect_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video metadata for {path}: fps={fps}, frames={frame_count}, "
            f"size={width}x{height}"
        )
    return VideoInfo(
        path=path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration=frame_count / fps,
        last_timestamp=(frame_count - 1) / fps,
    )


def resolve_track_columns(fieldnames: list[str], csv_path: Path) -> dict[str, str | None]:
    available = set(fieldnames)
    resolved: dict[str, str | None] = {}
    for semantic_name, aliases in TRACK_FIELD_ALIASES.items():
        resolved[semantic_name] = next(
            (name for name in aliases if name in available), None
        )
    if resolved["frame"] is None:
        raise ValueError(
            f"{csv_path} needs one of these frame columns: "
            f"{', '.join(TRACK_FIELD_ALIASES['frame'])}"
        )
    return resolved


def load_tracking_csv(path: Path) -> dict[int, list[TrackRecord]]:
    if not path.is_file():
        raise FileNotFoundError(f"Tracking CSV does not exist: {path}")
    records: dict[int, list[TrackRecord]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Tracking CSV has no header: {path}")
        columns = resolve_track_columns(reader.fieldnames, path)
        for line_number, row in enumerate(reader, start=2):
            raw_frame = row.get(str(columns["frame"]), "")
            try:
                frame = int(float(raw_frame))
            except (TypeError, ValueError):
                print(f"Warning: skipped {path.name}:{line_number}; invalid frame {raw_frame!r}")
                continue

            def get(name: str) -> str:
                column = columns[name]
                return str(row.get(column, "")).strip() if column else ""

            records.setdefault(frame, []).append(
                TrackRecord(
                    local_id=get("local_id"),
                    global_id=get("global_id"),
                    direction=get("direction"),
                    state=get("state"),
                    actual_bin=get("bin"),
                    cx=optional_float(get("cx")),
                    cy=optional_float(get("cy")),
                )
            )
    return records


def add_mapping(
    data: CrossCameraData, camera: str, global_id: str, factory_id: str
) -> None:
    if global_id and factory_id:
        key = (normalize_camera(camera), normalize_global_id(global_id))
        existing = data.factory_by_camera_global.get(key)
        if existing and existing != factory_id:
            raise ValueError(
                f"Conflicting Factory Roll IDs for {camera} global {global_id}: "
                f"{existing!r} versus {factory_id!r}"
            )
        data.factory_by_camera_global[key] = factory_id


def load_cross_camera_csv(path: Path | None) -> CrossCameraData:
    data = CrossCameraData()
    if path is None:
        return data
    if not path.is_file():
        raise FileNotFoundError(f"Cross-camera CSV does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Cross-camera CSV has no header: {path}")
        for row in reader:
            factory_id = first_value(row, FACTORY_ALIASES)
            cam0_global = first_value(row, CAM0_GLOBAL_ALIASES)
            cam1_global = first_value(row, CAM1_GLOBAL_ALIASES)
            add_mapping(data, "cam0", cam0_global, factory_id)
            add_mapping(data, "cam1", cam1_global, factory_id)

            gap = first_value(row, GAP_ALIASES)
            score = first_value(row, SCORE_ALIASES)
            cam0_exit_time = optional_float(first_value(row, CAM0_EXIT_TIME_ALIASES))
            cam1_entry_time = optional_float(first_value(row, CAM1_ENTRY_TIME_ALIASES))
            if cam0_exit_time is not None:
                data.events.append(
                    CrossCameraEvent(
                        "cam0", cam0_exit_time, factory_id, "EXIT", gap, score
                    )
                )
            if cam1_entry_time is not None:
                data.events.append(
                    CrossCameraEvent(
                        "cam1", cam1_entry_time, factory_id, "ENTRY", gap, score
                    )
                )

            # Also support an event-oriented CSV with one camera/event per row.
            camera = normalize_camera(
                first_value(row, ("camera", "camera_id", "camera_name"))
            )
            event = first_value(row, ("event", "event_type")).upper()
            event_time = optional_float(
                first_value(row, ("timestamp_seconds", "timestamp", "event_time"))
            )
            row_global = first_value(
                row, ("camera_global_id", "global_id", "global_roll_id")
            )
            if camera in {"cam0", "cam1"}:
                add_mapping(data, camera, row_global, factory_id)
            if camera in {"cam0", "cam1"} and event and event_time is not None:
                data.events.append(
                    CrossCameraEvent(
                        camera, event_time, factory_id, event, gap, score
                    )
                )
    return data


def fit_to_panel(
    frame: np.ndarray, panel_width: int, panel_height: int
) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    x_offset = (panel_width - resized_width) // 2
    y_offset = (panel_height - resized_height) // 2
    panel[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return panel, scale, x_offset, y_offset


def outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def identity_label(
    camera: str, record: TrackRecord, cross_data: CrossCameraData
) -> str:
    factory = cross_data.factory_id(camera, record.global_id)
    local = record.local_id or "---"
    if factory:
        return f"{factory} | {display_global_id(record.global_id)} | Local{local}"
    return f"Factory --- | {display_global_id(record.global_id)} | Local{local}"


def draw_camera_panel(
    camera: str,
    frame: np.ndarray | None,
    frame_index: int | None,
    info: VideoInfo,
    normalized_timestamp: float,
    records: list[TrackRecord],
    cross_data: CrossCameraData,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    if frame is None or frame_index is None:
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(
            "NO FRAME", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
        )[0]
        origin = (
            (panel_width - text_size[0]) // 2,
            (panel_height + text_size[1]) // 2,
        )
        outlined_text(panel, "NO FRAME", origin, 1.5, (120, 120, 255), 3)
        outlined_text(panel, camera.upper(), (18, 30), 0.75, (80, 220, 255), 2)
        outlined_text(
            panel,
            f"normalized {normalized_timestamp:.3f} s",
            (18, 57),
            0.55,
        )
        return panel

    panel, resize_scale, x_offset, y_offset = fit_to_panel(
        frame, panel_width, panel_height
    )
    native_timestamp = frame_index / info.fps
    header_lines = (
        camera.upper(),
        f"frame {frame_index}",
        f"video time {native_timestamp:.3f} s",
        f"normalized {normalized_timestamp:.3f} s",
    )
    for line_index, line in enumerate(header_lines):
        outlined_text(
            panel,
            line,
            (18, 28 + line_index * 23),
            0.55 if line_index else 0.72,
            (80, 220, 255) if line_index == 0 else (255, 255, 255),
            2 if line_index == 0 else 1,
        )

    summary_y = 125
    for record_index, record in enumerate(records):
        label = identity_label(camera, record, cross_data)
        details = " | ".join(
            part
            for part in (
                record.direction,
                record.state,
                f"bin {record.actual_bin}" if record.actual_bin else "",
            )
            if part
        )
        color = (80, 255, 120) if record.global_id else (180, 180, 180)
        if record.cx is not None and record.cy is not None:
            point = (
                int(round(record.cx * resize_scale)) + x_offset,
                int(round(record.cy * resize_scale)) + y_offset,
            )
            cv2.circle(panel, point, 5, color, -1, cv2.LINE_AA)
            outlined_text(
                panel,
                label,
                (min(point[0] + 8, max(5, panel_width - 360)), max(20, point[1] - 8)),
                0.43,
                color,
                1,
            )
        if record_index < 7:
            outlined_text(panel, label, (18, summary_y), 0.43, color, 1)
            summary_y += 19
            if details:
                outlined_text(
                    panel, details, (31, summary_y), 0.39, (225, 225, 225), 1
                )
                summary_y += 18

    events = cross_data.visible_events(camera, native_timestamp)
    for event_index, event in enumerate(events[:3]):
        banner = event.display_text()
        banner_y = panel_height - 25 - event_index * 28
        text_size = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )[0]
        cv2.rectangle(
            panel,
            (9, banner_y - 21),
            (min(panel_width - 9, 27 + text_size[0]), banner_y + 7),
            (0, 0, 0),
            -1,
        )
        outlined_text(panel, banner, (18, banner_y), 0.55, (0, 220, 255), 2)
    return panel


def validate_args(args: argparse.Namespace) -> None:
    if args.panel_height < 180:
        raise ValueError("--panel-height must be at least 180 pixels")
    if args.output_fps is not None and args.output_fps <= 0:
        raise ValueError("--output-fps must be greater than zero")
    if (
        args.start_time is not None
        and args.end_time is not None
        and args.end_time <= args.start_time
    ):
        raise ValueError("--end-time must be greater than --start-time")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}\nUse --overwrite to replace it."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    cam0_info = inspect_video(args.cam0_video)
    cam1_info = inspect_video(args.cam1_video)
    cam0_tracks = load_tracking_csv(args.cam0_csv)
    cam1_tracks = load_tracking_csv(args.cam1_csv)
    cross_data = load_cross_camera_csv(args.cross_camera_csv)

    cam0_normalized_start = args.cam0_offset
    cam1_normalized_start = args.cam1_offset
    cam0_normalized_end = args.cam0_offset + cam0_info.last_timestamp
    cam1_normalized_end = args.cam1_offset + cam1_info.last_timestamp
    union_start = min(cam0_normalized_start, cam1_normalized_start)
    union_end = max(cam0_normalized_end, cam1_normalized_end)
    overlap_start = max(cam0_normalized_start, cam1_normalized_start)
    overlap_end = min(cam0_normalized_end, cam1_normalized_end)
    overlap_duration = max(0.0, overlap_end - overlap_start)

    output_start = args.start_time if args.start_time is not None else union_start
    output_end = args.end_time if args.end_time is not None else union_end
    output_start = max(output_start, union_start)
    output_end = min(output_end, union_end)
    if output_end <= output_start:
        raise ValueError(
            f"Requested normalized range [{output_start:.3f}, {output_end:.3f}] "
            "does not contain video frames."
        )

    output_fps = args.output_fps or max(cam0_info.fps, cam1_info.fps)
    panel_aspect = max(
        cam0_info.width / cam0_info.height,
        cam1_info.width / cam1_info.height,
    )
    panel_width = int(math.ceil(args.panel_height * panel_aspect / 2.0) * 2)
    output_size = (panel_width * 2, args.panel_height)
    output_frame_count = int(math.floor((output_end - output_start) * output_fps)) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    print(f"Cam0 duration: {cam0_info.duration:.3f} s")
    print(f"Cam1 duration: {cam1_info.duration:.3f} s")
    print(f"Cam0 FPS: {cam0_info.fps:.3f}")
    print(f"Cam1 FPS: {cam1_info.fps:.3f}")
    print(f"Cam0 offset: {args.cam0_offset:+.3f} s")
    print(f"Cam1 offset: {args.cam1_offset:+.3f} s")
    print(
        f"Normalized overlap: {overlap_duration:.3f} s "
        f"({overlap_start:.3f} to {overlap_end:.3f})"
    )
    print(
        f"Output normalized range: {output_start:.3f} to {output_end:.3f} s"
    )
    print(f"Factory mappings loaded: {len(cross_data.factory_by_camera_global)}")
    print(f"Cross-camera event markers loaded: {len(cross_data.events)}")

    cam0_reader = TimestampFrameReader(cam0_info)
    cam1_reader = TimestampFrameReader(cam1_info)
    try:
        for output_index in range(output_frame_count):
            normalized_time = output_start + output_index / output_fps
            cam0_native_time = normalized_time - args.cam0_offset
            cam1_native_time = normalized_time - args.cam1_offset
            cam0_frame, cam0_index = cam0_reader.get(cam0_native_time)
            cam1_frame, cam1_index = cam1_reader.get(cam1_native_time)

            left = draw_camera_panel(
                "cam0",
                cam0_frame,
                cam0_index,
                cam0_info,
                normalized_time,
                cam0_tracks.get(cam0_index, []) if cam0_index is not None else [],
                cross_data,
                panel_width,
                args.panel_height,
            )
            right = draw_camera_panel(
                "cam1",
                cam1_frame,
                cam1_index,
                cam1_info,
                normalized_time,
                cam1_tracks.get(cam1_index, []) if cam1_index is not None else [],
                cross_data,
                panel_width,
                args.panel_height,
            )
            combined = np.hstack((left, right))
            cv2.line(
                combined,
                (panel_width, 0),
                (panel_width, args.panel_height - 1),
                (255, 255, 255),
                2,
            )
            writer.write(combined)
            if output_index and output_index % max(1, int(output_fps * 10)) == 0:
                print(
                    f"  wrote {output_index}/{output_frame_count} frames "
                    f"(normalized {normalized_time:.2f} s)"
                )
    finally:
        cam0_reader.close()
        cam1_reader.close()
        writer.release()

    print(f"Frames written: {output_frame_count}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
