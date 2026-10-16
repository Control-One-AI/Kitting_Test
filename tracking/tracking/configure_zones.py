#!/usr/bin/env python3
"""Interactively configure zone A, tracking, and zone B polygons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ZONE_ORDER = ("zone_a", "tracking_zone", "zone_b")
COLORS = {
    "zone_a": (80, 220, 80),
    "tracking_zone": (255, 210, 50),
    "zone_b": (80, 80, 255),
}
ZONE_DESCRIPTIONS = {
    "zone_a": "physical bin / endpoint A",
    "tracking_zone": "usable transport area around both bins",
    "zone_b": "physical bin / endpoint B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click lifecycle polygons on a representative video frame."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SCRIPT_DIR / "videos" / "cam1.mp4",
        help="Representative camera video (default: videos/cam1.mp4).",
    )
    parser.add_argument("--frame", type=int, default=260)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "configs" / "cam3_zones.json",
        help="Destination JSON (default: configs/cam3_zones.json).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frame < 0:
        parser.error("--frame must be zero or greater")
    return args


def read_frame(source: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {source}")
    return frame


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Zone config already exists: {output}\nUse --overwrite to replace it."
        )

    frame = read_frame(source, args.frame)
    polygons: dict[str, list[list[int]]] = {name: [] for name in ZONE_ORDER}
    zone_index = 0
    window = f"Configure lifecycle zones: {output.name}"

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and zone_index < len(ZONE_ORDER):
            zone_name = ZONE_ORDER[zone_index]
            polygons[zone_name].append([x, y])
            print(
                f"{zone_name} point {len(polygons[zone_name])}: [{x}, {y}]",
                flush=True,
            )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while zone_index < len(ZONE_ORDER):
        zone_name = ZONE_ORDER[zone_index]
        canvas = frame.copy()
        for name in ZONE_ORDER:
            points = polygons[name]
            if not points:
                continue
            array = np.asarray(points, dtype=np.int32)
            closed = name != zone_name and len(points) >= 3
            cv2.polylines(
                canvas, [array], closed, COLORS[name], 2, cv2.LINE_AA
            )
            for x, y in points:
                cv2.circle(canvas, (x, y), 4, COLORS[name], -1, cv2.LINE_AA)

        instructions = [
            f"Frame {args.frame} | Draw: {zone_name} | "
            f"{ZONE_DESCRIPTIONS[zone_name]}",
            f"Points selected: {len(polygons[zone_name])} (minimum 3)",
            "Left click: add point | U: undo | R: reset zone",
            "N/Space/Enter: finish zone | Esc: cancel",
        ]
        for index, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (20, 35 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("Cancelled; no zone file was written.")
            return
        if key in (ord("u"), 8) and polygons[zone_name]:
            polygons[zone_name].pop()
        elif key == ord("r"):
            polygons[zone_name].clear()
        elif key in (ord("n"), 13, 32):
            if len(polygons[zone_name]) < 3:
                print(f"{zone_name} needs at least 3 points")
                continue
            zone_index += 1

    # Show the complete calibration before allowing it to be written.
    while True:
        canvas = frame.copy()
        overlay = frame.copy()
        for name in ZONE_ORDER:
            array = np.asarray(polygons[name], dtype=np.int32)
            cv2.fillPoly(overlay, [array], COLORS[name], cv2.LINE_AA)
            cv2.polylines(canvas, [array], True, COLORS[name], 3, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0)
        cv2.putText(
            canvas,
            "Review all zones | S/Y/Space/Enter: SAVE | Esc: cancel",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("Cancelled after preview; no zone file was written.")
            return
        if key in (ord("s"), ord("y"), 13, 32):
            break

    cv2.destroyAllWindows()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(polygons, indent=2) + "\n", encoding="utf-8")
    if not output.is_file():
        raise RuntimeError(f"Zone config was not written: {output}")
    print(f"Saved zone config: {output}")
    for name in ZONE_ORDER:
        print(f"{name}: {polygons[name]}")


if __name__ == "__main__":
    main()
