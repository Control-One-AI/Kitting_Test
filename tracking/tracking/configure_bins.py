#!/usr/bin/env python3
"""Interactively map camera polygons to global physical BIN_1..BIN_4 IDs."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from bin_geometry import BIN_COLORS, PHYSICAL_BIN_IDS, empty_bin_config, load_bin_config


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw camera-specific polygons that map to global physical bin IDs."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--camera", required=True, help="Camera ID, e.g. cam0")
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "configs" / "bins.json",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Representative zero-based video frame.",
    )
    parser.add_argument(
        "--zone",
        choices=("occupancy", "entry"),
        default="occupancy",
        help="Draw the bin occupancy polygon or optional entry/mouth polygon.",
    )
    args = parser.parse_args()
    if args.frame < 0:
        parser.error("--frame must be zero or greater")
    if not args.camera.strip():
        parser.error("--camera must be a non-empty identifier")
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


def load_raw_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return empty_bin_config()
    load_bin_config(path)
    return json.loads(path.read_text(encoding="utf-8"))


def draw_polygon(
    canvas: np.ndarray,
    points: list[list[int]],
    color: tuple[int, int, int],
    *,
    closed: bool,
    thickness: int,
) -> None:
    if not points:
        return
    polygon = np.asarray(points, dtype=np.int32)
    cv2.polylines(canvas, [polygon], closed, color, thickness, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    camera_id = args.camera.strip()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    data = load_raw_config(config_path)
    cameras = data.setdefault("cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError("cameras must be a JSON object")
    camera_raw = cameras.setdefault(camera_id, {})
    if not isinstance(camera_raw, dict):
        raise ValueError(f"cameras.{camera_id} must be a JSON object")
    zone_key = (
        "occupancy_polygon" if args.zone == "occupancy" else "entry_polygon"
    )
    committed: dict[str, list[list[int]]] = {}
    for bin_id, raw_value in camera_raw.items():
        if isinstance(raw_value, dict):
            points = raw_value.get(zone_key, [])
        elif args.zone == "occupancy":
            points = raw_value
        else:
            points = []
        committed[bin_id] = [[int(x), int(y)] for x, y in points]

    frame = read_frame(source, args.frame)
    selected_index = 0
    selected_bin = PHYSICAL_BIN_IDS[selected_index]
    draft = [point[:] for point in committed.get(selected_bin, [])]
    confirm_save = False
    window = (
        f"Configure {args.zone} zones: {camera_id} | {config_path.name}"
    )

    def commit_draft() -> bool:
        nonlocal draft
        if len(draft) < 3:
            print(f"{selected_bin} needs at least 3 points", flush=True)
            return False
        committed[selected_bin] = [point[:] for point in draft]
        print(
            f"Committed {camera_id} {selected_bin} {args.zone}: "
            f"{committed[selected_bin]}",
            flush=True,
        )
        return True

    def select_bin(index: int) -> None:
        nonlocal selected_index, selected_bin, draft
        existing = committed.get(selected_bin)
        if draft != (existing or []):
            if len(draft) >= 3:
                commit_draft()
            elif draft:
                print(
                    f"Discarded incomplete edits for {selected_bin}", flush=True
                )
        selected_index = index
        selected_bin = PHYSICAL_BIN_IDS[selected_index]
        draft = [point[:] for point in committed.get(selected_bin, [])]
        print(f"Selected {selected_bin}", flush=True)

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and not confirm_save:
            draft.append([x, y])
            print(
                f"{camera_id} {selected_bin} point {len(draft)}: [{x}, {y}]",
                flush=True,
            )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    save_requested = False
    while True:
        canvas = frame.copy()
        overlay = frame.copy()
        for bin_id in PHYSICAL_BIN_IDS:
            points = (
                draft if bin_id == selected_bin and not confirm_save
                else committed.get(bin_id, [])
            )
            if len(points) < 3:
                continue
            polygon = np.asarray(points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], BIN_COLORS[bin_id], cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0.0)

        if args.zone == "entry":
            for bin_id, raw_value in camera_raw.items():
                occupancy = (
                    raw_value.get("occupancy_polygon", [])
                    if isinstance(raw_value, dict)
                    else raw_value
                )
                if len(occupancy) >= 3:
                    cv2.polylines(
                        canvas,
                        [np.asarray(occupancy, dtype=np.int32)],
                        True,
                        BIN_COLORS[bin_id],
                        1,
                        cv2.LINE_AA,
                    )

        for bin_id in PHYSICAL_BIN_IDS:
            points = (
                draft if bin_id == selected_bin and not confirm_save
                else committed.get(bin_id, [])
            )
            draw_polygon(
                canvas,
                points,
                BIN_COLORS[bin_id],
                closed=len(points) >= 3,
                thickness=4 if bin_id == selected_bin else 2,
            )
            if len(points) >= 3:
                polygon = np.asarray(points, dtype=np.int32)
                anchor = tuple(polygon[np.argmin(polygon[:, 1])].tolist())
                cv2.putText(
                    canvas,
                    f"{bin_id} {args.zone.upper()}",
                    (max(5, anchor[0]), max(25, anchor[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    BIN_COLORS[bin_id],
                    2,
                    cv2.LINE_AA,
                )

        if confirm_save:
            instructions = [
                f"Review {camera_id}: {', '.join(sorted(committed)) or 'no bins'}",
                "Y/Enter: WRITE CONFIG | N: return to edit | Esc: cancel",
            ]
        else:
            instructions = [
                f"{camera_id} frame {args.frame} | Selected: {selected_bin} "
                f"{args.zone.upper()}",
                "Keys 1-4: select global BIN ID | Left click: add point",
                "Enter: commit polygon | U: undo | R: reset | D: delete",
                "S: review/save config | Esc: cancel without writing",
            ]
        for index, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (18, 32 + index * 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF

        if confirm_save:
            if key in (ord("y"), 13):
                save_requested = True
                break
            if key == ord("n"):
                confirm_save = False
            elif key == 27:
                break
            continue

        if key == 27:
            break
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            select_bin(key - ord("1"))
        elif key in (ord("u"), 8) and draft:
            draft.pop()
        elif key == ord("r"):
            draft.clear()
        elif key == ord("d"):
            committed.pop(selected_bin, None)
            draft.clear()
            print(f"Deleted {camera_id} {selected_bin}", flush=True)
        elif key in (13, 32):
            commit_draft()
        elif key == ord("s"):
            existing = committed.get(selected_bin)
            if draft != (existing or []) and len(draft) >= 3:
                commit_draft()
            confirm_save = True

    cv2.destroyAllWindows()
    if not save_requested:
        print("Cancelled; bins.json was not changed.")
        return

    updated_camera = dict(camera_raw)
    for bin_id in PHYSICAL_BIN_IDS:
        existing = updated_camera.get(bin_id)
        points = committed.get(bin_id)
        if args.zone == "occupancy":
            if points is None:
                updated_camera.pop(bin_id, None)
            elif isinstance(existing, dict):
                updated = dict(existing)
                updated["occupancy_polygon"] = points
                updated_camera[bin_id] = updated
            else:
                updated_camera[bin_id] = points
        else:
            if existing is None:
                if points is not None:
                    raise ValueError(
                        f"Draw {bin_id} occupancy before its entry zone"
                    )
                continue
            occupancy = (
                existing.get("occupancy_polygon")
                if isinstance(existing, dict)
                else existing
            )
            updated = dict(existing) if isinstance(existing, dict) else {
                "occupancy_polygon": occupancy
            }
            if points is None:
                updated.pop("entry_polygon", None)
            else:
                updated["entry_polygon"] = points
            updated_camera[bin_id] = updated
    cameras[camera_id] = updated_camera
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = config_path.with_name(
            f"{config_path.stem}.backup-{stamp}{config_path.suffix}"
        )
        shutil.copy2(config_path, backup)
        print(f"Backup: {backup}")
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    load_bin_config(config_path)
    print(f"Saved global bin config: {config_path}")
    for bin_id, value in cameras[camera_id].items():
        print(f"{camera_id} {bin_id}: {value}")


if __name__ == "__main__":
    main()
