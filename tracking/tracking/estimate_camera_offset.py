#!/usr/bin/env python3
"""Calculate manual timestamp offsets for two independently recorded cameras."""

from __future__ import annotations

import argparse


def calculate_offsets(
    cam0_event_time: float,
    cam1_event_time: float,
    cam0_offset: float = 0.0,
) -> tuple[float, float]:
    """Return offsets that place the same event on one normalized timestamp.

    Normalized time is defined as ``video_time + camera_offset``.  Cam0's
    offset is selected by the caller (normally zero), and Cam1's offset is
    calculated so both reference-event timestamps normalize to the same time.
    """

    cam1_offset = cam0_event_time + cam0_offset - cam1_event_time
    return cam0_offset, cam1_offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate camera offsets from timestamps of the same physical "
            "event. Normalized time = video timestamp + camera offset."
        )
    )
    parser.add_argument(
        "--cam0-time",
        type=float,
        required=True,
        help="Cam0 native timestamp of the reference event, in seconds.",
    )
    parser.add_argument(
        "--cam1-time",
        type=float,
        required=True,
        help="Cam1 native timestamp of the same reference event, in seconds.",
    )
    parser.add_argument(
        "--cam0-offset",
        type=float,
        default=0.0,
        help="Chosen Cam0 offset (default: 0.0 seconds).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cam0_offset, cam1_offset = calculate_offsets(
        args.cam0_time, args.cam1_time, args.cam0_offset
    )
    normalized_event_time = args.cam0_time + cam0_offset
    relative_offset = cam1_offset - cam0_offset

    print("Manual camera-offset calibration")
    print(f"Cam0 event video time:       {args.cam0_time:.3f} s")
    print(f"Cam1 event video time:       {args.cam1_time:.3f} s")
    print(f"Cam1 offset - Cam0 offset:   {relative_offset:+.3f} s")
    print(f"Normalized reference time:   {normalized_event_time:.3f} s")
    print()
    print("Use these arguments:")
    print(f"  --cam0-offset {cam0_offset:.6f} --cam1-offset {cam1_offset:.6f}")
    print()
    print("Verification:")
    print(
        f"  Cam0: {args.cam0_time:.3f} + ({cam0_offset:+.3f}) "
        f"= {args.cam0_time + cam0_offset:.3f} s"
    )
    print(
        f"  Cam1: {args.cam1_time:.3f} + ({cam1_offset:+.3f}) "
        f"= {args.cam1_time + cam1_offset:.3f} s"
    )


if __name__ == "__main__":
    main()
