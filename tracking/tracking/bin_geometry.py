"""Global physical-bin geometry shared across camera-specific views.

This module deliberately contains no tracking, journey, or expected-bin logic.
Each camera polygon is only an observation of a globally named physical bin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


PHYSICAL_BIN_IDS = ("BIN_1", "BIN_2", "BIN_3", "BIN_4")
BIN_COLORS: dict[str, tuple[int, int, int]] = {
    "BIN_1": (80, 220, 80),
    "BIN_2": (80, 80, 255),
    "BIN_3": (255, 210, 50),
    "BIN_4": (220, 80, 220),
}


@dataclass(frozen=True)
class BinOverlap:
    """Membership evidence for one globally named physical bin."""

    physical_bin_id: str
    overlap_ratio: float
    intersection_area_pixels: int
    obb_area_pixels: int
    centroid_inside: bool


@dataclass(frozen=True)
class BinGeometry:
    """Validated physical-bin IDs and per-camera polygon observations."""

    physical_bins: tuple[str, ...]
    cameras: Mapping[str, Mapping[str, np.ndarray]]
    entry_zones: Mapping[str, Mapping[str, np.ndarray]] = field(
        default_factory=dict
    )

    def polygons_for_camera(self, camera_id: str) -> Mapping[str, np.ndarray]:
        return self.cameras.get(camera_id, {})

    def entry_polygons_for_camera(
        self, camera_id: str
    ) -> Mapping[str, np.ndarray]:
        return self.entry_zones.get(camera_id, {})


def empty_bin_config() -> dict[str, object]:
    """Return the canonical four-bin schema without invented coordinates."""

    return {
        "physical_bins": list(PHYSICAL_BIN_IDS),
        "cameras": {"cam0": {}, "cam1": {}},
    }


def load_bin_config(path: Path) -> BinGeometry:
    """Load and validate globally named bins and camera polygon mappings."""

    data = json.loads(path.read_text(encoding="utf-8"))
    physical_bins_raw = data.get("physical_bins")
    if not isinstance(physical_bins_raw, list) or not physical_bins_raw:
        raise ValueError("physical_bins must be a non-empty JSON list")
    physical_bins = tuple(str(value) for value in physical_bins_raw)
    if len(set(physical_bins)) != len(physical_bins):
        raise ValueError("physical_bins contains duplicate IDs")
    if set(physical_bins) != set(PHYSICAL_BIN_IDS):
        raise ValueError(
            "physical_bins must define exactly BIN_1, BIN_2, BIN_3, BIN_4"
        )

    cameras_raw = data.get("cameras")
    if not isinstance(cameras_raw, dict):
        raise ValueError("cameras must be a JSON object")
    cameras: dict[str, dict[str, np.ndarray]] = {}
    entry_zones: dict[str, dict[str, np.ndarray]] = {}
    for camera_id, camera_bins_raw in cameras_raw.items():
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera IDs must be non-empty strings")
        if not isinstance(camera_bins_raw, dict):
            raise ValueError(f"cameras.{camera_id} must be a JSON object")
        camera_bins: dict[str, np.ndarray] = {}
        camera_entries: dict[str, np.ndarray] = {}
        for physical_bin_id, points_raw in camera_bins_raw.items():
            if physical_bin_id not in physical_bins:
                raise ValueError(
                    f"cameras.{camera_id} uses unknown physical bin "
                    f"{physical_bin_id!r}"
                )
            entry_raw: object | None = None
            if isinstance(points_raw, dict):
                occupancy_raw = points_raw.get("occupancy_polygon")
                entry_raw = points_raw.get("entry_polygon")
            else:
                occupancy_raw = points_raw
            polygon = np.asarray(occupancy_raw, dtype=np.float32)
            if polygon.ndim != 2 or polygon.shape[1:] != (2,):
                raise ValueError(
                    f"cameras.{camera_id}.{physical_bin_id} must contain "
                    "[x, y] points"
                )
            if len(polygon) < 3:
                raise ValueError(
                    f"cameras.{camera_id}.{physical_bin_id} needs at least "
                    "three points"
                )
            if not np.isfinite(polygon).all():
                raise ValueError(
                    f"cameras.{camera_id}.{physical_bin_id} contains "
                    "non-finite coordinates"
                )
            if abs(float(cv2.contourArea(polygon))) <= 0.0:
                raise ValueError(
                    f"cameras.{camera_id}.{physical_bin_id} has zero area"
                )
            camera_bins[physical_bin_id] = polygon
            if entry_raw not in (None, []):
                entry_polygon = np.asarray(entry_raw, dtype=np.float32)
                if entry_polygon.ndim != 2 or entry_polygon.shape[1:] != (2,):
                    raise ValueError(
                        f"cameras.{camera_id}.{physical_bin_id}.entry_polygon "
                        "must contain [x, y] points"
                    )
                if len(entry_polygon) < 3:
                    raise ValueError(
                        f"cameras.{camera_id}.{physical_bin_id}.entry_polygon "
                        "needs at least three points"
                    )
                if not np.isfinite(entry_polygon).all():
                    raise ValueError(
                        f"cameras.{camera_id}.{physical_bin_id}.entry_polygon "
                        "contains non-finite coordinates"
                    )
                if abs(float(cv2.contourArea(entry_polygon))) <= 0.0:
                    raise ValueError(
                        f"cameras.{camera_id}.{physical_bin_id}.entry_polygon "
                        "has zero area"
                    )
                camera_entries[physical_bin_id] = entry_polygon
        cameras[camera_id] = camera_bins
        entry_zones[camera_id] = camera_entries
    return BinGeometry(
        physical_bins=physical_bins,
        cameras=cameras,
        entry_zones=entry_zones,
    )


def point_in_bin_entry_zone(
    geometry: BinGeometry,
    camera_id: str,
    bin_id: str,
    point: tuple[float, float],
) -> bool:
    """Return whether a point crosses an optional user-drawn bin mouth."""

    polygon = geometry.entry_polygons_for_camera(camera_id).get(bin_id)
    return bool(
        polygon is not None
        and cv2.pointPolygonTest(polygon, point, False) >= 0
    )


def point_bin_memberships(
    geometry: BinGeometry,
    camera_id: str,
    point: tuple[float, float],
) -> tuple[str, ...]:
    """Return every physical bin whose polygon contains the camera point."""

    matches = []
    camera_polygons = geometry.polygons_for_camera(camera_id)
    for physical_bin_id in geometry.physical_bins:
        polygon = camera_polygons.get(physical_bin_id)
        if polygon is None:
            continue
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            matches.append(physical_bin_id)
    return tuple(matches)


def point_in_bin(
    geometry: BinGeometry,
    camera_id: str,
    point: tuple[float, float],
) -> str | None:
    """Return one physical bin ID, or None when outside/ambiguous.

    Overlapping configured bin polygons are treated conservatively as
    ambiguous instead of silently choosing a camera-local identity.
    """

    matches = point_bin_memberships(geometry, camera_id, point)
    return matches[0] if len(matches) == 1 else None


def obb_bin_overlap(
    geometry: BinGeometry,
    camera_id: str,
    obb: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[BinOverlap, ...]:
    """Return per-bin OBB overlap evidence, highest overlap first.

    A small integer mask is used so concave operator-drawn polygons are
    supported without adding a geometry-library dependency.
    """

    obb_polygon = np.asarray(obb, dtype=np.float32)
    if obb_polygon.shape != (4, 2):
        raise ValueError(f"OBB must contain four [x, y] points; got {obb_polygon.shape}")
    if not np.isfinite(obb_polygon).all():
        raise ValueError("OBB contains non-finite coordinates")

    center = tuple(np.mean(obb_polygon, axis=0).tolist())
    memberships: list[BinOverlap] = []
    for physical_bin_id, bin_polygon in geometry.polygons_for_camera(
        camera_id
    ).items():
        all_points = np.vstack((obb_polygon, bin_polygon))
        minimum = np.floor(all_points.min(axis=0)).astype(int) - 1
        maximum = np.ceil(all_points.max(axis=0)).astype(int) + 1
        width, height = (maximum - minimum + 1).tolist()
        if width <= 0 or height <= 0:
            continue
        shifted_obb = np.rint(obb_polygon - minimum).astype(np.int32)
        shifted_bin = np.rint(bin_polygon - minimum).astype(np.int32)
        obb_mask = np.zeros((height, width), dtype=np.uint8)
        bin_mask = np.zeros_like(obb_mask)
        cv2.fillPoly(obb_mask, [shifted_obb], 1)
        cv2.fillPoly(bin_mask, [shifted_bin], 1)
        obb_area = int(np.count_nonzero(obb_mask))
        intersection_area = int(np.count_nonzero(obb_mask & bin_mask))
        overlap_ratio = intersection_area / obb_area if obb_area else 0.0
        memberships.append(
            BinOverlap(
                physical_bin_id=physical_bin_id,
                overlap_ratio=overlap_ratio,
                intersection_area_pixels=intersection_area,
                obb_area_pixels=obb_area,
                centroid_inside=(
                    cv2.pointPolygonTest(bin_polygon, center, False) >= 0
                ),
            )
        )
    return tuple(
        sorted(
            memberships,
            key=lambda value: (
                -value.overlap_ratio,
                geometry.physical_bins.index(value.physical_bin_id),
            ),
        )
    )
