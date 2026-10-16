#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bin_geometry import (
    load_bin_config,
    obb_bin_overlap,
    point_bin_memberships,
    point_in_bin_entry_zone,
    point_in_bin,
)


class BinGeometryTests(unittest.TestCase):
    def write_config(self, data: dict[str, object]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "bins.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @staticmethod
    def base_config() -> dict[str, object]:
        return {
            "physical_bins": ["BIN_1", "BIN_2", "BIN_3", "BIN_4"],
            "cameras": {
                "cam0": {
                    "BIN_3": [[0, 0], [100, 0], [100, 100], [0, 100]]
                },
                "cam1": {
                    "BIN_3": [[200, 0], [300, 0], [300, 100], [200, 100]]
                },
            },
        }

    def test_shared_camera_polygons_return_same_physical_bin(self) -> None:
        geometry = load_bin_config(self.write_config(self.base_config()))
        self.assertEqual(point_in_bin(geometry, "cam0", (50, 50)), "BIN_3")
        self.assertEqual(point_in_bin(geometry, "cam1", (250, 50)), "BIN_3")
        self.assertIsNot(
            geometry.polygons_for_camera("cam0")["BIN_3"],
            geometry.polygons_for_camera("cam1")["BIN_3"],
        )

    def test_point_outside_or_ambiguous_returns_none(self) -> None:
        data = self.base_config()
        data["cameras"]["cam0"].update(
            {
                "BIN_1": [[0, 0], [50, 0], [50, 50], [0, 50]],
                "BIN_2": [[25, 0], [75, 0], [75, 50], [25, 50]],
            }
        )
        geometry = load_bin_config(self.write_config(data))
        self.assertEqual(
            point_bin_memberships(geometry, "cam0", (30, 20)),
            ("BIN_1", "BIN_2", "BIN_3"),
        )
        self.assertIsNone(point_in_bin(geometry, "cam0", (30, 20)))
        self.assertIsNone(point_in_bin(geometry, "cam0", (500, 500)))

    def test_obb_overlap_reports_global_bin_and_ratio(self) -> None:
        geometry = load_bin_config(self.write_config(self.base_config()))
        obb = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32)
        memberships = obb_bin_overlap(geometry, "cam0", obb)
        self.assertEqual(memberships[0].physical_bin_id, "BIN_3")
        self.assertAlmostEqual(memberships[0].overlap_ratio, 1.0)
        self.assertTrue(memberships[0].centroid_inside)

    def test_unknown_camera_has_no_membership(self) -> None:
        geometry = load_bin_config(self.write_config(self.base_config()))
        self.assertIsNone(point_in_bin(geometry, "cam9", (50, 50)))
        self.assertEqual(
            obb_bin_overlap(
                geometry,
                "cam9",
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
            ),
            (),
        )

    def test_optional_nested_entry_polygon_preserves_occupancy(self) -> None:
        data = self.base_config()
        data["cameras"]["cam0"]["BIN_3"] = {
            "occupancy_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "entry_polygon": [[100, 20], [130, 20], [130, 80], [100, 80]],
        }
        geometry = load_bin_config(self.write_config(data))
        self.assertEqual(point_in_bin(geometry, "cam0", (50, 50)), "BIN_3")
        self.assertTrue(
            point_in_bin_entry_zone(geometry, "cam0", "BIN_3", (115, 50))
        )
        self.assertFalse(
            point_in_bin_entry_zone(geometry, "cam0", "BIN_3", (50, 50))
        )

    def test_unknown_physical_bin_is_rejected(self) -> None:
        data = self.base_config()
        data["cameras"]["cam0"]["CAM0_BIN_3"] = [
            [0, 0],
            [10, 0],
            [10, 10],
        ]
        with self.assertRaisesRegex(ValueError, "unknown physical bin"):
            load_bin_config(self.write_config(data))


if __name__ == "__main__":
    unittest.main()
