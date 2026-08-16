"""Unit tests for static, projected, and kinematic control models."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = REPO_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from nfc_north_player_tracking.config import (  # noqa: E402
    DEFAULT_PARAMETERS,
    OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
    OWNERSHIP_MODE_INTENDED_RECEIVER,
)
from nfc_north_player_tracking.dominant_regions import (  # noqa: E402
    build_distance_surface,
    evaluate_distance_target,
    evaluate_kinematic_target,
    simulate_arrival_times,
    voronoi_polygons,
)


def polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


class DominantRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = (-10.0, 10.0, -5.0, 10.0)
        self.parameters = replace(
            DEFAULT_PARAMETERS,
            distance_grid_resolution_yards=1.0,
            kinematic_grid_resolution_yards=1.0,
            kinematic_grid_time_step_seconds=0.05,
            kinematic_target_time_step_seconds=0.01,
            kinematic_max_time_seconds=6.0,
        )
        self.names = ["Receiver", "Defender"]
        self.sides = ["OFFENSE", "DEFENSE"]

    def test_static_target_owner(self) -> None:
        sites = np.array([[0.0, 0.0], [8.0, 0.0]])
        result = evaluate_distance_target(
            model_key="static",
            model_label="Static Voronoi",
            target=(2.0, 0.0),
            sites=sites,
            names=self.names,
            sides=self.sides,
            ownership_mode=OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
            intended_receiver_name="Receiver",
            near_boundary_threshold_yards=0.25,
        )
        self.assertTrue(result.target_in_offense_control)
        self.assertEqual(result.owner_name, "Receiver")
        self.assertAlmostEqual(result.control_margin, 4.0, places=7)

    def test_velocity_projection_can_change_owner(self) -> None:
        positions = np.array([[0.0, 0.0], [6.0, 0.0]])
        velocities = np.array([[6.0, 0.0], [0.0, 0.0]])
        target = (4.0, 0.0)
        static = evaluate_distance_target(
            "static",
            "Static Voronoi",
            target,
            positions,
            self.names,
            self.sides,
            OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
            "Receiver",
            0.25,
        )
        projected_sites = positions + 0.5 * velocities
        projected = evaluate_distance_target(
            "velocity",
            "Velocity-projected Voronoi",
            target,
            projected_sites,
            self.names,
            self.sides,
            OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
            "Receiver",
            0.25,
        )
        self.assertFalse(static.target_in_offense_control)
        self.assertTrue(projected.target_in_offense_control)
        self.assertEqual(projected.owner_name, "Receiver")

    def test_intended_receiver_mode_excludes_other_route_runners(self) -> None:
        sites = np.array([[8.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        names = ["Intended", "Other Receiver", "Defender"]
        sides = ["OFFENSE", "OFFENSE", "DEFENSE"]
        result = evaluate_distance_target(
            model_key="static",
            model_label="Static Voronoi",
            target=(0.0, 0.0),
            sites=sites,
            names=names,
            sides=sides,
            ownership_mode=OWNERSHIP_MODE_INTENDED_RECEIVER,
            intended_receiver_name="Intended",
            near_boundary_threshold_yards=0.25,
        )
        self.assertFalse(result.target_in_offense_control)
        self.assertEqual(result.owner_name, "Defender")
        self.assertEqual(result.offense_reference_name, "Intended")

    def test_kinematic_model_rewards_velocity_toward_target(self) -> None:
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])
        velocities = np.array([[4.0, 0.0], [-4.0, 0.0]])
        result = evaluate_kinematic_target(
            target=(7.0, 0.0),
            positions=positions,
            velocities=velocities,
            names=self.names,
            sides=self.sides,
            ownership_mode=OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
            intended_receiver_name="Receiver",
            parameters=self.parameters,
        )
        self.assertTrue(result.target_in_offense_control)
        self.assertLess(result.offense_value, result.defense_value)
        self.assertGreater(result.control_margin, 0.0)

    def test_stationary_symmetric_arrival_times_match(self) -> None:
        target = np.array([[0.0, 0.0]])
        left_time = simulate_arrival_times(
            np.array([-3.0, 0.0]),
            np.array([0.0, 0.0]),
            target,
            self.parameters.max_player_speed_yards_per_second,
            self.parameters.max_player_acceleration_yards_per_second_squared,
            self.parameters.kinematic_target_time_step_seconds,
            self.parameters.kinematic_max_time_seconds,
            self.parameters.kinematic_target_arrival_radius_yards,
        )[0]
        right_time = simulate_arrival_times(
            np.array([3.0, 0.0]),
            np.array([0.0, 0.0]),
            target,
            self.parameters.max_player_speed_yards_per_second,
            self.parameters.max_player_acceleration_yards_per_second_squared,
            self.parameters.kinematic_target_time_step_seconds,
            self.parameters.kinematic_max_time_seconds,
            self.parameters.kinematic_target_arrival_radius_yards,
        )[0]
        self.assertAlmostEqual(left_time, right_time, places=7)

    def test_voronoi_polygons_cover_rectangle(self) -> None:
        sites = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 6.0]])
        cells = voronoi_polygons(sites, self.bounds)
        rectangle_area = (
            (self.bounds[1] - self.bounds[0])
            * (self.bounds[3] - self.bounds[2])
        )
        self.assertAlmostEqual(
            sum(polygon_area(cell) for cell in cells),
            rectangle_area,
            places=6,
        )

    def test_distance_surface_has_valid_owners(self) -> None:
        sites = np.array([[0.0, 0.0], [8.0, 0.0]])
        surface = build_distance_surface(
            "static",
            "Static Voronoi",
            sites,
            self.names,
            self.sides,
            self.bounds,
            1.0,
        )
        self.assertGreater(surface.owner_index.size, 0)
        self.assertGreaterEqual(int(surface.owner_index.min()), 0)
        self.assertLess(int(surface.owner_index.max()), len(sites))


if __name__ == "__main__":
    unittest.main()
