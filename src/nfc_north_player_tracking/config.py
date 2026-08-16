"""Configuration for play-level coverage-control analysis.

Tracking positions are measured in yards and ``time_into_play`` is measured in
seconds. The first production model applies the same speed and acceleration
limits to every route runner and coverage defender.
"""

from __future__ import annotations

from dataclasses import dataclass


FIELD_HALF_WIDTH_YARDS = 26.65
PFF_PASS_WIDTH_CENTER = 26.5

RAW_LATERAL_COLUMN = "X"
RAW_LONGITUDINAL_COLUMN = "Y"

OFFENSE_ROUTE_ROLE = "PASS ROUTE"
DEFENSE_COVERAGE_ROLE = "COVERAGE"

DEFAULT_GAME_ID = "28430"
DEFAULT_PLAY_ID = "6408068"
DEFAULT_PLAY = f"{DEFAULT_GAME_ID}:{DEFAULT_PLAY_ID}"

OWNERSHIP_MODE_ANY_ROUTE_RUNNER = "ANY_ROUTE_RUNNER"
OWNERSHIP_MODE_INTENDED_RECEIVER = "INTENDED_RECEIVER_ONLY"
VALID_OWNERSHIP_MODES = {
    OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
    OWNERSHIP_MODE_INTENDED_RECEIVER,
}


@dataclass(frozen=True)
class CoverageParameters:
    """Shared parameters for the three coverage-control models."""

    velocity_projection_horizon_seconds: float = 0.50
    velocity_lookback_seconds: float = 0.40
    min_velocity_observations: int = 3

    max_player_speed_yards_per_second: float = 11.0
    max_player_acceleration_yards_per_second_squared: float = 7.0

    distance_grid_resolution_yards: float = 0.25
    kinematic_grid_resolution_yards: float = 0.25
    kinematic_grid_time_step_seconds: float = 0.05
    kinematic_target_time_step_seconds: float = 0.01
    kinematic_max_time_seconds: float = 9.0
    kinematic_target_arrival_radius_yards: float = 0.10

    distance_near_boundary_margin_yards: float = 0.25
    target_near_boundary_margin_seconds: float = 0.05

    minimum_plot_depth_yards: float = -10.0
    minimum_plot_max_depth_yards: float = 35.0
    maximum_plot_depth_yards: float = 60.0
    plot_depth_padding_yards: float = 10.0

    def validate(self) -> None:
        """Raise ``ValueError`` when a model parameter is unusable."""
        positive = {
            "velocity_projection_horizon_seconds": (
                self.velocity_projection_horizon_seconds
            ),
            "velocity_lookback_seconds": self.velocity_lookback_seconds,
            "max_player_speed_yards_per_second": (
                self.max_player_speed_yards_per_second
            ),
            "max_player_acceleration_yards_per_second_squared": (
                self.max_player_acceleration_yards_per_second_squared
            ),
            "distance_grid_resolution_yards": self.distance_grid_resolution_yards,
            "kinematic_grid_resolution_yards": (
                self.kinematic_grid_resolution_yards
            ),
            "kinematic_grid_time_step_seconds": (
                self.kinematic_grid_time_step_seconds
            ),
            "kinematic_target_time_step_seconds": (
                self.kinematic_target_time_step_seconds
            ),
            "kinematic_max_time_seconds": self.kinematic_max_time_seconds,
            "kinematic_target_arrival_radius_yards": (
                self.kinematic_target_arrival_radius_yards
            ),
            "distance_near_boundary_margin_yards": (
                self.distance_near_boundary_margin_yards
            ),
            "target_near_boundary_margin_seconds": (
                self.target_near_boundary_margin_seconds
            ),
            "plot_depth_padding_yards": self.plot_depth_padding_yards,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero; got {value}.")

        if self.min_velocity_observations < 2:
            raise ValueError("min_velocity_observations must be at least two.")
        if self.minimum_plot_depth_yards >= self.minimum_plot_max_depth_yards:
            raise ValueError("Plot minimum depth must be below plot maximum depth.")
        if self.minimum_plot_max_depth_yards > self.maximum_plot_depth_yards:
            raise ValueError("Minimum plot maximum exceeds the hard maximum.")

    @property
    def max_player_speed_miles_per_hour(self) -> float:
        """Return the speed ceiling in miles per hour."""
        return self.max_player_speed_yards_per_second * 3600.0 / 1760.0

    @property
    def max_player_acceleration_meters_per_second_squared(self) -> float:
        """Return the acceleration ceiling in meters per second squared."""
        return self.max_player_acceleration_yards_per_second_squared * 0.9144


DEFAULT_PARAMETERS = CoverageParameters()
DEFAULT_PARAMETERS.validate()
