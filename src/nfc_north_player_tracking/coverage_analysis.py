"""Orchestrate the three play-level coverage-control analyses."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import numpy as np
import polars as pl

from nfc_north_player_tracking.config import (
    FIELD_HALF_WIDTH_YARDS,
    OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
    CoverageParameters,
    DEFAULT_PARAMETERS,
)
from nfc_north_player_tracking.coverage_diagnostics import (
    build_coverage_diagnostics,
)
from nfc_north_player_tracking.coverage_queries import (
    intended_receiver_name,
    load_defense_play,
    load_offense_play,
    load_pbp_play,
    load_tracking_play,
    pass_target,
    passer_name,
    regular_pass_exclusion_reasons,
)
from nfc_north_player_tracking.dominant_regions import (
    ControlSurface,
    TargetControlResult,
    build_distance_surface,
    build_kinematic_surface,
    evaluate_distance_target,
    evaluate_kinematic_target,
)
from nfc_north_player_tracking.kinematics import (
    ball_row_at_time,
    build_release_state,
    context_release_state,
    event_time,
    optional_event_time,
    pass_outcome_event,
)
from nfc_north_player_tracking.player_resolution import (
    require_resolved_generators,
    resolve_context_player,
    resolve_generators,
)


@dataclass(frozen=True)
class CoverageAnalysisResult:
    """Complete play-level inputs, surfaces, diagnostics, and target metrics."""

    game_id: str
    play_id: str
    pbp_play: pl.DataFrame
    generator_audit: pl.DataFrame
    release_state: pl.DataFrame
    diagnostics: pl.DataFrame
    static_surface: ControlSurface
    velocity_surface: ControlSurface
    kinematic_surface: ControlSurface
    static_target: TargetControlResult
    velocity_target: TargetControlResult
    kinematic_target: TargetControlResult
    target: tuple[float, float]
    context: dict[str, object]
    plot_bounds: tuple[float, float, float, float]
    intended_receiver: str | None
    snap_time: float
    release_time: float
    arrival_time: float | None
    outcome_event: str | None
    outcome_time: float | None
    exclusion_reasons: tuple[str, ...]
    ownership_mode: str
    parameters: CoverageParameters

    @property
    def surfaces(self) -> tuple[ControlSurface, ControlSurface, ControlSurface]:
        return (
            self.static_surface,
            self.velocity_surface,
            self.kinematic_surface,
        )

    @property
    def target_results(
        self,
    ) -> tuple[TargetControlResult, TargetControlResult, TargetControlResult]:
        return (
            self.static_target,
            self.velocity_target,
            self.kinematic_target,
        )

    def result_frame(self) -> pl.DataFrame:
        """Return one tidy result row suitable for CSV output."""
        pbp = self.pbp_play.row(0, named=True)
        record: dict[str, object] = {
            "pff_GAMEID": self.game_id,
            "pff_PLAYID": self.play_id,
            "pff_GSISGAMEKEY": pbp.get("pff_GSISGAMEKEY"),
            "pff_GSISPLAYID": pbp.get("pff_GSISPLAYID"),
            "pff_GAMEDATE": pbp.get("pff_GAMEDATE"),
            "pff_WEEK": pbp.get("pff_WEEK"),
            "pff_OFFTEAM": pbp.get("pff_OFFTEAM"),
            "pff_DEFTEAM": pbp.get("pff_DEFTEAM"),
            "pff_PASSRESULT": pbp.get("pff_PASSRESULT"),
            "pff_PASSDEPTH": pbp.get("pff_PASSDEPTH"),
            "pff_PASSWIDTH": pbp.get("pff_PASSWIDTH"),
            "pff_PASSDIRECTION": pbp.get("pff_PASSDIRECTION"),
            "target_x": self.target[0],
            "target_y": self.target[1],
            "intended_receiver_name": self.intended_receiver,
            "ownership_mode": self.ownership_mode,
            "snap_time": self.snap_time,
            "release_time": self.release_time,
            "pass_arrived_time": self.arrival_time,
            "pass_outcome_event": self.outcome_event,
            "pass_outcome_time": self.outcome_time,
            "route_runners": self.release_state.filter(
                pl.col("side") == "OFFENSE"
            ).height,
            "coverage_defenders": self.release_state.filter(
                pl.col("side") == "DEFENSE"
            ).height,
            "regular_pass_eligible": len(self.exclusion_reasons) == 0,
            "regular_pass_exclusion_reasons": ";".join(self.exclusion_reasons),
            "velocity_projection_horizon_seconds": (
                self.parameters.velocity_projection_horizon_seconds
            ),
            "velocity_lookback_seconds": (
                self.parameters.velocity_lookback_seconds
            ),
            "max_player_speed_yards_per_second": (
                self.parameters.max_player_speed_yards_per_second
            ),
            "max_player_acceleration_yards_per_second_squared": (
                self.parameters.max_player_acceleration_yards_per_second_squared
            ),
            "kinematic_grid_resolution_yards": (
                self.parameters.kinematic_grid_resolution_yards
            ),
            "kinematic_grid_time_step_seconds": (
                self.parameters.kinematic_grid_time_step_seconds
            ),
        }
        for target_result in self.target_results:
            record.update(target_result.to_dict())

        record.update(
            {
                "kinematic_offense_arrival_time_s": (
                    self.kinematic_target.offense_value
                ),
                "kinematic_defense_arrival_time_s": (
                    self.kinematic_target.defense_value
                ),
                "kinematic_target_control_margin_s": (
                    self.kinematic_target.control_margin
                ),
            }
        )
        return pl.DataFrame([record])


def _rounded_down(value: float, increment: float = 5.0) -> float:
    return floor(value / increment) * increment


def _rounded_up(value: float, increment: float = 5.0) -> float:
    return ceil(value / increment) * increment


def determine_plot_bounds(
    release_state: pl.DataFrame,
    target: tuple[float, float],
    pbp_play: pl.DataFrame,
    parameters: CoverageParameters,
) -> tuple[float, float, float, float]:
    """Choose consistent field bounds for all three model panels."""
    release_min = float(release_state.get_column("release_y").min())
    release_max = float(release_state.get_column("release_y").max())
    target_y = float(target[1])

    y_min = min(
        parameters.minimum_plot_depth_yards,
        _rounded_down(release_min - 2.0),
    )
    desired_max = max(
        parameters.minimum_plot_max_depth_yards,
        target_y + parameters.plot_depth_padding_yards,
        release_max + parameters.plot_depth_padding_yards,
    )

    pbp = pbp_play.row(0, named=True)
    yards_to_goal = pbp.get("pff_YARDSTOGOALLINE")
    physical_max = parameters.maximum_plot_depth_yards
    if yards_to_goal is not None and float(yards_to_goal) >= 0:
        physical_max = min(physical_max, float(yards_to_goal) + 10.0)

    y_max = min(parameters.maximum_plot_depth_yards, physical_max)
    y_max = min(y_max, _rounded_up(desired_max))
    if y_max <= y_min + 10.0:
        y_max = min(parameters.maximum_plot_depth_yards, y_min + 15.0)

    return (-FIELD_HALF_WIDTH_YARDS, FIELD_HALF_WIDTH_YARDS, y_min, y_max)


def _release_arrays(
    release_state: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    positions = np.column_stack(
        [
            release_state.get_column("release_x").to_numpy(),
            release_state.get_column("release_y").to_numpy(),
        ]
    ).astype(float)
    velocities = np.column_stack(
        [
            release_state
            .get_column("velocity_x_yards_per_second")
            .to_numpy(),
            release_state
            .get_column("velocity_y_yards_per_second")
            .to_numpy(),
        ]
    ).astype(float)
    names = [str(value) for value in release_state.get_column("pff_PLAYERNAME")]
    sides = [str(value) for value in release_state.get_column("side")]
    return positions, velocities, names, sides


def analyze_coverage_play(
    game_id: object,
    play_id: object,
    pbp_csv: Path,
    offense_csv: Path,
    defense_csv: Path,
    tracking_csv: Path,
    parameters: CoverageParameters = DEFAULT_PARAMETERS,
    ownership_mode: str = OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
    allow_ineligible: bool = False,
) -> CoverageAnalysisResult:
    """Run all three coverage-control models for one PFF game/play key."""
    parameters.validate()
    game_id = str(game_id).strip().removesuffix(".0")
    play_id = str(play_id).strip().removesuffix(".0")

    pbp_play = load_pbp_play(pbp_csv, game_id, play_id)
    offense_rows = load_offense_play(offense_csv, game_id, play_id)
    defense_rows = load_defense_play(defense_csv, game_id, play_id)
    tracking = load_tracking_play(tracking_csv, pbp_play)
    if tracking.is_empty():
        raise ValueError(f"No tracking rows matched play {game_id}:{play_id}.")

    exclusions = regular_pass_exclusion_reasons(pbp_play, defense_rows)
    if exclusions and not allow_ineligible:
        raise ValueError(
            f"Play {game_id}:{play_id} is outside the regular-pass scope: "
            + ";".join(exclusions)
        )

    generator_audit = resolve_generators(offense_rows, defense_rows, tracking)
    resolved_generators = require_resolved_generators(generator_audit)

    snap_time = event_time(tracking, "ball_snap")
    release_time = event_time(tracking, "pass_forward")
    arrival_time = optional_event_time(tracking, "pass_arrived")
    outcome_event_name, outcome_time = pass_outcome_event(tracking)

    release_state = build_release_state(
        tracking,
        resolved_generators,
        release_time,
        parameters,
    )
    target = pass_target(pbp_play)
    intended = intended_receiver_name(offense_rows)

    passer = passer_name(offense_rows)
    resolved_passer = resolve_context_player(tracking, passer)
    passer_tracking_id = (
        None if resolved_passer is None else str(resolved_passer["pro_player_id"])
    )
    context = context_release_state(
        tracking,
        release_time,
        passer_tracking_id,
    )
    if arrival_time is not None:
        arrival_ball = ball_row_at_time(tracking, arrival_time)
        context["arrival_ball_x"] = float(arrival_ball["X"])
        context["arrival_ball_y"] = float(arrival_ball["Y"])
    else:
        context["arrival_ball_x"] = None
        context["arrival_ball_y"] = None

    if outcome_time is not None:
        outcome_ball = ball_row_at_time(tracking, outcome_time)
        context["outcome_event"] = outcome_event_name
        context["outcome_ball_x"] = float(outcome_ball["X"])
        context["outcome_ball_y"] = float(outcome_ball["Y"])
    else:
        context["outcome_event"] = None
        context["outcome_ball_x"] = None
        context["outcome_ball_y"] = None

    bounds = determine_plot_bounds(release_state, target, pbp_play, parameters)
    positions, velocities, names, sides = _release_arrays(release_state)
    projected_positions = (
        positions
        + parameters.velocity_projection_horizon_seconds * velocities
    )

    static_surface = build_distance_surface(
        model_key="static",
        model_label="Static Voronoi",
        sites=positions,
        names=names,
        sides=sides,
        bounds=bounds,
        resolution=parameters.distance_grid_resolution_yards,
    )
    static_target = evaluate_distance_target(
        model_key="static",
        model_label="Static Voronoi",
        target=target,
        sites=positions,
        names=names,
        sides=sides,
        ownership_mode=ownership_mode,
        intended_receiver_name=intended,
        near_boundary_threshold_yards=(
            parameters.distance_near_boundary_margin_yards
        ),
    )

    velocity_surface = build_distance_surface(
        model_key="velocity",
        model_label="Velocity-projected Voronoi",
        sites=projected_positions,
        names=names,
        sides=sides,
        bounds=bounds,
        resolution=parameters.distance_grid_resolution_yards,
    )
    velocity_target = evaluate_distance_target(
        model_key="velocity",
        model_label="Velocity-projected Voronoi",
        target=target,
        sites=projected_positions,
        names=names,
        sides=sides,
        ownership_mode=ownership_mode,
        intended_receiver_name=intended,
        near_boundary_threshold_yards=(
            parameters.distance_near_boundary_margin_yards
        ),
    )

    kinematic_surface = build_kinematic_surface(
        positions=positions,
        velocities=velocities,
        names=names,
        sides=sides,
        bounds=bounds,
        parameters=parameters,
    )
    kinematic_target = evaluate_kinematic_target(
        target=target,
        positions=positions,
        velocities=velocities,
        names=names,
        sides=sides,
        ownership_mode=ownership_mode,
        intended_receiver_name=intended,
        parameters=parameters,
    )

    diagnostics = build_coverage_diagnostics(
        exclusion_reasons=exclusions,
        pbp_play=pbp_play,
        tracking=tracking,
        generator_audit=generator_audit,
        release_state=release_state,
        snap_time=snap_time,
        release_time=release_time,
        target=target,
        plot_bounds=bounds,
        surfaces=[static_surface, velocity_surface, kinematic_surface],
        target_results=[static_target, velocity_target, kinematic_target],
        parameters=parameters,
        allow_ineligible=allow_ineligible,
    )

    return CoverageAnalysisResult(
        game_id=game_id,
        play_id=play_id,
        pbp_play=pbp_play,
        generator_audit=generator_audit,
        release_state=release_state,
        diagnostics=diagnostics,
        static_surface=static_surface,
        velocity_surface=velocity_surface,
        kinematic_surface=kinematic_surface,
        static_target=static_target,
        velocity_target=velocity_target,
        kinematic_target=kinematic_target,
        target=target,
        context=context,
        plot_bounds=bounds,
        intended_receiver=intended,
        snap_time=snap_time,
        release_time=release_time,
        arrival_time=arrival_time,
        outcome_event=outcome_event_name,
        outcome_time=outcome_time,
        exclusion_reasons=tuple(exclusions),
        ownership_mode=ownership_mode.upper(),
        parameters=parameters,
    )
