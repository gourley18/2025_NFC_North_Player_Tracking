"""Reconciliation and plausibility checks for play-level coverage analysis."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from nfc_north_player_tracking.config import FIELD_HALF_WIDTH_YARDS, CoverageParameters
from nfc_north_player_tracking.dominant_regions import (
    ControlSurface,
    TargetControlResult,
)
from nfc_north_player_tracking.kinematics import is_ball_expr


def _check(
    name: str,
    status: str,
    count: int | float | None,
    details: str,
) -> dict[str, object]:
    return {
        "check": name,
        "status": status,
        "count": count,
        "details": details,
    }


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


def _normalize_direction(value: object) -> str:
    token = re.sub(r"[^A-Z]", "", "" if value is None else str(value).upper())
    if token in {"L", "LEFT"}:
        return "L"
    if token in {"M", "MID", "MIDDLE", "C", "CENTER", "CENTRE"}:
        return "M"
    if token in {"R", "RIGHT"}:
        return "R"
    return token


def _width_bucket(raw_width: float | None) -> str:
    if raw_width is None:
        return ""
    if raw_width < 14:
        return "L"
    if raw_width < 40:
        return "M"
    return "R"


def build_coverage_diagnostics(
    *,
    exclusion_reasons: list[str],
    pbp_play: pl.DataFrame,
    tracking: pl.DataFrame,
    generator_audit: pl.DataFrame,
    release_state: pl.DataFrame,
    snap_time: float,
    release_time: float,
    target: tuple[float, float],
    plot_bounds: Sequence[float],
    surfaces: Iterable[ControlSurface],
    target_results: Iterable[TargetControlResult],
    parameters: CoverageParameters,
    allow_ineligible: bool,
) -> pl.DataFrame:
    """Build an audit table for joins, coordinates, kinematics, and models."""
    checks: list[dict[str, object]] = []
    surfaces = tuple(surfaces)
    target_results = tuple(target_results)
    pbp = pbp_play.row(0, named=True)

    eligible = len(exclusion_reasons) == 0
    checks.append(
        _check(
            "Regular-pass eligibility",
            "PASS" if eligible or allow_ineligible else "FAIL",
            len(exclusion_reasons),
            "No exclusions" if eligible else ";".join(exclusion_reasons),
        )
    )

    release_frame = tracking.filter(pl.col("time_into_play") == release_time)
    duplicate_release_entities = (
        release_frame
        .group_by("pro_player_id")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    checks.append(
        _check(
            "No duplicate release-frame entities",
            "PASS" if duplicate_release_entities.is_empty() else "FAIL",
            duplicate_release_entities.height,
            "Each tracking entity must have one row at pass_forward",
        )
    )

    ball_rows = release_frame.filter(is_ball_expr()).height
    checks.append(
        _check(
            "One football row at release",
            "PASS" if ball_rows == 1 else "FAIL",
            ball_rows,
            "The release frame must contain exactly one football observation",
        )
    )

    snap_ball = tracking.filter(
        (pl.col("time_into_play") == snap_time) & is_ball_expr()
    )
    snap_ball_y = (
        float(snap_ball.get_column("Y")[0]) if snap_ball.height == 1 else math.nan
    )
    checks.append(
        _check(
            "Football is near the LOS at ball_snap",
            "PASS" if snap_ball.height == 1 and abs(snap_ball_y) <= 0.50 else "FAIL",
            round(snap_ball_y, 3) if math.isfinite(snap_ball_y) else None,
            "The audited coordinate convention expects Y approximately zero at snap",
        )
    )

    pff_time_to_throw = pbp.get("pff_TIMETOTHROW")
    time_difference = (
        None
        if pff_time_to_throw is None
        else abs(release_time - float(pff_time_to_throw))
    )
    checks.append(
        _check(
            "Tracking release agrees with PFF time to throw",
            (
                "PASS"
                if time_difference is None or time_difference <= 0.20
                else "WARN"
            ),
            None if time_difference is None else round(time_difference, 3),
            "Tolerance is two 10-Hz tracking frames",
        )
    )

    resolution_failures = generator_audit.filter(
        pl.col("match_status") != "MATCHED"
    ).height
    checks.append(
        _check(
            "All generators resolve to tracking players",
            "PASS" if resolution_failures == 0 else "FAIL",
            resolution_failures,
            "Route runners and coverage defenders use the normalized-name bridge",
        )
    )

    offense_count = release_state.filter(pl.col("side") == "OFFENSE").height
    defense_count = release_state.filter(pl.col("side") == "DEFENSE").height
    checks.append(
        _check(
            "At least one route runner",
            "PASS" if offense_count > 0 else "FAIL",
            offense_count,
            "Offensive generators use pff_ROLE == Pass Route",
        )
    )
    checks.append(
        _check(
            "At least one coverage defender",
            "PASS" if defense_count > 0 else "FAIL",
            defense_count,
            "Defensive generators use pff_ROLE == Coverage",
        )
    )

    resolved_count = generator_audit.filter(pl.col("match_status") == "MATCHED").height
    checks.append(
        _check(
            "Release state reconciles to resolved generators",
            "PASS" if release_state.height == resolved_count else "FAIL",
            release_state.height,
            f"Expected {resolved_count} release-state rows",
        )
    )

    release_ids = set(
        release_frame
        .filter(~is_ball_expr())
        .get_column("pro_player_id")
        .to_list()
    )
    generator_ids = set(release_state.get_column("tracking_player_id").to_list())
    missing_ids = sorted(generator_ids - release_ids)
    checks.append(
        _check(
            "Every generator has a release-frame observation",
            "PASS" if not missing_ids else "FAIL",
            len(missing_ids),
            ",".join(str(value) for value in missing_ids),
        )
    )

    too_few_velocity_rows = release_state.filter(
        pl.col("velocity_observations") < parameters.min_velocity_observations
    ).height
    checks.append(
        _check(
            "Velocity history is sufficient",
            "PASS" if too_few_velocity_rows == 0 else "FAIL",
            too_few_velocity_rows,
            f"Minimum observations: {parameters.min_velocity_observations}",
        )
    )

    maximum_observed_speed = float(
        release_state.get_column("speed_yards_per_second").max()
    )
    checks.append(
        _check(
            "Observed release speeds are finite",
            "PASS" if math.isfinite(maximum_observed_speed) else "FAIL",
            round(maximum_observed_speed, 3),
            "Speed is calculated from the past-only linear fit",
        )
    )
    checks.append(
        _check(
            "Observed speed does not exceed model cap",
            (
                "PASS"
                if maximum_observed_speed
                <= parameters.max_player_speed_yards_per_second
                else "WARN"
            ),
            round(maximum_observed_speed, 3),
            f"Configured cap: {parameters.max_player_speed_yards_per_second:.3f} yd/s",
        )
    )

    outside_lateral_bounds = release_state.filter(
        pl.col("release_x").abs() > FIELD_HALF_WIDTH_YARDS + 1e-6
    ).height
    checks.append(
        _check(
            "Generator positions are inside the sidelines",
            "PASS" if outside_lateral_bounds == 0 else "FAIL",
            outside_lateral_bounds,
            f"Expected |X| <= {FIELD_HALF_WIDTH_YARDS}",
        )
    )

    target_x, target_y = target
    x_min, x_max, y_min, y_max = [float(value) for value in plot_bounds]
    target_inside = x_min <= target_x <= x_max and y_min <= target_y <= y_max
    checks.append(
        _check(
            "PFF target lies inside the modeled field domain",
            "PASS" if target_inside else "FAIL",
            None,
            f"Target=({target_x:.3f}, {target_y:.3f})",
        )
    )

    direction = _normalize_direction(pbp.get("pff_PASSDIRECTION"))
    raw_width = pbp.get("pff_PASSWIDTH")
    width_bucket = _width_bucket(
        None if raw_width is None else float(raw_width)
    )
    checks.append(
        _check(
            "PFF direction agrees with PFF pass-width bucket",
            "PASS" if direction == width_bucket else "FAIL",
            None,
            f"direction={direction}; width_bucket={width_bucket}",
        )
    )

    rectangle_area = (x_max - x_min) * (y_max - y_min)
    for surface in surfaces:
        finite_margin = np.isfinite(surface.control_margin).all()
        valid_owner = (
            surface.owner_index.min() >= 0
            and surface.owner_index.max() < release_state.height
        )
        nonempty_polygons = (
            True
            if surface.polygons is None
            else all(len(polygon) >= 3 for polygon in surface.polygons)
        )
        area_matches = True
        area_error = 0.0
        if surface.polygons is not None:
            total_area = sum(_polygon_area(polygon) for polygon in surface.polygons)
            area_error = abs(total_area - rectangle_area)
            area_matches = area_error <= max(1e-5, rectangle_area * 1e-7)
        status = (
            "PASS"
            if finite_margin and valid_owner and nonempty_polygons and area_matches
            else "FAIL"
        )
        checks.append(
            _check(
                f"{surface.model_label} surface is complete",
                status,
                int(surface.owner_index.size),
                "Finite margins, valid owners, non-empty cells; "
                f"polygon_area_error={area_error:.6g}",
            )
        )

    release_sides = np.asarray(
        [str(value).upper() for value in release_state.get_column("side")]
    )
    for surface, result in zip(surfaces, target_results, strict=True):
        finite_values = all(
            math.isfinite(value)
            for value in [
                result.owner_value,
                result.offense_value,
                result.defense_value,
                result.control_margin,
            ]
        )
        checks.append(
            _check(
                f"{result.model_label} target result is finite",
                "PASS" if finite_values else "FAIL",
                None,
                f"Owner={result.owner_name}; margin={result.control_margin:.4f} "
                f"{result.margin_unit}",
            )
        )

        x_index = int(np.argmin(abs(surface.x_coordinates - target_x)))
        y_index = int(np.argmin(abs(surface.y_coordinates - target_y)))
        grid_side = str(release_sides[surface.owner_index[y_index, x_index]])
        direct_side = "OFFENSE" if result.target_in_offense_control else "DEFENSE"
        grid_agrees = grid_side == direct_side
        checks.append(
            _check(
                f"{result.model_label} exact target agrees with plotted grid",
                (
                    "PASS"
                    if grid_agrees
                    else ("WARN" if result.target_near_boundary else "FAIL")
                ),
                None,
                f"exact={direct_side}; nearest_grid={grid_side}; "
                f"near_boundary={result.target_near_boundary}",
            )
        )

    return pl.DataFrame(checks)


def raise_for_failed_diagnostics(diagnostics: pl.DataFrame) -> None:
    """Raise when any coverage diagnostic is marked ``FAIL``."""
    failures = diagnostics.filter(pl.col("status") == "FAIL")
    if not failures.is_empty():
        raise ValueError("Coverage diagnostics failed:\n" + str(failures))
