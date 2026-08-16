"""Extract release-frame state and estimate player velocity from tracking data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl

from nfc_north_player_tracking.config import CoverageParameters


def is_ball_expr() -> pl.Expr:
    """Return a Polars expression that identifies the football entity."""
    return (
        (pl.col("pro_player_id") == "-1")
        | (
            pl.col("player_name")
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .str.to_lowercase()
            == "ball"
        )
    )


def event_time(tracking: pl.DataFrame, event_name: str) -> float:
    """Return the one distinct timestamp for a synchronized tracking event."""
    normalized = event_name.strip().lower()
    times = (
        tracking
        .filter(pl.col("event_normalized") == normalized)
        .get_column("time_into_play")
        .drop_nulls()
        .unique()
        .sort()
    )
    if len(times) != 1:
        raise ValueError(
            f"Expected one distinct {normalized!r} time; found "
            f"{len(times)}: {times.to_list()}."
        )
    return float(times[0])


def optional_event_time(tracking: pl.DataFrame, event_name: str) -> float | None:
    """Return an event time when present, otherwise ``None``."""
    normalized = event_name.strip().lower()
    times = (
        tracking
        .filter(pl.col("event_normalized") == normalized)
        .get_column("time_into_play")
        .drop_nulls()
        .unique()
        .sort()
    )
    if len(times) == 0:
        return None
    if len(times) != 1:
        raise ValueError(
            f"Expected at most one distinct {normalized!r} time; found "
            f"{len(times)}: {times.to_list()}."
        )
    return float(times[0])


def pass_outcome_event(tracking: pl.DataFrame) -> tuple[str | None, float | None]:
    """Return the optional unique ``pass_outcome_*`` event and timestamp."""
    outcomes = (
        tracking
        .filter(pl.col("event_normalized").str.starts_with("pass_outcome_"))
        .select("event_normalized", "time_into_play")
        .unique()
        .sort(["time_into_play", "event_normalized"])
    )
    if outcomes.is_empty():
        return None, None
    if outcomes.height != 1:
        raise ValueError(
            "Expected at most one pass_outcome_* event; found "
            f"{outcomes.to_dicts()}."
        )
    row = outcomes.row(0, named=True)
    return str(row["event_normalized"]), float(row["time_into_play"])


def frame_at_time(tracking: pl.DataFrame, time_value: float) -> pl.DataFrame:
    """Return all tracked entities at one observed frame time."""
    return tracking.filter(pl.col("time_into_play") == time_value)


def ball_row_at_time(tracking: pl.DataFrame, time_value: float) -> dict[str, object]:
    """Return the unique football observation at one frame time."""
    rows = frame_at_time(tracking, time_value).filter(is_ball_expr())
    if rows.height != 1:
        raise ValueError(
            f"Expected one football row at t={time_value}; found {rows.height}."
        )
    return rows.row(0, named=True)


def linear_velocity(
    times: Iterable[float],
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> tuple[float, float]:
    """Fit past-only linear position models and return ``vx`` and ``vy``."""
    time_array = np.asarray(list(times), dtype=float)
    x_array = np.asarray(list(x_values), dtype=float)
    y_array = np.asarray(list(y_values), dtype=float)

    if len(time_array) < 2:
        raise ValueError("At least two observations are required for velocity.")
    if not (
        np.isfinite(time_array).all()
        and np.isfinite(x_array).all()
        and np.isfinite(y_array).all()
    ):
        raise ValueError("Velocity observations contain non-finite values.")
    if np.unique(time_array).size < 2:
        raise ValueError("Velocity timestamps do not contain two unique values.")

    centered_time = time_array - time_array.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0:
        raise ValueError("Velocity regression has zero time variance.")

    vx = float(np.dot(centered_time, x_array - x_array.mean()) / denominator)
    vy = float(np.dot(centered_time, y_array - y_array.mean()) / denominator)
    return vx, vy


def _release_row_for_player(
    tracking: pl.DataFrame,
    player_id: str,
    release_time: float,
) -> dict[str, object]:
    rows = tracking.filter(
        (pl.col("pro_player_id") == player_id)
        & (pl.col("time_into_play") == release_time)
    )
    if rows.height != 1:
        raise ValueError(
            f"Expected one release row for tracking player {player_id}; "
            f"found {rows.height}."
        )
    return rows.row(0, named=True)


def _velocity_history(
    tracking: pl.DataFrame,
    player_id: str,
    release_time: float,
    lookback_seconds: float,
) -> pl.DataFrame:
    tolerance = 1e-9
    return (
        tracking
        .filter(
            (pl.col("pro_player_id") == player_id)
            & (
                pl.col("time_into_play")
                >= release_time - lookback_seconds - tolerance
            )
            & (pl.col("time_into_play") <= release_time + tolerance)
        )
        .select("time_into_play", "X", "Y")
        .unique()
        .sort("time_into_play")
    )


def build_release_state(
    tracking: pl.DataFrame,
    resolved_generators: pl.DataFrame,
    release_time: float,
    parameters: CoverageParameters,
) -> pl.DataFrame:
    """Return release position, velocity, and projected site for each generator."""
    parameters.validate()
    rows: list[dict[str, object]] = []

    for generator in resolved_generators.iter_rows(named=True):
        player_id = str(generator["tracking_player_id"])
        release = _release_row_for_player(tracking, player_id, release_time)
        history = _velocity_history(
            tracking,
            player_id,
            release_time,
            parameters.velocity_lookback_seconds,
        )
        if history.height < parameters.min_velocity_observations:
            raise ValueError(
                f"{generator['pff_PLAYERNAME']} has only {history.height} "
                "velocity observations in the configured lookback window."
            )

        vx, vy = linear_velocity(
            history.get_column("time_into_play").to_list(),
            history.get_column("X").to_list(),
            history.get_column("Y").to_list(),
        )
        speed = float(np.hypot(vx, vy))
        release_x = float(release["X"])
        release_y = float(release["Y"])
        horizon = parameters.velocity_projection_horizon_seconds

        rows.append(
            {
                "pff_GAMEID": generator["pff_GAMEID"],
                "pff_PLAYID": generator["pff_PLAYID"],
                "side": generator["side"],
                "analysis_role": generator["analysis_role"],
                "pff_ROLE": generator["pff_ROLE"],
                "pff_TEAM": generator["pff_TEAM"],
                "pff_PLAYERID": generator["pff_PLAYERID"],
                "pff_GSISPLAYERID": generator["pff_GSISPLAYERID"],
                "pff_PLAYERNAME": generator["pff_PLAYERNAME"],
                "tracking_player_id": player_id,
                "tracking_player_name": generator["tracking_player_name"],
                "team_id": generator["team_id"],
                "targeted_receiver": bool(generator["targeted_receiver"]),
                "release_time": release_time,
                "release_x": release_x,
                "release_y": release_y,
                "orientation_degrees": release.get("orientation"),
                "velocity_x_yards_per_second": vx,
                "velocity_y_yards_per_second": vy,
                "speed_yards_per_second": speed,
                "velocity_observations": history.height,
                "velocity_estimation_method": "PAST_LINEAR_REGRESSION",
                "projected_x": release_x + horizon * vx,
                "projected_y": release_y + horizon * vy,
                "projection_horizon_seconds": horizon,
            }
        )

    if not rows:
        raise ValueError("No route-runner or coverage-defender release states exist.")

    return pl.DataFrame(rows).sort(["side", "pff_PLAYERNAME"])


def context_release_state(
    tracking: pl.DataFrame,
    release_time: float,
    quarterback_tracking_id: str | None,
) -> dict[str, object]:
    """Return release-frame football and optional quarterback coordinates."""
    ball = ball_row_at_time(tracking, release_time)
    result: dict[str, object] = {
        "ball_x": float(ball["X"]),
        "ball_y": float(ball["Y"]),
        "quarterback_x": None,
        "quarterback_y": None,
        "quarterback_name": None,
    }

    if quarterback_tracking_id:
        qb = _release_row_for_player(
            tracking,
            str(quarterback_tracking_id),
            release_time,
        )
        result.update(
            {
                "quarterback_x": float(qb["X"]),
                "quarterback_y": float(qb["Y"]),
                "quarterback_name": str(qb["player_name"]),
            }
        )
    return result
