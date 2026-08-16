"""Audit the tracking coordinate convention on representative pass plays.

The validator tests the working hypothesis that tracking coordinates are
already standardized for passing analysis:

* ``X`` is lateral, with offense left negative and offense right positive.
* ``Y`` is longitudinal and relative to the line of scrimmage.
* downfield is positive ``Y``.
* the ball is near ``Y == 0`` at ``ball_snap``.

When ``--play`` is omitted, the validator uses ``DEFAULT_PLAYS``. Repeat
``--play`` to audit several plays in one run.

Examples:

    # Uses DEFAULT_PLAYS when --play is omitted.
    python tests/validate_tracking_coordinates.py

    # Override the default play. Repeat --play to audit multiple plays.
    python tests/validate_tracking_coordinates.py --play 28430:6408068

    # Audit more than one play.
    python tests/validate_tracking_coordinates.py \
        --play 28430:6408068 \
        --play 28430:6408079
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = REPO_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from nfc_north_player_tracking.config import (  # noqa: E402
    FIELD_HALF_WIDTH_YARDS,
    PFF_PASS_WIDTH_CENTER,
    RAW_LATERAL_COLUMN,
    RAW_LONGITUDINAL_COLUMN,
)

GAME_ID = "pff_GAMEID"
PLAY_ID = "pff_PLAYID"
GSIS_GAME_KEY = "pff_GSISGAMEKEY"
GSIS_PLAY_ID = "pff_GSISPLAYID"

DEFAULT_PLAYS = ["28430:6408068"]
SNAP_LOS_TOLERANCE_YARDS = 0.50
TIME_TO_THROW_DIFFERENCE_TOLERANCE_SECONDS = 0.15


def normalize_id_value(value: object) -> str:
    """Normalize an identifier such as 10635.0 to 10635."""
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def clean_id(column: str) -> pl.Expr:
    """Return a Polars expression that normalizes an identifier column."""
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.replace(r"\.0$", "")
    )


def parse_play(value: str) -> tuple[str, str]:
    """Parse ``GAME_ID:PLAY_ID`` from a command-line argument."""
    pieces = value.split(":", maxsplit=1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Play must use GAME_ID:PLAY_ID format.")

    game_id = normalize_id_value(pieces[0])
    play_id = normalize_id_value(pieces[1])
    if not game_id or not play_id:
        raise argparse.ArgumentTypeError("Both GAME_ID and PLAY_ID are required.")
    return game_id, play_id


def normalize_pff_direction(value: object) -> str:
    """Normalize PFF pass direction to ``L``, ``M``, or ``R`` when possible."""
    token = re.sub(r"[^A-Z]", "", "" if value is None else str(value).upper())
    if token in {"L", "LEFT"}:
        return "L"
    if token in {"M", "MID", "MIDDLE", "C", "CENTER", "CENTRE"}:
        return "M"
    if token in {"R", "RIGHT"}:
        return "R"
    return token


def load_pbp_play(pbp_csv: Path, game_id: str, play_id: str) -> pl.DataFrame:
    """Load the exact PBP row that supplies tracking keys and pass target."""
    columns = [
        GAME_ID,
        PLAY_ID,
        GSIS_GAME_KEY,
        GSIS_PLAY_ID,
        "pff_PASSDEPTH",
        "pff_PASSWIDTH",
        "pff_PASSDIRECTION",
        "pff_OFFTEAM",
        "pff_TIMETOTHROW",
    ]
    play = (
        pl.scan_csv(pbp_csv, infer_schema=False)
        .select(columns)
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
            pl.col("pff_PASSDEPTH").cast(pl.Float64, strict=False),
            pl.col("pff_PASSWIDTH").cast(pl.Float64, strict=False),
            pl.col("pff_TIMETOTHROW").cast(pl.Float64, strict=False),
        )
        .filter((pl.col(GAME_ID) == game_id) & (pl.col(PLAY_ID) == play_id))
        .collect()
        .unique()
    )
    if play.height != 1:
        raise ValueError(
            f"Expected one PBP row for {game_id}:{play_id}; found {play.height}."
        )
    return play


def load_tracking_play(tracking_csv: Path, pbp_play: pl.DataFrame) -> pl.DataFrame:
    """Load tracking observations that match all three available play keys."""
    key = pbp_play.row(0, named=True)
    return (
        pl.scan_csv(tracking_csv, infer_schema=False)
        .select(
            [
                "pff_play_id",
                "game_key",
                "gsis_play_id",
                "team_id",
                "pro_player_id",
                "player_name",
                "event",
                RAW_LATERAL_COLUMN,
                RAW_LONGITUDINAL_COLUMN,
                "time_into_play",
                "orientation",
            ]
        )
        .with_columns(
            clean_id("pff_play_id").alias("pff_play_id"),
            clean_id("game_key").alias("game_key"),
            clean_id("gsis_play_id").alias("gsis_play_id"),
            clean_id("pro_player_id").alias("pro_player_id"),
            pl.col("event")
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .str.to_lowercase()
            .alias("event_normalized"),
            pl.col(RAW_LATERAL_COLUMN).cast(pl.Float64, strict=False),
            pl.col(RAW_LONGITUDINAL_COLUMN).cast(pl.Float64, strict=False),
            pl.col("time_into_play").cast(pl.Float64, strict=False),
            pl.col("orientation").cast(pl.Float64, strict=False),
        )
        .filter(
            (pl.col("pff_play_id") == key[PLAY_ID])
            & (pl.col("game_key") == key[GSIS_GAME_KEY])
            & (pl.col("gsis_play_id") == key[GSIS_PLAY_ID])
            & pl.col("time_into_play").is_not_null()
            & pl.col(RAW_LATERAL_COLUMN).is_not_null()
            & pl.col(RAW_LONGITUDINAL_COLUMN).is_not_null()
        )
        .collect()
        .sort(["time_into_play", "pro_player_id"])
    )


def is_ball() -> pl.Expr:
    """Return a Polars expression that identifies the football row."""
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
    """Return the one distinct timestamp for a play event."""
    times = (
        tracking
        .filter(pl.col("event_normalized") == event_name)
        .get_column("time_into_play")
        .drop_nulls()
        .unique()
        .sort()
    )
    if len(times) != 1:
        raise ValueError(
            f"Expected one distinct {event_name!r} time; found {len(times)}: "
            f"{times.to_list()}"
        )
    return float(times[0])


def frame_at_time(tracking: pl.DataFrame, time_value: float) -> pl.DataFrame:
    """Return every tracked entity at an observed frame time."""
    return tracking.filter(pl.col("time_into_play") == time_value)


def ball_row_at_time(tracking: pl.DataFrame, time_value: float) -> dict[str, object]:
    """Return the unique football observation at one frame time."""
    rows = frame_at_time(tracking, time_value).filter(is_ball())
    if rows.height != 1:
        raise ValueError(
            f"Expected one ball row at t={time_value}; found {rows.height}."
        )
    return rows.row(0, named=True)


def pass_outcome_event(tracking: pl.DataFrame) -> tuple[str | None, float | None]:
    """Return the optional unique ``pass_outcome_*`` event and time."""
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
            "Expected at most one distinct pass_outcome_* event; found "
            f"{outcomes.to_dicts()}."
        )
    row = outcomes.row(0, named=True)
    return str(row["event_normalized"]), float(row["time_into_play"])


def target_bucket(target_x: float) -> str:
    """Map a centered PFF target width to Left, Middle, or Right."""
    if target_x < -13.0:
        return "L"
    if target_x > 13.0:
        return "R"
    return "M"


def draw_audit_plot(
    release_frame: pl.DataFrame,
    arrival_ball: dict[str, object],
    outcome_event_name: str | None,
    outcome_ball: dict[str, object] | None,
    target_x: float,
    target_y: float,
    game_id: str,
    play_id: str,
    output_path: Path,
) -> None:
    """Plot the release frame, tracked arrival point, and PFF target."""
    figure, axis = plt.subplots(figsize=(9, 12))
    axis.axhline(0.0, linewidth=2, label="LOS")
    axis.axvline(-FIELD_HALF_WIDTH_YARDS, linewidth=1)
    axis.axvline(FIELD_HALF_WIDTH_YARDS, linewidth=1)

    for row in release_frame.iter_rows(named=True):
        ball_row = (
            str(row["pro_player_id"]).strip() == "-1"
            or str(row["player_name"]).strip().lower() == "ball"
        )
        axis.scatter(
            float(row[RAW_LATERAL_COLUMN]),
            float(row[RAW_LONGITUDINAL_COLUMN]),
            marker="*" if ball_row else "o",
            s=180 if ball_row else 70,
        )
        if not ball_row:
            axis.annotate(
                str(row["player_name"]),
                (
                    float(row[RAW_LATERAL_COLUMN]),
                    float(row[RAW_LONGITUDINAL_COLUMN]),
                ),
                fontsize=8,
            )

    axis.scatter(
        float(arrival_ball[RAW_LATERAL_COLUMN]),
        float(arrival_ball[RAW_LONGITUDINAL_COLUMN]),
        marker="s",
        s=130,
        label="Tracked ball at pass_arrived",
    )
    if outcome_ball is not None:
        axis.scatter(
            float(outcome_ball[RAW_LATERAL_COLUMN]),
            float(outcome_ball[RAW_LONGITUDINAL_COLUMN]),
            marker="D",
            s=120,
            label=f"Tracked ball at {outcome_event_name}",
        )
    axis.scatter(target_x, target_y, marker="X", s=180, label="PFF target")

    axis.set_xlim(-FIELD_HALF_WIDTH_YARDS - 2, FIELD_HALF_WIDTH_YARDS + 2)
    axis.set_ylim(-15, max(35.0, target_y + 10.0))
    axis.set_title(f"Tracking coordinate audit: {game_id}:{play_id}")
    axis.set_xlabel("X: offense left (-) to right (+), yards")
    axis.set_ylabel("Y: yards relative to LOS, downfield positive")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def audit_play(
    pbp_csv: Path,
    tracking_csv: Path,
    output_directory: Path,
    game_id: str,
    play_id: str,
) -> pl.DataFrame:
    """Audit one play and write its release-frame table and visual plot."""
    pbp_play = load_pbp_play(pbp_csv, game_id, play_id)
    tracking = load_tracking_play(tracking_csv, pbp_play)
    if tracking.is_empty():
        raise ValueError(f"No tracking rows found for {game_id}:{play_id}.")

    snap_time = event_time(tracking, "ball_snap")
    release_time = event_time(tracking, "pass_forward")
    arrival_time = event_time(tracking, "pass_arrived")
    outcome_event_name, outcome_time = pass_outcome_event(tracking)
    if not snap_time < release_time < arrival_time:
        raise ValueError("Expected ball_snap < pass_forward < pass_arrived.")
    if outcome_time is not None and outcome_time < arrival_time:
        raise ValueError("Expected pass_outcome_* at or after pass_arrived.")

    snap_ball = ball_row_at_time(tracking, snap_time)
    release_ball = ball_row_at_time(tracking, release_time)
    arrival_ball = ball_row_at_time(tracking, arrival_time)
    outcome_ball = (
        ball_row_at_time(tracking, outcome_time)
        if outcome_time is not None
        else None
    )
    release_frame = frame_at_time(tracking, release_time)

    duplicate_release_entities = (
        release_frame
        .group_by("pro_player_id")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )

    pbp = pbp_play.row(0, named=True)
    if pbp["pff_PASSDEPTH"] is None or pbp["pff_PASSWIDTH"] is None:
        raise ValueError("PFF pass depth and width are required for the audit.")

    target_x = float(pbp["pff_PASSWIDTH"]) - PFF_PASS_WIDTH_CENTER
    target_y = float(pbp["pff_PASSDEPTH"])
    raw_pff_direction = str(pbp["pff_PASSDIRECTION"] or "").strip().upper()
    normalized_pff_direction = normalize_pff_direction(raw_pff_direction)
    computed_direction = target_bucket(target_x)

    arrival_x = float(arrival_ball[RAW_LATERAL_COLUMN])
    arrival_y = float(arrival_ball[RAW_LONGITUDINAL_COLUMN])
    target_arrival_distance = math.hypot(arrival_x - target_x, arrival_y - target_y)
    outcome_x = (
        float(outcome_ball[RAW_LATERAL_COLUMN])
        if outcome_ball is not None
        else None
    )
    outcome_y = (
        float(outcome_ball[RAW_LONGITUDINAL_COLUMN])
        if outcome_ball is not None
        else None
    )
    target_outcome_distance = (
        math.hypot(outcome_x - target_x, outcome_y - target_y)
        if outcome_x is not None and outcome_y is not None
        else None
    )
    snap_ball_y = float(snap_ball[RAW_LONGITUDINAL_COLUMN])

    tracking_time_to_throw = release_time - snap_time
    pff_time_to_throw = pbp["pff_TIMETOTHROW"]
    time_to_throw_difference = (
        abs(tracking_time_to_throw - float(pff_time_to_throw))
        if pff_time_to_throw is not None
        else None
    )
    time_to_throw_within_tolerance = (
        time_to_throw_difference <= TIME_TO_THROW_DIFFERENCE_TOLERANCE_SECONDS
        if time_to_throw_difference is not None
        else None
    )

    release_x = release_frame.get_column(RAW_LATERAL_COLUMN)
    release_x_within_field = bool(
        (
            (release_x >= -FIELD_HALF_WIDTH_YARDS)
            & (release_x <= FIELD_HALF_WIDTH_YARDS)
        ).all()
    )
    target_x_within_field = (
        -FIELD_HALF_WIDTH_YARDS <= target_x <= FIELD_HALF_WIDTH_YARDS
    )
    pff_direction_matches_width = (
        normalized_pff_direction in {"", computed_direction}
    )

    release_ball_rows = release_frame.filter(is_ball()).height
    release_player_rows = release_frame.height - release_ball_rows

    release_path = (
        output_directory / f"tracking_release_{game_id}_{play_id}.csv"
    )
    plot_path = (
        output_directory / f"tracking_coordinate_audit_{game_id}_{play_id}.png"
    )

    release_frame.with_columns(
        pl.col(RAW_LATERAL_COLUMN).alias("field_x"),
        pl.col(RAW_LONGITUDINAL_COLUMN).alias("field_y"),
    ).write_csv(release_path)

    draw_audit_plot(
        release_frame,
        arrival_ball,
        outcome_event_name,
        outcome_ball,
        target_x,
        target_y,
        game_id,
        play_id,
        plot_path,
    )

    failures: list[str] = []
    if abs(snap_ball_y) > SNAP_LOS_TOLERANCE_YARDS:
        failures.append("BALL_NOT_NEAR_LOS_AT_SNAP")
    if not pff_direction_matches_width:
        failures.append("PFF_DIRECTION_DISAGREES_WITH_WIDTH")
    if not release_x_within_field:
        failures.append("RELEASE_X_OUTSIDE_FIELD")
    if not target_x_within_field:
        failures.append("PFF_TARGET_X_OUTSIDE_FIELD")
    if release_ball_rows != 1:
        failures.append("RELEASE_BALL_ROW_COUNT")
    if not duplicate_release_entities.is_empty():
        failures.append("DUPLICATE_RELEASE_ENTITY")

    return pl.DataFrame(
        {
            GAME_ID: [game_id],
            PLAY_ID: [play_id],
            "pff_offense_team": [str(pbp["pff_OFFTEAM"] or "")],
            "snap_time": [snap_time],
            "release_time": [release_time],
            "arrival_time": [arrival_time],
            "pass_outcome_event": [outcome_event_name],
            "pass_outcome_time": [outcome_time],
            "tracking_time_to_throw_seconds": [tracking_time_to_throw],
            "pff_time_to_throw_seconds": [pff_time_to_throw],
            "time_to_throw_difference_seconds": [time_to_throw_difference],
            "time_to_throw_within_tolerance": [time_to_throw_within_tolerance],
            "snap_ball_y": [snap_ball_y],
            "snap_ball_near_los": [abs(snap_ball_y) <= SNAP_LOS_TOLERANCE_YARDS],
            "release_ball_x": [float(release_ball[RAW_LATERAL_COLUMN])],
            "release_ball_y": [float(release_ball[RAW_LONGITUDINAL_COLUMN])],
            "target_x": [target_x],
            "target_y": [target_y],
            "arrival_ball_x": [arrival_x],
            "arrival_ball_y": [arrival_y],
            "target_arrival_distance_yards": [target_arrival_distance],
            "outcome_ball_x": [outcome_x],
            "outcome_ball_y": [outcome_y],
            "target_outcome_distance_yards": [target_outcome_distance],
            "pff_pass_direction_raw": [raw_pff_direction],
            "pff_pass_direction_normalized": [normalized_pff_direction],
            "target_bucket_from_width": [computed_direction],
            "pff_direction_matches_width": [pff_direction_matches_width],
            "release_rows": [release_frame.height],
            "release_player_rows": [release_player_rows],
            "release_ball_rows": [release_ball_rows],
            "duplicate_release_entities": [duplicate_release_entities.height],
            "release_x_within_field": [release_x_within_field],
            "target_x_within_field": [target_x_within_field],
            "coordinate_hypothesis": ["field_x=X; field_y=Y"],
            "release_csv": [str(release_path)],
            "audit_plot": [str(plot_path)],
            "status": ["PASS" if not failures else "FAIL"],
            "failure_reasons": [";".join(failures)],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--play",
        action="append",
        default=None,
        metavar="GAME_ID:PLAY_ID",
        help=(
            "Selected play. Repeat for more than one play. Defaults to "
            f"{', '.join(DEFAULT_PLAYS)} when omitted."
        ),
    )
    parser.add_argument(
        "--pbp-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/pff_pbp.csv",
    )
    parser.add_argument(
        "--tracking-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/tracking_sample.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "outputs/diagnostics",
    )
    args = parser.parse_args()

    selected_plays = [parse_play(value) for value in (args.play or DEFAULT_PLAYS)]
    args.output_directory.mkdir(parents=True, exist_ok=True)

    summaries: list[pl.DataFrame] = []
    for game_id, play_id in selected_plays:
        play_summary = audit_play(
            args.pbp_csv,
            args.tracking_csv,
            args.output_directory,
            game_id,
            play_id,
        )
        summaries.append(play_summary)

        play_summary_path = (
            args.output_directory
            / f"tracking_coordinate_summary_{game_id}_{play_id}.csv"
        )
        play_summary.write_csv(play_summary_path)
        print(f"\nCoordinate audit: {game_id}:{play_id}")
        print(play_summary)
        print(f"Wrote: {play_summary_path}")

    summary = pl.concat(summaries, how="vertical_relaxed")
    summary_path = args.output_directory / "tracking_coordinate_summary.csv"
    summary.write_csv(summary_path)
    print(f"\nWrote combined summary: {summary_path}")

    failed = summary.filter(pl.col("status") == "FAIL")
    if not failed.is_empty():
        print("\nFAIL: one or more selected plays failed coordinate validation.")
        print(failed.select(GAME_ID, PLAY_ID, "failure_reasons"))
        return 1

    print("\nPASS: coordinate hypothesis is internally consistent for all plays.")
    print("Review target-distance fields and PNGs before generalizing further.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
