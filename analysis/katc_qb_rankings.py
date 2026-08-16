"""Rank selected quarterbacks by KATC agreement on eligible pass attempts.

KATC agreement is defined as follows:

    The observed PFF target location is offense-controlled at ``pass_forward``
    under Kinematic Arrival-Time Control (KATC).

The play result is intentionally not used in the ranking. Completion,
incompletion, and interception labels are used only to identify ordinary pass
attempts within the project's existing strict pass-population rules.

Default quarterbacks:

- Caleb Williams
- Carson Wentz
- Jared Goff
- J.J. McCarthy
- Jordan Love

Outputs:

- ``katc_qb_play_results.csv``: one row per successfully evaluated attempt.
- ``katc_qb_rankings.csv``: quarterback-level ranking and sample coverage.
- ``katc_qb_skipped_plays.csv``: plays excluded or unavailable to the model.

Examples from the repository root:

    python analysis/katc_qb_rankings.py

    python analysis/katc_qb_rankings.py \
        --tracking-csv data/raw/tracking_sample.csv \
        --minimum-attempts 10

    python analysis/katc_qb_rankings.py \
        --qb "Caleb Williams" \
        --qb "Jordan Love"
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = REPO_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from nfc_north_player_tracking.config import (  # noqa: E402
    DEFAULT_PARAMETERS,
    FIELD_HALF_WIDTH_YARDS,
    OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
    OWNERSHIP_MODE_INTENDED_RECEIVER,
    CoverageParameters,
)
from nfc_north_player_tracking.coverage_queries import (  # noqa: E402
    DEFENSE_COLUMNS,
    GAME_ID,
    GSIS_GAME_KEY,
    GSIS_PLAY_ID,
    OFFENSE_COLUMNS,
    PBP_COLUMNS,
    PLAY_ID,
    TRACKING_COLUMNS,
    clean_id,
    clean_upper,
    intended_receiver_name,
    pass_target,
    regular_pass_exclusion_reasons,
    truthy_expr,
)
from nfc_north_player_tracking.dominant_regions import (  # noqa: E402
    evaluate_kinematic_target,
)
from nfc_north_player_tracking.kinematics import (  # noqa: E402
    ball_row_at_time,
    build_release_state,
    event_time,
)
from nfc_north_player_tracking.player_resolution import (  # noqa: E402
    normalize_player_name,
    require_resolved_generators,
    resolve_generators,
)


DEFAULT_QUARTERBACKS = (
    "Caleb Williams",
    "Carson Wentz",
    "Jared Goff",
    "J.J. McCarthy",
    "Jordan Love",
)

PLAY_RESULTS_FILENAME = "katc_qb_play_results.csv"
RANKINGS_FILENAME = "katc_qb_rankings.csv"
SKIPPED_FILENAME = "katc_qb_skipped_plays.csv"


PLAY_RESULT_SCHEMA: dict[str, pl.DataType] = {
    "qb_name": pl.String,
    "qb_player_id": pl.String,
    "qb_team": pl.String,
    GAME_ID: pl.String,
    PLAY_ID: pl.String,
    GSIS_GAME_KEY: pl.String,
    GSIS_PLAY_ID: pl.String,
    "pff_GAMEDATE": pl.String,
    "pff_WEEK": pl.String,
    "pff_QUARTER": pl.String,
    "pff_DOWN": pl.String,
    "pff_CLOCK": pl.String,
    "pff_OFFTEAM": pl.String,
    "pff_DEFTEAM": pl.String,
    "target_x": pl.Float64,
    "target_y": pl.Float64,
    "intended_receiver_name": pl.String,
    "ownership_mode": pl.String,
    "snap_time": pl.Float64,
    "release_time": pl.Float64,
    "pff_time_to_throw_s": pl.Float64,
    "release_time_difference_s": pl.Float64,
    "route_runners": pl.Int64,
    "coverage_defenders": pl.Int64,
    "max_observed_speed_yards_per_second": pl.Float64,
    "observed_speed_exceeds_model_cap": pl.Boolean,
    "katc_target_owner_name": pl.String,
    "katc_target_owner_side": pl.String,
    "katc_target_owner_arrival_time_s": pl.Float64,
    "katc_offense_reference_name": pl.String,
    "katc_offense_arrival_time_s": pl.Float64,
    "katc_defense_reference_name": pl.String,
    "katc_defense_arrival_time_s": pl.Float64,
    "katc_control_margin_s": pl.Float64,
    "katc_target_in_offense_control": pl.Boolean,
    "katc_model_agrees_with_throw": pl.Boolean,
    "katc_target_near_boundary": pl.Boolean,
    "velocity_lookback_seconds": pl.Float64,
    "max_player_speed_yards_per_second": pl.Float64,
    "max_player_acceleration_yards_per_second_squared": pl.Float64,
    "katc_target_time_step_seconds": pl.Float64,
    "katc_target_arrival_radius_yards": pl.Float64,
}

SKIPPED_SCHEMA: dict[str, pl.DataType] = {
    "qb_name": pl.String,
    "qb_player_id": pl.String,
    "qb_team": pl.String,
    GAME_ID: pl.String,
    PLAY_ID: pl.String,
    "stage": pl.String,
    "skip_reason": pl.String,
    "details": pl.String,
}


def normalize_id(value: object) -> str:
    """Normalize an identifier such as ``10635.0`` to ``10635``."""
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalized_name_expr(column: str) -> pl.Expr:
    """Return a simple alphanumeric name key for batch quarterback lookup."""
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]", "")
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qb",
        action="append",
        help=(
            "Quarterback name to include. Repeat for multiple names. Uses the "
            "five project quarterbacks when omitted."
        ),
    )
    parser.add_argument(
        "--pbp-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/pff_pbp.csv",
    )
    parser.add_argument(
        "--offense-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/pff_offense.csv",
    )
    parser.add_argument(
        "--defense-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/pff_defense.csv",
    )
    parser.add_argument(
        "--tracking-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/tracking_sample.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "outputs/katc_rankings",
    )
    parser.add_argument(
        "--ownership-mode",
        choices=[
            OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
            OWNERSHIP_MODE_INTENDED_RECEIVER,
        ],
        default=OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
        help=(
            "Defines offensive control at the target. The default credits any "
            "route runner; the optional mode credits only the intended receiver."
        ),
    )
    parser.add_argument(
        "--velocity-lookback",
        type=float,
        default=DEFAULT_PARAMETERS.velocity_lookback_seconds,
        help="Past-only velocity-regression window in seconds.",
    )
    parser.add_argument(
        "--minimum-velocity-observations",
        type=int,
        default=DEFAULT_PARAMETERS.min_velocity_observations,
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=DEFAULT_PARAMETERS.max_player_speed_yards_per_second,
        help="Shared KATC speed ceiling in yards/second.",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=(
            DEFAULT_PARAMETERS.max_player_acceleration_yards_per_second_squared
        ),
        help="Shared KATC acceleration ceiling in yards/second^2.",
    )
    parser.add_argument(
        "--target-time-step",
        type=float,
        default=DEFAULT_PARAMETERS.kinematic_target_time_step_seconds,
        help="KATC exact-target simulation step in seconds.",
    )
    parser.add_argument(
        "--target-arrival-radius",
        type=float,
        default=DEFAULT_PARAMETERS.kinematic_target_arrival_radius_yards,
        help="Distance from the target considered an arrival, in yards.",
    )
    parser.add_argument(
        "--max-arrival-time",
        type=float,
        default=DEFAULT_PARAMETERS.kinematic_max_time_seconds,
        help="Maximum KATC simulation time in seconds.",
    )
    parser.add_argument(
        "--near-boundary-seconds",
        type=float,
        default=DEFAULT_PARAMETERS.target_near_boundary_margin_seconds,
        help="Absolute KATC margin treated as near the control boundary.",
    )
    parser.add_argument(
        "--minimum-attempts",
        type=int,
        default=1,
        help=(
            "Minimum successful KATC evaluations required for an official rank. "
            "All quarterbacks still appear in the output."
        ),
    )
    parser.add_argument(
        "--max-plays-per-qb",
        type=int,
        help="Optional development limit applied after play discovery.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first play-level KATC error instead of recording a skip.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each evaluated or skipped play.",
    )
    return parser.parse_args()


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Return an empty DataFrame with stable output dtypes."""
    return pl.DataFrame(schema=schema)


def _validate_inputs(args: argparse.Namespace) -> None:
    """Validate paths and scalar command-line options."""
    for label, path in {
        "PBP CSV": args.pbp_csv,
        "offense CSV": args.offense_csv,
        "defense CSV": args.defense_csv,
        "tracking CSV": args.tracking_csv,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if args.minimum_attempts < 1:
        raise ValueError("--minimum-attempts must be at least one.")
    if args.max_plays_per_qb is not None and args.max_plays_per_qb < 1:
        raise ValueError("--max-plays-per-qb must be at least one when provided.")


def _canonical_quarterbacks(values: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Return deduplicated display names and normalized-name lookup."""
    names: list[str] = []
    lookup: dict[str, str] = {}
    for raw_name in values:
        display = str(raw_name).strip()
        normalized = normalize_player_name(display)
        if not normalized:
            raise ValueError(f"Quarterback name is empty after normalization: {raw_name!r}")
        if normalized in lookup:
            continue
        lookup[normalized] = display
        names.append(display)
    if not names:
        raise ValueError("At least one quarterback name is required.")
    return names, lookup


def _partition_by_play(frame: pl.DataFrame) -> dict[tuple[str, str], pl.DataFrame]:
    """Partition a DataFrame into ``(GAME_ID, PLAY_ID)`` lookup entries."""
    if frame.is_empty():
        return {}
    partitions = frame.partition_by(
        [GAME_ID, PLAY_ID],
        as_dict=True,
        maintain_order=False,
    )
    output: dict[tuple[str, str], pl.DataFrame] = {}
    for raw_key, partition in partitions.items():
        key_values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        if len(key_values) != 2:
            raise ValueError(f"Unexpected partition key: {raw_key!r}")
        output[(normalize_id(key_values[0]), normalize_id(key_values[1]))] = partition
    return output


def _metadata_from_row(row: dict[str, object]) -> dict[str, object]:
    """Return the shared quarterback/play metadata for result and skip records."""
    return {
        "qb_name": row.get("qb_name"),
        "qb_player_id": row.get("qb_player_id"),
        "qb_team": row.get("qb_team"),
        GAME_ID: row.get(GAME_ID),
        PLAY_ID: row.get(PLAY_ID),
    }


def _skip_record(
    row: dict[str, object],
    *,
    stage: str,
    reason: str,
    details: object = "",
) -> dict[str, object]:
    """Build one stable skipped-play record."""
    return {
        **_metadata_from_row(row),
        "stage": stage,
        "skip_reason": reason,
        "details": "" if details is None else str(details),
    }


def load_qb_passer_plays(
    offense_csv: Path,
    qb_names: list[str],
    name_lookup: dict[str, str],
    max_plays_per_qb: int | None,
) -> pl.DataFrame:
    """Load exact passer-play keys for the requested quarterbacks."""
    requested_keys = list(name_lookup)
    lookup_frame = pl.DataFrame(
        {
            "normalized_qb_name": requested_keys,
            "qb_name": [name_lookup[key] for key in requested_keys],
        }
    )

    passer_plays = (
        pl.scan_csv(offense_csv, infer_schema=False)
        .select(
            GAME_ID,
            PLAY_ID,
            "pff_TEAM",
            "pff_PLAYERID",
            "pff_PLAYERNAME",
            "pff_POSITION",
            "pff_PASSER",
        )
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id("pff_PLAYERID").alias("pff_PLAYERID"),
            clean_upper("pff_TEAM").alias("pff_TEAM"),
            clean_upper("pff_POSITION").alias("pff_POSITION"),
            normalized_name_expr("pff_PLAYERNAME").alias("normalized_qb_name"),
        )
        .filter(
            pl.col("normalized_qb_name").is_in(requested_keys)
            & truthy_expr("pff_PASSER")
            & (pl.col(GAME_ID) != "")
            & (pl.col(PLAY_ID) != "")
            & (pl.col("pff_PLAYERID") != "")
        )
        .select(
            GAME_ID,
            PLAY_ID,
            pl.col("pff_TEAM").alias("qb_team"),
            pl.col("pff_PLAYERID").alias("qb_player_id"),
            pl.col("pff_PLAYERNAME").alias("source_qb_name"),
            "normalized_qb_name",
        )
        .unique()
        .collect()
        .join(lookup_frame, on="normalized_qb_name", how="left")
        .select(
            "qb_name",
            "qb_player_id",
            "qb_team",
            "source_qb_name",
            GAME_ID,
            PLAY_ID,
        )
        .sort(["qb_name", GAME_ID, PLAY_ID])
    )

    identity_conflicts = (
        passer_plays
        .select("qb_name", "qb_player_id")
        .unique()
        .group_by("qb_name")
        .agg(pl.col("qb_player_id").n_unique().alias("player_ids"))
        .filter(pl.col("player_ids") > 1)
    )
    if not identity_conflicts.is_empty():
        raise ValueError(
            "A requested quarterback resolved to multiple pff_PLAYERID values:\n"
            + str(identity_conflicts)
        )

    play_conflicts = (
        passer_plays
        .group_by([GAME_ID, PLAY_ID])
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    if not play_conflicts.is_empty():
        raise ValueError(
            "Multiple requested quarterbacks were marked as passer on the same play:\n"
            + str(play_conflicts)
        )

    if max_plays_per_qb is not None and not passer_plays.is_empty():
        limited: list[pl.DataFrame] = []
        for qb_name in qb_names:
            limited.append(
                passer_plays
                .filter(pl.col("qb_name") == qb_name)
                .sort([GAME_ID, PLAY_ID])
                .head(max_plays_per_qb)
            )
        passer_plays = pl.concat(limited, how="vertical_relaxed")

    return passer_plays


def load_candidate_pbp(
    pbp_csv: Path,
    qb_plays: pl.DataFrame,
) -> pl.DataFrame:
    """Load PBP rows for discovered quarterback passer-play keys."""
    metadata = qb_plays.select(
        "qb_name",
        "qb_player_id",
        "qb_team",
        GAME_ID,
        PLAY_ID,
    ).unique()
    return (
        pl.scan_csv(pbp_csv, infer_schema=False)
        .select(PBP_COLUMNS)
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
            pl.col("pff_PASSDEPTH").cast(pl.Float64, strict=False),
            pl.col("pff_PASSWIDTH").cast(pl.Float64, strict=False),
            pl.col("pff_TIMETOTHROW").cast(pl.Float64, strict=False),
            pl.col("pff_YARDSTOGOALLINE").cast(pl.Float64, strict=False),
            clean_upper("pff_RUNPASS").alias("pff_RUNPASS"),
            clean_upper("pff_PASSRESULT").alias("pff_PASSRESULT"),
            clean_upper("pff_PASSDIRECTION").alias("pff_PASSDIRECTION"),
            clean_upper("pff_INCOMPLETIONTYPE").alias("pff_INCOMPLETIONTYPE"),
        )
        .join(metadata.lazy(), on=[GAME_ID, PLAY_ID], how="inner")
        .collect()
        .unique()
        .sort(["qb_name", GAME_ID, PLAY_ID])
    )


def _load_player_rows(
    csv_path: Path,
    columns: list[str],
    play_keys: pl.DataFrame,
) -> pl.DataFrame:
    """Load and clean all player-play rows for a batch of selected plays."""
    return (
        pl.scan_csv(csv_path, infer_schema=False)
        .select(columns)
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
            clean_id("pff_PLAYERID").alias("pff_PLAYERID"),
            clean_id("pff_GSISPLAYERID").alias("pff_GSISPLAYERID"),
            clean_upper("pff_TEAM").alias("pff_TEAM"),
            clean_upper("pff_ROLE").alias("pff_ROLE"),
            clean_upper("pff_POSITION").alias("pff_POSITION"),
            clean_upper("pff_GAMEPOSITION").alias("pff_GAMEPOSITION"),
        )
        .join(play_keys.lazy(), on=[GAME_ID, PLAY_ID], how="inner")
        .collect()
        .sort([GAME_ID, PLAY_ID, "pff_PLAYERNAME"])
    )


def load_tracking_for_plays(
    tracking_csv: Path,
    eligible_pbp: pl.DataFrame,
) -> pl.DataFrame:
    """Scan tracking once and retain only the eligible PFF/GSIS play keys."""
    if eligible_pbp.is_empty():
        return pl.DataFrame()

    tracking_keys = (
        eligible_pbp
        .select(
            GAME_ID,
            PLAY_ID,
            pl.col(PLAY_ID).alias("pff_play_id"),
            pl.col(GSIS_GAME_KEY).alias("game_key"),
            pl.col(GSIS_PLAY_ID).alias("gsis_play_id"),
        )
        .filter(
            (pl.col("pff_play_id") != "")
            & (pl.col("game_key") != "")
            & (pl.col("gsis_play_id") != "")
        )
        .unique()
    )
    play_ids = tracking_keys.get_column("pff_play_id").unique().to_list()

    return (
        pl.scan_csv(tracking_csv, infer_schema=False)
        .select(TRACKING_COLUMNS)
        .with_columns(
            clean_id("pff_play_id").alias("pff_play_id"),
            clean_id("game_key").alias("game_key"),
            clean_id("gsis_play_id").alias("gsis_play_id"),
            clean_id("team_id").alias("team_id"),
            clean_id("pro_player_id").alias("pro_player_id"),
            pl.col("player_name")
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .alias("player_name"),
            pl.col("event")
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .str.to_lowercase()
            .alias("event_normalized"),
            pl.col("X").cast(pl.Float64, strict=False),
            pl.col("Y").cast(pl.Float64, strict=False),
            pl.col("rel_x").cast(pl.Float64, strict=False),
            pl.col("dist_to_ball").cast(pl.Float64, strict=False),
            pl.col("time_into_play").cast(pl.Float64, strict=False),
            pl.col("orientation").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("pff_play_id").is_in(play_ids)
            & pl.col("time_into_play").is_not_null()
            & pl.col("X").is_not_null()
            & pl.col("Y").is_not_null()
        )
        .join(
            tracking_keys.lazy(),
            on=["pff_play_id", "game_key", "gsis_play_id"],
            how="inner",
        )
        .collect()
        .sort([GAME_ID, PLAY_ID, "time_into_play", "pro_player_id"])
    )


def _release_arrays(
    release_state: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Convert the release-state table into KATC model arrays."""
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


def evaluate_katc_play(
    metadata: dict[str, object],
    pbp_play: pl.DataFrame,
    offense_rows: pl.DataFrame,
    defense_rows: pl.DataFrame,
    tracking: pl.DataFrame,
    parameters: CoverageParameters,
    ownership_mode: str,
) -> dict[str, object]:
    """Evaluate one eligible play at the exact PFF target using KATC only."""
    snap_time = event_time(tracking, "ball_snap")
    release_time = event_time(tracking, "pass_forward")

    snap_ball = ball_row_at_time(tracking, snap_time)
    ball_row_at_time(tracking, release_time)
    snap_ball_y = float(snap_ball["Y"])
    if abs(snap_ball_y) > 0.50:
        raise ValueError(
            f"Football is not near the LOS at ball_snap: Y={snap_ball_y:.3f}."
        )

    release_frame = tracking.filter(pl.col("time_into_play") == release_time)
    duplicate_entities = (
        release_frame
        .group_by("pro_player_id")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    if not duplicate_entities.is_empty():
        raise ValueError(
            "Duplicate tracking entities at pass_forward:\n"
            + str(duplicate_entities)
        )

    generator_audit = resolve_generators(offense_rows, defense_rows, tracking)
    resolved_generators = require_resolved_generators(generator_audit)
    release_state = build_release_state(
        tracking,
        resolved_generators,
        release_time,
        parameters,
    )

    if release_state.height != resolved_generators.height:
        raise ValueError(
            "Release-state rows do not reconcile to resolved generators: "
            f"{release_state.height} vs {resolved_generators.height}."
        )

    offense_count = release_state.filter(pl.col("side") == "OFFENSE").height
    defense_count = release_state.filter(pl.col("side") == "DEFENSE").height
    if offense_count == 0 or defense_count == 0:
        raise ValueError(
            "KATC requires at least one route runner and one coverage defender; "
            f"found offense={offense_count}, defense={defense_count}."
        )

    maximum_speed = float(
        release_state.get_column("speed_yards_per_second").max()
    )
    if not math.isfinite(maximum_speed):
        raise ValueError("Release-state velocity contains a non-finite speed.")

    target = pass_target(pbp_play)
    if abs(float(target[0])) > FIELD_HALF_WIDTH_YARDS + 1e-6:
        raise ValueError(
            f"PFF target X is outside the sidelines: {target[0]:.3f}."
        )

    intended_receiver = intended_receiver_name(offense_rows)
    if (
        ownership_mode == OWNERSHIP_MODE_INTENDED_RECEIVER
        and intended_receiver is None
    ):
        raise ValueError(
            "INTENDED_RECEIVER_ONLY requires one resolved targeted receiver."
        )

    positions, velocities, names, sides = _release_arrays(release_state)
    katc = evaluate_kinematic_target(
        target=target,
        positions=positions,
        velocities=velocities,
        names=names,
        sides=sides,
        ownership_mode=ownership_mode,
        intended_receiver_name=intended_receiver,
        parameters=parameters,
    )

    finite_values = [
        katc.owner_value,
        katc.offense_value,
        katc.defense_value,
        katc.control_margin,
    ]
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("KATC returned a non-finite exact-target result.")

    pbp = pbp_play.row(0, named=True)
    pff_time_to_throw = pbp.get("pff_TIMETOTHROW")
    release_difference = (
        None
        if pff_time_to_throw is None
        else abs(release_time - float(pff_time_to_throw))
    )

    agrees = bool(katc.target_in_offense_control)
    return {
        **_metadata_from_row(metadata),
        GSIS_GAME_KEY: pbp.get(GSIS_GAME_KEY),
        GSIS_PLAY_ID: pbp.get(GSIS_PLAY_ID),
        "pff_GAMEDATE": pbp.get("pff_GAMEDATE"),
        "pff_WEEK": pbp.get("pff_WEEK"),
        "pff_QUARTER": pbp.get("pff_QUARTER"),
        "pff_DOWN": pbp.get("pff_DOWN"),
        "pff_CLOCK": pbp.get("pff_CLOCK"),
        "pff_OFFTEAM": pbp.get("pff_OFFTEAM"),
        "pff_DEFTEAM": pbp.get("pff_DEFTEAM"),
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "intended_receiver_name": intended_receiver,
        "ownership_mode": ownership_mode,
        "snap_time": snap_time,
        "release_time": release_time,
        "pff_time_to_throw_s": pff_time_to_throw,
        "release_time_difference_s": release_difference,
        "route_runners": offense_count,
        "coverage_defenders": defense_count,
        "max_observed_speed_yards_per_second": maximum_speed,
        "observed_speed_exceeds_model_cap": (
            maximum_speed > parameters.max_player_speed_yards_per_second
        ),
        "katc_target_owner_name": katc.owner_name,
        "katc_target_owner_side": katc.owner_side,
        "katc_target_owner_arrival_time_s": katc.owner_value,
        "katc_offense_reference_name": katc.offense_reference_name,
        "katc_offense_arrival_time_s": katc.offense_value,
        "katc_defense_reference_name": katc.defense_reference_name,
        "katc_defense_arrival_time_s": katc.defense_value,
        "katc_control_margin_s": katc.control_margin,
        "katc_target_in_offense_control": agrees,
        "katc_model_agrees_with_throw": agrees,
        "katc_target_near_boundary": katc.target_near_boundary,
        "velocity_lookback_seconds": parameters.velocity_lookback_seconds,
        "max_player_speed_yards_per_second": (
            parameters.max_player_speed_yards_per_second
        ),
        "max_player_acceleration_yards_per_second_squared": (
            parameters.max_player_acceleration_yards_per_second_squared
        ),
        "katc_target_time_step_seconds": (
            parameters.kinematic_target_time_step_seconds
        ),
        "katc_target_arrival_radius_yards": (
            parameters.kinematic_target_arrival_radius_yards
        ),
    }


def wilson_interval(successes: int, attempts: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if attempts <= 0:
        return math.nan, math.nan
    proportion = successes / attempts
    denominator = 1.0 + (z * z / attempts)
    center = (proportion + z * z / (2.0 * attempts)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / attempts
            + z * z / (4.0 * attempts * attempts)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _count_unique_plays_by_qb(frame: pl.DataFrame, qb_name: str) -> int:
    """Return unique skipped play count for one quarterback."""
    if frame.is_empty() or "qb_name" not in frame.columns:
        return 0
    return (
        frame
        .filter(pl.col("qb_name") == qb_name)
        .select(GAME_ID, PLAY_ID)
        .unique()
        .height
    )


def compile_rankings(
    qb_names: list[str],
    qb_plays: pl.DataFrame,
    eligible_plays: pl.DataFrame,
    play_results: pl.DataFrame,
    skipped: pl.DataFrame,
    minimum_attempts: int,
    parameters: CoverageParameters,
    ownership_mode: str,
) -> pl.DataFrame:
    """Compile quarterback-level KATC agreement rates and ranks."""
    records: list[dict[str, object]] = []

    for qb_name in qb_names:
        candidates = qb_plays.filter(pl.col("qb_name") == qb_name)
        eligible = eligible_plays.filter(pl.col("qb_name") == qb_name)
        evaluated = (
            play_results.filter(pl.col("qb_name") == qb_name)
            if not play_results.is_empty()
            else _empty_frame(PLAY_RESULT_SCHEMA)
        )

        candidate_count = candidates.height
        eligible_count = eligible.height
        evaluated_count = evaluated.height
        agreements = (
            int(
                evaluated
                .get_column("katc_model_agrees_with_throw")
                .cast(pl.Int64)
                .sum()
            )
            if evaluated_count
            else 0
        )
        disagreements = evaluated_count - agreements

        margins = (
            np.asarray(
                evaluated.get_column("katc_control_margin_s").to_list(),
                dtype=float,
            )
            if evaluated_count
            else np.asarray([], dtype=float)
        )
        near_boundary = (
            int(
                evaluated
                .get_column("katc_target_near_boundary")
                .cast(pl.Int64)
                .sum()
            )
            if evaluated_count
            else 0
        )
        low, high = wilson_interval(agreements, evaluated_count)

        player_ids = (
            sorted(
                str(value)
                for value in candidates
                .get_column("qb_player_id")
                .drop_nulls()
                .unique()
                if str(value).strip()
            )
            if candidate_count
            else []
        )
        teams = (
            sorted(
                str(value)
                for value in candidates.get_column("qb_team").drop_nulls().unique()
                if str(value).strip()
            )
            if candidate_count
            else []
        )

        records.append(
            {
                "katc_rank": None,
                "rank_status": (
                    "RANKED"
                    if evaluated_count >= minimum_attempts
                    else (
                        "NO_EVALUATIONS"
                        if evaluated_count == 0
                        else "INSUFFICIENT_SAMPLE"
                    )
                ),
                "qb_name": qb_name,
                "qb_player_id": "|".join(str(value) for value in player_ids),
                "teams": "|".join(teams),
                "candidate_passer_plays": candidate_count,
                "strict_eligible_passes": eligible_count,
                "katc_evaluated_attempts": evaluated_count,
                "katc_agreement_attempts": agreements,
                "katc_disagreement_attempts": disagreements,
                "katc_agreement_rate_pct": (
                    100.0 * agreements / evaluated_count
                    if evaluated_count
                    else None
                ),
                "katc_agreement_wilson_low_pct": (
                    100.0 * low if evaluated_count else None
                ),
                "katc_agreement_wilson_high_pct": (
                    100.0 * high if evaluated_count else None
                ),
                "mean_katc_control_margin_s": (
                    float(np.mean(margins)) if evaluated_count else None
                ),
                "median_katc_control_margin_s": (
                    float(np.median(margins)) if evaluated_count else None
                ),
                "near_boundary_attempts": near_boundary,
                "near_boundary_rate_pct": (
                    100.0 * near_boundary / evaluated_count
                    if evaluated_count
                    else None
                ),
                "strict_eligibility_rate_pct": (
                    100.0 * eligible_count / candidate_count
                    if candidate_count
                    else None
                ),
                "model_evaluation_coverage_pct": (
                    100.0 * evaluated_count / eligible_count
                    if eligible_count
                    else None
                ),
                "eligible_not_evaluated": eligible_count - evaluated_count,
                "skipped_candidate_plays": _count_unique_plays_by_qb(
                    skipped, qb_name
                ),
                "ownership_mode": ownership_mode,
                "minimum_attempts_for_rank": minimum_attempts,
                "velocity_lookback_seconds": parameters.velocity_lookback_seconds,
                "max_player_speed_yards_per_second": (
                    parameters.max_player_speed_yards_per_second
                ),
                "max_player_acceleration_yards_per_second_squared": (
                    parameters.max_player_acceleration_yards_per_second_squared
                ),
                "katc_target_time_step_seconds": (
                    parameters.kinematic_target_time_step_seconds
                ),
                "katc_target_arrival_radius_yards": (
                    parameters.kinematic_target_arrival_radius_yards
                ),
            }
        )

    ranked_records = [
        record
        for record in records
        if record["rank_status"] == "RANKED"
    ]
    ranked_records.sort(
        key=lambda record: (
            -float(record["katc_agreement_rate_pct"]),
            -float(record["mean_katc_control_margin_s"]),
            -int(record["katc_evaluated_attempts"]),
            str(record["qb_name"]),
        )
    )
    ranks = {str(record["qb_name"]): index + 1 for index, record in enumerate(ranked_records)}
    for record in records:
        record["katc_rank"] = ranks.get(str(record["qb_name"]))

    records.sort(
        key=lambda record: (
            record["katc_rank"] is None,
            int(record["katc_rank"]) if record["katc_rank"] is not None else 10**9,
            str(record["qb_name"]),
        )
    )
    return pl.DataFrame(records)


def main() -> int:
    """Run the batch KATC ranking workflow."""
    args = parse_args()
    _validate_inputs(args)

    requested_names = args.qb or list(DEFAULT_QUARTERBACKS)
    qb_names, name_lookup = _canonical_quarterbacks(requested_names)

    parameters = replace(
        DEFAULT_PARAMETERS,
        velocity_lookback_seconds=args.velocity_lookback,
        min_velocity_observations=args.minimum_velocity_observations,
        max_player_speed_yards_per_second=args.max_speed,
        max_player_acceleration_yards_per_second_squared=args.max_acceleration,
        kinematic_target_time_step_seconds=args.target_time_step,
        kinematic_target_arrival_radius_yards=args.target_arrival_radius,
        kinematic_max_time_seconds=args.max_arrival_time,
        target_near_boundary_margin_seconds=args.near_boundary_seconds,
    )
    parameters.validate()

    args.output_directory.mkdir(parents=True, exist_ok=True)

    print("Discovering quarterback passer plays...")
    qb_plays = load_qb_passer_plays(
        args.offense_csv,
        qb_names,
        name_lookup,
        args.max_plays_per_qb,
    )
    if qb_plays.is_empty():
        print("No passer plays were found for the requested quarterbacks.")

    skipped_rows: list[dict[str, object]] = []

    print("Loading candidate PBP rows...")
    candidate_pbp = load_candidate_pbp(args.pbp_csv, qb_plays)
    pbp_keys = candidate_pbp.select(GAME_ID, PLAY_ID).unique()

    missing_pbp = qb_plays.join(pbp_keys, on=[GAME_ID, PLAY_ID], how="anti")
    for row in missing_pbp.iter_rows(named=True):
        skipped_rows.append(
            _skip_record(
                row,
                stage="PBP_JOIN",
                reason="NO_PBP_ROW",
                details="No PBP row matched pff_GAMEID + pff_PLAYID.",
            )
        )

    duplicate_pbp_keys = (
        candidate_pbp
        .group_by([GAME_ID, PLAY_ID])
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    duplicate_key_set = {
        (normalize_id(row[GAME_ID]), normalize_id(row[PLAY_ID]))
        for row in duplicate_pbp_keys.iter_rows(named=True)
    }
    if duplicate_key_set:
        duplicate_key_frame = pl.DataFrame(
            {
                GAME_ID: [key[0] for key in sorted(duplicate_key_set)],
                PLAY_ID: [key[1] for key in sorted(duplicate_key_set)],
            }
        )
        duplicate_metadata = qb_plays.join(
            duplicate_key_frame,
            on=[GAME_ID, PLAY_ID],
            how="inner",
        )
        for row in duplicate_metadata.iter_rows(named=True):
            skipped_rows.append(
                _skip_record(
                    row,
                    stage="PBP_JOIN",
                    reason="DUPLICATE_PBP_ROWS",
                    details="More than one distinct PBP row exists for the play key.",
                )
            )
        candidate_pbp = candidate_pbp.join(
            duplicate_key_frame,
            on=[GAME_ID, PLAY_ID],
            how="anti",
        )

    play_keys = candidate_pbp.select(GAME_ID, PLAY_ID).unique()
    print("Loading offense and defense player-play rows...")
    offense_rows = _load_player_rows(args.offense_csv, OFFENSE_COLUMNS, play_keys)
    defense_rows = _load_player_rows(args.defense_csv, DEFENSE_COLUMNS, play_keys)

    pbp_parts = _partition_by_play(candidate_pbp)
    offense_parts = _partition_by_play(offense_rows)
    defense_parts = _partition_by_play(defense_rows)
    empty_offense = offense_rows.head(0)
    empty_defense = defense_rows.head(0)

    eligible_metadata: list[dict[str, object]] = []
    eligible_pbp_parts: list[pl.DataFrame] = []

    print("Applying the strict regular-pass eligibility rules...")
    for metadata in qb_plays.iter_rows(named=True):
        key = (normalize_id(metadata[GAME_ID]), normalize_id(metadata[PLAY_ID]))
        pbp_play = pbp_parts.get(key)
        if pbp_play is None:
            continue
        defense_play = defense_parts.get(key, empty_defense)
        reasons = regular_pass_exclusion_reasons(pbp_play, defense_play)
        if reasons:
            skipped_rows.append(
                _skip_record(
                    metadata,
                    stage="ELIGIBILITY",
                    reason="STRICT_PASS_EXCLUSION",
                    details=";".join(reasons),
                )
            )
            if args.verbose:
                print(f"SKIP {key[0]}:{key[1]} — {';'.join(reasons)}")
            continue

        eligible_metadata.append(metadata)
        eligible_pbp_parts.append(pbp_play)

    eligible_plays = (
        pl.DataFrame(eligible_metadata)
        if eligible_metadata
        else qb_plays.head(0)
    )
    eligible_pbp = (
        pl.concat(eligible_pbp_parts, how="vertical_relaxed")
        if eligible_pbp_parts
        else candidate_pbp.head(0)
    )

    print(
        "Scanning tracking once for "
        f"{eligible_plays.height} strict-eligible passer plays..."
    )
    tracking_rows = load_tracking_for_plays(args.tracking_csv, eligible_pbp)
    tracking_parts = _partition_by_play(tracking_rows)

    result_rows: list[dict[str, object]] = []
    total_eligible = eligible_plays.height
    print("Evaluating exact-target KATC control...")
    for index, metadata in enumerate(
        eligible_plays.iter_rows(named=True),
        start=1,
    ):
        key = (normalize_id(metadata[GAME_ID]), normalize_id(metadata[PLAY_ID]))
        pbp_play = pbp_parts[key]
        offense_play = offense_parts.get(key, empty_offense)
        defense_play = defense_parts.get(key, empty_defense)
        tracking_play = tracking_parts.get(key)

        if tracking_play is None or tracking_play.is_empty():
            skipped_rows.append(
                _skip_record(
                    metadata,
                    stage="TRACKING_JOIN",
                    reason="NO_TRACKING_ROWS",
                    details=(
                        "No tracking rows matched pff_PLAYID + pff_GSISGAMEKEY + "
                        "pff_GSISPLAYID."
                    ),
                )
            )
            if args.verbose:
                print(f"SKIP {key[0]}:{key[1]} — no tracking rows")
            continue

        try:
            result = evaluate_katc_play(
                metadata,
                pbp_play,
                offense_play,
                defense_play,
                tracking_play,
                parameters,
                args.ownership_mode,
            )
        except Exception as error:  # noqa: BLE001 - recorded per play by design
            if args.fail_fast:
                raise
            skipped_rows.append(
                _skip_record(
                    metadata,
                    stage="KATC_EVALUATION",
                    reason=type(error).__name__,
                    details=error,
                )
            )
            if args.verbose:
                print(f"SKIP {key[0]}:{key[1]} — {type(error).__name__}: {error}")
            continue

        result_rows.append(result)
        if args.verbose:
            side = "OFFENSE" if result["katc_model_agrees_with_throw"] else "DEFENSE"
            print(
                f"PASS {key[0]}:{key[1]} — {side}, "
                f"margin={result['katc_control_margin_s']:+.3f} s"
            )
        elif index % 25 == 0 or index == total_eligible:
            print(f"Processed {index}/{total_eligible} eligible plays")

    play_results = (
        pl.DataFrame(result_rows, schema=PLAY_RESULT_SCHEMA)
        .sort(["qb_name", GAME_ID, PLAY_ID])
        if result_rows
        else _empty_frame(PLAY_RESULT_SCHEMA)
    )
    skipped = (
        pl.DataFrame(skipped_rows, schema=SKIPPED_SCHEMA)
        .sort(["qb_name", GAME_ID, PLAY_ID, "stage"])
        if skipped_rows
        else _empty_frame(SKIPPED_SCHEMA)
    )

    rankings = compile_rankings(
        qb_names=qb_names,
        qb_plays=qb_plays,
        eligible_plays=eligible_plays,
        play_results=play_results,
        skipped=skipped,
        minimum_attempts=args.minimum_attempts,
        parameters=parameters,
        ownership_mode=args.ownership_mode,
    )

    play_results_path = args.output_directory / PLAY_RESULTS_FILENAME
    rankings_path = args.output_directory / RANKINGS_FILENAME
    skipped_path = args.output_directory / SKIPPED_FILENAME

    play_results.write_csv(play_results_path)
    rankings.write_csv(rankings_path)
    skipped.write_csv(skipped_path)

    display_columns = [
        "katc_rank",
        "qb_name",
        "katc_evaluated_attempts",
        "katc_agreement_attempts",
        "katc_agreement_rate_pct",
        "mean_katc_control_margin_s",
        "model_evaluation_coverage_pct",
        "rank_status",
    ]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=160):
        print("\nKATC quarterback ranking")
        print(rankings.select(display_columns))

    print("\nAgreement definition:")
    print(
        "  The observed pass target was offense-controlled at pass_forward "
        "under KATC. Play result is not part of the ranking."
    )
    print(f"\nWrote play results: {play_results_path}")
    print(f"Wrote quarterback rankings: {rankings_path}")
    print(f"Wrote skipped-play audit: {skipped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
