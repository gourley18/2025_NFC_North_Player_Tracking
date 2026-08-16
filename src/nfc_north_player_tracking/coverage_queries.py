"""Load and classify play-level PFF and tracking data for coverage analysis."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from nfc_north_player_tracking.config import PFF_PASS_WIDTH_CENTER


GAME_ID = "pff_GAMEID"
PLAY_ID = "pff_PLAYID"
GSIS_GAME_KEY = "pff_GSISGAMEKEY"
GSIS_PLAY_ID = "pff_GSISPLAYID"

PBP_COLUMNS = [
    GAME_ID,
    PLAY_ID,
    GSIS_GAME_KEY,
    GSIS_PLAY_ID,
    "pff_GAMEDATE",
    "pff_WEEK",
    "pff_QUARTER",
    "pff_DOWN",
    "pff_CLOCK",
    "pff_OFFTEAM",
    "pff_DEFTEAM",
    "pff_RUNPASS",
    "pff_PASSRESULT",
    "pff_PASSDEPTH",
    "pff_PASSWIDTH",
    "pff_PASSDIRECTION",
    "pff_PASSRECEIVERTARGET",
    "pff_SCREEN",
    "pff_RUNPASSOPTION",
    "pff_TRICKPLAY",
    "pff_TRICKLOOK",
    "pff_INCOMPLETIONTYPE",
    "pff_NOPLAY",
    "pff_TIMETOTHROW",
    "pff_YARDSTOGOALLINE",
]

OFFENSE_COLUMNS = [
    GAME_ID,
    PLAY_ID,
    GSIS_GAME_KEY,
    GSIS_PLAY_ID,
    "pff_TEAM",
    "pff_PLAYERID",
    "pff_GSISPLAYERID",
    "pff_PLAYERNAME",
    "pff_POSITION",
    "pff_GAMEPOSITION",
    "pff_ROLE",
    "pff_PASSER",
    "pff_QB",
    "pff_TARGETEDRECEIVER",
]

DEFENSE_COLUMNS = [
    GAME_ID,
    PLAY_ID,
    GSIS_GAME_KEY,
    GSIS_PLAY_ID,
    "pff_TEAM",
    "pff_PLAYERID",
    "pff_GSISPLAYERID",
    "pff_PLAYERNAME",
    "pff_POSITION",
    "pff_GAMEPOSITION",
    "pff_ROLE",
    "pff_BATTEDPASS",
]

TRACKING_COLUMNS = [
    "pff_play_id",
    "game_key",
    "gsis_play_id",
    "team_id",
    "pro_player_id",
    "player_name",
    "event",
    "X",
    "Y",
    "rel_x",
    "dist_to_ball",
    "time_into_play",
    "orientation",
]

TRUTHY_VALUES = {"Y", "YES", "TRUE", "T", "1", "1.0"}
ELIGIBLE_PASS_RESULTS = {
    "COMPLETE",
    "COMPLETION",
    "INCOMPLETE",
    "INCOMPLETION",
    "INTERCEPTION",
    "INTERCEPTED",
    "INT",
}
RESULT_EXCLUSIONS = {
    "THROWNAWAY": "THROWAWAY",
    "THROWAWAY": "THROWAWAY",
    "BATTEDPASS": "BATTED_PASS",
    "HITASTHREW": "HIT_AS_THREW",
    "SPIKE": "SPIKE",
    "SACK": "NON_THROW_SACK",
    "RUN": "NON_THROW_RUN",
    "LATERAL": "LATERAL",
}
INCOMPLETION_TYPE_EXCLUSIONS = {
    "TA": "THROWAWAY",
    "BP": "BATTED_PASS",
    "HA": "HIT_AS_THREW",
}


def normalize_id_value(value: object) -> str:
    """Normalize an identifier such as ``10635.0`` to ``10635``."""
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_token(value: object) -> str:
    """Return uppercase alphanumeric text for category comparisons."""
    text = "" if value is None else str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def is_truthy(value: object) -> bool:
    """Recognize the common truthy encodings in PFF CSV exports."""
    return ("" if value is None else str(value).strip().upper()) in TRUTHY_VALUES


def clean_id(column: str) -> pl.Expr:
    """Return a Polars expression that normalizes an identifier column."""
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.replace(r"\.0$", "")
    )


def clean_upper(column: str) -> pl.Expr:
    """Return trimmed uppercase text for a CSV column."""
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )


def truthy_expr(column: str) -> pl.Expr:
    """Return a Boolean Polars expression for a PFF flag column."""
    return clean_upper(column).is_in(sorted(TRUTHY_VALUES))


def _scan_exact_play(
    csv_path: Path,
    columns: list[str],
    game_id: str,
    play_id: str,
) -> pl.LazyFrame:
    """Scan selected columns and retain one PFF game/play key."""
    return (
        pl.scan_csv(csv_path, infer_schema=False)
        .select(columns)
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
        )
        .filter((pl.col(GAME_ID) == game_id) & (pl.col(PLAY_ID) == play_id))
    )


def load_pbp_play(pbp_csv: Path, game_id: object, play_id: object) -> pl.DataFrame:
    """Load the exact PBP row that defines a coverage-analysis play."""
    game_id = normalize_id_value(game_id)
    play_id = normalize_id_value(play_id)
    play = (
        _scan_exact_play(pbp_csv, PBP_COLUMNS, game_id, play_id)
        .with_columns(
            pl.col("pff_PASSDEPTH").cast(pl.Float64, strict=False),
            pl.col("pff_PASSWIDTH").cast(pl.Float64, strict=False),
            pl.col("pff_TIMETOTHROW").cast(pl.Float64, strict=False),
            pl.col("pff_YARDSTOGOALLINE").cast(pl.Float64, strict=False),
            clean_upper("pff_RUNPASS").alias("pff_RUNPASS"),
            clean_upper("pff_PASSRESULT").alias("pff_PASSRESULT"),
            clean_upper("pff_PASSDIRECTION").alias("pff_PASSDIRECTION"),
            clean_upper("pff_INCOMPLETIONTYPE").alias("pff_INCOMPLETIONTYPE"),
        )
        .collect()
        .unique()
    )
    if play.height != 1:
        raise ValueError(
            f"Expected exactly one PBP row for {game_id}:{play_id}; "
            f"found {play.height}."
        )
    return play


def load_offense_play(
    offense_csv: Path,
    game_id: object,
    play_id: object,
) -> pl.DataFrame:
    """Load all offensive player-play rows for one play."""
    game_id = normalize_id_value(game_id)
    play_id = normalize_id_value(play_id)
    return (
        _scan_exact_play(offense_csv, OFFENSE_COLUMNS, game_id, play_id)
        .with_columns(
            clean_id("pff_PLAYERID").alias("pff_PLAYERID"),
            clean_id("pff_GSISPLAYERID").alias("pff_GSISPLAYERID"),
            clean_upper("pff_TEAM").alias("pff_TEAM"),
            clean_upper("pff_ROLE").alias("pff_ROLE"),
            clean_upper("pff_POSITION").alias("pff_POSITION"),
            clean_upper("pff_GAMEPOSITION").alias("pff_GAMEPOSITION"),
        )
        .collect()
        .sort("pff_PLAYERNAME")
    )


def load_defense_play(
    defense_csv: Path,
    game_id: object,
    play_id: object,
) -> pl.DataFrame:
    """Load all defensive player-play rows for one play."""
    game_id = normalize_id_value(game_id)
    play_id = normalize_id_value(play_id)
    return (
        _scan_exact_play(defense_csv, DEFENSE_COLUMNS, game_id, play_id)
        .with_columns(
            clean_id("pff_PLAYERID").alias("pff_PLAYERID"),
            clean_id("pff_GSISPLAYERID").alias("pff_GSISPLAYERID"),
            clean_upper("pff_TEAM").alias("pff_TEAM"),
            clean_upper("pff_ROLE").alias("pff_ROLE"),
            clean_upper("pff_POSITION").alias("pff_POSITION"),
            clean_upper("pff_GAMEPOSITION").alias("pff_GAMEPOSITION"),
        )
        .collect()
        .sort("pff_PLAYERNAME")
    )


def load_tracking_play(
    tracking_csv: Path,
    pbp_play: pl.DataFrame,
) -> pl.DataFrame:
    """Load tracking observations matching all three available play keys."""
    key = pbp_play.row(0, named=True)
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
            (pl.col("pff_play_id") == key[PLAY_ID])
            & (pl.col("game_key") == key[GSIS_GAME_KEY])
            & (pl.col("gsis_play_id") == key[GSIS_PLAY_ID])
            & pl.col("time_into_play").is_not_null()
            & pl.col("X").is_not_null()
            & pl.col("Y").is_not_null()
        )
        .collect()
        .sort(["time_into_play", "pro_player_id"])
    )


def pass_target(pbp_play: pl.DataFrame) -> tuple[float, float]:
    """Return the centered PFF pass target in tracking-field coordinates."""
    row = pbp_play.row(0, named=True)
    depth = row.get("pff_PASSDEPTH")
    width = row.get("pff_PASSWIDTH")
    if depth is None or width is None:
        raise ValueError("The play is missing PFF pass depth or width.")
    return float(width) - PFF_PASS_WIDTH_CENTER, float(depth)


def intended_receiver_name(offense_rows: pl.DataFrame) -> str | None:
    """Return the one PFF player marked as the targeted receiver."""
    targeted = offense_rows.filter(truthy_expr("pff_TARGETEDRECEIVER"))
    if targeted.is_empty():
        return None
    names = targeted.get_column("pff_PLAYERNAME").drop_nulls().unique().to_list()
    names = [str(name).strip() for name in names if str(name).strip()]
    if len(names) != 1:
        raise ValueError(
            "Expected zero or one targeted receiver name; found "
            f"{len(names)}: {names}."
        )
    return names[0]


def passer_name(offense_rows: pl.DataFrame) -> str | None:
    """Return the offensive player marked as the passer/QB for the play."""
    passer = offense_rows.filter(
        truthy_expr("pff_PASSER")
        | truthy_expr("pff_QB")
        | (pl.col("pff_POSITION") == "QB")
    )
    if passer.is_empty():
        return None
    names = passer.get_column("pff_PLAYERNAME").drop_nulls().unique().to_list()
    names = [str(name).strip() for name in names if str(name).strip()]
    if len(names) != 1:
        return None
    return names[0]


def has_defensive_batted_pass(defense_rows: pl.DataFrame) -> bool:
    """Return whether any defender is marked with a batted pass."""
    if defense_rows.is_empty():
        return False
    return bool(
        defense_rows
        .select(truthy_expr("pff_BATTEDPASS").any().alias("value"))
        .item()
    )


def regular_pass_exclusion_reasons(
    pbp_play: pl.DataFrame,
    defense_rows: pl.DataFrame,
) -> list[str]:
    """Return every reason a play is outside the initial regular-pass scope."""
    row = pbp_play.row(0, named=True)
    reasons: list[str] = []

    result = normalize_token(row.get("pff_PASSRESULT"))
    incompletion_type = normalize_token(row.get("pff_INCOMPLETIONTYPE"))

    if normalize_token(row.get("pff_RUNPASS")) != "P":
        reasons.append("NOT_PFF_PASS")
    if is_truthy(row.get("pff_NOPLAY")):
        reasons.append("NO_PLAY")
    if is_truthy(row.get("pff_SCREEN")):
        reasons.append("SCREEN")
    if is_truthy(row.get("pff_RUNPASSOPTION")):
        reasons.append("RPO")
    if is_truthy(row.get("pff_TRICKPLAY")) or is_truthy(row.get("pff_TRICKLOOK")):
        reasons.append("TRICK_PLAY")

    if result not in ELIGIBLE_PASS_RESULTS:
        reasons.append(RESULT_EXCLUSIONS.get(result, "UNSUPPORTED_PASS_RESULT"))

    incompletion_reason = INCOMPLETION_TYPE_EXCLUSIONS.get(incompletion_type)
    if incompletion_reason:
        reasons.append(incompletion_reason)

    if has_defensive_batted_pass(defense_rows):
        reasons.append("BATTED_PASS")

    if row.get("pff_PASSDEPTH") is None or row.get("pff_PASSWIDTH") is None:
        reasons.append("MISSING_TARGET_LOCATION")

    target = row.get("pff_PASSRECEIVERTARGET")
    if target is None or not str(target).strip():
        reasons.append("MISSING_INTENDED_RECEIVER")

    return list(dict.fromkeys(reasons))
