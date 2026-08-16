"""Validate the temporary PFF-name -> tracking-player bridge.

Run this before any Level 1 tessellation analysis. The test resolves only the
players that would become generators:

* pff_offense.pff_ROLE == "Pass Route"
* pff_defense.pff_ROLE == "Coverage"

Matching is exact after deterministic Unicode/case/punctuation normalization.
No fuzzy matching is performed. A selected play fails when any expected player
is unmatched, ambiguous, duplicated in the PFF role rows, or maps to a
conflicting tracking ID across the selected plays.

Example from the repository root:

    python tests/validate_tracking_player_names.py \
        --play 28430:6408068
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = REPO_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from nfc_north_player_tracking.config import (  # noqa: E402
    DEFENSE_COVERAGE_ROLE,
    OFFENSE_ROUTE_ROLE,
)

GAME_ID = "pff_GAMEID"
PLAY_ID = "pff_PLAYID"
GSIS_GAME_KEY = "pff_GSISGAMEKEY"
GSIS_PLAY_ID = "pff_GSISPLAYID"

DEFAULT_PLAYS = ["28430:6408068"]
FAILURE_STATUSES = {
    "EMPTY_PFF_NAME",
    "UNMATCHED",
    "AMBIGUOUS",
    "DUPLICATE_PFF_ROLE_ROW",
    "INCONSISTENT_PRO_PLAYER_ID",
}


def normalize_id_value(value: object) -> str:
    """Normalize an identifier such as 10635.0 to 10635."""
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_player_name(value: object) -> str:
    """Return a deterministic punctuation-insensitive player-name key."""
    text = "" if value is None else str(value)
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    return "".join(character for character in decomposed if character.isalnum())


def clean_id(column: str) -> pl.Expr:
    """Return a Polars expression that normalizes an ID column."""
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.replace(r"\.0$", "")
    )


def clean_upper(column: str) -> pl.Expr:
    """Return trimmed uppercase text."""
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )


def normalized_name(column: str, alias: str) -> pl.Expr:
    """Apply the deterministic Python normalizer to a Polars column."""
    return (
        pl.col(column)
        .map_elements(normalize_player_name, return_dtype=pl.String)
        .alias(alias)
    )


def parse_play(value: str) -> tuple[str, str]:
    """Parse GAME_ID:PLAY_ID from the command line."""
    pieces = value.split(":", maxsplit=1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Play must use GAME_ID:PLAY_ID format.")
    game_id = normalize_id_value(pieces[0])
    play_id = normalize_id_value(pieces[1])
    if not game_id or not play_id:
        raise argparse.ArgumentTypeError("Both GAME_ID and PLAY_ID are required.")
    return game_id, play_id


def load_pbp_key(
    pbp_csv: Path,
    game_id: str,
    play_id: str,
) -> dict[str, str]:
    """Return the single PBP key row used to locate tracking observations."""
    key = (
        pl.scan_csv(pbp_csv, infer_schema=False)
        .select([GAME_ID, PLAY_ID, GSIS_GAME_KEY, GSIS_PLAY_ID])
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
        )
        .filter(
            (pl.col(GAME_ID) == game_id)
            & (pl.col(PLAY_ID) == play_id)
        )
        .collect()
        .unique()
    )

    if key.height != 1:
        raise ValueError(
            f"Expected exactly one PBP key for {game_id}:{play_id}; "
            f"found {key.height}."
        )

    return key.row(0, named=True)


def load_expected_generators(
    offense_csv: Path,
    defense_csv: Path,
    game_id: str,
    play_id: str,
) -> pl.DataFrame:
    """Load PFF route runners and coverage defenders for one play."""
    offense = (
        pl.scan_csv(offense_csv, infer_schema=False)
        .select(
            [
                GAME_ID,
                PLAY_ID,
                "pff_TEAM",
                "pff_PLAYERID",
                "pff_PLAYERNAME",
                "pff_ROLE",
            ]
        )
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id("pff_PLAYERID").alias("pff_PLAYERID"),
            clean_upper("pff_ROLE").alias("pff_ROLE"),
            normalized_name("pff_PLAYERNAME", "normalized_pff_name"),
        )
        .filter(
            (pl.col(GAME_ID) == game_id)
            & (pl.col(PLAY_ID) == play_id)
            & (pl.col("pff_ROLE") == OFFENSE_ROUTE_ROLE)
        )
        .with_columns(
            pl.lit("OFFENSE").alias("side"),
            pl.lit("ROUTE_RUNNER").alias("analysis_role"),
        )
        .collect()
    )

    defense = (
        pl.scan_csv(defense_csv, infer_schema=False)
        .select(
            [
                GAME_ID,
                PLAY_ID,
                "pff_TEAM",
                "pff_PLAYERID",
                "pff_PLAYERNAME",
                "pff_ROLE",
            ]
        )
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id("pff_PLAYERID").alias("pff_PLAYERID"),
            clean_upper("pff_ROLE").alias("pff_ROLE"),
            normalized_name("pff_PLAYERNAME", "normalized_pff_name"),
        )
        .filter(
            (pl.col(GAME_ID) == game_id)
            & (pl.col(PLAY_ID) == play_id)
            & (pl.col("pff_ROLE") == DEFENSE_COVERAGE_ROLE)
        )
        .with_columns(
            pl.lit("DEFENSE").alias("side"),
            pl.lit("COVERAGE_DEFENDER").alias("analysis_role"),
        )
        .collect()
    )

    expected = pl.concat([offense, defense], how="vertical_relaxed")
    if expected.is_empty():
        return expected

    duplicate_key = [
        GAME_ID,
        PLAY_ID,
        "side",
        "pff_PLAYERID",
        "pff_ROLE",
    ]
    return (
        expected
        .with_columns(
            pl.len().over(duplicate_key).alias("pff_role_row_count")
        )
        .with_row_index("_pff_row_index")
        .sort(["side", "pff_PLAYERNAME"])
    )


def load_tracking_players(
    tracking_csv: Path,
    key: dict[str, str],
) -> pl.DataFrame:
    """Load unique non-ball tracking players for the exact PBP play key."""
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
            ]
        )
        .with_columns(
            clean_id("pff_play_id").alias("pff_play_id"),
            clean_id("game_key").alias("game_key"),
            clean_id("gsis_play_id").alias("gsis_play_id"),
            clean_id("pro_player_id").alias("pro_player_id"),
            clean_upper("player_name").alias("_player_name_upper"),
            normalized_name("player_name", "normalized_tracking_name"),
        )
        .filter(
            (pl.col("pff_play_id") == key[PLAY_ID])
            & (pl.col("game_key") == key[GSIS_GAME_KEY])
            & (pl.col("gsis_play_id") == key[GSIS_PLAY_ID])
            & (pl.col("pro_player_id") != "")
            & (pl.col("pro_player_id") != "-1")
            & (pl.col("_player_name_upper") != "BALL")
            & (pl.col("normalized_tracking_name") != "")
        )
        .select(
            "team_id",
            "pro_player_id",
            pl.col("player_name").alias("tracking_player_name"),
            "normalized_tracking_name",
        )
        .unique()
        .collect()
        .sort(["normalized_tracking_name", "pro_player_id"])
    )


def build_name_bridge_audit(
    expected: pl.DataFrame,
    tracking_players: pl.DataFrame,
) -> pl.DataFrame:
    """Build one human-readable audit row per PFF/candidate match."""
    if expected.is_empty():
        return pl.DataFrame(
            schema={
                GAME_ID: pl.String,
                PLAY_ID: pl.String,
                "pff_PLAYERNAME": pl.String,
                "match_status": pl.String,
            }
        )

    joined = expected.join(
        tracking_players,
        left_on="normalized_pff_name",
        right_on="normalized_tracking_name",
        how="left",
    )

    joined = joined.with_columns(
        pl.col("pro_player_id")
        .is_not_null()
        .sum()
        .over("_pff_row_index")
        .alias("candidate_count")
    )

    return (
        joined
        .with_columns(
            pl.when(pl.col("pff_role_row_count") > 1)
            .then(pl.lit("DUPLICATE_PFF_ROLE_ROW"))
            .when(pl.col("normalized_pff_name") == "")
            .then(pl.lit("EMPTY_PFF_NAME"))
            .when(pl.col("candidate_count") == 0)
            .then(pl.lit("UNMATCHED"))
            .when(pl.col("candidate_count") > 1)
            .then(pl.lit("AMBIGUOUS"))
            .otherwise(pl.lit("MATCHED"))
            .alias("match_status"),
            pl.lit("NORMALIZED_NAME").alias("resolution_method"),
        )
        .select(
            GAME_ID,
            PLAY_ID,
            "pff_TEAM",
            "side",
            "analysis_role",
            "pff_ROLE",
            "pff_PLAYERID",
            "pff_PLAYERNAME",
            "normalized_pff_name",
            "tracking_player_name",
            "pro_player_id",
            "team_id",
            "candidate_count",
            "pff_role_row_count",
            "resolution_method",
            "match_status",
        )
        .sort([GAME_ID, PLAY_ID, "side", "pff_PLAYERNAME", "pro_player_id"])
    )


def add_cross_play_consistency(audit: pl.DataFrame) -> pl.DataFrame:
    """Flag names that map to multiple tracking IDs across selected plays."""
    if audit.is_empty():
        return audit

    # PFF player ID is the authoritative identity. Two distinct players can
    # legitimately normalize to the same name, so cross-play consistency must
    # not be keyed by name alone.
    conflicts = (
        audit
        .filter(pl.col("match_status") == "MATCHED")
        .group_by("pff_PLAYERID")
        .agg(pl.col("pro_player_id").n_unique().alias("cross_play_id_count"))
        .filter(pl.col("cross_play_id_count") > 1)
        .select("pff_PLAYERID")
        .with_columns(pl.lit(True).alias("has_cross_play_id_conflict"))
    )

    return (
        audit
        .join(conflicts, on="pff_PLAYERID", how="left")
        .with_columns(
            pl.col("has_cross_play_id_conflict").fill_null(False),
            pl.when(
                pl.col("has_cross_play_id_conflict")
                & (pl.col("match_status") == "MATCHED")
            )
            .then(pl.lit("INCONSISTENT_PRO_PLAYER_ID"))
            .otherwise(pl.col("match_status"))
            .alias("match_status"),
        )
    )


def build_summary(audit: pl.DataFrame) -> pl.DataFrame:
    """Summarize matches by play, side, and status."""
    if audit.is_empty():
        return pl.DataFrame()
    return (
        audit
        .group_by([GAME_ID, PLAY_ID, "side", "match_status"])
        .agg(pl.len().alias("players"))
        .sort([GAME_ID, PLAY_ID, "side", "match_status"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--play",
        action="append",
        default=None,
        metavar="GAME_ID:PLAY_ID",
        help="Selected play. Repeat for more than one play.",
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
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write the audit but return success even when matches fail.",
    )
    args = parser.parse_args()

    selected_plays = [parse_play(value) for value in (args.play or DEFAULT_PLAYS)]
    args.output_directory.mkdir(parents=True, exist_ok=True)

    audit_tables: list[pl.DataFrame] = []
    for game_id, play_id in selected_plays:
        key = load_pbp_key(args.pbp_csv, game_id, play_id)
        expected = load_expected_generators(
            args.offense_csv,
            args.defense_csv,
            game_id,
            play_id,
        )
        if expected.is_empty():
            raise ValueError(
                f"No Pass Route/Coverage rows found for {game_id}:{play_id}."
            )
        tracking_players = load_tracking_players(args.tracking_csv, key)
        audit_tables.append(build_name_bridge_audit(expected, tracking_players))

    audit = add_cross_play_consistency(
        pl.concat(audit_tables, how="vertical_relaxed")
    )
    summary = build_summary(audit)

    audit_path = args.output_directory / "player_name_bridge_audit.csv"
    summary_path = args.output_directory / "player_name_bridge_summary.csv"
    audit.write_csv(audit_path)
    summary.write_csv(summary_path)

    print("\nPlayer-name bridge audit")
    print(audit)
    print("\nSummary")
    print(summary)
    print(f"\nWrote: {audit_path}")
    print(f"Wrote: {summary_path}")

    failed = audit.filter(pl.col("match_status").is_in(sorted(FAILURE_STATUSES)))
    if not failed.is_empty() and not args.no_strict:
        print("\nFAIL: resolve every listed failure before Level 1 analysis.")
        return 1

    print("\nPASS: all selected route runners and coverage defenders resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
