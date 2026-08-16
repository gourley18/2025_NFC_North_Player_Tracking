"""Audit and preliminarily classify the strict Level 1 pass population.

The output is intentionally PFF-only. Release-event and player-resolution
requirements are added after the corresponding validation gates pass.

Example:

    python tests/audit_pass_filters.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]

GAME_ID = "pff_GAMEID"
PLAY_ID = "pff_PLAYID"
GSIS_GAME_KEY = "pff_GSISGAMEKEY"
GSIS_PLAY_ID = "pff_GSISPLAYID"

AUDIT_FIELDS = [
    "pff_PASSRESULT",
    "pff_SCREEN",
    "pff_RUNPASSOPTION",
    "pff_TRICKPLAY",
    "pff_TRICKLOOK",
    "pff_INCOMPLETIONTYPE",
    "pff_NOPLAY",
    "pff_PASSDEPTH",
    "pff_PASSWIDTH",
    "pff_PASSDIRECTION",
    "pff_PASSRECEIVERTARGET",
]

ELIGIBLE_RESULTS = {"COMPLETE", "COMPLETION", "INTERCEPTION", "INTERCEPTED", "INT", "INCOMPLETE", "INCOMPLETION"}
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
TRUTHY_VALUES = {"Y", "YES", "TRUE", "T", "1", "1.0"}


def clean_id(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.replace(r"\.0$", "")
    )


def normalize_token(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def is_truthy(value: object) -> bool:
    return ("" if value is None else str(value).strip().upper()) in TRUTHY_VALUES


def exclusion_reasons(row: dict[str, object]) -> str:
    """Return every PFF-only Level 1 exclusion reason for one candidate row."""
    reasons: list[str] = []

    result = normalize_token(row.get("pff_PASSRESULT"))
    incompletion_type = normalize_token(row.get("pff_INCOMPLETIONTYPE"))

    if is_truthy(row.get("pff_NOPLAY")):
        reasons.append("NO_PLAY")
    if is_truthy(row.get("pff_SCREEN")):
        reasons.append("SCREEN")
    if is_truthy(row.get("pff_RUNPASSOPTION")):
        reasons.append("RPO")
    if is_truthy(row.get("pff_TRICKPLAY")) or is_truthy(row.get("pff_TRICKLOOK")):
        reasons.append("TRICK_PLAY")

    if result not in ELIGIBLE_RESULTS:
        reasons.append(RESULT_EXCLUSIONS.get(result, "UNSUPPORTED_PASS_RESULT"))

    incompletion_reason = INCOMPLETION_TYPE_EXCLUSIONS.get(incompletion_type)
    if incompletion_reason:
        reasons.append(incompletion_reason)

    if bool(row.get("has_defensive_batted_pass")):
        reasons.append("BATTED_PASS")

    if row.get("pff_PASSDEPTH") is None or row.get("pff_PASSWIDTH") is None:
        reasons.append("MISSING_TARGET_LOCATION")

    target = "" if row.get("pff_PASSRECEIVERTARGET") is None else str(row.get("pff_PASSRECEIVERTARGET")).strip()
    if not target:
        reasons.append("MISSING_INTENDED_RECEIVER")

    return ";".join(dict.fromkeys(reasons))


def load_defensive_batted_plays(defense_csv: Path) -> pl.DataFrame:
    return (
        pl.scan_csv(defense_csv, infer_schema=False)
        .select([GAME_ID, PLAY_ID, "pff_BATTEDPASS"])
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            pl.col("pff_BATTEDPASS")
            .map_elements(is_truthy, return_dtype=pl.Boolean)
            .alias("is_defensive_batted_pass"),
        )
        .filter((pl.col(GAME_ID) != "") & (pl.col(PLAY_ID) != ""))
        .group_by([GAME_ID, PLAY_ID])
        .agg(
            pl.col("is_defensive_batted_pass")
            .any()
            .alias("has_defensive_batted_pass")
        )
        .collect()
    )


def load_pass_candidates(pbp_csv: Path, defense_csv: Path) -> pl.DataFrame:
    columns = [
        GAME_ID,
        PLAY_ID,
        GSIS_GAME_KEY,
        GSIS_PLAY_ID,
        "pff_GAMEDATE",
        "pff_WEEK",
        "pff_OFFTEAM",
        "pff_DEFTEAM",
        "pff_RUNPASS",
        *AUDIT_FIELDS,
    ]
    candidates = (
        pl.scan_csv(pbp_csv, infer_schema=False)
        .select(columns)
        .with_columns(
            clean_id(GAME_ID).alias(GAME_ID),
            clean_id(PLAY_ID).alias(PLAY_ID),
            clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
            pl.col("pff_RUNPASS")
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .str.to_uppercase()
            .alias("pff_RUNPASS"),
            pl.col("pff_PASSDEPTH").cast(pl.Float64, strict=False),
            pl.col("pff_PASSWIDTH").cast(pl.Float64, strict=False),
        )
        .filter(
            (pl.col(GAME_ID) != "")
            & (pl.col(PLAY_ID) != "")
            & (pl.col("pff_RUNPASS") == "P")
        )
        .collect()
    )

    batted = load_defensive_batted_plays(defense_csv)
    candidates = candidates.join(
        batted,
        on=[GAME_ID, PLAY_ID],
        how="left",
    ).with_columns(
        pl.col("has_defensive_batted_pass").fill_null(False)
    )

    struct_columns = [
        "pff_PASSRESULT",
        "pff_SCREEN",
        "pff_RUNPASSOPTION",
        "pff_TRICKPLAY",
        "pff_TRICKLOOK",
        "pff_INCOMPLETIONTYPE",
        "pff_NOPLAY",
        "pff_PASSDEPTH",
        "pff_PASSWIDTH",
        "pff_PASSRECEIVERTARGET",
        "has_defensive_batted_pass",
    ]

    return (
        candidates
        .with_columns(
            pl.struct(struct_columns)
            .map_elements(exclusion_reasons, return_dtype=pl.String)
            .alias("level1_pff_exclusion_reasons")
        )
        .with_columns(
            (pl.col("level1_pff_exclusion_reasons") == "")
            .alias("level1_pff_eligible")
        )
        .sort([GAME_ID, PLAY_ID])
    )


def build_value_counts(candidates: pl.DataFrame) -> pl.DataFrame:
    tables: list[pl.DataFrame] = []
    for field in AUDIT_FIELDS:
        table = (
            candidates
            .with_columns(
                pl.col(field)
                .cast(pl.String)
                .fill_null("<NULL>")
                .str.strip_chars()
                .replace("", "<BLANK>")
                .alias("value")
            )
            .group_by("value")
            .agg(pl.len().alias("plays"))
            .with_columns(pl.lit(field).alias("field"))
            .select("field", "value", "plays")
        )
        tables.append(table)
    return pl.concat(tables, how="vertical_relaxed").sort(
        ["field", "plays", "value"],
        descending=[False, True, False],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pbp-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/pff_pbp.csv",
    )
    parser.add_argument(
        "--defense-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/pff_defense.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "outputs/diagnostics",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    candidates = load_pass_candidates(args.pbp_csv, args.defense_csv)
    value_counts = build_value_counts(candidates)

    candidates_path = args.output_directory / "pff_pass_candidates.csv"
    counts_path = args.output_directory / "pass_filter_value_counts.csv"
    candidates.write_csv(candidates_path)
    value_counts.write_csv(counts_path)

    summary = (
        candidates
        .group_by(
            ["level1_pff_eligible", "level1_pff_exclusion_reasons"]
        )
        .agg(pl.len().alias("plays"))
        .sort("plays", descending=True)
    )

    print("\nPFF-only Level 1 eligibility summary")
    print(summary)
    print(f"\nWrote: {candidates_path}")
    print(f"Wrote: {counts_path}")
    print(
        "\nNote: pass_forward, coordinate, and player-resolution requirements "
        "are intentionally not included until their validation gates pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
