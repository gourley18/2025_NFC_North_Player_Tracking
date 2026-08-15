"""Diagnostics for checking quarterback-query joins and row losses."""

import polars as pl

from nfc_north_player_tracking.qb_queries import (
    GAME_ID,
    GSIS_GAME_KEY,
    GSIS_PLAY_ID,
    OFFENSE_PLAYER_ID,
    OFFENSE_TEAM,
    PBP_OFFENSE_TEAM,
    PLAY_ID,
    get_qb_passer_rows,
    label_pass_outcomes,
)


def _duplicate_keys(data):
    """Return duplicated pff_GAMEID + pff_PLAYID combinations."""
    return (
        data
        .group_by([GAME_ID, PLAY_ID])
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
        .sort([GAME_ID, PLAY_ID])
    )


def build_qb_diagnostics(
    player_id,
    player_offense_rows,
    pass_play_keys,
    matched_pbp,
    pass_attempts,
    located_passes,
    unlocated_passes,
    area_summary,
):
    """Build audit tables that explain every join and filtering step."""
    player_id = str(player_id).strip().removesuffix(".0")
    passer_rows = get_qb_passer_rows(player_offense_rows)

    matched_keys = matched_pbp.select(GAME_ID, PLAY_ID).unique()
    qb_keys = pass_play_keys.select(GAME_ID, PLAY_ID).unique()

    unmatched_pass_keys = qb_keys.join(
        matched_keys,
        on=[GAME_ID, PLAY_ID],
        how="anti",
    ).sort([GAME_ID, PLAY_ID])

    leaked_pbp_keys = matched_keys.join(
        qb_keys,
        on=[GAME_ID, PLAY_ID],
        how="anti",
    ).sort([GAME_ID, PLAY_ID])

    duplicate_offense_passer_keys = _duplicate_keys(passer_rows)
    duplicate_pbp_keys = _duplicate_keys(matched_pbp)

    offense_key_details = (
        passer_rows
        .select(
            GAME_ID,
            PLAY_ID,
            GSIS_GAME_KEY,
            GSIS_PLAY_ID,
            OFFENSE_TEAM,
        )
        .unique()
    )

    pbp_key_details = (
        matched_pbp
        .select(
            GAME_ID,
            PLAY_ID,
            GSIS_GAME_KEY,
            GSIS_PLAY_ID,
            PBP_OFFENSE_TEAM,
        )
        .unique()
    )

    joined_key_details = offense_key_details.join(
        pbp_key_details,
        on=[GAME_ID, PLAY_ID],
        how="inner",
        suffix="_pbp",
    )

    foreign_key_mismatches = (
        joined_key_details
        .with_columns(
            (
                (pl.col(GSIS_GAME_KEY) != "")
                & (pl.col(f"{GSIS_GAME_KEY}_pbp") != "")
                & (pl.col(GSIS_GAME_KEY) != pl.col(f"{GSIS_GAME_KEY}_pbp"))
            ).alias("gsis_game_key_mismatch"),
            (
                (pl.col(GSIS_PLAY_ID) != "")
                & (pl.col(f"{GSIS_PLAY_ID}_pbp") != "")
                & (pl.col(GSIS_PLAY_ID) != pl.col(f"{GSIS_PLAY_ID}_pbp"))
            ).alias("gsis_play_id_mismatch"),
        )
        .filter(
            pl.col("gsis_game_key_mismatch")
            | pl.col("gsis_play_id_mismatch")
        )
        .sort([GAME_ID, PLAY_ID])
    )

    team_mismatches = (
        joined_key_details
        .with_columns(
            pl.col(OFFENSE_TEAM)
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .str.to_uppercase()
            .alias("offense_team_clean"),
            pl.col(PBP_OFFENSE_TEAM)
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .str.to_uppercase()
            .alias("pbp_offense_team_clean"),
        )
        .filter(
            (pl.col("offense_team_clean") != "")
            & (pl.col("pbp_offense_team_clean") != "")
            & (pl.col("offense_team_clean") != pl.col("pbp_offense_team_clean"))
        )
        .sort([GAME_ID, PLAY_ID])
    )

    labeled_pbp = label_pass_outcomes(matched_pbp)

    excluded_pbp_plays = (
        labeled_pbp
        .filter(
            pl.col("is_no_play")
            | pl.col("outcome").is_null()
        )
        .with_columns(
            pl.when(pl.col("is_no_play"))
            .then(pl.lit("NO_PLAY"))
            .when(pl.col("normalized_pass_result") == "")
            .then(pl.lit("MISSING_PASS_RESULT"))
            .otherwise(pl.lit("NON_ATTEMPT_OR_UNSUPPORTED_RESULT"))
            .alias("exclusion_reason")
        )
        .sort([GAME_ID, PLAY_ID])
    )

    excluded_result_counts = (
        excluded_pbp_plays
        .group_by(["exclusion_reason", "normalized_pass_result"])
        .agg(pl.len().alias("plays"))
        .sort(["exclusion_reason", "plays"], descending=[False, True])
    )

    wrong_player_rows = player_offense_rows.filter(
        pl.col(OFFENSE_PLAYER_ID) != player_id
    ).height

    matched_unique_count = matched_keys.height
    key_reconciliation_ok = (
        pass_play_keys.height
        == matched_unique_count + unmatched_pass_keys.height
    )

    pbp_reconciliation_ok = (
        matched_pbp.height
        == pass_attempts.height + excluded_pbp_plays.height
    )

    location_reconciliation_ok = (
        pass_attempts.height
        == located_passes.height + unlocated_passes.height
    )

    area_attempts = (
        area_summary.get_column("attempts").sum()
        if area_summary.height > 0
        else 0
    )
    area_reconciliation_ok = area_attempts == located_passes.height

    checks = [
        {
            "check": "Only requested player in offense rows",
            "status": "PASS" if wrong_player_rows == 0 else "FAIL",
            "count": wrong_player_rows,
            "details": "Rows belonging to another pff_PLAYERID",
        },
        {
            "check": "No duplicate QB passer keys in offense",
            "status": "PASS" if duplicate_offense_passer_keys.height == 0 else "FAIL",
            "count": duplicate_offense_passer_keys.height,
            "details": "Duplicate pff_GAMEID + pff_PLAYID keys",
        },
        {
            "check": "No PBP rows leaked from other plays",
            "status": "PASS" if leaked_pbp_keys.height == 0 else "FAIL",
            "count": leaked_pbp_keys.height,
            "details": "Matched PBP keys not present in QB passer keys",
        },
        {
            "check": "No duplicate matched PBP keys",
            "status": "PASS" if duplicate_pbp_keys.height == 0 else "FAIL",
            "count": duplicate_pbp_keys.height,
            "details": "Duplicate pff_GAMEID + pff_PLAYID rows can amplify counts",
        },
        {
            "check": "GSIS identifiers agree after PFF join",
            "status": "PASS" if foreign_key_mismatches.height == 0 else "FAIL",
            "count": foreign_key_mismatches.height,
            "details": "Secondary foreign keys disagree on a matched game/play",
        },
        {
            "check": "Offense team agrees after join",
            "status": "PASS" if team_mismatches.height == 0 else "FAIL",
            "count": team_mismatches.height,
            "details": "pff_offense.pff_TEAM differs from pff_pbp.pff_OFFTEAM",
        },
        {
            "check": "QB play keys reconcile",
            "status": "PASS" if key_reconciliation_ok else "FAIL",
            "count": unmatched_pass_keys.height,
            "details": "QB keys = matched unique keys + unmatched keys",
        },
        {
            "check": "Matched PBP rows reconcile to attempts/exclusions",
            "status": "PASS" if pbp_reconciliation_ok else "FAIL",
            "count": excluded_pbp_plays.height,
            "details": "Matched rows = classified attempts + explicitly excluded rows",
        },
        {
            "check": "Pass attempts reconcile to location split",
            "status": "PASS" if location_reconciliation_ok else "FAIL",
            "count": unlocated_passes.height,
            "details": "Attempts = located + unlocated",
        },
        {
            "check": "Area summary reconciles to plotted attempts",
            "status": "PASS" if area_reconciliation_ok else "FAIL",
            "count": int(area_attempts),
            "details": "Sum of area attempts equals located pass count",
        },
    ]

    return {
        "summary": pl.DataFrame(checks),
        "unmatched_pass_keys": unmatched_pass_keys,
        "leaked_pbp_keys": leaked_pbp_keys,
        "duplicate_offense_passer_keys": duplicate_offense_passer_keys,
        "duplicate_pbp_keys": duplicate_pbp_keys,
        "foreign_key_mismatches": foreign_key_mismatches,
        "team_mismatches": team_mismatches,
        "excluded_pbp_plays": excluded_pbp_plays,
        "excluded_result_counts": excluded_result_counts,
    }
