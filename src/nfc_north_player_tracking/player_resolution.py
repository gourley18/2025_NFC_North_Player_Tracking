"""Resolve PFF player-play rows to tracking entities for one play."""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict

import polars as pl

from nfc_north_player_tracking.config import (
    DEFENSE_COVERAGE_ROLE,
    OFFENSE_ROUTE_ROLE,
)
from nfc_north_player_tracking.coverage_queries import truthy_expr


MATCHED = "MATCHED"
UNMATCHED = "UNMATCHED"
AMBIGUOUS = "AMBIGUOUS"
EMPTY_PFF_NAME = "EMPTY_PFF_NAME"
DUPLICATE_PFF_ROLE_ROW = "DUPLICATE_PFF_ROLE_ROW"

FAILURE_STATUSES = {
    UNMATCHED,
    AMBIGUOUS,
    EMPTY_PFF_NAME,
    DUPLICATE_PFF_ROLE_ROW,
}


def normalize_player_name(value: object) -> str:
    """Return a deterministic case/punctuation-insensitive player-name key."""
    text = "" if value is None else str(value)
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    return "".join(character for character in decomposed if character.isalnum())


def _expected_generator_rows(
    offense_rows: pl.DataFrame,
    defense_rows: pl.DataFrame,
) -> list[dict[str, object]]:
    """Return route runners and coverage defenders as Python dictionaries."""
    offense = (
        offense_rows
        .filter(pl.col("pff_ROLE") == OFFENSE_ROUTE_ROLE)
        .with_columns(
            pl.lit("OFFENSE").alias("side"),
            pl.lit("ROUTE_RUNNER").alias("analysis_role"),
            truthy_expr("pff_TARGETEDRECEIVER").alias("targeted_receiver"),
        )
        .select(
            "pff_GAMEID",
            "pff_PLAYID",
            "pff_TEAM",
            "pff_PLAYERID",
            "pff_GSISPLAYERID",
            "pff_PLAYERNAME",
            "pff_ROLE",
            "side",
            "analysis_role",
            "targeted_receiver",
        )
    )

    defense = (
        defense_rows
        .filter(pl.col("pff_ROLE") == DEFENSE_COVERAGE_ROLE)
        .with_columns(
            pl.lit("DEFENSE").alias("side"),
            pl.lit("COVERAGE_DEFENDER").alias("analysis_role"),
            pl.lit(False).alias("targeted_receiver"),
        )
        .select(
            "pff_GAMEID",
            "pff_PLAYID",
            "pff_TEAM",
            "pff_PLAYERID",
            "pff_GSISPLAYERID",
            "pff_PLAYERNAME",
            "pff_ROLE",
            "side",
            "analysis_role",
            "targeted_receiver",
        )
    )

    return pl.concat([offense, defense], how="vertical_relaxed").to_dicts()


def _tracking_player_rows(tracking: pl.DataFrame) -> list[dict[str, object]]:
    """Return distinct non-ball tracking entities for one play."""
    return (
        tracking
        .filter(
            (pl.col("pro_player_id") != "-1")
            & (
                pl.col("player_name")
                .cast(pl.String)
                .fill_null("")
                .str.strip_chars()
                .str.to_lowercase()
                != "ball"
            )
        )
        .select("pro_player_id", "player_name", "team_id")
        .unique()
        .sort("player_name")
        .to_dicts()
    )


def resolve_generators(
    offense_rows: pl.DataFrame,
    defense_rows: pl.DataFrame,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    """Build an audit table for the temporary normalized-name bridge."""
    expected = _expected_generator_rows(offense_rows, defense_rows)
    tracking_players = _tracking_player_rows(tracking)

    role_counts = Counter(
        (
            str(row["side"]),
            normalize_player_name(row.get("pff_PLAYERNAME")),
        )
        for row in expected
    )

    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in tracking_players:
        candidates[normalize_player_name(row.get("player_name"))].append(row)

    audit_rows: list[dict[str, object]] = []
    for row in expected:
        normalized = normalize_player_name(row.get("pff_PLAYERNAME"))
        matches = candidates.get(normalized, [])
        role_count = role_counts[(str(row["side"]), normalized)]

        if not normalized:
            status = EMPTY_PFF_NAME
        elif role_count != 1:
            status = DUPLICATE_PFF_ROLE_ROW
        elif len(matches) == 0:
            status = UNMATCHED
        elif len(matches) > 1:
            status = AMBIGUOUS
        else:
            status = MATCHED

        match = matches[0] if len(matches) == 1 else {}
        audit_rows.append(
            {
                **row,
                "normalized_pff_name": normalized,
                "tracking_player_name": match.get("player_name"),
                "pro_player_id": match.get("pro_player_id"),
                "team_id": match.get("team_id"),
                "candidate_count": len(matches),
                "pff_role_row_count": role_count,
                "resolution_method": "NORMALIZED_NAME",
                "match_status": status,
            }
        )

    if not audit_rows:
        return pl.DataFrame(
            schema={
                "pff_GAMEID": pl.String,
                "pff_PLAYID": pl.String,
                "side": pl.String,
                "analysis_role": pl.String,
                "pff_PLAYERID": pl.String,
                "pff_PLAYERNAME": pl.String,
                "match_status": pl.String,
            }
        )

    return pl.DataFrame(audit_rows).sort(["side", "pff_PLAYERNAME"])


def require_resolved_generators(audit: pl.DataFrame) -> pl.DataFrame:
    """Return the matched generator table or raise with audit details."""
    failures = audit.filter(pl.col("match_status") != MATCHED)
    if not failures.is_empty():
        columns = [
            "side",
            "analysis_role",
            "pff_PLAYERNAME",
            "normalized_pff_name",
            "candidate_count",
            "pff_role_row_count",
            "match_status",
        ]
        raise ValueError(
            "Generator resolution failed:\n"
            + str(failures.select(columns))
        )

    duplicate_tracking_ids = (
        audit
        .group_by("pro_player_id")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    if not duplicate_tracking_ids.is_empty():
        raise ValueError(
            "Multiple generator rows resolved to the same tracking player:\n"
            + str(duplicate_tracking_ids)
        )

    return audit.with_columns(
        pl.col("pro_player_id").alias("tracking_player_id")
    )


def resolve_context_player(
    tracking: pl.DataFrame,
    player_name: str | None,
) -> dict[str, object] | None:
    """Resolve a single optional context player, such as the quarterback."""
    if player_name is None:
        return None
    normalized = normalize_player_name(player_name)
    matches = [
        row
        for row in _tracking_player_rows(tracking)
        if normalize_player_name(row.get("player_name")) == normalized
    ]
    if len(matches) != 1:
        return None
    return matches[0]
