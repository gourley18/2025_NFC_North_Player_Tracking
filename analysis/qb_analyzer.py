"""Create two Level 0 passing plots for each quarterback."""

import os

from nfc_north_player_tracking.field_plot import make_area_plot, make_pass_plot
from nfc_north_player_tracking.qb_diagnostics import build_qb_diagnostics
from nfc_north_player_tracking.qb_queries import (
    classify_pass_attempts,
    get_player_game_ids,
    get_qb_pass_play_keys,
    load_pbp_for_plays,
    load_player_offense_rows,
    split_passes_by_location,
    summarize_by_area,
)

OFFENSE_CSV = "data/raw/pff_offense.csv"
PBP_CSV = "data/raw/pff_pbp.csv"
OUTPUT_DIRECTORY = "outputs"

QUARTERBACKS = {
    10635: "Jared Goff",
    40306: "Jordan Love",
    144918: "JJ McCarthy",
    144622: "Caleb Williams",
}

def analyze_qb(player_id, player_name):
    """Query one quarterback, validate the data, and save two figures."""
    player_offense_rows = load_player_offense_rows(OFFENSE_CSV, player_id)

    if player_offense_rows.is_empty():
        raise ValueError(f"{player_name} ({player_id}) was not found in pff_offense.")

    game_ids = get_player_game_ids(player_offense_rows)
    pass_play_keys = get_qb_pass_play_keys(player_offense_rows)

    if pass_play_keys.is_empty():
        raise ValueError(f"No passer rows were found for {player_name}.")

    pbp_plays = load_pbp_for_plays(PBP_CSV, game_ids, pass_play_keys)

    if pbp_plays.is_empty():
        raise ValueError(f"No PBP rows matched {player_name}'s passer-play keys.")

    pass_attempts = classify_pass_attempts(pbp_plays)

    if pass_attempts.is_empty():
        raise ValueError(f"No pass attempts were found for {player_name}.")

    located_passes, unlocated_passes = split_passes_by_location(pass_attempts)

    if located_passes.is_empty():
        raise ValueError(f"No pass attempts had usable locations for {player_name}.")

    area_summary = summarize_by_area(located_passes)

    diagnostics = build_qb_diagnostics(
        player_id,
        player_offense_rows,
        pass_play_keys,
        pbp_plays,
        pass_attempts,
        located_passes,
        unlocated_passes,
        area_summary,
    )

    failed_checks = [
        row
        for row in diagnostics["summary"].iter_rows(named=True)
        if row["status"] == "FAIL"
    ]

    if failed_checks:
        print(diagnostics["summary"])
        raise ValueError(f"Diagnostic checks failed for {player_name}.")

    file_name = player_name.lower().replace(" ", "_")

    pass_figure = make_pass_plot(
        located_passes,
        f"{player_name}: Pass Attempts",
    )
    pass_figure.savefig(
        os.path.join(OUTPUT_DIRECTORY, f"{file_name}_pass_attempts.png"),
        dpi=200,
        bbox_inches="tight",
    )

    area_figure = make_area_plot(
        area_summary,
        f"{player_name}: Completion % by Field Area",
    )
    area_figure.savefig(
        os.path.join(OUTPUT_DIRECTORY, f"{file_name}_completion_by_area.png"),
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\n{player_name} ({player_id})")
    print(f"Games found: {len(game_ids)}")
    print(f"Passer-play keys: {pass_play_keys.height}")
    print(f"Matched PBP rows: {pbp_plays.height}")
    print(f"Pass attempts: {pass_attempts.height}")
    print(f"Located attempts plotted: {located_passes.height}")
    print(f"Unlocated attempts: {unlocated_passes.height}")
    print("Diagnostics: PASS")


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    for player_id, player_name in QUARTERBACKS.items():
        analyze_qb(player_id, player_name)

    print(f"\nCreated 8 figures in: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
