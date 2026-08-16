"""Print NFC North quarterbacks discovered by the production query logic."""

from nfc_north_player_tracking.qb_queries import find_team_quarterbacks

OFFENSE_CSV = "data/raw/pff_offense.csv"
NFC_NORTH_TEAMS = ["CHI", "DET", "GB", "MIN"]


if __name__ == "__main__":
    print(find_team_quarterbacks(OFFENSE_CSV, NFC_NORTH_TEAMS))
