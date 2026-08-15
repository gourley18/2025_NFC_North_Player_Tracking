# """Query and prepare quarterback passing data from the PFF CSV files."""

# import polars as pl

# # pff_offense identifies the player and the plays where that player was the passer.
# OFFENSE_PLAYER_ID = "pff_PLAYERID"
# OFFENSE_PLAYER_NAME = "pff_PLAYERNAME"
# OFFENSE_TEAM = "pff_TEAM"
# OFFENSE_PASSER_FLAG = "pff_PASSER"

# # These play identifiers connect pff_offense to pff_pbp.
# GAME_ID = "pff_GAMEID"
# PLAY_ID = "pff_PLAYID"
# GSIS_GAME_KEY = "pff_GSISGAMEKEY"
# GSIS_PLAY_ID = "pff_GSISPLAYID"

# # pff_pbp supplies the Level 0 pass result and target location.
# PASS_RESULT = "pff_PASSRESULT"
# PASS_DEPTH = "pff_PASSDEPTH"
# PASS_DIRECTION = "pff_PASSDIRECTION"
# PASS_WIDTH = "pff_PASSWIDTH"
# NO_PLAY = "pff_NOPLAY"

# OFFENSE_COLUMNS = [
#     OFFENSE_PLAYER_ID,
#     OFFENSE_PLAYER_NAME,
#     OFFENSE_TEAM,
#     OFFENSE_PASSER_FLAG,
#     GAME_ID,
#     PLAY_ID,
#     GSIS_GAME_KEY,
#     GSIS_PLAY_ID,
# ]

# # There is intentionally no pff_PASSER column in this list. The quarterback
# # has already been identified in pff_offense before pff_pbp is queried.
# PBP_COLUMNS = [
#     GAME_ID,
#     PLAY_ID,
#     GSIS_GAME_KEY,
#     GSIS_PLAY_ID,
#     "pff_GAMEDATE",
#     "pff_WEEK",
#     "pff_QUARTER",
#     "pff_DOWN",
#     "pff_CLOCK",
#     "pff_OFFTEAM",
#     PASS_RESULT,
#     PASS_DEPTH,
#     PASS_DIRECTION,
#     PASS_WIDTH,
#     NO_PLAY,
# ]

# TRACKING_COLUMNS = [
#     "pff_play_id",
#     "game_key",
#     "gsis_play_id",
#     "team_id",
#     "pro_player_id",
#     "player_name",
#     "event",
#     "X",
#     "Y",
#     "rel_x",
#     "dist_to_ball",
#     "time_into_play",
#     "orientation",
# ]


# def _clean_text(column):
#     """Return a CSV column as trimmed uppercase text."""
#     return (
#         pl.col(column)
#         .cast(pl.String)
#         .fill_null("")
#         .str.strip_chars()
#         .str.to_uppercase()
#     )


# def _clean_id(column):
#     """Normalize IDs so values such as 10635 and 10635.0 compare equally."""
#     return _clean_text(column).str.replace(r"\.0$", "")


# def _is_yes(column):
#     """Recognize common true values used by PFF flag columns."""
#     return _clean_text(column).is_in(["Y", "YES", "TRUE", "T", "1", "1.0"])


# def load_player_offense_rows(csv_path, player_id):
#     """Load every pff_offense row for one PFF player ID."""
#     player_id = str(player_id).strip().removesuffix(".0")

#     return (
#         pl.read_csv(
#             csv_path,
#             columns=OFFENSE_COLUMNS,
#             infer_schema=False,
#         )
#         .with_columns(
#             _clean_id(OFFENSE_PLAYER_ID).alias(OFFENSE_PLAYER_ID),
#             _clean_id(GAME_ID).alias(GAME_ID),
#             _clean_id(PLAY_ID).alias(PLAY_ID),
#             _clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
#             _clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
#         )
#         .filter(pl.col(OFFENSE_PLAYER_ID) == player_id)
#         .sort([GAME_ID, PLAY_ID])
#     )


# def get_player_game_ids(player_offense_rows):
#     """Return the player's distinct pff_GAMEID values as a Python set."""
#     return {
#         game_id
#         for game_id in player_offense_rows.get_column(GAME_ID).to_list()
#         if game_id
#     }


# def get_qb_pass_play_keys(player_offense_rows):
#     """Return play keys where the player's pff_offense row marks him passer."""
#     return (
#         player_offense_rows
#         .filter(_is_yes(OFFENSE_PASSER_FLAG))
#         .select(GAME_ID, PLAY_ID, GSIS_GAME_KEY, GSIS_PLAY_ID)
#         .filter(
#             (pl.col(GAME_ID) != "")
#             & (pl.col(PLAY_ID) != "")
#         )
#         .unique()
#         .sort([GAME_ID, PLAY_ID])
#     )


# def load_pbp_for_plays(csv_path, game_ids, pass_play_keys):
#     """Load pff_pbp rows matching the exact game/play keys from pff_offense."""
#     pbp = (
#         pl.read_csv(
#             csv_path,
#             columns=PBP_COLUMNS,
#             infer_schema=False,
#         )
#         .with_columns(
#             _clean_id(GAME_ID).alias(GAME_ID),
#             _clean_id(PLAY_ID).alias(PLAY_ID),
#             _clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
#             _clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
#         )
#         .filter(
#             (pl.col(GAME_ID) != "")
#             & (pl.col(PLAY_ID) != "")
#         )
#     )

#     if game_ids:
#         pbp = pbp.filter(pl.col(GAME_ID).is_in(sorted(game_ids)))

#     keys = pass_play_keys.select(GAME_ID, PLAY_ID).unique()

#     return (
#         pbp
#         .join(
#             keys,
#             on=[GAME_ID, PLAY_ID],
#             how="inner",
#         )
#         .filter(~_is_yes(NO_PLAY))
#         .sort([GAME_ID, PLAY_ID])
#     )


# def load_tracking_for_plays(csv_path, pbp_plays):
#     """Load tracking rows for PBP plays using all available play identifiers."""
#     tracking = (
#         pl.read_csv(
#             csv_path,
#             columns=TRACKING_COLUMNS,
#             infer_schema=False,
#         )
#         .with_columns(
#             _clean_id("pff_play_id").alias("pff_play_id"),
#             _clean_id("game_key").alias("game_key"),
#             _clean_id("gsis_play_id").alias("gsis_play_id"),
#         )
#         .filter(pl.col("pff_play_id") != "")
#     )

#     tracking_keys = (
#         pbp_plays
#         .select(
#             pl.col(PLAY_ID).alias("pff_play_id"),
#             pl.col(GSIS_GAME_KEY).alias("game_key"),
#             pl.col(GSIS_PLAY_ID).alias("gsis_play_id"),
#         )
#         .filter(
#             (pl.col("pff_play_id") != "")
#             & (pl.col("game_key") != "")
#             & (pl.col("gsis_play_id") != "")
#         )
#         .unique()
#     )

#     return tracking.join(
#         tracking_keys,
#         on=["pff_play_id", "game_key", "gsis_play_id"],
#         how="inner",
#     )


# def classify_pass_attempts(pbp_plays):
#     """Keep official pass-attempt outcomes and label their result and side."""
#     result = _clean_text(PASS_RESULT).str.replace_all(r"[^A-Z0-9]", "")
#     direction = _clean_text(PASS_DIRECTION).str.replace_all(r"[^A-Z]", "")

#     outcome = (
#         pl.when(result.is_in(["INTERCEPTION", "INTERCEPTED", "INT"]))
#         .then(pl.lit("Interception"))
#         .when(result.is_in(["COMPLETE", "COMPLETION", "COMP", "C"]))
#         .then(pl.lit("Completion"))
#         .when(
#             result.is_in(
#                 [
#                     "INCOMPLETE",
#                     "INCOMPLETION",
#                     "INCOMP",
#                     "INC",
#                     "THROWNAWAY",
#                     "THROWAWAY",
#                     "BATTEDPASS",
#                     "HITASTHREW",
#                     "SPIKE",
#                 ]
#             )
#         )
#         .then(pl.lit("Incompletion"))
#         .otherwise(None)
#         .alias("outcome")
#     )

#     field_side = (
#         pl.when(direction.is_in(["L", "LEFT"]))
#         .then(pl.lit("Left"))
#         .when(direction.is_in(["M", "MID", "MIDDLE", "C", "CENTER", "CENTRE"]))
#         .then(pl.lit("Center"))
#         .when(direction.is_in(["R", "RIGHT"]))
#         .then(pl.lit("Right"))
#         .otherwise(None)
#         .alias("field_side")
#     )

#     return (
#         pbp_plays
#         .with_columns(
#             outcome,
#             field_side,
#             pl.col(PASS_DEPTH)
#             .cast(pl.Float64, strict=False)
#             .alias("air_yards"),
#             pl.col(PASS_WIDTH)
#             .cast(pl.Float64, strict=False)
#             .alias("pass_width"),
#         )
#         .filter(pl.col("outcome").is_not_null())
#     )


# def split_passes_by_location(pass_attempts):
#     """Return located attempts for plotting and attempts without a location."""
#     has_location = (
#         pl.col("air_yards").is_not_null()
#         & pl.col("field_side").is_not_null()
#     )

#     located = (
#         pass_attempts
#         .filter(has_location)
#         .with_columns(
#             pl.when(pl.col("air_yards") < 7)
#             .then(pl.lit("Short"))
#             .when(pl.col("air_yards") < 20)
#             .then(pl.lit("Medium"))
#             .otherwise(pl.lit("Long"))
#             .alias("depth_bucket"),
#             pl.when(pl.col("pass_width").is_not_null())
#             .then(pl.col("pass_width") - 26.5)
#             .when(pl.col("field_side") == "Left")
#             .then(pl.lit(-20.0))
#             .when(pl.col("field_side") == "Center")
#             .then(pl.lit(0.0))
#             .otherwise(pl.lit(20.0))
#             .alias("plot_x"),
#         )
#     )

#     unlocated = pass_attempts.filter(~has_location)
#     return located, unlocated


# def summarize_by_area(located_passes):
#     """Summarize pass outcomes by depth bucket and horizontal field side."""
#     summary = located_passes.group_by(["depth_bucket", "field_side"]).agg(
#         pl.len().alias("attempts"),
#         (pl.col("outcome") == "Completion").sum().alias("completions"),
#         (pl.col("outcome") == "Incompletion").sum().alias("incompletions"),
#         (pl.col("outcome") == "Interception").sum().alias("interceptions"),
#         pl.col("air_yards").mean().round(1).alias("average_air_yards"),
#     )

#     return (
#         summary
#         .with_columns(
#             (100 * pl.col("completions") / pl.col("attempts"))
#             .round(1)
#             .alias("completion_pct"),
#             (100 * pl.col("interceptions") / pl.col("attempts"))
#             .round(1)
#             .alias("interception_pct"),
#             pl.when(pl.col("depth_bucket") == "Short")
#             .then(pl.lit(0))
#             .when(pl.col("depth_bucket") == "Medium")
#             .then(pl.lit(1))
#             .otherwise(pl.lit(2))
#             .alias("_depth_order"),
#             pl.when(pl.col("field_side") == "Left")
#             .then(pl.lit(0))
#             .when(pl.col("field_side") == "Center")
#             .then(pl.lit(1))
#             .otherwise(pl.lit(2))
#             .alias("_side_order"),
#         )
#         .sort(["_depth_order", "_side_order"])
#         .drop(["_depth_order", "_side_order"])
#     )
"""Query and prepare quarterback passing data from the PFF CSV files."""

import polars as pl

# pff_offense identifies the player and the plays where that player was the passer.
OFFENSE_PLAYER_ID = "pff_PLAYERID"
OFFENSE_PLAYER_NAME = "pff_PLAYERNAME"
OFFENSE_TEAM = "pff_TEAM"
OFFENSE_PASSER_FLAG = "pff_PASSER"

# These play identifiers connect pff_offense to pff_pbp.
GAME_ID = "pff_GAMEID"
PLAY_ID = "pff_PLAYID"
GSIS_GAME_KEY = "pff_GSISGAMEKEY"
GSIS_PLAY_ID = "pff_GSISPLAYID"

# pff_pbp supplies the Level 0 pass result and target location.
PBP_OFFENSE_TEAM = "pff_OFFTEAM"
PASS_RESULT = "pff_PASSRESULT"
PASS_DEPTH = "pff_PASSDEPTH"
PASS_DIRECTION = "pff_PASSDIRECTION"
PASS_WIDTH = "pff_PASSWIDTH"
NO_PLAY = "pff_NOPLAY"

OFFENSE_COLUMNS = [
    OFFENSE_PLAYER_ID,
    OFFENSE_PLAYER_NAME,
    OFFENSE_TEAM,
    OFFENSE_PASSER_FLAG,
    GAME_ID,
    PLAY_ID,
    GSIS_GAME_KEY,
    GSIS_PLAY_ID,
]

# There is intentionally no pff_PASSER column here. The quarterback is
# identified in pff_offense before pff_pbp is queried.
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
    PBP_OFFENSE_TEAM,
    PASS_RESULT,
    PASS_DEPTH,
    PASS_DIRECTION,
    PASS_WIDTH,
    NO_PLAY,
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


def _clean_text(column):
    """Return a CSV column as trimmed uppercase text."""
    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )


def _clean_id(column):
    """Normalize IDs so values such as 10635 and 10635.0 compare equally."""
    return _clean_text(column).str.replace(r"\.0$", "")


def _is_yes(column):
    """Recognize common true values used by PFF flag columns."""
    return _clean_text(column).is_in(["Y", "YES", "TRUE", "T", "1", "1.0"])


def load_player_offense_rows(csv_path, player_id):
    """Load every pff_offense row for one PFF player ID."""
    player_id = str(player_id).strip().removesuffix(".0")

    return (
        pl.read_csv(
            csv_path,
            columns=OFFENSE_COLUMNS,
            infer_schema=False,
        )
        .with_columns(
            _clean_id(OFFENSE_PLAYER_ID).alias(OFFENSE_PLAYER_ID),
            _clean_id(GAME_ID).alias(GAME_ID),
            _clean_id(PLAY_ID).alias(PLAY_ID),
            _clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            _clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
        )
        .filter(pl.col(OFFENSE_PLAYER_ID) == player_id)
        .sort([GAME_ID, PLAY_ID])
    )


def get_player_game_ids(player_offense_rows):
    """Return the player's distinct pff_GAMEID values as a Python set."""
    return {
        game_id
        for game_id in player_offense_rows.get_column(GAME_ID).to_list()
        if game_id
    }


def get_qb_passer_rows(player_offense_rows):
    """Return the player's raw pff_offense rows where pff_PASSER is true."""
    return player_offense_rows.filter(_is_yes(OFFENSE_PASSER_FLAG))


def get_qb_pass_play_keys(player_offense_rows):
    """Return unique game/play keys where the player is marked as the passer."""
    return (
        get_qb_passer_rows(player_offense_rows)
        .select(GAME_ID, PLAY_ID, GSIS_GAME_KEY, GSIS_PLAY_ID)
        .filter(
            (pl.col(GAME_ID) != "")
            & (pl.col(PLAY_ID) != "")
        )
        .unique()
        .sort([GAME_ID, PLAY_ID])
    )


def load_pbp_for_plays(csv_path, game_ids, pass_play_keys):
    """Load pff_pbp rows matching the quarterback's exact game/play keys."""
    pbp = (
        pl.read_csv(
            csv_path,
            columns=PBP_COLUMNS,
            infer_schema=False,
        )
        .with_columns(
            _clean_id(GAME_ID).alias(GAME_ID),
            _clean_id(PLAY_ID).alias(PLAY_ID),
            _clean_id(GSIS_GAME_KEY).alias(GSIS_GAME_KEY),
            _clean_id(GSIS_PLAY_ID).alias(GSIS_PLAY_ID),
        )
        .filter(
            (pl.col(GAME_ID) != "")
            & (pl.col(PLAY_ID) != "")
        )
    )

    if game_ids:
        pbp = pbp.filter(pl.col(GAME_ID).is_in(sorted(game_ids)))

    keys = pass_play_keys.select(GAME_ID, PLAY_ID).unique()

    # Keep no-play rows here. Diagnostics should distinguish a successful join
    # from a play that is later excluded from the passing analysis.
    return (
        pbp
        .join(
            keys,
            on=[GAME_ID, PLAY_ID],
            how="inner",
        )
        .sort([GAME_ID, PLAY_ID])
    )


def load_tracking_for_plays(csv_path, pbp_plays):
    """Load tracking rows for PBP plays using all available play identifiers."""
    tracking = (
        pl.read_csv(
            csv_path,
            columns=TRACKING_COLUMNS,
            infer_schema=False,
        )
        .with_columns(
            _clean_id("pff_play_id").alias("pff_play_id"),
            _clean_id("game_key").alias("game_key"),
            _clean_id("gsis_play_id").alias("gsis_play_id"),
        )
        .filter(pl.col("pff_play_id") != "")
    )

    tracking_keys = (
        pbp_plays
        .select(
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

    return tracking.join(
        tracking_keys,
        on=["pff_play_id", "game_key", "gsis_play_id"],
        how="inner",
    )


def label_pass_outcomes(pbp_plays):
    """Add normalized result, no-play, outcome, and location columns."""
    result = _clean_text(PASS_RESULT).str.replace_all(r"[^A-Z0-9]", "")
    direction = _clean_text(PASS_DIRECTION).str.replace_all(r"[^A-Z]", "")

    outcome = (
        pl.when(result.is_in(["INTERCEPTION", "INTERCEPTED", "INT"]))
        .then(pl.lit("Interception"))
        .when(result.is_in(["COMPLETE", "COMPLETION", "COMP", "C"]))
        .then(pl.lit("Completion"))
        .when(
            result.is_in(
                [
                    "INCOMPLETE",
                    "INCOMPLETION",
                    "INCOMP",
                    "INC",
                    "THROWNAWAY",
                    "THROWAWAY",
                    "BATTEDPASS",
                    "HITASTHREW",
                    "SPIKE",
                ]
            )
        )
        .then(pl.lit("Incompletion"))
        .otherwise(None)
        .alias("outcome")
    )

    field_side = (
        pl.when(direction.is_in(["L", "LEFT"]))
        .then(pl.lit("Left"))
        .when(direction.is_in(["M", "MID", "MIDDLE", "C", "CENTER", "CENTRE"]))
        .then(pl.lit("Center"))
        .when(direction.is_in(["R", "RIGHT"]))
        .then(pl.lit("Right"))
        .otherwise(None)
        .alias("field_side")
    )

    return pbp_plays.with_columns(
        result.alias("normalized_pass_result"),
        _is_yes(NO_PLAY).alias("is_no_play"),
        outcome,
        field_side,
        pl.col(PASS_DEPTH)
        .cast(pl.Float64, strict=False)
        .alias("air_yards"),
        pl.col(PASS_WIDTH)
        .cast(pl.Float64, strict=False)
        .alias("pass_width"),
    )


def classify_pass_attempts(pbp_plays):
    """Keep completions, incompletions, and interceptions on valid plays."""
    return (
        label_pass_outcomes(pbp_plays)
        .filter(
            (~pl.col("is_no_play"))
            & pl.col("outcome").is_not_null()
        )
    )


def split_passes_by_location(pass_attempts):
    """Return located attempts and attempts missing depth/direction."""
    has_location = (
        pl.col("air_yards").is_not_null()
        & pl.col("field_side").is_not_null()
    )

    located = (
        pass_attempts
        .filter(has_location)
        .with_columns(
            pl.when(pl.col("air_yards") < 7)
            .then(pl.lit("Short"))
            .when(pl.col("air_yards") < 20)
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("Long"))
            .alias("depth_bucket"),
            pl.when(pl.col("pass_width").is_not_null())
            .then(pl.col("pass_width") - 26.5)
            .when(pl.col("field_side") == "Left")
            .then(pl.lit(-20.0))
            .when(pl.col("field_side") == "Center")
            .then(pl.lit(0.0))
            .otherwise(pl.lit(20.0))
            .alias("plot_x"),
        )
    )

    unlocated = (
        pass_attempts
        .filter(~has_location)
        .with_columns(
            pl.when(
                pl.col("air_yards").is_null()
                & pl.col("field_side").is_null()
            )
            .then(pl.lit("Missing depth and direction"))
            .when(pl.col("air_yards").is_null())
            .then(pl.lit("Missing depth"))
            .otherwise(pl.lit("Missing direction"))
            .alias("location_issue")
        )
    )

    return located, unlocated


def summarize_by_area(located_passes):
    """Summarize pass outcomes by depth bucket and horizontal field side."""
    summary = located_passes.group_by(["depth_bucket", "field_side"]).agg(
        pl.len().alias("attempts"),
        (pl.col("outcome") == "Completion").sum().alias("completions"),
        (pl.col("outcome") == "Incompletion").sum().alias("incompletions"),
        (pl.col("outcome") == "Interception").sum().alias("interceptions"),
        pl.col("air_yards").mean().round(1).alias("average_air_yards"),
    )

    return (
        summary
        .with_columns(
            (100 * pl.col("completions") / pl.col("attempts"))
            .round(1)
            .alias("completion_pct"),
            (100 * pl.col("interceptions") / pl.col("attempts"))
            .round(1)
            .alias("interception_pct"),
            pl.when(pl.col("depth_bucket") == "Short")
            .then(pl.lit(0))
            .when(pl.col("depth_bucket") == "Medium")
            .then(pl.lit(1))
            .otherwise(pl.lit(2))
            .alias("_depth_order"),
            pl.when(pl.col("field_side") == "Left")
            .then(pl.lit(0))
            .when(pl.col("field_side") == "Center")
            .then(pl.lit(1))
            .otherwise(pl.lit(2))
            .alias("_side_order"),
        )
        .sort(["_depth_order", "_side_order"])
        .drop(["_depth_order", "_side_order"])
    )
