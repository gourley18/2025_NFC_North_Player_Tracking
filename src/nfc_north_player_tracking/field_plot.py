"""Create football-field plots for the Level 0 quarterback analysis."""

import matplotlib.pyplot as plt

FIELD_LEFT = -26.65
FIELD_RIGHT = 26.65
FIELD_MIN_DEPTH = -10
FIELD_MAX_DEPTH = 60

# In the supplied PFF sample, pass-width values 0-13 are Left, 14-39 are
# Center, and 40-52 are Right. Centering those coordinates gives +/- 13.
LEFT_CENTER_DIVIDER = -13
CENTER_RIGHT_DIVIDER = 13


def draw_field(axis):
    """Draw a football field measured relative to the line of scrimmage."""
    axis.set_facecolor("#2f6f3e")

    axis.plot(
        [FIELD_LEFT, FIELD_RIGHT, FIELD_RIGHT, FIELD_LEFT, FIELD_LEFT],
        [
            FIELD_MIN_DEPTH,
            FIELD_MIN_DEPTH,
            FIELD_MAX_DEPTH,
            FIELD_MAX_DEPTH,
            FIELD_MIN_DEPTH,
        ],
        color="white",
        linewidth=2,
    )

    # Horizontal field lines every 5 yards.
    for yard in range(FIELD_MIN_DEPTH, FIELD_MAX_DEPTH + 1, 5):
        axis.axhline(yard, color="white", linewidth=0.8, alpha=0.55)

    # Hash marks every yard.
    for yard in range(FIELD_MIN_DEPTH, FIELD_MAX_DEPTH + 1):
        axis.plot([-12.2, -11.4], [yard, yard], color="white", linewidth=0.5)
        axis.plot([11.4, 12.2], [yard, yard], color="white", linewidth=0.5)

    # The field is relative to the snap, so zero is always the LOS.
    axis.axhline(0, color="#ffa200", linewidth=3)
    axis.text(FIELD_RIGHT - 1, 0.8, "LOS", color="#ffa200", ha="right")

    # Boundaries for the nine analysis areas.
    axis.axvline(LEFT_CENTER_DIVIDER, color="white", linestyle="--", alpha=0.7)
    axis.axvline(CENTER_RIGHT_DIVIDER, color="white", linestyle="--", alpha=0.7)
    axis.axhline(7, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    axis.axhline(20, color="white", linestyle="--", linewidth=1.5, alpha=0.8)

    axis.set_xlim(FIELD_LEFT, FIELD_RIGHT)
    axis.set_ylim(FIELD_MIN_DEPTH, FIELD_MAX_DEPTH)
    axis.set_aspect("equal")
    axis.set_xticks([-20, 0, 20])
    axis.set_xticklabels(["LEFT", "CENTER", "RIGHT"])
    axis.set_ylabel("Pass depth relative to line of scrimmage (yards)")

    for spine in axis.spines.values():
        spine.set_visible(False)

    return axis


def make_pass_plot(passes, title):
    """Plot each located completion, incompletion, and interception."""
    figure, axis = plt.subplots(figsize=(9, 12))
    draw_field(axis)

    marker_styles = {
        "Completion": {
            "marker": "o",
            "facecolors": "none",
            "edgecolors": "white",
            "linewidths": 2,
            "label": "Completion (O)",
        },
        "Incompletion": {
            "marker": "x",
            "color": "#ef5350",
            "linewidths": 2,
            "label": "Incompletion (X)",
        },
        "Interception": {
            "marker": "^",
            "color": "#ffca28",
            "edgecolors": "black",
            "linewidths": 0.8,
            "label": "Interception (triangle)",
        },
    }

    for outcome, style in marker_styles.items():
        subset = passes.filter(passes["outcome"] == outcome)

        if subset.is_empty():
            continue

        axis.scatter(
            subset["plot_x"].to_list(),
            subset["air_yards"].to_list(),
            s=85,
            alpha=0.9,
            zorder=5,
            **style,
        )

    axis.set_title(title, fontsize=15, fontweight="bold", pad=14)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    figure.tight_layout()
    return figure


def make_area_plot(summary, title):
    """Shade the nine field areas by completion percentage."""
    figure, axis = plt.subplots(figsize=(9, 12))
    draw_field(axis)

    rows = {
        (row["depth_bucket"], row["field_side"]): row
        for row in summary.iter_rows(named=True)
    }

    x_ranges = {
        "Left": (FIELD_LEFT, LEFT_CENTER_DIVIDER),
        "Center": (LEFT_CENTER_DIVIDER, CENTER_RIGHT_DIVIDER),
        "Right": (CENTER_RIGHT_DIVIDER, FIELD_RIGHT),
    }

    # Short begins at the back of the plotted field, so negative air-yard
    # throws caught behind the LOS are included in the Short bucket.
    y_ranges = {
        "Short": (FIELD_MIN_DEPTH, 7),
        "Medium": (7, 20),
        "Long": (20, FIELD_MAX_DEPTH),
    }

    color_map = plt.get_cmap("RdYlGn")

    for depth_bucket, (y_min, y_max) in y_ranges.items():
        for field_side, (x_min, x_max) in x_ranges.items():
            row = rows.get((depth_bucket, field_side))

            if row is None:
                completion_pct = None
                attempts = 0
                fill_color = "gray"
            else:
                completion_pct = row["completion_pct"]
                attempts = row["attempts"]
                fill_color = color_map(completion_pct / 100)

            axis.fill(
                [x_min, x_max, x_max, x_min],
                [y_min, y_min, y_max, y_max],
                color=fill_color,
                alpha=0.55,
                zorder=1,
            )

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            if completion_pct is None:
                label = "No attempts"
            else:
                label = f"{completion_pct:.1f}%\n{attempts} att"

            axis.text(
                x_center,
                y_center,
                label,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=5,
            )

    axis.set_title(title, fontsize=15, fontweight="bold", pad=14)
    figure.tight_layout()
    return figure
