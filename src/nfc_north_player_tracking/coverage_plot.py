"""Matplotlib plotting for static, projected, and kinematic control maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from nfc_north_player_tracking.config import CoverageParameters
from nfc_north_player_tracking.dominant_regions import (
    ControlSurface,
    TargetControlResult,
)

if TYPE_CHECKING:
    from nfc_north_player_tracking.coverage_analysis import CoverageAnalysisResult

FIELD_COLOR = "#eef5e8"
FIELD_LINE_COLOR = "#ffffff"
LOS_COLOR = "#d97706"
OFFENSE_FILL = "#76b7e5"
DEFENSE_FILL = "#ef8a82"
OFFENSE_MARKER = "#0b559f"
DEFENSE_MARKER = "#a50f15"
TARGET_COLOR = "#7a3db8"
BALL_COLOR = "#4b2e1e"
QB_COLOR = "#f3c74f"
PROJECTED_SITE_COLOR = "#f6d55c"
TEAM_BOUNDARY_COLOR = "#111111"


def _short_name(name: str) -> str:
    pieces = [piece for piece in name.replace(".", "").split() if piece]
    return pieces[-1] if pieces else name


def _bounds_tuple(bounds: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bounds) != 4:
        raise ValueError("Plot bounds must be (x_min, x_max, y_min, y_max).")
    return tuple(float(value) for value in bounds)  # type: ignore[return-value]


def draw_field(axis: plt.Axes, bounds: Sequence[float]) -> None:
    """Draw a football field in offense-relative coordinates."""
    x_min, x_max, y_min, y_max = _bounds_tuple(bounds)
    axis.set_facecolor(FIELD_COLOR)
    axis.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color="black",
        linewidth=1.25,
        zorder=30,
    )
    first_yard = int(np.ceil(y_min / 5.0) * 5)
    last_yard = int(np.floor(y_max / 5.0) * 5)
    for yard in range(first_yard, last_yard + 1, 5):
        axis.axhline(
            yard,
            color=FIELD_LINE_COLOR,
            linewidth=0.8,
            alpha=0.85,
            zorder=0,
        )
    axis.axhline(0.0, color=LOS_COLOR, linewidth=2.5, zorder=31)
    axis.text(
        x_max - 0.7,
        0.55,
        "LOS",
        color=LOS_COLOR,
        fontsize=8.5,
        fontweight="bold",
        ha="right",
        va="bottom",
        zorder=32,
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Offense left (-) to right (+), yards")
    axis.set_ylabel("Yards relative to LOS; downfield positive")
    axis.grid(False)


def _release_rows(release_state: Any) -> list[dict[str, Any]]:
    if hasattr(release_state, "to_dicts"):
        return list(release_state.to_dicts())
    return [dict(row) for row in release_state]


def _draw_exact_polygons(
    axis: plt.Axes,
    surface: ControlSurface,
    release_rows: Sequence[Mapping[str, Any]],
) -> None:
    if surface.polygons is None:
        raise ValueError(f"{surface.model_label} does not contain exact polygons.")
    for polygon, row in zip(surface.polygons, release_rows, strict=True):
        if len(polygon) < 3:
            continue
        side = str(row["side"]).upper()
        fill = OFFENSE_FILL if side == "OFFENSE" else DEFENSE_FILL
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=fill,
                edgecolor="white",
                linewidth=0.75,
                alpha=0.55,
                zorder=3,
            )
        )


def _draw_team_control_surface(
    axis: plt.Axes,
    surface: ControlSurface,
    release_rows: Sequence[Mapping[str, Any]],
) -> None:
    sides = np.asarray([str(row["side"]).upper() for row in release_rows])
    offense_grid = (sides[surface.owner_index] == "OFFENSE").astype(float)
    x_grid, y_grid = np.meshgrid(surface.x_coordinates, surface.y_coordinates)
    axis.contourf(
        x_grid,
        y_grid,
        offense_grid,
        levels=[-0.5, 0.5, 1.5],
        colors=[DEFENSE_FILL, OFFENSE_FILL],
        alpha=0.55,
        antialiased=False,
        zorder=2,
    )


def _categorical_boundary_segments(
    surface: ControlSurface,
    release_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    owner = surface.owner_index
    sides = np.asarray([str(row["side"]).upper() for row in release_rows])
    owner_side = sides[owner]
    x = surface.x_coordinates
    y = surface.y_coordinates
    if len(x) < 2 or len(y) < 2:
        return [], []
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    individual: list[list[tuple[float, float]]] = []
    team: list[list[tuple[float, float]]] = []

    rows, columns = owner.shape
    for row in range(rows):
        y_low = float(y[row] - dy / 2.0)
        y_high = float(y[row] + dy / 2.0)
        for column in range(1, columns):
            if owner[row, column - 1] == owner[row, column]:
                continue
            x_edge = float((x[column - 1] + x[column]) / 2.0)
            segment = [(x_edge, y_low), (x_edge, y_high)]
            individual.append(segment)
            if owner_side[row, column - 1] != owner_side[row, column]:
                team.append(segment)

    for row in range(1, rows):
        y_edge = float((y[row - 1] + y[row]) / 2.0)
        for column in range(columns):
            if owner[row - 1, column] == owner[row, column]:
                continue
            x_low = float(x[column] - dx / 2.0)
            x_high = float(x[column] + dx / 2.0)
            segment = [(x_low, y_edge), (x_high, y_edge)]
            individual.append(segment)
            if owner_side[row - 1, column] != owner_side[row, column]:
                team.append(segment)
    return individual, team


def _draw_grid_boundaries(
    axis: plt.Axes,
    surface: ControlSurface,
    release_rows: Sequence[Mapping[str, Any]],
) -> None:
    individual, team = _categorical_boundary_segments(surface, release_rows)
    if individual:
        axis.add_collection(
            LineCollection(
                individual,
                colors="white",
                linewidths=0.25,
                alpha=0.55,
                zorder=5,
            )
        )


def _draw_team_boundary(axis: plt.Axes, surface: ControlSurface) -> None:
    x_grid, y_grid = np.meshgrid(surface.x_coordinates, surface.y_coordinates)
    margin = surface.control_margin
    if np.nanmin(margin) <= 0.0 <= np.nanmax(margin):
        axis.contour(
            x_grid,
            y_grid,
            margin,
            levels=[0.0],
            colors=TEAM_BOUNDARY_COLOR,
            linewidths=1.8,
            zorder=8,
        )


def _draw_players(
    axis: plt.Axes,
    surface: ControlSurface,
    release_rows: Sequence[Mapping[str, Any]],
    parameters: CoverageParameters,
    *,
    show_velocity: bool,
    show_projected_sites: bool,
) -> None:
    for index, row in enumerate(release_rows):
        side = str(row["side"]).upper()
        color = OFFENSE_MARKER if side == "OFFENSE" else DEFENSE_MARKER
        intended = bool(row.get("targeted_receiver", False))
        marker = "*" if intended else ("o" if side == "OFFENSE" else "s")
        size = 155 if intended else 65
        x = float(row["release_x"])
        y = float(row["release_y"])
        axis.scatter(
            [x],
            [y],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=0.75,
            zorder=30,
        )

        if show_velocity:
            projected_x = x + (
                parameters.velocity_projection_horizon_seconds
                * float(row["velocity_x_yards_per_second"])
            )
            projected_y = y + (
                parameters.velocity_projection_horizon_seconds
                * float(row["velocity_y_yards_per_second"])
            )
            axis.annotate(
                "",
                xy=(projected_x, projected_y),
                xytext=(x, y),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.1,
                    "color": color,
                    "alpha": 0.92,
                },
                zorder=28,
            )
            if show_projected_sites:
                projected_site = surface.site_positions[index]
                axis.scatter(
                    [float(projected_site[0])],
                    [float(projected_site[1])],
                    s=42,
                    marker="x",
                    color=PROJECTED_SITE_COLOR,
                    linewidth=1.8,
                    zorder=29,
                )

        horizontal_offset = 0.35 if x <= 0 else -0.35
        axis.text(
            x + horizontal_offset,
            y + (0.45 if index % 2 == 0 else -0.65),
            _short_name(str(row["pff_PLAYERNAME"])),
            ha="left" if horizontal_offset > 0 else "right",
            va="center",
            fontsize=7.2,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
            },
            zorder=35,
        )


def _draw_context(
    axis: plt.Axes,
    target: tuple[float, float],
    context: Mapping[str, object],
) -> None:
    ball_x = context.get("ball_x")
    ball_y = context.get("ball_y")
    if ball_x is not None and ball_y is not None:
        axis.scatter(
            [float(ball_x)],
            [float(ball_y)],
            s=35,
            marker="o",
            facecolor=BALL_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=38,
        )

    qb_x = context.get("quarterback_x")
    qb_y = context.get("quarterback_y")
    if qb_x is not None and qb_y is not None:
        axis.scatter(
            [float(qb_x)],
            [float(qb_y)],
            s=74,
            marker="D",
            facecolor=QB_COLOR,
            edgecolor="black",
            linewidth=0.8,
            zorder=37,
        )
        axis.plot(
            [float(qb_x), float(target[0])],
            [float(qb_y), float(target[1])],
            linestyle=":",
            linewidth=0.9,
            color=TARGET_COLOR,
            alpha=0.55,
            zorder=15,
        )

    axis.scatter(
        [float(target[0])],
        [float(target[1])],
        s=170,
        marker="X",
        facecolor=TARGET_COLOR,
        edgecolor="white",
        linewidth=1.1,
        zorder=40,
    )


def _model_title(
    surface: ControlSurface,
    target_result: TargetControlResult,
    parameters: CoverageParameters,
) -> str:
    controlled = "YES" if target_result.target_in_offense_control else "NO"
    owner = _short_name(target_result.owner_name)
    boundary = " | near boundary" if target_result.target_near_boundary else ""
    if surface.model_key == "static":
        heading = "Static Voronoi"
        detail = f"margin {target_result.control_margin:+.2f} yd"
    elif surface.model_key == "velocity":
        heading = (
            "Velocity-projected Voronoi "
            f"({parameters.velocity_projection_horizon_seconds:.2f} s)"
        )
        detail = f"margin {target_result.control_margin:+.2f} yd"
    else:
        heading = "Advanced kinematic arrival-time control"
        detail = f"margin {target_result.control_margin:+.2f} s"
    return (
        f"{heading}\n"
        f"OFFENSE-CONTROLLED TARGET: {controlled}\n"
        f"Owner: {owner} | {detail}{boundary}"
    )


def plot_control_panel(
    axis: plt.Axes,
    *,
    surface: ControlSurface,
    target_result: TargetControlResult,
    release_state: Any,
    target: tuple[float, float],
    context: Mapping[str, object],
    bounds: Sequence[float],
    parameters: CoverageParameters,
) -> None:
    """Draw one static, projected, or advanced coverage panel."""
    rows = _release_rows(release_state)
    draw_field(axis, bounds)

    if surface.polygons is not None:
        _draw_exact_polygons(axis, surface, rows)
        _draw_team_boundary(axis, surface)
    else:
        _draw_team_control_surface(axis, surface, rows)
        _draw_grid_boundaries(axis, surface, rows)
        _draw_team_boundary(axis, surface)

    _draw_players(
        axis,
        surface,
        rows,
        parameters,
        show_velocity=surface.model_key in {"velocity", "kinematic"},
        show_projected_sites=surface.model_key == "velocity",
    )
    _draw_context(axis, target, context)
    axis.set_title(_model_title(surface, target_result, parameters), fontsize=10.5)


def make_three_model_figure(result: "CoverageAnalysisResult") -> plt.Figure:
    """Create the final side-by-side three-model coverage figure."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 9), sharex=True, sharey=True)
    for axis, surface, target_result in zip(
        axes,
        result.surfaces,
        result.target_results,
        strict=True,
    ):
        plot_control_panel(
            axis,
            surface=surface,
            target_result=target_result,
            release_state=result.release_state,
            target=result.target,
            context=result.context,
            bounds=result.plot_bounds,
            parameters=result.parameters,
        )

    pbp = result.pbp_play.row(0, named=True)
    figure.suptitle(
        "Coverage control at pass release — "
        f"{pbp.get('pff_OFFTEAM')} vs {pbp.get('pff_DEFTEAM')} | "
        f"Game {result.game_id}, Play {result.play_id} | "
        f"{pbp.get('pff_PASSRESULT')}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=OFFENSE_MARKER,
            markeredgecolor="black", markersize=7, label="Route runner"
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor=DEFENSE_MARKER,
            markeredgecolor="black", markersize=7, label="Coverage defender"
        ),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor=OFFENSE_MARKER,
            markeredgecolor="black", markersize=11, label="Intended receiver"
        ),
        Line2D(
            [0], [0], marker="D", color="none", markerfacecolor=QB_COLOR,
            markeredgecolor="black", markersize=7, label="Quarterback"
        ),
        Line2D(
            [0], [0], marker="X", color="none", markerfacecolor=TARGET_COLOR,
            markeredgecolor="white", markersize=10, label="PFF target"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.88,
        bottom=0.11,
        wspace=0.08,
    )
    return figure


def save_single_model_figures(
    result: "CoverageAnalysisResult",
    output_directory: Path,
    *,
    dpi: int = 200,
) -> list[Path]:
    """Save one standalone figure for each model and return the paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for surface, target_result in zip(
        result.surfaces,
        result.target_results,
        strict=True,
    ):
        figure, axis = plt.subplots(figsize=(8, 10))
        plot_control_panel(
            axis,
            surface=surface,
            target_result=target_result,
            release_state=result.release_state,
            target=result.target,
            context=result.context,
            bounds=result.plot_bounds,
            parameters=result.parameters,
        )
        figure.tight_layout()
        path = output_directory / (
            f"{result.game_id}_{result.play_id}_{surface.model_key}_control.png"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
