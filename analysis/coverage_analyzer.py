"""Run three release-frame coverage-control analyses for selected pass plays.

Models:

1. Static Euclidean Voronoi control.
2. Velocity-projected Voronoi control.
3. Kinematic Arrival-Time control.

Examples from the repository root:

    # Representative default play.
    python analysis/coverage_analyzer.py

    # One explicitly selected play.
    python analysis/coverage_analyzer.py --play 28430:6408068

    # Multiple representative plays.
    python analysis/coverage_analyzer.py \
        --play 28430:6408068 \
        --play 28430:6408079
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = REPO_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from nfc_north_player_tracking.config import (  # noqa: E402
    DEFAULT_PARAMETERS,
    DEFAULT_PLAY,
    OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
    OWNERSHIP_MODE_INTENDED_RECEIVER,
)
from nfc_north_player_tracking.coverage_analysis import (  # noqa: E402
    CoverageAnalysisResult,
    analyze_coverage_play,
)
from nfc_north_player_tracking.coverage_diagnostics import (  # noqa: E402
    raise_for_failed_diagnostics,
)
from nfc_north_player_tracking.coverage_plot import (  # noqa: E402
    make_three_model_figure,
    save_single_model_figures,
)
from nfc_north_player_tracking.dominant_regions import (  # noqa: E402
    simulate_arrival_times,
)


def normalize_id(value: object) -> str:
    """Normalize a command-line identifier as text."""
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def parse_play(value: str) -> tuple[str, str]:
    """Parse ``GAME_ID:PLAY_ID`` from the command line."""
    pieces = value.split(":", maxsplit=1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Play must use GAME_ID:PLAY_ID format.")
    game_id = normalize_id(pieces[0])
    play_id = normalize_id(pieces[1])
    if not game_id or not play_id:
        raise argparse.ArgumentTypeError("Both GAME_ID and PLAY_ID are required.")
    return game_id, play_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--play",
        action="append",
        type=parse_play,
        help=(
            "PFF GAME_ID:PLAY_ID. Repeat for multiple plays. Uses the "
            "representative default play when omitted."
        ),
    )
    parser.add_argument(
        "--pbp-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/pff_pbp.csv",
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
        "--tracking-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/tracking_sample.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "outputs/coverage",
    )
    parser.add_argument(
        "--ownership-mode",
        choices=[
            OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
            OWNERSHIP_MODE_INTENDED_RECEIVER,
        ],
        default=OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
        help=(
            "Primary offense comparator at the target. The default treats any "
            "route runner as offensive control."
        ),
    )
    parser.add_argument(
        "--projection-horizon",
        type=float,
        default=DEFAULT_PARAMETERS.velocity_projection_horizon_seconds,
        help="Projected-site Voronoi horizon in seconds.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=DEFAULT_PARAMETERS.max_player_speed_yards_per_second,
        help="Shared player speed ceiling in yards/second.",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=(
            DEFAULT_PARAMETERS.max_player_acceleration_yards_per_second_squared
        ),
        help="Shared player acceleration ceiling in yards/second^2.",
    )
    parser.add_argument(
        "--distance-grid-spacing",
        type=float,
        default=DEFAULT_PARAMETERS.distance_grid_resolution_yards,
        help="Static/projected diagnostic surface spacing in yards.",
    )
    parser.add_argument(
        "--kinematic-grid-spacing",
        type=float,
        default=DEFAULT_PARAMETERS.kinematic_grid_resolution_yards,
        help="Advanced-model field-grid spacing in yards.",
    )
    parser.add_argument(
        "--kinematic-time-step",
        type=float,
        default=DEFAULT_PARAMETERS.kinematic_grid_time_step_seconds,
        help="Advanced-model field-surface simulation step in seconds.",
    )
    parser.add_argument(
        "--allow-ineligible-pass",
        action="store_true",
        help="Run a play even when it fails the strict regular-pass filter.",
    )
    parser.add_argument(
        "--keep-failed-diagnostics",
        action="store_true",
        help="Write outputs instead of raising on FAIL diagnostics.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def target_player_values(result: CoverageAnalysisResult) -> pl.DataFrame:
    """Return every generator's value at the exact PFF target."""
    rows = result.release_state.to_dicts()
    target = np.asarray(result.target, dtype=float)
    positions = np.column_stack(
        [
            result.release_state.get_column("release_x").to_numpy(),
            result.release_state.get_column("release_y").to_numpy(),
        ]
    ).astype(float)
    velocities = np.column_stack(
        [
            result.release_state
            .get_column("velocity_x_yards_per_second")
            .to_numpy(),
            result.release_state
            .get_column("velocity_y_yards_per_second")
            .to_numpy(),
        ]
    ).astype(float)
    projected = positions + (
        result.parameters.velocity_projection_horizon_seconds * velocities
    )

    static_distance = np.linalg.norm(positions - target[None, :], axis=1)
    projected_distance = np.linalg.norm(projected - target[None, :], axis=1)
    target_array = target.reshape(1, 2)
    kinematic_times = np.array(
        [
            simulate_arrival_times(
                start_position=position,
                start_velocity=velocity,
                targets=target_array,
                max_speed=result.parameters.max_player_speed_yards_per_second,
                max_acceleration=(
                    result.parameters
                    .max_player_acceleration_yards_per_second_squared
                ),
                time_step=result.parameters.kinematic_target_time_step_seconds,
                max_time=result.parameters.kinematic_max_time_seconds,
                arrival_radius=(
                    result.parameters.kinematic_target_arrival_radius_yards
                ),
            )[0]
            for position, velocity in zip(positions, velocities, strict=True)
        ],
        dtype=float,
    )

    output: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        output.append(
            {
                "pff_GAMEID": result.game_id,
                "pff_PLAYID": result.play_id,
                "player_name": row["pff_PLAYERNAME"],
                "side": row["side"],
                "analysis_role": row["analysis_role"],
                "is_intended_receiver": row["targeted_receiver"],
                "static_distance_yards": float(static_distance[index]),
                "velocity_distance_yards": float(projected_distance[index]),
                "kinematic_arrival_time_seconds": float(kinematic_times[index]),
            }
        )
    return pl.DataFrame(output).sort(
        ["kinematic_arrival_time_seconds", "side", "player_name"]
    )


def save_result(
    result: CoverageAnalysisResult,
    output_directory: Path,
    *,
    dpi: int,
) -> list[Path]:
    """Write all play-level figures, inputs, diagnostics, and result tables."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result.game_id}_{result.play_id}"

    combined = make_three_model_figure(result)
    combined_path = output_directory / f"{stem}_coverage_control.png"
    combined.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(combined)

    single_paths = save_single_model_figures(result, output_directory, dpi=dpi)

    summary_path = output_directory / f"{stem}_coverage_summary.csv"
    states_path = output_directory / f"{stem}_generator_states.csv"
    resolution_path = output_directory / f"{stem}_player_resolution.csv"
    diagnostics_path = output_directory / f"{stem}_diagnostics.csv"
    target_values_path = output_directory / f"{stem}_target_player_values.csv"

    result.result_frame().write_csv(summary_path)
    result.release_state.write_csv(states_path)
    result.generator_audit.write_csv(resolution_path)
    result.diagnostics.write_csv(diagnostics_path)
    target_player_values(result).write_csv(target_values_path)

    return [
        combined_path,
        *single_paths,
        summary_path,
        states_path,
        resolution_path,
        diagnostics_path,
        target_values_path,
    ]


def main() -> int:
    args = parse_args()
    plays = args.play or [parse_play(DEFAULT_PLAY)]
    parameters = replace(
        DEFAULT_PARAMETERS,
        velocity_projection_horizon_seconds=args.projection_horizon,
        max_player_speed_yards_per_second=args.max_speed,
        max_player_acceleration_yards_per_second_squared=args.max_acceleration,
        distance_grid_resolution_yards=args.distance_grid_spacing,
        kinematic_grid_resolution_yards=args.kinematic_grid_spacing,
        kinematic_grid_time_step_seconds=args.kinematic_time_step,
    )
    parameters.validate()

    summaries: list[pl.DataFrame] = []
    for game_id, play_id in plays:
        result = analyze_coverage_play(
            game_id=game_id,
            play_id=play_id,
            pbp_csv=args.pbp_csv,
            offense_csv=args.offense_csv,
            defense_csv=args.defense_csv,
            tracking_csv=args.tracking_csv,
            parameters=parameters,
            ownership_mode=args.ownership_mode,
            allow_ineligible=args.allow_ineligible_pass,
        )
        if not args.keep_failed_diagnostics:
            raise_for_failed_diagnostics(result.diagnostics)

        paths = save_result(result, args.output_directory, dpi=args.dpi)
        summary = result.result_frame()
        summaries.append(summary)

        print(f"\nCoverage analysis: {game_id}:{play_id}")
        print(summary)
        print("\nDiagnostics")
        print(result.diagnostics)
        for path in paths:
            print(f"Wrote: {path}")

    if summaries:
        combined = pl.concat(summaries, how="diagonal_relaxed")
        combined_path = args.output_directory / "coverage_summary.csv"
        combined.write_csv(combined_path)
        print(f"\nWrote combined summary: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
