# NFC North Quarterback Tracking and Coverage Control

A Python research project for evaluating NFC North quarterback play with PFF play-by-play, player-play charting, and player-tracking data.

The repository currently contains two connected analysis stages:

1. **Passing-location analysis** — establishes a trustworthy quarterback play population and summarizes pass outcomes across field areas.
2. **Release-frame coverage control** — uses player tracking at `pass_forward` to compare static, velocity-aware, and arrival-time-based control of the intended target location.

The project is built with **Python**, **Polars**, **NumPy**, and **Matplotlib**. Reusable analysis code lives under `src/`, executable runners live under `analysis/`, validation scripts and numerical tests live under `tests/`, raw data belongs under `data/raw/`, and generated artifacts are written under `outputs/`.

> The coverage-control outputs measure whether a target location is controlled by the offense or defense at release. They are not yet a complete good-decision/bad-decision grade for the quarterback.

## Project goals

This project is designed to move from descriptive quarterback statistics toward spatial and kinematic evaluation of passing decisions.

The current workflow can answer questions such as:

- Where does a quarterback complete, miss, or turn over passes?
- Which field areas produce the strongest and weakest completion percentages?
- Which route runner or coverage defender controls the intended target at release?
- How does player velocity change the apparent throwing window?
- How much earlier can the best offensive player reach the target than the best defender?
- How can these results support weekly attack planning and quarterback self-scouting?

## Analysis overview

### Passing-location analysis

The first stage identifies NFC North quarterbacks from `pff_offense.csv`, obtains their passer-play keys, joins those plays into `pff_pbp.csv`, classifies pass outcomes, and produces two figures per quarterback.

QB identity originates from:

```text
pff_offense.pff_PLAYERID
```

Passer plays are identified with:

```text
pff_offense.pff_PASSER
```

The primary PFF join is:

```text
pff_GAMEID + pff_PLAYID
```

GSIS identifiers are retained as secondary reconciliation keys and as the bridge to tracking data.

Passes are grouped into a 3 × 3 field matrix:

```text
Horizontal: Left | Center | Right
Depth:      Short (< 7) | Medium (7-20) | Long (20+)
```

Outputs:

```text
outputs/
├── <qb>_pass_attempts.png
└── <qb>_completion_by_area.png
```

The runner performs reconciliation checks before accepting the figures, including duplicate-key checks, unmatched-play checks, team agreement, GSIS agreement, attempt/exclusion reconciliation, and location-summary reconciliation.

### Release-frame coverage-control analysis

The second stage is play-centric. It evaluates selected pass plays at the tracking timestamp where:

```text
event == "pass_forward"
```

The generator population is intentionally role-aware:

```text
Offense: pff_offense.pff_ROLE == "Pass Route"
Defense: pff_defense.pff_ROLE == "Coverage"
```

Pass blockers and pass rushers are excluded from the receiving-space control map.

PFF player rows are currently connected to tracking entities through exact normalized player-name matching. The resolved tables retain both `pff_PLAYERID` and `pro_player_id`, allowing a formal ID crosswalk to replace the temporary name bridge later without changing the control-model APIs.

The validated tracking coordinate convention is:

```text
field_x = tracking.X
field_y = tracking.Y

negative X = offense left
positive X = offense right
Y = 0      = line of scrimmage
positive Y = downfield
```

The PFF target location is represented as:

```text
target_x = pff_PASSWIDTH - 26.5
target_y = pff_PASSDEPTH
```

## Coverage models

### 1. Static Voronoi

The static model uses each route runner's and coverage defender's observed position at `pass_forward`.

For a field point \(q\), ownership is assigned to the player with the smallest Euclidean distance:

```text
owner(q) = argmin_i distance(player_i, q)
```

This is the geometric baseline and primary sanity check.

### 2. Velocity-Projected Voronoi

The projected model estimates each player's release velocity from a past-only linear fit, then moves the Voronoi site forward by a configurable horizon:

```text
projected_position = release_position + velocity × projection_horizon
```

The default projection horizon is:

```text
0.50 seconds
```

The result remains a hard-edged polygonal Voronoi diagram, but regions expand and contract according to the direction and magnitude of player movement.

### 3. Kinematic Arrival-Time Control (KATC)

KATC assigns field control according to the earliest estimated player arrival time rather than simple distance.

The model uses:

- observed release position;
- observed release velocity;
- one shared maximum player speed;
- one shared maximum player acceleration.

Current defaults:

```text
Maximum speed:        11.0 yards/second
Maximum acceleration: 7.0 yards/second²
```

At the target point:

```text
KATC margin = best defense arrival time - best offense arrival time
```

Interpretation:

```text
Positive margin = offense arrives first
Negative margin = defense arrives first
Near zero       = contested target
```

The code currently uses `kinematic_*` names in several output columns and filenames. In project documentation and reporting, this model is referred to as **Kinematic Arrival-Time Control (KATC)**.

## Repository layout

```text
2025_NFC_North_Player_Tracking/
│
├── analysis/
│   ├── qb_analyzer.py
│   └── coverage_analyzer.py
│
├── data/
│   └── raw/
│       ├── pff_pbp.csv
│       ├── pff_offense.csv
│       ├── pff_defense.csv
│       └── tracking_sample.csv
│
├── outputs/
│   ├── <qb>_pass_attempts.png
│   ├── <qb>_completion_by_area.png
│   ├── coverage/
│   │   ├── <game>_<play>_coverage_control.png
│   │   ├── <game>_<play>_static_control.png
│   │   ├── <game>_<play>_velocity_control.png
│   │   ├── <game>_<play>_kinematic_control.png
│   │   ├── <game>_<play>_coverage_summary.csv
│   │   ├── <game>_<play>_generator_states.csv
│   │   ├── <game>_<play>_player_resolution.csv
│   │   ├── <game>_<play>_diagnostics.csv
│   │   ├── <game>_<play>_target_player_values.csv
│   │   └── coverage_summary.csv
│   └── diagnostics/
│       └── validation outputs
│
├── src/
│   └── nfc_north_player_tracking/
│       ├── __init__.py
│       ├── config.py
│       ├── qb_queries.py
│       ├── qb_diagnostics.py
│       ├── field_plot.py
│       ├── coverage_queries.py
│       ├── player_resolution.py
│       ├── kinematics.py
│       ├── dominant_regions.py
│       ├── coverage_analysis.py
│       ├── coverage_diagnostics.py
│       └── coverage_plot.py
│
├── tests/
│   ├── find_nfc_north_qbs.py
│   ├── audit_pass_filters.py
│   ├── validate_tracking_player_names.py
│   ├── validate_tracking_coordinates.py
│   └── test_dominant_regions.py
│
├── COVERAGE_ANALYSIS.md
└── README.md
```

## Data requirements

Place the following files in `data/raw/` using the exact filenames expected by the default runners:

```text
data/raw/pff_pbp.csv
data/raw/pff_offense.csv
data/raw/pff_defense.csv
data/raw/tracking_sample.csv
```

### `pff_pbp.csv`

Play-level data used for:

- game and play identifiers;
- GSIS tracking keys;
- pass result;
- pass depth, direction, and width;
- intended receiver;
- time to throw;
- pass-population filters;
- contextual football variables.

### `pff_offense.csv`

Player-play offensive data used for:

- authoritative quarterback identity;
- passer-play membership;
- route-runner identification;
- intended-receiver resolution;
- offensive player names and PFF IDs.

### `pff_defense.csv`

Player-play defensive data used for:

- coverage-defender identification;
- pass-rusher exclusion;
- batted-pass filtering;
- defensive player names and PFF IDs.

### `tracking_sample.csv`

Frame-level tracking data used for:

- `ball_snap` and `pass_forward` event timing;
- player and football positions;
- release velocity estimation;
- static, projected, and KATC field-control calculations.

The selected tracking play should contain the full set of tracked players, the football row, and a valid `pass_forward` event. The coverage runner joins tracking with all three available keys:

```text
pff_PLAYID     -> tracking.pff_play_id
pff_GSISGAMEKEY -> tracking.game_key
pff_GSISPLAYID  -> tracking.gsis_play_id
```

Follow the restrictions of your data license when storing or distributing raw PFF and tracking data.

## Environment setup with uv

The examples below use Python 3.12. Python 3.11 or newer is recommended for the current codebase.

### 1. Install uv

See the official uv installation guide:

<https://docs.astral.sh/uv/getting-started/installation/>

Common installation commands are:

**macOS and Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Confirm the installation:

```bash
uv --version
```

### 2. Create the virtual environment

From the repository root:

```bash
uv venv --python 3.12
```

### 3. Activate the environment

**macOS and Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install runtime dependencies

```bash
uv pip install polars numpy matplotlib
```

SciPy and Shapely are not required. The Voronoi cells are constructed with the project's internal half-plane clipping implementation.

### 5. Add `src/` to the Python path

The coverage runner adds `src/` automatically, but exporting `PYTHONPATH` makes all scripts and interactive sessions consistent, including the original quarterback runner.

**macOS and Linux**

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

**Windows PowerShell**

```powershell
$env:PYTHONPATH = "$PWD\src"
```

After activation and dependency installation, run all project commands with the environment's normal `python` executable.

### Optional project-managed uv workflow

The current repository can run with `uv venv` and `uv pip install` as shown above. If a `pyproject.toml` and `uv.lock` are added later, environment setup can be reduced to:

```bash
uv sync
```

Then activate `.venv` and continue using the `python` commands below.

## Quick start

From the repository root, after activating the virtual environment:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python analysis/qb_analyzer.py
python analysis/coverage_analyzer.py
```

On Windows PowerShell, set `$env:PYTHONPATH` as shown in the setup section before running the same Python scripts.

## Running the passing-location analysis

Run the NFC North quarterback analysis:

```bash
python analysis/qb_analyzer.py
```

The runner:

1. finds NFC North quarterbacks for `CHI`, `DET`, `GB`, and `MIN`;
2. isolates each quarterback's passer-play keys from `pff_offense.csv`;
3. joins those keys into `pff_pbp.csv`;
4. classifies completions, incompletions, and interceptions;
5. separates located and unlocated attempts;
6. runs leakage and reconciliation diagnostics;
7. saves two figures per quarterback.

## Running the coverage-control analysis

### Representative default play

```bash
python analysis/coverage_analyzer.py
```

The validated default is:

```text
pff_GAMEID = 28430
pff_PLAYID = 6408068
```

The default exists as a smoke test and development reference. It should not be interpreted as the only supported play.

### One selected play

```bash
python analysis/coverage_analyzer.py --play 28430:6408068
```

### Multiple selected plays

```bash
python analysis/coverage_analyzer.py \
    --play 28430:6408068 \
    --play 28430:6408079
```

### Intended-receiver-only ownership

By default, the offense controls a target when any route runner owns the location:

```text
ANY_ROUTE_RUNNER
```

To compare only the intended receiver against the defense:

```bash
python analysis/coverage_analyzer.py \
    --play 28430:6408068 \
    --ownership-mode INTENDED_RECEIVER_ONLY
```

### Parameter sensitivity

```bash
python analysis/coverage_analyzer.py \
    --play 28430:6408068 \
    --projection-horizon 0.50 \
    --max-speed 11.0 \
    --max-acceleration 7.0 \
    --distance-grid-spacing 0.25 \
    --kinematic-grid-spacing 0.25 \
    --kinematic-time-step 0.05
```

For a smoother KATC surface on selected plays, reduce the field-surface time step:

```bash
python analysis/coverage_analyzer.py \
    --play 28430:6408068 \
    --kinematic-time-step 0.025
```

This increases runtime but can reduce small numerical islands and staircase artifacts in the rendered control regions.

### Inspect all options

```bash
python analysis/coverage_analyzer.py --help
```

## Coverage outputs

For each requested play, the coverage runner writes:

```text
outputs/coverage/
├── <game>_<play>_coverage_control.png
├── <game>_<play>_static_control.png
├── <game>_<play>_velocity_control.png
├── <game>_<play>_kinematic_control.png
├── <game>_<play>_coverage_summary.csv
├── <game>_<play>_generator_states.csv
├── <game>_<play>_player_resolution.csv
├── <game>_<play>_diagnostics.csv
└── <game>_<play>_target_player_values.csv
```

The multi-play summary is:

```text
outputs/coverage/coverage_summary.csv
```

### Important result fields

```text
static_target_in_offense_control
velocity_target_in_offense_control
kinematic_target_in_offense_control

static_target_control_margin
velocity_target_control_margin
kinematic_target_control_margin_s

static_target_owner_name
velocity_target_owner_name
kinematic_target_owner_name
```

For static and projected Voronoi, the margin is measured in yards. For KATC, the margin is measured in seconds.

## Validation and tests

Run the data-validation gates before adding new representative plays.

### Player-name bridge

```bash
python tests/validate_tracking_player_names.py
```

Or select one or more plays:

```bash
python tests/validate_tracking_player_names.py \
    --play 28430:6408068
```

The play should not proceed to field-control analysis when a required route runner or coverage defender is unmatched or ambiguous.

### Tracking coordinates and events

```bash
python tests/validate_tracking_coordinates.py
```

This validates:

- one `ball_snap` time;
- one `pass_forward` time;
- football near the LOS at snap;
- complete release-frame rows;
- tracking/PFF time-to-throw agreement;
- target direction and width agreement;
- the working `X`/`Y` coordinate convention.

### Pass-filter audit

```bash
python tests/audit_pass_filters.py
```

The strict coverage population excludes, among other cases:

- no-plays;
- screens;
- RPO throws;
- trick plays;
- throwaways;
- batted passes;
- hit-as-threw plays;
- unsupported pass results;
- missing target locations;
- missing intended receivers.

### Numerical unit tests

```bash
python -m unittest tests/test_dominant_regions.py -v
```

Or discover all conventional unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Diagnostics philosophy

The project favors explicit failure over silent row loss or join amplification.

The coverage diagnostics check items such as:

- regular-pass eligibility;
- duplicate release-frame entities;
- exactly one football row at release;
- snap and release event agreement;
- successful player resolution;
- expected route-runner and defender counts;
- sufficient velocity history;
- physically plausible observed speeds;
- field and target bounds;
- complete control surfaces;
- agreement between exact target evaluation and plotted grid ownership.

By default, a failed coverage diagnostic raises an error. Use the following only for deliberate debugging:

```bash
python analysis/coverage_analyzer.py \
    --play 28430:6408068 \
    --keep-failed-diagnostics
```

Similarly, an otherwise excluded pass can be inspected deliberately with:

```bash
python analysis/coverage_analyzer.py \
    --play <game>:<play> \
    --allow-ineligible-pass
```

## Interpreting results

A positive control result means the model favors an offensive route runner at the PFF target location.

```text
target_in_offense_control = true
```

This does not automatically mean the quarterback made a good decision. The current receiving-space model does not yet fully incorporate:

- pass trajectory and ball speed;
- pass-lane defenders and tipped-ball risk;
- pressure and time remaining in the pocket;
- throw placement error;
- receiver catch probability;
- expected yards or expected points;
- the value of all alternative targets.

A future decision-quality model should compare the chosen target against every viable alternative and should separate target selection from throw execution.

## Configuration

Shared model defaults are defined in:

```text
src/nfc_north_player_tracking/config.py
```

Important parameters include:

```text
velocity_projection_horizon_seconds
velocity_lookback_seconds
max_player_speed_yards_per_second
max_player_acceleration_yards_per_second_squared
distance_grid_resolution_yards
kinematic_grid_resolution_yards
kinematic_grid_time_step_seconds
kinematic_target_time_step_seconds
kinematic_max_time_seconds
```

Prefer command-line overrides for experiments. Change `config.py` only when intentionally changing project-wide defaults.

## Troubleshooting

### `ModuleNotFoundError: nfc_north_player_tracking`

Run from the repository root and set `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### Input file not found

Confirm the expected files exist:

```text
data/raw/pff_pbp.csv
data/raw/pff_offense.csv
data/raw/pff_defense.csv
data/raw/tracking_sample.csv
```

Custom paths can be supplied to the coverage runner:

```bash
python analysis/coverage_analyzer.py \
    --pbp-csv /path/to/pff_pbp.csv \
    --offense-csv /path/to/pff_offense.csv \
    --defense-csv /path/to/pff_defense.csv \
    --tracking-csv /path/to/tracking.csv
```

### Coverage analysis reports unresolved players

Run:

```bash
python tests/validate_tracking_player_names.py --play <game>:<play>
```

Review the generated name-bridge audit before modifying the resolver. The current implementation intentionally avoids fuzzy matching.

### Coverage map has unexpected orientation

Run:

```bash
python tests/validate_tracking_coordinates.py --play <game>:<play>
```

Do not add a coordinate reflection or rotation until the audit demonstrates that the current offense-relative convention fails for that play.

### KATC plot has small islands or jagged boundaries

Try a smaller simulation time step:

```bash
python analysis/coverage_analyzer.py \
    --play <game>:<play> \
    --kinematic-time-step 0.025
```

The exact target calculation may remain stable even when the field-wide display contains coarse-grid artifacts. Review the target values and diagnostics CSV alongside the figure.

### Headless server cannot open a display

The scripts save figures rather than requiring an interactive window. If a Matplotlib backend error occurs, set:

```bash
export MPLBACKEND=Agg
```

## Current limitations and roadmap

Current limitations:

- player identity uses normalized names rather than a formal PFF-to-tracking ID crosswalk;
- the model evaluates receiving-space control, not the full football flight path;
- the current KATC motion model uses shared physical limits for every player;
- field-control maps are evaluated only at the release snapshot;
- the analysis is currently optimized for selected representative plays rather than every pass in a season;
- target control is not yet converted into expected completion probability or expected value.

Planned extensions:

1. Replace normalized-name matching with a persistent player-ID crosswalk.
2. Add ball-trajectory and pass-lane interception analysis.
3. Compare intended-receiver ownership with any-route-runner ownership.
4. Evaluate every route runner as a counterfactual target.
5. Add pressure, coverage family, formation, and route-concept splits.
6. Aggregate control metrics into attack-plan and quarterback self-scout reports.
7. Separate quarterback decision quality from throw execution and outcome.

## Research use

This repository is intended as an auditable research and scouting workflow. When presenting results:

- report the model and parameter values used;
- retain diagnostics with every selected play;
- distinguish exact target metrics from visual grid approximations;
- describe KATC as an implementation of arrival-time or dominant-region field control;
- avoid presenting target control alone as a definitive quarterback decision grade.
