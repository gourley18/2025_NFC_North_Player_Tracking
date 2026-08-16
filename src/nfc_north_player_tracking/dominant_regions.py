"""Static, projected-site, and acceleration-limited coverage-control models."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np

from nfc_north_player_tracking.config import (
    OWNERSHIP_MODE_ANY_ROUTE_RUNNER,
    OWNERSHIP_MODE_INTENDED_RECEIVER,
    VALID_OWNERSHIP_MODES,
    CoverageParameters,
)


@dataclass(frozen=True)
class ControlSurface:
    """A field-wide ownership surface for one control model."""

    model_key: str
    model_label: str
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    owner_index: np.ndarray
    control_margin: np.ndarray
    metric_unit: str
    site_positions: np.ndarray
    polygons: tuple[np.ndarray, ...] | None = None


@dataclass(frozen=True)
class TargetControlResult:
    """Exact target-point ownership and offense-versus-defense margin."""

    model_key: str
    model_label: str
    target_x: float
    target_y: float
    owner_index: int
    owner_name: str
    owner_side: str
    owner_value: float
    offense_reference_name: str
    offense_value: float
    defense_reference_name: str
    defense_value: float
    control_margin: float
    margin_unit: str
    target_in_offense_control: bool
    target_near_boundary: bool
    ownership_mode: str

    def to_dict(self, prefix: str | None = None) -> dict[str, object]:
        """Return a flat record suitable for one-row CSV output."""
        key = self.model_key if prefix is None else prefix
        return {
            f"{key}_target_owner_name": self.owner_name,
            f"{key}_target_owner_side": self.owner_side,
            f"{key}_target_owner_value": self.owner_value,
            f"{key}_offense_reference_name": self.offense_reference_name,
            f"{key}_offense_value": self.offense_value,
            f"{key}_defense_reference_name": self.defense_reference_name,
            f"{key}_defense_value": self.defense_value,
            f"{key}_target_control_margin": self.control_margin,
            f"{key}_margin_unit": self.margin_unit,
            f"{key}_target_in_offense_control": (
                self.target_in_offense_control
            ),
            f"{key}_target_near_boundary": self.target_near_boundary,
            f"{key}_ownership_mode": self.ownership_mode,
        }


def _validate_inputs(
    positions: np.ndarray,
    names: Sequence[str],
    sides: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    names_array = np.asarray(list(names), dtype=object)
    sides_array = np.asarray([str(side).upper() for side in sides], dtype=object)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (players, 2).")
    if len(positions) == 0:
        raise ValueError("At least one generator is required.")
    if len(names_array) != len(positions) or len(sides_array) != len(positions):
        raise ValueError("positions, names, and sides must have equal lengths.")
    if not np.isfinite(positions).all():
        raise ValueError("Generator positions contain non-finite values.")
    if not np.any(sides_array == "OFFENSE"):
        raise ValueError("At least one offensive generator is required.")
    if not np.any(sides_array == "DEFENSE"):
        raise ValueError("At least one defensive generator is required.")
    return positions, names_array, sides_array


def _grid_axis(minimum: float, maximum: float, resolution: float) -> np.ndarray:
    if maximum <= minimum:
        raise ValueError("Grid maximum must exceed its minimum.")
    if resolution <= 0:
        raise ValueError("Grid resolution must be positive.")
    points = int(ceil((maximum - minimum) / resolution)) + 1
    return np.linspace(minimum, maximum, points)


def build_grid(
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x coordinates, y coordinates, and flattened field points."""
    x_min, x_max, y_min, y_max = bounds
    x_values = _grid_axis(x_min, x_max, resolution)
    y_values = _grid_axis(y_min, y_max, resolution)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    return x_values, y_values, points


def _control_margin_from_costs(
    costs: np.ndarray,
    sides: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return owner indices and positive-is-offense control margins."""
    if costs.ndim < 2:
        raise ValueError("costs must have a leading player dimension.")
    owner = np.argmin(costs, axis=0)
    offense_cost = np.min(costs[sides == "OFFENSE"], axis=0)
    defense_cost = np.min(costs[sides == "DEFENSE"], axis=0)
    return owner, defense_cost - offense_cost


def _clip_polygon_to_half_plane(
    polygon: np.ndarray,
    normal: np.ndarray,
    offset: float,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Clip a polygon to ``normal dot point <= offset``."""
    if len(polygon) == 0:
        return polygon

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(np.dot(normal, previous) - offset)
    previous_inside = previous_value <= tolerance

    for current in polygon:
        current_value = float(np.dot(normal, current) - offset)
        current_inside = current_value <= tolerance

        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(np.dot(normal, direction))
            if abs(denominator) > tolerance:
                fraction = (offset - float(np.dot(normal, previous))) / denominator
                fraction = min(1.0, max(0.0, fraction))
                output.append(previous + fraction * direction)

        if current_inside:
            output.append(current)

        previous = current
        previous_inside = current_inside

    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def voronoi_polygons(
    sites: np.ndarray,
    bounds: tuple[float, float, float, float],
    duplicate_tolerance: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    """Construct exact rectangularly clipped Euclidean Voronoi cells.

    The implementation uses pairwise half-plane clipping and therefore needs no
    SciPy or geometry-library dependency.
    """
    sites = np.asarray(sites, dtype=float)
    x_min, x_max, y_min, y_max = bounds
    rectangle = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=float,
    )

    pairwise = sites[:, None, :] - sites[None, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    duplicate_mask = (distances < duplicate_tolerance) & (
        ~np.eye(len(sites), dtype=bool)
    )
    if duplicate_mask.any():
        pairs = np.argwhere(np.triu(duplicate_mask, k=1))
        raise ValueError(f"Voronoi sites are duplicated at player pairs {pairs.tolist()}.")

    cells: list[np.ndarray] = []
    for index, site in enumerate(sites):
        polygon = rectangle.copy()
        for other_index, other_site in enumerate(sites):
            if other_index == index:
                continue
            normal = 2.0 * (other_site - site)
            offset = float(np.dot(other_site, other_site) - np.dot(site, site))
            polygon = _clip_polygon_to_half_plane(polygon, normal, offset)
            if len(polygon) == 0:
                break
        cells.append(polygon)
    return tuple(cells)


def build_distance_surface(
    model_key: str,
    model_label: str,
    sites: np.ndarray,
    names: Sequence[str],
    sides: Sequence[str],
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> ControlSurface:
    """Build a distance-based field ownership surface and exact polygons."""
    sites, _, side_array = _validate_inputs(sites, names, sides)
    x_values, y_values, points = build_grid(bounds, resolution)
    distances = np.linalg.norm(
        points[None, :, :] - sites[:, None, :],
        axis=2,
    )
    owner, margin = _control_margin_from_costs(distances, side_array)
    shape = (len(y_values), len(x_values))
    return ControlSurface(
        model_key=model_key,
        model_label=model_label,
        x_coordinates=x_values,
        y_coordinates=y_values,
        owner_index=owner.reshape(shape),
        control_margin=margin.reshape(shape),
        metric_unit="yards",
        site_positions=sites,
        polygons=voronoi_polygons(sites, bounds),
    )


def _offense_candidate_indices(
    names: np.ndarray,
    sides: np.ndarray,
    ownership_mode: str,
    intended_receiver_name: str | None,
) -> np.ndarray:
    mode = ownership_mode.upper()
    if mode not in VALID_OWNERSHIP_MODES:
        raise ValueError(
            f"Unknown ownership mode {ownership_mode!r}; expected one of "
            f"{sorted(VALID_OWNERSHIP_MODES)}."
        )
    if mode == OWNERSHIP_MODE_ANY_ROUTE_RUNNER:
        return np.flatnonzero(sides == "OFFENSE")

    if not intended_receiver_name:
        raise ValueError(
            "INTENDED_RECEIVER_ONLY ownership requires an intended receiver."
        )
    indices = np.flatnonzero(names == intended_receiver_name)
    indices = indices[sides[indices] == "OFFENSE"]
    if len(indices) != 1:
        raise ValueError(
            f"Expected one offensive generator named {intended_receiver_name!r}; "
            f"found {len(indices)}."
        )
    return indices


def evaluate_target_values(
    model_key: str,
    model_label: str,
    target: tuple[float, float],
    values: np.ndarray,
    names: Sequence[str],
    sides: Sequence[str],
    ownership_mode: str,
    intended_receiver_name: str | None,
    margin_unit: str,
    near_boundary_threshold: float,
) -> TargetControlResult:
    """Evaluate exact player costs/times at the PFF target point."""
    values = np.asarray(values, dtype=float)
    names_array = np.asarray(list(names), dtype=object)
    sides_array = np.asarray([str(side).upper() for side in sides], dtype=object)
    if values.shape != (len(names_array),):
        raise ValueError("Target values must contain one value per generator.")
    if not np.isfinite(values).all():
        raise ValueError("Target evaluation contains non-finite player values.")

    offense_indices = _offense_candidate_indices(
        names_array,
        sides_array,
        ownership_mode,
        intended_receiver_name,
    )
    defense_indices = np.flatnonzero(sides_array == "DEFENSE")
    if len(defense_indices) == 0:
        raise ValueError("Target evaluation requires a defensive generator.")

    eligible_owner_indices = np.concatenate([offense_indices, defense_indices])
    owner_local = int(np.argmin(values[eligible_owner_indices]))
    owner_index = int(eligible_owner_indices[owner_local])
    offense_local = int(np.argmin(values[offense_indices]))
    offense_index = int(offense_indices[offense_local])
    defense_local = int(np.argmin(values[defense_indices]))
    defense_index = int(defense_indices[defense_local])

    offense_value = float(values[offense_index])
    defense_value = float(values[defense_index])
    margin = defense_value - offense_value

    return TargetControlResult(
        model_key=model_key,
        model_label=model_label,
        target_x=float(target[0]),
        target_y=float(target[1]),
        owner_index=owner_index,
        owner_name=str(names_array[owner_index]),
        owner_side=str(sides_array[owner_index]),
        owner_value=float(values[owner_index]),
        offense_reference_name=str(names_array[offense_index]),
        offense_value=offense_value,
        defense_reference_name=str(names_array[defense_index]),
        defense_value=defense_value,
        control_margin=margin,
        margin_unit=margin_unit,
        target_in_offense_control=bool(margin > 0.0),
        target_near_boundary=bool(abs(margin) < near_boundary_threshold),
        ownership_mode=ownership_mode.upper(),
    )


def evaluate_distance_target(
    model_key: str,
    model_label: str,
    target: tuple[float, float],
    sites: np.ndarray,
    names: Sequence[str],
    sides: Sequence[str],
    ownership_mode: str,
    intended_receiver_name: str | None,
    near_boundary_threshold_yards: float,
) -> TargetControlResult:
    """Evaluate static or projected-site distance control at one target."""
    sites, _, _ = _validate_inputs(sites, names, sides)
    target_array = np.asarray(target, dtype=float)
    distances = np.linalg.norm(sites - target_array[None, :], axis=1)
    return evaluate_target_values(
        model_key=model_key,
        model_label=model_label,
        target=target,
        values=distances,
        names=names,
        sides=sides,
        ownership_mode=ownership_mode,
        intended_receiver_name=intended_receiver_name,
        margin_unit="yards",
        near_boundary_threshold=near_boundary_threshold_yards,
    )


def simulate_arrival_times(
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    targets: np.ndarray,
    max_speed: float,
    max_acceleration: float,
    time_step: float,
    max_time: float,
    arrival_radius: float,
) -> np.ndarray:
    """Approximate earliest arrival times under shared speed/acceleration caps.

    At each step the athlete selects a desired velocity toward each queried
    point. Desired speed is limited by both ``max_speed`` and the braking speed
    ``sqrt(2 * max_acceleration * remaining_distance)``. The velocity vector can
    change by at most ``max_acceleration * time_step`` per step, so movement
    already directed away from or across the target must first be redirected.
    """
    start_position = np.asarray(start_position, dtype=float).reshape(2)
    start_velocity = np.asarray(start_velocity, dtype=float).reshape(2)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)

    for name, value in {
        "max_speed": max_speed,
        "max_acceleration": max_acceleration,
        "time_step": time_step,
        "max_time": max_time,
        "arrival_radius": arrival_radius,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value}.")
    if not (
        np.isfinite(start_position).all()
        and np.isfinite(start_velocity).all()
        and np.isfinite(targets).all()
    ):
        raise ValueError("Kinematic inputs contain non-finite values.")

    target_count = len(targets)
    positions = np.repeat(start_position[None, :], target_count, axis=0)

    initial_speed = float(np.linalg.norm(start_velocity))
    if initial_speed > max_speed:
        start_velocity = start_velocity * (max_speed / initial_speed)
    velocities = np.repeat(start_velocity[None, :], target_count, axis=0)

    arrival_times = np.full(target_count, np.inf, dtype=float)
    active = np.ones(target_count, dtype=bool)
    initial_distance = np.linalg.norm(targets - positions, axis=1)
    initially_arrived = initial_distance <= arrival_radius
    arrival_times[initially_arrived] = 0.0
    active[initially_arrived] = False

    step_count = int(ceil(max_time / time_step))
    for step in range(step_count):
        if not active.any():
            break

        active_indices = np.flatnonzero(active)
        position = positions[active_indices]
        velocity = velocities[active_indices]
        target = targets[active_indices]

        displacement = target - position
        distance = np.linalg.norm(displacement, axis=1)
        direction = np.divide(
            displacement,
            distance[:, None],
            out=np.zeros_like(displacement),
            where=distance[:, None] > 1e-12,
        )

        remaining = np.maximum(0.0, distance - arrival_radius)
        braking_speed = np.sqrt(2.0 * max_acceleration * remaining)
        desired_speed = np.minimum(max_speed, braking_speed)
        desired_velocity = direction * desired_speed[:, None]

        velocity_change = desired_velocity - velocity
        change_norm = np.linalg.norm(velocity_change, axis=1)
        maximum_change = max_acceleration * time_step
        scale = np.ones_like(change_norm)
        exceeds_acceleration = change_norm > maximum_change
        scale[exceeds_acceleration] = (
            maximum_change / change_norm[exceeds_acceleration]
        )
        next_velocity = velocity + velocity_change * scale[:, None]

        next_speed = np.linalg.norm(next_velocity, axis=1)
        exceeds_speed = next_speed > max_speed
        next_velocity[exceeds_speed] *= (
            max_speed / next_speed[exceeds_speed]
        )[:, None]

        next_position = position + 0.5 * (velocity + next_velocity) * time_step

        segment = next_position - position
        segment_length_squared = np.sum(segment * segment, axis=1)
        fraction = np.divide(
            np.sum((target - position) * segment, axis=1),
            segment_length_squared,
            out=np.zeros_like(segment_length_squared),
            where=segment_length_squared > 1e-14,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = position + fraction[:, None] * segment
        closest_distance = np.linalg.norm(target - closest, axis=1)
        arrived = closest_distance <= arrival_radius

        if arrived.any():
            arrived_indices = active_indices[arrived]
            arrival_times[arrived_indices] = (
                step * time_step + fraction[arrived] * time_step
            )
            active[arrived_indices] = False

        remaining_mask = ~arrived
        remaining_indices = active_indices[remaining_mask]
        positions[remaining_indices] = next_position[remaining_mask]
        velocities[remaining_indices] = next_velocity[remaining_mask]

    return arrival_times


def build_kinematic_surface(
    positions: np.ndarray,
    velocities: np.ndarray,
    names: Sequence[str],
    sides: Sequence[str],
    bounds: tuple[float, float, float, float],
    parameters: CoverageParameters,
) -> ControlSurface:
    """Build the advanced acceleration-limited time-to-arrival surface."""
    positions, _, side_array = _validate_inputs(positions, names, sides)
    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions.")
    if not np.isfinite(velocities).all():
        raise ValueError("Generator velocities contain non-finite values.")

    x_values, y_values, points = build_grid(
        bounds,
        parameters.kinematic_grid_resolution_yards,
    )
    arrival_radius = max(
        0.05,
        parameters.kinematic_grid_resolution_yards / 2.0,
    )
    player_times = []
    for position, velocity in zip(positions, velocities, strict=True):
        player_times.append(
            simulate_arrival_times(
                start_position=position,
                start_velocity=velocity,
                targets=points,
                max_speed=parameters.max_player_speed_yards_per_second,
                max_acceleration=(
                    parameters.max_player_acceleration_yards_per_second_squared
                ),
                time_step=parameters.kinematic_grid_time_step_seconds,
                max_time=parameters.kinematic_max_time_seconds,
                arrival_radius=arrival_radius,
            )
        )
    costs = np.vstack(player_times)
    unreachable = np.all(~np.isfinite(costs), axis=0)
    if unreachable.any():
        raise ValueError(
            f"The kinematic model did not reach {int(unreachable.sum())} grid points "
            "within the configured maximum time."
        )

    owner, margin = _control_margin_from_costs(costs, side_array)
    shape = (len(y_values), len(x_values))
    return ControlSurface(
        model_key="kinematic",
        model_label="Advanced kinematic",
        x_coordinates=x_values,
        y_coordinates=y_values,
        owner_index=owner.reshape(shape),
        control_margin=margin.reshape(shape),
        metric_unit="seconds",
        site_positions=positions,
        polygons=None,
    )


def evaluate_kinematic_target(
    target: tuple[float, float],
    positions: np.ndarray,
    velocities: np.ndarray,
    names: Sequence[str],
    sides: Sequence[str],
    ownership_mode: str,
    intended_receiver_name: str | None,
    parameters: CoverageParameters,
) -> TargetControlResult:
    """Calculate exact target arrival times and the control margin in seconds."""
    positions, _, _ = _validate_inputs(positions, names, sides)
    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions.")

    target_array = np.asarray(target, dtype=float).reshape(1, 2)
    values = np.array(
        [
            simulate_arrival_times(
                start_position=position,
                start_velocity=velocity,
                targets=target_array,
                max_speed=parameters.max_player_speed_yards_per_second,
                max_acceleration=(
                    parameters.max_player_acceleration_yards_per_second_squared
                ),
                time_step=parameters.kinematic_target_time_step_seconds,
                max_time=parameters.kinematic_max_time_seconds,
                arrival_radius=parameters.kinematic_target_arrival_radius_yards,
            )[0]
            for position, velocity in zip(positions, velocities, strict=True)
        ],
        dtype=float,
    )
    return evaluate_target_values(
        model_key="kinematic",
        model_label="Advanced kinematic",
        target=target,
        values=values,
        names=names,
        sides=sides,
        ownership_mode=ownership_mode,
        intended_receiver_name=intended_receiver_name,
        margin_unit="seconds",
        near_boundary_threshold=parameters.target_near_boundary_margin_seconds,
    )
