from __future__ import annotations
"""nuPlan runtime adapter bundled with the v9.5.31 predicate package.

This file is based on the original ``nuplan_predicate_kg`` adapter recovered
from the user's base project.  The scenario-builder construction is made
compatible with the sensor-data-era nuPlan devkit API while preserving the
same FrameRecord interface consumed by v9.5.31.
"""

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class FrameRecord:
    scenario_token: str
    log_name: str
    scenario_type: str
    timestamp_us: int
    ego_state: Any
    tracked_objects: list[Any]
    traffic_lights: list[Any]
    map_api: Any
    route_roadblock_ids: list[str]


def _subsample_ratio(sample_interval_s: float) -> float:
    """Return the nuPlan 20 Hz subsampling ratio for the requested interval."""
    interval = float(sample_interval_s)
    if interval <= 0:
        raise ValueError("sample_interval_s must be > 0")
    # nuPlan lidar_pc base frequency is 20 Hz => 0.05 s.
    return max(0.0, min(1.0, 0.05 / interval))


def _find_db_files(dataset_root: Path) -> list[str]:
    """Find local nuPlan DB files, preferring the configured directory itself."""
    direct = sorted(str(p) for p in dataset_root.glob("*.db") if p.is_file())
    if direct:
        return direct
    nested = sorted(str(p) for p in dataset_root.rglob("*.db") if p.is_file())
    if nested:
        return nested
    raise FileNotFoundError(f"No nuPlan .db files found under: {dataset_root}")


def _infer_sensor_root(dataset_root: Path, cfg: dict[str, Any]) -> Path:
    """Resolve sensor_root without requiring it in the predicate CLI."""
    explicit = cfg.get("sensor_root") or os.environ.get("NUPLAN_SENSOR_ROOT")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(str(explicit)).expanduser())

    # User's current mini layout:
    #   <nuplan_data>/mini/data/cache/mini/*.db
    #   <nuplan_data>/downloads/mini_sensors/
    ancestors = list(dataset_root.parents)
    if len(ancestors) >= 4:
        candidates.append(ancestors[3] / "downloads" / "mini_sensors")

    # Common nuPlan v1.2 layout variants.
    for ancestor in ancestors[:6]:
        candidates.extend(
            [
                ancestor / "sensor_blobs",
                ancestor / "mini_sensors",
                ancestor / "downloads" / "mini_sensors",
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    # include_cameras=False means the predicate extractor does not read camera
    # blobs. Newer NuPlanScenarioBuilder versions still require a sensor_root
    # constructor argument, so use the DB root as a harmless existing fallback.
    return dataset_root


def _make_scenario_mapping(ScenarioMapping: Any, sample_interval_s: float) -> Any:
    ratio = _subsample_ratio(sample_interval_s)
    try:
        return ScenarioMapping({}, ratio)
    except TypeError:
        # Compatibility with the older API used by the recovered base project.
        return ScenarioMapping({}, 20.0, sample_interval_s)


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keyword arguments accepted by the installed nuPlan version."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def build_mini_scenarios(cfg: dict[str, Any]) -> list[Any]:
    """Build nuPlan scenarios; v9.5.31 applies YAML/max-scenario filtering later."""
    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
            NuPlanScenarioBuilder,
        )
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
            ScenarioMapping,
        )
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    except ImportError as exc:
        raise RuntimeError(
            "The official nuPlan devkit is required in the active environment."
        ) from exc

    dataset_root = Path(str(cfg["dataset_root"])).expanduser().resolve()
    map_root = Path(str(cfg["map_root"])).expanduser().resolve()
    map_version = str(cfg.get("map_version", "nuplan-maps-v1.0"))
    sample_interval_s = float(cfg.get("sample_interval_s", 0.5))
    verbose = bool(cfg.get("verbose", True))

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    if not map_root.exists():
        raise FileNotFoundError(f"map_root does not exist: {map_root}")

    db_files = _find_db_files(dataset_root)
    sensor_root = _infer_sensor_root(dataset_root, cfg)
    scenario_mapping = _make_scenario_mapping(ScenarioMapping, sample_interval_s)

    builder_kwargs = {
        "data_root": str(dataset_root),
        "map_root": str(map_root),
        "sensor_root": str(sensor_root),
        "db_files": db_files,
        "map_version": map_version,
        "include_cameras": False,
        "max_workers": None,
        "verbose": verbose,
        "scenario_mapping": scenario_mapping,
        "vehicle_parameters": get_pacifica_parameters(),
    }
    builder = NuPlanScenarioBuilder(
        **_supported_kwargs(NuPlanScenarioBuilder, builder_kwargs)
    )

    filter_kwargs = {
        "scenario_types": None,
        "scenario_tokens": None,
        "log_names": None,
        "map_names": None,
        # Do not pre-limit here. The v9.5.31 pipeline applies its YAML filter
        # first and then --max-scenarios, preserving requested scenario tokens.
        "num_scenarios_per_type": None,
        "limit_total_scenarios": None,
        "timestamp_threshold_s": None,
        "ego_displacement_minimum_m": None,
        "expand_scenarios": False,
        "remove_invalid_goals": False,
        "shuffle": False,
        "ego_start_speed_threshold": None,
        "ego_stop_speed_threshold": None,
        "speed_noise_tolerance": None,
        "token_set_path": None,
        "fraction_in_token_set_threshold": None,
        "ego_route_radius": None,
    }
    scenario_filter = ScenarioFilter(
        **_supported_kwargs(ScenarioFilter, filter_kwargs)
    )
    return list(builder.get_scenarios(scenario_filter, Sequential()))


def iter_frames(scenario: Any) -> Iterable[FrameRecord]:
    """Yield the normalized frame interface consumed by v9.5.31."""
    iterations = int(scenario.get_number_of_iterations())
    try:
        route_ids = [str(value) for value in scenario.get_route_roadblock_ids()]
    except Exception:
        route_ids = []

    for i in range(iterations):
        try:
            traffic_lights = list(scenario.get_traffic_light_status_at_iteration(i))
        except Exception:
            traffic_lights = []
        detections = scenario.get_tracked_objects_at_iteration(i)
        tracked = list(getattr(detections, "tracked_objects", detections))
        yield FrameRecord(
            scenario_token=str(scenario.token),
            log_name=str(scenario.log_name),
            scenario_type=str(scenario.scenario_type),
            timestamp_us=int(scenario.get_time_point(i).time_us),
            ego_state=scenario.get_ego_state_at_iteration(i),
            tracked_objects=tracked,
            traffic_lights=traffic_lights,
            map_api=scenario.map_api,
            route_roadblock_ids=route_ids,
        )
