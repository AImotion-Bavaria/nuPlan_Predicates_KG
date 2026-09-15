"""Hierarchical interaction detection for v9.5.26.

The existing ``np:follows`` implementation is preserved unchanged.
``np:changesLane`` adds a binary vehicle-like-to-target-driving-primitive
interaction using an offline topology-constrained temporal decoder over Lane and
LaneConnector candidates, center containment, local geometry, lateral motion,
and track continuity. Directed longitudinal continuation is suppressed while
lateral path transfers are expanded into left/right maneuvers with confirmed,
probable, or censored status. ``np:overtakes`` uses an internal two-stage
Case-1 trajectory search: a high-recall directed-pair scan followed by strict
footprint-aware lane matching and verification of a fully observed same-lane
departure, pass, and return.

``np:mergesInFrontOf`` and ``np:mergesBehind`` are vehicle-to-vehicle
relations anchored to a decoded lane change and a pre-existing vehicle stream
in the target Lane/LaneConnector path. ``np:crossesInFrontOf`` adds an observed
dynamic-road-user crossing-order event using the validated trajectory-conflict search.
``np:yieldsTo`` (v9.5.42) adds retrospectively verified behavioral priority giving:
it reuses verified crossing/merge events, requires a measurable subject motion
concession and later progression, and excludes ordinary stopping explained by an
unambiguous relevant RED signal. Unrelated map ambiguity, footprint overlap,
stopped-vehicle passing, and passing without return remain omitted.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point as ShapelyPoint
from shapely.ops import nearest_points

from ..base import *
from .geometry import footprint_metrics, oriented_box_polygon
from .map import (
    _baseline_length,
    _map_edge_ids,
    _project_progress_on_map_object,
    _record_direction,
)


FOLLOW_RULE_ID = "R-FOLLOW-HIERARCHICAL-001"

# Bicycles/cyclists are intentionally excluded from the first motor-vehicle
# car-following implementation because their lateral filtering and mixed-traffic
# behaviour require a separately defined rule.
_EXCLUDED_FOLLOW_TYPE_MARKERS = (
    "BICYCLE",
    "CYCLIST",
    "PEDESTRIAN",
    "STATIC",
    "BARRIER",
    "TRAFFIC_CONE",
    "CZONE_SIGN",
    "GENERIC_OBJECT",
)
_INCLUDED_FOLLOW_TYPE_MARKERS = (
    "EGO",
    "VEHICLE",
    "CAR",
    "TRUCK",
    "BUS",
    "MOTORCYCLE",
    "TRAILER",
    "CONSTRUCTION_VEHICLE",
)


@dataclass(frozen=True)
class _PathNode:
    object_id: str
    kind: str
    map_object: Any
    length_m: float


@dataclass(frozen=True)
class _PathEvidence:
    nodes: tuple[_PathNode, ...]
    relation: str
    subject_interval_m: tuple[float, float]
    object_interval_m: tuple[float, float]
    subject_center_progress_m: float
    object_center_progress_m: float
    center_path_distance_m: float
    bumper_gap_m: float


@dataclass(frozen=True)
class _LeaderAssessment:
    subject: dict[str, Any]
    obj: dict[str, Any]
    path: _PathEvidence
    geometry: dict[str, Any]
    subject_forward_speed_mps: Optional[float]
    object_forward_speed_mps: Optional[float]
    headway_s: Optional[float]
    mode: Optional[str]
    valid_follow_envelope: bool
    omission_reason: Optional[str]


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_follow_vehicle(record: dict[str, Any]) -> bool:
    if str(record.get("track_token")) == "ego":
        return True
    agent_type = _normalized_agent_type(record)
    if any(marker in agent_type for marker in _EXCLUDED_FOLLOW_TYPE_MARKERS):
        return False
    return any(marker in agent_type for marker in _INCLUDED_FOLLOW_TYPE_MARKERS)


def _map_kind_from_layer(layer: Any) -> str:
    name = str(getattr(layer, "name", layer)).upper()
    return "lane_connector" if "CONNECTOR" in name else "lane"


def _resolve_map_object(
    frame: Any,
    object_id: str,
    cache: dict[str, Optional[_PathNode]],
) -> Optional[_PathNode]:
    object_id = str(object_id)
    if object_id in cache:
        return cache[object_id]

    result = None
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        try:
            map_object = frame.map_api.get_map_object(object_id, layer)
        except Exception:
            map_object = None
        if map_object is None:
            continue
        length = _finite_float(_baseline_length(map_object))
        if length is None or length <= 0.0:
            continue
        result = _PathNode(
            object_id=object_id,
            kind=_map_kind_from_layer(layer),
            map_object=map_object,
            length_m=length,
        )
        break

    cache[object_id] = result
    return result


def _record_path_node(record: dict[str, Any]) -> Optional[_PathNode]:
    object_id = record.get("primary_map_object_id")
    kind = record.get("primary_map_kind")
    map_object = record.get("primary_map_object")
    if not object_id or kind not in {"lane", "lane_connector"} or map_object is None:
        return None
    length = _finite_float(record.get("baseline_length_m"))
    if length is None:
        length = _finite_float(_baseline_length(map_object))
    if length is None or length <= 0.0:
        return None
    return _PathNode(str(object_id), str(kind), map_object, length)


def _unique_forward_path(
    snapshot: dict[str, Any],
    subject: dict[str, Any],
    obj: dict[str, Any],
    object_cache: dict[str, Optional[_PathNode]],
) -> tuple[Optional[tuple[_PathNode, ...]], Optional[str]]:
    """Return one unbranched directed path from subject map object to object.

    The detector intentionally refuses to infer an unobserved route choice at a
    split.  Each intermediate object must have exactly one outgoing edge until
    the target object is reached.  This still supports lane→connector→lane,
    connector→lane, merges, roundabout continuation, and longer unbranched
    sequences while avoiding false positives across alternative turn branches.
    """
    if subject.get("map_match_ambiguous") or obj.get("map_match_ambiguous"):
        return None, "ambiguous_primary_map_match"

    start = _record_path_node(subject)
    target = _record_path_node(obj)
    if start is None or target is None:
        return None, "missing_primary_map_object"

    object_cache.setdefault(start.object_id, start)
    object_cache.setdefault(target.object_id, target)

    if start.object_id == target.object_id:
        if start.kind != target.kind:
            return None, "map_object_kind_conflict"
        return (start,), "same_map"

    max_hops = max(1, int(getattr(ARGS, "follow_max_path_hops", 3)))
    path = [start]
    visited = {start.object_id}
    current = start

    for _ in range(max_hops):
        outgoing_ids = sorted(str(value) for value in _map_edge_ids(current.map_object, "outgoing"))
        if not outgoing_ids:
            return None, "no_forward_map_continuation"

        # A target on one of several outgoing branches does not establish the
        # subject's intended branch; wait until both vehicles occupy a uniquely
        # resolved shared continuation.
        if len(outgoing_ids) != 1:
            return None, "ambiguous_forward_branch"

        next_id = outgoing_ids[0]
        if next_id in visited:
            return None, "cyclic_map_path"
        next_node = _resolve_map_object(snapshot["frame"], next_id, object_cache)
        if next_node is None:
            return None, "unresolved_forward_map_object"

        path.append(next_node)
        visited.add(next_id)
        current = next_node

        if current.object_id == target.object_id:
            if current.kind != target.kind:
                return None, "target_map_kind_conflict"
            # Preserve the already matched target object instance and metrics.
            path[-1] = target
            relation = "successor" if len(path) == 2 else "forward_path"
            return tuple(path), relation

    return None, "target_outside_follow_path_horizon"


def _box_corner_points(record: dict[str, Any]) -> list[tuple[float, float]]:
    polygon = oriented_box_polygon(record)
    if polygon is None:
        return []
    try:
        coordinates = list(polygon.exterior.coords)
    except Exception:
        return []
    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates = coordinates[:-1]
    result = []
    for coordinate in coordinates:
        try:
            result.append((float(coordinate[0]), float(coordinate[1])))
        except Exception:
            continue
    return result


def _projected_footprint_interval(
    record: dict[str, Any], map_object: Any, map_length_m: float
) -> Optional[tuple[float, float, float]]:
    """Project the full oriented footprint to one map baseline.

    Corner projection is preferred.  A heading-aware rectangular extent around
    the projected centre is used only as a deterministic fallback.
    """
    x, y, body_heading = record["xyh"]
    center_progress = _finite_float(_project_progress_on_map_object(map_object, x, y))
    if center_progress is None:
        return None

    projected = []
    for px, py in _box_corner_points(record):
        progress = _finite_float(_project_progress_on_map_object(map_object, px, py))
        if progress is not None:
            projected.append(progress)

    if len(projected) >= 3:
        rear = min(projected)
        front = max(projected)
    else:
        dimensions = record.get("dimensions", {})
        length = _finite_float(dimensions.get("length_m"))
        width = _finite_float(dimensions.get("width_m"))
        map_heading = _finite_float(record.get("map_heading_rad"))
        if length is None or width is None or map_heading is None:
            return None
        delta = wrap_signed(float(body_heading) - map_heading)
        half_extent = 0.5 * (
            abs(math.cos(delta)) * length + abs(math.sin(delta)) * width
        )
        rear = center_progress - half_extent
        front = center_progress + half_extent

    # The baseline projection API naturally saturates around segment ends for
    # boundary-straddling boxes.  Clamping makes the interval deterministic and
    # avoids extrapolating beyond the represented map segment.
    rear = min(max(0.0, float(rear)), float(map_length_m))
    front = min(max(0.0, float(front)), float(map_length_m))
    if front < rear:
        rear, front = front, rear
    return rear, front, min(max(0.0, center_progress), float(map_length_m))


def _path_evidence(
    path: tuple[_PathNode, ...],
    subject: dict[str, Any],
    obj: dict[str, Any],
    relation: str,
) -> Optional[_PathEvidence]:
    subject_interval = _projected_footprint_interval(
        subject, path[0].map_object, path[0].length_m
    )
    object_interval = _projected_footprint_interval(
        obj, path[-1].map_object, path[-1].length_m
    )
    if subject_interval is None or object_interval is None:
        return None

    target_offset = sum(node.length_m for node in path[:-1])
    subject_rear, subject_front, subject_center = subject_interval
    object_rear_local, object_front_local, object_center_local = object_interval
    object_rear = target_offset + object_rear_local
    object_front = target_offset + object_front_local
    object_center = target_offset + object_center_local

    return _PathEvidence(
        nodes=path,
        relation=relation,
        subject_interval_m=(subject_rear, subject_front),
        object_interval_m=(object_rear, object_front),
        subject_center_progress_m=subject_center,
        object_center_progress_m=object_center,
        center_path_distance_m=object_center - subject_center,
        bumper_gap_m=object_rear - subject_front,
    )


def _forward_speed_on_path(record: dict[str, Any]) -> Optional[float]:
    velocity = record.get("velocity")
    map_heading = _finite_float(record.get("map_heading_rad"))
    if velocity is None or map_heading is None:
        return None
    try:
        vx, vy = float(velocity[0]), float(velocity[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(vx) or not math.isfinite(vy):
        return None
    return float(math.cos(map_heading) * vx + math.sin(map_heading) * vy)


def _assess_forward_object(
    snapshot: dict[str, Any],
    subject: dict[str, Any],
    obj: dict[str, Any],
    object_cache: dict[str, Optional[_PathNode]],
) -> tuple[Optional[_LeaderAssessment], Optional[str]]:
    path_nodes, path_relation_or_reason = _unique_forward_path(
        snapshot, subject, obj, object_cache
    )
    if path_nodes is None:
        return None, path_relation_or_reason

    path = _path_evidence(path_nodes, subject, obj, str(path_relation_or_reason))
    if path is None:
        return None, "missing_path_projected_footprint"
    if path.center_path_distance_m <= ARGS.distance_epsilon_m:
        return None, "object_not_forward_on_path"

    geometry = footprint_metrics(subject, obj)
    subject_direction = _record_direction(subject)
    object_direction = _record_direction(obj)
    if not subject_direction.reliable or subject_direction.heading_rad is None:
        omission = "subject_travel_direction_unreliable"
    elif not object_direction.reliable or object_direction.heading_rad is None:
        omission = "object_travel_direction_unreliable"
    elif geometry.get("overlapping") is None:
        omission = "missing_footprint_geometry"
    elif geometry.get("overlapping") is True:
        omission = "overlapping_footprints"
    elif path.bumper_gap_m <= ARGS.distance_epsilon_m:
        omission = "non_positive_path_bumper_gap"
    else:
        omission = None

    subject_forward = _forward_speed_on_path(subject)
    object_forward = _forward_speed_on_path(obj)
    reverse_tolerance = float(getattr(ARGS, "follow_reverse_speed_tolerance_mps", 0.30))
    max_distance = float(getattr(ARGS, "maximum_follow_distance_m", 80.0))
    queue_subject_speed = float(getattr(ARGS, "follow_queue_speed_threshold_mps", 2.0))
    queue_leader_speed = float(getattr(ARGS, "follow_queue_leader_speed_threshold_mps", 4.0))
    max_queue_gap = float(getattr(ARGS, "maximum_queue_follow_gap_m", 12.0))

    headway = None
    moving_mode = False
    queue_mode = False
    if omission is None:
        if subject_forward is None or object_forward is None:
            omission = "missing_path_forward_speed"
        elif subject_forward < -reverse_tolerance or object_forward < -reverse_tolerance:
            omission = "reverse_motion_on_forward_path"
        elif path.bumper_gap_m > max_distance:
            omission = "outside_follow_distance_horizon"
        else:
            if subject_forward > ARGS.minimum_forward_speed_mps:
                headway = path.bumper_gap_m / subject_forward
                moving_mode = bool(0.0 < headway <= ARGS.maximum_follow_headway_s)
            queue_mode = bool(
                subject_forward <= queue_subject_speed
                and object_forward <= queue_leader_speed
                and path.bumper_gap_m <= max_queue_gap
            )
            if not moving_mode and not queue_mode:
                omission = "outside_moving_and_queue_follow_envelopes"

    mode = "moving_headway" if moving_mode else ("queue_or_stop_and_go" if queue_mode else None)
    assessment = _LeaderAssessment(
        subject=subject,
        obj=obj,
        path=path,
        geometry=geometry,
        subject_forward_speed_mps=subject_forward,
        object_forward_speed_mps=object_forward,
        headway_s=headway,
        mode=mode,
        valid_follow_envelope=omission is None,
        omission_reason=omission,
    )
    return assessment, None


def _select_nearest_forward_assessments(
    snapshot: dict[str, Any],
) -> dict[str, tuple[_LeaderAssessment, Optional[float]]]:
    """Select one unique nearest forward motor vehicle per subject.

    Unlike ``_select_nearest_leaders``, this helper does not require the
    moving/queue follow envelope.  It is used by the overtaking detector so a
    stable direct approach can start an overtaking event even when the initial
    time headway is larger than the following threshold.  Geometry, directed
    path ordering, and nearest-leader uniqueness are still enforced.
    """
    entities = [
        record for record in snapshot["entities"].values()
        if _is_follow_vehicle(record)
    ]
    selected: dict[str, tuple[_LeaderAssessment, Optional[float]]] = {}
    object_cache: dict[str, Optional[_PathNode]] = {}
    tie_tolerance = float(getattr(ARGS, "follow_leader_tie_tolerance_m", 0.50))

    for subject in entities:
        candidates: list[_LeaderAssessment] = []
        for obj in entities:
            if str(subject.get("track_token")) == str(obj.get("track_token")):
                continue
            assessment, _ = _assess_forward_object(
                snapshot, subject, obj, object_cache
            )
            if assessment is not None:
                candidates.append(assessment)

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: (
                item.path.center_path_distance_m,
                item.path.bumper_gap_m,
                str(item.obj.get("track_token")),
            )
        )
        leader = candidates[0]
        second_distance = (
            candidates[1].path.center_path_distance_m
            if len(candidates) > 1 else None
        )
        if (
            second_distance is not None
            and second_distance - leader.path.center_path_distance_m
            <= tie_tolerance
        ):
            continue

        selected[str(subject["track_token"])] = (leader, second_distance)

    return selected


def _select_nearest_leaders(
    snapshot: dict[str, Any],
) -> dict[str, tuple[_LeaderAssessment, Optional[float]]]:
    """Select at most one unique valid following leader per subject."""
    return {
        subject_token: value
        for subject_token, value
        in _select_nearest_forward_assessments(snapshot).items()
        if value[0].valid_follow_envelope
    }


def derive_following_predicates(snapshots):
    """Emit only temporally persistent ``np:follows(subject, leader)`` relations."""
    assertions = []
    if not ARGS.derive_semantic_predicates:
        return assertions

    streaks = {}
    previous_snapshot = None
    for snapshot in snapshots:
        frame = snapshot["frame"]
        current_us = int(snapshot["timestamp_us"])
        current_frame_index = int(snapshot.get("frame_index", 0))
        continuous_snapshot = False
        if previous_snapshot is not None:
            continuous_snapshot = observations_are_continuous(
                int(previous_snapshot.get("frame_index", current_frame_index - 1)),
                current_frame_index,
                int(previous_snapshot["timestamp_us"]),
                current_us,
                sample_interval_s=ARGS.sample_interval_s,
                maximum_gap_factor=ARGS.maximum_temporal_gap_factor,
            )

        selected = _select_nearest_leaders(snapshot)
        active_keys = set()
        for _, (assessment, second_distance) in selected.items():
            subject = assessment.subject
            obj = assessment.obj
            pair_key = (
                str(subject["track_token"]),
                str(obj["track_token"]),
                "follows",
            )
            active_keys.add(pair_key)
            streak = update_condition_streak(
                streaks,
                pair_key,
                condition=True,
                current_timestamp_us=current_us,
                current_frame_index=current_frame_index,
                continuous_with_previous=continuous_snapshot,
            )

            queue_streak = None
            if assessment.mode == "queue_or_stop_and_go":
                queue_key = (
                    str(subject["track_token"]),
                    str(obj["track_token"]),
                    "queuesBehind",
                )
                active_keys.add(queue_key)
                queue_streak = update_condition_streak(
                    streaks,
                    queue_key,
                    condition=True,
                    current_timestamp_us=current_us,
                    current_frame_index=current_frame_index,
                    continuous_with_previous=continuous_snapshot,
                )

            if streak.duration_s < ARGS.minimum_follow_duration_s:
                continue

            subject_direction = _record_direction(subject)
            object_direction = _record_direction(obj)
            evidence = {
                "follow_strategy": "hierarchical_unique_path_nearest_leader_v9_5_6",
                "supporting_predicates": [
                    "np:hasPrimaryLane",
                    "np:hasPrimaryLaneConnector",
                    "np:hasEffectiveTravelHeading",
                    "np:hasSubjectForwardSpeed",
                    "np:hasSignedPathDistanceTo",
                    "np:hasFreeSpaceDistanceTo",
                    "np:hasPairObservedDuration",
                ],
                "leader_selection_scope": "all_observed_motor_vehicles",
                "leader_rank": 1,
                "nearest_leader_unique": True,
                "second_candidate_center_path_distance_m": second_distance,
                "leader_tie_tolerance_m": float(
                    getattr(ARGS, "follow_leader_tie_tolerance_m", 0.50)
                ),
                "subject_id": subject["entity_id"],
                "object_id": obj["entity_id"],
                "subject_track_token": str(subject["track_token"]),
                "object_track_token": str(obj["track_token"]),
                "subject_agent_type": _normalized_agent_type(subject),
                "object_agent_type": _normalized_agent_type(obj),
                "subject_primary_map_object_id": subject.get("primary_map_object_id"),
                "object_primary_map_object_id": obj.get("primary_map_object_id"),
                "subject_primary_map_kind": subject.get("primary_map_kind"),
                "object_primary_map_kind": obj.get("primary_map_kind"),
                "path_relation": assessment.path.relation,
                "path_object_ids": [node.object_id for node in assessment.path.nodes],
                "path_object_kinds": [node.kind for node in assessment.path.nodes],
                "path_hops": max(0, len(assessment.path.nodes) - 1),
                "unique_unbranched_forward_path": True,
                "subject_path_interval_m": list(assessment.path.subject_interval_m),
                "object_path_interval_m": list(assessment.path.object_interval_m),
                "subject_path_center_progress_m": assessment.path.subject_center_progress_m,
                "object_path_center_progress_m": assessment.path.object_center_progress_m,
                "path_center_distance_m": assessment.path.center_path_distance_m,
                "path_bumper_gap_m": assessment.path.bumper_gap_m,
                "map_bumper_gap_m": assessment.path.bumper_gap_m,
                "headway_gap_source": "path_projected_oriented_footprints",
                "subject_forward_speed_mps": assessment.subject_forward_speed_mps,
                "object_forward_speed_mps": assessment.object_forward_speed_mps,
                "bumper_headway_s": assessment.headway_s,
                "follow_mode": assessment.mode,
                "maximum_follow_headway_s": ARGS.maximum_follow_headway_s,
                "maximum_follow_distance_m": float(
                    getattr(ARGS, "maximum_follow_distance_m", 80.0)
                ),
                "follow_queue_speed_threshold_mps": float(
                    getattr(ARGS, "follow_queue_speed_threshold_mps", 2.0)
                ),
                "follow_queue_leader_speed_threshold_mps": float(
                    getattr(ARGS, "follow_queue_leader_speed_threshold_mps", 4.0)
                ),
                "maximum_queue_follow_gap_m": float(
                    getattr(ARGS, "maximum_queue_follow_gap_m", 12.0)
                ),
                "subject_travel_heading_rad": subject_direction.heading_rad,
                "object_travel_heading_rad": object_direction.heading_rad,
                "subject_travel_direction_source": subject_direction.source,
                "object_travel_direction_source": object_direction.source,
                "condition_duration_s": streak.duration_s,
                "condition_frame_count": streak.frame_count,
                "minimum_follow_duration_s": ARGS.minimum_follow_duration_s,
                **assessment.geometry,
            }
            assertions.append(
                derived_relation(
                    "np:follows",
                    subject["entity_id"],
                    obj["entity_id"],
                    frame,
                    FOLLOW_RULE_ID,
                    evidence,
                    start_time_us=streak.start_timestamp_us,
                    end_time_us=current_us,
                )
            )

            if (
                queue_streak is not None
                and queue_streak.duration_s >= ARGS.minimum_follow_duration_s
            ):
                queue_evidence = {
                    **evidence,
                    "follow_mode": "queue_or_stop_and_go",
                    "queue_condition_duration_s": queue_streak.duration_s,
                    "queue_condition_frame_count": queue_streak.frame_count,
                    "queue_definition": (
                        "subject_path_speed<=2.0m/s; leader_path_speed<=4.0m/s; "
                        "0<path_bumper_gap<=12m; same path/leader/persistence gates as follows"
                    ),
                }
                assertions.append(
                    derived_relation(
                        "np:queuesBehind",
                        subject["entity_id"],
                        obj["entity_id"],
                        frame,
                        FOLLOW_RULE_ID,
                        queue_evidence,
                        start_time_us=queue_streak.start_timestamp_us,
                        end_time_us=current_us,
                    )
                )

        # A leader change, absent track, invalid envelope, or map uncertainty
        # ends the old streak immediately.  The new leader starts a new streak.
        for key in list(streaks):
            if key not in active_keys:
                streaks.pop(key, None)
        previous_snapshot = snapshot

    return assertions

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Complete vehicle-to-target-lane interaction
# ---------------------------------------------------------------------------

LANE_CHANGE_RULE_ID = "R-LANE-CHANGE-INTERACTION-004"


def _lane_change_arg(name: str, default: float) -> float:
    try:
        return float(getattr(ARGS, name, default))
    except (TypeError, ValueError):
        return float(default)


def _lane_change_int_arg(name: str, default: int) -> int:
    try:
        return int(getattr(ARGS, name, default))
    except (TypeError, ValueError):
        return int(default)


def _lane_change_entity_id(kind: Any, value: Any) -> Optional[str]:
    """Return a canonical drivable-primitive entity ID.

    Lane changes are defined over both normal lanes and lane connectors in
    v9.5.26.  Keeping the type in the identity prevents an unrelated lane and
    connector with the same raw nuPlan token from being collapsed together.
    """
    if value is None:
        return None
    kind = str(kind or "").strip().lower()
    text = str(value)
    if text.startswith("lane:") or text.startswith("lane_connector:"):
        return text
    if kind == "lane":
        return f"lane:{text}"
    if kind == "lane_connector":
        return f"lane_connector:{text}"
    return None


def _lane_change_split_entity_id(value: Any) -> tuple[Optional[str], Optional[str]]:
    if value is None:
        return None, None
    text = str(value)
    if text.startswith("lane_connector:"):
        return "lane_connector", text.split(":", 1)[1]
    if text.startswith("lane:"):
        return "lane", text.split(":", 1)[1]
    return None, text


def _lane_change_lane_id(record: dict[str, Any]) -> Optional[str]:
    """Return the selected primary drivable primitive (Lane or LaneConnector).

    The previous implementation intentionally reduced the lane-change decoder
    to normal lanes.  That missed real lateral transfers that occur while a
    vehicle is inside an intersection/merge connector.  v9.5.26 therefore uses
    the actual selected primary map primitive whenever it is a Lane or a
    LaneConnector.
    """
    kind = record.get("primary_map_kind")
    value = record.get("primary_map_object_id")
    entity_id = _lane_change_entity_id(kind, value)
    if entity_id is not None:
        return entity_id
    if record.get("primary_lane_id") is not None:
        return _lane_change_entity_id("lane", record.get("primary_lane_id"))
    if record.get("primary_connector_id") is not None:
        return _lane_change_entity_id("lane_connector", record.get("primary_connector_id"))
    return None


def _lane_change_speed(record: dict[str, Any]) -> float:
    speed = _finite_float(record.get("speed"))
    if speed is not None:
        return max(0.0, speed)
    velocity = record.get("velocity")
    if velocity is None:
        return 0.0
    try:
        return math.hypot(float(velocity[0]), float(velocity[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _lane_change_map_object(record: dict[str, Any], lane_id: str) -> Any:
    """Resolve a Lane/LaneConnector object from its canonical entity ID."""
    lane_id = str(lane_id)
    if (
        _lane_change_lane_id(record) == lane_id
        and record.get("primary_map_object") is not None
    ):
        return record.get("primary_map_object")
    wanted_kind, wanted_raw = _lane_change_split_entity_id(lane_id)
    for candidate in record.get("map_candidates", []) or []:
        kind = str(candidate.get("kind") or "")
        if kind not in {"lane", "lane_connector"}:
            continue
        candidate_entity = _lane_change_entity_id(kind, candidate.get("object_id"))
        if candidate_entity == lane_id:
            return candidate.get("map_object")
        # Backward-compatible raw-ID lookup for old synthetic fixtures.
        candidate_raw = _lane_change_edge_id(candidate.get("object_id"))
        if wanted_kind is None and wanted_raw is not None and str(candidate_raw) == str(wanted_raw):
            return candidate.get("map_object")
    return None


def _lane_change_overlap_ratio(record: dict[str, Any], lane_object: Any) -> float:
    footprint = oriented_box_polygon(record)
    polygon = getattr(lane_object, "polygon", None)
    if footprint is None or polygon is None:
        return 0.0
    try:
        if footprint.is_empty or polygon.is_empty:
            return 0.0
        return float(footprint.intersection(polygon).area) / max(
            float(footprint.area), 1e-9
        )
    except Exception:
        return 0.0


def _lane_change_baseline_line(lane_object: Any) -> Optional[LineString]:
    baseline = getattr(lane_object, "baseline_path", None)
    discrete = getattr(baseline, "discrete_path", None)
    if not discrete:
        return None
    coordinates = []
    for state in discrete:
        try:
            x = float(state.x)
            y = float(state.y)
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            coordinates.append((x, y))
    if len(coordinates) < 2:
        return None
    try:
        return LineString(coordinates)
    except Exception:
        return None


def _lane_change_tangent_heading(line: LineString, progress: float) -> Optional[float]:
    if line is None or float(line.length) <= 0.0:
        return None
    progress = min(max(0.0, float(progress)), float(line.length))
    p0 = line.interpolate(max(0.0, progress - 0.50))
    p1 = line.interpolate(min(float(line.length), progress + 0.50))
    dx = float(p1.x) - float(p0.x)
    dy = float(p1.y) - float(p0.y)
    if math.hypot(dx, dy) <= 1e-6:
        return None
    return math.atan2(dy, dx)


def _lane_change_signed_lateral(
    line: Optional[LineString], record: dict[str, Any]
) -> Optional[float]:
    if line is None:
        return None
    x, y, _ = record["xyh"]
    point = ShapelyPoint(float(x), float(y))
    progress = float(line.project(point))
    base = line.interpolate(progress)
    heading = _lane_change_tangent_heading(line, progress)
    if heading is None:
        return None
    dx = float(x) - float(base.x)
    dy = float(y) - float(base.y)
    return float(-math.sin(heading) * dx + math.cos(heading) * dy)


def _lane_change_baseline_anchor(
    lane_object: Any, record: dict[str, Any]
) -> Optional[tuple[float, float]]:
    line = _lane_change_baseline_line(lane_object)
    if line is None:
        return None
    x, y, _ = record["xyh"]
    point = line.interpolate(float(line.project(ShapelyPoint(float(x), float(y)))))
    return float(point.x), float(point.y)


def _lane_change_edge_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.startswith("lane:") or value.startswith("lane_connector:"):
            return value.split(":", 1)[1]
        return value
    raw = getattr(value, "id", None)
    return None if raw is None else str(raw)


def _lane_change_edge_objects(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        value = value() if callable(value) else value
    except Exception:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None]
    return [value]


def _lane_change_edge_ids(lane: Any, attributes: tuple[str, ...]) -> set[str]:
    result: set[str] = set()
    for attribute in attributes:
        for item in _lane_change_edge_objects(getattr(lane, attribute, None)):
            item_id = _lane_change_edge_id(item)
            if item_id:
                result.add(item_id)
    return result


def _lane_change_outgoing_objects(lane: Any) -> list[Any]:
    result = []
    seen = set()
    for attribute in ("outgoing_edges", "outgoing_edge", "next_edges", "successors"):
        for item in _lane_change_edge_objects(getattr(lane, attribute, None)):
            item_id = _lane_change_edge_id(item)
            if item_id and item_id not in seen:
                seen.add(item_id)
                result.append(item)
    return result


def _lane_change_reachable_forward(
    start_lane: Any, target_lane_id: Optional[str], max_hops: int = 5
) -> bool:
    start_id = _lane_change_edge_id(start_lane)
    target_id = _lane_change_edge_id(target_lane_id)
    if start_id is None or target_id is None:
        return False
    if start_id == target_id:
        return True
    frontier = [start_lane]
    visited = {start_id}
    for _ in range(max(1, int(max_hops))):
        next_frontier = []
        for lane in frontier:
            for successor in _lane_change_outgoing_objects(lane):
                successor_id = _lane_change_edge_id(successor)
                if successor_id is None:
                    continue
                if successor_id == target_id:
                    return True
                if successor_id in visited:
                    continue
                visited.add(successor_id)
                # Some SDK edge references contain only an ID.  They are still
                # sufficient for the direct-successor check above.
                if any(
                    getattr(successor, attribute, None) is not None
                    for attribute in ("outgoing_edges", "outgoing_edge", "next_edges", "successors")
                ):
                    next_frontier.append(successor)
        if not next_frontier:
            break
        frontier = next_frontier
    return False


def _lane_change_match(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    lane_id = _lane_change_lane_id(record)
    lane_object = None if lane_id is None else _lane_change_map_object(record, lane_id)
    candidates = []
    for candidate in record.get("map_candidates", []) or []:
        if candidate.get("kind") != "lane":
            continue
        candidate_id = _lane_change_edge_id(candidate.get("object_id"))
        candidate_object = candidate.get("map_object")
        if candidate_id is None or candidate_object is None:
            continue
        candidates.append({
            "lane_id": candidate_id,
            "lane_object": candidate_object,
            "overlap_ratio": float(candidate.get("overlap_ratio") or 0.0),
            "center_covered": bool(candidate.get("center_covered")),
        })
    if lane_id is None or lane_object is None:
        ranked = sorted(
            candidates,
            key=lambda row: (
                bool(row["center_covered"]),
                float(row["overlap_ratio"]),
                str(row["lane_id"]),
            ),
            reverse=True,
        )
        if ranked:
            first = ranked[0]
            second_overlap = float(ranked[1]["overlap_ratio"]) if len(ranked) > 1 else 0.0
            clear_candidate = (
                bool(first["center_covered"])
                and (
                    len(ranked) == 1
                    or not bool(ranked[1]["center_covered"])
                    or float(first["overlap_ratio"]) - second_overlap >= 0.15
                )
            ) or (
                float(first["overlap_ratio"]) >= 0.55
                and float(first["overlap_ratio"]) - second_overlap >= 0.15
            )
            if clear_candidate:
                lane_id = str(first["lane_id"])
                lane_object = first["lane_object"]
    if lane_id is None or lane_object is None:
        return None

    overlap = None
    center_covered = False
    for candidate in candidates:
        if str(candidate["lane_id"]) == str(lane_id):
            overlap = float(candidate["overlap_ratio"])
            center_covered = bool(candidate["center_covered"])
            break
    if overlap is None:
        overlap = _lane_change_overlap_ratio(record, lane_object)
        try:
            x, y, _ = record["xyh"]
            center_covered = bool(
                getattr(lane_object, "polygon", None).covers(
                    ShapelyPoint(float(x), float(y))
                )
            )
        except Exception:
            center_covered = False
    return {
        "lane_id": str(lane_id),
        "lane_object": lane_object,
        "overlap_ratio": float(overlap),
        "center_covered": bool(center_covered),
        "ambiguous": bool(record.get("map_match_ambiguous", False)),
    }


def _lane_change_same_longitudinal_corridor(
    first: Optional[dict[str, Any]], second: Optional[dict[str, Any]]
) -> bool:
    if first is None or second is None:
        return False
    first_id = str(first.get("lane_id") or "")
    second_id = str(second.get("lane_id") or "")
    if not first_id or not second_id:
        return False
    if first_id == second_id:
        return True
    max_hops = _lane_change_int_arg("lane_change_max_lane_path_hops", 5)
    return bool(
        _lane_change_reachable_forward(first.get("lane_object"), second_id, max_hops)
        or _lane_change_reachable_forward(second.get("lane_object"), first_id, max_hops)
    )


def _lane_change_official_adjacency_side(
    source_record: dict[str, Any],
    source_lane: Any,
    target_lane: Any,
    target_lane_id: str,
) -> Optional[str]:
    target_lane_id = str(target_lane_id)
    _, target_raw_id = _lane_change_split_entity_id(target_lane_id)
    target_lane_id = str(target_raw_id or target_lane_id)
    source_lane_id = _lane_change_edge_id(source_lane)
    left_ids = {
        str(value).split(":", 1)[1] if str(value).startswith("lane:") else str(value)
        for value in source_record.get("left_adjacent_object_ids", set()) or set()
    }
    right_ids = {
        str(value).split(":", 1)[1] if str(value).startswith("lane:") else str(value)
        for value in source_record.get("right_adjacent_object_ids", set()) or set()
    }
    left_ids.update(_lane_change_edge_ids(
        source_lane,
        ("left_adjacent_edge", "adjacent_left", "left_adjacent_ids"),
    ))
    right_ids.update(_lane_change_edge_ids(
        source_lane,
        ("right_adjacent_edge", "adjacent_right", "right_adjacent_ids"),
    ))
    try:
        adjacent = getattr(source_lane, "adjacent_edges", None)
        adjacent = adjacent() if callable(adjacent) else adjacent
        if adjacent and len(adjacent) >= 2:
            left_id = _lane_change_edge_id(adjacent[0])
            right_id = _lane_change_edge_id(adjacent[1])
            if left_id:
                left_ids.add(left_id)
            if right_id:
                right_ids.add(right_id)
    except Exception:
        pass
    if target_lane_id in left_ids and target_lane_id not in right_ids:
        return "left"
    if target_lane_id in right_ids and target_lane_id not in left_ids:
        return "right"

    if target_lane is not None and source_lane_id is not None:
        target_left = _lane_change_edge_ids(
            target_lane,
            ("left_adjacent_edge", "adjacent_left", "left_adjacent_ids"),
        )
        target_right = _lane_change_edge_ids(
            target_lane,
            ("right_adjacent_edge", "adjacent_right", "right_adjacent_ids"),
        )
        if source_lane_id in target_right and source_lane_id not in target_left:
            return "left"
        if source_lane_id in target_left and source_lane_id not in target_right:
            return "right"
    return None


def _lane_change_geometric_adjacency_side(
    source_lane: Any, target_lane: Any
) -> Optional[str]:
    source_polygon = getattr(source_lane, "polygon", None)
    target_polygon = getattr(target_lane, "polygon", None)
    source_line = _lane_change_baseline_line(source_lane)
    target_line = _lane_change_baseline_line(target_lane)
    if source_polygon is None or target_polygon is None or source_line is None or target_line is None:
        return None
    try:
        polygon_distance = float(source_polygon.distance(target_polygon))
    except Exception:
        return None
    if polygon_distance > _lane_change_arg(
        "lane_change_geometric_max_polygon_distance_m", 1.50
    ):
        return None
    source_mid = source_line.interpolate(0.5 * float(source_line.length))
    source_heading = _lane_change_tangent_heading(
        source_line, 0.5 * float(source_line.length)
    )
    target_progress = float(target_line.project(source_mid))
    target_heading = _lane_change_tangent_heading(target_line, target_progress)
    if source_heading is None or target_heading is None:
        return None
    if angular_difference(source_heading, target_heading) > _lane_change_arg(
        "lane_change_geometric_max_direction_difference_rad", 0.30
    ):
        return None
    target_point = target_line.interpolate(target_progress)
    centerline_distance = float(source_mid.distance(target_point))
    if not (
        _lane_change_arg("lane_change_geometric_min_centerline_distance_m", 2.0)
        <= centerline_distance
        <= _lane_change_arg("lane_change_geometric_max_centerline_distance_m", 6.5)
    ):
        return None
    dx = float(target_point.x) - float(source_mid.x)
    dy = float(target_point.y) - float(source_mid.y)
    lateral = -math.sin(source_heading) * dx + math.cos(source_heading) * dy
    if abs(lateral) <= 0.25:
        return None
    return "left" if lateral > 0.0 else "right"


def _lane_change_local_geometric_adjacency_side(
    source_lane: Any, target_lane: Any, reference_record: dict[str, Any]
) -> Optional[str]:
    """Infer left/right from local parallel baselines near the subject.

    Whole-lane midpoint geometry fails when the source and target tokens begin
    at different longitudinal positions.  Evaluating both baselines at the
    subject position is substantially more tolerant while still rejecting
    longitudinal successor transitions.
    """
    source_line = _lane_change_baseline_line(source_lane)
    target_line = _lane_change_baseline_line(target_lane)
    if source_line is None or target_line is None:
        return None
    try:
        x, y, _ = reference_record["xyh"]
        subject_point = ShapelyPoint(float(x), float(y))
        source_progress = float(source_line.project(subject_point))
        target_progress = float(target_line.project(subject_point))
        source_point = source_line.interpolate(source_progress)
        target_point = target_line.interpolate(target_progress)
        source_heading = _lane_change_tangent_heading(source_line, source_progress)
        target_heading = _lane_change_tangent_heading(target_line, target_progress)
    except Exception:
        return None
    if source_heading is None or target_heading is None:
        return None
    if angular_difference(source_heading, target_heading) > _lane_change_arg(
        "lane_change_local_max_direction_difference_rad", 0.60
    ):
        return None
    separation = float(source_point.distance(target_point))
    if not (
        _lane_change_arg("lane_change_local_min_centerline_distance_m", 1.25)
        <= separation
        <= _lane_change_arg("lane_change_local_max_centerline_distance_m", 6.5)
    ):
        return None
    dx = float(target_point.x) - float(source_point.x)
    dy = float(target_point.y) - float(source_point.y)
    lateral = -math.sin(source_heading) * dx + math.cos(source_heading) * dy
    if abs(lateral) <= 0.25:
        return None
    return "left" if lateral > 0.0 else "right"


def _lane_change_adjacency_side(
    source_record: dict[str, Any], source_lane: Any, target_lane: Any, target_lane_id: str
) -> tuple[Optional[str], Optional[str]]:
    side = _lane_change_official_adjacency_side(
        source_record, source_lane, target_lane, target_lane_id
    )
    if side is not None:
        return side, "official_map_adjacency"
    if bool(getattr(ARGS, "lane_change_allow_geometric_adjacency_fallback", True)):
        side = _lane_change_local_geometric_adjacency_side(
            source_lane, target_lane, source_record
        )
        if side is not None:
            return side, "local_geometric_fallback"
        side = _lane_change_geometric_adjacency_side(source_lane, target_lane)
        if side is not None:
            return side, "conservative_geometric_fallback"
    return None, None


def _lane_change_fill_short_same_lane_unknowns(
    values: list[Optional[str]], maximum_run: int
) -> list[Optional[str]]:
    result = list(values)
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        end = index - 1
        left = result[start - 1] if start > 0 else None
        right = result[index] if index < len(result) else None
        if end - start + 1 <= int(maximum_run) and left is not None and left == right:
            for fill_index in range(start, end + 1):
                result[fill_index] = left
    return result


def _lane_change_corridor_labels(
    matches: list[Optional[dict[str, Any]]],
) -> list[Optional[str]]:
    labels: list[Optional[str]] = []
    current_label = None
    last_match = None
    for index, match in enumerate(matches):
        if match is None:
            labels.append(None)
            continue
        if last_match is not None and _lane_change_same_longitudinal_corridor(
            last_match, match
        ):
            labels.append(current_label)
            last_match = match
            continue
        current_label = f"corridor:{index}:{match['lane_id']}"
        labels.append(current_label)
        last_match = match
    return labels


def _lane_change_segments(values: list[Optional[str]]) -> list[dict[str, Any]]:
    if not values:
        return []
    result = []
    start = 0
    current = values[0]
    for index in range(1, len(values)):
        if values[index] == current:
            continue
        result.append({
            "value": current,
            "start_position": start,
            "end_position": index - 1,
            "length": index - start,
        })
        start = index
        current = values[index]
    result.append({
        "value": current,
        "start_position": start,
        "end_position": len(values) - 1,
        "length": len(values) - start,
    })
    return result


def _lane_change_candidate_transitions(segments: list[dict[str, Any]]):
    maximum_unknown = _lane_change_int_arg(
        "lane_change_max_transition_unknown_frames", 4
    )
    index = 0
    while index < len(segments) - 1:
        source = segments[index]
        if source["value"] is None:
            index += 1
            continue
        following = segments[index + 1]
        unknown = None
        if following["value"] is not None:
            target = following
            next_index = index + 1
        elif (
            following["length"] <= maximum_unknown
            and index + 2 < len(segments)
            and segments[index + 2]["value"] is not None
        ):
            unknown = following
            target = segments[index + 2]
            next_index = index + 2
        else:
            index += 1
            continue
        if source["value"] != target["value"]:
            yield source, unknown, target
        index = next_index


def _lane_change_transition_pair(
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    match_by_frame: dict[int, Optional[dict[str, Any]]],
    source_frames: list[int],
    target_frames: list[int],
) -> Optional[dict[str, Any]]:
    search_frames = max(2, _lane_change_int_arg("lane_change_adjacency_search_frames", 6))
    source_candidates = source_frames[-search_frames:]
    target_candidates = target_frames[:search_frames]
    rows = []
    for source_frame in source_candidates:
        source_pair = track.get(source_frame)
        source_match = match_by_frame.get(source_frame)
        if source_pair is None or source_match is None:
            continue
        source_record = source_pair[1]
        for target_frame in target_candidates:
            target_match = match_by_frame.get(target_frame)
            if target_match is None:
                continue
            if _lane_change_same_longitudinal_corridor(source_match, target_match):
                continue
            side, adjacency_source = _lane_change_adjacency_side(
                source_record,
                source_match["lane_object"],
                target_match["lane_object"],
                target_match["lane_id"],
            )
            if side is None:
                continue
            rows.append({
                "source_frame": int(source_frame),
                "target_frame": int(target_frame),
                "source_match": source_match,
                "target_match": target_match,
                "side": side,
                "adjacency_source": adjacency_source,
                "score": (
                    0 if adjacency_source == "official_map_adjacency" else 1,
                    abs(int(target_frame) - int(source_frame)),
                    -float(source_match.get("overlap_ratio") or 0.0)
                    - float(target_match.get("overlap_ratio") or 0.0),
                ),
            })
    if not rows:
        return None
    rows.sort(key=lambda row: row["score"])
    return rows[0]


def _lane_change_boundary_evidence(
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    source_lane: Any,
    target_lane: Any,
    source_end_frame: int,
    target_start_frame: int,
    target_confirmation_frame: int,
) -> dict[str, Any]:
    window = _lane_change_int_arg("lane_change_boundary_evidence_window_frames", 3)
    search_start = max(min(track), int(source_end_frame) - window)
    search_end = min(max(track), int(target_confirmation_frame) + window)
    rows = []
    for frame_index in range(search_start, search_end + 1):
        pair = track.get(frame_index)
        if pair is None:
            continue
        _, record = pair
        source_overlap = _lane_change_overlap_ratio(record, source_lane)
        target_overlap = _lane_change_overlap_ratio(record, target_lane)
        rows.append({
            "frame_index": int(frame_index),
            "source_overlap_ratio": float(source_overlap),
            "target_overlap_ratio": float(target_overlap),
            "dual_overlap_ratio": float(min(source_overlap, target_overlap)),
        })
    if not rows:
        return {
            "has_dual_lane_overlap": False,
            "best_frame": None,
            "best_source_overlap_ratio": 0.0,
            "best_target_overlap_ratio": 0.0,
            "best_dual_overlap_ratio": 0.0,
        }
    best = max(
        rows,
        key=lambda row: (
            row["dual_overlap_ratio"],
            row["source_overlap_ratio"] + row["target_overlap_ratio"],
            -abs(row["frame_index"] - int(target_start_frame)),
        ),
    )
    threshold = _lane_change_arg("lane_change_minimum_dual_lane_overlap_ratio", 0.03)
    return {
        "has_dual_lane_overlap": best["dual_overlap_ratio"] >= threshold,
        "best_frame": int(best["frame_index"]),
        "best_source_overlap_ratio": float(best["source_overlap_ratio"]),
        "best_target_overlap_ratio": float(best["target_overlap_ratio"]),
        "best_dual_overlap_ratio": float(best["dual_overlap_ratio"]),
    }


def _lane_change_first_sustained_frame(
    rows: list[tuple[int, bool]], required_frames: int
) -> Optional[tuple[int, int]]:
    required_frames = max(1, int(required_frames))
    for start in range(len(rows)):
        window = rows[start : start + required_frames]
        if len(window) < required_frames:
            break
        frame_indices = [value[0] for value in window]
        if any(b != a + 1 for a, b in zip(frame_indices[:-1], frame_indices[1:])):
            continue
        if all(bool(value[1]) for value in window):
            return int(frame_indices[0]), int(frame_indices[-1])
    return None


def _lane_change_corridor_line(
    match_by_frame: dict[int, Optional[dict[str, Any]]],
    frame_indices: list[int],
) -> Optional[LineString]:
    objects = []
    seen = set()
    for frame_index in frame_indices:
        match = match_by_frame.get(frame_index)
        if match is None:
            continue
        lane_object = match.get("lane_object")
        lane_id = _lane_change_edge_id(lane_object)
        if lane_object is None or lane_id is None or lane_id in seen:
            continue
        seen.add(lane_id)
        objects.append(lane_object)
    coordinates: list[tuple[float, float]] = []
    for lane_object in objects:
        line = _lane_change_baseline_line(lane_object)
        if line is None:
            continue
        current = [(float(x), float(y)) for x, y in line.coords]
        if coordinates:
            forward_gap = math.hypot(
                current[0][0] - coordinates[-1][0],
                current[0][1] - coordinates[-1][1],
            )
            reverse_gap = math.hypot(
                current[-1][0] - coordinates[-1][0],
                current[-1][1] - coordinates[-1][1],
            )
            if reverse_gap < forward_gap:
                current.reverse()
            if math.hypot(
                current[0][0] - coordinates[-1][0],
                current[0][1] - coordinates[-1][1],
            ) <= 1e-6:
                current = current[1:]
        coordinates.extend(current)
    if len(coordinates) < 2:
        return None
    try:
        return LineString(coordinates)
    except Exception:
        return None


def _lane_change_physical_window(
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    match_by_frame: dict[int, Optional[dict[str, Any]]],
    corridor_label_by_frame: dict[int, Optional[str]],
    source_label: str,
    target_label: str,
    source_corridor_line: Optional[LineString],
    target_entry_lane: Any,
    side: str,
    source_segment_start: int,
    source_end_frame: int,
    boundary_frame: int,
    target_start_frame: int,
    target_segment_end: int,
) -> dict[str, Any]:
    side_sign = 1.0 if side == "left" else -1.0
    boundary_pair = track.get(boundary_frame)
    boundary_time_us = int(boundary_pair[0]["timestamp_us"]) if boundary_pair else None
    search_rows = []
    for frame_index in range(source_segment_start, boundary_frame + 1):
        pair = track.get(frame_index)
        if pair is None:
            continue
        snapshot, record = pair
        if boundary_time_us is not None and (
            boundary_time_us - int(snapshot["timestamp_us"])
        ) / 1e6 > _lane_change_arg("lane_change_onset_search_s", 6.0):
            continue
        lateral = _lane_change_signed_lateral(source_corridor_line, record)
        if lateral is None:
            continue
        search_rows.append({
            "frame": int(frame_index),
            "time_us": int(snapshot["timestamp_us"]),
            "lateral": float(lateral),
            "target_overlap": _lane_change_overlap_ratio(record, target_entry_lane),
        })

    onset_frame = int(source_end_frame)
    onset_source = "source_lane_end_fallback"
    if search_rows:
        baseline_count = min(
            len(search_rows),
            max(2, _lane_change_int_arg("lane_change_source_stable_frames", 2) + 1),
        )
        baseline_values = sorted(row["lateral"] for row in search_rows[:baseline_count])
        middle = len(baseline_values) // 2
        baseline = (
            baseline_values[middle]
            if len(baseline_values) % 2
            else 0.5 * (baseline_values[middle - 1] + baseline_values[middle])
        )
        for row in search_rows:
            row["directed_displacement"] = side_sign * (row["lateral"] - baseline)
        final_displacement = max(
            row["directed_displacement"] for row in search_rows
        )
        displacement_threshold = _lane_change_arg(
            "lane_change_lateral_onset_threshold_m", 0.15
        )
        velocity_threshold = _lane_change_arg(
            "lane_change_lateral_velocity_onset_threshold_mps", 0.15
        )
        future_gain_threshold = _lane_change_arg(
            "lane_change_onset_future_gain_m", 0.40
        )
        onset_stable_frames = max(
            1, _lane_change_int_arg("lane_change_onset_stable_frames", 2)
        )
        for index, row in enumerate(search_rows):
            if final_displacement - row["directed_displacement"] < future_gain_threshold:
                continue
            velocity = 0.0
            if index > 0:
                previous = search_rows[index - 1]
                dt = (row["time_us"] - previous["time_us"]) / 1e6
                if dt > 0.0:
                    velocity = (
                        row["directed_displacement"]
                        - previous["directed_displacement"]
                    ) / dt
            condition = (
                row["directed_displacement"] >= displacement_threshold
                or velocity >= velocity_threshold
                or row["target_overlap"] > 0.01
            )
            if not condition:
                continue
            future = search_rows[index : index + onset_stable_frames]
            if len(future) < onset_stable_frames:
                continue
            if any(
                next_row["directed_displacement"]
                < row["directed_displacement"] - 0.10
                for next_row in future
            ):
                continue
            onset_frame = int(row["frame"])
            onset_source = (
                "early_lateral_motion"
                if velocity >= velocity_threshold
                else "sustained_lateral_displacement"
                if row["directed_displacement"] >= displacement_threshold
                else "first_target_lane_overlap"
            )
            break
        else:
            positive_rows = [
                row for row in search_rows
                if row["target_overlap"] > 0.01
            ]
            if positive_rows:
                onset_frame = int(positive_rows[0]["frame"])
                onset_source = "first_target_lane_overlap"

    completion_rows = []
    target_required = _lane_change_arg(
        "lane_change_target_complete_overlap_ratio", 0.60
    )
    for frame_index in range(max(boundary_frame, target_start_frame), target_segment_end + 1):
        pair = track.get(frame_index)
        match = match_by_frame.get(frame_index)
        if pair is None or match is None:
            continue
        _, record = pair
        if corridor_label_by_frame.get(frame_index) != target_label:
            continue
        target_overlap = _lane_change_overlap_ratio(record, match["lane_object"])
        completion_rows.append((
            int(frame_index),
            bool(
                target_overlap >= target_required
                and (
                    bool(match.get("center_covered"))
                    or target_overlap >= min(0.85, target_required + 0.15)
                )
            ),
        ))
    completion_run = _lane_change_first_sustained_frame(
        completion_rows,
        _lane_change_int_arg("lane_change_completion_stable_frames", 2),
    )
    if completion_run is not None:
        completion_frame = int(completion_run[0])
        completion_confirmation_frame = int(completion_run[1])
        completion_source = "stable_target_occupancy"
    else:
        completion_frame = None
        completion_confirmation_frame = None
        if bool(getattr(
            ARGS,
            "lane_change_allow_single_frame_high_confidence_completion",
            True,
        )):
            high_threshold = _lane_change_arg(
                "lane_change_single_frame_target_overlap_ratio", 0.80
            )
            for frame_index in range(target_segment_end, target_start_frame - 1, -1):
                pair = track.get(frame_index)
                match = match_by_frame.get(frame_index)
                if pair is None or match is None:
                    continue
                if corridor_label_by_frame.get(frame_index) != target_label:
                    continue
                overlap = _lane_change_overlap_ratio(pair[1], match["lane_object"])
                if overlap >= high_threshold and bool(match.get("center_covered")):
                    completion_frame = int(frame_index)
                    completion_confirmation_frame = int(frame_index)
                    completion_source = "single_frame_high_confidence_target_occupancy"
                    break
        if completion_frame is None:
            completion_frame = int(target_start_frame)
            completion_confirmation_frame = int(target_start_frame)
            completion_source = "target_entry_fallback"

    completion_frame = max(int(completion_frame), int(onset_frame))
    return {
        "maneuver_onset_frame": int(onset_frame),
        "completion_frame": int(completion_frame),
        "completion_confirmation_frame": int(completion_confirmation_frame),
        "onset_source": onset_source,
        "completion_source": completion_source,
    }


def _lane_change_phase(frame_index: int, boundary_frame: int) -> tuple[int, str, str]:
    if int(frame_index) < int(boundary_frame):
        return 1, "departing_source_lane", "Departing source lane"
    if int(frame_index) == int(boundary_frame):
        return 2, "crossing_lane_boundary", "Crossing lane boundary"
    return 3, "entering_target_lane", "Entering target lane"



def _lane_change_all_matches(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every usable Lane/LaneConnector match for temporal decoding.

    A drivable path in nuPlan may be represented as Lane -> LaneConnector ->
    Lane without any physical lane change.  Conversely, a vehicle can move
    laterally from one LaneConnector to a parallel LaneConnector.  We therefore
    retain both map-object kinds and let topology distinguish longitudinal
    continuation from lateral transfer.
    """
    rows: dict[str, dict[str, Any]] = {}
    primary_entity_id = _lane_change_lane_id(record)

    for candidate in record.get("map_candidates", []) or []:
        kind = str(candidate.get("kind") or "")
        if kind not in {"lane", "lane_connector"}:
            continue
        entity_id = _lane_change_entity_id(kind, candidate.get("object_id"))
        map_object = candidate.get("map_object")
        if entity_id is None or map_object is None:
            continue
        overlap = _finite_float(candidate.get("overlap_ratio"))
        if overlap is None:
            overlap = _lane_change_overlap_ratio(record, map_object)
        _, raw_id = _lane_change_split_entity_id(entity_id)
        rows[str(entity_id)] = {
            "lane_id": str(entity_id),
            "map_entity_id": str(entity_id),
            "map_object_id": str(raw_id),
            "map_kind": kind,
            "lane_object": map_object,
            "overlap_ratio": float(max(0.0, overlap)),
            "center_covered": bool(candidate.get("center_covered")),
            "primary": str(entity_id) == str(primary_entity_id),
            "ambiguous": bool(record.get("map_match_ambiguous", False)),
        }

    if primary_entity_id is not None:
        map_object = _lane_change_map_object(record, primary_entity_id)
        if map_object is not None:
            row = rows.get(str(primary_entity_id))
            kind, raw_id = _lane_change_split_entity_id(primary_entity_id)
            if row is None:
                overlap = _lane_change_overlap_ratio(record, map_object)
                try:
                    x, y, _ = record["xyh"]
                    center_covered = bool(
                        getattr(map_object, "polygon", None).covers(
                            ShapelyPoint(float(x), float(y))
                        )
                    )
                except Exception:
                    center_covered = False
                row = {
                    "lane_id": str(primary_entity_id),
                    "map_entity_id": str(primary_entity_id),
                    "map_object_id": str(raw_id),
                    "map_kind": str(kind),
                    "lane_object": map_object,
                    "overlap_ratio": float(max(0.0, overlap)),
                    "center_covered": center_covered,
                    "primary": True,
                    "ambiguous": bool(record.get("map_match_ambiguous", False)),
                }
                rows[str(primary_entity_id)] = row
            else:
                row["primary"] = True

    center_bonus = _lane_change_arg("lane_change_candidate_center_bonus", 0.35)
    primary_bonus = _lane_change_arg("lane_change_candidate_primary_bonus", 0.10)
    ambiguity_penalty = _lane_change_arg("lane_change_candidate_ambiguity_penalty", 0.05)
    for row in rows.values():
        row["occupancy_score"] = float(
            row["overlap_ratio"]
            + (center_bonus if row["center_covered"] else 0.0)
            + (primary_bonus if row["primary"] else 0.0)
            - (ambiguity_penalty if row["ambiguous"] else 0.0)
        )
    return sorted(
        rows.values(),
        key=lambda row: (
            float(row["occupancy_score"]),
            float(row["overlap_ratio"]),
            bool(row["center_covered"]),
            str(row["lane_id"]),
        ),
        reverse=True,
    )

def _lane_change_corridor_groups(
    matches_by_frame: dict[int, list[dict[str, Any]]],
) -> dict[str, str]:
    """Collapse only *unambiguous* longitudinal primitive chains.

    This is deliberately stricter than v9.5.24.  At an intersection a lane
    may have several outgoing LaneConnectors; globally unioning all of those
    alternatives destroys the distinction between parallel connector paths and
    makes connector-to-connector lane changes impossible to see.  We therefore
    collapse a source into its successor only when the source has exactly one
    outgoing map edge among the observed primitives.  Branched alternatives
    remain distinct decoder states and are classified locally as either
    longitudinal continuation or lateral transfer.
    """
    primitive_objects: dict[str, Any] = {}
    first_seen: dict[str, int] = {}
    raw_to_entities: dict[str, list[str]] = defaultdict(list)
    sequence_number = 0
    for frame_index in sorted(matches_by_frame):
        for match in matches_by_frame[frame_index]:
            entity_id = str(match["lane_id"])
            primitive_objects.setdefault(entity_id, match["lane_object"])
            if entity_id not in first_seen:
                first_seen[entity_id] = sequence_number
                sequence_number += 1
            raw_id = str(match.get("map_object_id") or _lane_change_edge_id(match["lane_object"]) or "")
            if raw_id and entity_id not in raw_to_entities[raw_id]:
                raw_to_entities[raw_id].append(entity_id)

    parent = {entity_id: entity_id for entity_id in primitive_objects}

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_seen[first_root] <= first_seen[second_root]:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    for entity_id, primitive in primitive_objects.items():
        outgoing_raw = []
        for successor in _lane_change_outgoing_objects(primitive):
            successor_id = _lane_change_edge_id(successor)
            if successor_id and successor_id not in outgoing_raw:
                outgoing_raw.append(successor_id)
        if len(outgoing_raw) != 1:
            continue
        candidate_entities = raw_to_entities.get(str(outgoing_raw[0]), [])
        if len(candidate_entities) == 1:
            union(entity_id, candidate_entities[0])

    groups: dict[str, list[str]] = defaultdict(list)
    for entity_id in primitive_objects:
        groups[find(entity_id)].append(entity_id)

    primitive_to_corridor: dict[str, str] = {}
    for members in groups.values():
        representative = min(members, key=lambda value: first_seen[value])
        label = f"corridor:{representative}"
        for entity_id in members:
            primitive_to_corridor[entity_id] = label
    return primitive_to_corridor

def _lane_change_frame_corridor_matches(
    matches_by_frame: dict[int, list[dict[str, Any]]],
    lane_to_corridor: dict[str, str],
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for frame_index, matches in matches_by_frame.items():
        grouped: dict[str, dict[str, Any]] = {}
        for match in matches:
            corridor = lane_to_corridor.get(str(match["lane_id"]))
            if corridor is None:
                continue
            candidate = dict(match)
            candidate["corridor_id"] = corridor
            previous = grouped.get(corridor)
            if previous is None or (
                float(candidate["occupancy_score"]),
                float(candidate["overlap_ratio"]),
                bool(candidate["center_covered"]),
            ) > (
                float(previous["occupancy_score"]),
                float(previous["overlap_ratio"]),
                bool(previous["center_covered"]),
            ):
                grouped[corridor] = candidate
        result[frame_index] = grouped
    return result


def _lane_change_strong_corridor_matches(
    corridor_matches: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    minimum_overlap = _lane_change_arg(
        "lane_change_minimum_stable_lane_overlap_ratio", 0.45
    )
    result = {}
    for corridor, match in corridor_matches.items():
        overlap = float(match.get("overlap_ratio") or 0.0)
        if bool(match.get("center_covered")) or overlap >= minimum_overlap:
            result[corridor] = match
    return result


def _lane_change_find_stable_corridor(
    continuous_frames: list[int],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
    search_position: int,
    excluded_corridor: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Find the earliest temporally persistent lane-corridor occupancy.

    A small look-ahead window tolerates one ambiguous/flickering frame.  This
    is important at both the outward and return boundaries of an overtake.
    """
    required = max(
        1,
        _lane_change_int_arg("lane_change_temporal_required_frames", 2),
        _lane_change_int_arg("lane_change_source_stable_frames", 2),
        _lane_change_int_arg("lane_change_target_stable_frames", 2),
    )
    configured_window = _lane_change_int_arg(
        "lane_change_temporal_window_frames", required + 1
    )
    window_size = max(required, configured_window)

    for start_position in range(max(0, int(search_position)), len(continuous_frames)):
        end_position = min(len(continuous_frames), start_position + window_size)
        if end_position - start_position < required:
            break
        aggregate: dict[str, dict[str, Any]] = {}
        for position in range(start_position, end_position):
            frame_index = continuous_frames[position]
            strong = _lane_change_strong_corridor_matches(
                frame_corridor_matches.get(frame_index, {})
            )
            for corridor, match in strong.items():
                if excluded_corridor is not None and corridor == excluded_corridor:
                    continue
                item = aggregate.setdefault(corridor, {
                    "positions": [],
                    "frames": [],
                    "score": 0.0,
                    "center_count": 0,
                    "primary_count": 0,
                })
                item["positions"].append(position)
                item["frames"].append(frame_index)
                item["score"] += float(match.get("occupancy_score") or 0.0)
                item["center_count"] += int(bool(match.get("center_covered")))
                item["primary_count"] += int(bool(match.get("primary")))

        qualified = []
        for corridor, item in aggregate.items():
            if len(item["positions"]) < required:
                continue
            # The evidence must persist to the end of the local window rather
            # than being an old lane that disappeared at its beginning.
            if item["positions"][-1] < end_position - 2:
                continue
            required_positions = item["positions"][:required]
            confirmation_position = required_positions[-1]
            qualified.append((
                confirmation_position,
                -len(item["positions"]),
                -item["center_count"],
                -item["score"],
                -item["primary_count"],
                corridor,
                item,
            ))
        if not qualified:
            continue
        qualified.sort()
        _, _, _, _, _, corridor, item = qualified[0]
        evidence_positions = item["positions"][:required]
        return {
            "corridor_id": corridor,
            "evidence_start_position": int(evidence_positions[0]),
            "confirmation_position": int(evidence_positions[-1]),
            "evidence_positions": tuple(int(value) for value in evidence_positions),
            "window_start_position": int(start_position),
            "window_end_position": int(end_position - 1),
        }
    return None


def _lane_change_decode_episodes(
    continuous_frames: list[int],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Decode every stable lane occupancy episode for one vehicle-like track."""
    initial = _lane_change_find_stable_corridor(
        continuous_frames, frame_corridor_matches, 0, None
    )
    if initial is None:
        return []
    episodes = [initial]
    current = str(initial["corridor_id"])
    search_position = int(initial["confirmation_position"]) + 1

    while search_position < len(continuous_frames):
        target = _lane_change_find_stable_corridor(
            continuous_frames,
            frame_corridor_matches,
            search_position,
            current,
        )
        if target is None:
            break
        # Ignore candidates that are not later than the current confirmation.
        if int(target["confirmation_position"]) <= int(episodes[-1]["confirmation_position"]):
            search_position += 1
            continue
        episodes.append(target)
        current = str(target["corridor_id"])
        search_position = int(target["confirmation_position"]) + 1

    for index, episode in enumerate(episodes):
        next_start = (
            int(episodes[index + 1]["evidence_start_position"])
            if index + 1 < len(episodes)
            else len(continuous_frames)
        )
        episode["episode_end_position"] = max(
            int(episode["confirmation_position"]), next_start - 1
        )
    return episodes



def _lane_change_filter_adjacent_episodes(
    episodes: list[dict[str, Any]],
    continuous_frames: list[int],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Reject map-matching state jumps that are not lateral adjacency changes.

    Rejected states do not become the new source state.  Therefore a brief
    non-adjacent map glitch cannot hide a later genuine ``A -> B`` transition.
    """
    if not episodes:
        return []
    accepted = [dict(episodes[0])]
    search_frames = max(
        2, _lane_change_int_arg("lane_change_adjacency_search_frames", 6)
    )
    for candidate in episodes[1:]:
        source = accepted[-1]
        source_corridor = str(source["corridor_id"])
        target_corridor = str(candidate["corridor_id"])
        target_start = int(candidate["evidence_start_position"])
        source_start = int(source["evidence_start_position"])
        source_positions = [
            position
            for position in range(source_start, max(source_start, target_start))
            if source_corridor
            in _lane_change_strong_corridor_matches(
                frame_corridor_matches.get(continuous_frames[position], {})
            )
        ][-search_frames:]
        target_positions = [
            position
            for position in candidate.get("evidence_positions", ())
            if target_corridor
            in _lane_change_strong_corridor_matches(
                frame_corridor_matches.get(continuous_frames[position], {})
            )
        ][:search_frames]
        if not source_positions or not target_positions:
            continue
        source_frames = [continuous_frames[position] for position in source_positions]
        target_frames = [continuous_frames[position] for position in target_positions]
        local_matches: dict[int, Optional[dict[str, Any]]] = {}
        for frame_index in source_frames:
            local_matches[frame_index] = frame_corridor_matches[frame_index][source_corridor]
        for frame_index in target_frames:
            local_matches[frame_index] = frame_corridor_matches[frame_index][target_corridor]
        if _lane_change_transition_pair(
            track, local_matches, source_frames, target_frames
        ) is None:
            continue
        accepted.append(dict(candidate))

    for index, episode in enumerate(accepted):
        next_start = (
            int(accepted[index + 1]["evidence_start_position"])
            if index + 1 < len(accepted)
            else len(continuous_frames)
        )
        episode["episode_end_position"] = max(
            int(episode["confirmation_position"]), next_start - 1
        )
    return accepted

def _lane_change_episode_frames(
    episode: dict[str, Any],
    continuous_frames: list[int],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
    *,
    start_position: Optional[int] = None,
    end_position: Optional[int] = None,
    strong_only: bool = False,
) -> list[int]:
    corridor = str(episode["corridor_id"])
    start = int(
        episode["evidence_start_position"] if start_position is None else start_position
    )
    end = int(
        episode["episode_end_position"] if end_position is None else end_position
    )
    result = []
    for position in range(max(0, start), min(len(continuous_frames) - 1, end) + 1):
        frame_index = continuous_frames[position]
        match = frame_corridor_matches.get(frame_index, {}).get(corridor)
        if match is None:
            continue
        if strong_only and corridor not in _lane_change_strong_corridor_matches(
            {corridor: match}
        ):
            continue
        result.append(frame_index)
    return result



def _lane_change_primary_match(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return the explicitly selected primary normal lane for one frame.

    Lane-change confirmation in v9.5.26 is intentionally based on the
    temporally persistent ``primary_lane_id`` transition.  Candidate-lane
    overlaps remain diagnostic evidence only and never replace the selected
    primary lane.
    """
    lane_id = _lane_change_lane_id(record)
    if lane_id is None:
        return None
    lane_object = _lane_change_map_object(record, lane_id)
    if lane_object is None:
        return None

    overlap = None
    center_covered = False
    for candidate in record.get("map_candidates", []) or []:
        if candidate.get("kind") != "lane":
            continue
        candidate_id = _lane_change_edge_id(candidate.get("object_id"))
        if candidate_id is None or str(candidate_id) != str(lane_id):
            continue
        overlap = _finite_float(candidate.get("overlap_ratio"))
        center_covered = bool(candidate.get("center_covered"))
        break

    if overlap is None:
        overlap = _lane_change_overlap_ratio(record, lane_object)
        try:
            x, y, _ = record["xyh"]
            polygon = getattr(lane_object, "polygon", None)
            center_covered = bool(
                polygon is not None
                and polygon.covers(ShapelyPoint(float(x), float(y)))
            )
        except Exception:
            center_covered = False

    return {
        "lane_id": str(lane_id),
        "lane_object": lane_object,
        "overlap_ratio": float(max(0.0, overlap or 0.0)),
        "center_covered": bool(center_covered),
        "ambiguous": bool(record.get("map_match_ambiguous", False)),
        "primary": True,
        "occupancy_score": 1.0,
    }


def _lane_change_consecutive(values: list[int]) -> bool:
    return all(second == first + 1 for first, second in zip(values[:-1], values[1:]))


def _lane_change_primary_onset(
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    source_lane: Any,
    target_lane: Any,
    side: str,
    search_start_frame: int,
    source_end_frame: int,
    primary_switch_frame: int,
) -> tuple[int, str]:
    """Backdate a confirmed primary-lane switch to lateral-motion onset.

    The switch itself confirms the destination.  This helper only estimates
    when the physical maneuver began.  If the sampled trajectory contains no
    reliable early lateral evidence, the last source-primary frame is used so
    the interaction is never activated later than the primary-lane switch.
    """
    source_line = _lane_change_baseline_line(source_lane)
    if source_line is None:
        return int(source_end_frame), "last_source_primary_frame_fallback"

    switch_pair = track.get(primary_switch_frame)
    switch_time_us = (
        int(switch_pair[0]["timestamp_us"])
        if switch_pair is not None
        else None
    )
    rows: list[dict[str, float | int]] = []
    for frame_index in range(int(search_start_frame), int(primary_switch_frame) + 1):
        pair = track.get(frame_index)
        if pair is None:
            continue
        snapshot, record = pair
        if switch_time_us is not None:
            age_s = (switch_time_us - int(snapshot["timestamp_us"])) / 1e6
            if age_s > _lane_change_arg("lane_change_onset_search_s", 6.0):
                continue
        lateral = _lane_change_signed_lateral(source_line, record)
        if lateral is None:
            continue
        rows.append({
            "frame": int(frame_index),
            "time_us": int(snapshot["timestamp_us"]),
            "lateral": float(lateral),
            "target_overlap": float(_lane_change_overlap_ratio(record, target_lane)),
        })

    if len(rows) < 2:
        return int(source_end_frame), "last_source_primary_frame_fallback"

    baseline_count = min(
        len(rows),
        max(2, _lane_change_int_arg("lane_change_source_stable_frames", 2) + 1),
    )
    baseline_values = sorted(float(row["lateral"]) for row in rows[:baseline_count])
    middle = len(baseline_values) // 2
    baseline = (
        baseline_values[middle]
        if len(baseline_values) % 2
        else 0.5 * (baseline_values[middle - 1] + baseline_values[middle])
    )
    side_sign = 1.0 if side == "left" else -1.0
    for row in rows:
        row["directed_displacement"] = side_sign * (
            float(row["lateral"]) - float(baseline)
        )

    final_displacement = max(float(row["directed_displacement"]) for row in rows)
    displacement_threshold = _lane_change_arg(
        "lane_change_lateral_onset_threshold_m", 0.15
    )
    velocity_threshold = _lane_change_arg(
        "lane_change_lateral_velocity_onset_threshold_mps", 0.15
    )
    future_gain_threshold = _lane_change_arg(
        "lane_change_onset_future_gain_m", 0.40
    )
    onset_frames = max(1, _lane_change_int_arg("lane_change_onset_stable_frames", 2))

    for index, row in enumerate(rows):
        directed = float(row["directed_displacement"])
        if final_displacement - directed < future_gain_threshold:
            continue
        velocity = 0.0
        if index > 0:
            previous = rows[index - 1]
            dt = (int(row["time_us"]) - int(previous["time_us"])) / 1e6
            if dt > 0.0:
                velocity = (
                    directed - float(previous["directed_displacement"])
                ) / dt
        trigger = (
            directed >= displacement_threshold
            or velocity >= velocity_threshold
            or float(row["target_overlap"]) > 0.01
        )
        if not trigger:
            continue
        future = rows[index : index + onset_frames]
        if len(future) < onset_frames:
            continue
        if any(
            float(item["directed_displacement"]) < directed - 0.10
            for item in future
        ):
            continue
        source = (
            "early_lateral_velocity"
            if velocity >= velocity_threshold
            else "sustained_lateral_displacement"
            if directed >= displacement_threshold
            else "first_target_lane_overlap"
        )
        return int(row["frame"]), source

    positive_overlap = [row for row in rows if float(row["target_overlap"]) > 0.01]
    if positive_overlap:
        return int(positive_overlap[0]["frame"]), "first_target_lane_overlap"
    return int(source_end_frame), "last_source_primary_frame_fallback"


def _lane_change_corridor_lane_objects(
    matches_by_frame: dict[int, list[dict[str, Any]]],
    lane_to_corridor: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Return every concrete lane object grouped by decoded corridor."""
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for matches in matches_by_frame.values():
        for match in matches:
            lane_id = str(match["lane_id"])
            corridor = lane_to_corridor.get(lane_id)
            if corridor is not None:
                grouped[corridor].setdefault(lane_id, match["lane_object"])
    return grouped


def _lane_change_best_match_near(
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
    corridor: str,
    frame_index: int,
    *,
    direction: int,
    maximum_steps: int = 8,
) -> Optional[tuple[int, dict[str, Any]]]:
    for step in range(maximum_steps + 1):
        candidate_frame = int(frame_index + direction * step)
        match = frame_corridor_matches.get(candidate_frame, {}).get(corridor)
        if match is not None:
            return candidate_frame, match
    return None


def _lane_change_local_transition_side(
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
    source_corridor: str,
    target_corridor: str,
    previous_frame: int,
    current_frame: int,
) -> Optional[tuple[str, str, int, dict[str, Any], int, dict[str, Any]]]:
    """Resolve one local adjacent-corridor transition and its left/right side."""
    source_row = _lane_change_best_match_near(
        frame_corridor_matches, source_corridor, previous_frame,
        direction=-1, maximum_steps=10,
    )
    target_row = _lane_change_best_match_near(
        frame_corridor_matches, target_corridor, current_frame,
        direction=1, maximum_steps=10,
    )
    if source_row is None or target_row is None:
        return None
    source_frame, source_match = source_row
    target_frame, target_match = target_row
    record_pair = track.get(source_frame) or track.get(previous_frame) or track.get(current_frame)
    if record_pair is None:
        return None

    # A Lane -> LaneConnector -> Lane progression along directed topology is
    # map segmentation / intersection traversal, not a lane change.  Reject a
    # direct forward successor before attempting geometric lateral adjacency.
    _, target_raw_id = _lane_change_split_entity_id(target_match.get("lane_id"))
    if target_raw_id is not None and _lane_change_reachable_forward(
        source_match["lane_object"], target_raw_id, 1
    ):
        return None

    side, source = _lane_change_adjacency_side(
        record_pair[1],
        source_match["lane_object"],
        target_match["lane_object"],
        str(target_match["lane_id"]),
    )
    if side is None:
        return None
    return (
        str(side), str(source), int(source_frame), source_match,
        int(target_frame), target_match,
    )


def _lane_change_observation_cost(
    record: dict[str, Any],
    match: Optional[dict[str, Any]],
    *,
    any_match: bool,
) -> float:
    """Negative log-like cost for assigning one frame to one lane corridor."""
    if match is None:
        return 5.0 if any_match else 2.7
    overlap = max(0.0, min(1.0, float(match.get("overlap_ratio") or 0.0)))
    score = max(0.0, float(match.get("occupancy_score") or 0.0))
    cost = 3.4
    cost -= min(1.5, score) * 1.25
    cost -= overlap * 1.20
    if bool(match.get("center_covered")):
        cost -= 1.15
    if bool(match.get("primary")):
        cost -= 1.00
    if bool(match.get("ambiguous")):
        cost += 0.20
    return max(0.0, float(cost))


def _lane_change_transition_cost(
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
    previous_state: str,
    current_state: str,
    previous_frame: int,
    current_frame: int,
) -> float:
    unknown = "__UNKNOWN__"
    if previous_state == current_state:
        return 0.05
    if previous_state == unknown or current_state == unknown:
        return 1.20

    transition = _lane_change_local_transition_side(
        track,
        frame_corridor_matches,
        previous_state,
        current_state,
        previous_frame,
        current_frame,
    )
    if transition is None:
        return 50.0
    side, _, source_frame, source_match, target_frame, target_match = transition
    cost = 2.75
    previous_record = track.get(previous_frame, track.get(source_frame))[1]
    current_record = track.get(current_frame, track.get(target_frame))[1]

    source_overlap_previous = _lane_change_overlap_ratio(
        previous_record, source_match["lane_object"]
    )
    source_overlap_current = _lane_change_overlap_ratio(
        current_record, source_match["lane_object"]
    )
    target_overlap_previous = _lane_change_overlap_ratio(
        previous_record, target_match["lane_object"]
    )
    target_overlap_current = _lane_change_overlap_ratio(
        current_record, target_match["lane_object"]
    )
    transfer_gain = (
        (target_overlap_current - target_overlap_previous)
        + (source_overlap_previous - source_overlap_current)
    )
    cost -= max(-0.5, min(1.0, transfer_gain)) * 1.25

    source_line = _lane_change_baseline_line(source_match["lane_object"])
    previous_lateral = _lane_change_signed_lateral(source_line, previous_record)
    current_lateral = _lane_change_signed_lateral(source_line, current_record)
    if previous_lateral is not None and current_lateral is not None:
        directed = (current_lateral - previous_lateral) * (1.0 if side == "left" else -1.0)
        if directed > 0.15:
            cost -= min(1.25, directed * 0.8)
        elif directed < -0.15:
            cost += min(2.0, abs(directed))

    target_frame_match = frame_corridor_matches.get(current_frame, {}).get(current_state)
    if target_frame_match is not None:
        if bool(target_frame_match.get("primary")):
            cost -= 1.00
        if bool(target_frame_match.get("center_covered")):
            cost -= 0.75
    gap = max(1, int(current_frame) - int(previous_frame))
    if gap > 1:
        cost += min(1.0, 0.15 * (gap - 1))
    return max(0.20, float(cost))


def _lane_change_decode_corridor_path(
    track: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    observed_frames: list[int],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
) -> tuple[list[str], float]:
    """Viterbi-style topology-constrained lane-corridor decoding."""
    unknown = "__UNKNOWN__"
    corridors = sorted({
        corridor
        for matches in frame_corridor_matches.values()
        for corridor in matches
    })
    states = corridors + [unknown]
    if not corridors:
        return [unknown] * len(observed_frames), 0.0

    backpointers: list[dict[str, str]] = []
    previous_costs: dict[str, float] = {}
    for position, frame_index in enumerate(observed_frames):
        pair = track[frame_index]
        record = pair[1]
        matches = frame_corridor_matches.get(frame_index, {})
        any_match = bool(matches)
        current_costs: dict[str, float] = {}
        current_back: dict[str, str] = {}
        for state in states:
            if state == unknown:
                emission = 3.4 if any_match else 0.8
            else:
                emission = _lane_change_observation_cost(
                    record, matches.get(state), any_match=any_match
                )
            if position == 0:
                current_costs[state] = float(emission)
                current_back[state] = state
                continue
            previous_frame = observed_frames[position - 1]
            best_previous = None
            best_cost = math.inf
            for previous_state, accumulated in previous_costs.items():
                transition = _lane_change_transition_cost(
                    track,
                    frame_corridor_matches,
                    previous_state,
                    state,
                    previous_frame,
                    frame_index,
                )
                candidate = accumulated + transition + emission
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = previous_state
            current_costs[state] = float(best_cost)
            current_back[state] = str(best_previous)
        previous_costs = current_costs
        backpointers.append(current_back)

    final_state = min(previous_costs, key=previous_costs.get)
    total_cost = float(previous_costs[final_state])
    path = [final_state]
    for position in range(len(observed_frames) - 1, 0, -1):
        final_state = backpointers[position][final_state]
        path.append(final_state)
    path.reverse()
    return path, total_cost


def _lane_change_smooth_decoded_path(
    path: list[str],
    observed_frames: list[int],
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
) -> list[str]:
    """Fill short unknown gaps and reject unsupported one-frame A-B-A flicker."""
    unknown = "__UNKNOWN__"
    result = list(path)
    maximum_unknown = max(
        0, _lane_change_int_arg("lane_change_decoder_max_unknown_gap_frames", 3)
    )
    index = 0
    while index < len(result):
        if result[index] != unknown:
            index += 1
            continue
        end = index
        while end + 1 < len(result) and result[end + 1] == unknown:
            end += 1
        previous = result[index - 1] if index > 0 else None
        following = result[end + 1] if end + 1 < len(result) else None
        if previous is not None and previous == following and end - index + 1 <= maximum_unknown:
            for position in range(index, end + 1):
                result[position] = previous
        index = end + 1

    for position in range(1, len(result) - 1):
        previous, current, following = result[position - 1 : position + 2]
        if current in {unknown, previous} or previous != following:
            continue
        # A single sampled B state between A states cannot establish and
        # leave an adjacent lane physically. Always treat A→B→A in one
        # observation as assignment flicker, even if B is marked primary.
        result[position] = previous
    return result


def _lane_change_path_segments(
    path: list[str], observed_frames: list[int]
) -> list[dict[str, Any]]:
    if not path:
        return []
    segments = []
    start = 0
    for position in range(1, len(path) + 1):
        if position < len(path) and path[position] == path[start]:
            continue
        segments.append({
            "state": path[start],
            "start_position": start,
            "end_position": position - 1,
            "frames": observed_frames[start:position],
            "length": position - start,
        })
        start = position
    return segments


def _lane_change_target_completion_frame(
    target_frames: list[int],
    target_corridor: str,
    frame_corridor_matches: dict[int, dict[str, dict[str, Any]]],
) -> tuple[int, str, bool]:
    required = max(1, _lane_change_int_arg("lane_change_decoder_target_stable_frames", 2))
    for start in range(len(target_frames)):
        window = target_frames[start : start + required]
        if len(window) < required:
            break
        if all(
            (
                (match := frame_corridor_matches.get(frame, {}).get(target_corridor))
                is not None
                and (
                    bool(match.get("center_covered"))
                    or float(match.get("overlap_ratio") or 0.0) >= 0.50
                    or bool(match.get("primary"))
                )
            )
            for frame in window
        ):
            return int(window[-1]), "decoded_stable_target_occupancy", True
    return int(target_frames[0]), "decoded_target_entry_censored", False


def _lane_change_event_score(
    *,
    adjacency_source: str,
    source_frames: list[int],
    target_frames: list[int],
    source_match: dict[str, Any],
    target_match: dict[str, Any],
    source_record: dict[str, Any],
    target_record: dict[str, Any],
    side: str,
    target_stable: bool,
    onset_censored: bool,
    completion_censored: bool,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if adjacency_source == "official_map_adjacency":
        score += 2.0; reasons.append("official_lateral_adjacency")
    else:
        score += 1.25; reasons.append("local_geometric_adjacency")
    if len(source_frames) >= 2:
        score += 1.0; reasons.append("source_temporal_support")
    elif source_frames:
        score += 0.5; reasons.append("short_source_support")
    if target_stable:
        score += 2.0; reasons.append("stable_target_occupancy")
    elif target_frames:
        score += 0.75; reasons.append("censored_target_support")
    if bool(target_match.get("primary")):
        score += 1.5; reasons.append("target_primary_lane")
    if bool(target_match.get("center_covered")):
        score += 1.5; reasons.append("target_center_containment")
    target_overlap = float(target_match.get("overlap_ratio") or 0.0)
    source_overlap = float(source_match.get("overlap_ratio") or 0.0)
    if target_overlap >= 0.50:
        score += 1.0; reasons.append("dominant_target_overlap")
    source_line = _lane_change_baseline_line(source_match["lane_object"])
    source_lateral = _lane_change_signed_lateral(source_line, source_record)
    target_lateral = _lane_change_signed_lateral(source_line, target_record)
    if source_lateral is not None and target_lateral is not None:
        directed = (target_lateral - source_lateral) * (1.0 if side == "left" else -1.0)
        if directed >= 0.75:
            score += 1.5; reasons.append("directed_lateral_transfer")
        elif directed >= 0.25:
            score += 0.75; reasons.append("weak_directed_lateral_transfer")
        elif directed < -0.25:
            score -= 2.0; reasons.append("opposite_lateral_motion")
    if target_overlap > source_overlap:
        score += 0.5; reasons.append("occupancy_transfer")
    if onset_censored:
        score -= 0.25; reasons.append("left_censored")
    if completion_censored:
        score -= 0.50; reasons.append("right_censored")
    return float(score), reasons


def _maneuver_history_requirements() -> tuple[int, float]:
    """Return the shared minimum precondition history for temporal maneuvers."""
    try:
        minimum_frames = max(1, int(getattr(ARGS, "minimum_maneuver_history_frames", 1)))
    except (TypeError, ValueError):
        minimum_frames = 1
    try:
        minimum_seconds = max(0.0, float(getattr(ARGS, "minimum_maneuver_history_s", 0.5)))
    except (TypeError, ValueError):
        minimum_seconds = 0.5
    return minimum_frames, minimum_seconds


def _lane_change_observed_source_history(track, source_frames, onset_frame):
    """Measure observed stable-source history strictly before maneuver onset."""
    prior = [int(frame) for frame in source_frames if int(frame) < int(onset_frame) and int(frame) in track]
    if not prior or int(onset_frame) not in track:
        return {"frame_count": len(prior), "duration_s": 0.0, "frames": prior}
    onset_us = int(track[int(onset_frame)][0]["timestamp_us"])
    first_us = int(track[prior[0]][0]["timestamp_us"])
    return {
        "frame_count": len(prior),
        "duration_s": max(0.0, (onset_us - first_us) / 1e6),
        "frames": prior,
        "first_time_us": first_us,
        "last_time_us": int(track[prior[-1]][0]["timestamp_us"]),
    }


def derive_lane_change_predicates(snapshots):
    """Decode complete vehicle trajectories into topology-consistent lane states.

    v9.5.26 uses an offline two-stage decoder.  First, a Viterbi-style path
    chooses the most plausible lane corridor for every observed frame from
    primary-lane, footprint-overlap, center-containment, geometry, topology,
    and temporal-continuity evidence.  Second, each adjacent decoded corridor
    transition is expanded into CHANGING_LEFT or CHANGING_RIGHT from its
    retrospectively estimated physical onset through stable target occupancy.

    Confirmed, probable, and boundary-censored changes are exported rather
    than silently discarded.  The scan continues after every event, so
    L0 -> L1 -> L0 yields two independent lane changes for ego or any other
    vehicle-like agent.
    """
    if not ARGS.derive_semantic_predicates or not snapshots:
        return []
    snapshots = list(snapshots)
    tracks: dict[str, dict[int, tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for snapshot in snapshots:
        frame_index = int(snapshot["frame_index"])
        for token, record in snapshot["entities"].items():
            if _is_follow_vehicle(record):
                tracks[str(token)][frame_index] = (snapshot, record)

    assertions = []
    minimum_speed = _lane_change_arg("lane_change_minimum_subject_speed_mps", 0.10)
    emit_probable = bool(getattr(ARGS, "lane_change_decoder_emit_probable", True))
    minimum_probable_score = _lane_change_arg(
        "lane_change_decoder_minimum_probable_score", 3.25
    )
    minimum_confirmed_score = _lane_change_arg(
        "lane_change_decoder_minimum_confirmed_score", 6.0
    )

    for subject_token, track in tracks.items():
        observed_frames = sorted(track)
        if len(observed_frames) < 2:
            continue
        matches_by_frame = {
            frame: _lane_change_all_matches(track[frame][1])
            for frame in observed_frames
        }
        lane_to_corridor = _lane_change_corridor_groups(matches_by_frame)
        frame_corridor_matches = _lane_change_frame_corridor_matches(
            matches_by_frame, lane_to_corridor
        )
        decoded_path, decoder_cost = _lane_change_decode_corridor_path(
            track, observed_frames, frame_corridor_matches
        )
        decoded_path = _lane_change_smooth_decoded_path(
            decoded_path, observed_frames, frame_corridor_matches
        )
        segments = _lane_change_path_segments(decoded_path, observed_frames)
        unknown = "__UNKNOWN__"
        emitted_keys: set[tuple[str, str, int, int]] = set()
        event_rows: list[dict[str, Any]] = []

        for segment_index in range(len(segments) - 1):
            source_segment = segments[segment_index]
            next_index = segment_index + 1
            while next_index < len(segments) and segments[next_index]["state"] == unknown:
                next_index += 1
            if next_index >= len(segments):
                break
            target_segment = segments[next_index]
            source_corridor = str(source_segment["state"])
            target_corridor = str(target_segment["state"])
            # A one-observation A→B→A excursion is physically too short to be
            # a completed lane change at nuPlan sampling rates. Treat it as
            # map-assignment flicker even when that single observation is strong.
            following_index = next_index + 1
            while following_index < len(segments) and segments[following_index]["state"] == unknown:
                following_index += 1
            if (
                int(target_segment["length"]) <= 1
                and following_index < len(segments)
                and str(segments[following_index]["state"]) == source_corridor
            ):
                continue
            if source_corridor == unknown or target_corridor == unknown:
                continue
            if source_corridor == target_corridor:
                continue
            source_frames = list(source_segment["frames"])
            target_frames = list(target_segment["frames"])
            if not source_frames or not target_frames:
                continue
            transition = _lane_change_local_transition_side(
                track,
                frame_corridor_matches,
                source_corridor,
                target_corridor,
                source_frames[-1],
                target_frames[0],
            )
            if transition is None:
                continue
            side, adjacency_source, source_frame, source_match, target_entry_frame, target_match = transition
            completion_frame, completion_source, target_stable = _lane_change_target_completion_frame(
                target_frames, target_corridor, frame_corridor_matches
            )
            completion_match = frame_corridor_matches.get(completion_frame, {}).get(target_corridor)
            if completion_match is None:
                completion_match = target_match
                completion_frame = int(target_entry_frame)
                target_stable = False
                completion_source = "nearest_target_match_fallback"
            source_pair = track.get(source_frame)
            completion_pair = track.get(completion_frame)
            if source_pair is None or completion_pair is None:
                continue
            if max(
                _lane_change_speed(source_pair[1]),
                _lane_change_speed(completion_pair[1]),
            ) < minimum_speed:
                continue

            onset_censored = bool(segment_index == 0 and len(source_frames) <= 1)
            completion_censored = not bool(target_stable)
            if onset_censored:
                onset_frame = int(observed_frames[0])
                onset_source = "decoded_left_censored_track_start"
            else:
                onset_frame, onset_source = _lane_change_primary_onset(
                    track,
                    source_match["lane_object"],
                    target_match["lane_object"],
                    side,
                    source_frames[0],
                    source_frame,
                    target_frames[0],
                )
            onset_frame = max(observed_frames[0], min(int(onset_frame), int(target_frames[0])))
            if onset_frame not in track:
                candidates = [frame for frame in observed_frames if frame <= onset_frame]
                onset_frame = candidates[-1] if candidates else observed_frames[0]
                onset_source = "nearest_observed_decoder_frame"
            active_frames = [
                frame for frame in observed_frames
                if onset_frame <= frame <= completion_frame
            ]
            if not active_frames:
                continue
            onset_frame = int(active_frames[0])

            score, score_reasons = _lane_change_event_score(
                adjacency_source=adjacency_source,
                source_frames=source_frames,
                target_frames=target_frames,
                source_match=source_match,
                target_match=completion_match,
                source_record=source_pair[1],
                target_record=completion_pair[1],
                side=side,
                target_stable=target_stable,
                onset_censored=onset_censored,
                completion_censored=completion_censored,
            )
            if score >= minimum_confirmed_score and not completion_censored:
                confidence_status = "confirmed"
            elif score >= minimum_probable_score and emit_probable:
                confidence_status = "censored" if (onset_censored or completion_censored) else "probable"
            else:
                continue

            event_key = (
                source_corridor, target_corridor, onset_frame, completion_frame
            )
            if event_key in emitted_keys:
                continue
            emitted_keys.add(event_key)
            event_rows.append({
                "source_corridor": source_corridor,
                "target_corridor": target_corridor,
                "source_frames": source_frames,
                "target_frames": target_frames,
                "source_frame": int(source_frame),
                "target_entry_frame": int(target_frames[0]),
                "completion_frame": int(completion_frame),
                "source_match": source_match,
                "target_entry_match": target_match,
                "target_match": completion_match,
                "side": side,
                "adjacency_source": adjacency_source,
                "onset_frame": onset_frame,
                "onset_source": onset_source,
                "completion_source": completion_source,
                "target_stable": target_stable,
                "onset_censored": onset_censored,
                "completion_censored": completion_censored,
                "confidence_score": score,
                "confidence_status": confidence_status,
                "confidence_reasons": score_reasons,
            })

        # Left-censored recovery when the decoded path begins in the target
        # corridor but an adjacent source lane is still visible in early
        # footprint evidence and fades as target occupancy grows.
        if bool(getattr(ARGS, "lane_change_decoder_recover_left_censored", True)) and segments:
            first_segment = next((s for s in segments if s["state"] != unknown), None)
            if first_segment is not None and first_segment["frames"]:
                target_corridor = str(first_segment["state"])
                early_frames = list(first_segment["frames"][:4])
                target_match = frame_corridor_matches.get(early_frames[0], {}).get(target_corridor)
                if target_match is not None:
                    candidates = []
                    for source_match in matches_by_frame.get(early_frames[0], []):
                        source_corridor = lane_to_corridor.get(str(source_match["lane_id"]))
                        if source_corridor is None or source_corridor == target_corridor:
                            continue
                        transition = _lane_change_local_transition_side(
                            track, frame_corridor_matches, source_corridor,
                            target_corridor, early_frames[0], early_frames[0]
                        )
                        if transition is None:
                            continue
                        side, adjacency_source, source_frame, resolved_source, _, resolved_target = transition
                        source_values = [
                            _lane_change_overlap_ratio(track[f][1], resolved_source["lane_object"])
                            for f in early_frames
                        ]
                        target_values = [
                            _lane_change_overlap_ratio(track[f][1], resolved_target["lane_object"])
                            for f in early_frames
                        ]
                        decay = source_values[0] - source_values[-1] if len(source_values) > 1 else 0.0
                        gain = target_values[-1] - target_values[0] if len(target_values) > 1 else 0.0
                        if max(source_values, default=0.0) >= 0.05 and max(decay, gain) >= 0.04:
                            candidates.append((decay + gain + max(source_values), source_corridor, side, adjacency_source, resolved_source, resolved_target))
                    if candidates:
                        candidates.sort(reverse=True, key=lambda row: row[0])
                        _, source_corridor, side, adjacency_source, source_match, target_match = candidates[0]
                        completion_frame, completion_source, target_stable = _lane_change_target_completion_frame(
                            list(first_segment["frames"]), target_corridor, frame_corridor_matches
                        )
                        key_exists = any(row["target_corridor"] == target_corridor and row["onset_frame"] == observed_frames[0] for row in event_rows)
                        if not key_exists:
                            event_rows.append({
                                "source_corridor": str(source_corridor),
                                "target_corridor": target_corridor,
                                "source_frames": [observed_frames[0]],
                                "target_frames": list(first_segment["frames"]),
                                "source_frame": observed_frames[0],
                                "target_entry_frame": observed_frames[0],
                                "completion_frame": completion_frame,
                                "source_match": source_match,
                                "target_entry_match": target_match,
                                "target_match": frame_corridor_matches.get(completion_frame, {}).get(target_corridor) or target_match,
                                "side": side,
                                "adjacency_source": "decoder_left_censored_adjacency",
                                "onset_frame": observed_frames[0],
                                "onset_source": "decoded_left_censored_track_start",
                                "completion_source": completion_source,
                                "target_stable": target_stable,
                                "onset_censored": True,
                                "completion_censored": not target_stable,
                                "confidence_score": 4.5,
                                "confidence_status": "censored",
                                "confidence_reasons": ["adjacent_source_overlap_decay", "target_occupancy_gain", "left_censored"],
                            })

        event_rows.sort(key=lambda row: (row["onset_frame"], row["completion_frame"]))
        minimum_history_frames, minimum_history_s = _maneuver_history_requirements()
        for sequence_index, event in enumerate(event_rows, start=1):
            onset_frame = int(event["onset_frame"])
            # A normal confirmed temporal maneuver must have observable source
            # history before its physical onset. Track-start / left-censored
            # events are intentionally not exported as np:changesLane.
            history = _lane_change_observed_source_history(
                track, event.get("source_frames", []), onset_frame
            )
            if event.get("onset_censored") is True:
                continue
            if int(history.get("frame_count", 0)) < minimum_history_frames:
                continue
            if float(history.get("duration_s", 0.0)) + 1e-9 < minimum_history_s:
                continue
            completion_frame = int(event["completion_frame"])
            boundary_frame = int(event["target_entry_frame"])
            source_match = event["source_match"]
            target_entry_match = event.get("target_entry_match", event["target_match"])
            target_match = event["target_match"]
            source_lane = source_match["lane_object"]
            target_entry_lane = target_entry_match["lane_object"]
            target_lane = target_match["lane_object"]
            source_entity_id = str(source_match.get("map_entity_id") or source_match["lane_id"])
            target_entry_entity_id = str(target_entry_match.get("map_entity_id") or target_entry_match["lane_id"])
            target_entity_id = str(target_match.get("map_entity_id") or target_match["lane_id"])
            source_kind, source_lane_id = _lane_change_split_entity_id(source_entity_id)
            target_entry_kind, target_entry_lane_id = _lane_change_split_entity_id(target_entry_entity_id)
            target_kind, target_lane_id = _lane_change_split_entity_id(target_entity_id)
            source_lane_id = str(source_lane_id or source_entity_id)
            target_entry_lane_id = str(target_entry_lane_id or target_entry_entity_id)
            target_lane_id = str(target_lane_id or target_entity_id)
            onset_snapshot = track[onset_frame][0]
            completion_snapshot, completion_record = track[completion_frame]
            arrow_end = _lane_change_baseline_anchor(target_lane, completion_record)
            if arrow_end is None:
                continue
            start_time_us = int(onset_snapshot["timestamp_us"])
            boundary_time_us = int(track[boundary_frame][0]["timestamp_us"])
            completion_time_us = int(completion_snapshot["timestamp_us"])
            event_id = stable_id(
                str(onset_snapshot.get("scenario_token", "")),
                "np:changesLane",
                subject_token,
                event["source_corridor"],
                event["target_corridor"],
                start_time_us,
                sequence_index,
            )
            boundary = _lane_change_boundary_evidence(
                track, source_lane, target_entry_lane,
                int(event["source_frame"]), boundary_frame, completion_frame,
            )
            source_record = track[int(event["source_frame"])][1]
            source_line = _lane_change_baseline_line(source_lane)
            source_lateral = _lane_change_signed_lateral(source_line, source_record)
            target_lateral = _lane_change_signed_lateral(source_line, completion_record)
            lateral_shift = None if source_lateral is None or target_lateral is None else float(target_lateral - source_lateral)
            active_frames = [f for f in observed_frames if onset_frame <= f <= completion_frame]
            base_evidence = {
                "lane_change_strategy": "offline_drivable_primitive_path_transfer_decoder_v9_5_25",
                "interaction_kind": "vehicle_to_structure",
                "subject_track_token": subject_token,
                "subject_agent_type": _normalized_agent_type(completion_record),
                "source_corridor_id": str(event["source_corridor"]),
                "target_corridor_id": str(event["target_corridor"]),
                "source_lane_id": source_lane_id,
                "target_entry_lane_id": target_entry_lane_id,
                "target_lane_id": target_lane_id,
                "source_map_kind": str(source_kind or source_match.get("map_kind") or "lane"),
                "target_entry_map_kind": str(target_entry_kind or target_entry_match.get("map_kind") or "lane"),
                "target_map_kind": str(target_kind or target_match.get("map_kind") or "lane"),
                "source_map_entity_id": source_entity_id,
                "target_entry_map_entity_id": target_entry_entity_id,
                "target_map_entity_id": target_entity_id,
                # Backward-compatible field names; values now correctly support
                # both lane:<id> and lane_connector:<id> structures.
                "source_lane_entity_id": source_entity_id,
                "target_entry_lane_entity_id": target_entry_entity_id,
                "target_lane_entity_id": target_entity_id,
                "lane_change_side": str(event["side"]),
                "directional_predicate_id": "np:changesLaneLeft" if event["side"] == "left" else "np:changesLaneRight",
                "adjacency_source": str(event["adjacency_source"]),
                "ordinary_adjacent_lane_change": True,
                "source_and_target_are_distinct_corridors": True,
                "event_id": event_id,
                "lane_change_sequence_index": int(sequence_index),
                "start_time_us": start_time_us,
                "boundary_time_us": boundary_time_us,
                "completion_time_us": completion_time_us,
                "maneuver_onset_frame_index": onset_frame,
                "source_segment_start_frame_index": int(event["source_frames"][0]),
                "source_end_frame_index": int(event["source_frame"]),
                "boundary_frame_index": boundary_frame,
                "target_entry_frame_index": boundary_frame,
                "completion_frame_index": completion_frame,
                "completion_confirmation_frame_index": completion_frame,
                "target_segment_end_frame_index": int(event["target_frames"][-1]),
                "onset_source": str(event["onset_source"]),
                "completion_source": str(event["completion_source"]),
                "maneuver_history_gate_passed": True,
                "maneuver_history_frame_count": int(history.get("frame_count", 0)),
                "maneuver_history_duration_s": float(history.get("duration_s", 0.0)),
                "maneuver_history_frames": list(history.get("frames", [])),
                "minimum_maneuver_history_frames": int(minimum_history_frames),
                "minimum_maneuver_history_s": float(minimum_history_s),
                "left_censored_events_exported": False,
                "condition_duration_s": float((completion_time_us - start_time_us) / 1e6),
                "lane_transition_duration_s": float((completion_time_us - boundary_time_us) / 1e6),
                "measured_lateral_shift_m": lateral_shift,
                "decoded_lane_state_model": "KEEPING_LANE|CHANGING_LEFT|CHANGING_RIGHT|UNKNOWN",
                "decoded_maneuver_state": "CHANGING_LEFT" if event["side"] == "left" else "CHANGING_RIGHT",
                "decoder_total_path_cost": float(decoder_cost),
                "event_confidence_score": float(event["confidence_score"]),
                "event_confidence_status": str(event["confidence_status"]),
                "event_confidence_reasons": list(event["confidence_reasons"]),
                "onset_censored": bool(event["onset_censored"]),
                "completion_censored": bool(event["completion_censored"]),
                "left_censored_source_inference": bool(event["onset_censored"]),
                "source_primary_lane_observed": bool(source_match.get("primary")),
                "target_primary_lane_observed": bool(target_match.get("primary")),
                "source_primary_lane_stable": len(event["source_frames"]) >= 2,
                "target_primary_lane_stable": bool(event["target_stable"]),
                "source_primary_persistence_frames": int(len(event["source_frames"])),
                "target_primary_persistence_frames": int(len(event["target_frames"])),
                "source_stable_overlap_ratio": float(source_match.get("overlap_ratio") or 0.0),
                "target_stable_overlap_ratio": float(target_match.get("overlap_ratio") or 0.0),
                "source_stable_center_covered": bool(source_match.get("center_covered")),
                "target_stable_center_covered": bool(target_match.get("center_covered")),
                "has_dual_lane_overlap": bool(boundary["has_dual_lane_overlap"]),
                "has_confirmed_lane_assignment_transition": bool(target_match.get("primary")),
                "boundary_evidence_type": "decoded_topology_constrained_corridor_transition",
                "boundary_source_overlap_ratio": float(boundary["best_source_overlap_ratio"]),
                "boundary_target_overlap_ratio": float(boundary["best_target_overlap_ratio"]),
                "boundary_dual_overlap_ratio": float(boundary["best_dual_overlap_ratio"]),
                "longitudinal_lane_successors_collapsed": True,
                "lane_and_lane_connector_primitives_supported": True,
                "direct_successor_transition_is_not_lane_change": True,
                "lateral_lane_connector_transfer_supported": True,
                "multiple_lane_changes_per_subject_supported": True,
                "primary_lane_is_event_confirmation": False,
                "candidate_lane_overlap_is_diagnostic_only": False,
                "track_gaps_tolerated": True,
                "arrow_activation_scope": "maneuver_onset_through_completion",
                "arrow_start_kind": "current_subject_position",
                "arrow_end_kind": "target_lane_baseline_at_maneuver_completion",
                "lane_change_arrow_end_x": float(arrow_end[0]),
                "lane_change_arrow_end_y": float(arrow_end[1]),
                "supporting_predicates": [
                    "np:hasPrimaryLane",
                    "np:hasPrimaryLaneConnector",
                    "np:inLane",
                    "np:inLaneConnector",
                    "np:intersectsLane",
                    "np:intersectsLaneConnector",
                    "np:hasBaselineLateralOffset",
                ],
            }
            for frame_index in active_frames:
                snapshot, current = track[frame_index]
                phase_number, phase_id, phase_label = _lane_change_phase(frame_index, boundary_frame)
                evidence = dict(base_evidence)
                evidence.update({
                    "lane_change_active": True,
                    "lane_change_completed": frame_index == completion_frame,
                    "lane_change_phase": phase_id,
                    "lane_change_phase_number": phase_number,
                    "lane_change_phase_label": phase_label,
                    "lane_change_current_frame_index": int(frame_index),
                    "lane_change_current_time_us": int(snapshot["timestamp_us"]),
                    "lane_change_current_subject_x": float(current["xyh"][0]),
                    "lane_change_current_subject_y": float(current["xyh"][1]),
                    "lane_change_arrow_start_x": float(current["xyh"][0]),
                    "lane_change_arrow_start_y": float(current["xyh"][1]),
                    "current_source_lane_overlap_ratio": _lane_change_overlap_ratio(current, source_lane),
                    "current_target_lane_overlap_ratio": _lane_change_overlap_ratio(current, target_lane),
                    "condition_duration_s": float((int(snapshot["timestamp_us"]) - start_time_us) / 1e6),
                })
                assertions.append(derived_relation(
                    "np:changesLane",
                    current["entity_id"],
                    target_entity_id,
                    snapshot["frame"],
                    LANE_CHANGE_RULE_ID,
                    evidence,
                    start_time_us=start_time_us,
                    end_time_us=int(snapshot["timestamp_us"]),
                ))
    return assertions

# ---------------------------------------------------------------------------
# Vehicle-to-vehicle merge interactions derived from confirmed lane changes
# ---------------------------------------------------------------------------

MERGE_RULE_ID = "R-MERGE-INTERACTION-001"
MERGE_IN_FRONT_PREDICATE = "np:mergesInFrontOf"
MERGE_BEHIND_PREDICATE = "np:mergesBehind"


def _merge_arg(name: str, default: float) -> float:
    try:
        return float(getattr(ARGS, name, default))
    except (TypeError, ValueError):
        return float(default)


def _merge_int_arg(name: str, default: int) -> int:
    try:
        return int(getattr(ARGS, name, default))
    except (TypeError, ValueError):
        return int(default)


def _merge_assertion_value(assertion: Any, name: str, default: Any = None) -> Any:
    if isinstance(assertion, dict):
        return assertion.get(name, default)
    return getattr(assertion, name, default)


def _merge_lane_change_events(lane_change_assertions: list[Any]) -> list[dict[str, Any]]:
    """Return one canonical record per decoded lane-change event.

    ``np:changesLane`` is assigned on every active maneuver frame.  Merge
    reasoning must run once per physical maneuver, therefore rows are grouped by
    ``event_id`` and the completion row is preferred as the canonical event.
    """
    groups: dict[str, list[Any]] = defaultdict(list)
    fallback_counter = 0
    for assertion in lane_change_assertions or []:
        if str(_merge_assertion_value(assertion, "predicate_id", "")) != "np:changesLane":
            continue
        evidence = dict(_merge_assertion_value(assertion, "evidence", {}) or {})
        event_id = str(evidence.get("event_id") or "")
        if not event_id:
            fallback_counter += 1
            event_id = "fallback:{}:{}:{}".format(
                _merge_assertion_value(assertion, "subject_id", ""),
                evidence.get("start_time_us"),
                fallback_counter,
            )
        groups[event_id].append(assertion)

    result = []
    for event_id, rows in groups.items():
        def row_key(row: Any) -> tuple[int, int]:
            evidence = dict(_merge_assertion_value(row, "evidence", {}) or {})
            completed = 1 if evidence.get("lane_change_completed") is True else 0
            try:
                current_frame = int(evidence.get("lane_change_current_frame_index", -1))
            except (TypeError, ValueError):
                current_frame = -1
            return completed, current_frame

        canonical = max(rows, key=row_key)
        evidence = dict(_merge_assertion_value(canonical, "evidence", {}) or {})
        evidence["event_id"] = event_id
        result.append({
            "assertion": canonical,
            "event_id": event_id,
            "subject_id": str(_merge_assertion_value(canonical, "subject_id", "") or ""),
            "subject_token": str(
                evidence.get("subject_track_token")
                or _merge_assertion_value(canonical, "subject_id", "")
                or ""
            ).replace("entity:", ""),
            "evidence": evidence,
            "rows": rows,
        })
    result.sort(key=lambda event: (
        int(event["evidence"].get("start_time_us") or 0),
        event["subject_token"],
        event["event_id"],
    ))
    return result


def _merge_exact_match(record: dict[str, Any], entity_id: str) -> Optional[dict[str, Any]]:
    wanted = str(entity_id or "")
    if not wanted:
        return None
    for match in _lane_change_all_matches(record):
        if str(match.get("lane_id") or "") == wanted:
            return match
    return None


def _merge_node_from_match(match: dict[str, Any]) -> Optional[_PathNode]:
    if not match:
        return None
    map_object = match.get("lane_object")
    raw_id = str(match.get("map_object_id") or _lane_change_edge_id(map_object) or "")
    kind = str(match.get("map_kind") or "")
    length = _finite_float(_baseline_length(map_object)) if map_object is not None else None
    if not raw_id or kind not in {"lane", "lane_connector"} or map_object is None:
        return None
    if length is None or length <= 0.0:
        return None
    return _PathNode(raw_id, kind, map_object, float(length))


def _merge_unique_forward_path_between_matches(
    snapshot: dict[str, Any],
    start_match: dict[str, Any],
    target_match: dict[str, Any],
    max_hops: int,
) -> Optional[tuple[_PathNode, ...]]:
    """Return an unbranched directed path between two selected primitives.

    This intentionally mirrors the conservative path logic used by ``follows``.
    A predecessor at a split is not considered to prove occupancy of a specific
    future branch.  Once a Lane/LaneConnector branch is actually observed, the
    relation becomes resolvable.
    """
    start = _merge_node_from_match(start_match)
    target = _merge_node_from_match(target_match)
    if start is None or target is None:
        return None
    if start.object_id == target.object_id:
        return (start,) if start.kind == target.kind else None

    cache: dict[str, Optional[_PathNode]] = {
        start.object_id: start,
        target.object_id: target,
    }
    path = [start]
    visited = {start.object_id}
    current = start
    for _ in range(max(1, int(max_hops))):
        outgoing_ids = sorted(
            str(value) for value in _map_edge_ids(current.map_object, "outgoing")
        )
        if len(outgoing_ids) != 1:
            return None
        next_id = outgoing_ids[0]
        if next_id in visited:
            return None
        next_node = _resolve_map_object(snapshot["frame"], next_id, cache)
        if next_node is None:
            return None
        path.append(next_node)
        visited.add(next_id)
        current = next_node
        if current.object_id == target.object_id:
            if current.kind != target.kind:
                return None
            path[-1] = target
            return tuple(path)
    return None


def _merge_matches_same_target_path(
    snapshot: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    max_hops: int,
) -> bool:
    if not first or not second:
        return False
    return bool(
        _merge_unique_forward_path_between_matches(snapshot, first, second, max_hops)
        or _merge_unique_forward_path_between_matches(snapshot, second, first, max_hops)
    )


def _merge_match_is_strong(match: Optional[dict[str, Any]]) -> bool:
    if match is None:
        return False
    overlap = float(match.get("overlap_ratio") or 0.0)
    return bool(
        match.get("primary")
        or match.get("center_covered")
        or overlap >= _merge_arg("merge_minimum_path_overlap_ratio", 0.20)
    )


def _merge_best_match_on_anchor_path(
    snapshot: dict[str, Any],
    record: dict[str, Any],
    anchor_match: dict[str, Any],
    max_hops: int,
) -> Optional[dict[str, Any]]:
    candidates = []
    for match in _lane_change_all_matches(record):
        if not _merge_match_is_strong(match):
            continue
        if _merge_matches_same_target_path(snapshot, match, anchor_match, max_hops):
            candidates.append(match)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.get("occupancy_score") or 0.0),
            float(row.get("overlap_ratio") or 0.0),
            bool(row.get("center_covered")),
            bool(row.get("primary")),
        ),
    )


def _merge_record_for_match(record: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    node = _merge_node_from_match(match)
    if node is None:
        return result
    result.update({
        "primary_map_object_id": node.object_id,
        "primary_map_kind": node.kind,
        "primary_map_object": node.map_object,
        "baseline_length_m": node.length_m,
        # The ambiguity of the original primary match has already been resolved
        # by selecting one strong target-path candidate above.
        "map_match_ambiguous": False,
    })
    return result


def _merge_pair_order(
    snapshot: dict[str, Any],
    subject: dict[str, Any],
    subject_match: dict[str, Any],
    obj: dict[str, Any],
    object_match: dict[str, Any],
    max_hops: int,
) -> Optional[dict[str, Any]]:
    """Return subject order on the shared target path using footprint intervals."""
    subject_norm = _merge_record_for_match(subject, subject_match)
    object_norm = _merge_record_for_match(obj, object_match)

    # Object -> subject means the subject is longitudinally ahead.
    object_to_subject = _merge_unique_forward_path_between_matches(
        snapshot, object_match, subject_match, max_hops
    )
    if object_to_subject is not None:
        evidence = _path_evidence(object_to_subject, object_norm, subject_norm, "merge_target_path")
        if (
            evidence is not None
            and evidence.center_path_distance_m > ARGS.distance_epsilon_m
            and evidence.bumper_gap_m > ARGS.distance_epsilon_m
        ):
            return {
                "order": "subject_ahead",
                "bumper_gap_m": float(evidence.bumper_gap_m),
                "center_path_distance_m": float(evidence.center_path_distance_m),
                "path_object_ids": [node.object_id for node in evidence.nodes],
                "path_object_kinds": [node.kind for node in evidence.nodes],
            }

    subject_to_object = _merge_unique_forward_path_between_matches(
        snapshot, subject_match, object_match, max_hops
    )
    if subject_to_object is not None:
        evidence = _path_evidence(subject_to_object, subject_norm, object_norm, "merge_target_path")
        if (
            evidence is not None
            and evidence.center_path_distance_m > ARGS.distance_epsilon_m
            and evidence.bumper_gap_m > ARGS.distance_epsilon_m
        ):
            return {
                "order": "subject_behind",
                "bumper_gap_m": float(evidence.bumper_gap_m),
                "center_path_distance_m": float(evidence.center_path_distance_m),
                "path_object_ids": [node.object_id for node in evidence.nodes],
                "path_object_kinds": [node.kind for node in evidence.nodes],
            }
    return None


def _merge_same_flow(subject: dict[str, Any], obj: dict[str, Any]) -> tuple[bool, Optional[float]]:
    subject_direction = _record_direction(subject)
    object_direction = _record_direction(obj)
    if (
        not subject_direction.reliable
        or subject_direction.heading_rad is None
        or not object_direction.reliable
        or object_direction.heading_rad is None
    ):
        return False, None
    difference = angular_difference(
        float(subject_direction.heading_rad), float(object_direction.heading_rad)
    )
    return (
        difference <= _merge_arg("merge_same_flow_threshold_rad", 0.55),
        float(difference),
    )


def _merge_overlapping_lane_change_spans(
    lane_change_events: list[dict[str, Any]],
) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for event in lane_change_events:
        evidence = event["evidence"]
        try:
            start = int(evidence.get("start_time_us"))
            end = int(evidence.get("completion_time_us"))
        except (TypeError, ValueError):
            continue
        result[event["subject_token"]].append((start, end))
    return result


def _merge_track_changes_during(
    spans: dict[str, list[tuple[int, int]]],
    token: str,
    start_us: int,
    end_us: int,
) -> bool:
    return any(not (finish < start_us or begin > end_us) for begin, finish in spans.get(token, []))


def _merge_object_preexisting_target_support(
    snapshots_by_frame: dict[int, dict[str, Any]],
    observed_frames: list[int],
    object_token: str,
    target_anchor: dict[str, Any],
    source_anchor: dict[str, Any],
    boundary_frame: int,
    boundary_time_us: int,
    max_hops: int,
) -> dict[str, Any]:
    lookback_us = int(round(_merge_arg("merge_target_preexistence_window_s", 2.0) * 1e6))
    target_rows = []
    for frame_index in observed_frames:
        if frame_index > boundary_frame:
            break
        snapshot = snapshots_by_frame[frame_index]
        timestamp_us = int(snapshot["timestamp_us"])
        if timestamp_us < boundary_time_us - lookback_us:
            continue
        obj = snapshot.get("entities", {}).get(object_token)
        if obj is None:
            continue
        target_match = _merge_best_match_on_anchor_path(
            snapshot, obj, target_anchor, max_hops
        )
        if target_match is None:
            continue
        source_match = _merge_best_match_on_anchor_path(
            snapshot, obj, source_anchor, max_hops
        )
        # The other vehicle must already belong to the target stream, not to
        # the lane-changing subject's source stream.
        if source_match is not None:
            source_score = float(source_match.get("occupancy_score") or 0.0)
            target_score = float(target_match.get("occupancy_score") or 0.0)
            if source_score >= target_score - 0.05:
                continue
        target_rows.append((frame_index, timestamp_us, target_match))
    return {
        "frame_count": len(target_rows),
        "frames": [row[0] for row in target_rows],
        "first_time_us": target_rows[0][1] if target_rows else None,
        "last_time_us": target_rows[-1][1] if target_rows else None,
    }


def _merge_stable_post_order(
    snapshots_by_frame: dict[int, dict[str, Any]],
    observed_frames: list[int],
    subject_token: str,
    object_token: str,
    target_anchor: dict[str, Any],
    completion_frame: int,
    completion_time_us: int,
    max_hops: int,
) -> Optional[dict[str, Any]]:
    window_us = int(round(_merge_arg("merge_order_confirmation_window_s", 2.0) * 1e6))
    maximum_gap = _merge_arg("merge_maximum_neighbor_gap_m", 40.0)
    maximum_headway_s = _merge_arg("merge_maximum_headway_s", 5.0)
    minimum_headway_speed_mps = _merge_arg("minimum_forward_speed_mps", 0.30)
    samples = []
    for frame_index in observed_frames:
        if frame_index < completion_frame:
            continue
        snapshot = snapshots_by_frame[frame_index]
        timestamp_us = int(snapshot["timestamp_us"])
        if timestamp_us > completion_time_us + window_us:
            break
        subject = snapshot.get("entities", {}).get(subject_token)
        obj = snapshot.get("entities", {}).get(object_token)
        if subject is None or obj is None:
            continue
        subject_match = _merge_best_match_on_anchor_path(
            snapshot, subject, target_anchor, max_hops
        )
        object_match = _merge_best_match_on_anchor_path(
            snapshot, obj, target_anchor, max_hops
        )
        if subject_match is None or object_match is None:
            continue
        same_flow, direction_difference = _merge_same_flow(subject, obj)
        if not same_flow:
            continue
        geometry = footprint_metrics(subject, obj)
        if geometry.get("overlapping") is True:
            continue
        order = _merge_pair_order(
            snapshot, subject, subject_match, obj, object_match, max_hops
        )
        if order is None or order["bumper_gap_m"] > maximum_gap:
            continue

        # A merge is an interaction with the nearest target-stream neighbor,
        # not merely a final ahead/behind ordering.  Measure time headway from
        # the *trailing* vehicle to the leading one after the subject becomes
        # established in the target stream.  In queues / near-stop traffic,
        # headway is intentionally left undefined and the metric gap gate
        # remains authoritative.
        trailing_record = obj if order["order"] == "subject_ahead" else subject
        trailing_forward_speed_mps = _forward_speed_on_path(trailing_record)
        post_merge_headway_s = None
        headway_gate_applied = False
        if (
            trailing_forward_speed_mps is not None
            and math.isfinite(trailing_forward_speed_mps)
            and trailing_forward_speed_mps > minimum_headway_speed_mps
        ):
            post_merge_headway_s = float(order["bumper_gap_m"] / trailing_forward_speed_mps)
            headway_gate_applied = True
            if post_merge_headway_s > maximum_headway_s:
                continue

        samples.append({
            "frame_index": int(frame_index),
            "timestamp_us": timestamp_us,
            "subject": subject,
            "object": obj,
            "subject_match": subject_match,
            "object_match": object_match,
            "direction_difference_rad": direction_difference,
            "trailing_forward_speed_mps": trailing_forward_speed_mps,
            "post_merge_headway_s": post_merge_headway_s,
            "headway_gate_applied": headway_gate_applied,
            **order,
        })

    if not samples:
        return None
    first_order = samples[0]["order"]
    consistent = []
    for sample in samples:
        if sample["order"] != first_order:
            break
        consistent.append(sample)
    required = max(1, _merge_int_arg("merge_order_stable_frames", 2))
    if len(consistent) < required:
        # At the end of a scenario/track we retain one strong final observation
        # as explicitly censored evidence instead of silently losing the event.
        allow_terminal = bool(getattr(ARGS, "merge_allow_terminal_single_frame", True))
        last_observed = observed_frames[-1] if observed_frames else completion_frame
        if not (allow_terminal and len(consistent) == 1 and consistent[0]["frame_index"] >= last_observed - 1):
            return None
        status = "censored"
    else:
        status = "confirmed"

    representative = consistent[min(len(consistent), required) - 1]
    return {
        "order": first_order,
        "status": status,
        "samples": consistent,
        "stable_frame_count": len(consistent),
        "confirmation_frame_index": int(representative["frame_index"]),
        "confirmation_time_us": int(representative["timestamp_us"]),
        "bumper_gap_m": float(representative["bumper_gap_m"]),
        "center_path_distance_m": float(representative["center_path_distance_m"]),
        "path_object_ids": list(representative["path_object_ids"]),
        "path_object_kinds": list(representative["path_object_kinds"]),
        "direction_difference_rad": representative["direction_difference_rad"],
        "trailing_forward_speed_mps": representative.get("trailing_forward_speed_mps"),
        "post_merge_headway_s": representative.get("post_merge_headway_s"),
        "headway_gate_applied": bool(representative.get("headway_gate_applied", False)),
        "object_entity_id": str(representative["object"].get("entity_id") or object_token),
        "object_target_map_entity_id": str(representative["object_match"].get("lane_id") or ""),
    }


def derive_merging_predicates(snapshots, lane_change_assertions=None):
    """Derive vehicle-to-vehicle ``mergesInFrontOf`` / ``mergesBehind``.

    A merge interaction is anchored to one already decoded lateral corridor
    transfer.  The other vehicle must have occupied the *target* stream before
    the subject crossed into it and must remain on that same directed target
    path after completion.  Final longitudinal order is measured with
    path-projected oriented footprints and must persist.  For each lane change
    we select at most one nearest target-stream neighbor behind and one nearest
    neighbor ahead, so a vehicle entering a gap can legitimately satisfy both
    relations simultaneously.

    This definition works uniformly for Lane and LaneConnector primitives and
    deliberately rejects normal Lane -> LaneConnector -> Lane progression,
    because no ``np:changesLane`` event exists for longitudinal continuation.
    """
    if not ARGS.derive_semantic_predicates or not snapshots:
        return []
    snapshots = list(snapshots)
    if lane_change_assertions is None:
        lane_change_assertions = derive_lane_change_predicates(snapshots)
    lane_change_assertions = list(lane_change_assertions or [])
    events = _merge_lane_change_events(lane_change_assertions)
    if not events:
        return []

    snapshots_by_frame = {
        int(snapshot["frame_index"]): snapshot for snapshot in snapshots
    }
    observed_frames = sorted(snapshots_by_frame)
    if not observed_frames:
        return []
    spans = _merge_overlapping_lane_change_spans(events)
    max_hops = max(1, _merge_int_arg("merge_max_target_path_hops", 6))
    minimum_pre_frames = max(1, _merge_int_arg("merge_minimum_preexisting_target_frames", 1))
    vehicle_tokens = {
        str(token)
        for snapshot in snapshots
        for token, record in snapshot.get("entities", {}).items()
        if _is_follow_vehicle(record)
    }

    assertions = []
    for event in events:
        evidence = event["evidence"]
        # Final order must be observable. A right-censored lane change cannot
        # establish that the subject actually became ahead/behind in the target.
        if evidence.get("completion_censored") is True:
            continue
        # Merge activation inherits the lane-change onset. Never promote a
        # track-start / left-censored lane change into a confirmed merge.
        minimum_history_frames, minimum_history_s = _maneuver_history_requirements()
        if evidence.get("onset_censored") is True:
            continue
        try:
            lane_history_frames = int(evidence.get("maneuver_history_frame_count", 0))
            lane_history_duration_s = float(evidence.get("maneuver_history_duration_s", 0.0))
        except (TypeError, ValueError):
            continue
        if lane_history_frames < minimum_history_frames:
            continue
        if lane_history_duration_s + 1e-9 < minimum_history_s:
            continue
        try:
            onset_frame = int(evidence["maneuver_onset_frame_index"])
            boundary_frame = int(evidence["boundary_frame_index"])
            completion_frame = int(evidence["completion_frame_index"])
            start_time_us = int(evidence["start_time_us"])
            boundary_time_us = int(evidence["boundary_time_us"])
            completion_time_us = int(evidence["completion_time_us"])
        except (KeyError, TypeError, ValueError):
            continue
        if completion_frame not in snapshots_by_frame or boundary_frame not in snapshots_by_frame:
            continue

        subject_token = str(event["subject_token"])
        completion_snapshot = snapshots_by_frame[completion_frame]
        completion_subject = completion_snapshot.get("entities", {}).get(subject_token)
        if completion_subject is None:
            continue
        target_entity_id = str(evidence.get("target_map_entity_id") or evidence.get("target_lane_entity_id") or "")
        source_entity_id = str(evidence.get("source_map_entity_id") or evidence.get("source_lane_entity_id") or "")
        target_anchor = _merge_exact_match(completion_subject, target_entity_id)
        if target_anchor is None:
            continue

        source_snapshot = snapshots_by_frame.get(int(evidence.get("source_end_frame_index", onset_frame)))
        source_subject = None if source_snapshot is None else source_snapshot.get("entities", {}).get(subject_token)
        source_anchor = None if source_subject is None else _merge_exact_match(source_subject, source_entity_id)
        if source_anchor is None:
            # Left-censored lane changes may have source evidence only in the
            # event's first frame. Search backward/forward within the event.
            for frame_index in observed_frames:
                if frame_index < onset_frame or frame_index > boundary_frame:
                    continue
                subject = snapshots_by_frame[frame_index].get("entities", {}).get(subject_token)
                if subject is not None:
                    source_anchor = _merge_exact_match(subject, source_entity_id)
                    if source_anchor is not None:
                        break
        if source_anchor is None:
            continue

        candidate_rows = {"subject_ahead": [], "subject_behind": []}
        candidate_tokens = vehicle_tokens - {subject_token}

        for object_token in sorted(candidate_tokens):
            # A simultaneous lateral maneuver by the other vehicle makes
            # "merges into O's corridor" ambiguous; omit instead of guessing.
            if _merge_track_changes_during(
                spans, object_token, boundary_time_us, completion_time_us
            ):
                continue
            pre_support = _merge_object_preexisting_target_support(
                snapshots_by_frame,
                observed_frames,
                object_token,
                target_anchor,
                source_anchor,
                boundary_frame,
                boundary_time_us,
                max_hops,
            )
            if int(pre_support["frame_count"]) < minimum_pre_frames:
                continue
            order = _merge_stable_post_order(
                snapshots_by_frame,
                observed_frames,
                subject_token,
                object_token,
                target_anchor,
                completion_frame,
                completion_time_us,
                max_hops,
            )
            if order is None:
                continue
            candidate_rows[order["order"]].append({
                "object_token": object_token,
                "pre_support": pre_support,
                "order": order,
            })

        # Select the nearest relevant neighbor on each side of the inserted
        # subject. This prevents one lane change from relating the subject to
        # every vehicle in a long target-lane queue.
        selected = []
        for order_name, rows in candidate_rows.items():
            if not rows:
                continue
            rows.sort(key=lambda row: (
                float(row["order"]["bumper_gap_m"]),
                row["object_token"],
            ))
            selected.append(rows[0])

        active_frames = [
            frame for frame in observed_frames
            if onset_frame <= frame <= completion_frame
            and subject_token in snapshots_by_frame[frame].get("entities", {})
        ]
        for selected_row in selected:
            object_token = selected_row["object_token"]
            order = selected_row["order"]
            pre_support = selected_row["pre_support"]
            predicate_id = (
                MERGE_IN_FRONT_PREDICATE
                if order["order"] == "subject_ahead"
                else MERGE_BEHIND_PREDICATE
            )
            relation_label = (
                "merges_in_front_of"
                if order["order"] == "subject_ahead"
                else "merges_behind"
            )
            merge_event_id = stable_id(
                str(completion_snapshot.get("scenario_token", "")),
                predicate_id,
                event["event_id"],
                subject_token,
                object_token,
            )
            base_evidence = {
                "merge_strategy": "lane_change_anchored_target_stream_neighbor_v9_5_31",
                "interaction_kind": "vehicle_to_vehicle",
                "merge_event_id": merge_event_id,
                "subject_track_token": subject_token,
                "object_track_token": object_token,
                "source_lane_change_event_id": event["event_id"],
                "source_lane_change_sequence_index": evidence.get("lane_change_sequence_index"),
                "source_lane_change_side": evidence.get("lane_change_side"),
                "source_map_entity_id": source_entity_id,
                "target_map_entity_id": target_entity_id,
                "source_map_kind": evidence.get("source_map_kind"),
                "target_map_kind": evidence.get("target_map_kind"),
                "lane_and_lane_connector_targets_supported": True,
                "longitudinal_continuation_is_not_merge": True,
                "other_vehicle_preexists_in_target_stream": True,
                "other_vehicle_target_support_frame_count": int(pre_support["frame_count"]),
                "other_vehicle_target_support_frames": list(pre_support["frames"]),
                "other_vehicle_target_support_first_time_us": pre_support["first_time_us"],
                "other_vehicle_target_support_last_time_us": pre_support["last_time_us"],
                "subject_final_order": order["order"],
                "merge_relation_label": relation_label,
                "post_merge_bumper_gap_m": float(order["bumper_gap_m"]),
                "post_merge_maximum_neighbor_gap_m": float(_merge_arg("merge_maximum_neighbor_gap_m", 40.0)),
                "post_merge_headway_s": order.get("post_merge_headway_s"),
                "post_merge_maximum_headway_s": float(_merge_arg("merge_maximum_headway_s", 5.0)),
                "post_merge_trailing_forward_speed_mps": order.get("trailing_forward_speed_mps"),
                "post_merge_headway_gate_applied": bool(order.get("headway_gate_applied", False)),
                "post_merge_headway_reference": (
                    "object_as_trailing_vehicle"
                    if order["order"] == "subject_ahead"
                    else "subject_as_trailing_vehicle"
                ),
                "post_merge_center_path_distance_m": float(order["center_path_distance_m"]),
                "post_merge_order_stable_frames": int(order["stable_frame_count"]),
                "post_merge_order_status": str(order["status"]),
                "post_merge_order_confirmation_frame_index": int(order["confirmation_frame_index"]),
                "post_merge_order_confirmation_time_us": int(order["confirmation_time_us"]),
                "post_merge_target_path_object_ids": list(order["path_object_ids"]),
                "post_merge_target_path_object_kinds": list(order["path_object_kinds"]),
                "post_merge_direction_difference_rad": order["direction_difference_rad"],
                "other_vehicle_target_map_entity_id": order["object_target_map_entity_id"],
                "nearest_neighbor_selection": True,
                "nearest_neighbor_scope": "nearest_ahead_and_nearest_behind_on_preexisting_target_stream",
                "start_time_us": start_time_us,
                "boundary_time_us": boundary_time_us,
                "completion_time_us": completion_time_us,
                "maneuver_onset_frame_index": onset_frame,
                "boundary_frame_index": boundary_frame,
                "completion_frame_index": completion_frame,
                "condition_duration_s": float((completion_time_us - start_time_us) / 1e6),
                "supporting_predicates": ["np:changesLane"],
                "maneuver_history_gate_passed": True,
                "maneuver_history_frame_count": lane_history_frames,
                "maneuver_history_duration_s": lane_history_duration_s,
                "minimum_maneuver_history_frames": int(minimum_history_frames),
                "minimum_maneuver_history_s": float(minimum_history_s),
                "history_role": "pre_merge_source_state_observation",
            }
            for frame_index in active_frames:
                snapshot = snapshots_by_frame[frame_index]
                subject = snapshot["entities"][subject_token]
                obj = snapshot.get("entities", {}).get(object_token)
                current_time_us = int(snapshot["timestamp_us"])
                current_evidence = dict(base_evidence)
                current_evidence.update({
                    "merge_active": True,
                    "merge_completed": frame_index == completion_frame,
                    "merge_current_frame_index": int(frame_index),
                    "merge_current_time_us": current_time_us,
                    "merge_current_subject_x": float(subject["xyh"][0]),
                    "merge_current_subject_y": float(subject["xyh"][1]),
                    "merge_current_object_x": None if obj is None else float(obj["xyh"][0]),
                    "merge_current_object_y": None if obj is None else float(obj["xyh"][1]),
                    "condition_duration_s": float((current_time_us - start_time_us) / 1e6),
                })
                assertions.append(derived_relation(
                    predicate_id,
                    subject["entity_id"],
                    order["object_entity_id"],
                    snapshot["frame"],
                    MERGE_RULE_ID,
                    current_evidence,
                    start_time_us=start_time_us,
                    end_time_us=current_time_us,
                ))
    return assertions


# ---------------------------------------------------------------------------

# Notebook-faithful fully observed same-lane-start overtaking event
# ---------------------------------------------------------------------------

OVERTAKE_RULE_ID = "R-OVERTAKE-NOTEBOOK-CASE1-006"

OVERTAKE_CASE_1 = "case1_same_lane_start"

_OVERTAKE_CASE_LABELS = {
    OVERTAKE_CASE_1: "Case 1: same-lane start",
}

# The following constants reproduce the fully observed Case 1 detector in
# the internal Case-1 specification. Case 2 (already in the adjacent lane when the
# observation begins) is intentionally unsupported from v9.5.26 onward.
# CLI values with the corresponding names may override these defaults.
_NB_PAIR_QUERY_RADIUS_M = 80.0
_NB_MIN_COMMON_FRAMES = 5
_NB_BROAD_MAX_FRAME_GAP_S = 2.5
_NB_BROAD_MAX_EVENT_DURATION_S = 35.0
_NB_BROAD_MIN_INITIAL_BEHIND_GAP_M = 0.5
_NB_BROAD_FINAL_SAME_LANE_LATERAL_M = 3.0
_NB_BROAD_MIN_SIDE_BY_SIDE_LATERAL_M = 1.0
_NB_BROAD_MAX_SIDE_BY_SIDE_CENTER_LONGITUDINAL_M = 8.0
_NB_BROAD_CASE1_MAX_INITIAL_LATERAL_M = 3.0
_NB_BROAD_CASE1_MIN_LATERAL_EXCURSION_M = 1.25
_NB_BROAD_CASE1_MAX_RETURN_LATERAL_DELTA_M = 2.5
_NB_BROAD_MIN_SUBJECT_SPEED_MPS = 1.5
_NB_BROAD_MIN_OBJECT_SPEED_MPS = 0.2
_NB_BROAD_MIN_RELATIVE_SPEED_MPS = 0.1
_NB_BROAD_SAME_FLOW_THRESHOLD_RAD = 0.75
_NB_BROAD_MIN_SAME_FLOW_FRACTION = 0.65
_NB_SEARCH_PADDING_S = 4.0
_NB_MIN_INITIAL_BEHIND_GAP_M = 1.0
_NB_MIN_SIDE_BY_SIDE_LATERAL_M = 1.25
_NB_MIN_FULL_CLEARANCE_M = 1.0
_NB_MIN_SUBJECT_SPEED_MPS = 2.0
_NB_MIN_OBJECT_SPEED_MPS = 0.5
_NB_MIN_RELATIVE_SPEED_MPS = 0.3
_NB_SAME_FLOW_THRESHOLD_RAD = 0.55
_NB_MIN_SAME_FLOW_FRACTION = 0.80
_NB_MIN_FINAL_STABLE_FRAMES = 2
_NB_MIN_FINAL_STABLE_DURATION_S = 0.5
_NB_MAX_EVENT_DURATION_S = 30.0
_NB_MAX_FRAME_GAP_S = 1.5
_NB_LANE_QUERY_RADIUS_M = 8.0
_NB_MIN_LANE_FOOTPRINT_OVERLAP_RATIO = 0.10
_NB_LANE_AMBIGUITY_OVERLAP_MARGIN = 0.05
_NB_MAX_LANE_PATH_HOPS = 8
_NB_POSITIVE_DEDUPLICATION_TOLERANCE_S = 2.0


def _overtake_arg(name: str, default: float) -> float:
    try:
        return float(getattr(ARGS, name, default))
    except (TypeError, ValueError):
        return float(default)


def _overtake_int_arg(name: str, default: int) -> int:
    try:
        return int(getattr(ARGS, name, default))
    except (TypeError, ValueError):
        return int(default)


def _overtake_relation_family(subject_token: str, object_token: str) -> str:
    subject_is_ego = str(subject_token).lower() == "ego"
    object_is_ego = str(object_token).lower() == "ego"
    if subject_is_ego and not object_is_ego:
        return "ego_to_agent"
    if object_is_ego and not subject_is_ego:
        return "agent_to_ego"
    return "agent_to_agent"


def _is_overtake_vehicle(record: dict[str, Any]) -> bool:
    if str(record.get("track_token", "")).lower() == "ego":
        return True
    agent_type = _normalized_agent_type(record)
    if any(marker in agent_type for marker in _EXCLUDED_FOLLOW_TYPE_MARKERS):
        return False
    return any(marker in agent_type for marker in _INCLUDED_FOLLOW_TYPE_MARKERS)


def _overtake_circular_mean(angle_a: float, angle_b: float) -> float:
    x_component = math.cos(float(angle_a)) + math.cos(float(angle_b))
    y_component = math.sin(float(angle_a)) + math.sin(float(angle_b))
    if abs(x_component) + abs(y_component) < 1e-9:
        return float(angle_a)
    return math.atan2(y_component, x_component)


def _overtake_polygon_points(record: dict[str, Any]) -> list[tuple[float, float]]:
    polygon = oriented_box_polygon(record)
    if polygon is None:
        return []
    try:
        coordinates = list(polygon.exterior.coords)
    except Exception:
        return []
    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates = coordinates[:-1]
    points = []
    for coordinate in coordinates:
        try:
            points.append((float(coordinate[0]), float(coordinate[1])))
        except Exception:
            continue
    return points


def _project_overtake_box_interval(
    record: dict[str, Any], reference_heading: float
) -> Optional[tuple[float, float]]:
    axis_x = math.cos(float(reference_heading))
    axis_y = math.sin(float(reference_heading))
    projections = [
        x * axis_x + y * axis_y
        for x, y in _overtake_polygon_points(record)
    ]
    if not projections:
        return None
    return float(min(projections)), float(max(projections))


def _overtake_pair_geometry(
    subject: dict[str, Any], obj: dict[str, Any]
) -> Optional[dict[str, Any]]:
    try:
        sx, sy, subject_heading = map(float, subject["xyh"])
        ox, oy, object_heading = map(float, obj["xyh"])
    except Exception:
        return None

    reference_heading = _overtake_circular_mean(subject_heading, object_heading)
    longitudinal_axis = (
        math.cos(reference_heading),
        math.sin(reference_heading),
    )
    lateral_axis = (
        -math.sin(reference_heading),
        math.cos(reference_heading),
    )
    relative_x = sx - ox
    relative_y = sy - oy

    subject_interval = _project_overtake_box_interval(subject, reference_heading)
    object_interval = _project_overtake_box_interval(obj, reference_heading)
    if subject_interval is None or object_interval is None:
        return None

    signed_overlap_m = (
        min(subject_interval[1], object_interval[1])
        - max(subject_interval[0], object_interval[0])
    )
    subject_polygon = oriented_box_polygon(subject)
    object_polygon = oriented_box_polygon(obj)
    try:
        intersection_area = float(subject_polygon.intersection(object_polygon).area)
    except Exception:
        intersection_area = 0.0

    distance_epsilon = _overtake_arg("distance_epsilon_m", 1e-3)
    overlap_epsilon = _overtake_arg("overlap_area_epsilon_m2", 1e-4)
    return {
        "reference_heading_rad": float(reference_heading),
        "subject_center_longitudinal_m": float(
            relative_x * longitudinal_axis[0]
            + relative_y * longitudinal_axis[1]
        ),
        "subject_center_lateral_m": float(
            relative_x * lateral_axis[0]
            + relative_y * lateral_axis[1]
        ),
        "subject_rear_progress": float(subject_interval[0]),
        "subject_front_progress": float(subject_interval[1]),
        "object_rear_progress": float(object_interval[0]),
        "object_front_progress": float(object_interval[1]),
        "subject_behind_gap_m": float(object_interval[0] - subject_interval[1]),
        "subject_clearance_gap_m": float(subject_interval[0] - object_interval[1]),
        "longitudinal_overlap_m": float(max(0.0, signed_overlap_m)),
        "longitudinal_intervals_overlap": bool(
            signed_overlap_m >= -distance_epsilon
        ),
        "center_distance_m": float(math.hypot(sx - ox, sy - oy)),
        "heading_difference_rad": float(
            angular_difference(subject_heading, object_heading)
        ),
        "footprints_overlap": bool(intersection_area > overlap_epsilon),
        "footprint_intersection_area_m2": float(intersection_area),
    }


def _compute_overtake_forward_speeds(
    observations: list[tuple[int, dict[str, Any]]]
) -> list[Optional[float]]:
    speeds: list[Optional[float]] = [None] * len(observations)
    for index in range(1, len(observations)):
        previous_time, previous = observations[index - 1]
        current_time, current = observations[index]
        delta_time_s = (int(current_time) - int(previous_time)) / 1e6
        if delta_time_s <= 0.0:
            continue
        try:
            px, py, ph = map(float, previous["xyh"])
            cx, cy, ch = map(float, current["xyh"])
        except Exception:
            continue
        average_heading = _overtake_circular_mean(ph, ch)
        forward_displacement = (
            math.cos(average_heading) * (cx - px)
            + math.sin(average_heading) * (cy - py)
        )
        speeds[index] = float(forward_displacement / delta_time_s)
    if len(speeds) > 1 and speeds[0] is None and speeds[1] is not None:
        speeds[0] = speeds[1]
    return speeds


def _map_object_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    identifier = getattr(value, "id", value)
    return str(identifier) if identifier is not None else None


def _safe_parent(map_object: Any) -> Any:
    value = getattr(map_object, "parent", None)
    try:
        return value() if callable(value) else value
    except Exception:
        return None


def _roadblock_id(map_object: Any) -> Optional[str]:
    try:
        value = map_object.get_roadblock_id()
        if value is not None:
            return str(value)
    except Exception:
        pass
    return _map_object_id(_safe_parent(map_object))


def _intersection_id(map_object: Any) -> Optional[str]:
    for candidate in (map_object, _safe_parent(map_object)):
        if candidate is None:
            continue
        value = getattr(candidate, "intersection", None)
        try:
            value = value() if callable(value) else value
        except Exception:
            value = None
        identifier = _map_object_id(value)
        if identifier:
            return identifier
    return None


def _edge_objects(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        value = value() if callable(value) else value
    except Exception:
        return []
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None]
    try:
        return [item for item in list(value) if item is not None]
    except TypeError:
        return [value]


def _edge_ids(value: Any) -> set[str]:
    return {
        identifier
        for item in _edge_objects(value)
        for identifier in [_map_object_id(item)]
        if identifier is not None
    }


def _outgoing_lane_objects(lane: Any) -> list[Any]:
    objects = []
    for attribute in (
        "outgoing_edges",
        "outgoing_edge",
        "next_edges",
        "successors",
    ):
        objects.extend(_edge_objects(getattr(lane, attribute, None)))
    unique = {}
    for item in objects:
        identifier = _map_object_id(item)
        if identifier:
            unique[identifier] = item
    return list(unique.values())


def _adjacent_lane_ids(lane: Any) -> tuple[set[str], set[str]]:
    left: set[str] = set()
    right: set[str] = set()
    try:
        adjacent = getattr(lane, "adjacent_edges")
        adjacent = adjacent() if callable(adjacent) else adjacent
        if adjacent and len(adjacent) >= 2:
            if adjacent[0] is not None:
                left.add(str(adjacent[0].id))
            if adjacent[1] is not None:
                right.add(str(adjacent[1].id))
    except Exception:
        pass
    for attribute, target in (
        ("left_adjacent_edge", left),
        ("adjacent_left", left),
        ("right_adjacent_edge", right),
        ("adjacent_right", right),
    ):
        target.update(_edge_ids(getattr(lane, attribute, None)))
    # Package map records already expose the same official adjacency IDs.
    left.update(str(value) for value in getattr(lane, "left_adjacent_ids", set()) or set())
    right.update(str(value) for value in getattr(lane, "right_adjacent_ids", set()) or set())
    return left, right


def _baseline_measurements(
    lane: Any, x: float, y: float
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    baseline = getattr(lane, "baseline_path", None)
    if baseline is None:
        return None, None, None
    try:
        point = Point2D(float(x), float(y))
        progress = float(baseline.get_nearest_arc_length_from_position(point))
        pose = baseline.get_nearest_pose_from_position(point)
        heading = wrap_signed(float(pose.heading))
        dx = float(x) - float(pose.x)
        dy = float(y) - float(pose.y)
        lateral = -math.sin(heading) * dx + math.cos(heading) * dy
        return progress, float(lateral), heading
    except Exception:
        return None, None, None


def _fallback_lane_match(record: dict[str, Any]) -> dict[str, Any]:
    """Resolve the selected drivable primitive from extractor map evidence.

    Overtake reasoning in v9.5.31 deliberately uses the same Lane +
    LaneConnector identity model as the lane-change decoder.  ``lane_id`` is
    retained as the raw nuPlan token for backward-compatible evidence, while
    ``map_entity_id`` is the typed identity used for topology/corridor logic.
    """
    entity_id = _lane_change_lane_id(record)
    kind, raw_id = _lane_change_split_entity_id(entity_id)
    lane_object = None if entity_id is None else _lane_change_map_object(record, entity_id)

    matches = _lane_change_all_matches(record)
    if entity_id is None and matches:
        ranked = sorted(
            matches,
            key=lambda row: (
                bool(row.get("primary")),
                bool(row.get("center_covered")),
                float(row.get("overlap_ratio") or 0.0),
                float(row.get("occupancy_score") or 0.0),
            ),
            reverse=True,
        )
        best = ranked[0]
        entity_id = str(best.get("map_entity_id") or best.get("lane_id"))
        kind, raw_id = _lane_change_split_entity_id(entity_id)
        lane_object = best.get("lane_object")

    candidate_entity_ids = {
        str(row.get("map_entity_id") or row.get("lane_id"))
        for row in matches
        if row.get("map_entity_id") or row.get("lane_id")
    }
    candidate_raw_ids = {
        str(_lane_change_split_entity_id(value)[1] or value)
        for value in candidate_entity_ids
    }
    left_ids = {
        str(value).split(":", 1)[1]
        if str(value).startswith(("lane:", "lane_connector:"))
        else str(value)
        for value in record.get("left_adjacent_object_ids", set()) or set()
    }
    right_ids = {
        str(value).split(":", 1)[1]
        if str(value).startswith(("lane:", "lane_connector:"))
        else str(value)
        for value in record.get("right_adjacent_object_ids", set()) or set()
    }
    return {
        "lane_id": None if raw_id is None else str(raw_id),
        "map_entity_id": None if entity_id is None else str(entity_id),
        "map_kind": kind,
        "lane_object": lane_object,
        "ambiguous": bool(record.get("map_match_ambiguous", False)),
        "candidate_lane_ids": candidate_raw_ids,
        "candidate_map_entity_ids": candidate_entity_ids,
        "intersection_id": record.get("primary_map_intersection_id"),
        "left_adjacent_lane_ids": left_ids,
        "right_adjacent_lane_ids": right_ids,
        "outgoing_lane_ids": _edge_ids(getattr(lane_object, "outgoing_edges", None)),
        "roadblock_id": record.get("primary_map_roadblock_id"),
        "progress_m": record.get("map_progress_m"),
        "lateral_offset_m": record.get("map_lateral_offset_m"),
        "map_heading_rad": record.get("map_heading_rad"),
    }

def _overtake_lane_match(
    snapshot: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Return one Lane/LaneConnector map state for strict overtake reasoning.

    Extractor-provided primary/candidate map evidence is preferred because it
    is exactly the evidence used by ``np:changesLane``.  A direct map query is
    only a fallback when that evidence is unavailable.
    """
    selected = _fallback_lane_match(record)
    if selected.get("map_entity_id") is not None and selected.get("lane_object") is not None:
        return selected

    frame = snapshot["frame"]
    map_api = getattr(frame, "map_api", None)
    query_supported = bool(
        map_api is not None
        and hasattr(map_api, "get_proximal_map_objects")
        and "Point2D" in globals()
        and "Point" in globals()
    )
    if not query_supported:
        return selected

    try:
        x, y, _ = map(float, record["xyh"])
        point = Point2D(x, y)
        layers = [SemanticMapLayer.LANE]
        connector_layer = getattr(SemanticMapLayer, "LANE_CONNECTOR", None)
        if connector_layer is not None:
            layers.append(connector_layer)
        nearby = map_api.get_proximal_map_objects(
            point,
            _overtake_arg("overtake_lane_query_radius_m", _NB_LANE_QUERY_RADIUS_M),
            layers,
        )
    except Exception:
        nearby = {}

    try:
        center_point = Point(x, y)
        footprint = oriented_box_polygon(record)
        footprint_area = max(float(footprint.area), 1e-6)
    except Exception:
        return selected

    minimum_overlap = _overtake_arg(
        "overtake_lane_min_overlap_ratio",
        _NB_MIN_LANE_FOOTPRINT_OVERLAP_RATIO,
    )
    candidates = []
    layer_kinds = [(SemanticMapLayer.LANE, "lane")]
    connector_layer = getattr(SemanticMapLayer, "LANE_CONNECTOR", None)
    if connector_layer is not None:
        layer_kinds.append((connector_layer, "lane_connector"))
    for layer, kind in layer_kinds:
        for lane in nearby.get(layer, []) or []:
            polygon = getattr(lane, "polygon", None)
            if polygon is None:
                continue
            try:
                center_covered = bool(polygon.covers(center_point))
                intersection_area = float(polygon.intersection(footprint).area)
            except Exception:
                continue
            overlap_ratio = intersection_area / footprint_area
            if not center_covered and overlap_ratio < minimum_overlap:
                continue
            progress, lateral, map_heading = _baseline_measurements(lane, x, y)
            left_ids, right_ids = _adjacent_lane_ids(lane)
            entity_id = _lane_change_entity_id(kind, getattr(lane, "id", None))
            candidates.append({
                "lane_id": str(lane.id),
                "map_entity_id": entity_id,
                "map_kind": kind,
                "lane_object": lane,
                "center_covered": center_covered,
                "overlap_ratio": overlap_ratio,
                "progress_m": progress,
                "lateral_offset_m": lateral,
                "map_heading_rad": map_heading,
                "roadblock_id": _roadblock_id(lane),
                "intersection_id": _intersection_id(lane),
                "left_adjacent_lane_ids": left_ids,
                "right_adjacent_lane_ids": right_ids,
                "outgoing_lane_ids": _edge_ids(getattr(lane, "outgoing_edges", None)),
            })

    candidates.sort(
        key=lambda row: (
            bool(row["center_covered"]),
            float(row["overlap_ratio"]),
            -abs(float(row["lateral_offset_m"]))
            if row["lateral_offset_m"] is not None
            else -1e9,
        ),
        reverse=True,
    )
    if not candidates:
        return selected

    selected = dict(candidates[0])
    ambiguous = False
    if len(candidates) > 1:
        first, second = candidates[:2]
        same_center_class = bool(first["center_covered"]) == bool(second["center_covered"])
        close_overlap = abs(
            float(first["overlap_ratio"]) - float(second["overlap_ratio"])
        ) < _overtake_arg(
            "overtake_lane_ambiguity_margin",
            _NB_LANE_AMBIGUITY_OVERLAP_MARGIN,
        )
        ambiguous = bool(same_center_class and close_overlap)
    selected["ambiguous"] = ambiguous
    selected["candidate_lane_ids"] = {row["lane_id"] for row in candidates}
    selected["candidate_map_entity_ids"] = {
        row["map_entity_id"] for row in candidates if row.get("map_entity_id")
    }
    return selected

def _overtake_corridor_match(lane: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not lane:
        return None
    entity_id = lane.get("map_entity_id")
    if entity_id is None:
        kind = lane.get("map_kind") or "lane"
        entity_id = _lane_change_entity_id(kind, lane.get("lane_id"))
    if entity_id is None or lane.get("lane_object") is None:
        return None
    return {
        "lane_id": str(entity_id),
        "lane_object": lane.get("lane_object"),
    }


def _lanes_are_adjacent(
    subject_lane: dict[str, Any],
    object_lane: dict[str, Any],
    object_record: Optional[dict[str, Any]] = None,
) -> tuple[bool, Optional[str]]:
    """Return whether subject occupies a lateral corridor beside object.

    The side is expressed from the object's path: ``left`` means the subject
    is on the object's left.  The lane-change decoder's adjacency machinery is
    reused so LaneConnector-to-LaneConnector and mixed Lane/Connector cases are
    handled consistently.
    """
    subject_match = _overtake_corridor_match(subject_lane)
    object_match = _overtake_corridor_match(object_lane)
    if subject_match is None or object_match is None:
        return False, None
    if _lane_change_same_longitudinal_corridor(subject_match, object_match):
        return False, None
    side, _ = _lane_change_adjacency_side(
        object_record or {},
        object_match["lane_object"],
        subject_match["lane_object"],
        subject_match["lane_id"],
    )
    if side is not None:
        return True, str(side)
    reverse_side, _ = _lane_change_adjacency_side(
        {},
        subject_match["lane_object"],
        object_match["lane_object"],
        object_match["lane_id"],
    )
    if reverse_side == "left":
        return True, "right"
    if reverse_side == "right":
        return True, "left"
    return False, None

def _lane_reachable_forward(
    start_lane_object: Any,
    target_lane_id: Optional[str],
    max_hops: Optional[int] = None,
) -> bool:
    if start_lane_object is None or target_lane_id is None:
        return False
    start_id = _map_object_id(start_lane_object)
    target_id = str(target_lane_id)
    if start_id == target_id:
        return True
    if max_hops is None:
        max_hops = _overtake_int_arg(
            "overtake_max_lane_path_hops", _NB_MAX_LANE_PATH_HOPS
        )
    frontier = [start_lane_object]
    visited = {start_id} if start_id is not None else set()
    for _ in range(int(max_hops)):
        next_frontier = []
        for lane in frontier:
            for successor in _outgoing_lane_objects(lane):
                successor_id = _map_object_id(successor)
                if successor_id is None:
                    continue
                if successor_id == target_id:
                    return True
                if successor_id in visited:
                    continue
                visited.add(successor_id)
                next_frontier.append(successor)
        if not next_frontier:
            break
        frontier = next_frontier
    return False


def _lanes_belong_to_same_longitudinal_path(
    first_lane: dict[str, Any], second_lane: dict[str, Any]
) -> bool:
    """True when two selected primitives are the same directed corridor.

    This delegates to the Lane/LaneConnector topology used by the lane-change
    decoder rather than assuming that a corridor consists only of Lane IDs.
    """
    first = _overtake_corridor_match(first_lane)
    second = _overtake_corridor_match(second_lane)
    return bool(_lane_change_same_longitudinal_corridor(first, second))


def _lane_transition_is_lateral(
    previous_lane: dict[str, Any],
    current_lane: dict[str, Any],
    reference_record: Optional[dict[str, Any]] = None,
) -> bool:
    previous = _overtake_corridor_match(previous_lane)
    current = _overtake_corridor_match(current_lane)
    if previous is None or current is None:
        return False
    if _lane_change_same_longitudinal_corridor(previous, current):
        return False
    side, _ = _lane_change_adjacency_side(
        reference_record or {},
        previous["lane_object"],
        current["lane_object"],
        current["lane_id"],
    )
    return side is not None

def _collect_vehicle_observations_by_snapshot(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    observations = []
    for snapshot in snapshots:
        frame_observations = {}
        for token, record in snapshot.get("entities", {}).items():
            if _is_overtake_vehicle(record):
                frame_observations[str(token)] = record
        observations.append(frame_observations)
    return observations


def _nearby_pair_common_frames(
    observations_by_frame: list[dict[str, dict[str, Any]]],
) -> dict[tuple[str, str], list[int]]:
    radius = _overtake_arg("overtake_pair_query_radius_m", _NB_PAIR_QUERY_RADIUS_M)
    minimum_frames = _overtake_int_arg("overtake_min_common_frames", _NB_MIN_COMMON_FRAMES)
    pair_frames: dict[tuple[str, str], list[int]] = defaultdict(list)
    for frame_index, observations in enumerate(observations_by_frame):
        tokens = sorted(observations)
        for first_index, first_token in enumerate(tokens):
            first = observations[first_token]
            try:
                first_x, first_y, _ = map(float, first["xyh"])
            except Exception:
                continue
            for second_token in tokens[first_index + 1:]:
                second = observations[second_token]
                try:
                    second_x, second_y, _ = map(float, second["xyh"])
                except Exception:
                    continue
                if math.hypot(first_x - second_x, first_y - second_y) <= radius:
                    pair_frames[(first_token, second_token)].append(frame_index)
    return {
        pair: sorted(set(frame_indices))
        for pair, frame_indices in pair_frames.items()
        if len(set(frame_indices)) >= minimum_frames
    }


def _aligned_pair_records(
    snapshots: list[dict[str, Any]],
    observations_by_frame: list[dict[str, dict[str, Any]]],
    common_frame_indices: list[int],
    subject_token: str,
    object_token: str,
) -> list[dict[str, Any]]:
    records = []
    for frame_index in common_frame_indices:
        observations = observations_by_frame[frame_index]
        subject = observations.get(subject_token)
        obj = observations.get(object_token)
        if subject is None or obj is None:
            continue
        geometry = _overtake_pair_geometry(subject, obj)
        if geometry is None:
            continue
        records.append({
            "snapshot": snapshots[frame_index],
            "frame_index": int(snapshots[frame_index].get("frame_index", frame_index)),
            "timestamp_us": int(snapshots[frame_index]["timestamp_us"]),
            "subject": subject,
            "object": obj,
            **geometry,
        })
    if not records:
        return records
    subject_speeds = _compute_overtake_forward_speeds([
        (row["timestamp_us"], row["subject"]) for row in records
    ])
    object_speeds = _compute_overtake_forward_speeds([
        (row["timestamp_us"], row["object"]) for row in records
    ])
    for index, row in enumerate(records):
        row["subject_speed_mps"] = subject_speeds[index]
        row["object_speed_mps"] = object_speeds[index]
        row["relative_speed_mps"] = (
            None
            if subject_speeds[index] is None or object_speeds[index] is None
            else float(subject_speeds[index] - object_speeds[index])
        )
    return records


def _best_broad_case1_candidate(
    records: list[dict[str, Any]],
    subject_token: str,
    object_token: str,
) -> Optional[dict[str, Any]]:
    """Return the best high-recall fully observed same-lane-start candidate."""
    minimum_frames = _overtake_int_arg(
        "overtake_min_common_frames", _NB_MIN_COMMON_FRAMES
    )
    if len(records) < minimum_frames:
        return None
    timestamps = [int(row["timestamp_us"]) for row in records]
    if len(timestamps) > 1 and max(
        (second - first) / 1e6
        for first, second in zip(timestamps, timestamps[1:])
    ) > _NB_BROAD_MAX_FRAME_GAP_S:
        return None

    best = None
    for start_index, start in enumerate(records):
        initial_signed_lateral = float(start["subject_center_lateral_m"])
        initial_lateral = abs(initial_signed_lateral)
        subject_speed = start.get("subject_speed_mps")
        object_speed = start.get("object_speed_mps")
        if start["subject_behind_gap_m"] < _NB_BROAD_MIN_INITIAL_BEHIND_GAP_M:
            continue
        if start["heading_difference_rad"] > _NB_BROAD_SAME_FLOW_THRESHOLD_RAD:
            continue
        if subject_speed is None or subject_speed < _NB_BROAD_MIN_SUBJECT_SPEED_MPS:
            continue
        if object_speed is None or object_speed < _NB_BROAD_MIN_OBJECT_SPEED_MPS:
            continue
        if initial_lateral > _NB_BROAD_CASE1_MAX_INITIAL_LATERAL_M:
            continue

        side_index = None
        maximum_lateral_excursion = 0.0
        for index in range(start_index, len(records)):
            row = records[index]
            lateral_excursion = abs(
                float(row["subject_center_lateral_m"]) - initial_signed_lateral
            )
            maximum_lateral_excursion = max(
                maximum_lateral_excursion, lateral_excursion
            )
            side_valid = bool(
                row["longitudinal_intervals_overlap"]
                and abs(row["subject_center_lateral_m"])
                >= _NB_BROAD_MIN_SIDE_BY_SIDE_LATERAL_M
                and abs(row["subject_center_longitudinal_m"])
                <= _NB_BROAD_MAX_SIDE_BY_SIDE_CENTER_LONGITUDINAL_M
                and row["heading_difference_rad"]
                <= _NB_BROAD_SAME_FLOW_THRESHOLD_RAD
                and lateral_excursion >= _NB_BROAD_CASE1_MIN_LATERAL_EXCURSION_M
            )
            if side_valid:
                side_index = index
                break
        if side_index is None:
            continue

        reversal_index = next((
            index
            for index in range(side_index, len(records))
            if records[index]["subject_center_longitudinal_m"] > 0.0
        ), None)
        if reversal_index is None:
            continue
        clearance_index = next((
            index
            for index in range(reversal_index, len(records))
            if records[index]["subject_clearance_gap_m"] >= 0.0
        ), None)
        if clearance_index is None:
            continue

        return_index = None
        for index in range(clearance_index, len(records)):
            row = records[index]
            final_signed_lateral = float(row["subject_center_lateral_m"])
            final_lateral = abs(final_signed_lateral)
            if final_lateral > _NB_BROAD_FINAL_SAME_LANE_LATERAL_M:
                continue
            if row["subject_clearance_gap_m"] < 0.0:
                continue
            if abs(
                final_signed_lateral - initial_signed_lateral
            ) > _NB_BROAD_CASE1_MAX_RETURN_LATERAL_DELTA_M:
                continue
            return_index = index
            break
        if return_index is None:
            continue

        event_records = records[start_index:return_index + 1]
        event_duration_s = (
            event_records[-1]["timestamp_us"]
            - event_records[0]["timestamp_us"]
        ) / 1e6
        if (
            event_duration_s <= 0.0
            or event_duration_s > _NB_BROAD_MAX_EVENT_DURATION_S
        ):
            continue
        flow_fraction = sum(
            row["heading_difference_rad"]
            <= _NB_BROAD_SAME_FLOW_THRESHOLD_RAD
            for row in event_records
        ) / len(event_records)
        if flow_fraction < _NB_BROAD_MIN_SAME_FLOW_FRACTION:
            continue
        relative_speeds = [
            row["relative_speed_mps"]
            for row in event_records
            if row.get("relative_speed_mps") is not None
        ]
        maximum_relative_speed = (
            max(relative_speeds) if relative_speeds else float("-inf")
        )
        if maximum_relative_speed < _NB_BROAD_MIN_RELATIVE_SPEED_MPS:
            continue

        completion = records[return_index]
        final_lateral = abs(float(completion["subject_center_lateral_m"]))
        lateral_convergence = initial_lateral - final_lateral
        score = 0.0
        score += min(20.0, 4.0 * max(0.0, start["subject_behind_gap_m"]))
        score += min(20.0, 8.0 * max(0.0, maximum_lateral_excursion))
        score += 15.0
        score += min(
            15.0,
            5.0 * max(0.0, completion["subject_clearance_gap_m"] + 1.0),
        )
        score += min(15.0, 10.0 * max(0.0, maximum_relative_speed))
        score += 10.0 * flow_fraction
        score += 5.0 * max(
            0.0,
            1.0
            - final_lateral
            / max(_NB_BROAD_FINAL_SAME_LANE_LATERAL_M, 1e-6),
        )
        candidate = {
            "broad_case_hint": OVERTAKE_CASE_1,
            "subject_track_token": str(subject_token),
            "object_track_token": str(object_token),
            "relation_family": _overtake_relation_family(
                subject_token, object_token
            ),
            "start_time_us": int(records[start_index]["timestamp_us"]),
            "end_time_us": int(records[return_index]["timestamp_us"]),
            "start_frame_index": int(records[start_index]["frame_index"]),
            "side_by_side_frame_index": int(
                records[side_index]["frame_index"]
            ),
            "order_reversal_frame_index": int(
                records[reversal_index]["frame_index"]
            ),
            "clearance_frame_index": int(
                records[clearance_index]["frame_index"]
            ),
            "approximate_return_frame_index": int(
                records[return_index]["frame_index"]
            ),
            "initial_behind_gap_m": float(start["subject_behind_gap_m"]),
            "initial_lateral_offset_m": initial_signed_lateral,
            "final_lateral_offset_m": float(
                completion["subject_center_lateral_m"]
            ),
            "maximum_lateral_excursion_m": float(
                maximum_lateral_excursion
            ),
            "lateral_convergence_m": float(lateral_convergence),
            "maximum_relative_speed_mps": float(maximum_relative_speed),
            "same_flow_fraction": float(flow_fraction),
            "broad_event_duration_s": float(event_duration_s),
            "candidate_score": float(score),
            "candidate_rule": (
                "full_mini_case1_same_lane_depart_pass_return_v1_0_9"
            ),
        }
        if best is None or candidate["candidate_score"] > best["candidate_score"]:
            best = candidate
    return best


def _broad_overtake_candidates(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations_by_frame = _collect_vehicle_observations_by_snapshot(snapshots)
    pair_frames = _nearby_pair_common_frames(observations_by_frame)
    candidates = []
    for (first_token, second_token), common_frames in pair_frames.items():
        for subject_token, object_token in (
            (first_token, second_token),
            (second_token, first_token),
        ):
            records = _aligned_pair_records(
                snapshots,
                observations_by_frame,
                common_frames,
                subject_token,
                object_token,
            )
            candidate = _best_broad_case1_candidate(
                records, subject_token, object_token
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _collect_strict_pair_records(
    snapshots: list[dict[str, Any]], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    subject_token = str(candidate["subject_track_token"])
    object_token = str(candidate["object_track_token"])
    padding_s = _overtake_arg("overtake_search_padding_s", _NB_SEARCH_PADDING_S)
    start_us = int(candidate["start_time_us"] - padding_s * 1e6)
    end_us = int(candidate["end_time_us"] + padding_s * 1e6)
    records = []
    for snapshot in snapshots:
        timestamp_us = int(snapshot["timestamp_us"])
        if timestamp_us < start_us or timestamp_us > end_us:
            continue
        subject = snapshot.get("entities", {}).get(subject_token)
        obj = snapshot.get("entities", {}).get(object_token)
        if subject is None or obj is None:
            continue
        geometry = _overtake_pair_geometry(subject, obj)
        if geometry is None:
            continue
        records.append({
            "snapshot": snapshot,
            "frame": snapshot["frame"],
            "frame_index": int(snapshot.get("frame_index", len(records))),
            "timestamp_us": timestamp_us,
            "subject": subject,
            "object": obj,
            "subject_token": subject_token,
            "object_token": object_token,
            "subject_lane": _overtake_lane_match(snapshot, subject),
            "object_lane": _overtake_lane_match(snapshot, obj),
            **geometry,
        })
    if not records:
        return records
    subject_speeds = _compute_overtake_forward_speeds([
        (row["timestamp_us"], row["subject"]) for row in records
    ])
    object_speeds = _compute_overtake_forward_speeds([
        (row["timestamp_us"], row["object"]) for row in records
    ])
    for index, record in enumerate(records):
        record["subject_speed_mps"] = subject_speeds[index]
        record["object_speed_mps"] = object_speeds[index]
        record["relative_speed_mps"] = (
            None
            if subject_speeds[index] is None or object_speeds[index] is None
            else float(subject_speeds[index] - object_speeds[index])
        )
    return records


def _annotate_object_original_lane_status(
    records: list[dict[str, Any]],
    start_index: int,
    original_object_lane: dict[str, Any],
    initial_subject_passing_lane: dict[str, Any],
) -> tuple[Optional[int], Optional[str]]:
    """Track whether O remains in its original *logical corridor*.

    Lane -> LaneConnector -> Lane progression is accepted when each observed
    primitive is a directed longitudinal continuation.  A lateral transfer by
    O still rejects the strict Case-1 overtake because the object must
    preserve its stream while S passes it.
    """
    del initial_subject_passing_lane  # Retained in the signature for compatibility.
    previous_object_lane = original_object_lane
    first_failure_index = None
    failure_reason = None
    for index in range(start_index, len(records)):
        row = records[index]
        current_object_lane = row["object_lane"]
        row["object_on_original_lane_path"] = False
        row["object_original_lane_failure_reason"] = None
        if first_failure_index is not None:
            row["object_original_lane_failure_reason"] = failure_reason
            continue
        if current_object_lane.get("map_entity_id") is None:
            first_failure_index = index
            failure_reason = "object_corridor_unresolved"
        elif current_object_lane.get("ambiguous", False):
            first_failure_index = index
            failure_reason = "object_corridor_ambiguous"
        elif index == start_index:
            if not _lanes_belong_to_same_longitudinal_path(
                original_object_lane, current_object_lane
            ):
                first_failure_index = index
                failure_reason = "object_not_on_original_corridor_at_start"
        elif _lanes_belong_to_same_longitudinal_path(
            previous_object_lane, current_object_lane
        ):
            previous_object_lane = current_object_lane
        elif _lane_transition_is_lateral(
            previous_object_lane, current_object_lane, row.get("object")
        ):
            first_failure_index = index
            failure_reason = "object_changed_corridor"
        else:
            first_failure_index = index
            failure_reason = "object_left_original_corridor"
        if first_failure_index is None:
            row["object_on_original_lane_path"] = True
        else:
            row["object_original_lane_failure_reason"] = failure_reason
    return first_failure_index, failure_reason


def _annotate_subject_passing_corridor_status(
    records: list[dict[str, Any]], departure_index: int
) -> None:
    """Follow S along a passing corridor across Lane/Connector primitives."""
    previous = None
    still_passing = True
    for index in range(departure_index, len(records)):
        row = records[index]
        current = row["subject_lane"]
        row["subject_on_passing_corridor"] = False
        if not still_passing:
            continue
        if current.get("map_entity_id") is None or current.get("ambiguous", False):
            still_passing = False
            continue
        if previous is None:
            row["subject_on_passing_corridor"] = True
            previous = current
            continue
        if _lanes_belong_to_same_longitudinal_path(previous, current):
            row["subject_on_passing_corridor"] = True
            previous = current
            continue
        # A transition back into O's current preserved stream ends the passing
        # corridor rather than being treated as map discontinuity.
        if (
            row.get("object_on_original_lane_path", False)
            and _lanes_belong_to_same_longitudinal_path(current, row["object_lane"])
        ):
            still_passing = False
            continue
        # Any other lateral or disconnected transition leaves the strict
        # one-passing-corridor Case-1 model.
        still_passing = False

def _consecutive_stable_original_lane_completion(
    records: list[dict[str, Any]], return_start_index: int
) -> tuple[Optional[int], Optional[int]]:
    """Confirm stable post-pass occupancy of the same logical corridor."""
    stable_start = None
    stable_count = 0
    minimum_clearance = _overtake_arg(
        "minimum_overtake_clearance_m", _NB_MIN_FULL_CLEARANCE_M
    )
    same_flow_threshold = _overtake_arg(
        "overtake_same_flow_threshold_rad", _NB_SAME_FLOW_THRESHOLD_RAD
    )
    minimum_frames = _overtake_int_arg(
        "overtake_lane_stability_frames", _NB_MIN_FINAL_STABLE_FRAMES
    )
    minimum_duration = _overtake_arg(
        "minimum_overtake_completion_duration_s",
        _NB_MIN_FINAL_STABLE_DURATION_S,
    )
    for index in range(return_start_index, len(records)):
        row = records[index]
        subject_lane = row["subject_lane"]
        object_lane = row["object_lane"]
        valid = bool(
            row.get("object_on_original_lane_path", False)
            and subject_lane.get("map_entity_id") is not None
            and object_lane.get("map_entity_id") is not None
            and _lanes_belong_to_same_longitudinal_path(subject_lane, object_lane)
            and not subject_lane.get("ambiguous", False)
            and not object_lane.get("ambiguous", False)
            and row["subject_clearance_gap_m"] >= minimum_clearance
            and row["heading_difference_rad"] <= same_flow_threshold
        )
        if not valid:
            stable_start = None
            stable_count = 0
            continue
        if stable_start is None:
            stable_start = index
        stable_count += 1
        duration_s = (
            records[index]["timestamp_us"] - records[stable_start]["timestamp_us"]
        ) / 1e6
        if stable_count >= minimum_frames and duration_s >= minimum_duration:
            return index, stable_start
    return None, None

def _valid_strict_lane_state(row: dict[str, Any]) -> bool:
    """Require unambiguous drivable primitives, including LaneConnectors."""
    subject_lane = row["subject_lane"]
    object_lane = row["object_lane"]
    return bool(
        subject_lane.get("map_entity_id") is not None
        and object_lane.get("map_entity_id") is not None
        and not subject_lane.get("ambiguous", False)
        and not object_lane.get("ambiguous", False)
    )

def _valid_strict_initial_motion(row: dict[str, Any]) -> bool:
    subject_speed = row.get("subject_speed_mps")
    object_speed = row.get("object_speed_mps")
    return bool(
        row["subject_behind_gap_m"]
        >= _overtake_arg(
            "minimum_overtake_initial_behind_gap_m",
            _NB_MIN_INITIAL_BEHIND_GAP_M,
        )
        and row["heading_difference_rad"]
        <= _overtake_arg(
            "overtake_same_flow_threshold_rad",
            _NB_SAME_FLOW_THRESHOLD_RAD,
        )
        and subject_speed is not None
        and subject_speed
        >= _overtake_arg(
            "minimum_overtake_subject_speed_mps", _NB_MIN_SUBJECT_SPEED_MPS
        )
        and object_speed is not None
        and object_speed
        >= _overtake_arg(
            "minimum_overtaken_vehicle_speed_mps", _NB_MIN_OBJECT_SPEED_MPS
        )
    )


def _evaluate_strict_case1_overtake(
    candidate: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Strictly verify a fully observed same-lane-start overtake."""
    result = {
        **candidate,
        "overtake_case": OVERTAKE_CASE_1,
        "overtake_case_number": 1,
        "overtake_case_label": _OVERTAKE_CASE_LABELS[OVERTAKE_CASE_1],
        "overtake_detected": False,
        "overtake_rejection_reason": None,
    }
    minimum_frames = _overtake_int_arg(
        "overtake_min_common_frames", _NB_MIN_COMMON_FRAMES
    )
    if len(records) < minimum_frames:
        result["overtake_rejection_reason"] = "insufficient_common_frames"
        return result
    timestamps = [int(row["timestamp_us"]) for row in records]
    if len(timestamps) > 1 and max(
        (second - first) / 1e6
        for first, second in zip(timestamps, timestamps[1:])
    ) > _overtake_arg("overtake_max_frame_gap_s", _NB_MAX_FRAME_GAP_S):
        result["overtake_rejection_reason"] = "discontinuous_pair_track"
        return result

    start_index = None
    departure_index = None
    passing_side = None
    original_object_lane = None
    passing_lane = None

    for index, row in enumerate(records):
        if not _valid_strict_lane_state(row) or not _valid_strict_initial_motion(row):
            continue
        if not _lanes_belong_to_same_longitudinal_path(
            row["subject_lane"], row["object_lane"]
        ):
            continue
        candidate_original_lane = row["object_lane"]
        previous_object_lane = candidate_original_lane
        for later_index in range(index + 1, len(records)):
            later = records[later_index]
            if not _valid_strict_lane_state(later):
                continue
            if not _lanes_belong_to_same_longitudinal_path(
                previous_object_lane, later["object_lane"]
            ):
                break
            previous_object_lane = later["object_lane"]
            adjacent, side = _lanes_are_adjacent(
                later["subject_lane"], later["object_lane"], later.get("object")
            )
            if adjacent:
                start_index = index
                departure_index = later_index
                passing_side = side
                original_object_lane = candidate_original_lane
                passing_lane = later["subject_lane"]
                break
        if start_index is not None:
            break
    if start_index is None:
        result["overtake_rejection_reason"] = (
            "no_initial_behind_same_lane_then_departure_state"
        )
        return result

    assert departure_index is not None
    assert original_object_lane is not None
    assert passing_lane is not None
    initial_lane_relation = "same_lane"

    object_failure_index, object_failure_reason = _annotate_object_original_lane_status(
        records, start_index, original_object_lane, passing_lane
    )
    _annotate_subject_passing_corridor_status(records, departure_index)

    minimum_side_lateral = _overtake_arg(
        "minimum_overtake_side_by_side_lateral_m",
        _NB_MIN_SIDE_BY_SIDE_LATERAL_M,
    )
    same_flow_threshold = _overtake_arg(
        "overtake_same_flow_threshold_rad", _NB_SAME_FLOW_THRESHOLD_RAD
    )
    side_index = None
    for index in range(departure_index, len(records)):
        row = records[index]
        if not row.get("object_on_original_lane_path", False):
            break
        subject_on_passing_path = bool(
            row.get("subject_on_passing_corridor", False)
        )
        if (
            subject_on_passing_path
            and row["longitudinal_intervals_overlap"]
            and abs(row["subject_center_lateral_m"]) >= minimum_side_lateral
            and row["heading_difference_rad"] <= same_flow_threshold
        ):
            side_index = index
            break
    if side_index is None:
        result["overtake_rejection_reason"] = (
            object_failure_reason
            if object_failure_index is not None
            else "no_side_by_side_phase_in_passing_lane"
        )
        return result

    reversal_index = next((
        index
        for index in range(side_index, len(records))
        if records[index].get("object_on_original_lane_path", False)
        and records[index]["subject_center_longitudinal_m"] > 0.0
    ), None)
    if reversal_index is None:
        result["overtake_rejection_reason"] = (
            object_failure_reason or "no_longitudinal_order_reversal"
        )
        return result

    minimum_clearance = _overtake_arg(
        "minimum_overtake_clearance_m", _NB_MIN_FULL_CLEARANCE_M
    )
    clearance_index = next((
        index
        for index in range(reversal_index, len(records))
        if records[index].get("object_on_original_lane_path", False)
        and records[index]["subject_clearance_gap_m"] >= minimum_clearance
    ), None)
    if clearance_index is None:
        result["overtake_rejection_reason"] = (
            object_failure_reason or "no_full_longitudinal_clearance"
        )
        return result

    event_until_clearance = records[start_index:clearance_index + 1]
    relative_speeds = [
        row["relative_speed_mps"]
        for row in event_until_clearance
        if row.get("relative_speed_mps") is not None
    ]
    maximum_relative_speed = (
        max(relative_speeds) if relative_speeds else float("-inf")
    )
    if maximum_relative_speed < _overtake_arg(
        "minimum_overtake_relative_speed_mps", _NB_MIN_RELATIVE_SPEED_MPS
    ):
        result["overtake_rejection_reason"] = (
            "insufficient_positive_relative_speed"
        )
        return result

    flow_fraction = sum(
        row["heading_difference_rad"] <= same_flow_threshold
        for row in event_until_clearance
    ) / len(event_until_clearance)
    if flow_fraction < _overtake_arg(
        "minimum_overtake_same_flow_fraction", _NB_MIN_SAME_FLOW_FRACTION
    ):
        result["overtake_rejection_reason"] = "insufficient_same_flow_fraction"
        return result

    return_index = next((
        index
        for index in range(clearance_index, len(records))
        if records[index].get("object_on_original_lane_path", False)
        and _lanes_belong_to_same_longitudinal_path(
            records[index]["subject_lane"], records[index]["object_lane"]
        )
    ), None)
    if return_index is None:
        result["overtake_rejection_reason"] = (
            object_failure_reason or "subject_did_not_return_to_original_lane"
        )
        return result

    completion_index, stable_start_index = (
        _consecutive_stable_original_lane_completion(records, return_index)
    )
    if completion_index is None or stable_start_index is None:
        result["overtake_rejection_reason"] = (
            object_failure_reason or "return_to_original_lane_not_stable"
        )
        return result

    event_duration_s = (
        records[completion_index]["timestamp_us"]
        - records[start_index]["timestamp_us"]
    ) / 1e6
    if event_duration_s > _overtake_arg(
        "maximum_overtake_duration_s", _NB_MAX_EVENT_DURATION_S
    ):
        result["overtake_rejection_reason"] = "event_duration_too_long"
        return result

    completion = records[completion_index]
    passing_lane_frame_count = sum(
        bool(records[index].get("subject_on_passing_corridor", False))
        for index in range(departure_index, return_index + 1)
    )
    final_stable_frame_count = completion_index - stable_start_index + 1
    maximum_longitudinal_gain = max(
        row["subject_center_longitudinal_m"]
        for row in records[start_index:completion_index + 1]
    )
    result.update({
        "overtake_detected": True,
        "overtake_rejection_reason": "accepted",
        "overtake_initial_lane_relation": initial_lane_relation,
        "overtake_start_index": start_index,
        "overtake_departure_index": departure_index,
        "overtake_side_by_side_index": side_index,
        "overtake_order_reversal_index": reversal_index,
        "overtake_clearance_index": clearance_index,
        "overtake_return_index": return_index,
        "overtake_stable_start_index": stable_start_index,
        "overtake_completion_index": completion_index,
        "overtake_start_frame_index": records[start_index]["frame_index"],
        "overtake_departure_frame_index": records[departure_index]["frame_index"],
        "overtake_side_by_side_frame_index": records[side_index]["frame_index"],
        "overtake_order_reversal_frame_index": records[reversal_index]["frame_index"],
        "overtake_clearance_frame_index": records[clearance_index]["frame_index"],
        "overtake_return_frame_index": records[return_index]["frame_index"],
        "overtake_completion_frame_index": completion["frame_index"],
        "overtake_start_time_us": records[start_index]["timestamp_us"],
        "overtake_departure_time_us": records[departure_index]["timestamp_us"],
        "overtake_side_by_side_time_us": records[side_index]["timestamp_us"],
        "overtake_order_reversal_time_us": records[reversal_index]["timestamp_us"],
        "overtake_clearance_time_us": records[clearance_index]["timestamp_us"],
        "overtake_return_time_us": records[return_index]["timestamp_us"],
        "overtake_stable_start_time_us": records[stable_start_index]["timestamp_us"],
        "overtake_completion_time_us": completion["timestamp_us"],
        "overtake_passing_side": passing_side,
        # Legacy lane fields are corridor anchors so old consumers continue to
        # see A -> B -> A even when the physical map sequence is L -> LC -> L.
        "overtake_original_lane_id": original_object_lane.get("lane_id"),
        "overtake_passing_lane_id": passing_lane.get("lane_id"),
        "overtake_initial_subject_lane_id": original_object_lane.get("lane_id"),
        "overtake_initial_object_lane_id": original_object_lane.get("lane_id"),
        "overtake_final_subject_lane_id": original_object_lane.get("lane_id"),
        "overtake_final_object_lane_id": original_object_lane.get("lane_id"),
        "overtake_original_map_entity_id": original_object_lane.get("map_entity_id"),
        "overtake_passing_map_entity_id": passing_lane.get("map_entity_id"),
        "overtake_initial_subject_map_entity_id": records[start_index]["subject_lane"].get("map_entity_id"),
        "overtake_initial_object_map_entity_id": records[start_index]["object_lane"].get("map_entity_id"),
        "overtake_final_subject_map_entity_id": completion["subject_lane"].get("map_entity_id"),
        "overtake_final_object_map_entity_id": completion["object_lane"].get("map_entity_id"),
        "overtake_original_map_kind": original_object_lane.get("map_kind"),
        "overtake_passing_map_kind": passing_lane.get("map_kind"),
        "overtake_final_subject_map_kind": completion["subject_lane"].get("map_kind"),
        "overtake_final_object_map_kind": completion["object_lane"].get("map_kind"),
        "overtake_subject_returned_to_original_lane": True,
        "overtake_object_remained_in_original_lane": True,
        "overtake_maximum_relative_speed_mps": float(maximum_relative_speed),
        "overtake_same_flow_fraction": float(flow_fraction),
        "overtake_event_duration_s": float(event_duration_s),
        "overtake_final_clearance_gap_m": float(completion["subject_clearance_gap_m"]),
        "overtake_initial_behind_gap_m": float(records[start_index]["subject_behind_gap_m"]),
        "overtake_maximum_longitudinal_gain_m": float(maximum_longitudinal_gain),
        "overtake_passing_lane_frame_count": int(passing_lane_frame_count),
        "overtake_final_stable_frame_count": int(final_stable_frame_count),
        "overtake_final_stable_duration_s": float(
            (
                completion["timestamp_us"]
                - records[stable_start_index]["timestamp_us"]
            ) / 1e6
        ),
        "overtake_intersection_involved": any(
            row["subject_lane"].get("intersection_id") is not None
            or row["object_lane"].get("intersection_id") is not None
            for row in records[start_index:completion_index + 1]
        ),
        "overtake_completion_record": completion,
    })
    return result


def _deduplicate_overtake_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tolerance_us = int(
        _overtake_arg(
            "overtake_positive_deduplication_tolerance_s",
            _NB_POSITIVE_DEDUPLICATION_TOLERANCE_S,
        ) * 1e6
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(
            str(result["subject_track_token"]),
            str(result["object_track_token"]),
        )].append(result)

    selected = []
    for group in grouped.values():
        group.sort(key=lambda row: int(row["overtake_completion_time_us"]))
        cluster = []
        previous_time = None
        for row in group:
            current_time = int(row["overtake_completion_time_us"])
            if previous_time is None or current_time - previous_time <= tolerance_us:
                cluster.append(row)
            else:
                selected.append(max(
                    cluster,
                    key=lambda item: (
                        float(item.get("candidate_score", 0.0)),
                        float(item.get("overtake_final_clearance_gap_m", 0.0)),
                        -float(item.get("overtake_event_duration_s", float("inf"))),
                    ),
                ))
                cluster = [row]
            previous_time = current_time
        if cluster:
            selected.append(max(
                cluster,
                key=lambda item: (
                    float(item.get("candidate_score", 0.0)),
                    float(item.get("overtake_final_clearance_gap_m", 0.0)),
                    -float(item.get("overtake_event_duration_s", float("inf"))),
                ),
            ))

    return sorted(
        selected,
        key=lambda row: int(row["overtake_completion_time_us"]),
    )


def _completed_case1_overtake_assertion(result: dict[str, Any]):
    completion = result["overtake_completion_record"]
    subject = completion["subject"]
    obj = completion["object"]
    start_us = int(result["overtake_start_time_us"])
    completion_us = int(result["overtake_completion_time_us"])
    evidence = {
        "overtake_strategy": "package_internal_two_stage_case1_corridor_v2",
        "overtake_corridor_model": "lane_and_lane_connector_directed_continuity_v9_5_31",
        "lane_and_lane_connector_primitives_supported": True,
        "longitudinal_lane_connector_progression_preserves_corridor": True,
        "lateral_lane_connector_passing_supported": True,
        "intersection_connector_frames_allowed_with_same_flow_guard": True,
        "legacy_lane_fields_are_corridor_anchors": True,
        "implementation_source": "nuplan_predicates_modular.categories.interaction",
        "supporting_predicates": [
            "np:inLane",
            "np:inLaneConnector",
            "np:intersectsLane",
            "np:intersectsLaneConnector",
            "np:hasPrimaryLane",
            "np:hasPrimaryLaneConnector",
            "np:hasSubjectForwardSpeed",
            "np:hasLongitudinalRelativeSpeedTo",
            "np:inFrontOf",
            "np:behind",
        ],
        "overtake_case": result["overtake_case"],
        "overtake_case_number": result["overtake_case_number"],
        "overtake_case_label": result["overtake_case_label"],
        "overtake_initial_lane_relation": result["overtake_initial_lane_relation"],
        "overtake_observation_mode": "fully_observed",
        "complete_event_observed": True,
        "lane_departure_observed": True,
        "overtake_entry_mode": "same_lane_departure",
        "initial_follow_observed": False,
        "initial_follow_mode": None,
        "overtake_completion_mode": "returned_to_original_lane",
        "relation_family": result["relation_family"],
        "subject_id": subject["entity_id"],
        "object_id": obj["entity_id"],
        "subject_track_token": result["subject_track_token"],
        "object_track_token": result["object_track_token"],
        "subject_agent_type": _normalized_agent_type(subject),
        "object_agent_type": _normalized_agent_type(obj),
        "initial_lane_id": result["overtake_original_lane_id"],
        "original_lane_id": result["overtake_original_lane_id"],
        "initial_subject_lane_id": result["overtake_initial_subject_lane_id"],
        "initial_object_lane_id": result["overtake_initial_object_lane_id"],
        "passing_lane_id": result["overtake_passing_lane_id"],
        "final_lane_id": result["overtake_final_subject_lane_id"],
        "final_object_lane_id": result["overtake_final_object_lane_id"],
        "original_map_entity_id": result.get("overtake_original_map_entity_id"),
        "passing_map_entity_id": result.get("overtake_passing_map_entity_id"),
        "initial_subject_map_entity_id": result.get("overtake_initial_subject_map_entity_id"),
        "initial_object_map_entity_id": result.get("overtake_initial_object_map_entity_id"),
        "final_subject_map_entity_id": result.get("overtake_final_subject_map_entity_id"),
        "final_object_map_entity_id": result.get("overtake_final_object_map_entity_id"),
        "original_map_kind": result.get("overtake_original_map_kind"),
        "passing_map_kind": result.get("overtake_passing_map_kind"),
        "final_subject_map_kind": result.get("overtake_final_subject_map_kind"),
        "final_object_map_kind": result.get("overtake_final_object_map_kind"),
        "passing_side": result["overtake_passing_side"],
        "returned_to_original_lane": True,
        "remained_in_passing_lane": False,
        "original_lane_preserved_by_object": True,
        "intersection_involved": bool(result.get("overtake_intersection_involved", False)),
        "start_time_us": start_us,
        "lane_departure_time_us": int(result["overtake_departure_time_us"]),
        "passing_lane_observation_start_time_us": int(result["overtake_departure_time_us"]),
        "side_by_side_time_us": int(result["overtake_side_by_side_time_us"]),
        "order_reversal_time_us": int(result["overtake_order_reversal_time_us"]),
        "clearance_time_us": int(result["overtake_clearance_time_us"]),
        "return_time_us": int(result["overtake_return_time_us"]),
        "final_order_stable_start_time_us": int(result["overtake_stable_start_time_us"]),
        "completion_time_us": completion_us,
        "condition_duration_s": float(result["overtake_event_duration_s"]),
        "approach_frame_count": int(
            result["overtake_departure_index"]
            - result["overtake_start_index"]
            + 1
        ),
        "departure_transition_frame_count": 0,
        "passing_lane_frame_count": int(result["overtake_passing_lane_frame_count"]),
        "return_transition_frame_count": 0,
        "return_frame_count": int(result["overtake_final_stable_frame_count"]),
        "final_order_frame_count": int(result["overtake_final_stable_frame_count"]),
        "initial_bumper_gap_m": float(result["overtake_initial_behind_gap_m"]),
        "final_clearance_gap_m": float(result["overtake_final_clearance_gap_m"]),
        "maximum_longitudinal_gain_m": float(result["overtake_maximum_longitudinal_gain_m"]),
        "maximum_relative_speed_mps": float(result["overtake_maximum_relative_speed_mps"]),
        "same_flow_fraction": float(result["overtake_same_flow_fraction"]),
        "final_stable_duration_s": float(result["overtake_final_stable_duration_s"]),
        "candidate_score": float(result.get("candidate_score", 0.0)),
        "broad_case_hint": result.get("broad_case_hint"),
        "candidate_rule": result.get("candidate_rule"),
        "minimum_overtake_initial_behind_gap_m": _overtake_arg(
            "minimum_overtake_initial_behind_gap_m", _NB_MIN_INITIAL_BEHIND_GAP_M
        ),
        "minimum_overtake_side_by_side_lateral_m": _overtake_arg(
            "minimum_overtake_side_by_side_lateral_m", _NB_MIN_SIDE_BY_SIDE_LATERAL_M
        ),
        "minimum_overtake_relative_speed_mps": _overtake_arg(
            "minimum_overtake_relative_speed_mps", _NB_MIN_RELATIVE_SPEED_MPS
        ),
        "minimum_overtake_clearance_m": _overtake_arg(
            "minimum_overtake_clearance_m", _NB_MIN_FULL_CLEARANCE_M
        ),
        "minimum_overtake_same_flow_fraction": _overtake_arg(
            "minimum_overtake_same_flow_fraction", _NB_MIN_SAME_FLOW_FRACTION
        ),
        "overtake_same_flow_threshold_rad": _overtake_arg(
            "overtake_same_flow_threshold_rad", _NB_SAME_FLOW_THRESHOLD_RAD
        ),
        "overtake_lane_stability_frames": _overtake_int_arg(
            "overtake_lane_stability_frames", _NB_MIN_FINAL_STABLE_FRAMES
        ),
        "minimum_overtake_completion_duration_s": _overtake_arg(
            "minimum_overtake_completion_duration_s", _NB_MIN_FINAL_STABLE_DURATION_S
        ),
        "maximum_overtake_duration_s": _overtake_arg(
            "maximum_overtake_duration_s", _NB_MAX_EVENT_DURATION_S
        ),
        "overtake_lane_query_radius_m": _overtake_arg(
            "overtake_lane_query_radius_m", _NB_LANE_QUERY_RADIUS_M
        ),
        "overtake_lane_min_overlap_ratio": _overtake_arg(
            "overtake_lane_min_overlap_ratio", _NB_MIN_LANE_FOOTPRINT_OVERLAP_RATIO
        ),
        "overtake_lane_ambiguity_margin": _overtake_arg(
            "overtake_lane_ambiguity_margin", _NB_LANE_AMBIGUITY_OVERLAP_MARGIN
        ),
        "overtake_max_lane_path_hops": _overtake_int_arg(
            "overtake_max_lane_path_hops", _NB_MAX_LANE_PATH_HOPS
        ),
        "overtake_start_frame_index": int(result["overtake_start_frame_index"]),
        "overtake_departure_frame_index": int(result["overtake_departure_frame_index"]),
        "overtake_side_by_side_frame_index": int(result["overtake_side_by_side_frame_index"]),
        "overtake_order_reversal_frame_index": int(result["overtake_order_reversal_frame_index"]),
        "overtake_clearance_frame_index": int(result["overtake_clearance_frame_index"]),
        "overtake_return_frame_index": int(result["overtake_return_frame_index"]),
        "overtake_completion_frame_index": int(result["overtake_completion_frame_index"]),
    }
    return derived_relation(
        "np:overtakes",
        subject["entity_id"],
        obj["entity_id"],
        completion["frame"],
        OVERTAKE_RULE_ID,
        evidence,
        start_time_us=start_us,
        end_time_us=completion_us,
    )


def _overtake_assignment_start_index(result: dict[str, Any]) -> int:
    """Return the physical departure frame after observed behind-history.

    The initial same-corridor/behind observations are evidence history, not an
    active overtake interval.  The red overtake arrow therefore begins at the
    verified lateral departure and never at scenario frame 0 merely because the
    future trajectory later confirms an overtake.
    """
    departure = int(result["overtake_departure_index"])
    completion = int(result["overtake_completion_index"])
    return min(max(departure, 0), completion)


def _overtake_phase_for_index(
    result: dict[str, Any],
    current_index: int,
    assignment_start_index: int,
) -> tuple[int, str, str]:
    """Return the four presentation phases of a fully observed Case 1 event."""
    side_index = int(result["overtake_side_by_side_index"])
    clearance_index = int(result["overtake_clearance_index"])
    return_index = int(result["overtake_return_index"])
    if current_index < side_index:
        return 1, "initial_behind_departure", "Initial behind / departure"
    if current_index < clearance_index:
        return 2, "passing", "Passing"
    if current_index < return_index:
        return 3, "cleared", "Cleared"
    return 4, "returned", "Returned"


def _full_maneuver_case1_overtake_assertions(
    result: dict[str, Any],
) -> list[Any]:
    """Emit one interval endpoint per frame of a verified maneuver.

    Detection remains retrospective and strict: no assertion is emitted unless
    the complete fully observed Case 1 event passes the internal strict verifier.  Once it is
    verified, the semantic relation is assigned from the verified same-lane start
    through stable completion.  Each row uses the maneuver start as
    ``valid_time_us`` and the displayed frame as ``end_time_us``, matching the
    interval-endpoint convention already used by ``np:follows``.
    """
    records = result["overtake_records"]
    assignment_start_index = _overtake_assignment_start_index(result)
    completion_index = int(result["overtake_completion_index"])
    assignment_start = records[assignment_start_index]
    completion = records[completion_index]
    assignment_start_us = int(assignment_start["timestamp_us"])
    completion_us = int(completion["timestamp_us"])

    # Reuse the complete detector evidence from the accepted event, then add
    # frame-specific assignment and phase information below.
    completed = _completed_case1_overtake_assertion(result)
    base_evidence = dict(completed.evidence)
    detector_start_us = int(base_evidence["start_time_us"])
    detector_start_frame_index = int(base_evidence["overtake_start_frame_index"])
    total_duration_s = (completion_us - assignment_start_us) / 1e6

    assertions = []
    for current_index in range(assignment_start_index, completion_index + 1):
        current = records[current_index]
        current_us = int(current["timestamp_us"])
        phase_number, phase_id, phase_label = _overtake_phase_for_index(
            result,
            current_index,
            assignment_start_index,
        )
        evidence = dict(base_evidence)
        evidence.update({
            "overtake_assignment_strategy": (
                "package_internal_case1_full_maneuver_v1"
            ),
            "overtake_assignment_scope": "full_verified_maneuver",
            "overtake_active": True,
            "overtake_completed": current_index == completion_index,
            "overtake_phase": phase_id,
            "overtake_phase_number": phase_number,
            "overtake_phase_label": phase_label,
            "overtake_current_time_us": current_us,
            "overtake_current_frame_index": int(current["frame_index"]),
            "overtake_assignment_start_time_us": assignment_start_us,
            "overtake_assignment_start_frame_index": int(
                assignment_start["frame_index"]
            ),
            "maneuver_history_gate_passed": True,
            "maneuver_history_frame_count": int(result.get("overtake_history_frame_count", 0)),
            "maneuver_history_duration_s": float(result.get("overtake_history_duration_s", 0.0)),
            "maneuver_history_frames": list(result.get("overtake_history_frame_indices", [])),
            "minimum_maneuver_history_frames": int(result.get("minimum_maneuver_history_frames", 1)),
            "minimum_maneuver_history_s": float(result.get("minimum_maneuver_history_s", 0.5)),
            "history_role": "precondition_only_not_active_overtake",
            "overtake_assignment_start_center_longitudinal_m": float(
                assignment_start["subject_center_longitudinal_m"]
            ),
            "overtake_detector_start_time_us": detector_start_us,
            "overtake_detector_start_frame_index": detector_start_frame_index,
            "overtake_total_event_duration_s": float(total_duration_s),
            "condition_duration_s": float(
                (current_us - assignment_start_us) / 1e6
            ),
            "start_time_us": assignment_start_us,
            "overtake_start_frame_index": int(
                assignment_start["frame_index"]
            ),
        })
        assertions.append(derived_relation(
            "np:overtakes",
            current["subject"]["entity_id"],
            current["object"]["entity_id"],
            current["frame"],
            OVERTAKE_RULE_ID,
            evidence,
            start_time_us=assignment_start_us,
            end_time_us=current_us,
        ))
    return assertions


def derive_overtaking_predicates(snapshots):
    """Detect and assign only fully observed same-lane-start overtakes.

    Stage A performs the package's high-recall directed-pair trajectory scan
    for Case 1. Stage B uses the same Lane + LaneConnector topology model as
    ``np:changesLane`` and requires: initial same-corridor/behind state,
    departure to a lateral adjacent corridor, side-by-side overlap, order
    reversal, full clearance, preservation of O's logical corridor, and stable
    return to that corridor. Longitudinal L -> LC -> L progression does not
    break either the original or passing corridor. The initial same-corridor/behind frames are retained only as history. The relation is assigned from the verified lateral departure through completion.

    Already-in-progress adjacent-lane starts (the former Case 2) are rejected.
    """
    if not ARGS.derive_semantic_predicates or not snapshots:
        return []
    snapshot_list = list(snapshots)
    candidates = _broad_overtake_candidates(snapshot_list)
    accepted = []
    for candidate in candidates:
        records = _collect_strict_pair_records(snapshot_list, candidate)
        result = _evaluate_strict_case1_overtake(candidate, records)
        if result.get("overtake_detected") is True:
            minimum_history_frames, minimum_history_s = _maneuver_history_requirements()
            start_index = int(result["overtake_start_index"])
            departure_index = int(result["overtake_departure_index"])
            history_indices = list(range(start_index, departure_index))
            history_duration_s = 0.0
            if history_indices and departure_index < len(records):
                history_duration_s = max(
                    0.0,
                    (int(records[departure_index]["timestamp_us"]) - int(records[history_indices[0]]["timestamp_us"])) / 1e6,
                )
            if len(history_indices) < minimum_history_frames:
                continue
            if history_duration_s + 1e-9 < minimum_history_s:
                continue
            result["overtake_history_frame_count"] = len(history_indices)
            result["overtake_history_duration_s"] = float(history_duration_s)
            result["overtake_history_frame_indices"] = [int(records[i]["frame_index"]) for i in history_indices]
            result["minimum_maneuver_history_frames"] = int(minimum_history_frames)
            result["minimum_maneuver_history_s"] = float(minimum_history_s)
            result["overtake_records"] = records
            accepted.append(result)
    unique_results = _deduplicate_overtake_results(accepted)
    assertions = []
    for result in unique_results:
        assertions.extend(_full_maneuver_case1_overtake_assertions(result))
    return assertions



# ============================================================================
# crossesInFrontOf: observed dynamic-road-user crossing order
# ============================================================================

CROSSES_IN_FRONT_RULE_ID = "R-CROSSES-IN-FRONT-001"
CROSSES_IN_FRONT_PREDICATE = "np:crossesInFrontOf"
CROSSES_IN_FRONT_RAY_LENGTH_M = 10.0
CROSSES_IN_FRONT_ETA_SPEED_EPS_MPS = 0.05



def _crosses_agent_type(record: dict[str, Any]) -> str:
    return _normalized_agent_type(record)


def _crosses_canonical_type(record: dict[str, Any]) -> str:
    agent_type = _crosses_agent_type(record)
    if "PEDESTRIAN" in agent_type:
        return "PEDESTRIAN"
    if "BICYCLE" in agent_type or "CYCLIST" in agent_type:
        return "BICYCLE"
    if "MOTORCYCLE" in agent_type:
        return "MOTORCYCLE"
    if "BUS" in agent_type:
        return "BUS"
    if "TRUCK" in agent_type:
        return "TRUCK"
    if str(record.get("track_token")) == "ego" or "EGO" in agent_type:
        return "EGO"
    if "VEHICLE" in agent_type or "CAR" in agent_type:
        return "VEHICLE"
    return agent_type or "UNKNOWN"


def _crosses_is_pedestrian(record: dict[str, Any]) -> bool:
    return _crosses_canonical_type(record) == "PEDESTRIAN"


def _crosses_is_bicycle(record: dict[str, Any]) -> bool:
    return _crosses_canonical_type(record) == "BICYCLE"


def _crosses_record_dimensions(record: dict[str, Any]) -> tuple[float, float]:
    dimensions = record.get("dimensions") or {}
    length = _finite_float(dimensions.get("length_m"))
    width = _finite_float(dimensions.get("width_m"))

    canonical = _crosses_canonical_type(record)
    if length is None or length <= 0.0:
        length = 0.8 if canonical == "PEDESTRIAN" else 4.5
    if width is None or width <= 0.0:
        width = 0.8 if canonical == "PEDESTRIAN" else 2.0
    return float(length), float(width)


def _crosses_record_speed(record: dict[str, Any]) -> Optional[float]:
    speed = _finite_float(record.get("speed"))
    if speed is not None:
        return float(speed)
    velocity = record.get("velocity")
    if velocity is None:
        return None
    try:
        return float(math.hypot(float(velocity[0]), float(velocity[1])))
    except Exception:
        return None


def _crosses_preferred_heading(item: dict[str, Any]) -> float:
    record = item["record"]
    velocity = record.get("velocity")
    speed = _crosses_record_speed(record)
    if velocity is not None and speed is not None and speed >= 0.50:
        try:
            return float(math.atan2(float(velocity[1]), float(velocity[0])))
        except Exception:
            pass
    return float(record["xyh"][2])


def _crosses_motion_heading(track: list[dict[str, Any]], index: int) -> float:
    i0 = max(0, int(index) - 1)
    i1 = min(len(track) - 1, int(index) + 1)
    x0, y0, _ = track[i0]["record"]["xyh"]
    x1, y1, _ = track[i1]["record"]["xyh"]
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    if math.hypot(dx, dy) >= 0.20:
        return float(math.atan2(dy, dx))
    return _crosses_preferred_heading(track[index])


def _crosses_forward_lateral(
    origin_xy: tuple[float, float],
    heading: float,
    target_xy: tuple[float, float],
) -> tuple[float, float]:
    dx = float(target_xy[0]) - float(origin_xy[0])
    dy = float(target_xy[1]) - float(origin_xy[1])
    c = math.cos(float(heading))
    s = math.sin(float(heading))
    return (
        float(c * dx + s * dy),
        float(-s * dx + c * dy),
    )


def _crosses_forward_ray_intersection(
    subject_xy: tuple[float, float],
    subject_heading: float,
    object_xy: tuple[float, float],
    object_heading: float,
    *,
    max_distance_m: Optional[float] = None,
    epsilon: float = 1e-9,
) -> Optional[dict[str, float]]:
    """Return the unique intersection of the two forward rays/segments.

    v9.5.34 uses this exact geometry for both detection and visualization.
    When ``max_distance_m`` is provided, each ray is a finite segment from
    0 to that distance.  Therefore an intersection is valid only when it is
    forward of both agents *and* lies inside both finite forward segments.
    """
    sx, sy = map(float, subject_xy)
    ox, oy = map(float, object_xy)
    sh = float(subject_heading)
    oh = float(object_heading)

    dsx, dsy = math.cos(sh), math.sin(sh)
    dox, doy = math.cos(oh), math.sin(oh)
    denominator = dsx * doy - dsy * dox

    # Parallel/collinear rays have no unique crossing point.
    if abs(denominator) <= float(epsilon):
        return None

    rx = ox - sx
    ry = oy - sy
    subject_lambda_m = (rx * doy - ry * dox) / denominator
    object_mu_m = (rx * dsy - ry * dsx) / denominator

    tolerance = max(float(epsilon), 1e-9)
    if subject_lambda_m < -tolerance or object_mu_m < -tolerance:
        return None

    subject_lambda_m = max(0.0, float(subject_lambda_m))
    object_mu_m = max(0.0, float(object_mu_m))

    if max_distance_m is not None:
        limit = max(0.0, float(max_distance_m))
        if (
            subject_lambda_m > limit + tolerance
            or object_mu_m > limit + tolerance
        ):
            return None

    return {
        "x": float(sx + subject_lambda_m * dsx),
        "y": float(sy + subject_lambda_m * dsy),
        "subject_lambda_m": subject_lambda_m,
        "object_mu_m": object_mu_m,
    }

def _crosses_heading_difference(a: float, b: float) -> float:
    return abs(wrap_signed(float(a) - float(b)))


def _crosses_projected_half_extent(
    length_m: float,
    width_m: float,
    actor_heading: float,
    reference_heading: float,
) -> float:
    delta = _crosses_heading_difference(actor_heading, reference_heading)
    return 0.5 * (
        float(length_m) * abs(math.cos(delta))
        + float(width_m) * abs(math.sin(delta))
    )


def _crosses_track_displacement(track: list[dict[str, Any]]) -> float:
    if len(track) < 2:
        return 0.0
    x0, y0, _ = track[0]["record"]["xyh"]
    x1, y1, _ = track[-1]["record"]["xyh"]
    return float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))


def _crosses_track_motion_summary(
    track: list[dict[str, Any]],
) -> dict[str, float]:
    """Summarize observed motion without changing any other predicate state."""
    if not track:
        return {
            "duration_s": 0.0,
            "max_displacement_m": 0.0,
            "median_speed_mps": float("inf"),
        }

    x0, y0, _ = track[0]["record"]["xyh"]
    maximum_displacement = 0.0
    speeds = []
    for item in track:
        x, y, _ = item["record"]["xyh"]
        maximum_displacement = max(
            maximum_displacement,
            math.hypot(float(x) - float(x0), float(y) - float(y0)),
        )
        speed = _crosses_record_speed(item["record"])
        if speed is not None and math.isfinite(float(speed)):
            speeds.append(float(speed))

    duration_s = 0.0
    if len(track) >= 2:
        duration_s = max(
            0.0,
            float(track[-1]["time_s"]) - float(track[0]["time_s"]),
        )

    return {
        "duration_s": float(duration_s),
        "max_displacement_m": float(maximum_displacement),
        "median_speed_mps": (
            float(median(speeds)) if speeds else float("inf")
        ),
    }


def _crosses_map_context(
    item: dict[str, Any],
) -> dict[str, Any]:
    """Return local nuPlan map distances used only by crossesInFrontOf.

    The query intentionally does not mutate the entity record or shared map
    predicates.  If map access is unavailable, ``available`` is False and the
    caller preserves the previous detector behavior.
    """
    unavailable = {
        "available": False,
        "road_distance_m": None,
        "lane_distance_m": None,
        "intersection_distance_m": None,
        "crosswalk_distance_m": None,
        "carpark_distance_m": None,
        "carpark_contains_center": False,
        "carpark_footprint_overlap": False,
        "carpark_footprint_overlap_area_m2": 0.0,
        "carpark_area_ids": [],
    }

    if not MAP_QUERY_AVAILABLE or not SHAPELY_AVAILABLE:
        return unavailable

    snapshot = item.get("snapshot") or {}
    frame = snapshot.get("frame")
    map_api = getattr(frame, "map_api", None)
    if map_api is None or not hasattr(map_api, "get_proximal_map_objects"):
        return unavailable

    try:
        x, y, _ = map(float, item["record"]["xyh"])
        query_point = Point2D(float(x), float(y))
        geometry_point = Point(float(x), float(y))
        footprint = oriented_box_polygon(item["record"])
    except Exception:
        return unavailable

    layer_names = (
        "LANE",
        "LANE_CONNECTOR",
        "DRIVABLE_AREA",
        "INTERSECTION",
        "CROSSWALK",
        "CARPARK_AREA",
    )
    layers = []
    layer_name_by_value = {}
    for name in layer_names:
        layer = getattr(SemanticMapLayer, name, None)
        if layer is not None:
            layers.append(layer)
            layer_name_by_value[layer] = name

    if not layers:
        return unavailable

    try:
        nearby = map_api.get_proximal_map_objects(
            query_point,
            float(ARGS.crosses_in_front_map_query_radius_m),
            layers,
        )
    except Exception:
        return unavailable

    distances = {name: float("inf") for name in layer_names}
    carpark_contains_center = False
    carpark_footprint_overlap = False
    carpark_footprint_overlap_area_m2 = 0.0
    carpark_area_ids: list[str] = []

    for layer, objects in (nearby or {}).items():
        name = layer_name_by_value.get(layer)
        if name is None:
            name = str(getattr(layer, "name", layer)).upper()
        if name not in distances:
            continue

        for map_object in objects or []:
            polygon = getattr(map_object, "polygon", None)
            if polygon is None:
                continue
            try:
                distance_m = max(0.0, float(polygon.distance(geometry_point)))
            except Exception:
                continue
            distances[name] = min(distances[name], distance_m)

            # Strong parking exclusion for crossesInFrontOf: if a motor
            # vehicle's current physical footprint actually occupies a mapped
            # CARPARK_AREA, the pair is rejected immediately later, regardless
            # of instantaneous speed.  Center containment is kept as a robust
            # fallback when footprint geometry is unavailable or degenerate.
            if name == "CARPARK_AREA":
                object_id = str(
                    getattr(map_object, "id", getattr(map_object, "object_id", ""))
                )
                try:
                    contains_center = bool(polygon.covers(geometry_point))
                except Exception:
                    contains_center = False
                if contains_center:
                    carpark_contains_center = True
                    if object_id and object_id not in carpark_area_ids:
                        carpark_area_ids.append(object_id)

                if footprint is not None:
                    try:
                        overlap_area = max(
                            0.0,
                            float(footprint.intersection(polygon).area),
                        )
                    except Exception:
                        overlap_area = 0.0
                    if overlap_area > 1e-6:
                        carpark_footprint_overlap = True
                        carpark_footprint_overlap_area_m2 = max(
                            carpark_footprint_overlap_area_m2,
                            overlap_area,
                        )
                        if object_id and object_id not in carpark_area_ids:
                            carpark_area_ids.append(object_id)

    lane_distance = min(
        distances["LANE"],
        distances["LANE_CONNECTOR"],
    )
    road_distance = min(
        lane_distance,
        distances["DRIVABLE_AREA"],
        distances["INTERSECTION"],
    )

    def _finite_or_none(value: float) -> Optional[float]:
        return float(value) if math.isfinite(float(value)) else None

    return {
        "available": True,
        "road_distance_m": _finite_or_none(road_distance),
        "lane_distance_m": _finite_or_none(lane_distance),
        "intersection_distance_m": _finite_or_none(
            distances["INTERSECTION"]
        ),
        "crosswalk_distance_m": _finite_or_none(
            distances["CROSSWALK"]
        ),
        "carpark_distance_m": _finite_or_none(
            distances["CARPARK_AREA"]
        ),
        "carpark_contains_center": bool(carpark_contains_center),
        "carpark_footprint_overlap": bool(carpark_footprint_overlap),
        "carpark_footprint_overlap_area_m2": float(
            carpark_footprint_overlap_area_m2
        ),
        "carpark_area_ids": list(carpark_area_ids),
    }


def _crosses_distance_exceeds(
    value: Optional[float],
    threshold: float,
) -> bool:
    """Treat a missing queried layer as farther than the local threshold."""
    return value is None or float(value) > float(threshold)


def _crosses_vehicle_occupies_carpark(
    record: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    """Return True when a motor vehicle occupies a mapped CARPARK_AREA.

    This is intentionally stronger than the stationary parked-object fallback:
    occupation of an explicit nuPlan car-park polygon is enough to exclude the
    actor from crossesInFrontOf, even if its reported velocity is noisy/nonzero.
    """
    if not bool(context.get("available")):
        return False

    canonical = _crosses_canonical_type(record)
    agent_type = _crosses_agent_type(record)
    motor_vehicle = (
        canonical in {"VEHICLE", "MOTORCYCLE", "BUS", "TRUCK", "EGO"}
        or "TRAILER" in agent_type
        or "CONSTRUCTION_VEHICLE" in agent_type
    )
    if not motor_vehicle:
        return False

    return bool(
        context.get("carpark_footprint_overlap")
        or context.get("carpark_contains_center")
    )


def _crosses_track_is_parked(
    track: list[dict[str, Any]],
    context: dict[str, Any],
) -> bool:
    """Conservatively reject persistent stationary roadside/car-park actors.

    A low-speed actor is *not* called parked merely because it stopped.  It must
    also be in a car-park area or remain outside the lane/lane-connector
    corridor. This preserves vehicles waiting at signals/intersections.
    """
    if not track:
        return False

    first_record = track[0]["record"]
    canonical = _crosses_canonical_type(first_record)
    if canonical in {"PEDESTRIAN", "EGO"}:
        return False

    motion = _crosses_track_motion_summary(track)
    if motion["duration_s"] < float(
        ARGS.crosses_in_front_parked_min_duration_s
    ):
        return False
    if motion["max_displacement_m"] > float(
        ARGS.crosses_in_front_parked_max_displacement_m
    ):
        return False
    if motion["median_speed_mps"] > float(
        ARGS.crosses_in_front_parked_max_speed_mps
    ):
        return False

    if not bool(context.get("available")):
        return False

    carpark_distance = context.get("carpark_distance_m")
    if (
        carpark_distance is not None
        and float(carpark_distance)
        <= float(ARGS.crosses_in_front_parked_max_lane_distance_m)
    ):
        return True

    lane_distance = context.get("lane_distance_m")
    return _crosses_distance_exceeds(
        lane_distance,
        float(ARGS.crosses_in_front_parked_max_lane_distance_m),
    )


def _crosses_actor_map_relevant(
    record: dict[str, Any],
    context: dict[str, Any],
) -> tuple[bool, str]:
    """Apply the requested road/crosswalk relevance gate to one actor."""
    if not bool(context.get("available")):
        # Synthetic tests and degraded map runs preserve the old behavior.
        return True, "map_unavailable_fail_open"

    canonical = _crosses_canonical_type(record)
    road_distance = context.get("road_distance_m")
    crosswalk_distance = context.get("crosswalk_distance_m")

    if canonical == "PEDESTRIAN":
        intersection_distance = context.get("intersection_distance_m")
        near_crosswalk = not _crosses_distance_exceeds(
            crosswalk_distance,
            float(ARGS.crosses_in_front_crosswalk_near_distance_m),
        )
        near_intersection = not _crosses_distance_exceeds(
            intersection_distance,
            float(ARGS.crosses_in_front_crosswalk_near_distance_m),
        )
        near_road = not _crosses_distance_exceeds(
            road_distance,
            float(ARGS.crosses_in_front_bicycle_road_distance_m),
        )

        if near_crosswalk:
            return True, "pedestrian_near_crosswalk"
        if near_intersection and near_road:
            return True, "pedestrian_near_intersection_road"
        return False, "pedestrian_not_near_crosswalk_or_intersection"

    if canonical == "BICYCLE":
        # A bicycle may cross at a crosswalk, but unlike a pedestrian it can
        # also legitimately cross through a lane/connector/intersection.
        near_crosswalk = not _crosses_distance_exceeds(
            crosswalk_distance,
            float(ARGS.crosses_in_front_crosswalk_near_distance_m),
        )
        near_road = not _crosses_distance_exceeds(
            road_distance,
            float(ARGS.crosses_in_front_bicycle_road_distance_m),
        )
        if not (near_crosswalk or near_road):
            return False, "bicycle_not_near_crosswalk_or_road"
        return True, "bicycle_map_relevant"

    if _crosses_distance_exceeds(
        road_distance,
        float(ARGS.crosses_in_front_max_road_distance_m),
    ):
        return False, "dynamic_actor_far_from_road"

    return True, "road_relevant"


def _crosses_pair_map_filter(
    subject_track: list[dict[str, Any]],
    object_track: list[dict[str, Any]],
    subject_item: dict[str, Any],
    object_item: dict[str, Any],
) -> dict[str, Any]:
    """Predicate-local parked/off-road/infrastructure gate."""
    context_cache_key = "_crosses_map_context_v9534"
    subject_context = subject_item.get(context_cache_key)
    if not isinstance(subject_context, dict):
        subject_context = _crosses_map_context(subject_item)
        subject_item[context_cache_key] = subject_context

    object_context = object_item.get(context_cache_key)
    if not isinstance(object_context, dict):
        object_context = _crosses_map_context(object_item)
        object_item[context_cache_key] = object_context

    # Explicit mapped car parks are the strongest parking signal.  Reject a
    # motor vehicle immediately if its current footprint overlaps CARPARK_AREA;
    # no low-speed or persistence requirement is needed for this case.
    if _crosses_vehicle_occupies_carpark(
        subject_item["record"], subject_context
    ):
        return {
            "accepted": False,
            "reason": "subject_in_carpark_area",
            "subject_context": subject_context,
            "object_context": object_context,
        }
    if _crosses_vehicle_occupies_carpark(
        object_item["record"], object_context
    ):
        return {
            "accepted": False,
            "reason": "object_in_carpark_area",
            "subject_context": subject_context,
            "object_context": object_context,
        }

    if _crosses_track_is_parked(subject_track, subject_context):
        return {
            "accepted": False,
            "reason": "subject_parked",
            "subject_context": subject_context,
            "object_context": object_context,
        }
    if _crosses_track_is_parked(object_track, object_context):
        return {
            "accepted": False,
            "reason": "object_parked",
            "subject_context": subject_context,
            "object_context": object_context,
        }

    subject_ok, subject_reason = _crosses_actor_map_relevant(
        subject_item["record"], subject_context
    )
    if not subject_ok:
        return {
            "accepted": False,
            "reason": subject_reason,
            "subject_context": subject_context,
            "object_context": object_context,
        }

    object_ok, object_reason = _crosses_actor_map_relevant(
        object_item["record"], object_context
    )
    if not object_ok:
        return {
            "accepted": False,
            "reason": object_reason,
            "subject_context": subject_context,
            "object_context": object_context,
        }

    return {
        "accepted": True,
        "reason": "map_relevant",
        "subject_reason": subject_reason,
        "object_reason": object_reason,
        "subject_context": subject_context,
        "object_context": object_context,
    }


def _crosses_build_tracks(
    snapshots: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for snapshot in snapshots:
        for token, record in snapshot.get("entities", {}).items():
            if not is_dynamic_road_user(record):
                continue
            tracks[str(token)].append({
                "token": str(token),
                "record": record,
                "snapshot": snapshot,
                "timestamp_us": int(snapshot["timestamp_us"]),
                "time_s": float(snapshot["timestamp_us"]) / 1e6,
                "frame_index": int(snapshot.get("frame_index", 0)),
            })

    minimum_observations = max(
        2, int(ARGS.crosses_in_front_min_observations)
    )
    minimum_displacement = float(
        ARGS.crosses_in_front_min_track_displacement_m
    )

    cleaned: dict[str, list[dict[str, Any]]] = {}
    for token, track in tracks.items():
        track = sorted(track, key=lambda row: int(row["timestamp_us"]))
        if len(track) < minimum_observations:
            continue

        first_record = track[0]["record"]
        if (
            _crosses_is_pedestrian(first_record)
            or _crosses_is_bicycle(first_record)
        ):
            if _crosses_track_displacement(track) < minimum_displacement:
                continue

        cleaned[token] = track

    return cleaned


def _crosses_bbox_distance(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> float:
    ax = [float(row["record"]["xyh"][0]) for row in first]
    ay = [float(row["record"]["xyh"][1]) for row in first]
    bx = [float(row["record"]["xyh"][0]) for row in second]
    by = [float(row["record"]["xyh"][1]) for row in second]

    dx = max(
        0.0,
        max(min(ax), min(bx)) - min(max(ax), max(bx)),
    )
    dy = max(
        0.0,
        max(min(ay), min(by)) - min(max(ay), max(by)),
    )
    return float(math.hypot(dx, dy))


def _crosses_common_track_items(
    subject_track: list[dict[str, Any]],
    object_track: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    subject_by_time = {
        int(row["timestamp_us"]): row for row in subject_track
    }
    object_by_time = {
        int(row["timestamp_us"]): row for row in object_track
    }
    times = sorted(set(subject_by_time).intersection(object_by_time))
    return [
        (subject_by_time[timestamp_us], object_by_time[timestamp_us])
        for timestamp_us in times
    ]


def _crosses_row_metrics(
    subject_track: list[dict[str, Any]],
    object_track: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    common = _crosses_common_track_items(subject_track, object_track)
    subject_index_by_time = {
        int(item["timestamp_us"]): index
        for index, item in enumerate(subject_track)
    }

    rows: list[dict[str, Any]] = []
    for subject_item, object_item in common:
        subject_record = subject_item["record"]
        object_record = object_item["record"]

        subject_index = subject_index_by_time[int(subject_item["timestamp_us"])]
        object_heading = _crosses_preferred_heading(object_item)
        subject_heading = _crosses_motion_heading(subject_track, subject_index)

        sx, sy, _ = subject_record["xyh"]
        ox, oy, _ = object_record["xyh"]

        longitudinal_m, lateral_m = _crosses_forward_lateral(
            (float(ox), float(oy)),
            object_heading,
            (float(sx), float(sy)),
        )

        subject_length, subject_width = _crosses_record_dimensions(
            subject_record
        )
        object_length, object_width = _crosses_record_dimensions(
            object_record
        )
        subject_extent = _crosses_projected_half_extent(
            subject_length,
            subject_width,
            subject_heading,
            object_heading,
        )
        object_front = 0.5 * float(object_length)

        front_clearance_m = float(
            longitudinal_m - object_front - subject_extent
        )
        corridor_half_width_m = float(
            0.5 * object_width
            + 0.5 * subject_width
            + float(ARGS.crosses_in_front_side_buffer_m)
        )

        crossing_speed_mps = None
        velocity = subject_record.get("velocity")
        if velocity is not None:
            try:
                normal_x = -math.sin(object_heading)
                normal_y = math.cos(object_heading)
                crossing_speed_mps = abs(
                    float(velocity[0]) * normal_x
                    + float(velocity[1]) * normal_y
                )
            except Exception:
                crossing_speed_mps = None

        rows.append({
            "timestamp_us": int(subject_item["timestamp_us"]),
            "time_s": float(subject_item["time_s"]),
            "subject_item": subject_item,
            "object_item": object_item,
            "subject_heading": float(subject_heading),
            "object_heading": float(object_heading),
            "longitudinal_m": float(longitudinal_m),
            "lateral_m": float(lateral_m),
            "front_clearance_m": float(front_clearance_m),
            "corridor_half_width_m": float(corridor_half_width_m),
            "crossing_angle_deg": math.degrees(
                _crosses_heading_difference(
                    subject_heading,
                    object_heading,
                )
            ),
            "subject_crossing_speed_mps": crossing_speed_mps,
        })

    return rows


def _crosses_find_side_evidence(
    rows: list[dict[str, Any]],
    left_index: int,
    right_index: int,
) -> Optional[dict[str, Any]]:
    crossing_time_s = 0.5 * (
        float(rows[left_index]["time_s"])
        + float(rows[right_index]["time_s"])
    )
    margin_m = max(
        float(rows[left_index]["corridor_half_width_m"]),
        float(rows[right_index]["corridor_half_width_m"]),
    )
    search_window_s = float(
        ARGS.crosses_in_front_side_search_window_s
    )

    # crossesInFrontOf uses O's FORWARD RAY, not an infinite heading line.
    # Side-to-side evidence is therefore valid only while S is in the forward
    # half-plane of O.  Samples behind O must not contribute evidence for an
    # "in front of" crossing.
    before = [
        (index, row)
        for index, row in enumerate(rows[: left_index + 1])
        if (
            crossing_time_s - search_window_s
            <= float(row["time_s"])
            <= float(rows[left_index]["time_s"])
            and float(row["longitudinal_m"]) > 0.0
        )
    ]
    after = [
        (right_index + offset, row)
        for offset, row in enumerate(rows[right_index:])
        if (
            float(rows[right_index]["time_s"])
            <= float(row["time_s"])
            <= crossing_time_s + search_window_s
            and float(row["longitudinal_m"]) > 0.0
        )
    ]

    negative_before = [
        (index, row)
        for index, row in before
        if float(row["lateral_m"]) <= -margin_m
    ]
    positive_before = [
        (index, row)
        for index, row in before
        if float(row["lateral_m"]) >= margin_m
    ]
    negative_after = [
        (index, row)
        for index, row in after
        if float(row["lateral_m"]) <= -margin_m
    ]
    positive_after = [
        (index, row)
        for index, row in after
        if float(row["lateral_m"]) >= margin_m
    ]

    possibilities = []
    if negative_before and positive_after:
        possibilities.append((
            negative_before[-1],
            positive_after[0],
            "negative_to_positive",
        ))
    if positive_before and negative_after:
        possibilities.append((
            positive_before[-1],
            negative_after[0],
            "positive_to_negative",
        ))

    if not possibilities:
        return None

    previous, following, direction = min(
        possibilities,
        key=lambda value: (
            float(value[1][1]["time_s"])
            - float(value[0][1]["time_s"])
        ),
    )

    return {
        "pre_index": int(previous[0]),
        "pre_row": previous[1],
        "post_index": int(following[0]),
        "post_row": following[1],
        "direction": str(direction),
        "margin_m": float(margin_m),
    }


def _crosses_interpolate_zero_crossing(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    lateral_first = float(first["lateral_m"])
    lateral_second = float(second["lateral_m"])
    denominator = lateral_second - lateral_first

    if abs(denominator) < 1e-9:
        alpha = 0.5
    else:
        alpha = max(
            0.0,
            min(1.0, -lateral_first / denominator),
        )

    def lerp(a: float, b: float) -> float:
        return float(a) + float(alpha) * (float(b) - float(a))

    first_subject = first["subject_item"]["record"]
    second_subject = second["subject_item"]["record"]
    first_object = first["object_item"]["record"]
    second_object = second["object_item"]["record"]

    return {
        "alpha": float(alpha),
        "time_s": lerp(first["time_s"], second["time_s"]),
        "timestamp_us": int(round(
            lerp(first["timestamp_us"], second["timestamp_us"])
        )),
        "x": lerp(
            first_subject["xyh"][0],
            second_subject["xyh"][0],
        ),
        "y": lerp(
            first_subject["xyh"][1],
            second_subject["xyh"][1],
        ),
        "object_x": lerp(
            first_object["xyh"][0],
            second_object["xyh"][0],
        ),
        "object_y": lerp(
            first_object["xyh"][1],
            second_object["xyh"][1],
        ),
        "longitudinal_m": lerp(
            first["longitudinal_m"],
            second["longitudinal_m"],
        ),
        "front_clearance_m": lerp(
            first["front_clearance_m"],
            second["front_clearance_m"],
        ),
    }


def _crosses_object_conflict_arrival(
    object_track: list[dict[str, Any]],
    conflict_xy: tuple[float, float],
    after_time_s: float,
) -> Optional[dict[str, Any]]:
    """Estimate when O's physical footprint first reaches static conflict C."""
    previous = None
    footprint_buffer_m = float(
        ARGS.crosses_in_front_conflict_footprint_buffer_m
    )

    for item in object_track:
        time_s = float(item["time_s"])
        if time_s < float(after_time_s):
            previous = item
            continue

        record = item["record"]
        heading = _crosses_preferred_heading(item)
        x, y, _ = record["xyh"]
        forward_m, lateral_m = _crosses_forward_lateral(
            (float(x), float(y)),
            heading,
            conflict_xy,
        )

        length_m, width_m = _crosses_record_dimensions(record)
        half_length_m = 0.5 * length_m + footprint_buffer_m
        half_width_m = 0.5 * width_m + footprint_buffer_m

        if (
            abs(lateral_m) <= half_width_m
            and -half_length_m <= forward_m <= half_length_m
        ):
            return {
                "time_s": time_s,
                "timestamp_us": int(item["timestamp_us"]),
                "mode": "sampled_footprint_contains_conflict",
            }

        # Sparse-sampling fallback: detect O's front crossing the static
        # conflict plane between consecutive observations.
        if previous is not None:
            previous_record = previous["record"]
            previous_heading = _crosses_preferred_heading(previous)
            px, py, _ = previous_record["xyh"]
            previous_forward, previous_lateral = _crosses_forward_lateral(
                (float(px), float(py)),
                previous_heading,
                conflict_xy,
            )
            previous_length, previous_width = _crosses_record_dimensions(
                previous_record
            )
            previous_half_length = (
                0.5 * previous_length + footprint_buffer_m
            )
            previous_half_width = (
                0.5 * previous_width + footprint_buffer_m
            )

            previous_gap = previous_forward - previous_half_length
            current_gap = forward_m - half_length_m

            if (
                abs(previous_lateral) <= previous_half_width + 0.5
                and abs(lateral_m) <= half_width_m + 0.5
                and previous_gap > 0.0 >= current_gap
            ):
                denominator = previous_gap - current_gap
                alpha = (
                    1.0
                    if abs(denominator) < 1e-9
                    else max(
                        0.0,
                        min(1.0, previous_gap / denominator),
                    )
                )
                arrival_time_s = (
                    float(previous["time_s"])
                    + alpha * (
                        time_s - float(previous["time_s"])
                    )
                )
                # If the interpolation bracket begins before S's entry,
                # retain it only when O's estimated physical arrival occurs
                # at/after the crossing interval begins.
                if arrival_time_s >= float(after_time_s) - 1e-9:
                    return {
                        "time_s": float(arrival_time_s),
                        "timestamp_us": int(round(arrival_time_s * 1e6)),
                        "mode": "interpolated_front_plane_crossing",
                    }

        previous = item

    return None


def _crosses_frame_candidate(
    subject_track: list[dict[str, Any]],
    object_track: list[dict[str, Any]],
    subject_item: dict[str, Any],
    object_item: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Evaluate the v9.5.34 finite-ray rule at exactly one frame.

    The predicate is intentionally frame-local: if the same 10 m rays drawn in
    the video do not intersect at this frame, no ``crossesInFrontOf`` assertion
    can be emitted for this frame.
    """
    subject_record = subject_item["record"]
    object_record = object_item["record"]

    if (
        not bool(ARGS.crosses_in_front_include_pedestrian_pedestrian)
        and _crosses_is_pedestrian(subject_record)
        and _crosses_is_pedestrian(object_record)
    ):
        return None

    # Use the exact current xyh headings that the video uses.  Keeping one
    # authoritative direction source prevents detector/overlay disagreement.
    sx, sy, subject_heading = map(float, subject_record["xyh"])
    ox, oy, object_heading = map(float, object_record["xyh"])

    crossing_angle_deg = math.degrees(
        _crosses_heading_difference(subject_heading, object_heading)
    )
    if not (
        float(ARGS.crosses_in_front_min_crossing_angle_deg)
        <= crossing_angle_deg
        <= float(ARGS.crosses_in_front_max_crossing_angle_deg)
    ):
        return None

    conflict = _crosses_forward_ray_intersection(
        (sx, sy),
        subject_heading,
        (ox, oy),
        object_heading,
        max_distance_m=CROSSES_IN_FRONT_RAY_LENGTH_M,
    )
    if conflict is None:
        return None

    subject_speed_mps = _crosses_record_speed(subject_record)
    if (
        subject_speed_mps is None
        or float(subject_speed_mps)
        < float(ARGS.crosses_in_front_min_subject_crossing_speed_mps)
    ):
        return None

    object_speed_mps = _crosses_record_speed(object_record)
    if object_speed_mps is None:
        # Without an object speed we cannot establish the directional
        # "S before O" semantics reliably.
        return None

    subject_eta_s = float(conflict["subject_lambda_m"]) / max(
        float(subject_speed_mps), CROSSES_IN_FRONT_ETA_SPEED_EPS_MPS
    )

    if float(object_speed_mps) <= CROSSES_IN_FRONT_ETA_SPEED_EPS_MPS:
        object_eta_s = float("inf")
        object_stationary_mode = True
        eta_gap_s = None
    else:
        object_eta_s = float(conflict["object_mu_m"]) / float(object_speed_mps)
        object_stationary_mode = False
        eta_gap_s = float(object_eta_s - subject_eta_s)

        # S must reach the common conflict point first; otherwise this ordered
        # relation is not crossesInFrontOf(S,O).
        if eta_gap_s < float(ARGS.crosses_in_front_order_margin_s):
            return None
        if eta_gap_s > float(
            ARGS.crosses_in_front_max_arrival_time_difference_s
        ):
            return None

    # Geometry/order is deliberately evaluated before map queries for speed.
    # Semantically the parking/map gate is still mandatory; this ordering only
    # avoids expensive repeated map lookups for pairs whose 10 m rays cannot
    # possibly produce the predicate.
    map_filter = _crosses_pair_map_filter(
        subject_track,
        object_track,
        subject_item,
        object_item,
    )
    if not bool(map_filter["accepted"]):
        return None

    # A stopped O has effectively infinite ETA.  S still needs to be moving;
    # mapped car-park and roadside-parking filters have now been applied.
    candidate_score = (
        max(0.0, 3.0 - abs(crossing_angle_deg - 90.0) / 30.0)
        + max(
            0.0,
            2.0
            - max(
                float(conflict["subject_lambda_m"]),
                float(conflict["object_mu_m"]),
            )
            / CROSSES_IN_FRONT_RAY_LENGTH_M,
        )
        + min(2.0, float(subject_speed_mps))
    )

    return {
        "subject_token": str(subject_item["token"]),
        "object_token": str(object_item["token"]),
        "subject_type": _crosses_canonical_type(subject_record),
        "object_type": _crosses_canonical_type(object_record),
        "timestamp_us": int(subject_item["timestamp_us"]),
        "time_s": float(subject_item["time_s"]),
        "subject_item": subject_item,
        "object_item": object_item,
        "conflict_x": float(conflict["x"]),
        "conflict_y": float(conflict["y"]),
        "ray_length_m": float(CROSSES_IN_FRONT_RAY_LENGTH_M),
        "subject_ray_distance_m": float(conflict["subject_lambda_m"]),
        "object_ray_distance_m": float(conflict["object_mu_m"]),
        "crossing_angle_deg": float(crossing_angle_deg),
        "subject_speed_mps": float(subject_speed_mps),
        "object_speed_mps": float(object_speed_mps),
        "subject_eta_s": float(subject_eta_s),
        "object_eta_s": (
            None if not math.isfinite(object_eta_s) else float(object_eta_s)
        ),
        "eta_gap_s": eta_gap_s,
        "object_stationary_mode": bool(object_stationary_mode),
        "reference_geometry": "finite_forward_ray_segments_10m",
        "map_context_filter": "carpark_parked_offroad_infrastructure",
        "subject_map_filter_reason": str(
            map_filter.get("subject_reason", "accepted")
        ),
        "object_map_filter_reason": str(
            map_filter.get("object_reason", "accepted")
        ),
        "candidate_score": float(candidate_score),
    }


def _crosses_finalize_active_run(
    run: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collapse one contiguous sequence of valid ray-intersection frames."""
    representative = min(
        run,
        key=lambda row: (
            max(
                float(row["subject_ray_distance_m"]),
                float(row["object_ray_distance_m"]),
            ),
            -float(row["candidate_score"]),
        ),
    )
    first = run[0]
    last = run[-1]
    return {
        "subject_token": str(first["subject_token"]),
        "object_token": str(first["object_token"]),
        "subject_type": str(first["subject_type"]),
        "object_type": str(first["object_type"]),
        "entry_time_s": float(first["time_s"]),
        "entry_time_us": int(first["timestamp_us"]),
        "clearance_time_s": float(last["time_s"]),
        "clearance_time_us": int(last["timestamp_us"]),
        "crossing_time_s": float(representative["time_s"]),
        "crossing_time_us": int(representative["timestamp_us"]),
        "duration_s": max(0.0, float(last["time_s"]) - float(first["time_s"])),
        "conflict_x": float(representative["conflict_x"]),
        "conflict_y": float(representative["conflict_y"]),
        "crossing_angle_deg": float(representative["crossing_angle_deg"]),
        "ray_length_m": float(representative["ray_length_m"]),
        "subject_ray_distance_m": float(
            representative["subject_ray_distance_m"]
        ),
        "object_ray_distance_m": float(
            representative["object_ray_distance_m"]
        ),
        "subject_speed_mps": float(representative["subject_speed_mps"]),
        "object_speed_mps": float(representative["object_speed_mps"]),
        "subject_eta_s": float(representative["subject_eta_s"]),
        "object_eta_s": representative["object_eta_s"],
        "eta_gap_s": representative["eta_gap_s"],
        "object_stationary_mode": bool(
            representative["object_stationary_mode"]
        ),
        "reference_geometry": str(representative["reference_geometry"]),
        "map_context_filter": str(representative["map_context_filter"]),
        "subject_map_filter_reason": str(
            representative["subject_map_filter_reason"]
        ),
        "object_map_filter_reason": str(
            representative["object_map_filter_reason"]
        ),
        "candidate_score": max(float(row["candidate_score"]) for row in run),
        "active_frames": list(run),
    }


def _crosses_evaluate_ordered_pair(
    subject_track: list[dict[str, Any]],
    object_track: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return contiguous v9.5.34 10 m-ray events for one ordered pair."""
    common = _crosses_common_track_items(subject_track, object_track)
    if len(common) < int(ARGS.crosses_in_front_min_observations):
        return []

    simultaneous_s = (
        int(common[-1][0]["timestamp_us"])
        - int(common[0][0]["timestamp_us"])
    ) / 1e6
    if simultaneous_s < float(
        ARGS.crosses_in_front_min_simultaneous_observation_s
    ):
        return []

    if _crosses_bbox_distance(
        subject_track, object_track
    ) > float(ARGS.crosses_in_front_max_pair_distance_m):
        return []

    events: list[dict[str, Any]] = []
    active_run: list[dict[str, Any]] = []

    for subject_item, object_item in common:
        candidate = _crosses_frame_candidate(
            subject_track,
            object_track,
            subject_item,
            object_item,
        )
        if candidate is None:
            if active_run:
                events.append(_crosses_finalize_active_run(active_run))
                active_run = []
            continue

        active_run.append(candidate)

    if active_run:
        events.append(_crosses_finalize_active_run(active_run))

    return events


def _crosses_find_events(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find directional finite-ray crossing runs without symmetric duplicates."""
    tracks = _crosses_build_tracks(snapshots)
    tokens = sorted(tracks)
    events: list[dict[str, Any]] = []

    # Evaluate each unordered pair once.  At any frame the ETA-order condition
    # normally permits at most one direction; if both survive due to numerical
    # edge cases, retain the stronger candidate for that timestamp.
    for first_index, first_token in enumerate(tokens):
        for second_token in tokens[first_index + 1 :]:
            first_track = tracks[first_token]
            second_track = tracks[second_token]

            forward_events = _crosses_evaluate_ordered_pair(
                first_track, second_track
            )
            reverse_events = _crosses_evaluate_ordered_pair(
                second_track, first_track
            )

            # Merge at frame level, then regroup so only one direction can be
            # active for an unordered pair at any timestamp.
            by_time: dict[int, dict[str, Any]] = {}
            for event in [*forward_events, *reverse_events]:
                for candidate in event["active_frames"]:
                    timestamp_us = int(candidate["timestamp_us"])
                    previous = by_time.get(timestamp_us)
                    if (
                        previous is None
                        or float(candidate["candidate_score"])
                        > float(previous["candidate_score"])
                    ):
                        by_time[timestamp_us] = candidate

            run: list[dict[str, Any]] = []
            previous_direction: Optional[tuple[str, str]] = None
            previous_common_index: Optional[int] = None

            common_times = [
                int(subject_item["timestamp_us"])
                for subject_item, _ in _crosses_common_track_items(
                    first_track, second_track
                )
            ]
            common_index_by_time = {
                timestamp_us: index
                for index, timestamp_us in enumerate(common_times)
            }

            for timestamp_us in sorted(by_time):
                candidate = by_time[timestamp_us]
                direction = (
                    str(candidate["subject_token"]),
                    str(candidate["object_token"]),
                )
                common_index = common_index_by_time.get(timestamp_us)
                contiguous = (
                    previous_common_index is None
                    or (
                        common_index is not None
                        and common_index == previous_common_index + 1
                    )
                )
                if run and (
                    direction != previous_direction or not contiguous
                ):
                    events.append(_crosses_finalize_active_run(run))
                    run = []
                run.append(candidate)
                previous_direction = direction
                previous_common_index = common_index

            if run:
                events.append(_crosses_finalize_active_run(run))

    return sorted(
        events,
        key=lambda row: (
            int(row["entry_time_us"]),
            str(row["subject_token"]),
            str(row["object_token"]),
        ),
    )


def derive_crosses_in_front_predicates(snapshots):
    """Emit v9.5.34 ``np:crossesInFrontOf(S,O)`` assertions.

    Definition at every active frame:

    * parked/map-irrelevant actors are rejected first;
    * S and O each contribute exactly one 10 m forward segment from the current
      geometric center along the current heading used by the video;
    * the two finite segments must intersect at a unique point;
    * the crossing angle must be within the configured angle limits; and
    * S must be moving and estimated to reach the conflict point before O.

    Assertions are emitted only for frames satisfying that exact geometry.
    Consequently the video cannot display ``crossesInFrontOf`` on a frame where
    its two displayed 10 m reference rays do not intersect.
    """
    if not ARGS.derive_semantic_predicates or not snapshots:
        return []

    snapshots = list(snapshots)
    assertions = []

    for event in _crosses_find_events(snapshots):
        run_start_us = int(event["entry_time_us"])

        for candidate in event["active_frames"]:
            current_time_us = int(candidate["timestamp_us"])
            subject_record = candidate["subject_item"]["record"]
            object_record = candidate["object_item"]["record"]
            frame = candidate["subject_item"]["snapshot"]["frame"]

            evidence = {
                "crosses_in_front_strategy": "v9_5_34_finite_10m_forward_rays",
                "interaction_semantics": (
                    "current_frame_path_intersection_with_subject_arriving_first"
                ),
                "temporal_assignment": "only_frames_with_valid_10m_ray_intersection",
                "subject_track_token": str(candidate["subject_token"]),
                "object_track_token": str(candidate["object_token"]),
                "subject_agent_type": str(candidate["subject_type"]),
                "object_agent_type": str(candidate["object_type"]),
                "conflict_x_m": float(candidate["conflict_x"]),
                "conflict_y_m": float(candidate["conflict_y"]),
                "crossing_time_us": current_time_us,
                "subject_conflict_entry_time_us": run_start_us,
                "subject_conflict_clearance_time_us": int(event["clearance_time_us"]),
                "current_timestamp_us": current_time_us,
                "subject_conflict_duration_s": float(event["duration_s"]),
                "crossing_angle_deg": float(candidate["crossing_angle_deg"]),
                "ray_length_m": float(candidate["ray_length_m"]),
                "subject_ray_distance_m": float(candidate["subject_ray_distance_m"]),
                "object_ray_distance_m": float(candidate["object_ray_distance_m"]),
                "subject_speed_mps": float(candidate["subject_speed_mps"]),
                "object_speed_mps": float(candidate["object_speed_mps"]),
                "subject_eta_s": float(candidate["subject_eta_s"]),
                "object_eta_s": candidate["object_eta_s"],
                "eta_gap_s": candidate["eta_gap_s"],
                "object_stationary_mode": bool(
                    candidate["object_stationary_mode"]
                ),
                "reference_geometry": str(candidate["reference_geometry"]),
                "map_context_filter": str(candidate["map_context_filter"]),
                "subject_map_filter_reason": str(
                    candidate["subject_map_filter_reason"]
                ),
                "object_map_filter_reason": str(
                    candidate["object_map_filter_reason"]
                ),
                "directional_candidate_score": float(
                    candidate["candidate_score"]
                ),
            }

            assertions.append(
                derived_relation(
                    CROSSES_IN_FRONT_PREDICATE,
                    subject_record["entity_id"],
                    object_record["entity_id"],
                    frame,
                    CROSSES_IN_FRONT_RULE_ID,
                    evidence,
                    start_time_us=run_start_us,
                    end_time_us=current_time_us,
                )
            )

    return assertions


# ============================================================================
# yieldsTo: retrospectively verified crossing-priority giving (v9.5.42)
# ============================================================================

YIELD_RULE_ID = "R-INTERACTION-YIELD-001"
YIELD_PREDICATE = "np:yieldsTo"


def _yield_speed(record: dict[str, Any]) -> float:
    speed = _crosses_record_speed(record)
    return 0.0 if speed is None or not math.isfinite(float(speed)) else float(speed)


def _yield_normalized_signal_state(value: Any) -> str:
    text = str(getattr(value, "name", value) if value is not None else "UNKNOWN").strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return {
        "AMBER": "YELLOW",
        "ORANGE": "YELLOW",
        "UNAVAILABLE": "UNKNOWN",
        "NONE": "UNKNOWN",
        "": "UNKNOWN",
    }.get(text, text if text in {"RED", "YELLOW", "GREEN", "UNKNOWN"} else "UNKNOWN")


def _yield_relevant_red_signal(item: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Conservatively identify ordinary pre-entry RED-signal stopping.

    This mirrors the v9.5.38 pre-entry signal semantics without depending on
    the traffic-light category being active in the same run.  A RED is treated
    as explanatory only when the subject is still on a lane and exactly one of
    its immediate outgoing connectors is currently signal controlled.
    """
    snapshot = item.get("snapshot") or {}
    record = item.get("record") or {}
    if str(record.get("primary_map_kind", "")) != "lane":
        return False, None

    grouped: dict[str, set[str]] = defaultdict(set)
    for row in snapshot.get("traffic_lights", []) or []:
        connector_id = str(row.get("connector_id", "")).strip()
        if not connector_id or connector_id.lower() == "unknown":
            continue
        grouped[connector_id].add(_yield_normalized_signal_state(row.get("status")))
    if not grouped:
        return False, None

    outgoing = {str(value) for value in (record.get("outgoing_object_ids", set()) or set())}
    controlled = sorted(outgoing & set(grouped))
    if len(controlled) != 1:
        return False, None

    connector_id = controlled[0]
    states = grouped.get(connector_id, set())
    canonical = next(iter(states)) if len(states) == 1 else "UNKNOWN"
    return canonical == "RED", connector_id


def _yield_window(
    track: list[dict[str, Any]],
    start_us: int,
    end_us: int,
) -> list[dict[str, Any]]:
    return [
        item for item in track
        if int(start_us) <= int(item["timestamp_us"]) <= int(end_us)
    ]


def _yield_conflict_distance(item: dict[str, Any], point: tuple[float, float]) -> float:
    x, y, _ = item["record"]["xyh"]
    return float(math.hypot(float(x) - float(point[0]), float(y) - float(point[1])))


def _yield_conflict_forward(item: dict[str, Any], point: tuple[float, float]) -> tuple[float, float]:
    x, y, _ = item["record"]["xyh"]
    heading = _crosses_preferred_heading(item)
    return _crosses_forward_lateral((float(x), float(y)), heading, point)


def _yield_point_outside_footprint(
    item: dict[str, Any],
    point: tuple[float, float],
    buffer_m: float,
) -> bool:
    if SHAPELY_AVAILABLE:
        try:
            polygon = oriented_box_polygon(item["record"])
            if polygon is not None:
                return float(polygon.distance(ShapelyPoint(float(point[0]), float(point[1])))) > float(buffer_m)
        except Exception:
            pass

    length_m, width_m = _crosses_record_dimensions(item["record"])
    radius = 0.5 * math.hypot(length_m, width_m)
    return _yield_conflict_distance(item, point) > radius + float(buffer_m)


def _yield_find_motion_concession(
    subject_track: list[dict[str, Any]],
    *,
    event_start_us: int,
    event_end_us: int,
    conflict_point: Optional[tuple[float, float]] = None,
    pre_window_s: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Find a significant pre-event speed concession and its onset."""
    pre_window_s = float(
        ARGS.yield_pre_event_window_s if pre_window_s is None else pre_window_s
    )
    window_start_us = int(event_start_us - round(pre_window_s * 1e6))
    items = _yield_window(subject_track, window_start_us, int(event_end_us))
    if len(items) < 2:
        return None

    baseline_items = [
        item for item in items
        if int(item["timestamp_us"]) <= int(event_start_us)
    ]
    if not baseline_items:
        return None

    baseline_speed = max(_yield_speed(item["record"]) for item in baseline_items)
    if baseline_speed + 1e-9 < float(ARGS.yield_min_baseline_speed_mps):
        return None

    eligible = []
    for item in items:
        if conflict_point is not None:
            distance = _yield_conflict_distance(item, conflict_point)
            forward_m, _ = _yield_conflict_forward(item, conflict_point)
            if distance > float(ARGS.yield_max_subject_conflict_distance_m):
                continue
            # The response must occur before the subject has passed the conflict.
            if forward_m < -0.25:
                continue
        eligible.append(item)
    if not eligible:
        return None

    minimum_item = min(eligible, key=lambda row: _yield_speed(row["record"]))
    minimum_speed = _yield_speed(minimum_item["record"])
    required_drop = max(
        float(ARGS.yield_min_speed_drop_mps),
        0.15 * float(baseline_speed),
    )
    measured_drop = max(0.0, float(baseline_speed) - float(minimum_speed))
    if measured_drop + 1e-9 < required_drop:
        return None

    reaction = None
    half_drop = 0.5 * required_drop
    for item in eligible:
        timestamp_us = int(item["timestamp_us"])
        if timestamp_us > int(event_end_us):
            break
        speed = _yield_speed(item["record"])
        acceleration = _finite_float(item["record"].get("estimated_acceleration_mps2"))
        decelerating = (
            acceleration is not None
            and acceleration <= -float(ARGS.acceleration_threshold_mps2)
        )
        if baseline_speed - speed >= half_drop or decelerating:
            # Do not mark an onset unless the verified full drop is observed
            # at or after this candidate onset.
            later_min = min(
                _yield_speed(row["record"])
                for row in eligible
                if int(row["timestamp_us"]) >= timestamp_us
            )
            if baseline_speed - later_min + 1e-9 >= required_drop:
                reaction = item
                break
    if reaction is None:
        return None

    red_connector_ids = []
    response_items = [
        item for item in eligible
        if int(reaction["timestamp_us"]) <= int(item["timestamp_us"]) <= int(event_end_us)
    ]
    for item in response_items:
        is_red, connector_id = _yield_relevant_red_signal(item)
        if is_red:
            red_connector_ids.append(str(connector_id))

    return {
        "reaction_item": reaction,
        "reaction_time_us": int(reaction["timestamp_us"]),
        "reaction_frame_index": int(reaction.get("frame_index", 0)),
        "baseline_speed_mps": float(baseline_speed),
        "minimum_speed_mps": float(minimum_speed),
        "speed_drop_mps": float(measured_drop),
        "required_speed_drop_mps": float(required_drop),
        "reached_slow_state": bool(minimum_speed <= float(ARGS.yield_slow_speed_mps)),
        "red_signal_detected": bool(red_connector_ids),
        "red_signal_connector_ids": sorted(set(red_connector_ids)),
    }


def _yield_initial_competition(
    subject_track: list[dict[str, Any]],
    object_track: list[dict[str, Any]],
    *,
    conflict_point: tuple[float, float],
    reaction_time_us: int,
) -> Optional[dict[str, Any]]:
    """Verify that the crossing was relevant before the subject conceded."""
    start_us = int(reaction_time_us - round(float(ARGS.yield_pre_event_window_s) * 1e6))
    subject_by_time = {
        int(item["timestamp_us"]): item
        for item in subject_track
        if start_us <= int(item["timestamp_us"]) <= int(reaction_time_us)
    }
    object_by_time = {
        int(item["timestamp_us"]): item
        for item in object_track
        if start_us <= int(item["timestamp_us"]) <= int(reaction_time_us)
    }
    common_times = sorted(set(subject_by_time).intersection(object_by_time), reverse=True)
    for timestamp_us in common_times:
        subject_item = subject_by_time[timestamp_us]
        object_item = object_by_time[timestamp_us]
        subject_speed = _yield_speed(subject_item["record"])
        object_speed = _yield_speed(object_item["record"])
        if subject_speed < 0.50 or object_speed < 0.30:
            continue

        subject_forward, _ = _yield_conflict_forward(subject_item, conflict_point)
        object_forward, _ = _yield_conflict_forward(object_item, conflict_point)
        if subject_forward <= 0.0 or object_forward <= 0.0:
            continue

        subject_distance = _yield_conflict_distance(subject_item, conflict_point)
        object_distance = _yield_conflict_distance(object_item, conflict_point)
        subject_eta = subject_distance / max(subject_speed, 0.05)
        object_eta = object_distance / max(object_speed, 0.05)
        eta_difference = subject_eta - object_eta
        if abs(eta_difference) <= float(ARGS.yield_max_initial_eta_difference_s):
            return {
                "timestamp_us": int(timestamp_us),
                "frame_index": int(subject_item.get("frame_index", 0)),
                "subject_eta_s": float(subject_eta),
                "object_eta_s": float(object_eta),
                "eta_difference_s": float(eta_difference),
                "subject_speed_mps": float(subject_speed),
                "object_speed_mps": float(object_speed),
                "subject_distance_to_conflict_m": float(subject_distance),
                "object_distance_to_conflict_m": float(object_distance),
            }
    return None


def _yield_observed_object_clearance(
    object_track: list[dict[str, Any]],
    *,
    conflict_point: tuple[float, float],
    crossing_time_us: int,
) -> Optional[dict[str, Any]]:
    """Find an observed post-conflict state of the priority road user."""
    end_us = int(crossing_time_us + round(float(ARGS.yield_post_event_window_s) * 1e6))
    items = _yield_window(
        object_track,
        int(crossing_time_us - round(float(ARGS.sample_interval_s) * 1e6)),
        end_us,
    )
    if len(items) < 2:
        return None

    # Require the observed trajectory to come close to the detector's conflict
    # point before accepting a later behind-conflict sample as clearance.
    nearest_index, nearest_item = min(
        enumerate(items),
        key=lambda pair: _yield_conflict_distance(pair[1], conflict_point),
    )
    length_m, width_m = _crosses_record_dimensions(nearest_item["record"])
    proximity_limit = max(3.0, 0.5 * math.hypot(length_m, width_m) + 1.0)
    nearest_distance = _yield_conflict_distance(nearest_item, conflict_point)
    if nearest_distance > proximity_limit:
        return None

    for item in items[nearest_index:]:
        if int(item["timestamp_us"]) < int(crossing_time_us):
            continue
        forward_m, lateral_m = _yield_conflict_forward(item, conflict_point)
        length_m, width_m = _crosses_record_dimensions(item["record"])
        clearance_threshold = 0.5 * length_m + float(ARGS.yield_conflict_clearance_buffer_m)
        if forward_m <= -clearance_threshold:
            return {
                "item": item,
                "time_us": int(item["timestamp_us"]),
                "frame_index": int(item.get("frame_index", 0)),
                "forward_m": float(forward_m),
                "lateral_m": float(lateral_m),
                "nearest_distance_m": float(nearest_distance),
            }
    return None


def _yield_subject_remains_outside_conflict(
    subject_track: list[dict[str, Any]],
    *,
    conflict_point: tuple[float, float],
    start_us: int,
    end_us: int,
) -> bool:
    items = _yield_window(subject_track, int(start_us), int(end_us))
    if not items:
        return False
    buffer_m = float(ARGS.yield_conflict_clearance_buffer_m)
    return all(
        _yield_point_outside_footprint(item, conflict_point, buffer_m)
        for item in items
    )


def _yield_subject_proceeds_after(
    subject_track: list[dict[str, Any]],
    *,
    after_time_us: int,
    anchor_item: dict[str, Any],
    conflict_point: Optional[tuple[float, float]] = None,
) -> Optional[dict[str, Any]]:
    end_us = int(after_time_us + round(float(ARGS.yield_post_event_window_s) * 1e6))
    x0, y0, _ = anchor_item["record"]["xyh"]
    for item in _yield_window(subject_track, int(after_time_us), end_us):
        if int(item["timestamp_us"]) <= int(after_time_us):
            continue
        speed = _yield_speed(item["record"])
        if speed + 1e-9 < float(ARGS.yield_min_proceed_speed_mps):
            continue
        x, y, _ = item["record"]["xyh"]
        displacement = math.hypot(float(x) - float(x0), float(y) - float(y0))
        if displacement + 1e-9 < float(ARGS.yield_min_proceed_displacement_m):
            continue

        if conflict_point is not None:
            reaction_distance = _yield_conflict_distance(anchor_item, conflict_point)
            current_distance = _yield_conflict_distance(item, conflict_point)
            forward_m, _ = _yield_conflict_forward(item, conflict_point)
            if current_distance >= reaction_distance and forward_m > 0.0:
                continue
        return {
            "item": item,
            "time_us": int(item["timestamp_us"]),
            "frame_index": int(item.get("frame_index", 0)),
            "speed_mps": float(speed),
            "displacement_from_reaction_m": float(displacement),
        }
    return None


def _yield_crossing_events_from_assertions(
    crossing_assertions,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for assertion in list(crossing_assertions or []):
        if str(getattr(assertion, "predicate_id", "")) != CROSSES_IN_FRONT_PREDICATE:
            continue
        evidence = dict(getattr(assertion, "evidence", {}) or {})
        subject_token = str(evidence.get("subject_track_token", ""))
        object_token = str(evidence.get("object_track_token", ""))
        if not subject_token or not object_token:
            continue
        event_start_us = int(
            evidence.get("subject_conflict_entry_time_us")
            or getattr(assertion, "start_time_us", 0)
            or getattr(assertion, "valid_time_us", 0)
        )
        key = (subject_token, object_token, event_start_us)
        groups[key].append(assertion)

    events = []
    for (priority_token, yielding_token, event_start_us), rows in groups.items():
        representative = min(
            rows,
            key=lambda assertion: max(
                float((assertion.evidence or {}).get("subject_ray_distance_m") or float("inf")),
                float((assertion.evidence or {}).get("object_ray_distance_m") or float("inf")),
            ),
        )
        evidence = dict(representative.evidence or {})
        try:
            conflict_point = (
                float(evidence["conflict_x_m"]),
                float(evidence["conflict_y_m"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        events.append({
            "priority_token": priority_token,
            "yielding_token": yielding_token,
            "event_start_us": int(event_start_us),
            "detector_clearance_time_us": int(
                evidence.get("subject_conflict_clearance_time_us")
                or max(int(getattr(row, "end_time_us", 0) or 0) for row in rows)
            ),
            "crossing_time_us": int(evidence.get("crossing_time_us") or getattr(representative.provenance, "timestamp_us", event_start_us)),
            "conflict_point": conflict_point,
            "crossing_angle_deg": evidence.get("crossing_angle_deg"),
            "representative_evidence": evidence,
        })
    return sorted(events, key=lambda row: (row["event_start_us"], row["yielding_token"], row["priority_token"]))



def _yield_emit_interval(
    *,
    snapshots_by_time: dict[int, dict[str, Any]],
    subject_token: str,
    object_token: str,
    start_time_us: int,
    end_time_us: int,
    base_evidence: dict[str, Any],
) -> list[Any]:
    assertions = []
    for timestamp_us in sorted(snapshots_by_time):
        if timestamp_us < int(start_time_us) or timestamp_us > int(end_time_us):
            continue
        snapshot = snapshots_by_time[timestamp_us]
        subject = snapshot.get("entities", {}).get(subject_token)
        obj = snapshot.get("entities", {}).get(object_token)
        if subject is None or obj is None:
            continue
        current_evidence = dict(base_evidence)
        current_evidence.update({
            "yield_active": True,
            "current_timestamp_us": int(timestamp_us),
            "current_frame_index": int(snapshot.get("frame_index", 0)),
            "current_subject_speed_mps": _yield_speed(subject),
            "current_object_speed_mps": _yield_speed(obj),
        })
        assertions.append(
            derived_relation(
                YIELD_PREDICATE,
                subject["entity_id"],
                obj["entity_id"],
                snapshot["frame"],
                YIELD_RULE_ID,
                current_evidence,
                start_time_us=int(start_time_us),
                end_time_us=int(timestamp_us),
            )
        )
    return assertions


def _derive_crossing_yields(
    snapshots: list[dict[str, Any]],
    tracks: dict[str, list[dict[str, Any]]],
    crossing_assertions,
) -> list[Any]:
    snapshots_by_time = {int(snapshot["timestamp_us"]): snapshot for snapshot in snapshots}
    assertions = []
    for event in _yield_crossing_events_from_assertions(crossing_assertions):
        subject_token = str(event["yielding_token"])
        object_token = str(event["priority_token"])
        subject_track = tracks.get(subject_token)
        object_track = tracks.get(object_token)
        if not subject_track or not object_track:
            continue

        # v9.5.39 initially supports motor/lane-using road users yielding to
        # vehicles, cyclists, or pedestrians. Pedestrian-as-yielder is omitted.
        subject_record = subject_track[0]["record"]
        object_record = object_track[0]["record"]
        if not is_vehicle_like(subject_record) or not is_dynamic_road_user(object_record):
            continue
        if _crosses_is_pedestrian(subject_record):
            continue

        conflict_point = event["conflict_point"]
        concession = _yield_find_motion_concession(
            subject_track,
            event_start_us=int(event["event_start_us"]),
            event_end_us=int(event["detector_clearance_time_us"]),
            conflict_point=conflict_point,
        )
        if concession is None or concession["red_signal_detected"]:
            continue

        competition = _yield_initial_competition(
            subject_track,
            object_track,
            conflict_point=conflict_point,
            reaction_time_us=int(concession["reaction_time_us"]),
        )
        if competition is None:
            continue

        clearance = _yield_observed_object_clearance(
            object_track,
            conflict_point=conflict_point,
            crossing_time_us=int(event["crossing_time_us"]),
        )
        if clearance is None:
            continue

        if not _yield_subject_remains_outside_conflict(
            subject_track,
            conflict_point=conflict_point,
            start_us=int(concession["reaction_time_us"]),
            end_us=int(clearance["time_us"]),
        ):
            continue

        proceed = _yield_subject_proceeds_after(
            subject_track,
            after_time_us=int(clearance["time_us"]),
            anchor_item=concession["reaction_item"],
            conflict_point=conflict_point,
        )
        if proceed is None:
            continue

        event_id = stable_id(
            str(snapshots[0].get("scenario_token", "")),
            YIELD_PREDICATE,
            "crossing_conflict",
            subject_token,
            object_token,
            int(concession["reaction_time_us"]),
            int(clearance["time_us"]),
        )
        base_evidence = {
            "yield_strategy": "v9_5_39_verified_behavioral_priority",
            "yield_event_id": event_id,
            "yield_mode": "crossing_conflict",
            "interaction_semantics": "subject_concedes_motion_and_priority_object_clears_first",
            "subject_track_token": subject_token,
            "object_track_token": object_token,
            "subject_agent_type": _crosses_canonical_type(subject_record),
            "object_agent_type": _crosses_canonical_type(object_record),
            "reaction_start_time_us": int(concession["reaction_time_us"]),
            "reaction_start_frame_index": int(concession["reaction_frame_index"]),
            "yield_end_time_us": int(clearance["time_us"]),
            "object_clearance_time_us": int(clearance["time_us"]),
            "subject_proceed_time_us": int(proceed["time_us"]),
            "subject_baseline_speed_mps": float(concession["baseline_speed_mps"]),
            "subject_minimum_speed_mps": float(concession["minimum_speed_mps"]),
            "subject_speed_drop_mps": float(concession["speed_drop_mps"]),
            "subject_required_speed_drop_mps": float(concession["required_speed_drop_mps"]),
            "subject_reached_slow_state": bool(concession["reached_slow_state"]),
            "initial_competition_time_us": int(competition["timestamp_us"]),
            "initial_subject_eta_s": float(competition["subject_eta_s"]),
            "initial_object_eta_s": float(competition["object_eta_s"]),
            "initial_eta_difference_s": float(competition["eta_difference_s"]),
            "conflict_x_m": float(conflict_point[0]),
            "conflict_y_m": float(conflict_point[1]),
            "yield_reference_x_m": float(conflict_point[0]),
            "yield_reference_y_m": float(conflict_point[1]),
            "subject_remained_outside_conflict": True,
            "object_proceeded_first": True,
            "subject_proceeded_after_object": True,
            "red_signal_exclusion_applied": True,
            "red_signal_detected_during_yield_response": False,
            "red_signal_connector_ids": [],
            "supporting_predicates": [CROSSES_IN_FRONT_PREDICATE],
            "crossing_angle_deg": event.get("crossing_angle_deg"),
            "crosses_in_front_direction": "object_crosses_in_front_of_subject",
            "crosses_detector_event_start_us": int(event["event_start_us"]),
            "crosses_detector_clearance_time_us": int(event["detector_clearance_time_us"]),
            "object_observed_clearance_frame_index": int(clearance["frame_index"]),
            "subject_proceed_frame_index": int(proceed["frame_index"]),
            "subject_proceed_speed_mps": float(proceed["speed_mps"]),
            "subject_proceed_displacement_m": float(proceed["displacement_from_reaction_m"]),
            "retrospectively_verified": True,
        }
        assertions.extend(_yield_emit_interval(
            snapshots_by_time=snapshots_by_time,
            subject_token=subject_token,
            object_token=object_token,
            start_time_us=int(concession["reaction_time_us"]),
            end_time_us=int(clearance["time_us"]),
            base_evidence=base_evidence,
        ))
    return assertions



def derive_yield_predicates(
    snapshots,
    *,
    crossing_assertions=None,
):
    """Emit v9.5.43 ``np:yieldsTo(S,O)`` assertions.

    ``np:crossesInFrontOf(O,S)`` establishes the local crossing order.
    ``np:yieldsTo(S,O)`` is emitted only when the subject additionally shows
    verified behavioral concession: initial temporal competition, a measurable
    speed reduction close to the crossing conflict, remaining outside while O
    clears, later subject progression, and exclusion of ordinary relevant
    RED-signal stopping.
    """
    if not ARGS.derive_semantic_predicates or not snapshots:
        return []

    snapshots = list(snapshots)
    if crossing_assertions is None:
        crossing_assertions = derive_crosses_in_front_predicates(snapshots)

    tracks = _crosses_build_tracks(snapshots)
    assertions = _derive_crossing_yields(
        snapshots,
        tracks,
        crossing_assertions,
    )

    unique = {}
    for assertion in assertions:
        unique.setdefault(str(assertion.assertion_id), assertion)
    return sorted(
        unique.values(),
        key=lambda assertion: (
            int(assertion.valid_time_us),
            int(assertion.end_time_us or assertion.valid_time_us),
            str(assertion.subject_id),
            str(assertion.object_id),
        ),
    )


def append_pair_instant_interactions(assertions, frame, pair_id, context, state=None):
    """Compatibility hook: v9.5.26 emits no instant interaction predicates.

    The supported interaction assertions are derived temporally by
    ``derive_following_predicates``, ``derive_lane_change_predicates``,
    ``derive_merging_predicates``, ``derive_crosses_in_front_predicates``,
    ``derive_yield_predicates``, and ``derive_overtaking_predicates``.
    Relative-motion measurements remain in the motion and risk categories.

    This function is intentionally retained because the directed-pair
    coordinator imports and calls it for every pair.
    """
    return None
