"""Map grounding, map topology, and map-owned pairwise measurements.

This module keeps the existing package architecture and public helper names, but
replaces the draft map matcher with a conservative geometry-first matcher.
Important design choices:

* Exact point membership and footprint intersection are represented separately.
* A primary lane/connector is emitted only when the best geometric candidate is
  unambiguous under explicit overlap and lateral-offset margins.
* Expert-route membership never changes the primary geometric map match.
* Lane and lane-connector parent objects are represented by separate predicates.
* Pair progress difference is restricted to one shared baseline, while signed
  path distance is reserved for same/successor/predecessor topology.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import *
from .geometry import oriented_box_polygon
from .motion import append_effective_travel_direction


MAP_RULE_ID = "R-MAP-MEMBERSHIP-EXACT-001"
PAIR_MAP_RULE_ID = "R-MAP-MOTION-SPATIAL-001"
ROUTE_RULE_ID = "R-EGO-RELEVANCE-001"


def _layer_cache_key(layer):
    return str(getattr(layer, "name", layer))


def _cached_map_object(state, map_api, object_id, layer):
    if not ARGS.map_cache or state is None:
        try:
            return map_api.get_map_object(str(object_id), layer)
        except Exception:
            return None
    cache = state.setdefault("map_object_cache", {})
    key = (str(object_id), _layer_cache_key(layer))
    if key not in cache:
        try:
            cache[key] = map_api.get_map_object(str(object_id), layer)
        except Exception:
            cache[key] = None
    return cache[key]


def _cached_proximal_map_objects(state, frame, x, y, radius, layers):
    resolution = float(getattr(ARGS, "proximal_map_cache_resolution_m", 0.0) or 0.0)
    if not ARGS.map_cache or state is None or resolution <= 0.0:
        return frame.map_api.get_proximal_map_objects(
            Point2D(float(x), float(y)), float(radius), layers
        )
    qx = round(float(x) / resolution)
    qy = round(float(y) / resolution)
    key = (
        qx,
        qy,
        round(float(radius), 3),
        tuple(sorted(_layer_cache_key(layer) for layer in layers)),
    )
    cache = state.setdefault("proximal_map_cache", {})
    if key not in cache:
        cache[key] = frame.map_api.get_proximal_map_objects(
            Point2D(float(x), float(y)), float(radius), layers
        )
    return cache[key]


def query_exact_lane_membership(frame: Any, x: float, y: float):
    """Return exact center-point lane and connector memberships."""
    if not MAP_QUERY_AVAILABLE or not SHAPELY_AVAILABLE:
        return []
    point = Point2D(float(x), float(y))
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    try:
        nearby = frame.map_api.get_proximal_map_objects(point, 8.0, layers)
    except Exception as exc:
        raise RuntimeError(f"Lane map query failed: {exc}") from exc

    memberships = []
    shapely_point = Point(float(x), float(y))
    for layer, objects in nearby.items():
        for map_object in objects:
            polygon = getattr(map_object, "polygon", None)
            if polygon is not None and polygon.covers(shapely_point):
                memberships.append((layer, str(getattr(map_object, "id", "unknown"))))
    return memberships


def _find_map_object(map_api, object_id, layer, state=None):
    return _cached_map_object(state, map_api, object_id, layer)


def _edge_id(edge):
    if edge is None:
        return None
    return str(getattr(edge, "id", edge))


def _adjacent_lane_ids(map_api, lane_id, state=None):
    """Return official nuPlan left/right adjacent lane IDs."""
    if not MAP_QUERY_AVAILABLE:
        return set(), set()
    lane = _find_map_object(map_api, lane_id, SemanticMapLayer.LANE, state=state)
    if lane is None:
        return set(), set()
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
    for attr, target in (
        ("left_adjacent_edge", left),
        ("adjacent_left", left),
        ("right_adjacent_edge", right),
        ("adjacent_right", right),
    ):
        value = getattr(lane, attr, None)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            item_id = _edge_id(item)
            if item_id:
                target.add(item_id)
    return left, right


def _map_edge_ids(map_object: Any, direction: str) -> set[str]:
    """Read predecessor/successor edge IDs across nuPlan object variants."""
    if map_object is None:
        return set()
    attributes = (
        ("incoming_edges", "incoming_edge", "predecessors", "predecessor")
        if direction == "incoming"
        else ("outgoing_edges", "outgoing_edge", "successors", "successor")
    )
    result: set[str] = set()
    for attr in attributes:
        value = getattr(map_object, attr, None)
        if value is None:
            continue
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            item_id = _edge_id(item)
            if item_id:
                result.add(item_id)
    return result


def _safe_parent(map_object: Any) -> Any:
    if map_object is None:
        return None
    value = getattr(map_object, "parent", None)
    try:
        return value() if callable(value) else value
    except Exception:
        return None


def _safe_parent_id(map_object: Any) -> Optional[str]:
    if map_object is None:
        return None
    try:
        value = map_object.get_roadblock_id()
        if value is not None and str(value):
            return str(value)
    except Exception:
        pass
    parent = _safe_parent(map_object)
    value = getattr(parent, "id", None) if parent is not None else None
    return str(value) if value is not None and str(value) else None


def _safe_intersection_id(map_object: Any) -> Optional[str]:
    parent = _safe_parent(map_object)
    for candidate in (map_object, parent):
        if candidate is None:
            continue
        value = getattr(candidate, "intersection", None)
        try:
            value = value() if callable(value) else value
        except Exception:
            value = None
        if value is not None and getattr(value, "id", None) is not None:
            return str(value.id)
    return None


def _safe_discrete_path(map_object: Any) -> list[Any]:
    baseline = getattr(map_object, "baseline_path", None)
    path = getattr(baseline, "discrete_path", None)
    if path is None:
        return []
    try:
        return list(path)
    except TypeError:
        return []


def _baseline_measurements(
    map_object: Any, x: float, y: float
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return nearest arc progress, signed left-positive offset, and heading."""
    baseline = getattr(map_object, "baseline_path", None)
    if baseline is not None:
        try:
            point = Point2D(float(x), float(y))
            progress = float(baseline.get_nearest_arc_length_from_position(point))
            pose = baseline.get_nearest_pose_from_position(point)
            heading = wrap_signed(float(pose.heading))
            dx, dy = float(x) - float(pose.x), float(y) - float(pose.y)
            lateral = -math.sin(heading) * dx + math.cos(heading) * dy
            return progress, float(lateral), heading
        except Exception:
            pass

    path = _safe_discrete_path(map_object)
    points = []
    for pose in path:
        try:
            points.append((float(pose.x), float(pose.y), wrap_signed(float(pose.heading))))
        except Exception:
            continue
    if not points:
        return None, None, None
    nearest_index = min(
        range(len(points)),
        key=lambda i: (points[i][0] - x) ** 2 + (points[i][1] - y) ** 2,
    )
    nx, ny, nh = points[nearest_index]
    progress = sum(
        math.hypot(
            points[i][0] - points[i - 1][0],
            points[i][1] - points[i - 1][1],
        )
        for i in range(1, nearest_index + 1)
    )
    dx, dy = x - nx, y - ny
    lateral = -math.sin(nh) * dx + math.cos(nh) * dy
    return float(progress), float(lateral), float(nh)


def _baseline_length(map_object: Any) -> Optional[float]:
    baseline = getattr(map_object, "baseline_path", None)
    value = getattr(baseline, "length", None)
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        path = _safe_discrete_path(map_object)
        if len(path) < 2:
            return None
        try:
            return float(
                sum(
                    math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))
                    for a, b in zip(path, path[1:])
                )
            )
        except Exception:
            return None


def _project_progress_on_map_object(map_object: Any, x: float, y: float) -> Optional[float]:
    return _baseline_measurements(map_object, x, y)[0]


def _candidate_query_radius(record: dict[str, Any]) -> float:
    dimensions = record.get("dimensions", {})
    length = float(dimensions.get("length_m") or 4.0)
    width = float(dimensions.get("width_m") or 2.0)
    half_diagonal = 0.5 * math.hypot(max(0.0, length), max(0.0, width))
    return max(8.0, half_diagonal + 4.0)


def _query_map_candidates(frame: Any, record: dict[str, Any], state=None) -> list[dict[str, Any]]:
    if not MAP_QUERY_AVAILABLE or not SHAPELY_AVAILABLE:
        return []
    x, y, _ = record["xyh"]
    direction_hint = record.get("velocity_heading_rad")
    direction_hint_source = "velocity"
    if direction_hint is None:
        direction_hint = record.get("displacement_heading_rad")
        direction_hint_source = "temporal_displacement"
    if direction_hint is None:
        direction_hint_source = "none"

    try:
        nearby = _cached_proximal_map_objects(
            state,
            frame,
            x,
            y,
            _candidate_query_radius(record),
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
    except Exception:
        return []

    footprint = oriented_box_polygon(record)
    center = Point(float(x), float(y))
    footprint_area = (
        float(footprint.area)
        if footprint is not None and math.isfinite(float(footprint.area)) and footprint.area > 0
        else None
    )
    candidates = []
    for layer, objects in nearby.items():
        layer_name = str(getattr(layer, "name", layer)).upper()
        kind = "lane_connector" if "LANE_CONNECTOR" in layer_name else "lane"
        for map_object in objects:
            polygon = getattr(map_object, "polygon", None)
            if polygon is None:
                continue
            try:
                center_covered = bool(polygon.covers(center))
            except Exception:
                center_covered = False
            intersection_area = 0.0
            overlap_ratio = 0.0
            if footprint is not None and footprint_area:
                try:
                    intersection_area = max(0.0, float(footprint.intersection(polygon).area))
                    overlap_ratio = max(0.0, min(1.0, intersection_area / footprint_area))
                except Exception:
                    intersection_area = 0.0
                    overlap_ratio = 0.0
            progress, lateral, map_heading = _baseline_measurements(map_object, x, y)
            heading_difference = (
                None
                if map_heading is None or direction_hint is None
                else angular_difference(float(direction_hint), float(map_heading))
            )
            object_id = str(getattr(map_object, "id", "unknown"))
            candidates.append(
                {
                    "kind": kind,
                    "object_id": object_id,
                    "entity_id": f"{kind}:{object_id}",
                    "map_object": map_object,
                    "center_covered": center_covered,
                    "intersection_area_m2": intersection_area,
                    "overlap_ratio": overlap_ratio,
                    "progress_m": progress,
                    "lateral_offset_m": lateral,
                    "map_heading_rad": map_heading,
                    "heading_difference_rad": heading_difference,
                    "direction_hint_source": direction_hint_source,
                    "parent_id": _safe_parent_id(map_object),
                    "intersection_id": _safe_intersection_id(map_object),
                    "baseline_length_m": _baseline_length(map_object),
                }
            )
    return candidates


def _candidate_sort_key(candidate: dict[str, Any]):
    lateral = candidate.get("lateral_offset_m")
    lateral_abs = abs(float(lateral)) if lateral is not None and math.isfinite(float(lateral)) else math.inf
    heading_difference = candidate.get("heading_difference_rad")
    heading_rank = (
        -float(heading_difference)
        if heading_difference is not None and math.isfinite(float(heading_difference))
        else -math.pi
    )
    # Geometry dominates. Heading is only a late deterministic tie-breaker and
    # never turns a geometrically inferior candidate into the winner.
    return (
        1 if candidate.get("center_covered") else 0,
        float(candidate.get("overlap_ratio") or 0.0),
        -lateral_abs,
        heading_rank,
        1 if candidate.get("kind") == "lane" else 0,
        str(candidate.get("object_id") or ""),
    )


def _select_primary_candidate(candidates: list[dict[str, Any]]):
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("center_covered")
        or float(candidate.get("overlap_ratio") or 0.0)
        >= float(ARGS.primary_lane_min_overlap_ratio)
    ]
    eligible.sort(key=_candidate_sort_key, reverse=True)
    if not eligible:
        return None, False, "no_eligible_candidate", eligible
    if len(eligible) == 1:
        return eligible[0], False, "single_eligible_candidate", eligible

    first, second = eligible[0], eligible[1]
    if bool(first.get("center_covered")) != bool(second.get("center_covered")):
        return first, False, "center_coverage_dominates", eligible

    overlap_margin = float(first.get("overlap_ratio") or 0.0) - float(
        second.get("overlap_ratio") or 0.0
    )
    if overlap_margin >= float(ARGS.primary_lane_ambiguity_margin):
        return first, False, "overlap_margin", eligible

    first_lateral = first.get("lateral_offset_m")
    second_lateral = second.get("lateral_offset_m")
    lateral_margin = float(getattr(ARGS, "primary_map_lateral_tiebreak_margin_m", 0.50))
    if (
        first_lateral is not None
        and second_lateral is not None
        and math.isfinite(float(first_lateral))
        and math.isfinite(float(second_lateral))
        and abs(float(first_lateral)) + lateral_margin < abs(float(second_lateral))
    ):
        return first, False, "lateral_offset_margin", eligible

    return None, True, "top_candidates_indistinguishable", eligible


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": candidate.get("entity_id"),
        "kind": candidate.get("kind"),
        "object_id": candidate.get("object_id"),
        "center_covered": bool(candidate.get("center_covered")),
        "intersection_area_m2": candidate.get("intersection_area_m2"),
        "overlap_ratio": candidate.get("overlap_ratio"),
        "progress_m": candidate.get("progress_m"),
        "lateral_offset_m": candidate.get("lateral_offset_m"),
        "map_heading_rad": candidate.get("map_heading_rad"),
        "heading_difference_rad": candidate.get("heading_difference_rad"),
        "parent_id": candidate.get("parent_id"),
        "intersection_id": candidate.get("intersection_id"),
        "baseline_length_m": candidate.get("baseline_length_m"),
    }


def _query_context_map_layers(frame, record, state=None) -> list[dict[str, Any]]:
    if not MAP_QUERY_AVAILABLE or not SHAPELY_AVAILABLE:
        return []
    layers = []
    if hasattr(SemanticMapLayer, "INTERSECTION"):
        layers.append(SemanticMapLayer.INTERSECTION)
    if hasattr(SemanticMapLayer, "CROSSWALK"):
        layers.append(SemanticMapLayer.CROSSWALK)
    if not layers:
        return []

    x, y, _ = record["xyh"]
    footprint = oriented_box_polygon(record)
    center = Point(float(x), float(y))
    try:
        nearby = _cached_proximal_map_objects(state, frame, x, y, 12.0, layers)
    except Exception:
        return []

    results = []
    for layer, objects in nearby.items():
        layer_name = str(getattr(layer, "name", layer)).upper()
        kind = "crosswalk" if "CROSSWALK" in layer_name else "intersection"
        for map_object in objects:
            polygon = getattr(map_object, "polygon", None)
            if polygon is None:
                continue
            try:
                center_covered = bool(polygon.covers(center))
            except Exception:
                center_covered = False
            intersection_area = 0.0
            if footprint is not None:
                try:
                    intersection_area = max(0.0, float(footprint.intersection(polygon).area))
                except Exception:
                    intersection_area = 0.0
            results.append(
                {
                    "kind": kind,
                    "object_id": str(getattr(map_object, "id", "unknown")),
                    "map_object": map_object,
                    "center_covered": center_covered,
                    "intersection_area_m2": intersection_area,
                }
            )
    return results


def _append_context_map_assertions(assertions, frame, record, context_objects):
    record["intersection_ids"] = set()
    record["intersection_footprint_ids"] = set()
    record["crosswalk_ids"] = set()
    record["crosswalk_footprint_ids"] = set()
    record["in_crosswalk"] = False
    area_epsilon = float(getattr(ARGS, "overlap_area_epsilon_m2", 1e-4))

    for item in context_objects:
        kind = item["kind"]
        object_id = item["object_id"]
        entity_id = f"{kind}:{object_id}"
        evidence = {
            "map_object_id": object_id,
            "map_object_kind": kind,
            "center_covered": bool(item["center_covered"]),
            "footprint_intersection_area_m2": float(item["intersection_area_m2"]),
            "overlap_area_epsilon_m2": area_epsilon,
        }
        if kind == "intersection":
            if item["center_covered"]:
                record["intersection_ids"].add(object_id)
                assertions.append(
                    derived_relation(
                        "np:inIntersection",
                        record["entity_id"],
                        entity_id,
                        frame,
                        MAP_RULE_ID,
                        evidence,
                    )
                )
            if float(item["intersection_area_m2"]) > area_epsilon:
                record["intersection_footprint_ids"].add(object_id)
                assertions.append(
                    derived_relation(
                        "np:intersectsIntersection",
                        record["entity_id"],
                        entity_id,
                        frame,
                        MAP_RULE_ID,
                        evidence,
                    )
                )
        else:
            if item["center_covered"]:
                record["crosswalk_ids"].add(object_id)
                record["in_crosswalk"] = True
                assertions.append(
                    derived_relation(
                        "np:inCrosswalk",
                        record["entity_id"],
                        entity_id,
                        frame,
                        MAP_RULE_ID,
                        evidence,
                    )
                )
            if float(item["intersection_area_m2"]) > area_epsilon:
                record["crosswalk_footprint_ids"].add(object_id)
                record["in_crosswalk"] = True
                assertions.append(
                    derived_relation(
                        "np:intersectsCrosswalk",
                        record["entity_id"],
                        entity_id,
                        frame,
                        MAP_RULE_ID,
                        evidence,
                    )
                )


def append_map_memberships(assertions, frame, record, state=None):
    """Append map-grounding facts and populate map context on an entity record."""
    candidates = _query_map_candidates(frame, record, state=state)
    context_objects = _query_context_map_layers(frame, record, state=state)
    selected, ambiguous, selection_reason, eligible = _select_primary_candidate(candidates)

    record["map_candidates"] = candidates
    record["lane_ids"] = set()
    record["footprint_lane_ids"] = set()
    record["primary_lane_id"] = None
    record["primary_connector_id"] = None
    record["primary_map_id"] = None
    record["primary_map_object_id"] = None
    record["primary_map_kind"] = None
    record["map_heading_rad"] = None
    record["baseline_progress_m"] = None
    record["baseline_lateral_offset_m"] = None
    record["baseline_curvature_1pm"] = None
    record["baseline_length_m"] = None
    record["map_speed_limit_mps"] = None
    record["left_adjacent_object_ids"] = set()
    record["right_adjacent_object_ids"] = set()
    record["incoming_object_ids"] = set()
    record["outgoing_object_ids"] = set()
    record["map_match_ambiguous"] = bool(ambiguous)
    record["map_match_status"] = "ambiguous" if ambiguous else "unmatched"
    record["primary_map_object"] = None
    record["roadblock_id"] = None
    record["parent_roadblock_kind"] = None
    record["primary_map_intersection_id"] = None
    record["route_roadblock_rank"] = None
    record["on_expert_route"] = False
    record["on_ego_route_corridor"] = False

    _append_context_map_assertions(assertions, frame, record, context_objects)

    x, y, _ = record["xyh"]
    area_epsilon = float(getattr(ARGS, "overlap_area_epsilon_m2", 1e-4))
    for candidate in candidates:
        membership_evidence = {
            "center_point": [float(x), float(y)],
            "map_object_id": candidate["object_id"],
            "map_object_kind": candidate["kind"],
            "center_covered": bool(candidate["center_covered"]),
            "footprint_intersection_area_m2": float(candidate["intersection_area_m2"]),
            "footprint_overlap_ratio": float(candidate["overlap_ratio"]),
            "overlap_area_epsilon_m2": area_epsilon,
        }
        if candidate["center_covered"]:
            pid = (
                "np:inLaneConnector"
                if candidate["kind"] == "lane_connector"
                else "np:inLane"
            )
            record["lane_ids"].add(candidate["entity_id"])
            assertions.append(
                derived_relation(
                    pid,
                    record["entity_id"],
                    candidate["entity_id"],
                    frame,
                    MAP_RULE_ID,
                    membership_evidence,
                )
            )
        if float(candidate["intersection_area_m2"]) > area_epsilon:
            pid = (
                "np:intersectsLaneConnector"
                if candidate["kind"] == "lane_connector"
                else "np:intersectsLane"
            )
            record["footprint_lane_ids"].add(candidate["entity_id"])
            assertions.append(
                derived_relation(
                    pid,
                    record["entity_id"],
                    candidate["entity_id"],
                    frame,
                    MAP_RULE_ID,
                    membership_evidence,
                )
            )

    if ambiguous:
        append_positive_boolean_semantic(
            assertions,
            "np:hasAmbiguousMapMatch",
            record["entity_id"],
            True,
            frame,
            MAP_RULE_ID,
            {
                "selection_reason": selection_reason,
                "candidate_count": len(candidates),
                "eligible_candidate_count": len(eligible),
                "primary_lane_min_overlap_ratio": ARGS.primary_lane_min_overlap_ratio,
                "primary_lane_ambiguity_margin": ARGS.primary_lane_ambiguity_margin,
                "primary_map_lateral_tiebreak_margin_m": getattr(
                    ARGS, "primary_map_lateral_tiebreak_margin_m", 0.50
                ),
                "candidate_summaries": [_candidate_summary(item) for item in eligible[:5]],
            },
            state=state,
        )

    if selected is None:
        append_effective_travel_direction(assertions, frame, record, state)
        return

    record["map_match_status"] = "matched"
    record["primary_map_id"] = selected["entity_id"]
    record["primary_map_object_id"] = selected["object_id"]
    record["primary_map_kind"] = selected["kind"]
    selected_object = selected["map_object"]
    record["primary_map_object"] = selected_object
    record["incoming_object_ids"] = _map_edge_ids(selected_object, "incoming")
    record["outgoing_object_ids"] = _map_edge_ids(selected_object, "outgoing")
    record["baseline_length_m"] = selected.get("baseline_length_m")

    if selected["kind"] == "lane":
        left_ids, right_ids = _adjacent_lane_ids(
            frame.map_api, selected["object_id"], state=state
        )
        record["left_adjacent_object_ids"] = set(left_ids)
        record["right_adjacent_object_ids"] = set(right_ids)
        record["primary_lane_id"] = selected["entity_id"]
        primary_predicate = "np:hasPrimaryLane"
    else:
        record["primary_connector_id"] = selected["entity_id"]
        primary_predicate = "np:hasPrimaryLaneConnector"

    evidence = {
        "map_object_id": selected["object_id"],
        "map_object_kind": selected["kind"],
        "selected_entity_id": selected["entity_id"],
        "selection_reason": selection_reason,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "center_covered": bool(selected["center_covered"]),
        "footprint_intersection_area_m2": float(selected["intersection_area_m2"]),
        "footprint_overlap_ratio": float(selected["overlap_ratio"]),
        "baseline_progress_m": selected["progress_m"],
        "baseline_lateral_offset_m": selected["lateral_offset_m"],
        "map_heading_rad": selected["map_heading_rad"],
        "heading_difference_rad": selected["heading_difference_rad"],
        "baseline_length_m": selected["baseline_length_m"],
        "primary_lane_min_overlap_ratio": ARGS.primary_lane_min_overlap_ratio,
        "primary_lane_ambiguity_margin": ARGS.primary_lane_ambiguity_margin,
        "candidate_summaries": [_candidate_summary(item) for item in eligible[:5]],
    }
    assertions.append(
        derived_relation(
            primary_predicate,
            record["entity_id"],
            selected["entity_id"],
            frame,
            MAP_RULE_ID,
            evidence,
        )
    )
    assertions.append(
        derived_value(
            "np:hasPrimaryMapOverlapRatio",
            record["entity_id"],
            float(selected["overlap_ratio"]),
            ValueType.FLOAT,
            frame,
            MAP_RULE_ID,
            evidence,
        )
    )

    parent_id = selected.get("parent_id") or _safe_parent_id(selected_object)
    if parent_id:
        record["roadblock_id"] = str(parent_id)
        if selected["kind"] == "lane":
            record["parent_roadblock_kind"] = "roadblock"
            parent_entity_id = f"roadblock:{parent_id}"
            parent_predicate = "np:hasParentRoadblock"
        else:
            record["parent_roadblock_kind"] = "roadblock_connector"
            parent_entity_id = f"roadblock_connector:{parent_id}"
            parent_predicate = "np:hasParentRoadblockConnector"
        assertions.append(
            derived_relation(
                parent_predicate,
                record["entity_id"],
                parent_entity_id,
                frame,
                MAP_RULE_ID,
                {
                    **evidence,
                    "parent_id": str(parent_id),
                    "parent_kind": record["parent_roadblock_kind"],
                },
            )
        )

        route_rank_map = (
            state.get("current_route_roadblock_rank", {}) if state is not None else {}
        )
        if str(parent_id) in route_rank_map:
            rank = int(route_rank_map[str(parent_id)])
            record["route_roadblock_rank"] = rank
            record["on_expert_route"] = True
            record["on_ego_route_corridor"] = True
            assertions.append(
                derived_value(
                    "np:hasRouteRoadblockRank",
                    record["entity_id"],
                    rank,
                    ValueType.INTEGER,
                    frame,
                    ROUTE_RULE_ID,
                    {"roadblock_id": str(parent_id), "route_rank": rank},
                )
            )

    topological_intersection_id = selected.get("intersection_id") or _safe_intersection_id(
        selected_object
    )
    if topological_intersection_id:
        record["primary_map_intersection_id"] = str(topological_intersection_id)
        assertions.append(
            derived_relation(
                "np:hasPrimaryMapIntersection",
                record["entity_id"],
                f"intersection:{topological_intersection_id}",
                frame,
                MAP_RULE_ID,
                {
                    **evidence,
                    "intersection_id": str(topological_intersection_id),
                    "relation_kind": "primary_map_object_topology",
                },
            )
        )

    speed_limit = getattr(selected_object, "speed_limit_mps", None)
    try:
        speed_limit = float(speed_limit)
        if math.isfinite(speed_limit) and speed_limit >= 0.0:
            record["map_speed_limit_mps"] = speed_limit
            assertions.append(
                derived_value(
                    "np:hasMapSpeedLimit",
                    record["entity_id"],
                    speed_limit,
                    ValueType.FLOAT,
                    frame,
                    MAP_RULE_ID,
                    {**evidence, "map_speed_limit_mps": speed_limit},
                )
            )
    except (TypeError, ValueError):
        pass

    if selected["progress_m"] is not None:
        record["baseline_progress_m"] = float(selected["progress_m"])
        assertions.append(
            derived_value(
                "np:hasBaselineProgress",
                record["entity_id"],
                float(selected["progress_m"]),
                ValueType.FLOAT,
                frame,
                MAP_RULE_ID,
                evidence,
            )
        )
        baseline = getattr(selected_object, "baseline_path", None)
        try:
            curvature = float(
                baseline.get_curvature_at_arc_length(float(selected["progress_m"]))
            )
            if math.isfinite(curvature):
                record["baseline_curvature_1pm"] = curvature
                assertions.append(
                    derived_value(
                        "np:hasBaselineCurvature",
                        record["entity_id"],
                        curvature,
                        ValueType.FLOAT,
                        frame,
                        MAP_RULE_ID,
                        {**evidence, "baseline_curvature_1pm": curvature},
                    )
                )
        except Exception:
            pass

    if selected["lateral_offset_m"] is not None:
        record["baseline_lateral_offset_m"] = float(selected["lateral_offset_m"])
        assertions.append(
            derived_value(
                "np:hasBaselineLateralOffset",
                record["entity_id"],
                float(selected["lateral_offset_m"]),
                ValueType.FLOAT,
                frame,
                MAP_RULE_ID,
                evidence,
            )
        )

    if selected["map_heading_rad"] is not None:
        record["map_heading_rad"] = wrap_signed(float(selected["map_heading_rad"]))
        assertions.append(
            derived_value(
                "np:hasMapHeading",
                record["entity_id"],
                record["map_heading_rad"],
                ValueType.FLOAT,
                frame,
                MAP_RULE_ID,
                evidence,
            )
        )
    append_effective_travel_direction(assertions, frame, record, state)


def _same_primary_map(subject: dict[str, Any], obj: dict[str, Any]) -> Optional[str]:
    if subject.get("map_match_ambiguous") or obj.get("map_match_ambiguous"):
        return None
    sid = subject.get("primary_map_id")
    oid = obj.get("primary_map_id")
    return sid if sid is not None and sid == oid else None


def _same_primary_lane(subject: dict[str, Any], obj: dict[str, Any]) -> Optional[str]:
    shared = _same_primary_map(subject, obj)
    if shared is None:
        return None
    if subject.get("primary_map_kind") != "lane" or obj.get("primary_map_kind") != "lane":
        return None
    return shared


def _record_direction(record: dict[str, Any]) -> DirectionEvidence:
    value = record.get("travel_direction")
    if isinstance(value, DirectionEvidence):
        return value
    return DirectionEvidence(
        heading_rad=None,
        source="unreliable",
        reliable=False,
        omission_reason="travel_direction_not_computed",
        map_heading_rad=record.get("map_heading_rad"),
        velocity_heading_rad=record.get("velocity_heading_rad"),
        displacement_heading_rad=record.get("displacement_heading_rad"),
        velocity_displacement_difference_rad=None,
        map_motion_difference_rad=None,
    )


def _shared_intersection(subject: dict[str, Any], obj: dict[str, Any]) -> bool:
    subject_ids = set(subject.get("intersection_ids", set())) | set(
        subject.get("intersection_footprint_ids", set())
    )
    object_ids = set(obj.get("intersection_ids", set())) | set(
        obj.get("intersection_footprint_ids", set())
    )
    subject_primary = subject.get("primary_map_intersection_id")
    object_primary = obj.get("primary_map_intersection_id")
    if subject_primary:
        subject_ids.add(str(subject_primary))
    if object_primary:
        object_ids.add(str(object_primary))
    return bool(subject_ids & object_ids)


def _pair_map_relation(subject: dict[str, Any], obj: dict[str, Any]) -> str:
    """Return the conservative topological relation of object map to subject map."""
    if subject.get("map_match_ambiguous") or obj.get("map_match_ambiguous"):
        return "ambiguous"
    sid = subject.get("primary_map_object_id")
    oid = obj.get("primary_map_object_id")
    if not sid or not oid:
        return "unrelated"
    if sid == oid and subject.get("primary_map_kind") == obj.get("primary_map_kind"):
        return "same_map"
    if oid in subject.get("left_adjacent_object_ids", set()):
        return "adjacent_left"
    if oid in subject.get("right_adjacent_object_ids", set()):
        return "adjacent_right"
    if sid in obj.get("right_adjacent_object_ids", set()):
        return "adjacent_left"
    if sid in obj.get("left_adjacent_object_ids", set()):
        return "adjacent_right"
    if oid in subject.get("outgoing_object_ids", set()) or sid in obj.get(
        "incoming_object_ids", set()
    ):
        return "successor"
    if oid in subject.get("incoming_object_ids", set()) or sid in obj.get(
        "outgoing_object_ids", set()
    ):
        return "predecessor"
    return "unrelated"


def _same_baseline_progress_difference(
    subject: dict[str, Any], obj: dict[str, Any], map_relation: str
) -> Optional[float]:
    if map_relation != "same_map":
        return None
    subject_progress = subject.get("baseline_progress_m")
    object_progress = obj.get("baseline_progress_m")
    if subject_progress is None or object_progress is None:
        return None
    return float(object_progress) - float(subject_progress)


def _signed_path_distance(
    subject: dict[str, Any], obj: dict[str, Any], map_relation: str
) -> Optional[float]:
    """Return signed along-path distance for same/successor/predecessor only."""
    relation = str(map_relation)
    subject_progress = subject.get("baseline_progress_m")
    object_progress = obj.get("baseline_progress_m")
    if relation == "same_map" and subject_progress is not None and object_progress is not None:
        return float(object_progress) - float(subject_progress)
    if relation == "successor" and subject_progress is not None and object_progress is not None:
        subject_length = subject.get("baseline_length_m") or _baseline_length(
            subject.get("primary_map_object")
        )
        if subject_length is not None:
            return max(0.0, float(subject_length) - float(subject_progress)) + max(
                0.0, float(object_progress)
            )
    if relation == "predecessor" and subject_progress is not None and object_progress is not None:
        object_length = obj.get("baseline_length_m") or _baseline_length(
            obj.get("primary_map_object")
        )
        if object_length is not None:
            return -(
                max(0.0, float(object_length) - float(object_progress))
                + max(0.0, float(subject_progress))
            )
    return None


def _map_longitudinal_reference_difference(
    subject: dict[str, Any], obj: dict[str, Any], map_relation: str
) -> Optional[float]:
    """Longitudinal coordinate used internally by spatial/relevance logic.

    For same/successor/predecessor topology this equals signed path distance.
    For adjacent lanes only, both positions are projected onto the subject
    baseline to preserve the established longitudinal comparison behaviour.
    This adjacent projection is intentionally not serialized as
    ``np:hasSignedPathDistanceTo`` because it is not a lane-graph path length.
    """
    path_distance = _signed_path_distance(subject, obj, map_relation)
    if path_distance is not None:
        return path_distance
    if map_relation not in {"adjacent_left", "adjacent_right"}:
        return None
    subject_map = subject.get("primary_map_object")
    if subject_map is None:
        return None
    sx, sy, _ = subject["xyh"]
    ox, oy, _ = obj["xyh"]
    subject_projection = _project_progress_on_map_object(subject_map, sx, sy)
    object_projection = _project_progress_on_map_object(subject_map, ox, oy)
    if subject_projection is None or object_projection is None:
        return None
    return float(object_projection) - float(subject_projection)


def _map_pair_evidence(context: dict[str, Any]) -> dict[str, Any]:
    subject = context["subject"]
    obj = context["object"]
    evidence = dict(context["evidence"])
    evidence.update(
        {
            "subject_primary_map_object_id": subject.get("primary_map_object_id"),
            "object_primary_map_object_id": obj.get("primary_map_object_id"),
            "subject_primary_map_kind": subject.get("primary_map_kind"),
            "object_primary_map_kind": obj.get("primary_map_kind"),
            "subject_map_match_ambiguous": bool(subject.get("map_match_ambiguous")),
            "object_map_match_ambiguous": bool(obj.get("map_match_ambiguous")),
            "subject_baseline_progress_m": subject.get("baseline_progress_m"),
            "object_baseline_progress_m": obj.get("baseline_progress_m"),
            "subject_baseline_length_m": subject.get("baseline_length_m"),
            "object_baseline_length_m": obj.get("baseline_length_m"),
            "subject_left_adjacent_object_ids": sorted(
                subject.get("left_adjacent_object_ids", set())
            ),
            "subject_right_adjacent_object_ids": sorted(
                subject.get("right_adjacent_object_ids", set())
            ),
            "subject_incoming_object_ids": sorted(subject.get("incoming_object_ids", set())),
            "subject_outgoing_object_ids": sorted(subject.get("outgoing_object_ids", set())),
            "object_left_adjacent_object_ids": sorted(obj.get("left_adjacent_object_ids", set())),
            "object_right_adjacent_object_ids": sorted(obj.get("right_adjacent_object_ids", set())),
            "object_incoming_object_ids": sorted(obj.get("incoming_object_ids", set())),
            "object_outgoing_object_ids": sorted(obj.get("outgoing_object_ids", set())),
            "subject_intersection_ids": sorted(
                set(subject.get("intersection_ids", set()))
                | set(subject.get("intersection_footprint_ids", set()))
                | ({str(subject["primary_map_intersection_id"])} if subject.get("primary_map_intersection_id") else set())
            ),
            "object_intersection_ids": sorted(
                set(obj.get("intersection_ids", set()))
                | set(obj.get("intersection_footprint_ids", set()))
                | ({str(obj["primary_map_intersection_id"])} if obj.get("primary_map_intersection_id") else set())
            ),
            "map_relation": context["map_relation"],
            "map_progress_delta_m": context.get("map_progress_delta_m"),
            "same_baseline_progress_delta_m": context.get(
                "same_baseline_progress_delta_m"
            ),
            "signed_path_distance_m": context.get("signed_path_distance_m"),
        }
    )
    return evidence


def append_pair_map_measurements(assertions, frame, pair_id, context):
    """Emit pairwise measurements owned by the map category."""
    evidence = _map_pair_evidence(context)
    progress_delta = context.get("same_baseline_progress_delta_m")
    if progress_delta is not None and math.isfinite(float(progress_delta)):
        assertions.append(
            derived_value(
                "np:hasMapProgressDifferenceTo",
                pair_id,
                float(progress_delta),
                ValueType.FLOAT,
                frame,
                PAIR_MAP_RULE_ID,
                evidence,
            )
        )
    signed_path = context.get("signed_path_distance_m")
    if signed_path is not None and math.isfinite(float(signed_path)):
        assertions.append(
            derived_value(
                "np:hasSignedPathDistanceTo",
                pair_id,
                float(signed_path),
                ValueType.FLOAT,
                frame,
                ROUTE_RULE_ID,
                evidence,
            )
        )
    assertions.append(
        derived_value(
            "np:hasSpatialMapRelation",
            pair_id,
            context["map_relation"],
            ValueType.STRING,
            frame,
            PAIR_MAP_RULE_ID,
            evidence,
        )
    )


def append_pair_map_semantics(assertions, frame, pair_id, context, state=None):
    subject = context["subject"]
    obj = context["object"]
    evidence = _map_pair_evidence(context)
    if _same_primary_lane(subject, obj) is not None:
        append_positive_boolean_semantic(
            assertions,
            "np:inSameLaneAs",
            pair_id,
            True,
            frame,
            MAP_RULE_ID,
            evidence,
            state=state,
        )
    if _shared_intersection(subject, obj):
        append_positive_boolean_semantic(
            assertions,
            "np:sharesIntersectionWith",
            pair_id,
            True,
            frame,
            MAP_RULE_ID,
            evidence,
            state=state,
        )


def _query_stop_lines(frame, x, y, state=None):
    if not MAP_QUERY_AVAILABLE or not SHAPELY_AVAILABLE:
        return []
    if not hasattr(SemanticMapLayer, "STOP_LINE"):
        return []
    try:
        nearby = _cached_proximal_map_objects(
            state,
            frame,
            x,
            y,
            float(ARGS.stop_line_search_radius_m),
            [SemanticMapLayer.STOP_LINE],
        )
    except Exception:
        return []
    result = []
    for stop_line in nearby.get(SemanticMapLayer.STOP_LINE, []):
        polygon = getattr(stop_line, "polygon", None)
        if polygon is None:
            continue
        result.append((str(getattr(stop_line, "id", "unknown")), polygon, stop_line))
    return result


def _traffic_light_map(snapshot):
    result = {}
    for item in snapshot.get("traffic_lights", []):
        connector_id = str(item["connector_id"])
        result[connector_id] = item
    return result
