#!/usr/bin/env python3
"""Conservative relevance and route-aware pair selection for nuPlan predicates.

The extractor should retain every tracked object as an entity, but it should not
create ego-agent pair nodes for every object.  A pair is created only when there
is positive evidence that the object matters to the ego's current or near-future
motion context.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RelevanceThresholds:
    # Hard guardrails
    hard_max_center_distance_m: float = 120.0
    max_relevant_ego_agents: int = 14
    max_noncritical_relevant_agents: int = 10

    # Same-route / same-flow relevance
    same_lane_ahead_m: float = 100.0
    same_lane_behind_m: float = 35.0
    adjacent_lane_m: float = 45.0
    connected_lane_m: float = 70.0
    route_roadblock_lookahead: int = 3
    route_roadblock_lookbehind: int = 1

    # Immediate geometric relevance
    always_relevant_free_space_m: float = 6.0
    vulnerable_road_user_distance_m: float = 35.0
    static_obstacle_distance_m: float = 25.0

    # Constant-velocity conflict screening
    cpa_horizon_s: float = 8.0
    cpa_clearance_margin_m: float = 2.0
    critical_cpa_time_s: float = 4.0
    critical_cpa_clearance_m: float = 1.0

    # Cross-traffic / intersection relevance
    shared_intersection_distance_m: float = 55.0
    traffic_control_distance_m: float = 45.0

    # Directional semantics should be stricter than pair relevance
    directional_same_lane_ahead_m: float = 80.0
    directional_same_lane_behind_m: float = 25.0
    directional_adjacent_m: float = 35.0
    directional_connected_m: float = 50.0


DEFAULT_RELEVANCE_THRESHOLDS = RelevanceThresholds()


def relevance_thresholds_as_dict() -> dict[str, float | int]:
    return asdict(DEFAULT_RELEVANCE_THRESHOLDS)


@dataclass(frozen=True)
class CpaResult:
    time_s: Optional[float]
    center_distance_m: Optional[float]
    clearance_m: Optional[float]
    closing: bool


@dataclass(frozen=True)
class PairRelevanceDecision:
    relevant: bool
    critical: bool
    score: float
    reasons: tuple[str, ...]
    primary_reason: str
    map_relation: str
    route_relation: str
    center_distance_m: Optional[float]
    free_space_distance_m: Optional[float]
    signed_path_distance_m: Optional[float]
    cpa_time_s: Optional[float]
    cpa_center_distance_m: Optional[float]
    cpa_clearance_m: Optional[float]
    directional_semantics_allowed: bool
    directional_omission_reason: Optional[str]
    mandatory_selection: bool = False


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _xy(record: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    pose = record.get("xyh")
    if not isinstance(pose, Sequence) or len(pose) < 2:
        return None
    x, y = _finite(pose[0]), _finite(pose[1])
    return None if x is None or y is None else (x, y)


def _velocity(record: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    value = record.get("velocity")
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    x, y = _finite(value[0]), _finite(value[1])
    return None if x is None or y is None else (x, y)


def _half_diagonal(record: Mapping[str, Any]) -> float:
    dimensions = record.get("dimensions") or {}
    length = _finite(dimensions.get("length_m")) or 0.0
    width = _finite(dimensions.get("width_m")) or 0.0
    return 0.5 * math.hypot(length, width)


def constant_velocity_cpa(
    subject: Mapping[str, Any],
    obj: Mapping[str, Any],
    *,
    horizon_s: float,
) -> CpaResult:
    """Compute closest approach under a short constant-velocity model."""
    sxy, oxy = _xy(subject), _xy(obj)
    sv, ov = _velocity(subject), _velocity(obj)
    if sxy is None or oxy is None or sv is None or ov is None:
        return CpaResult(None, None, None, False)
    rx, ry = oxy[0] - sxy[0], oxy[1] - sxy[1]
    vx, vy = ov[0] - sv[0], ov[1] - sv[1]
    vv = vx * vx + vy * vy
    closing = (rx * vx + ry * vy) < 0.0
    if vv <= 1e-9:
        t = 0.0
    else:
        t = max(0.0, min(float(horizon_s), -(rx * vx + ry * vy) / vv))
    dx, dy = rx + vx * t, ry + vy * t
    center = math.hypot(dx, dy)
    clearance = center - _half_diagonal(subject) - _half_diagonal(obj)
    return CpaResult(float(t), float(center), float(clearance), bool(closing))


def normalized_agent_type(record: Mapping[str, Any]) -> str:
    return str(record.get("agent_type") or record.get("tracked_object_type") or "UNKNOWN").upper()


def is_vulnerable_road_user(record: Mapping[str, Any]) -> bool:
    name = normalized_agent_type(record)
    return any(token in name for token in ("PEDESTRIAN", "BICYCLE", "CYCLIST"))


def is_static_or_barrier(record: Mapping[str, Any]) -> bool:
    name = normalized_agent_type(record)
    return any(token in name for token in ("STATIC", "BARRIER", "CZONE_SIGN", "GENERIC_OBJECT"))


def route_relation(subject: Mapping[str, Any], obj: Mapping[str, Any]) -> str:
    srank = subject.get("route_roadblock_rank")
    orank = obj.get("route_roadblock_rank")
    if srank is None or orank is None:
        if obj.get("on_expert_route"):
            return "object_on_route"
        return "off_or_unknown_route"
    delta = int(orank) - int(srank)
    if delta == 0:
        return "same_route_roadblock"
    if delta > 0:
        return f"route_ahead_{delta}"
    return f"route_behind_{abs(delta)}"


def evaluate_pair_relevance(
    *,
    subject: Mapping[str, Any],
    obj: Mapping[str, Any],
    map_relation: str,
    center_distance_m: Optional[float],
    free_space_distance_m: Optional[float],
    signed_path_distance_m: Optional[float],
    same_flow: bool,
    shared_intersection: bool,
    relevant_traffic_control: bool,
    subject_is_ego: bool,
    thresholds: RelevanceThresholds = DEFAULT_RELEVANCE_THRESHOLDS,
    additional_reasons: Sequence[str] = (),
    mandatory_selection: bool = False,
) -> PairRelevanceDecision:
    """Return a conservative relevance decision and directional gate.

    Relevance is disjunctive: proximity, path occupancy, route topology,
    intersection conflict, and CPA can independently make an object relevant.
    Directional predicates are stricter and require a coherent lane/path frame.
    """
    center = _finite(center_distance_m)
    free = _finite(free_space_distance_m)
    path = _finite(signed_path_distance_m)
    relation = str(map_relation or "unrelated")
    rrel = route_relation(subject, obj)

    if center is None or center > thresholds.hard_max_center_distance_m:
        return PairRelevanceDecision(
            False, False, 0.0, tuple(), "outside_hard_range", relation, rrel,
            center, free, path, None, None, None, False, "outside_hard_range",
            False,
        )

    reasons: list[str] = []
    score = 0.0
    critical = False

    if free is not None and free <= thresholds.always_relevant_free_space_m:
        reasons.append("immediate_proximity")
        score += 100.0 - min(50.0, free * 5.0)
        critical = free <= 1.0

    if relation == "same_map" and same_flow and path is not None:
        if 0.0 < path <= thresholds.same_lane_ahead_m:
            reasons.append("same_lane_ahead")
            score += 70.0 + max(0.0, 30.0 * (1.0 - path / thresholds.same_lane_ahead_m))
        elif -thresholds.same_lane_behind_m <= path < 0.0:
            reasons.append("same_lane_behind")
            score += 45.0 + max(0.0, 20.0 * (1.0 - abs(path) / thresholds.same_lane_behind_m))

    if relation in {"adjacent_left", "adjacent_right"} and same_flow and center <= thresholds.adjacent_lane_m:
        reasons.append(relation)
        score += 45.0 + max(0.0, 20.0 * (1.0 - center / thresholds.adjacent_lane_m))

    if relation in {"successor", "predecessor"} and same_flow and center <= thresholds.connected_lane_m:
        reasons.append(f"connected_{relation}")
        score += 55.0 + max(0.0, 15.0 * (1.0 - center / thresholds.connected_lane_m))

    srank = subject.get("route_roadblock_rank")
    orank = obj.get("route_roadblock_rank")
    if subject_is_ego and srank is not None and orank is not None:
        delta = int(orank) - int(srank)
        route_range_ok = (
            0 <= delta <= thresholds.route_roadblock_lookahead
            and center <= thresholds.same_lane_ahead_m
        ) or (
            -thresholds.route_roadblock_lookbehind <= delta < 0
            and center <= thresholds.same_lane_behind_m
        )
        if route_range_ok:
            reasons.append("ego_route_corridor")
            score += 35.0 - 5.0 * abs(delta)

    if shared_intersection and center <= thresholds.shared_intersection_distance_m:
        reasons.append("shared_intersection_conflict")
        score += 65.0 + max(0.0, 20.0 * (1.0 - center / thresholds.shared_intersection_distance_m))

    if relevant_traffic_control and center <= thresholds.traffic_control_distance_m:
        reasons.append("shared_traffic_control")
        score += 55.0

    if is_vulnerable_road_user(obj) and center <= thresholds.vulnerable_road_user_distance_m:
        route_or_conflict = bool(
            obj.get("on_ego_route_corridor")
            or obj.get("in_crosswalk")
            or shared_intersection
            or relevant_traffic_control
            or (free is not None and free <= thresholds.always_relevant_free_space_m)
        )
        if route_or_conflict:
            reasons.append("vulnerable_road_user_near_ego_path")
            score += 90.0
            critical = critical or center <= 12.0

    if is_static_or_barrier(obj) and center <= thresholds.static_obstacle_distance_m:
        if obj.get("on_ego_route_corridor") or (free is not None and free <= thresholds.always_relevant_free_space_m):
            reasons.append("static_obstacle_on_ego_path")
            score += 80.0

    cpa = constant_velocity_cpa(subject, obj, horizon_s=thresholds.cpa_horizon_s)
    if (
        cpa.time_s is not None
        and cpa.clearance_m is not None
        and cpa.closing
        and 0.0 <= cpa.time_s <= thresholds.cpa_horizon_s
        and cpa.clearance_m <= thresholds.cpa_clearance_margin_m
    ):
        reasons.append("predicted_close_approach")
        score += 95.0 - min(40.0, cpa.time_s * 5.0)
        if cpa.time_s <= thresholds.critical_cpa_time_s and cpa.clearance_m <= thresholds.critical_cpa_clearance_m:
            critical = True

    for reason in additional_reasons:
        reason = str(reason).strip()
        if reason and reason not in reasons:
            reasons.append(reason)

    relevant = bool(reasons)
    primary = max(reasons, key=lambda reason: {
        "predicted_close_approach": 100,
        "immediate_proximity": 95,
        "vulnerable_road_user_near_ego_path": 90,
        "static_obstacle_on_ego_path": 85,
        "shared_intersection_conflict": 80,
        "same_lane_ahead": 75,
        "connected_successor": 70,
        "connected_predecessor": 65,
        "shared_traffic_control": 60,
        "same_lane_behind": 55,
        "adjacent_left": 50,
        "adjacent_right": 50,
        "ego_route_corridor": 40,
        "generator_within_distance": 110,
        "generator_forward_corridor": 105,
        "generator_same_lane": 100,
        "generator_predicted_path_intersection": 115,
        "reverse_of_selected_pair": 10,
    }.get(reason, 0)) if reasons else "not_relevant"

    directional_allowed = False
    omission = None
    if not relevant:
        omission = "pair_not_relevant"
    elif not same_flow:
        omission = "cross_or_opposite_flow"
    elif relation == "same_map" and path is not None:
        directional_allowed = (
            -thresholds.directional_same_lane_behind_m
            <= path
            <= thresholds.directional_same_lane_ahead_m
            and abs(path) > 1e-6
        )
        if not directional_allowed:
            omission = "same_lane_outside_directional_range"
    elif relation in {"adjacent_left", "adjacent_right"}:
        directional_allowed = center <= thresholds.directional_adjacent_m
        if not directional_allowed:
            omission = "adjacent_lane_outside_directional_range"
    elif relation in {"successor", "predecessor"}:
        directional_allowed = center <= thresholds.directional_connected_m
        if not directional_allowed:
            omission = "connected_lane_outside_directional_range"
    else:
        omission = "no_common_lane_path_frame"

    return PairRelevanceDecision(
        relevant=relevant,
        critical=critical,
        score=float(score),
        reasons=tuple(sorted(set(reasons))),
        primary_reason=primary,
        map_relation=relation,
        route_relation=rrel,
        center_distance_m=center,
        free_space_distance_m=free,
        signed_path_distance_m=path,
        cpa_time_s=cpa.time_s,
        cpa_center_distance_m=cpa.center_distance_m,
        cpa_clearance_m=cpa.clearance_m,
        directional_semantics_allowed=directional_allowed,
        directional_omission_reason=omission,
        mandatory_selection=bool(mandatory_selection and relevant),
    )


def select_top_relevant(
    decisions: Sequence[tuple[Mapping[str, Any], PairRelevanceDecision]],
    *,
    thresholds: RelevanceThresholds = DEFAULT_RELEVANCE_THRESHOLDS,
) -> list[tuple[Mapping[str, Any], PairRelevanceDecision]]:
    """Keep all critical objects and the highest scoring non-critical objects."""
    relevant = [(record, decision) for record, decision in decisions if decision.relevant]
    mandatory = [(r, d) for r, d in relevant if d.mandatory_selection]
    mandatory_keys = {str(r.get("track_token")) for r, _ in mandatory}
    critical = [
        (r, d) for r, d in relevant
        if d.critical and str(r.get("track_token")) not in mandatory_keys
    ]
    protected_keys = mandatory_keys | {str(r.get("track_token")) for r, _ in critical}
    ordinary = sorted(
        [(r, d) for r, d in relevant if str(r.get("track_token")) not in protected_keys],
        key=lambda item: (-item[1].score, item[1].center_distance_m or math.inf, str(item[0].get("track_token"))),
    )
    # Mandatory agents are never removed by the historical top-k cap.
    remaining = max(0, thresholds.max_relevant_ego_agents - len(mandatory) - len(critical))
    ordinary_limit = min(thresholds.max_noncritical_relevant_agents, remaining)
    selected = mandatory + critical + ordinary[:ordinary_limit]
    return sorted(
        selected,
        key=lambda item: (
            -int(item[1].mandatory_selection),
            -int(item[1].critical),
            -item[1].score,
            str(item[0].get("track_token")),
        ),
    )
