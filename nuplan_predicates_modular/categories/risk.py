"""Current-state conflict-risk predicates used by the paper.

No observed future trajectory, collision label, or retrospective ground truth is
used here.  The final safety gate follows the UN R157 thresholds reported in the
paper appendix.

First predicate
---------------
``np:hasConflictRiskWith(A, B)`` is a symmetric relation between two dynamic
road users.  It is asserted when a short constant-velocity continuation of the
*current* states predicts that their oriented footprints will approach within
an explicit safety clearance during the prediction horizon, while the pair also
has interaction-relevance evidence and a meaningful radial closing motion.

This is a risk predicate, not a collision predicate: the recorded nuPlan future
may subsequently brake, yield, turn, or otherwise avoid the predicted conflict.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import *
from .geometry import footprint_metrics, oriented_box_polygon


CONFLICT_RISK_RULE_ID = "R-CONFLICT-RISK-CV-002"
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0
_NUMERICAL_EPS = 1e-6


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_velocity(record: dict[str, Any]) -> Optional[tuple[float, float]]:
    velocity = record.get("velocity")
    if not isinstance(velocity, (tuple, list)) or len(velocity) < 2:
        return None
    vx, vy = _finite(velocity[0]), _finite(velocity[1])
    if vx is None or vy is None:
        return None
    return float(vx), float(vy)


def _half_diagonal(record: dict[str, Any]) -> Optional[float]:
    dimensions = record.get("dimensions") or {}
    length = _finite(dimensions.get("length_m"))
    width = _finite(dimensions.get("width_m"))
    if length is None or width is None or length <= 0.0 or width <= 0.0:
        return None
    return 0.5 * math.hypot(length, width)


def _relative_motion(subject: dict[str, Any], obj: dict[str, Any]):
    sv = _valid_velocity(subject)
    ov = _valid_velocity(obj)
    if sv is None or ov is None:
        return None

    sx, sy, _ = subject["xyh"]
    ox, oy, _ = obj["xyh"]
    rx, ry = float(ox) - float(sx), float(oy) - float(sy)
    dvx, dvy = float(ov[0]) - float(sv[0]), float(ov[1]) - float(sv[1])
    relative_speed = math.hypot(dvx, dvy)
    return rx, ry, dvx, dvy, relative_speed, sv, ov




def _speed(velocity: tuple[float, float]) -> float:
    return float(math.hypot(float(velocity[0]), float(velocity[1])))


def _radial_closing_speed(
    rx: float,
    ry: float,
    dvx: float,
    dvy: float,
) -> float:
    """Positive when the current center-to-center separation is decreasing."""
    distance = math.hypot(rx, ry)
    if distance <= _NUMERICAL_EPS:
        return 0.0
    return float(-(rx * dvx + ry * dvy) / distance)


def _angle_difference_rad(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _motion_direction_difference(
    subject: dict[str, Any],
    obj: dict[str, Any],
    subject_velocity: tuple[float, float],
    object_velocity: tuple[float, float],
) -> float:
    """Current direction difference, preferring velocity and falling back to pose heading."""
    ss = _speed(subject_velocity)
    os = _speed(object_velocity)
    if ss > 0.2:
        sh = math.atan2(subject_velocity[1], subject_velocity[0])
    else:
        sh = float(subject["xyh"][2])
    if os > 0.2:
        oh = math.atan2(object_velocity[1], object_velocity[0])
    else:
        oh = float(obj["xyh"][2])
    return float(_angle_difference_rad(sh, oh))


def _is_vulnerable_road_user(record: dict[str, Any]) -> bool:
    name = str(record.get("agent_type") or record.get("tracked_object_type") or "").upper()
    return any(token in name for token in ("PEDESTRIAN", "BICYCLE", "CYCLIST"))


def _interaction_relevant_candidate(
    context: dict[str, Any],
    subject: dict[str, Any],
    obj: dict[str, Any],
    *,
    current_clearance_m: float,
    center_distance_m: float,
    subject_velocity: tuple[float, float],
    object_velocity: tuple[float, float],
) -> tuple[bool, str, float]:
    """Conservative internal candidate gate for conflict-risk reasoning.

    This gate is evidence used by ``hasConflictRiskWith``; it is deliberately
    not materialized as another KG predicate.  It prevents the constant-velocity
    model from turning unrelated nearby traffic into semantic risk assertions.
    """
    relation = str(context.get("map_relation") or "unrelated")
    direction_diff = _motion_direction_difference(
        subject, obj, subject_velocity, object_velocity
    )

    # Strongest evidence: both road users occupy the same lane/path object or
    # immediately connected lane/lane-connector objects.
    if relation in {"same_map", "successor", "predecessor"}:
        return True, "lane_path_connected", direction_diff

    # Vulnerable road users often have no usable lane topology.  Keep only a
    # local neighborhood; the later OBB, closing-speed and clearance tests are
    # still required before the predicate can be emitted.
    if (_is_vulnerable_road_user(subject) or _is_vulnerable_road_user(obj)) and (
        center_distance_m <= float(ARGS.conflict_risk_vru_max_center_distance_m)
    ):
        return True, "vulnerable_road_user_local", direction_diff

    # Map matching may be unavailable for a valid rear-end interaction.  Allow
    # a tight same-flow geometric corridor as a fallback, but not a broad
    # adjacent-road corridor.
    geometric = context.get("geometric_spatial")
    lateral = None if geometric is None else _finite(
        getattr(geometric, "travel_lateral_m", None)
    )
    same_flow_limit = math.radians(15.0)
    if (
        lateral is not None
        and abs(lateral) <= float(ARGS.conflict_risk_same_flow_corridor_lateral_m)
        and center_distance_m <= float(ARGS.conflict_risk_same_flow_corridor_center_m)
        and direction_diff <= same_flow_limit
    ):
        return True, "tight_same_flow_corridor", direction_diff

    # For otherwise unconnected agents, retain only genuinely crossing or
    # opposite-flow motion in a local interaction region.  This rejects close
    # parallel traffic on different roads/lane objects.
    crossing_limit = math.radians(float(ARGS.conflict_risk_crossing_angle_deg))
    if (
        direction_diff >= crossing_limit
        and center_distance_m <= float(ARGS.conflict_risk_cross_traffic_max_center_distance_m)
    ):
        return True, "cross_or_opposite_motion", direction_diff

    return False, "no_interaction_relevance", direction_diff


def _center_cpa_time_and_distance(
    rx: float,
    ry: float,
    dvx: float,
    dvy: float,
    horizon_s: float,
) -> tuple[float, float]:
    vv = dvx * dvx + dvy * dvy
    if vv <= _NUMERICAL_EPS:
        return 0.0, math.hypot(rx, ry)
    time_s = -(rx * dvx + ry * dvy) / vv
    time_s = min(max(0.0, float(time_s)), float(horizon_s))
    dx = rx + dvx * time_s
    dy = ry + dvy * time_s
    return float(time_s), float(math.hypot(dx, dy))


def _predicted_clearance_function(
    subject: dict[str, Any],
    obj: dict[str, Any],
    dvx: float,
    dvy: float,
):
    """Return footprint clearance under fixed-heading relative translation.

    Common translation by the subject velocity does not change pairwise
    clearance.  Therefore the subject footprint can remain fixed while the
    object footprint is translated by the relative velocity ``v_obj-v_subj``.
    """
    subject_polygon = oriented_box_polygon(subject)
    object_polygon = oriented_box_polygon(obj)
    if subject_polygon is None or object_polygon is None:
        return None

    def clearance(time_s: float) -> float:
        moved_object = translate(
            object_polygon,
            xoff=float(dvx) * float(time_s),
            yoff=float(dvy) * float(time_s),
        )
        return float(subject_polygon.distance(moved_object))

    return clearance


def _minimum_future_clearance(
    clearance_fn,
    *,
    horizon_s: float,
    center_cpa_time_s: float,
    iterations: int,
) -> tuple[float, float]:
    """Numerically minimize convex footprint clearance over the short horizon.

    With fixed headings and constant velocities, the relative footprint motion
    is a straight translation.  Oriented boxes are convex, so a bounded
    golden-section search is a deterministic and inexpensive way to locate the
    minimum clearance.  End points and center-CPA time are explicitly checked
    to make the boundary behaviour deterministic.
    """
    left, right = 0.0, float(horizon_s)
    c = right - _GOLDEN_RATIO * (right - left)
    d = left + _GOLDEN_RATIO * (right - left)
    fc, fd = clearance_fn(c), clearance_fn(d)

    for _ in range(max(8, int(iterations))):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - _GOLDEN_RATIO * (right - left)
            fc = clearance_fn(c)
        else:
            left, c, fc = c, d, fd
            d = left + _GOLDEN_RATIO * (right - left)
            fd = clearance_fn(d)

    candidates = [
        (0.0, clearance_fn(0.0)),
        (float(horizon_s), clearance_fn(float(horizon_s))),
        (float(center_cpa_time_s), clearance_fn(float(center_cpa_time_s))),
        (float(c), float(fc)),
        (float(d), float(fd)),
        (0.5 * (left + right), clearance_fn(0.5 * (left + right))),
    ]
    time_s, clearance_m = min(candidates, key=lambda item: (item[1], item[0]))
    return float(time_s), float(clearance_m)


def _collision_avoidance_braking_demand_mps2(
    closing_speed_mps: float,
    current_clearance_m: float,
    protected_clearance_m: float,
) -> float:
    """Return constant-deceleration demand needed to stop closing before the envelope.

    The available relative stopping distance is the current free-space clearance
    minus the protected clearance envelope.  ``v^2/(2d)`` is used for positive
    radial closing speed; if the pair is already inside the protected envelope,
    any positive closing motion is treated as unbounded braking demand.
    """
    speed = max(0.0, float(closing_speed_mps))
    if speed <= _NUMERICAL_EPS:
        return 0.0
    distance = float(current_clearance_m) - float(protected_clearance_m)
    if distance <= _NUMERICAL_EPS:
        return float("inf")
    return float((speed * speed) / (2.0 * distance))


def _is_ego_record(record: dict[str, Any]) -> bool:
    token = str(record.get("track_token") or "").lower()
    agent_type = str(record.get("agent_type") or "").upper()
    return token == "ego" or agent_type == "EGO"


def _lane_change_safety_assessment(
    context: dict[str, Any],
    subject: dict[str, Any],
    obj: dict[str, Any],
    subject_velocity: tuple[float, float],
    object_velocity: tuple[float, float],
    *,
    current_clearance_m: float,
    predicted_min_clearance_m: float,
) -> dict[str, Any]:
    """Evaluate the UN R157 lane-change gap criteria when the ego is involved."""
    relation = str(context.get("map_relation") or "unrelated")
    lane_change_related = relation in {"adjacent_left", "adjacent_right"}
    if not lane_change_related or not (_is_ego_record(subject) or _is_ego_record(obj)):
        return {
            "applicable": False,
            "unsafe": False,
            "target_required_deceleration_mps2": 0.0,
            "minimum_gap_threshold_m": 0.0,
            "gap_below_one_second_travel": False,
        }

    if _is_ego_record(subject):
        ego_velocity, target_velocity = subject_velocity, object_velocity
    else:
        ego_velocity, target_velocity = object_velocity, subject_velocity

    ego_speed = _speed(ego_velocity)
    target_speed = _speed(target_velocity)
    approach_speed = max(0.0, ego_speed - target_speed)
    if current_clearance_m <= _NUMERICAL_EPS and approach_speed > 0.0:
        target_decel = float("inf")
    elif approach_speed <= _NUMERICAL_EPS:
        target_decel = 0.0
    else:
        target_decel = float(
            (approach_speed * approach_speed)
            / (2.0 * max(float(current_clearance_m), _NUMERICAL_EPS))
        )

    gap_time_s = float(ARGS.conflict_risk_lane_change_min_gap_time_s)
    minimum_gap_m = ego_speed * gap_time_s
    gap_below = float(predicted_min_clearance_m) < minimum_gap_m
    unsafe = bool(
        target_decel > float(ARGS.conflict_risk_lane_change_max_target_decel_mps2)
        or gap_below
    )
    return {
        "applicable": True,
        "unsafe": unsafe,
        "target_required_deceleration_mps2": float(target_decel),
        "minimum_gap_threshold_m": float(minimum_gap_m),
        "gap_below_one_second_travel": bool(gap_below),
        "ego_speed_mps": float(ego_speed),
        "target_speed_mps": float(target_speed),
    }


def _safety_gate(
    context: dict[str, Any],
    subject: dict[str, Any],
    obj: dict[str, Any],
    subject_velocity: tuple[float, float],
    object_velocity: tuple[float, float],
    *,
    closing_speed_mps: float,
    current_clearance_m: float,
    predicted_min_clearance_m: float,
    protected_clearance_m: float,
) -> tuple[bool, dict[str, Any]]:
    braking_demand = _collision_avoidance_braking_demand_mps2(
        closing_speed_mps, current_clearance_m, protected_clearance_m
    )
    imminent = braking_demand >= float(ARGS.conflict_risk_imminent_braking_demand_mps2)
    lane_change = _lane_change_safety_assessment(
        context, subject, obj, subject_velocity, object_velocity,
        current_clearance_m=current_clearance_m,
        predicted_min_clearance_m=predicted_min_clearance_m,
    )
    established = bool(imminent or (lane_change["applicable"] and lane_change["unsafe"]))
    evidence = {
        "safety_gate_established": established,
        "collision_avoidance_braking_demand_mps2": float(braking_demand),
        "imminent_collision_threshold_mps2": float(ARGS.conflict_risk_imminent_braking_demand_mps2),
        "imminent_collision_condition": bool(imminent),
        "lane_change_safety": lane_change,
        "lane_change_max_target_deceleration_mps2": float(ARGS.conflict_risk_lane_change_max_target_decel_mps2),
        "lane_change_min_gap_time_s": float(ARGS.conflict_risk_lane_change_min_gap_time_s),
    }
    return established, evidence


def _claim_unordered_pair_once(
    state: Optional[dict[str, Any]],
    frame,
    subject: dict[str, Any],
    obj: dict[str, Any],
) -> tuple[bool, tuple[str, str]]:
    """Suppress duplicate A->B/B->A emission while retaining directed pair processing."""
    canonical = canonical_symmetric_pair(subject["track_token"], obj["track_token"])
    if state is None:
        return True, canonical

    timestamp_us = int(frame.timestamp_us)
    cache = state.setdefault("risk_unordered_pair_cache", {})
    if cache.get("timestamp_us") != timestamp_us:
        cache.clear()
        cache["timestamp_us"] = timestamp_us
        cache["seen"] = set()
    seen = cache.setdefault("seen", set())
    if canonical in seen:
        return False, canonical
    seen.add(canonical)
    return True, canonical


def append_pair_risk_predicates(
    assertions,
    frame,
    pair_id,
    context,
    *,
    state: Optional[dict[str, Any]] = None,
):
    """Append ``np:hasConflictRiskWith`` when current-state evidence establishes it."""
    subject = context["subject"]
    obj = context["object"]
    evidence = context["evidence"]

    emit_this_direction, canonical = _claim_unordered_pair_once(
        state, frame, subject, obj
    )
    if not emit_this_direction:
        return

    pid = "np:hasConflictRiskWith"

    if not is_dynamic_road_user(subject) or not is_dynamic_road_user(obj):
        record_semantic_omission(state, pid, "non_dynamic_road_user")
        return

    geometry = context.get("geometry") or {}
    current_clearance = _finite(geometry.get("free_space_distance_m"))
    if current_clearance is None:
        record_semantic_omission(state, pid, "missing_oriented_footprint")
        return
    if geometry.get("overlapping") is True:
        # The predicate represents a future conflict risk, not an already
        # overlapping/colliding state.
        record_semantic_omission(state, pid, "already_overlapping")
        return

    motion = _relative_motion(subject, obj)
    if motion is None:
        record_semantic_omission(state, pid, "missing_velocity")
        return
    rx, ry, dvx, dvy, relative_speed, subject_velocity, object_velocity = motion

    center_distance_m = float(math.hypot(rx, ry))
    closing_speed_mps = _radial_closing_speed(rx, ry, dvx, dvy)
    if closing_speed_mps < float(ARGS.conflict_risk_min_closing_speed_mps):
        record_semantic_omission(state, pid, "insufficient_closing_speed")
        return

    eligible, eligibility_reason, direction_difference_rad = _interaction_relevant_candidate(
        context,
        subject,
        obj,
        current_clearance_m=float(current_clearance),
        center_distance_m=center_distance_m,
        subject_velocity=subject_velocity,
        object_velocity=object_velocity,
    )
    if not eligible:
        record_semantic_omission(state, pid, "pair_not_interaction_relevant")
        return

    horizon_s = float(ARGS.conflict_risk_horizon_s)
    clearance_threshold_m = float(ARGS.conflict_risk_clearance_m)
    center_cpa_time_s, center_cpa_distance_m = _center_cpa_time_and_distance(
        rx, ry, dvx, dvy, horizon_s
    )

    # Exact necessary broad-phase test: if the *minimum center distance* is
    # larger than both footprint half-diagonals plus the safety clearance, the
    # oriented footprints cannot possibly come within the safety envelope.
    subject_radius = _half_diagonal(subject)
    object_radius = _half_diagonal(obj)
    if subject_radius is None or object_radius is None:
        record_semantic_omission(state, pid, "missing_dimensions")
        return
    broad_phase_limit = subject_radius + object_radius + clearance_threshold_m
    if center_cpa_distance_m > broad_phase_limit:
        record_semantic_omission(state, pid, "cannot_reach_conflict_envelope")
        return

    clearance_fn = _predicted_clearance_function(subject, obj, dvx, dvy)
    if clearance_fn is None:
        record_semantic_omission(state, pid, "missing_oriented_footprint")
        return

    min_time_s, min_clearance_m = _minimum_future_clearance(
        clearance_fn,
        horizon_s=horizon_s,
        center_cpa_time_s=center_cpa_time_s,
        iterations=int(ARGS.conflict_risk_optimization_iterations),
    )

    # A future risk must be strictly future-facing and the predicted clearance
    # must become smaller than it is now.  This rejects parallel close traffic
    # and pairs that are already receding from one another.
    clearance_reduction_m = current_clearance - min_clearance_m
    minimum_reduction_m = float(ARGS.conflict_risk_min_clearance_reduction_m)
    establishes_risk = bool(
        min_time_s > _NUMERICAL_EPS
        and min_time_s <= horizon_s + _NUMERICAL_EPS
        and min_clearance_m <= clearance_threshold_m + _NUMERICAL_EPS
        and clearance_reduction_m + _NUMERICAL_EPS >= minimum_reduction_m
    )
    if not establishes_risk:
        record_semantic_omission(state, pid, "future_conflict_not_established")
        return

    safety_established, safety_evidence = _safety_gate(
        context,
        subject,
        obj,
        subject_velocity,
        object_velocity,
        closing_speed_mps=closing_speed_mps,
        current_clearance_m=float(current_clearance),
        predicted_min_clearance_m=float(min_clearance_m),
        protected_clearance_m=clearance_threshold_m,
    )
    if not safety_established:
        record_semantic_omission(state, pid, "un_r157_safety_gate_not_established")
        return

    risk_evidence = {
        **evidence,
        "canonical_symmetric_pair": list(canonical),
        "risk_model": "constant_velocity_fixed_heading_oriented_footprints",
        "uses_observed_future": False,
        "current_free_space_clearance_m": float(current_clearance),
        "predicted_minimum_clearance_m": float(min_clearance_m),
        "predicted_time_to_minimum_clearance_s": float(min_time_s),
        "predicted_clearance_reduction_m": float(clearance_reduction_m),
        "center_cpa_time_s": float(center_cpa_time_s),
        "center_cpa_distance_m": float(center_cpa_distance_m),
        "relative_speed_mps": float(relative_speed),
        "closing_speed_mps": float(closing_speed_mps),
        "interaction_eligibility_reason": eligibility_reason,
        "motion_direction_difference_deg": float(math.degrees(direction_difference_rad)),
        "subject_velocity_mps": [float(subject_velocity[0]), float(subject_velocity[1])],
        "object_velocity_mps": [float(object_velocity[0]), float(object_velocity[1])],
        "prediction_horizon_s": horizon_s,
        "conflict_clearance_threshold_m": clearance_threshold_m,
        "minimum_closing_speed_mps": float(ARGS.conflict_risk_min_closing_speed_mps),
        "minimum_clearance_reduction_m": float(ARGS.conflict_risk_min_clearance_reduction_m),
        **safety_evidence,
    }

    assertions.append(
        derived_relation(
            pid,
            subject["entity_id"],
            obj["entity_id"],
            frame,
            CONFLICT_RISK_RULE_ID,
            risk_evidence,
        )
    )
