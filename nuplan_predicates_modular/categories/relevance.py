"""Ego-agent and agent-agent relevance selection and audit evidence."""
from ..base import *
from .geometry import relative_kinematics, footprint_metrics
from .map import (
    _map_longitudinal_reference_difference,
    _pair_map_relation,
    _record_direction,
    _shared_intersection,
)

def _traffic_light_connector_ids(frame) -> set[str]:
    return {str(getattr(light, "lane_connector_id", "")) for light in frame.traffic_lights}


def _generator_ego_selection_features(ego_record, agent, context):
    """Return the four configurable ego-agent inclusion conditions.

    The conditions are combined with logical OR and are marked mandatory, so
    they are never removed by the historical top-k relevance cap.
    """
    kin, geometry, relation, path_distance, same_flow, shared_intersection, relevant_light = context
    center_distance = kin.get("center_distance_m")

    ego_direction = _record_direction(ego_record)
    longitudinal = lateral = None
    if ego_direction.reliable and ego_direction.heading_rad is not None:
        longitudinal, lateral = project_relative_position(
            ego_record["xyh"][:2], agent["xyh"][:2], ego_direction.heading_rad
        )

    within_distance = bool(
        ARGS.relevance_include_within_distance
        and center_distance is not None
        and math.isfinite(float(center_distance))
        and float(center_distance) <= ARGS.relevance_distance_threshold_m
    )
    inside_forward_corridor = bool(
        ARGS.relevance_include_forward_corridor
        and longitudinal is not None
        and lateral is not None
        and 0.0 <= float(longitudinal) <= ARGS.relevance_forward_corridor_length_m
        and abs(float(lateral)) <= ARGS.relevance_forward_corridor_half_width_m
    )
    same_lane = bool(
        ARGS.relevance_include_same_lane
        and relation == "same_map"
        and same_flow
    )

    cpa = constant_velocity_cpa(
        ego_record, agent, horizon_s=ARGS.relevance_prediction_horizon_s
    )
    predicted_path_intersection = bool(
        ARGS.relevance_include_predicted_path_intersection
        and cpa.time_s is not None
        and cpa.clearance_m is not None
        and 0.0 <= float(cpa.time_s) <= ARGS.relevance_prediction_horizon_s
        and float(cpa.clearance_m) <= ARGS.relevance_path_intersection_clearance_m
        and (cpa.closing or float(cpa.time_s) <= 1e-9)
    )

    reasons = []
    if within_distance:
        reasons.append("generator_within_distance")
    if inside_forward_corridor:
        reasons.append("generator_forward_corridor")
    if same_lane:
        reasons.append("generator_same_lane")
    if predicted_path_intersection:
        reasons.append("generator_predicted_path_intersection")

    return {
        "within_distance": within_distance,
        "inside_forward_corridor": inside_forward_corridor,
        "same_lane": same_lane,
        "predicted_path_intersection": predicted_path_intersection,
        "generator_selection_reasons": tuple(reasons),
        "ego_longitudinal_m": longitudinal,
        "ego_lateral_m": lateral,
        "prediction_cpa_time_s": cpa.time_s,
        "prediction_cpa_center_distance_m": cpa.center_distance_m,
        "prediction_cpa_clearance_m": cpa.clearance_m,
    }


def _evaluate_pair_context(subject, obj, frame, *, subject_is_ego, additional_reasons=(), mandatory_selection=False):
    context = _pair_context_for_relevance(subject, obj, frame)
    kin, geometry, relation, path_distance, same_flow, shared_intersection, relevant_light = context
    decision = evaluate_pair_relevance(
        subject=subject, obj=obj, map_relation=relation,
        center_distance_m=kin.get("center_distance_m"),
        free_space_distance_m=geometry.get("free_space_distance_m"),
        signed_path_distance_m=path_distance, same_flow=same_flow,
        shared_intersection=shared_intersection,
        relevant_traffic_control=relevant_light,
        subject_is_ego=subject_is_ego,
        thresholds=DEFAULT_RELEVANCE_THRESHOLDS,
        additional_reasons=additional_reasons,
        mandatory_selection=mandatory_selection,
    )
    return context, decision


def reverse_selected_pair_decision(subject, obj, frame, selected_decision):
    """Re-evaluate the reverse direction instead of reusing forward evidence."""
    context, decision = _evaluate_pair_context(
        subject, obj, frame, subject_is_ego=False,
        additional_reasons=("reverse_of_selected_pair",),
        mandatory_selection=selected_decision.mandatory_selection,
    )
    return decision


def _pair_context_for_relevance(subject, obj, frame):
    kin = relative_kinematics(subject, obj)
    geometry = footprint_metrics(subject, obj)
    relation = _pair_map_relation(subject, obj)
    path_distance = _map_longitudinal_reference_difference(subject, obj, relation)
    sd, od = _record_direction(subject), _record_direction(obj)
    difference = angular_difference(sd.heading_rad, od.heading_rad) if sd.reliable and od.reliable else None
    same_flow = difference is not None and difference <= ARGS.spatial_same_flow_threshold_rad
    shared_intersection = _shared_intersection(subject, obj)
    light_connectors = _traffic_light_connector_ids(frame)
    subject_connectors = {
        str(subject.get("primary_map_object_id") or ""),
        *{str(x) for x in subject.get("outgoing_object_ids", set())},
    }
    object_connectors = {
        str(obj.get("primary_map_object_id") or ""),
        *{str(x) for x in obj.get("outgoing_object_ids", set())},
    }
    relevant_light = bool(light_connectors & subject_connectors & object_connectors)
    return kin, geometry, relation, path_distance, same_flow, shared_intersection, relevant_light


def _select_relevant_ego_agents(ego_record, agent_records, frame, state, direction_modes=None):
    candidates = []
    contexts = {}
    for agent in agent_records:
        context = _pair_context_for_relevance(ego_record, agent, frame)
        features = _generator_ego_selection_features(ego_record, agent, context)
        reasons = features["generator_selection_reasons"]
        _, decision = _evaluate_pair_context(
            ego_record, agent, frame, subject_is_ego=True,
            additional_reasons=reasons,
            mandatory_selection=bool(reasons),
        )
        contexts[str(agent["track_token"])] = (context, decision, features)
        candidates.append((agent, decision))

    direction_modes = direction_modes or {"ego-agent": "relevance", "agent-ego": "relevance"}
    relevant_selected = select_top_relevant(
        candidates, thresholds=DEFAULT_RELEVANCE_THRESHOLDS
    )
    relevant_tokens = {str(record["track_token"]) for record, _ in relevant_selected}
    selected_tokens_by_direction = {
        direction: (
            {str(record["track_token"]) for record, _ in candidates}
            if mode == "all"
            else set(relevant_tokens)
        )
        for direction, mode in direction_modes.items()
    }
    decisions = {str(record["track_token"]): decision for record, decision in candidates}

    for agent, decision in candidates:
        token = str(agent["track_token"])
        _, _, features = contexts[token]
        state.setdefault("relevance_audit", []).append({
            "scenario_token": str(frame.scenario_token),
            "timestamp_us": int(frame.timestamp_us),
            "track_token": token,
            "selected": any(token in values for values in selected_tokens_by_direction.values()),
            "selected_ego_agent": token in selected_tokens_by_direction.get("ego-agent", set()),
            "selected_agent_ego": token in selected_tokens_by_direction.get("agent-ego", set()),
            "mandatory_selection": decision.mandatory_selection,
            "relevant_before_cap": decision.relevant,
            "critical": decision.critical,
            "score": decision.score,
            "primary_reason": decision.primary_reason,
            "reasons": "|".join(decision.reasons),
            "generator_selection_reasons": "|".join(features["generator_selection_reasons"]),
            "generator_within_distance": features["within_distance"],
            "generator_inside_forward_corridor": features["inside_forward_corridor"],
            "generator_same_lane": features["same_lane"],
            "generator_predicted_path_intersection": features["predicted_path_intersection"],
            "ego_longitudinal_m": features["ego_longitudinal_m"],
            "ego_lateral_m": features["ego_lateral_m"],
            "prediction_cpa_time_s": features["prediction_cpa_time_s"],
            "prediction_cpa_center_distance_m": features["prediction_cpa_center_distance_m"],
            "prediction_cpa_clearance_m": features["prediction_cpa_clearance_m"],
            "center_distance_m": decision.center_distance_m,
            "free_space_distance_m": decision.free_space_distance_m,
            "signed_path_distance_m": decision.signed_path_distance_m,
            "cpa_time_s": decision.cpa_time_s,
            "cpa_clearance_m": decision.cpa_clearance_m,
            "directional_allowed": decision.directional_semantics_allowed,
            "directional_omission_reason": decision.directional_omission_reason,
        })
    return candidates, relevant_selected, decisions, contexts


def _agent_agent_candidate_pairs(agent_records, selection_mode=None):
    """Return cheap unordered candidate index pairs using a uniform spatial grid.

    The lossless ``pair-selection=all`` path intentionally bypasses this helper.
    Grid lookup avoids constructing the full O(N^2) pair list in dense frames.
    """
    count = len(agent_records)
    if count < 2:
        return []
    selection_mode = selection_mode or ARGS.pair_selection_by_direction.get("agent-agent", "all")
    if selection_mode == "all":
        return [(i, j) for i in range(count) for j in range(i + 1, count)]

    radius = max(float(ARGS.agent_agent_candidate_radius_m), 0.0)
    if radius <= 0.0:
        return [(i, j) for i in range(count) for j in range(i + 1, count)]

    cell_size = radius
    radius_sq = radius * radius
    grid = defaultdict(list)
    positions = []
    for index, record in enumerate(agent_records):
        x, y = map(float, record["xyh"][:2])
        positions.append((x, y))
        grid[(math.floor(x / cell_size), math.floor(y / cell_size))].append(index)

    pairs = []
    for i, (x, y) in enumerate(positions):
        cx, cy = math.floor(x / cell_size), math.floor(y / cell_size)
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                for j in grid.get((gx, gy), ()):
                    if j <= i:
                        continue
                    ox, oy = positions[j]
                    dx, dy = ox - x, oy - y
                    if dx * dx + dy * dy <= radius_sq:
                        pairs.append((i, j))
    return pairs


def _agent_agent_rank_key(item):
    """Sort critical and high-confidence pairs before ordinary nearby pairs."""
    key, decision = item
    distance = decision.free_space_distance_m
    if distance is None or not math.isfinite(float(distance)):
        distance = decision.center_distance_m
    if distance is None or not math.isfinite(float(distance)):
        distance = float("inf")
    return (
        0 if decision.critical else 1,
        0 if decision.mandatory_selection else 1,
        -float(decision.score),
        float(distance),
        key,
    )


def _select_relevant_agent_pairs(agent_records, frame, state=None, selection_mode=None):
    """Select agent-agent pairs with cheap candidate generation and safe caps.

    ``pair-selection=all`` retains the original lossless behavior. Relevance mode
    first applies a spatial grid prefilter, then executes the existing detailed
    relevance logic, and finally caps only non-critical pairs.
    """
    state = state if state is not None else {}
    total_possible = len(agent_records) * max(len(agent_records) - 1, 0) // 2
    selection_mode = selection_mode or ARGS.pair_selection_by_direction.get("agent-agent", "all")
    candidate_indices = _agent_agent_candidate_pairs(agent_records, selection_mode=selection_mode)
    evaluated = []

    for i, j in candidate_indices:
        a, b = agent_records[i], agent_records[j]
        context, decision = _evaluate_pair_context(
            a, b, frame, subject_is_ego=False
        )
        key = (str(a["track_token"]), str(b["track_token"]))
        if selection_mode == "all" or decision.relevant:
            evaluated.append((key, decision))

        if selection_mode == "relevance":
            state.setdefault("agent_agent_relevance_audit", []).append({
                "scenario_token": str(frame.scenario_token),
                "timestamp_us": int(frame.timestamp_us),
                "subject_token": key[0],
                "object_token": key[1],
                "selected_before_cap": bool(decision.relevant),
                "critical": bool(decision.critical),
                "score": float(decision.score),
                "primary_reason": decision.primary_reason,
                "reasons": "|".join(decision.reasons),
                "center_distance_m": decision.center_distance_m,
                "free_space_distance_m": decision.free_space_distance_m,
                "signed_path_distance_m": decision.signed_path_distance_m,
                "cpa_time_s": decision.cpa_time_s,
                "cpa_clearance_m": decision.cpa_clearance_m,
            })

    if selection_mode == "all":
        selected_items = evaluated
        removed_by_cap = 0
    else:
        ordered = sorted(evaluated, key=_agent_agent_rank_key)
        critical = [item for item in ordered if item[1].critical]
        ordinary = [item for item in ordered if not item[1].critical]

        neighbor_cap = max(int(ARGS.agent_agent_max_neighbors_per_agent), 0)
        frame_cap = max(int(ARGS.agent_agent_max_pairs_per_frame), 0)
        counts = Counter()
        kept_ordinary = []
        for item in ordinary:
            (first, second), _ = item
            if neighbor_cap and (counts[first] >= neighbor_cap or counts[second] >= neighbor_cap):
                continue
            if frame_cap and len(kept_ordinary) >= frame_cap:
                continue
            kept_ordinary.append(item)
            counts[first] += 1
            counts[second] += 1
        selected_items = critical + kept_ordinary
        removed_by_cap = len(ordinary) - len(kept_ordinary)

    selected_keys = {key for key, _ in selected_items}
    for row in state.get("agent_agent_relevance_audit", []):
        if (row.get("subject_token"), row.get("object_token")) in selected_keys:
            row["selected"] = True
            row["removed_by_cap"] = False
        elif "selected" not in row:
            row["selected"] = False
            row["removed_by_cap"] = bool(row.get("selected_before_cap"))

    stats = state.setdefault("agent_agent_relevance_stats", Counter())
    stats["possible_pairs"] += total_possible
    stats["candidates_after_prefilter"] += len(candidate_indices)
    stats["fully_evaluated"] += len(candidate_indices)
    stats["relevant_before_cap"] += len(evaluated)
    stats["selected"] += len(selected_items)
    stats["rejected"] += max(total_possible - len(selected_items), 0)
    stats["critical_selected"] += sum(1 for _, d in selected_items if d.critical)
    stats["removed_by_cap"] += removed_by_cap

    return [key for key, _ in selected_items], dict(selected_items)


def append_pair_relevance_assertions(assertions, frame, pair_id, pair_tag, context, state=None):
    decision = context.get("relevance_decision")
    if decision is None:
        return
    evidence = context["evidence"]
    relevance_pid = (
        "np:isRelevantToEgo"
        if pair_tag in {"ego_to_agent", "agent_to_ego"}
        else "np:isRelevantPair"
    )
    append_positive_boolean_semantic(
        assertions, relevance_pid, pair_id, bool(decision.relevant),
        frame, "R-EGO-RELEVANCE-001", evidence, state=state,
    )
    assertions.append(
        derived_value(
            "np:hasRelevanceScore", pair_id, float(decision.score), ValueType.FLOAT,
            frame, "R-EGO-RELEVANCE-001", evidence,
        )
    )
    assertions.append(
        derived_value(
            "np:hasPrimaryRelevanceReason", pair_id, decision.primary_reason,
            ValueType.STRING, frame, "R-EGO-RELEVANCE-001", evidence,
        )
    )
    assertions.append(
        derived_value(
            "np:hasRouteRelation", pair_id, decision.route_relation,
            ValueType.STRING, frame, "R-EGO-RELEVANCE-001", evidence,
        )
    )
    for reason in decision.reasons:
        assertions.append(
            derived_value(
                "np:hasRelevanceReason", pair_id, reason, ValueType.STRING,
                frame, "R-EGO-RELEVANCE-001", evidence,
            )
        )
