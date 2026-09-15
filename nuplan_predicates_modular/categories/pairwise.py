"""Thin directed-pair coordinator.

Category-specific emissions are delegated to geometry, motion, map, relevance,
spatial, heading, interaction, risk, and temporal modules.
"""
from ..base import *
from ..predicate_logic import classify_body_frame_spatial_relation
from .geometry import relative_kinematics, footprint_metrics, append_pair_geometry_measurements
from .motion import append_pair_motion_measurements
from .map import (
    _same_primary_map,
    _pair_map_relation,
    _record_direction,
    _same_baseline_progress_difference,
    _signed_path_distance,
    _map_longitudinal_reference_difference,
    append_pair_map_measurements,
    append_pair_map_semantics,
)
from .relevance import append_pair_relevance_assertions
from .spatial import append_pair_spatial_semantics
from .heading import append_pair_heading_semantics
from .interaction import append_pair_instant_interactions
from .risk import append_pair_risk_predicates
from .temporal import append_pair_temporal_measurements


from .snapshot_utils import _pair_id_for_snapshot, _snapshot_pairs, _records_by_token

def _build_pair_context(subject, obj, relevance_decision=None):
    sid, oid = subject["entity_id"], obj["entity_id"]
    kin = relative_kinematics(subject, obj)
    geometry = footprint_metrics(subject, obj)
    shared_primary = _same_primary_map(subject, obj)
    map_relation = _pair_map_relation(subject, obj)
    subject_direction = _record_direction(subject)
    object_direction = _record_direction(obj)
    same_baseline_progress_delta = _same_baseline_progress_difference(
        subject, obj, map_relation
    )
    signed_path_distance = _signed_path_distance(subject, obj, map_relation)
    map_progress_delta = _map_longitudinal_reference_difference(
        subject, obj, map_relation
    )
    spatial = classify_relevant_spatial_relation(
        subject_xy=subject["xyh"][:2],
        object_xy=obj["xyh"][:2],
        center_distance_m=kin.get("center_distance_m"),
        overlapping=bool(geometry.get("overlapping")),
        subject_direction=subject_direction,
        object_direction=object_direction,
        map_relation=map_relation,
        map_progress_delta_m=map_progress_delta,
        require_map_context=ARGS.require_map_context_for_spatial,
        relevance_max_distance_m=ARGS.spatial_relevance_max_distance_m,
        adjacent_relevance_max_distance_m=ARGS.spatial_adjacent_relevance_max_distance_m,
        diagonal_max_longitudinal_m=ARGS.spatial_diagonal_max_longitudinal_m,
        motion_only_fallback_max_distance_m=ARGS.spatial_motion_only_fallback_max_distance_m,
        motion_only_max_lateral_m=ARGS.spatial_motion_only_max_lateral_m,
        same_flow_threshold_rad=ARGS.spatial_same_flow_threshold_rad,
        longitudinal_deadband_m=ARGS.spatial_longitudinal_deadband_m,
        lateral_deadband_m=ARGS.spatial_lateral_deadband_m,
        side_longitudinal_band_m=ARGS.spatial_side_longitudinal_band_m,
        front_lateral_base_m=ARGS.spatial_front_lateral_base_m,
        front_lateral_ratio=ARGS.spatial_front_lateral_ratio,
        colocated_epsilon_m=ARGS.colocated_epsilon_m,
    )
    geometric_spatial = classify_body_frame_spatial_relation(
        subject_xyh=subject["xyh"],
        object_xy=obj["xyh"][:2],
        overlapping=bool(geometry.get("overlapping")),
        longitudinal_deadband_m=ARGS.spatial_longitudinal_deadband_m,
        lateral_deadband_m=ARGS.spatial_lateral_deadband_m,
        side_longitudinal_band_m=ARGS.spatial_side_longitudinal_band_m,
        front_lateral_base_m=ARGS.spatial_front_lateral_base_m,
        front_lateral_ratio=ARGS.spatial_front_lateral_ratio,
        colocated_epsilon_m=ARGS.colocated_epsilon_m,
    )
    evidence = {
        "subject_id": sid,
        "object_id": oid,
        "subject_track_token": str(subject["track_token"]),
        "object_track_token": str(obj["track_token"]),
        "subject_agent_type": _normalized_agent_type(subject),
        "object_agent_type": _normalized_agent_type(obj),
        "subject_pose": list(subject["xyh"]),
        "object_pose": list(obj["xyh"]),
        "shared_primary_map_id": shared_primary,
        "map_relation": map_relation,
        "map_progress_delta_m": map_progress_delta,
        "same_baseline_progress_delta_m": same_baseline_progress_delta,
        "signed_path_distance_m": signed_path_distance,
        "subject_primary_map_id": subject.get("primary_map_id"),
        "object_primary_map_id": obj.get("primary_map_id"),
        "subject_travel_heading_rad": subject_direction.heading_rad,
        "object_travel_heading_rad": object_direction.heading_rad,
        "subject_travel_direction_source": subject_direction.source,
        "object_travel_direction_source": object_direction.source,
        "travel_direction_difference_rad": spatial.travel_direction_difference_rad,
        "travel_longitudinal_distance_m": spatial.travel_longitudinal_m,
        "travel_lateral_distance_m": spatial.travel_lateral_m,
        "body_frame_spatial_predicate_id": geometric_spatial.predicate_id,
        "body_frame_longitudinal_distance_m": geometric_spatial.travel_longitudinal_m,
        "body_frame_lateral_distance_m": geometric_spatial.travel_lateral_m,
        "body_frame_reference_heading_rad": geometric_spatial.reference_heading_rad,
        **kin,
        **geometry,
        "relevance": (
            None
            if relevance_decision is None
            else {
                "relevant": relevance_decision.relevant,
                "critical": relevance_decision.critical,
                "score": relevance_decision.score,
                "reasons": list(relevance_decision.reasons),
                "primary_reason": relevance_decision.primary_reason,
                "route_relation": relevance_decision.route_relation,
                "signed_path_distance_m": relevance_decision.signed_path_distance_m,
                "cpa_time_s": relevance_decision.cpa_time_s,
                "cpa_clearance_m": relevance_decision.cpa_clearance_m,
                "directional_semantics_allowed": relevance_decision.directional_semantics_allowed,
                "directional_omission_reason": relevance_decision.directional_omission_reason,
            }
        ),
    }
    return {
        "subject": subject,
        "object": obj,
        "kinematics": kin,
        "geometry": geometry,
        "shared_primary_map_id": shared_primary,
        "map_relation": map_relation,
        "subject_direction": subject_direction,
        "object_direction": object_direction,
        "map_progress_delta_m": map_progress_delta,
        "same_baseline_progress_delta_m": same_baseline_progress_delta,
        "signed_path_distance_m": signed_path_distance,
        "spatial": spatial,
        "geometric_spatial": geometric_spatial,
        "relevance_decision": relevance_decision,
        "evidence": evidence,
    }


def append_pairwise_measurements(
    assertions,
    frame,
    state,
    subject,
    obj,
    pair_tag,
    relevance_decision: Optional[PairRelevanceDecision] = None,
):
    sid, oid = subject["entity_id"], obj["entity_id"]
    if str(subject.get("track_token")) == str(obj.get("track_token")):
        return None

    pair_key = (
        str(subject["track_token"]),
        str(obj["track_token"]),
        str(pair_tag),
    )
    pair_id = (
        f"pair:{frame.scenario_token}:{frame.timestamp_us}:{pair_tag}:"
        f"{stable_id(sid, oid)}"
    )
    context = _build_pair_context(subject, obj, relevance_decision)
    evidence = context["evidence"]

    assertions.append(
        derived_relation(
            "np:hasPairwiseState", sid, pair_id, frame,
            "R-RELATIVE-KINEMATICS-001", evidence,
        )
    )
    assertions.append(
        derived_relation(
            "np:pairSubject", pair_id, sid, frame,
            "R-RELATIVE-KINEMATICS-001", evidence,
        )
    )
    assertions.append(
        derived_relation(
            "np:pairObject", pair_id, oid, frame,
            "R-RELATIVE-KINEMATICS-001", evidence,
        )
    )

    append_pair_relevance_assertions(
        assertions, frame, pair_id, pair_tag, context, state=state
    )
    append_pair_geometry_measurements(assertions, frame, pair_id, context)
    append_pair_motion_measurements(assertions, frame, pair_id, context)
    append_pair_map_measurements(assertions, frame, pair_id, context)
    append_pair_risk_predicates(assertions, frame, pair_id, context, state=state)
    append_pair_map_semantics(assertions, frame, pair_id, context, state=state)
    append_pair_spatial_semantics(assertions, frame, pair_id, context, state=state)
    append_pair_heading_semantics(assertions, frame, pair_id, context, state=state)
    append_pair_instant_interactions(assertions, frame, pair_id, context, state=state)
    append_pair_temporal_measurements(
        assertions, frame, state, pair_id, pair_key, context
    )

    result = {
        "pair_id": pair_id,
        "pair_key": pair_key,
        "kinematics": context["kinematics"],
        "geometry": context["geometry"],
        "shared_primary_map_id": context["shared_primary_map_id"],
        "map_relation": context["map_relation"],
        "spatial_classification": context["spatial"],
    }
    state.setdefault("current_pair_measurements", {})[pair_key] = result
    return result
