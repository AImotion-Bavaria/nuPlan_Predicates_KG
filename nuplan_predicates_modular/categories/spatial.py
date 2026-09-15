"""V4 corridor-based directional spatial predicates with relevance gating.

The directional class itself is the v4 body/travel-frame geometric classifier
(``context["geometric_spatial"]``): long/lat deadbands, pure lateral band, and
widening front/rear corridor are unchanged.

The only added protection is that the directional predicate is emitted only
when ``relevance_decision.directional_semantics_allowed`` is true. This keeps
unrelated roadblocks, opposite/cross traffic, and out-of-range pairs from
receiving directional spatial semantics.

The footprint contact/distance state remains independent.
"""
from ..base import *


def append_pair_spatial_semantics(assertions, frame, pair_id, context, state=None):
    if not getattr(ARGS, "derive_spatial_predicates", True):
        return

    subject = context["subject"]
    obj = context["object"]
    geometric_spatial = context.get("geometric_spatial")
    map_aware_spatial = context.get("spatial")
    relevance_decision = context.get("relevance_decision")
    geometry = context["geometry"]
    evidence = context["evidence"]

    # ------------------------------------------------------------
    # V4 directional spatial relation + relevance/map/direction gate.
    # ------------------------------------------------------------
    if not (is_dynamic_road_user(subject) and is_dynamic_road_user(obj)):
        record_semantic_omission(
            state,
            "primary_spatial_sector",
            "non_dynamic_road_user",
        )
    elif relevance_decision is None:
        record_semantic_omission(
            state,
            "primary_spatial_sector",
            "missing_relevance_decision",
        )
    elif not relevance_decision.directional_semantics_allowed:
        record_semantic_omission(
            state,
            "primary_spatial_sector",
            relevance_decision.directional_omission_reason
            or "directional_semantics_not_allowed",
        )
    elif geometric_spatial is not None and geometric_spatial.predicate_id is not None:
        spatial_evidence = {
            **evidence,
            "exclusive_group": "primary_spatial_sector",
            "classification_source": geometric_spatial.source,
            "classification_reference": "subject_body_heading_v4_corridor",
            "expected_predicate_id": geometric_spatial.predicate_id,
            "body_frame_longitudinal_m": geometric_spatial.travel_longitudinal_m,
            "body_frame_lateral_m": geometric_spatial.travel_lateral_m,
            "body_frame_reference_heading_rad": geometric_spatial.reference_heading_rad,
            "directional_semantics_allowed": True,
            "directional_relevance_reason": relevance_decision.primary_reason,
            "directional_route_relation": relevance_decision.route_relation,
            "directional_map_relation": relevance_decision.map_relation,
            "map_aware_candidate_predicate_id": (
                None if map_aware_spatial is None else map_aware_spatial.predicate_id
            ),
            "map_aware_classification_source": (
                None if map_aware_spatial is None else map_aware_spatial.source
            ),
            "map_aware_omission_reason": (
                None if map_aware_spatial is None else map_aware_spatial.omission_reason
            ),
            "spatial_thresholds": {
                "longitudinal_deadband_m": ARGS.spatial_longitudinal_deadband_m,
                "lateral_deadband_m": ARGS.spatial_lateral_deadband_m,
                "side_longitudinal_band_m": ARGS.spatial_side_longitudinal_band_m,
                "front_lateral_base_m": ARGS.spatial_front_lateral_base_m,
                "front_lateral_ratio": ARGS.spatial_front_lateral_ratio,
                "colocated_epsilon_m": ARGS.colocated_epsilon_m,
            },
        }
        append_positive_boolean_semantic(
            assertions,
            geometric_spatial.predicate_id,
            pair_id,
            True,
            frame,
            "R-SEMANTIC-THRESHOLD-001",
            spatial_evidence,
            state=state,
        )
    else:
        record_semantic_omission(
            state,
            "primary_spatial_sector",
            (
                None
                if geometric_spatial is None
                else geometric_spatial.omission_reason
            )
            or "invalid_body_frame_spatial_geometry",
        )

    # ------------------------------------------------------------
    # Distance/contact state is independent of directional semantics.
    # ------------------------------------------------------------
    distance_pid = classify_distance_state(
        geometry.get("free_space_distance_m"),
        intersection_area_m2=geometry.get("intersection_area_m2"),
        touching=geometry.get("touching"),
        overlap_area_epsilon_m2=ARGS.overlap_area_epsilon_m2,
        distance_epsilon_m=ARGS.distance_epsilon_m,
        very_near_m=ARGS.very_near_distance_m,
        near_m=ARGS.near_distance_m,
    )
    if distance_pid is not None:
        append_positive_boolean_semantic(
            assertions,
            distance_pid,
            pair_id,
            True,
            frame,
            (
                "R-SEMANTIC-THRESHOLD-001"
                if distance_pid in {"np:near", "np:veryNear"}
                else "R-FOOTPRINT-GEOMETRY-001"
            ),
            {**evidence, "exclusive_group": "distance_state"},
            state=state,
        )
