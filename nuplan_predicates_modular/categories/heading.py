"""Pairwise travel-heading semantic states."""
from ..base import *


def append_pair_heading_semantics(assertions, frame, pair_id, context, state=None):
    if not ARGS.derive_semantic_predicates:
        return
    spatial = context["spatial"]
    evidence = context["evidence"]
    heading_pid = classify_heading_state(
        spatial.travel_direction_difference_rad,
        ARGS.alignment_threshold_rad,
        ARGS.opposite_threshold_rad,
    )
    if heading_pid is not None:
        append_positive_boolean_semantic(
            assertions, heading_pid, pair_id, True, frame,
            "R-MAP-MOTION-SPATIAL-001",
            {**evidence, "exclusive_group": "travel_heading_state"},
            state=state,
        )
