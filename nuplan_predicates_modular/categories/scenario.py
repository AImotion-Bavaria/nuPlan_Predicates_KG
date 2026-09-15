"""Scenario-level category orchestration."""
from ..base import *
from .future_observation import derive_future_predicates
from .maneuver import derive_maneuver_predicates
from .interaction import (
    derive_following_predicates,
    derive_lane_change_predicates,
    derive_merging_predicates,
    derive_crosses_in_front_predicates,
    derive_yield_predicates,
    derive_overtaking_predicates,
)
from .intent import derive_intent_predicates

def derive_scenario_level_predicates(state):
    snapshots = state.get("snapshots", [])
    if not snapshots:
        return []
    assertions = []
    if ARGS.derive_observed_future_labels and "future_observation" in ACTIVE_CATEGORIES:
        assertions.extend(derive_future_predicates(snapshots))
    if ARGS.derive_semantic_predicates:
        if "maneuver" in ACTIVE_CATEGORIES:
            assertions.extend(derive_maneuver_predicates(snapshots))
        if "interaction" in ACTIVE_CATEGORIES:
            assertions.extend(derive_following_predicates(snapshots))
            lane_change_assertions = derive_lane_change_predicates(snapshots)
            assertions.extend(lane_change_assertions)
            merge_assertions = derive_merging_predicates(
                snapshots, lane_change_assertions=lane_change_assertions
            )
            assertions.extend(merge_assertions)
            crossing_assertions = derive_crosses_in_front_predicates(snapshots)
            assertions.extend(crossing_assertions)
            assertions.extend(derive_yield_predicates(
                snapshots,
                crossing_assertions=crossing_assertions,
            ))
            assertions.extend(derive_overtaking_predicates(snapshots))
        if ARGS.derive_intent_candidates and "intent" in ACTIVE_CATEGORIES:
            assertions.extend(derive_intent_predicates(snapshots))
    return assertions
