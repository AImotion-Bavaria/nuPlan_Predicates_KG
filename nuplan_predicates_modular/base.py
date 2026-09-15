#!/usr/bin/env python3
"""Shared runtime, catalogue, configuration, and assertion helpers.

Category logic must not be added here. Put it in ``categories/<category>.py``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from bisect import bisect_left
from statistics import median
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS
from rdflib.namespace import PROV
from tqdm.auto import tqdm

from .predicate_logic import (
    DEFAULT_THRESHOLDS,
    thresholds_as_dict,
    SEMANTIC_BOOLEAN_PREDICATES,
    canonical_symmetric_pair,
    classify_acceleration_state,
    classify_distance_state,
    classify_heading_state,
    DirectionEvidence,
    classify_relevant_spatial_relation,
    project_relative_position,
    select_travel_direction,
    angular_difference,
    classify_primary_spatial_sector,
    classify_signed_state,
    classify_speed_state,
    observations_are_continuous,
    parse_boolean,
    update_condition_streak,
    check_threshold_relationships,
)
from .relevance_logic import (
    DEFAULT_RELEVANCE_THRESHOLDS,
    PairRelevanceDecision,
    constant_velocity_cpa,
    evaluate_pair_relevance,
    relevance_thresholds_as_dict,
    select_top_relevant,
)


def parse_optional_limit(value: str) -> Optional[int]:
    normalized = str(value).strip().lower()
    if normalized in {"all", "max", "none", "unlimited"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Limit must be positive or 'all'.")
    return parsed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract nuPlan native values, continuous measurements, exact map "
            "relations, temporal history, and positive semantic predicates using built-in thresholds."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("NUPLAN_DATASET_ROOT", Path(__file__).resolve().parents[1] / "data" / "nuplan")),
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path(os.environ.get("NUPLAN_MAP_ROOT", Path(__file__).resolve().parents[1] / "data" / "maps")),
    )
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--output-name", default="paper_release")
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_THRESHOLDS.sample_interval_s)
    parser.add_argument(
        "--max-scenarios",
        type=parse_optional_limit,
        default=None,
        metavar="N|all",
    )
    parser.add_argument(
        "--max-frames-per-scenario",
        type=parse_optional_limit,
        default=None,
        metavar="N|all",
    )
    parser.add_argument(
        "--pairwise-agent-relations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated compatibility switch. Agent-agent processing is now "
            "controlled by whether 'agent-agent' is present in --pair-directions."
        ),
    )
    parser.add_argument(
        "--pair-selection",
        default="all",
        help=(
            "Pair policy per enabled direction. Supply one value to apply it to all "
            "directions, e.g. --pair-selection relevance, or one comma-separated "
            "value for each item in --pair-directions, e.g. "
            "--pair-selection all,relevance,relevance with "
            "--pair-directions ego-agent,agent-ego,agent-agent. Allowed values are "
            "all and relevance."
        ),
    )
    parser.add_argument(
        "--pair-directions",
        default="ego-agent,agent-ego,agent-agent",
        help=(
            "Comma-separated directed relation families to evaluate: ego-agent, "
            "agent-ego, agent-agent. Default enables all three. Presence of "
            "agent-agent is sufficient to enable non-ego agent-pair processing."
        ),
    )
    parser.add_argument(
        "--scenario-filter-yaml",
        type=Path,
        default=None,
        help=(
            "Optional post-builder filter supporting scenario_types, "
            "scenario_tokens, log_names, map_names, num_scenarios_per_type, "
            "limit_total_scenarios, and shuffle."
        ),
    )
    parser.add_argument("--filter-random-seed", type=int, default=42)

    # Expanded ego-agent selection. These rules are combined with logical OR
    # and are mandatory additions to the conservative legacy relevance rules.
    parser.add_argument(
        "--relevance-distance-threshold-m",
        type=float,
        default=40.0,
        help="Select every ego-agent pair whose center distance is at most this value.",
    )
    parser.add_argument(
        "--relevance-include-within-distance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--relevance-include-forward-corridor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--relevance-forward-corridor-length-m",
        type=float,
        default=70.0,
    )
    parser.add_argument(
        "--relevance-forward-corridor-half-width-m",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--relevance-include-same-lane",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--relevance-include-predicted-path-intersection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--relevance-prediction-horizon-s",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--relevance-path-intersection-clearance-m",
        type=float,
        default=3.0,
        help=("Select an agent when constant-velocity footprint clearance to ego "
              "falls below this value during the prediction horizon."),
    )

    # Agent-agent relevance candidate generation and bounded selection.
    # These options affect only --pair-selection relevance. The lossless
    # --pair-selection all mode remains unchanged.
    parser.add_argument(
        "--agent-agent-candidate-radius-m",
        type=float,
        default=60.0,
        help=("Cheap spatial prefilter radius for agent-agent relevance candidates. "
              "Only used when agent-agent selection is relevance."),
    )
    parser.add_argument(
        "--agent-agent-max-neighbors-per-agent",
        type=int,
        default=10,
        help=("Maximum non-critical relevant agent-agent neighbors retained per agent. "
              "Use 0 for no per-agent cap."),
    )
    parser.add_argument(
        "--agent-agent-max-pairs-per-frame",
        type=int,
        default=150,
        help=("Maximum non-critical unordered agent-agent pairs retained per frame. "
              "Critical pairs are always retained. Use 0 for no frame cap."),
    )

    # Scenario-level future, maneuver, traffic-control, visibility, and intent rules
    parser.add_argument("--future-horizon-s", type=float, default=DEFAULT_THRESHOLDS.future_horizon_s)
    parser.add_argument("--future-step-s", type=float, default=DEFAULT_THRESHOLDS.future_step_s)
    parser.add_argument("--stopped-speed-mps", type=float, default=DEFAULT_THRESHOLDS.stopped_speed_mps)
    parser.add_argument("--moving-speed-mps", type=float, default=DEFAULT_THRESHOLDS.moving_speed_mps)
    parser.add_argument("--acceleration-threshold-mps2", type=float, default=DEFAULT_THRESHOLDS.acceleration_threshold_mps2)
    parser.add_argument("--near-distance-m", type=float, default=DEFAULT_THRESHOLDS.near_distance_m)
    parser.add_argument("--very-near-distance-m", type=float, default=DEFAULT_THRESHOLDS.very_near_distance_m)
    parser.add_argument("--alignment-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.alignment_threshold_rad)
    parser.add_argument("--opposite-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.opposite_threshold_rad)
    parser.add_argument("--closing-speed-threshold-mps", type=float, default=DEFAULT_THRESHOLDS.closing_speed_threshold_mps)
    parser.add_argument("--lateral-same-lane-tolerance-m", type=float, default=DEFAULT_THRESHOLDS.lateral_same_lane_tolerance_m)
    parser.add_argument("--maximum-follow-headway-s", type=float, default=DEFAULT_THRESHOLDS.maximum_follow_headway_s)
    parser.add_argument("--minimum-follow-duration-s", type=float, default=DEFAULT_THRESHOLDS.minimum_follow_duration_s)
    parser.add_argument("--maximum-follow-distance-m", type=float, default=DEFAULT_THRESHOLDS.maximum_follow_distance_m)
    parser.add_argument("--follow-queue-speed-threshold-mps", type=float, default=DEFAULT_THRESHOLDS.follow_queue_speed_threshold_mps)
    parser.add_argument("--follow-queue-leader-speed-threshold-mps", type=float, default=DEFAULT_THRESHOLDS.follow_queue_leader_speed_threshold_mps)
    parser.add_argument("--maximum-queue-follow-gap-m", type=float, default=DEFAULT_THRESHOLDS.maximum_queue_follow_gap_m)
    parser.add_argument("--follow-reverse-speed-tolerance-mps", type=float, default=DEFAULT_THRESHOLDS.follow_reverse_speed_tolerance_mps)
    parser.add_argument("--follow-leader-tie-tolerance-m", type=float, default=DEFAULT_THRESHOLDS.follow_leader_tie_tolerance_m)
    parser.add_argument("--follow-max-path-hops", type=int, default=DEFAULT_THRESHOLDS.follow_max_path_hops)
    parser.add_argument("--minimum-overtake-approach-duration-s", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_approach_duration_s)
    parser.add_argument("--minimum-overtake-passing-duration-s", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_passing_duration_s)
    parser.add_argument("--minimum-overtake-completion-duration-s", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_completion_duration_s)
    parser.add_argument("--minimum-overtake-relative-speed-mps", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_relative_speed_mps)
    parser.add_argument("--minimum-overtake-subject-speed-mps", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_subject_speed_mps)
    parser.add_argument("--minimum-overtaken-vehicle-speed-mps", type=float, default=DEFAULT_THRESHOLDS.minimum_overtaken_vehicle_speed_mps)
    parser.add_argument("--minimum-overtake-clearance-m", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_clearance_m)
    parser.add_argument("--maximum-overtake-entry-gap-m", type=float, default=DEFAULT_THRESHOLDS.maximum_overtake_entry_gap_m)
    parser.add_argument("--overtake-departure-max-lead-m", type=float, default=DEFAULT_THRESHOLDS.overtake_departure_max_lead_m)
    parser.add_argument("--overtake-same-flow-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.overtake_same_flow_threshold_rad)
    parser.add_argument("--overtake-lane-stability-frames", type=int, default=DEFAULT_THRESHOLDS.overtake_lane_stability_frames)
    parser.add_argument("--maximum-overtake-lane-transition-duration-s", type=float, default=DEFAULT_THRESHOLDS.maximum_overtake_lane_transition_duration_s)
    parser.add_argument("--maximum-overtake-duration-s", type=float, default=DEFAULT_THRESHOLDS.maximum_overtake_duration_s)
    parser.add_argument("--overtake-pair-query-radius-m", type=float, default=DEFAULT_THRESHOLDS.overtake_pair_query_radius_m)
    parser.add_argument("--overtake-min-common-frames", type=int, default=DEFAULT_THRESHOLDS.overtake_min_common_frames)
    parser.add_argument("--overtake-search-padding-s", type=float, default=DEFAULT_THRESHOLDS.overtake_search_padding_s)
    parser.add_argument("--overtake-max-frame-gap-s", type=float, default=DEFAULT_THRESHOLDS.overtake_max_frame_gap_s)
    parser.add_argument("--minimum-overtake-initial-behind-gap-m", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_initial_behind_gap_m)
    parser.add_argument("--minimum-overtake-side-by-side-lateral-m", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_side_by_side_lateral_m)
    parser.add_argument("--minimum-overtake-same-flow-fraction", type=float, default=DEFAULT_THRESHOLDS.minimum_overtake_same_flow_fraction)
    parser.add_argument("--overtake-lane-query-radius-m", type=float, default=DEFAULT_THRESHOLDS.overtake_lane_query_radius_m)
    parser.add_argument("--overtake-lane-min-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.overtake_lane_min_overlap_ratio)
    parser.add_argument("--overtake-lane-ambiguity-margin", type=float, default=DEFAULT_THRESHOLDS.overtake_lane_ambiguity_margin)
    parser.add_argument("--overtake-max-lane-path-hops", type=int, default=DEFAULT_THRESHOLDS.overtake_max_lane_path_hops)
    parser.add_argument("--overtake-positive-deduplication-tolerance-s", type=float, default=DEFAULT_THRESHOLDS.overtake_positive_deduplication_tolerance_s)
    parser.add_argument("--minimum-maneuver-history-frames", type=int, default=DEFAULT_THRESHOLDS.minimum_maneuver_history_frames, help="Minimum observed precondition frames required before changesLane/overtakes/merge activation.")
    parser.add_argument("--minimum-maneuver-history-s", type=float, default=DEFAULT_THRESHOLDS.minimum_maneuver_history_s, help="Minimum observed precondition history duration before temporal maneuver activation.")
    parser.add_argument("--lane-change-source-stable-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_source_stable_frames)
    parser.add_argument("--lane-change-target-stable-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_target_stable_frames)
    parser.add_argument("--lane-change-max-transition-unknown-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_max_transition_unknown_frames)
    parser.add_argument("--lane-change-minimum-subject-speed-mps", type=float, default=DEFAULT_THRESHOLDS.lane_change_minimum_subject_speed_mps)
    parser.add_argument("--lane-change-maximum-transition-duration-s", type=float, default=DEFAULT_THRESHOLDS.lane_change_maximum_transition_duration_s)
    parser.add_argument("--lane-change-minimum-lateral-shift-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_minimum_lateral_shift_m)
    parser.add_argument("--lane-change-minimum-stable-lane-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.lane_change_minimum_stable_lane_overlap_ratio)
    parser.add_argument(
        "--lane-change-require-dual-lane-overlap",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.lane_change_require_dual_lane_overlap,
        help="Require at least one footprint observation overlapping source and target lanes.",
    )
    parser.add_argument("--lane-change-boundary-evidence-window-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_boundary_evidence_window_frames)
    parser.add_argument("--lane-change-minimum-dual-lane-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.lane_change_minimum_dual_lane_overlap_ratio)
    parser.add_argument("--lane-change-onset-search-s", type=float, default=DEFAULT_THRESHOLDS.lane_change_onset_search_s)
    parser.add_argument("--lane-change-lateral-onset-threshold-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_lateral_onset_threshold_m)
    parser.add_argument("--lane-change-lateral-velocity-onset-threshold-mps", type=float, default=DEFAULT_THRESHOLDS.lane_change_lateral_velocity_onset_threshold_mps)
    parser.add_argument("--lane-change-onset-future-gain-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_onset_future_gain_m)
    parser.add_argument("--lane-change-onset-stable-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_onset_stable_frames)
    parser.add_argument("--lane-change-target-complete-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.lane_change_target_complete_overlap_ratio)
    parser.add_argument("--lane-change-source-remaining-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.lane_change_source_remaining_overlap_ratio)
    parser.add_argument("--lane-change-completion-stable-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_completion_stable_frames)
    parser.add_argument(
        "--lane-change-allow-single-frame-high-confidence-completion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.lane_change_allow_single_frame_high_confidence_completion,
        help="Allow one final high-overlap target-lane frame when the scenario ends immediately after completion.",
    )
    parser.add_argument("--lane-change-single-frame-target-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.lane_change_single_frame_target_overlap_ratio)
    parser.add_argument("--lane-change-max-lane-path-hops", type=int, default=DEFAULT_THRESHOLDS.lane_change_max_lane_path_hops)
    parser.add_argument("--lane-change-adjacency-search-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_adjacency_search_frames)
    parser.add_argument(
        "--lane-change-allow-geometric-adjacency-fallback",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.lane_change_allow_geometric_adjacency_fallback,
        help="Allow conservative geometric adjacency only when official lane adjacency is unavailable.",
    )
    parser.add_argument("--lane-change-geometric-max-polygon-distance-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_geometric_max_polygon_distance_m)
    parser.add_argument("--lane-change-geometric-min-centerline-distance-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_geometric_min_centerline_distance_m)
    parser.add_argument("--lane-change-geometric-max-centerline-distance-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_geometric_max_centerline_distance_m)
    parser.add_argument("--lane-change-geometric-max-direction-difference-rad", type=float, default=DEFAULT_THRESHOLDS.lane_change_geometric_max_direction_difference_rad)
    parser.add_argument("--lane-change-local-min-centerline-distance-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_local_min_centerline_distance_m)
    parser.add_argument("--lane-change-local-max-centerline-distance-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_local_max_centerline_distance_m)
    parser.add_argument("--lane-change-local-max-direction-difference-rad", type=float, default=DEFAULT_THRESHOLDS.lane_change_local_max_direction_difference_rad)
    parser.add_argument(
        "--lane-change-allow-single-frame-primary-target",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.lane_change_allow_single_frame_primary_target,
        help="Allow one high-confidence terminal target-primary frame.",
    )
    parser.add_argument("--lane-change-primary-flicker-max-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_primary_flicker_max_frames)
    parser.add_argument("--lane-change-single-frame-primary-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.lane_change_single_frame_primary_overlap_ratio)
    parser.add_argument(
        "--lane-change-recover-left-censored-events",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.lane_change_recover_left_censored_events,
        help="Recover a track-start lane change from decreasing overlap with one adjacent source lane.",
    )
    parser.add_argument("--lane-change-left-censored-min-source-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.lane_change_left_censored_min_source_overlap_ratio)
    parser.add_argument("--lane-change-left-censored-min-overlap-change", type=float, default=DEFAULT_THRESHOLDS.lane_change_left_censored_min_overlap_change)
    parser.add_argument("--lane-change-temporal-required-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_temporal_required_frames)
    parser.add_argument("--lane-change-temporal-window-frames", type=int, default=DEFAULT_THRESHOLDS.lane_change_temporal_window_frames)
    parser.add_argument("--lane-change-candidate-center-bonus", type=float, default=DEFAULT_THRESHOLDS.lane_change_candidate_center_bonus)
    parser.add_argument("--lane-change-candidate-primary-bonus", type=float, default=DEFAULT_THRESHOLDS.lane_change_candidate_primary_bonus)
    parser.add_argument("--lane-change-candidate-ambiguity-penalty", type=float, default=DEFAULT_THRESHOLDS.lane_change_candidate_ambiguity_penalty)
    parser.add_argument("--lane-change-max-grouping-lane-ids", type=int, default=DEFAULT_THRESHOLDS.lane_change_max_grouping_lane_ids)
    parser.add_argument("--merge-target-preexistence-window-s", type=float, default=DEFAULT_THRESHOLDS.merge_target_preexistence_window_s)
    parser.add_argument("--merge-order-confirmation-window-s", type=float, default=DEFAULT_THRESHOLDS.merge_order_confirmation_window_s)
    parser.add_argument("--merge-order-stable-frames", type=int, default=DEFAULT_THRESHOLDS.merge_order_stable_frames)
    parser.add_argument("--merge-minimum-preexisting-target-frames", type=int, default=DEFAULT_THRESHOLDS.merge_minimum_preexisting_target_frames)
    parser.add_argument("--merge-minimum-path-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.merge_minimum_path_overlap_ratio)
    parser.add_argument("--merge-same-flow-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.merge_same_flow_threshold_rad)
    parser.add_argument("--merge-maximum-neighbor-gap-m", type=float, default=DEFAULT_THRESHOLDS.merge_maximum_neighbor_gap_m)
    parser.add_argument("--merge-maximum-headway-s", type=float, default=DEFAULT_THRESHOLDS.merge_maximum_headway_s)
    parser.add_argument("--merge-max-target-path-hops", type=int, default=DEFAULT_THRESHOLDS.merge_max_target_path_hops)
    parser.add_argument(
        "--merge-allow-terminal-single-frame",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.merge_allow_terminal_single_frame,
        help="Allow one final target-stream order observation when the scenario ends immediately after the merge.",
    )

    # crossesInFrontOf: V2 local, footprint-aware side-to-side detector.
    parser.add_argument("--crosses-in-front-min-observations", type=int, default=DEFAULT_THRESHOLDS.crosses_in_front_min_observations)
    parser.add_argument("--crosses-in-front-min-track-displacement-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_min_track_displacement_m)
    parser.add_argument("--crosses-in-front-min-simultaneous-observation-s", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_min_simultaneous_observation_s)
    parser.add_argument("--crosses-in-front-max-pair-distance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_max_pair_distance_m)
    parser.add_argument("--crosses-in-front-max-front-distance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_max_front_distance_m)
    parser.add_argument("--crosses-in-front-min-front-clearance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_min_front_clearance_m)
    parser.add_argument("--crosses-in-front-min-interval-front-clearance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_min_interval_front_clearance_m)
    parser.add_argument("--crosses-in-front-side-buffer-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_side_buffer_m)
    parser.add_argument("--crosses-in-front-side-search-window-s", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_side_search_window_s)
    parser.add_argument("--crosses-in-front-min-crossing-angle-deg", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_min_crossing_angle_deg)
    parser.add_argument("--crosses-in-front-max-crossing-angle-deg", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_max_crossing_angle_deg)
    parser.add_argument("--crosses-in-front-min-subject-crossing-speed-mps", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_min_subject_crossing_speed_mps)
    parser.add_argument("--crosses-in-front-order-margin-s", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_order_margin_s)
    parser.add_argument("--crosses-in-front-max-arrival-time-difference-s", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_max_arrival_time_difference_s)
    parser.add_argument("--crosses-in-front-conflict-footprint-buffer-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_conflict_footprint_buffer_m)
    parser.add_argument("--crosses-in-front-max-stationary-object-speed-mps", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_max_stationary_object_speed_mps)
    parser.add_argument("--crosses-in-front-max-stationary-object-front-distance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_max_stationary_object_front_distance_m)
    parser.add_argument("--crosses-in-front-map-query-radius-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_map_query_radius_m)
    parser.add_argument("--crosses-in-front-max-road-distance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_max_road_distance_m)
    parser.add_argument("--crosses-in-front-crosswalk-near-distance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_crosswalk_near_distance_m)
    parser.add_argument("--crosses-in-front-bicycle-road-distance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_bicycle_road_distance_m)
    parser.add_argument("--crosses-in-front-parked-max-speed-mps", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_parked_max_speed_mps)
    parser.add_argument("--crosses-in-front-parked-min-duration-s", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_parked_min_duration_s)
    parser.add_argument("--crosses-in-front-parked-max-displacement-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_parked_max_displacement_m)
    parser.add_argument("--crosses-in-front-parked-max-lane-distance-m", type=float, default=DEFAULT_THRESHOLDS.crosses_in_front_parked_max_lane_distance_m)
    parser.add_argument(
        "--crosses-in-front-include-pedestrian-pedestrian",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.crosses_in_front_include_pedestrian_pedestrian,
        help="Include pedestrian-pedestrian crossing relations; disabled by default.",
    )

    # yieldsTo: retrospectively verified behavioral priority giving.
    parser.add_argument("--yield-pre-event-window-s", type=float, default=DEFAULT_THRESHOLDS.yield_pre_event_window_s)
    parser.add_argument("--yield-post-event-window-s", type=float, default=DEFAULT_THRESHOLDS.yield_post_event_window_s)
    parser.add_argument("--yield-max-initial-eta-difference-s", type=float, default=DEFAULT_THRESHOLDS.yield_max_initial_eta_difference_s)
    parser.add_argument("--yield-min-baseline-speed-mps", type=float, default=DEFAULT_THRESHOLDS.yield_min_baseline_speed_mps)
    parser.add_argument("--yield-min-speed-drop-mps", type=float, default=DEFAULT_THRESHOLDS.yield_min_speed_drop_mps)
    parser.add_argument("--yield-slow-speed-mps", type=float, default=DEFAULT_THRESHOLDS.yield_slow_speed_mps)
    parser.add_argument("--yield-max-subject-conflict-distance-m", type=float, default=DEFAULT_THRESHOLDS.yield_max_subject_conflict_distance_m)
    parser.add_argument("--yield-conflict-clearance-buffer-m", type=float, default=DEFAULT_THRESHOLDS.yield_conflict_clearance_buffer_m)
    parser.add_argument("--yield-min-proceed-speed-mps", type=float, default=DEFAULT_THRESHOLDS.yield_min_proceed_speed_mps)
    parser.add_argument("--yield-min-proceed-displacement-m", type=float, default=DEFAULT_THRESHOLDS.yield_min_proceed_displacement_m)

    parser.add_argument("--turn-angle-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.turn_angle_threshold_rad)
    parser.add_argument("--uturn-angle-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.uturn_angle_threshold_rad)
    parser.add_argument("--stop-line-search-radius-m", type=float, default=DEFAULT_THRESHOLDS.stop_line_search_radius_m)
    parser.add_argument("--stop-line-near-distance-m", type=float, default=DEFAULT_THRESHOLDS.stop_line_near_distance_m)
    parser.add_argument("--minimum-red-wait-duration-s", type=float, default=DEFAULT_THRESHOLDS.minimum_red_wait_duration_s)
    parser.add_argument("--ego-fov-deg", type=float, default=DEFAULT_THRESHOLDS.ego_fov_deg)
    parser.add_argument("--ego-fov-range-m", type=float, default=DEFAULT_THRESHOLDS.ego_fov_range_m)
    parser.add_argument("--minimum-intent-duration-s", type=float, default=DEFAULT_THRESHOLDS.minimum_intent_duration_s)
    parser.add_argument("--gap-increase-threshold-m", type=float, default=DEFAULT_THRESHOLDS.gap_increase_threshold_m)
    parser.add_argument("--arrival-time-tolerance-s", type=float, default=DEFAULT_THRESHOLDS.arrival_time_tolerance_s)
    # v9.5.45 risk1 refinement: conservative current-state conflict risk.
    parser.add_argument(
        "--conflict-risk-horizon-s", type=float, default=2.0,
        help="Short constant-velocity prediction horizon for np:hasConflictRiskWith.",
    )
    parser.add_argument(
        "--conflict-risk-clearance-m", type=float, default=0.5,
        help="Maximum predicted oriented-footprint free-space clearance for conflict risk.",
    )
    parser.add_argument(
        "--conflict-risk-min-closing-speed-mps",
        "--conflict-risk-min-relative-speed-mps",
        dest="conflict_risk_min_closing_speed_mps",
        type=float,
        default=0.5,
        help=(
            "Minimum radial closing speed required for conflict risk. "
            "The old --conflict-risk-min-relative-speed-mps spelling is retained as an alias."
        ),
    )
    parser.add_argument(
        "--conflict-risk-min-clearance-reduction-m", type=float, default=1.0,
        help="Minimum decrease from current to predicted free-space clearance.",
    )
    parser.add_argument(
        "--conflict-risk-cross-traffic-max-center-distance-m", type=float, default=40.0,
        help="Maximum current center distance for an otherwise-unconnected crossing/opposite-flow candidate.",
    )
    parser.add_argument(
        "--conflict-risk-vru-max-center-distance-m", type=float, default=30.0,
        help="Maximum current center distance for a candidate containing a vulnerable road user.",
    )
    parser.add_argument(
        "--conflict-risk-same-flow-corridor-lateral-m", type=float, default=2.5,
        help="Maximum body-frame lateral offset for the map-fallback same-flow corridor candidate.",
    )
    parser.add_argument(
        "--conflict-risk-same-flow-corridor-center-m", type=float, default=45.0,
        help="Maximum center distance for the map-fallback same-flow corridor candidate.",
    )
    parser.add_argument(
        "--conflict-risk-crossing-angle-deg", type=float, default=25.0,
        help="Minimum current motion-direction difference for an unconnected crossing/opposite-flow candidate.",
    )
    parser.add_argument(
        "--conflict-risk-optimization-iterations", type=int, default=24,
        help="Deterministic bounded-search iterations for predicted footprint clearance.",
    )
    parser.add_argument(
        "--conflict-risk-imminent-braking-demand-mps2", type=float, default=5.0,
        help="UN R157 imminent-collision braking-demand threshold (m/s^2).",
    )
    parser.add_argument(
        "--conflict-risk-lane-change-max-target-decel-mps2", type=float, default=3.0,
        help="UN R157 maximum target-lane vehicle deceleration for regular lane changes (m/s^2).",
    )
    parser.add_argument(
        "--conflict-risk-lane-change-min-gap-time-s", type=float, default=1.0,
        help="UN R157 minimum lane-change gap expressed as ALKS travel time (s).",
    )
    parser.add_argument("--maximum-temporal-gap-factor", type=float, default=DEFAULT_THRESHOLDS.maximum_temporal_gap_factor)
    parser.add_argument("--minimum-acceleration-samples", type=int, default=DEFAULT_THRESHOLDS.minimum_acceleration_samples)
    parser.add_argument("--distance-epsilon-m", type=float, default=DEFAULT_THRESHOLDS.distance_epsilon_m)
    parser.add_argument("--overlap-area-epsilon-m2", type=float, default=DEFAULT_THRESHOLDS.overlap_area_epsilon_m2)
    parser.add_argument("--colocated-epsilon-m", type=float, default=DEFAULT_THRESHOLDS.colocated_epsilon_m)
    parser.add_argument(
        "--spatial-longitudinal-deadband-m", type=float,
        default=DEFAULT_THRESHOLDS.spatial_longitudinal_deadband_m,
        help="Minimum reliable front/rear displacement in the subject frame.",
    )
    parser.add_argument(
        "--spatial-lateral-deadband-m", type=float,
        default=DEFAULT_THRESHOLDS.spatial_lateral_deadband_m,
        help="Minimum reliable left/right displacement in the subject frame.",
    )
    parser.add_argument(
        "--spatial-side-longitudinal-band-m", type=float,
        default=DEFAULT_THRESHOLDS.spatial_side_longitudinal_band_m,
        help="Pure left/right is allowed only inside this longitudinal band.",
    )
    parser.add_argument(
        "--spatial-front-lateral-base-m", type=float,
        default=DEFAULT_THRESHOLDS.spatial_front_lateral_base_m,
        help="Base half-width of the front/rear corridor.",
    )
    parser.add_argument(
        "--spatial-front-lateral-ratio", type=float,
        default=DEFAULT_THRESHOLDS.spatial_front_lateral_ratio,
        help="Distance-dependent widening ratio of the front/rear corridor.",
    )
    parser.add_argument("--spatial-relevance-max-distance-m", type=float, default=DEFAULT_THRESHOLDS.spatial_relevance_max_distance_m)
    parser.add_argument("--spatial-adjacent-relevance-max-distance-m", type=float, default=DEFAULT_THRESHOLDS.spatial_adjacent_relevance_max_distance_m)
    parser.add_argument("--spatial-diagonal-max-longitudinal-m", type=float, default=DEFAULT_THRESHOLDS.spatial_diagonal_max_longitudinal_m)
    parser.add_argument("--spatial-motion-only-fallback-max-distance-m", type=float, default=DEFAULT_THRESHOLDS.spatial_motion_only_fallback_max_distance_m)
    parser.add_argument("--spatial-motion-only-max-lateral-m", type=float, default=DEFAULT_THRESHOLDS.spatial_motion_only_max_lateral_m)
    parser.add_argument("--spatial-min-direction-speed-mps", type=float, default=DEFAULT_THRESHOLDS.spatial_min_direction_speed_mps)
    parser.add_argument("--spatial-min-displacement-for-direction-m", type=float, default=DEFAULT_THRESHOLDS.spatial_min_displacement_for_direction_m)
    parser.add_argument("--spatial-direction-agreement-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.spatial_direction_agreement_threshold_rad)
    parser.add_argument("--spatial-map-motion-tolerance-rad", type=float, default=DEFAULT_THRESHOLDS.spatial_map_motion_tolerance_rad)
    parser.add_argument("--spatial-same-flow-threshold-rad", type=float, default=DEFAULT_THRESHOLDS.spatial_same_flow_threshold_rad)
    parser.add_argument("--spatial-connected-sign-tolerance-m", type=float, default=DEFAULT_THRESHOLDS.spatial_connected_sign_tolerance_m)
    parser.add_argument(
        "--require-map-context-for-spatial",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_THRESHOLDS.spatial_require_map_context,
        help=("Require same, adjacent, predecessor, or successor map context before "
              "emitting front/behind/left/right semantics. Enabled by default."),
    )
    parser.add_argument("--minimum-forward-speed-mps", type=float, default=DEFAULT_THRESHOLDS.minimum_forward_speed_mps)
    parser.add_argument("--minimum-future-coverage-ratio", type=float, default=DEFAULT_THRESHOLDS.minimum_future_coverage_ratio)
    parser.add_argument("--lane-change-lateral-threshold-m", type=float, default=DEFAULT_THRESHOLDS.lane_change_lateral_threshold_m)
    parser.add_argument("--lane-stability-frames", type=int, default=DEFAULT_THRESHOLDS.lane_stability_frames)
    parser.add_argument("--primary-lane-min-overlap-ratio", type=float, default=DEFAULT_THRESHOLDS.primary_lane_min_overlap_ratio)
    parser.add_argument("--primary-lane-ambiguity-margin", type=float, default=DEFAULT_THRESHOLDS.primary_lane_ambiguity_margin)
    parser.add_argument(
        "--primary-map-lateral-tiebreak-margin-m",
        type=float,
        default=DEFAULT_THRESHOLDS.primary_map_lateral_tiebreak_margin_m,
        help=("Select the best geometric map candidate when its absolute baseline "
              "lateral offset is better than the runner-up by at least this margin."),
    )
    parser.add_argument(
        "--derive-observed-future-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Derive retrospective observed-future labels. Disabled by default to prevent leakage.",
    )
    parser.add_argument(
        "--derive-intent-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Derive conservative inferred intent candidates. Disabled by default.",
    )
    parser.add_argument(
        "--derive-spatial-predicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Emit the spatial relation predicates (front/behind/left/right, "
            "overlap, near, and related spatial states). Enabled by default."
        ),
    )
    parser.add_argument(
        "--derive-semantic-predicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Emit higher-level semantic predicates such as interaction, risk, "
            "and traffic-control relations. Enabled by default for the paper release."
        ),
    )
    parser.add_argument(
        "--categories",
        default="all",
        help=("Comma-separated predicate categories to write, e.g. "
              "'spatial,map,geometry'. Extraction dependencies are still computed."),
    )
    parser.add_argument(
        "--include-category-dependencies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also WRITE dependency-category assertions. Dependencies required for "
            "computation are always available internally. Disabled by default so "
            "--categories is a strict persisted-output selection."
        ),
    )
    parser.add_argument(
        "--write-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write detailed audit/debug CSV files (relevance audits, temporal "
            "continuity details, semantic omissions). Disabled by default to keep "
            "the output compact. Summary counts are still retained."
        ),
    )
    parser.add_argument("--write-jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-rdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compact-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write a compact Parquet schema containing the assertion identity, "
            "relation/value, time, scenario/rule provenance, and semantic evidence. "
            "Repeated run-level provenance fields are omitted. Enabled by default."
        ),
    )
    parser.add_argument("--batch-size-scenarios", type=int, default=5)
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help=("Number of isolated scenario workers. Workers write separate output "
              "directories and keep independent map caches. Start with 2-4 on "
              "low RAM; increase on a high-RAM machine."),
    )
    parser.add_argument(
        "--worker-index", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-count", type=int, default=1, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-progress-file", type=Path, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--task-queue-file", type=Path, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--parquet-rows-per-part", type=int, default=5000,
        help=("Maximum assertions per Parquet part. Use 5,000-20,000 for low RAM; "
              "100,000-500,000 on a high-RAM machine."),
    )
    parser.add_argument(
        "--parquet-compression", choices=("zstd", "snappy", "gzip", "none"),
        default="zstd", help="Parquet compression. snappy is faster; zstd is smaller.",
    )
    parser.add_argument(
        "--map-cache", action=argparse.BooleanOptionalAction, default=True,
        help="Cache static map objects and topology for each scenario.",
    )
    parser.add_argument(
        "--proximal-map-cache-resolution-m", type=float, default=0.0,
        help=("Optional spatial quantization for proximal map-query caching. 0 disables "
              "approximate query caching and preserves exact candidate queries. Try 0.5 "
              "or 1.0 only when maximum speed is preferred."),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fail-on-frame-error", action="store_true")
    args = parser.parse_args(argv)

    if args.sample_interval_s <= 0:
        parser.error("--sample-interval-s must be positive")
    if args.batch_size_scenarios <= 0:
        parser.error("--batch-size-scenarios must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.worker_count <= 0:
        parser.error("--worker-count must be positive")
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        parser.error("--worker-index must satisfy 0 <= index < worker-count")
    if args.parquet_rows_per_part <= 0:
        parser.error("--parquet-rows-per-part must be positive")
    if args.proximal_map_cache_resolution_m < 0:
        parser.error("--proximal-map-cache-resolution-m must be non-negative")

    allowed_pair_directions = {"ego-agent", "agent-ego", "agent-agent"}
    parsed_pair_direction_order = [
        item.strip().lower() for item in str(args.pair_directions).split(",") if item.strip()
    ]
    unknown_pair_directions = set(parsed_pair_direction_order) - allowed_pair_directions
    if unknown_pair_directions:
        parser.error(
            f"Unknown --pair-directions values: {sorted(unknown_pair_directions)}. "
            f"Allowed: {sorted(allowed_pair_directions)}"
        )
    if not parsed_pair_direction_order:
        parser.error("--pair-directions must enable at least one relation family")
    if len(set(parsed_pair_direction_order)) != len(parsed_pair_direction_order):
        parser.error("--pair-directions must not contain duplicate relation families")

    allowed_pair_selections = {"all", "relevance"}
    parsed_pair_selections = [
        item.strip().lower() for item in str(args.pair_selection).split(",") if item.strip()
    ]
    if not parsed_pair_selections:
        parser.error("--pair-selection must contain at least one value")
    unknown_pair_selections = set(parsed_pair_selections) - allowed_pair_selections
    if unknown_pair_selections:
        parser.error(
            f"Unknown --pair-selection values: {sorted(unknown_pair_selections)}. "
            f"Allowed: {sorted(allowed_pair_selections)}"
        )
    if len(parsed_pair_selections) == 1:
        parsed_pair_selections = parsed_pair_selections * len(parsed_pair_direction_order)
    elif len(parsed_pair_selections) != len(parsed_pair_direction_order):
        parser.error(
            "--pair-selection must contain either one value or exactly one value "
            "for each item in --pair-directions"
        )

    args.pair_direction_order = tuple(parsed_pair_direction_order)
    args.pair_directions = set(parsed_pair_direction_order)
    # v3: --pair-directions is the single source of truth. Keep the legacy
    # attribute synchronized for old internal/reporting code.
    args.pairwise_agent_relations = "agent-agent" in args.pair_directions
    args.pair_selection_by_direction = dict(
        zip(parsed_pair_direction_order, parsed_pair_selections)
    )
    # Backward-compatible scalar when all enabled directions share one mode.
    args.pair_selection = (
        parsed_pair_selections[0]
        if len(set(parsed_pair_selections)) == 1
        else ",".join(parsed_pair_selections)
    )

    for name in (
        "future_horizon_s", "future_step_s", "stopped_speed_mps",
        "moving_speed_mps", "near_distance_m", "very_near_distance_m",
        "alignment_threshold_rad", "opposite_threshold_rad",
        "closing_speed_threshold_mps", "lateral_same_lane_tolerance_m",
        "maximum_follow_headway_s", "minimum_follow_duration_s",
        "maximum_follow_distance_m", "follow_queue_speed_threshold_mps",
        "follow_queue_leader_speed_threshold_mps", "maximum_queue_follow_gap_m",
        "follow_reverse_speed_tolerance_mps", "follow_leader_tie_tolerance_m",
        "minimum_overtake_approach_duration_s",
        "minimum_overtake_passing_duration_s",
        "minimum_overtake_completion_duration_s",
        "minimum_overtake_relative_speed_mps",
        "minimum_overtake_subject_speed_mps",
        "minimum_overtaken_vehicle_speed_mps",
        "minimum_overtake_clearance_m",
        "maximum_overtake_entry_gap_m",
        "overtake_departure_max_lead_m",
        "overtake_same_flow_threshold_rad",
        "maximum_overtake_lane_transition_duration_s",
        "maximum_overtake_duration_s",
        "overtake_pair_query_radius_m",
        "overtake_search_padding_s",
        "overtake_max_frame_gap_s",
        "minimum_overtake_initial_behind_gap_m",
        "minimum_overtake_side_by_side_lateral_m",
        "minimum_overtake_same_flow_fraction",
        "overtake_lane_query_radius_m",
        "overtake_lane_min_overlap_ratio",
        "overtake_lane_ambiguity_margin",
        "overtake_positive_deduplication_tolerance_s",
        "minimum_maneuver_history_s",
        "lane_change_minimum_subject_speed_mps",
        "lane_change_maximum_transition_duration_s",
        "lane_change_minimum_lateral_shift_m",
        "lane_change_minimum_stable_lane_overlap_ratio",
        "lane_change_minimum_dual_lane_overlap_ratio",
        "lane_change_onset_search_s",
        "lane_change_lateral_onset_threshold_m",
        "lane_change_target_complete_overlap_ratio",
        "lane_change_source_remaining_overlap_ratio",
        "lane_change_geometric_max_polygon_distance_m",
        "lane_change_geometric_min_centerline_distance_m",
        "lane_change_geometric_max_centerline_distance_m",
        "lane_change_geometric_max_direction_difference_rad",
        "merge_target_preexistence_window_s", "merge_order_confirmation_window_s",
        "merge_minimum_path_overlap_ratio", "merge_same_flow_threshold_rad",
        "merge_maximum_neighbor_gap_m", "merge_maximum_headway_s",
        "crosses_in_front_min_track_displacement_m",
        "crosses_in_front_min_simultaneous_observation_s",
        "crosses_in_front_max_pair_distance_m",
        "crosses_in_front_max_front_distance_m",
        "crosses_in_front_side_buffer_m",
        "crosses_in_front_side_search_window_s",
        "crosses_in_front_min_crossing_angle_deg",
        "crosses_in_front_max_crossing_angle_deg",
        "crosses_in_front_min_subject_crossing_speed_mps",
        "crosses_in_front_max_arrival_time_difference_s",
        "crosses_in_front_order_margin_s",
        "crosses_in_front_conflict_footprint_buffer_m",
        "crosses_in_front_max_stationary_object_speed_mps",
        "crosses_in_front_max_stationary_object_front_distance_m",
        "crosses_in_front_map_query_radius_m",
        "crosses_in_front_max_road_distance_m",
        "crosses_in_front_crosswalk_near_distance_m",
        "crosses_in_front_bicycle_road_distance_m",
        "crosses_in_front_parked_max_speed_mps",
        "crosses_in_front_parked_min_duration_s",
        "crosses_in_front_parked_max_displacement_m",
        "crosses_in_front_parked_max_lane_distance_m",
        "yield_pre_event_window_s", "yield_post_event_window_s",
        "yield_max_initial_eta_difference_s", "yield_min_baseline_speed_mps",
        "yield_min_speed_drop_mps", "yield_slow_speed_mps",
        "yield_max_subject_conflict_distance_m", "yield_conflict_clearance_buffer_m",
        "yield_min_proceed_speed_mps", "yield_min_proceed_displacement_m",
        "turn_angle_threshold_rad", "uturn_angle_threshold_rad",
        "stop_line_search_radius_m", "stop_line_near_distance_m",
        "minimum_red_wait_duration_s", "ego_fov_deg", "ego_fov_range_m",
        "minimum_intent_duration_s", "gap_increase_threshold_m",
        "arrival_time_tolerance_s",
        "maximum_temporal_gap_factor", "distance_epsilon_m",
        "overlap_area_epsilon_m2", "colocated_epsilon_m",
        "minimum_forward_speed_mps", "minimum_future_coverage_ratio",
        "lane_change_lateral_threshold_m", "primary_lane_min_overlap_ratio",
        "primary_lane_ambiguity_margin", "primary_map_lateral_tiebreak_margin_m",
        "spatial_longitudinal_deadband_m",
        "spatial_lateral_deadband_m", "spatial_side_longitudinal_band_m",
        "spatial_front_lateral_base_m", "spatial_front_lateral_ratio",
        "spatial_relevance_max_distance_m", "spatial_adjacent_relevance_max_distance_m",
        "spatial_diagonal_max_longitudinal_m", "spatial_motion_only_fallback_max_distance_m",
        "spatial_motion_only_max_lateral_m", "spatial_min_direction_speed_mps",
        "spatial_min_displacement_for_direction_m", "spatial_direction_agreement_threshold_rad",
        "spatial_map_motion_tolerance_rad", "spatial_same_flow_threshold_rad",
        "spatial_connected_sign_tolerance_m",
        "relevance_distance_threshold_m",
        "relevance_forward_corridor_length_m",
        "relevance_forward_corridor_half_width_m",
        "relevance_prediction_horizon_s",
        "relevance_path_intersection_clearance_m",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.conflict_risk_horizon_s <= 0:
        parser.error("--conflict-risk-horizon-s must be positive")
    if args.conflict_risk_clearance_m < 0:
        parser.error("--conflict-risk-clearance-m must be non-negative")
    if args.conflict_risk_min_closing_speed_mps < 0:
        parser.error("--conflict-risk-min-closing-speed-mps must be non-negative")
    if args.conflict_risk_min_clearance_reduction_m < 0:
        parser.error("--conflict-risk-min-clearance-reduction-m must be non-negative")
    if args.conflict_risk_cross_traffic_max_center_distance_m <= 0:
        parser.error("--conflict-risk-cross-traffic-max-center-distance-m must be positive")
    if args.conflict_risk_vru_max_center_distance_m <= 0:
        parser.error("--conflict-risk-vru-max-center-distance-m must be positive")
    if args.conflict_risk_same_flow_corridor_lateral_m <= 0:
        parser.error("--conflict-risk-same-flow-corridor-lateral-m must be positive")
    if args.conflict_risk_same_flow_corridor_center_m <= 0:
        parser.error("--conflict-risk-same-flow-corridor-center-m must be positive")
    if not 0.0 <= args.conflict_risk_crossing_angle_deg <= 180.0:
        parser.error("--conflict-risk-crossing-angle-deg must be in [0, 180]")
    if args.conflict_risk_optimization_iterations < 8:
        parser.error("--conflict-risk-optimization-iterations must be at least 8")
    if args.conflict_risk_imminent_braking_demand_mps2 <= 0:
        parser.error("--conflict-risk-imminent-braking-demand-mps2 must be positive")
    if args.conflict_risk_lane_change_max_target_decel_mps2 <= 0:
        parser.error("--conflict-risk-lane-change-max-target-decel-mps2 must be positive")
    if args.conflict_risk_lane_change_min_gap_time_s <= 0:
        parser.error("--conflict-risk-lane-change-min-gap-time-s must be positive")
    threshold_errors = check_threshold_relationships(args)
    if threshold_errors:
        parser.error("; ".join(threshold_errors))
    if args.lane_stability_frames < 2:
        parser.error("--lane-stability-frames must be at least 2")
    if args.overtake_lane_stability_frames < 2:
        parser.error("--overtake-lane-stability-frames must be at least 2")
    if args.overtake_min_common_frames < 2:
        parser.error("--overtake-min-common-frames must be at least 2")
    if args.minimum_maneuver_history_frames < 1:
        parser.error("--minimum-maneuver-history-frames must be at least 1")
    if args.overtake_max_lane_path_hops < 1:
        parser.error("--overtake-max-lane-path-hops must be at least 1")
    if args.lane_change_source_stable_frames < 1:
        parser.error("--lane-change-source-stable-frames must be at least 1")
    if args.lane_change_target_stable_frames < 2:
        parser.error("--lane-change-target-stable-frames must be at least 2")
    if args.lane_change_max_transition_unknown_frames < 0:
        parser.error("--lane-change-max-transition-unknown-frames must be non-negative")
    if args.lane_change_boundary_evidence_window_frames < 0:
        parser.error("--lane-change-boundary-evidence-window-frames must be non-negative")
    if args.lane_change_onset_stable_frames < 1:
        parser.error("--lane-change-onset-stable-frames must be at least 1")
    if args.lane_change_completion_stable_frames < 1:
        parser.error("--lane-change-completion-stable-frames must be at least 1")
    if args.lane_change_temporal_required_frames < 2:
        parser.error("--lane-change-temporal-required-frames must be at least 2")
    if args.lane_change_temporal_window_frames < args.lane_change_temporal_required_frames:
        parser.error(
            "--lane-change-temporal-window-frames must be at least "
            "--lane-change-temporal-required-frames"
        )
    if args.lane_change_max_grouping_lane_ids < 1:
        parser.error("--lane-change-max-grouping-lane-ids must be at least 1")
    if args.merge_order_stable_frames < 1:
        parser.error("--merge-order-stable-frames must be at least 1")
    if args.merge_minimum_preexisting_target_frames < 1:
        parser.error("--merge-minimum-preexisting-target-frames must be at least 1")
    if args.merge_max_target_path_hops < 1:
        parser.error("--merge-max-target-path-hops must be at least 1")
    if args.crosses_in_front_min_observations < 2:
        parser.error("--crosses-in-front-min-observations must be at least 2")
    for option_name in (
        "yield_pre_event_window_s",
        "yield_post_event_window_s",
        "yield_max_initial_eta_difference_s",
        "yield_min_baseline_speed_mps",
        "yield_min_speed_drop_mps",
        "yield_slow_speed_mps",
        "yield_max_subject_conflict_distance_m",
        "yield_conflict_clearance_buffer_m",
        "yield_min_proceed_speed_mps",
        "yield_min_proceed_displacement_m",
    ):
        if getattr(args, option_name) < 0:
            parser.error(f"--{option_name.replace('_', '-')} must be non-negative")
    if args.yield_pre_event_window_s <= 0 or args.yield_post_event_window_s <= 0:
        parser.error("--yield-pre-event-window-s and --yield-post-event-window-s must be positive")
    if args.yield_max_initial_eta_difference_s <= 0:
        parser.error("--yield-max-initial-eta-difference-s must be positive")
    if not 0 <= args.merge_minimum_path_overlap_ratio <= 1:
        parser.error("--merge-minimum-path-overlap-ratio must be in [0, 1]")
    for option_name in (
        "lane_change_candidate_center_bonus",
        "lane_change_candidate_primary_bonus",
        "lane_change_candidate_ambiguity_penalty",
    ):
        if getattr(args, option_name) < 0:
            parser.error(f"--{option_name.replace('_', '-')} must be non-negative")
    for option_name in (
        "lane_change_minimum_stable_lane_overlap_ratio",
        "lane_change_minimum_dual_lane_overlap_ratio",
        "lane_change_target_complete_overlap_ratio",
        "lane_change_source_remaining_overlap_ratio",
    ):
        if not 0 <= getattr(args, option_name) <= 1:
            parser.error(f"--{option_name.replace('_', '-')} must be in [0, 1]")
    if (
        args.lane_change_geometric_min_centerline_distance_m
        > args.lane_change_geometric_max_centerline_distance_m
    ):
        parser.error(
            "--lane-change-geometric-min-centerline-distance-m must not exceed "
            "--lane-change-geometric-max-centerline-distance-m"
        )
    if not 0 < args.minimum_overtake_same_flow_fraction <= 1:
        parser.error("--minimum-overtake-same-flow-fraction must be in (0, 1]")
    if not 0 <= args.overtake_lane_min_overlap_ratio <= 1:
        parser.error("--overtake-lane-min-overlap-ratio must be in [0, 1]")
    if not 0 <= args.overtake_lane_ambiguity_margin <= 1:
        parser.error("--overtake-lane-ambiguity-margin must be in [0, 1]")
    if not 0 <= args.primary_lane_min_overlap_ratio <= 1:
        parser.error("--primary-lane-min-overlap-ratio must be in [0, 1]")
    if not 0 <= args.primary_lane_ambiguity_margin <= 1:
        parser.error("--primary-lane-ambiguity-margin must be in [0, 1]")
    if args.primary_map_lateral_tiebreak_margin_m < 0:
        parser.error("--primary-map-lateral-tiebreak-margin-m must be non-negative")
    if not any((args.write_jsonl, args.write_parquet, args.write_rdf)):
        parser.error("At least one output format must be enabled")
    return args


# Importing the package from Python remains side-effect free.  The public CLI
# passes its arguments through a short-lived environment variable *before* this
# module is imported.  This lets the rule/predicate catalog be constructed from
# the actual runtime thresholds while normal library/test imports still use only
# parser defaults and never consume the host process command line.
_runtime_argv_json = os.environ.pop("PREDICATE_KG_CLI_ARGS_JSON", None)
if _runtime_argv_json:
    try:
        _runtime_argv = json.loads(_runtime_argv_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid PREDICATE_KG_CLI_ARGS_JSON") from exc
    if not isinstance(_runtime_argv, list):
        raise RuntimeError("PREDICATE_KG_CLI_ARGS_JSON must encode a list")
else:
    _runtime_argv = []
ARGS = parse_args(_runtime_argv)
PROJECT_ROOT = ARGS.project_root.expanduser().resolve()
DATASET_ROOT = ARGS.dataset_root.expanduser().resolve()
MAP_ROOT = ARGS.map_root.expanduser().resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs_python" / ARGS.output_name
BATCH_DIR = OUTPUT_DIR / "batches"


def initialize_runtime_paths(*, validate_paths: bool = True) -> argparse.Namespace:
    """Validate configured input paths and create output directories."""
    if validate_paths:
        if not PROJECT_ROOT.exists():
            raise FileNotFoundError(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")
        if not DATASET_ROOT.exists():
            raise FileNotFoundError(
                f"nuPlan dataset root does not exist: {DATASET_ROOT}. "
                "Pass --dataset-root or set NUPLAN_DATASET_ROOT."
            )
        if not MAP_ROOT.exists():
            raise FileNotFoundError(
                f"nuPlan map root does not exist: {MAP_ROOT}. "
                "Pass --map-root or set NUPLAN_MAP_ROOT."
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    return ARGS


def configure_runtime(argv=None, *, validate_paths: bool = True) -> argparse.Namespace:
    """Programmatic runtime configuration helper.

    For command-line execution prefer :mod:`nuplan_predicate_kg.cli`, which
    configures arguments before this module is imported so rule metadata and
    implementation thresholds are guaranteed to be identical.
    """
    global PROJECT_ROOT, DATASET_ROOT, MAP_ROOT, OUTPUT_DIR, BATCH_DIR
    parsed = parse_args(argv)
    ARGS.__dict__.clear()
    ARGS.__dict__.update(vars(parsed))
    PROJECT_ROOT = ARGS.project_root.expanduser().resolve()
    DATASET_ROOT = ARGS.dataset_root.expanduser().resolve()
    MAP_ROOT = ARGS.map_root.expanduser().resolve()
    OUTPUT_DIR = PROJECT_ROOT / "outputs_python" / ARGS.output_name
    BATCH_DIR = OUTPUT_DIR / "batches"
    return initialize_runtime_paths(validate_paths=validate_paths)

# Prefer the runtime bundled with this package. Keep PROJECT_ROOT/src only as
# a fallback for compatibility with older deployments. Because sys.path.insert(0)
# reverses priority when called repeatedly, add the fallback first and the bundled
# runtime last so the bundled v9.5.39 adapter is always resolved first.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_SRC = PACKAGE_ROOT
for source_root in (PROJECT_ROOT, BUNDLED_SRC):
    source_text = str(source_root)
    if source_root.exists():
        # Remove an existing occurrence before reinserting, otherwise an editable
        # install or parent-project PYTHONPATH can still shadow the bundled runtime.
        while source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)

from nuplan_predicate_kg.adapters.nuplan_adapter import build_mini_scenarios, iter_frames
from nuplan_predicate_kg.models import (
    AssertionKind,
    PredicateAssertion,
    PredicateDefinition,
    Provenance,
    RuleDefinition,
    ValueType,
)

try:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    MAP_QUERY_AVAILABLE = True
except Exception:
    # The package remains importable without nuPlan; map support is required only
    # when the extraction pipeline is executed.
    MAP_QUERY_AVAILABLE = False
    MAP_QUERY_AVAILABLE = False

try:
    from shapely.affinity import rotate, translate
    from shapely.geometry import Point, box as shapely_box

    SHAPELY_AVAILABLE = True
except Exception as exc:
    print(f"Shapely unavailable: {exc}")
    SHAPELY_AVAILABLE = False


NP = Namespace("https://w3id.org/nuplan-predicate-kg/")


def stable_id(*parts: object) -> str:
    text = "|".join(map(str, parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def make_predicate(
    predicate_id: str,
    category: str,
    description: str,
    subject_types: list[str],
    object_types: list[str],
    assertion_kind: AssertionKind,
    value_type: ValueType,
    *,
    rule_id: Optional[str] = None,
    units: Optional[str] = None,
    source_fields: Optional[list[str]] = None,
    temporal_scope: str = "instant",
    symmetric: bool = False,
    inverse_of: Optional[str] = None,
) -> PredicateDefinition:
    return PredicateDefinition(
        predicate_id=predicate_id,
        label=predicate_id.split(":")[-1],
        category=category,
        description=description,
        subject_types=subject_types,
        object_types=object_types,
        value_type=value_type,
        assertion_kind=assertion_kind,
        source_fields=source_fields or [],
        rule_id=rule_id,
        units=units,
        temporal_scope=temporal_scope,
        symmetric=symmetric,
        inverse_of=inverse_of,
    )


NATIVE = AssertionKind.NATIVE
DERIVED = AssertionKind.DERIVED

RULES = [
    RuleDefinition(
        rule_id="R-SPEED-MAGNITUDE-001",
        name="Planar speed magnitude",
        description="Compute planar speed from global velocity components.",
        inputs=["velocity_x", "velocity_y"],
        parameters={},
        expression="sqrt(vx^2 + vy^2)",
        implementation="math.hypot",
    ),
    RuleDefinition(
        rule_id="R-ACCELERATION-MAGNITUDE-001",
        name="Planar acceleration magnitude",
        description="Compute acceleration magnitude from acceleration components.",
        inputs=["acceleration_x", "acceleration_y"],
        parameters={},
        expression="sqrt(ax^2 + ay^2)",
        implementation="math.hypot",
    ),
    RuleDefinition(
        rule_id="R-TEMPORAL-DIFFERENCE-001",
        name="Consecutive-observation temporal differences",
        description=(
            "Compute elapsed time, displacement, wrapped heading change, speed "
            "change, and finite-difference acceleration between two consecutive "
            "available observations of the same track. No semantic threshold is applied."
        ),
        inputs=["previous_state", "current_state"],
        parameters={},
        expression="difference(current, previous)",
        implementation="append_temporal_measurements",
    ),
    RuleDefinition(
        rule_id="R-RELATIVE-KINEMATICS-001",
        name="Directed pairwise kinematics",
        description=(
            "Compute relative position and relative velocity in the subject-oriented frame."
        ),
        inputs=["subject_pose", "object_pose", "subject_velocity", "object_velocity"],
        parameters={},
        expression="rotate(object-subject, -subject_heading)",
        implementation="relative_kinematics",
    ),
    RuleDefinition(
        rule_id="R-FOOTPRINT-GEOMETRY-001",
        name="Oriented footprint geometry",
        description="Compute footprint distance and exact polygon overlap.",
        inputs=["subject_pose", "object_pose", "subject_dimensions", "object_dimensions"],
        parameters={},
        expression="polygon_distance_and_intersection",
        implementation="footprint_metrics",
    ),
    RuleDefinition(
        rule_id="R-MAP-MEMBERSHIP-EXACT-001",
        name="Conservative geometric and topological map grounding",
        description=(
            "Represent exact center coverage and positive-area footprint intersection "
            "separately, then select at most one primary lane or lane connector using "
            "explicit overlap and lateral-offset ambiguity margins."
        ),
        inputs=[
            "entity_center", "entity_oriented_footprint", "map_polygon",
            "baseline_path", "lane_graph", "parent_map_object",
        ],
        parameters={"built_in_threshold_profile": thresholds_as_dict()},
        expression=(
            "exact_membership = polygon.covers(center); "
            "footprint_intersection = area(footprint ∩ polygon) > epsilon; "
            "primary = unique_best_geometry_or_ambiguous"
        ),
        implementation="append_map_memberships",
    ),
    RuleDefinition(
        rule_id="R-PAIR-HISTORY-001",
        name="Directed pair observation history",
        description=(
            "Count consecutive observations of a directed pair and preserve the "
            "elapsed observed duration and changes in pair distances."
        ),
        inputs=["previous_pair_state", "current_pair_state"],
        parameters={},
        expression="pair_history_update",
        implementation="append_pairwise_measurements",
    ),
]


PREDICATES = [
    # Structure and scenario
    make_predicate("np:hasTimestamp", "structure", "Observation timestamp in microseconds.", ["Observation"], ["Literal"], NATIVE, ValueType.INTEGER, source_fields=["lidar_pc.timestamp"]),
    make_predicate("np:hasScenarioType", "scenario", "Native nuPlan scenario type.", ["Scenario"], ["Literal"], NATIVE, ValueType.STRING, source_fields=["scenario.scenario_type"]),
    make_predicate("np:hasLogName", "scenario", "Native nuPlan log name.", ["Scenario"], ["Literal"], NATIVE, ValueType.STRING, source_fields=["scenario.log_name"]),
    make_predicate("np:hasEgoState", "structure", "Links an observation to its ego state.", ["Observation"], ["EgoState"], NATIVE, ValueType.ENTITY, source_fields=["ego_state"]),
    make_predicate("np:hasTrackedObject", "structure", "Links an observation to a tracked object.", ["Observation"], ["Agent"], NATIVE, ValueType.ENTITY, source_fields=["tracked_objects"]),
    make_predicate("np:hasTrackToken", "agent", "Stable nuPlan track token.", ["Agent"], ["Literal"], NATIVE, ValueType.STRING, source_fields=["metadata.track_token"]),
    make_predicate("np:hasAgentType", "agent", "Native tracked-object type.", ["Agent"], ["Literal"], NATIVE, ValueType.STRING, source_fields=["tracked_object_type"]),
    # Pose and geometry
    make_predicate("np:hasX", "position_values", "Global geometric-center x coordinate.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="meter", source_fields=["center.x"]),
    make_predicate("np:hasY", "position_values", "Global geometric-center y coordinate.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="meter", source_fields=["center.y"]),
    make_predicate("np:hasHeading", "position_values", "Global geometric-center heading.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="radian", source_fields=["center.heading"]),
    make_predicate("np:hasRearAxleX", "position_values", "Native ego rear-axle x coordinate.", ["EgoState"], ["Literal"], NATIVE, ValueType.FLOAT, units="meter", source_fields=["rear_axle.x"]),
    make_predicate("np:hasRearAxleY", "position_values", "Native ego rear-axle y coordinate.", ["EgoState"], ["Literal"], NATIVE, ValueType.FLOAT, units="meter", source_fields=["rear_axle.y"]),
    make_predicate("np:hasLength", "geometry", "Object length.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="meter", source_fields=["oriented_box.length"]),
    make_predicate("np:hasWidth", "geometry", "Object width.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="meter", source_fields=["oriented_box.width"]),
    make_predicate("np:hasHeight", "geometry", "Object height when available.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="meter", source_fields=["oriented_box.height"]),
    make_predicate("np:hasArea", "geometry", "Rectangular footprint area length times width.", ["EgoState", "Agent"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-FOOTPRINT-GEOMETRY-001", units="square_meter"),
    # Motion measurements
    make_predicate("np:hasVelocity", "motion", "Planar global velocity vector [vx, vy].", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.VECTOR2, units="m/s", source_fields=["velocity.x", "velocity.y"]),
    make_predicate("np:hasVelocityX", "motion", "Global x velocity component.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="m/s", source_fields=["velocity.x"]),
    make_predicate("np:hasVelocityY", "motion", "Global y velocity component.", ["EgoState", "Agent"], ["Literal"], NATIVE, ValueType.FLOAT, units="m/s", source_fields=["velocity.y"]),
    make_predicate("np:hasSpeed", "motion", "Planar speed magnitude.", ["EgoState", "Agent"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-SPEED-MAGNITUDE-001", units="m/s"),
    make_predicate("np:hasAccelerationX", "motion", "Native ego x acceleration component.", ["EgoState"], ["Literal"], NATIVE, ValueType.FLOAT, units="m/s^2", source_fields=["rear_axle_acceleration_2d.x"]),
    make_predicate("np:hasAccelerationY", "motion", "Native ego y acceleration component.", ["EgoState"], ["Literal"], NATIVE, ValueType.FLOAT, units="m/s^2", source_fields=["rear_axle_acceleration_2d.y"]),
    make_predicate("np:hasAcceleration", "motion", "Planar acceleration magnitude.", ["EgoState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-ACCELERATION-MAGNITUDE-001", units="m/s^2"),
    # Temporal measurements
    make_predicate("np:precedes", "temporal", "Links the previous available temporal instance of the same track to the current instance.", ["SpatialEntity"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-TEMPORAL-DIFFERENCE-001", temporal_scope="interval"),
    make_predicate("np:hasDeltaTimeFromPrevious", "temporal", "Elapsed time from the previous available observation of the same track.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="second", temporal_scope="interval"),
    make_predicate("np:hasDisplacementFromPrevious", "temporal", "Center displacement from the previous available observation.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="meter", temporal_scope="interval"),
    make_predicate("np:hasHeadingChangeFromPrevious", "temporal", "Signed wrapped heading change from the previous available observation.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="radian", temporal_scope="interval"),
    make_predicate("np:hasSpeedChangeFromPrevious", "temporal", "Speed change from the previous available observation.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="m/s", temporal_scope="interval"),
    make_predicate("np:hasEstimatedAcceleration", "temporal", "Finite-difference acceleration from speed change divided by elapsed time.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="m/s^2", temporal_scope="interval"),
    make_predicate("np:hasObservedFrameCount", "temporal", "Number of available observations of this track within the current scenario processing window.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.INTEGER, rule_id="R-TEMPORAL-DIFFERENCE-001", temporal_scope="interval"),
    make_predicate("np:hasObservedDuration", "temporal", "Elapsed time since the first available observation of this track in the current scenario processing window.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="second", temporal_scope="interval"),
    # Exact map and control facts
    make_predicate("np:inLane", "map", "Entity geometric center is covered by an unbuffered lane polygon.", ["SpatialEntity"], ["Lane"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:inLaneConnector", "map", "Entity geometric center is covered by an unbuffered lane-connector polygon.", ["SpatialEntity"], ["LaneConnector"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:intersectsLane", "map", "Entity footprint has positive-area intersection with an unbuffered lane polygon.", ["SpatialEntity"], ["Lane"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:intersectsLaneConnector", "map", "Entity footprint has positive-area intersection with an unbuffered lane-connector polygon.", ["SpatialEntity"], ["LaneConnector"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:onExpertRoute", "route", "Roadblock belongs to the native expert-route roadblock sequence.", ["Roadblock"], ["Route"], NATIVE, ValueType.ENTITY, source_fields=["route_roadblock_ids"]),
    # Pair structure and continuous measurements
    make_predicate("np:hasPairwiseState", "pairwise", "Links an entity to a directed pairwise-state node.", ["SpatialEntity"], ["PairwiseState"], DERIVED, ValueType.ENTITY, rule_id="R-RELATIVE-KINEMATICS-001"),
    make_predicate("np:pairSubject", "pairwise", "Directed pair subject.", ["PairwiseState"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-RELATIVE-KINEMATICS-001"),
    make_predicate("np:pairObject", "pairwise", "Directed pair object.", ["PairwiseState"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-RELATIVE-KINEMATICS-001"),
    make_predicate("np:hasCenterDistanceTo", "geometry", "Center-to-center planar distance.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="meter"),
    make_predicate("np:hasFreeSpaceDistanceTo", "geometry", "Minimum distance between oriented footprints.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-FOOTPRINT-GEOMETRY-001", units="meter"),
    make_predicate("np:hasLongitudinalDistanceTo", "geometry", "Object longitudinal displacement in the subject frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="meter"),
    make_predicate("np:hasLateralDistanceTo", "geometry", "Object lateral displacement in the subject frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="meter"),
    make_predicate("np:hasBearingTo", "geometry", "Object bearing in the subject frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="radian"),
    make_predicate("np:hasHeadingDifferenceTo", "geometry", "Absolute wrapped heading difference.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="radian"),
    make_predicate("np:hasRelativeSpeedTo", "motion", "Relative velocity magnitude.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="m/s"),
    make_predicate("np:hasLongitudinalRelativeSpeedTo", "motion", "Relative longitudinal velocity in the subject frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="m/s"),
    make_predicate("np:hasLateralRelativeSpeedTo", "motion", "Relative lateral velocity in the subject frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="m/s"),
    make_predicate("np:hasClosingSpeedTo", "motion", "Signed radial closing speed; positive means decreasing center distance.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="m/s"),
    make_predicate("np:hasVelocityTowardTarget", "motion", "Subject velocity component along the direction to the object.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="m/s"),
    make_predicate("np:hasSubjectForwardSpeed", "motion", "Subject velocity component along its heading.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-RELATIVE-KINEMATICS-001", units="m/s"),
    make_predicate("np:overlapping", "spatial", "Positive-area overlap of the two oriented footprints above the built-in numerical epsilon.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-FOOTPRINT-GEOMETRY-001"),
    make_predicate("np:inSameLaneAs", "map", "Subject and object have the same unambiguous primary lane; travel-direction agreement is not required.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:sharesIntersectionWith", "map", "Subject and object share an independently established geometric or topological intersection context.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasPairObservedFrameCount", "temporal", "Number of consecutive available observations of this directed pair.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.INTEGER, rule_id="R-PAIR-HISTORY-001", temporal_scope="interval"),
    make_predicate("np:hasPairObservedDuration", "temporal", "Elapsed duration of consecutive available observations of this directed pair.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-PAIR-HISTORY-001", units="second", temporal_scope="interval"),
    make_predicate("np:hasCenterDistanceChangeFromPrevious", "temporal", "Change in directed-pair center distance from the preceding sampled frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-PAIR-HISTORY-001", units="meter", temporal_scope="interval"),
    make_predicate("np:hasFreeSpaceDistanceChangeFromPrevious", "temporal", "Change in directed-pair footprint clearance from the preceding sampled frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-PAIR-HISTORY-001", units="meter", temporal_scope="interval"),
]


RULES.extend([
    RuleDefinition(
        rule_id="R-SEMANTIC-THRESHOLD-001",
        name="Configurable semantic classification",
        description="Create positive-only semantic labels from stored measurements using the built-in v6 route-, map-, and relevance-aware threshold profile; CLI overrides remain optional.",
        inputs=["stored_measurements"],
        parameters={"built_in_profile": "route_map_relevance_v6", "cli_override_optional": True},
        expression="predicate(measurements, thresholds)",
        implementation="derive_current_semantics",
    ),
    RuleDefinition(
        rule_id="R-FOLLOW-HIERARCHICAL-001",
        name="Hierarchical nearest-leader following",
        description=(
            "Derive only np:follows from unique directed map topology, path-projected "
            "oriented footprints, nearest-leader selection over all observed motor "
            "vehicles, moving/queue envelopes, and temporal persistence."
        ),
        inputs=[
            "primary_map_membership", "map_topology", "oriented_footprints",
            "effective_travel_direction", "velocity", "scenario_history",
        ],
        parameters={
            "maximum_follow_headway_s": ARGS.maximum_follow_headway_s,
            "minimum_follow_duration_s": ARGS.minimum_follow_duration_s,
            "maximum_follow_distance_m": ARGS.maximum_follow_distance_m,
            "maximum_queue_follow_gap_m": ARGS.maximum_queue_follow_gap_m,
            "follow_max_path_hops": ARGS.follow_max_path_hops,
        },
        expression=(
            "unique_forward_path AND nearest_forward_motor_vehicle AND "
            "(moving_headway OR queue_gap) continuously"
        ),
        implementation="derive_following_predicates",
    ),
    RuleDefinition(
        rule_id="R-LANE-CHANGE-INTERACTION-003",
        name="Complete vehicle-to-lane change interaction",
        description=(
            "Recall-first confirmation of each lane change from a selected primary "
            "lane transition to a laterally adjacent target. One observed source "
            "frame is sufficient, target persistence rejects A-B-A flicker, local "
            "adjacency tolerates offset lane tokens, track gaps are retained, and "
            "a narrow left-censored recovery handles tracks starting mid-change. "
            "Every transition is emitted independently, including outward and "
            "return changes of a complete overtake."
        ),
        inputs=[
            "vehicle_track_history", "primary_lane_membership",
            "lane_baselines", "lane_adjacency",
        ],
        parameters={
            "source_primary_persistence_frames": ARGS.lane_change_source_stable_frames,
            "target_primary_persistence_frames": ARGS.lane_change_target_stable_frames,
            "maximum_unknown_frames": ARGS.lane_change_max_transition_unknown_frames,
            "onset_search_s": ARGS.lane_change_onset_search_s,
        },
        expression=(
            "observed_primary_lane(source) -> persistent_primary_lane(target) "
            "AND locally_laterally_adjacent(source,target); backdate onset from lateral motion"
        ),
        implementation="derive_lane_change_predicates",
    ),
    RuleDefinition(
        rule_id="R-MERGE-INTERACTION-001",
        name="Lane-change-anchored vehicle merge interaction",
        description=(
            "Derive mergesInFrontOf / mergesBehind only from an already decoded "
            "lateral Lane/LaneConnector corridor transfer. The other vehicle must "
            "pre-exist on the target directed stream, remain on that target path, "
            "and have a stable post-merge longitudinal order. The nearest target-"
            "stream neighbor behind and ahead are selected independently."
        ),
        inputs=[
            "decoded_lane_change_event", "lane_and_connector_topology",
            "oriented_footprints", "effective_travel_direction",
            "target_stream_vehicle_history",
        ],
        parameters={
            "target_preexistence_window_s": ARGS.merge_target_preexistence_window_s,
            "order_confirmation_window_s": ARGS.merge_order_confirmation_window_s,
            "order_stable_frames": ARGS.merge_order_stable_frames,
            "minimum_preexisting_target_frames": ARGS.merge_minimum_preexisting_target_frames,
            "minimum_path_overlap_ratio": ARGS.merge_minimum_path_overlap_ratio,
            "same_flow_threshold_rad": ARGS.merge_same_flow_threshold_rad,
            "maximum_neighbor_gap_m": ARGS.merge_maximum_neighbor_gap_m,
            "maximum_headway_s": ARGS.merge_maximum_headway_s,
            "max_target_path_hops": ARGS.merge_max_target_path_hops,
        },
        expression=(
            "changesLane(S,targetPath) AND O_preexists_on(targetPath) AND "
            "stable_post_merge_order(S,O) AND nearest_target_stream_neighbor"
        ),
        implementation="derive_merging_predicates",
    ),
    RuleDefinition(
        rule_id="R-CROSSES-IN-FRONT-001",
        name="Observed dynamic-road-user crossing order",
        description=(
            "Derive crossesInFrontOf(S,O) frame-locally with two finite 10 m forward "
            "segments starting at the current S/O geometric centers. The exact segments "
            "drawn in the video must intersect, the crossing angle must be within the "
            "configured range, parked/map-irrelevant actors are rejected, and S must be "
            "estimated to reach the common conflict point before O."
        ),
        inputs=[
            "dynamic_road_user_track_history", "geometric_centers",
            "headings_and_velocities", "object_dimensions", "scenario_history",
            "semantic_map_drivable_crosswalk_intersection_carpark_context",
        ],
        parameters={
            "minimum_observations": ARGS.crosses_in_front_min_observations,
            "minimum_track_displacement_m": ARGS.crosses_in_front_min_track_displacement_m,
            "minimum_simultaneous_observation_s": ARGS.crosses_in_front_min_simultaneous_observation_s,
            "maximum_pair_distance_m": ARGS.crosses_in_front_max_pair_distance_m,
            "forward_ray_segment_length_m": 10.0,
            "minimum_crossing_angle_deg": ARGS.crosses_in_front_min_crossing_angle_deg,
            "maximum_crossing_angle_deg": ARGS.crosses_in_front_max_crossing_angle_deg,
            "minimum_subject_crossing_speed_mps": ARGS.crosses_in_front_min_subject_crossing_speed_mps,
            "maximum_arrival_time_difference_s": ARGS.crosses_in_front_max_arrival_time_difference_s,
            "order_margin_s": ARGS.crosses_in_front_order_margin_s,
            "map_query_radius_m": ARGS.crosses_in_front_map_query_radius_m,
            "maximum_road_distance_m": ARGS.crosses_in_front_max_road_distance_m,
            "crosswalk_near_distance_m": ARGS.crosses_in_front_crosswalk_near_distance_m,
            "bicycle_road_distance_m": ARGS.crosses_in_front_bicycle_road_distance_m,
            "parked_max_speed_mps": ARGS.crosses_in_front_parked_max_speed_mps,
            "parked_min_duration_s": ARGS.crosses_in_front_parked_min_duration_s,
            "parked_max_displacement_m": ARGS.crosses_in_front_parked_max_displacement_m,
            "parked_max_lane_distance_m": ARGS.crosses_in_front_parked_max_lane_distance_m,
        },
        expression=(
            "dynamic(S) AND dynamic(O) AND not_parked(S,O) AND "
            "intersect(forward_segment_10m(S),forward_segment_10m(O)) AND "
            "clear_crossing_angle AND ETA(S,C)+order_margin<=ETA(O,C)"
        ),
        implementation="derive_crosses_in_front_predicates",
    ),
    RuleDefinition(
        rule_id="R-INTERACTION-YIELD-001",
        name="Retrospectively verified behavioral yielding",
        description=(
            "Derive yieldsTo(S,O) only when S makes a measurable motion concession "
            "during a verified crossing-conflict interaction. S must remain outside the "
            "conflict while O clears first, have prior motion evidence, show a significant "
            "speed reduction, and subsequently proceed. Ordinary RED-signal stopping is "
            "conservatively excluded. Merge predicates are explicitly not used."
        ),
        inputs=[
            "verified_crossesInFrontOf_event",
            "subject_motion_history", "object_motion_history",
            "crossing_conflict_geometry", "traffic_light_status",
            "post_event_subject_motion",
        ],
        parameters={
            "pre_event_window_s": ARGS.yield_pre_event_window_s,
            "post_event_window_s": ARGS.yield_post_event_window_s,
            "maximum_initial_eta_difference_s": ARGS.yield_max_initial_eta_difference_s,
            "minimum_baseline_speed_mps": ARGS.yield_min_baseline_speed_mps,
            "minimum_speed_drop_mps": ARGS.yield_min_speed_drop_mps,
            "slow_speed_mps": ARGS.yield_slow_speed_mps,
            "maximum_subject_conflict_distance_m": ARGS.yield_max_subject_conflict_distance_m,
            "conflict_clearance_buffer_m": ARGS.yield_conflict_clearance_buffer_m,
            "minimum_proceed_speed_mps": ARGS.yield_min_proceed_speed_mps,
            "minimum_proceed_displacement_m": ARGS.yield_min_proceed_displacement_m,
        },
        expression=(
            "verified_crossesInFrontOf(O,S) AND prior_temporal_competition AND "
            "motion_concession(S) AND red_signal_not_explanatory(S) AND "
            "O_proceeds_first AND S_proceeds_after(O)"
        ),
        implementation="derive_yield_predicates",
    ),
    RuleDefinition(
        rule_id="R-OVERTAKE-NOTEBOOK-CASE1-006",
        name="Notebook-faithful fully observed Case 1 overtaking",
        description=(
            "Use only the fully observed Case 1 strategy from the supplied v1.0.9 "
            "notebook: first perform a high-recall same-lane-start trajectory scan, "
            "then independently reconstruct each candidate with footprint-aware map "
            "matching and require behind, departure to an adjacent passing lane, "
            "side-by-side overlap, order reversal, full clearance, object lane "
            "preservation, and stable return to the original lane."
        ),
        inputs=[
            "directed_vehicle_pair_trajectories", "oriented_footprints",
            "displacement_based_forward_speed", "lane_and_connector_polygons",
            "directed_lane_graph", "scenario_history",
        ],
        parameters={
            "overtake_pair_query_radius_m": ARGS.overtake_pair_query_radius_m,
            "overtake_min_common_frames": ARGS.overtake_min_common_frames,
            "overtake_search_padding_s": ARGS.overtake_search_padding_s,
            "overtake_max_frame_gap_s": ARGS.overtake_max_frame_gap_s,
            "minimum_overtake_initial_behind_gap_m": ARGS.minimum_overtake_initial_behind_gap_m,
            "minimum_overtake_side_by_side_lateral_m": ARGS.minimum_overtake_side_by_side_lateral_m,
            "minimum_overtake_relative_speed_mps": ARGS.minimum_overtake_relative_speed_mps,
            "minimum_overtake_subject_speed_mps": ARGS.minimum_overtake_subject_speed_mps,
            "minimum_overtaken_vehicle_speed_mps": ARGS.minimum_overtaken_vehicle_speed_mps,
            "minimum_overtake_clearance_m": ARGS.minimum_overtake_clearance_m,
            "overtake_same_flow_threshold_rad": ARGS.overtake_same_flow_threshold_rad,
            "minimum_overtake_same_flow_fraction": ARGS.minimum_overtake_same_flow_fraction,
            "overtake_lane_stability_frames": ARGS.overtake_lane_stability_frames,
            "minimum_overtake_completion_duration_s": ARGS.minimum_overtake_completion_duration_s,
            "overtake_lane_query_radius_m": ARGS.overtake_lane_query_radius_m,
            "overtake_lane_min_overlap_ratio": ARGS.overtake_lane_min_overlap_ratio,
            "overtake_lane_ambiguity_margin": ARGS.overtake_lane_ambiguity_margin,
            "overtake_max_lane_path_hops": ARGS.overtake_max_lane_path_hops,
            "maximum_overtake_duration_s": ARGS.maximum_overtake_duration_s,
            "minimum_maneuver_history_frames": ARGS.minimum_maneuver_history_frames,
            "minimum_maneuver_history_s": ARGS.minimum_maneuver_history_s,
        },
        expression=(
            "broad_candidate_scan(case1_same_lane_start) THEN strict_map_recheck: "
            "same_lane_behind -> adjacent_passing_lane -> side_by_side -> "
            "order_reversal -> full_clearance -> stable_return_to_original_lane"
        ),
        implementation="derive_overtaking_predicates",
    ),
    RuleDefinition(
        rule_id="R-CONFLICT-RISK-CV-002",
        name="Current-state short-horizon conflict risk",
        description=(
            "Predict relative motion from the current positions and velocities only. "
            "First require an interaction-relevant candidate from lane/path connectivity, "
            "a tight same-flow corridor, vulnerable-road-user proximity, or crossing/opposite "
            "motion. Then require positive radial closing speed, a meaningful reduction "
            "in oriented-footprint clearance within the short horizon, and the UN R157 "
            "safety gate reported in the paper appendix."
        ),
        inputs=[
            "current_subject_pose", "current_object_pose",
            "current_subject_velocity", "current_object_velocity",
            "oriented_footprint_dimensions",
        ],
        parameters={
            "prediction_horizon_s": ARGS.conflict_risk_horizon_s,
            "conflict_clearance_m": ARGS.conflict_risk_clearance_m,
            "minimum_closing_speed_mps": ARGS.conflict_risk_min_closing_speed_mps,
            "minimum_clearance_reduction_m": ARGS.conflict_risk_min_clearance_reduction_m,
            "cross_traffic_max_center_distance_m": ARGS.conflict_risk_cross_traffic_max_center_distance_m,
            "vru_max_center_distance_m": ARGS.conflict_risk_vru_max_center_distance_m,
            "same_flow_corridor_lateral_m": ARGS.conflict_risk_same_flow_corridor_lateral_m,
            "same_flow_corridor_center_m": ARGS.conflict_risk_same_flow_corridor_center_m,
            "crossing_angle_deg": ARGS.conflict_risk_crossing_angle_deg,
            "optimization_iterations": ARGS.conflict_risk_optimization_iterations,
            "imminent_braking_demand_mps2": ARGS.conflict_risk_imminent_braking_demand_mps2,
            "lane_change_max_target_decel_mps2": ARGS.conflict_risk_lane_change_max_target_decel_mps2,
            "lane_change_min_gap_time_s": ARGS.conflict_risk_lane_change_min_gap_time_s,
        },
        expression=(
            "dynamic_pair AND interaction_relevant_candidate AND closing_speed>=v_close AND "
            "min_{0<t<=H} footprint_clearance_cv(t)<=d_risk AND "
            "current_clearance-min_clearance>=delta_d_min AND R_safety"
        ),
        implementation="append_pair_risk_predicates",
    ),
    RuleDefinition(
        rule_id="R-FUTURE-OBSERVATION-001",
        name="Observed future trajectory analysis",
        description="Use later observed states of the same tracks to compute future links, minimum future distance, and observed footprint-intersection TTC.",
        inputs=["scenario_track_history", "future_horizon_s"],
        parameters={"future_horizon_s": ARGS.future_horizon_s},
        expression="scan later observed footprints within horizon",
        implementation="derive_future_predicates",
    ),
    RuleDefinition(
        rule_id="R-MANEUVER-TOPOLOGY-001",
        name="Temporal maneuver and map-topology analysis",
        description="Use lane and lane-connector membership transitions, map adjacency, and accumulated heading change to infer maneuvers.",
        inputs=["track_lane_history", "map_topology", "heading_history"],
        parameters={"turn_angle_threshold_rad": ARGS.turn_angle_threshold_rad},
        expression="classify stable map-membership transitions",
        implementation="derive_maneuver_predicates",
    ),
    RuleDefinition(
        rule_id="R-TRAFFIC-LIGHT-CONTROLS-001",
        name="Traffic signal controls movement",
        description=(
            "Ground a dataset-native traffic-control association into the canonical "
            "TrafficSignal -> ControlledMovement relation. In nuPlan, the native "
            "controlled movement is a lane connector referenced by traffic_light_status."
        ),
        inputs=["traffic_light_status.lane_connector_id", "lane_connector_map_object"],
        parameters={},
        expression="native traffic-control association -> canonical controls(signal, movement)",
        implementation="derive_traffic_control_at_snapshot",
    ),
    RuleDefinition(
        rule_id="R-TRAFFIC-LIGHT-STATE-001",
        name="Traffic signal canonical state",
        description=(
            "Map the dataset-native current traffic-light status onto the canonical "
            "SignalState vocabulary RED, YELLOW, GREEN, or UNKNOWN while preserving "
            "the stable TrafficSignal identity."
        ),
        inputs=["traffic_light_status.lane_connector_id", "traffic_light_status.status"],
        parameters={},
        expression="native signal status -> canonical hasSignalState(signal, state)",
        implementation="derive_traffic_light_predicates_at_snapshot",
    ),
    RuleDefinition(
        rule_id="R-TRAFFIC-LIGHT-RELEVANCE-001",
        name="Relevant traffic signal for agent",
        description=(
            "Link a road user to the traffic signal that deterministically applies to its "
            "current controlled movement. The rule uses direct controlled-connector "
            "occupancy or a unique signal-controlled immediate successor and abstains "
            "when multiple controlled movements are possible."
        ),
        inputs=[
            "agent_map_membership",
            "lane_topology.outgoing_edges",
            "traffic_light_status.lane_connector_id",
        ],
        parameters={},
        expression=(
            "occupies controlled movement OR unique controlled immediate successor "
            "-> isRelevantSignal(agent, signal)"
        ),
        implementation="derive_traffic_light_predicates_at_snapshot",
    ),
    RuleDefinition(
        rule_id="R-GEOMETRIC-VISIBILITY-001",
        name="Geometric ego visibility",
        description="Use ego field of view and line-segment occlusion by agent footprints to derive geometric visibility.",
        inputs=["ego_pose", "target_pose", "agent_footprints"],
        parameters={"ego_fov_deg": ARGS.ego_fov_deg, "ego_fov_range_m": ARGS.ego_fov_range_m},
        expression="inside FOV and unoccluded line of sight",
        implementation="derive_visibility_predicates",
    ),
    RuleDefinition(
        rule_id="R-INTENT-INTERACTION-001",
        name="Rule-based interaction intent",
        description="Infer yielding, waiting, gap creation, and gap competition from future conflict, temporal motion, and map context.",
        inputs=["future_pair_history", "maneuver_history", "motion_history"],
        parameters={"minimum_intent_duration_s": ARGS.minimum_intent_duration_s},
        expression="explicit behavioral rules",
        implementation="derive_intent_predicates",
    ),
])


PREDICATES.extend([
    # Threshold-derived current semantics
    make_predicate("np:isStopped", "motion_state", "Speed is at or below the configured stopped threshold.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:isSlow", "motion_state", "Speed is between the configured stopped and moving thresholds.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:isMoving", "motion_state", "Speed is above the configured moving threshold.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:isAccelerating", "motion_state", "Estimated signed acceleration exceeds the configured positive threshold.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001", temporal_scope="interval"),
    make_predicate("np:isDecelerating", "motion_state", "Estimated signed acceleration is below the configured negative threshold.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001", temporal_scope="interval"),
    make_predicate("np:near", "spatial", "Exclusive near class: footprint clearance is above very-near and at or below the built-in near threshold.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:veryNear", "spatial", "Exclusive very-near class: positive footprint clearance is at or below the built-in very-near threshold.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:inFrontOf", "spatial", "The pair object is ahead of the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:behind", "spatial", "The pair object is behind the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:leftOf", "spatial", "The pair object is primarily left of the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:rightOf", "spatial", "The pair object is primarily right of the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:frontLeftOf", "spatial", "The pair object is ahead and left of the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:frontRightOf", "spatial", "The pair object is ahead and right of the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:rearLeftOf", "spatial", "The pair object is behind and left of the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:rearRightOf", "spatial", "The pair object is behind and right of the pair subject in the subject body-heading frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:alignedWith", "heading", "Heading difference is below the configured alignment threshold.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate("np:oppositeDirectionTo", "heading", "Heading difference is close to pi.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-SEMANTIC-THRESHOLD-001"),
    make_predicate(
        "np:follows",
        "interaction",
        (
            "Subject persistently follows its unique nearest observed motor-vehicle "
            "leader on the same unambiguous directed lane/connector path, using a "
            "path-projected bumper gap and moving-headway or queue-follow envelope."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-FOLLOW-HIERARCHICAL-001",
        temporal_scope="interval",
    ),
    make_predicate(
        "np:queuesBehind",
        "interaction",
        (
            "Queue-specific longitudinal following: the subject persistently follows "
            "its unique nearest valid forward leader while subject path speed is at "
            "most 2.0 m/s, leader path speed is at most 4.0 m/s, and bumper gap is "
            "positive and at most 12 m."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-FOLLOW-HIERARCHICAL-001",
        temporal_scope="interval",
    ),
    make_predicate(
        "np:changesLane",
        "interaction",
        (
            "A vehicle-like subject transitions between laterally adjacent lane "
            "corridors. An offline topology-constrained decoder combines primary-lane "
            "assignment, footprint overlap, center containment, local geometry, "
            "lateral motion, and temporal continuity. Confirmed, probable, and "
            "boundary-censored changes are retained, and each active row carries an "
            "arrow from the current subject position to the fixed completion point "
            "on the decoded target lane."
        ),
        ["SpatialEntity"],
        ["Lane"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-LANE-CHANGE-INTERACTION-003",
        temporal_scope="interval",
    ),
    make_predicate(
        "np:mergesInFrontOf",
        "interaction",
        (
            "Subject performs a decoded lateral corridor transfer into a Lane/LaneConnector "
            "target stream already occupied by the object and becomes stably ahead of the "
            "nearest target-stream vehicle behind it."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-MERGE-INTERACTION-001",
        temporal_scope="interval",
    ),
    make_predicate(
        "np:mergesBehind",
        "interaction",
        (
            "Subject performs a decoded lateral corridor transfer into a Lane/LaneConnector "
            "target stream already occupied by the object and becomes stably behind the "
            "nearest target-stream vehicle ahead of it."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-MERGE-INTERACTION-001",
        temporal_scope="interval",
    ),
    make_predicate(
        "np:crossesInFrontOf",
        "interaction",
        (
            "Subject S traverses a genuine crossing-conflict region in front of "
            "dynamic road user O while O is still approaching or waiting upstream. "
            "The relation records observed crossing order, not right-of-way or yielding."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-CROSSES-IN-FRONT-001",
        temporal_scope="interval",
    ),
    make_predicate(
        "np:yieldsTo",
        "interaction",
        (
            "Subject gives priority to the object during a verified crossing-conflict "
            "interaction by measurably slowing or stopping, while the object proceeds "
            "first, and the subject subsequently proceeds. Crossing order alone is not "
            "sufficient, and ordinary RED-signal stopping is excluded. Merge predicates "
            "are not inputs to this rule."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-INTERACTION-YIELD-001",
        temporal_scope="interval",
    ),
    make_predicate(
        "np:overtakes",
        "interaction",
        (
            "Subject is performing a strictly verified, fully observed same-direction "
            "overtaking maneuver. Subject and object begin behind/in front on the same "
            "directed lane path; the subject departs to an official adjacent passing lane, "
            "passes with full clearance, and returns stably to the original lane. The "
            "relation is assigned continuously over the full verified maneuver."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-OVERTAKE-NOTEBOOK-CASE1-006",
        temporal_scope="interval",
    ),

    # Risk (v9.5.45: refined conservative current-state conflict risk)
    make_predicate(
        "np:hasConflictRiskWith",
        "risk",
        (
            "Two dynamic road users have a short-horizon conflict risk when they are "
            "interaction-relevant, have sufficient radial closing motion, and constant-"
            "velocity continuation predicts a meaningful reduction of oriented-footprint "
            "clearance into the configured safety envelope. The predicate uses no observed-"
            "future ground truth and does not require an actual collision to occur."
        ),
        ["SpatialEntity"],
        ["SpatialEntity"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-CONFLICT-RISK-CV-002",
        symmetric=True,
    ),

    # Future observed facts
    make_predicate("np:hasFuturePosition", "future_observation", "Links an entity instance to a later observed instance of the same track.", ["SpatialEntity"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-FUTURE-OBSERVATION-001", temporal_scope="future"),
    make_predicate("np:hasFutureFootprint", "future_observation", "Links an entity instance to a later observed footprint instance of the same track.", ["SpatialEntity"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-FUTURE-OBSERVATION-001", temporal_scope="future"),

    # Maneuvers
    make_predicate("np:keepsLane", "maneuver", "Track maintains a stable lane over the temporal window.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:changesLaneLeft", "maneuver", "Track transitions to a left-adjacent lane.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:changesLaneRight", "maneuver", "Track transitions to a right-adjacent lane.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:entersLaneConnector", "maneuver", "Track transitions from a lane into a connected lane connector.", ["SpatialEntity"], ["LaneConnector"], DERIVED, ValueType.ENTITY, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:exitsLaneConnector", "maneuver", "Track transitions from a lane connector into a successor lane.", ["SpatialEntity"], ["Lane"], DERIVED, ValueType.ENTITY, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:merges", "maneuver", "Track enters a converging target lane while another stream occupies that lane.", ["SpatialEntity"], ["Lane"], DERIVED, ValueType.ENTITY, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:turnsLeft", "maneuver", "Accumulated heading change through a connector indicates a left turn.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:turnsRight", "maneuver", "Accumulated heading change through a connector indicates a right turn.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:goesStraight", "maneuver", "Small accumulated heading change through a connector indicates straight movement.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),
    make_predicate("np:makesUTurn", "maneuver", "Accumulated heading change near pi indicates a U-turn.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MANEUVER-TOPOLOGY-001", temporal_scope="interval"),

    # Traffic light (v9.5.38: compact dataset-independent grounding)
    make_predicate(
        "np:controls",
        "traffic_light",
        "A traffic signal controls a canonical movement through the road network.",
        ["TrafficSignal"],
        ["ControlledMovement"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-TRAFFIC-LIGHT-CONTROLS-001",
    ),
    make_predicate(
        "np:hasSignalState",
        "traffic_light",
        "Current canonical state of a traffic signal: RED, YELLOW, GREEN, or UNKNOWN.",
        ["TrafficSignal"],
        ["SignalState"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-TRAFFIC-LIGHT-STATE-001",
    ),
    make_predicate(
        "np:isRelevantSignal",
        "traffic_light",
        "Traffic signal that deterministically applies to the agent's current or uniquely determined immediate movement.",
        ["Agent"],
        ["TrafficSignal"],
        DERIVED,
        ValueType.ENTITY,
        rule_id="R-TRAFFIC-LIGHT-RELEVANCE-001",
    ),

    # Geometric visibility
    make_predicate("np:withinEgoFieldOfView", "visibility", "At least part of the target footprint lies inside the configured geometric ego range and field of view.", ["Agent"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-GEOMETRIC-VISIBILITY-001"),
    make_predicate("np:hasLineOfSightTo", "visibility", "Ego-to-target center segment is not blocked by another modeled agent footprint.", ["EgoState"], ["Agent"], DERIVED, ValueType.ENTITY, rule_id="R-GEOMETRIC-VISIBILITY-001"),
    make_predicate("np:occludedByAgent", "visibility", "Target line of sight is blocked by a closer agent footprint.", ["Agent"], ["Agent"], DERIVED, ValueType.ENTITY, rule_id="R-GEOMETRIC-VISIBILITY-001"),
    make_predicate("np:geometricallyVisibleToEgo", "visibility", "Target footprint is within the geometric ego field of view and has an unblocked modeled line of sight.", ["Agent"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-GEOMETRIC-VISIBILITY-001"),

    # Sensor-level catalogue entries; emitted only when matching sensor data/calibration is available.
    make_predicate("np:projectsIntoCamera", "sensor_visibility", "Target 3D box projects into a configured camera image.", ["Agent"], ["Camera"], DERIVED, ValueType.ENTITY, rule_id="R-GEOMETRIC-VISIBILITY-001"),
    make_predicate("np:visibleInCamera", "sensor_visibility", "Target has valid positive-depth visible projection in a camera.", ["Agent"], ["Camera"], DERIVED, ValueType.ENTITY, rule_id="R-GEOMETRIC-VISIBILITY-001"),
    make_predicate("np:visibleInLidar", "sensor_visibility", "Target is supported by LiDAR observations when point association is available.", ["Agent"], ["Lidar"], DERIVED, ValueType.ENTITY, rule_id="R-GEOMETRIC-VISIBILITY-001"),
    make_predicate("np:partiallyOccluded", "sensor_visibility", "Target projection is only partly visible according to sensor-level evidence.", ["Agent"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-GEOMETRIC-VISIBILITY-001"),

    # Intent
    make_predicate("np:yieldingTo", "intent", "Subject slows or stops before a future conflict while object clears first.", ["SpatialEntity"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-INTENT-INTERACTION-001", temporal_scope="interval"),
    make_predicate("np:waitingFor", "intent", "Subject remains stopped because the object occupies or blocks its relevant future path.", ["SpatialEntity"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-INTENT-INTERACTION-001", temporal_scope="interval"),
    make_predicate("np:creatingGapFor", "intent", "Subject increases an available merge gap and the object later enters it.", ["SpatialEntity"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-INTENT-INTERACTION-001", temporal_scope="interval"),
    make_predicate("np:competingForGapWith", "intent", "Two entities approach the same target gap with similar arrival times and no clear yielding relation.", ["SpatialEntity"], ["SpatialEntity"], DERIVED, ValueType.ENTITY, rule_id="R-INTENT-INTERACTION-001", temporal_scope="interval", symmetric=True),
])


# Additional logic-consistency measurements and conservative semantic facts.
RULES.append(
    RuleDefinition(
        rule_id="R-MAP-MOTION-SPATIAL-001",
        name="Map- and motion-aware relevant spatial relation",
        description=(
            "Select a reliable travel direction from map flow, velocity, and temporal displacement; "
            "then emit one spatial relation only for same, adjacent, predecessor, or successor map context."
        ),
        inputs=["primary_map_match", "map_topology", "velocity", "temporal_displacement", "pair_geometry"],
        parameters={"built_in_threshold_profile": thresholds_as_dict()},
        expression="relevant_spatial_relation(map_context, travel_direction, relative_position)",
        implementation="classify_relevant_spatial_relation",
    )
)


PREDICATES.extend([
    make_predicate("np:touching", "spatial", "Footprints touch within numerical tolerance without positive-area overlap.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-FOOTPRINT-GEOMETRY-001"),
    make_predicate("np:hasIntersectionArea", "geometry", "Positive-area intersection of the two oriented footprints.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-FOOTPRINT-GEOMETRY-001", units="square_meter"),
    make_predicate("np:hasLongitudinalFootprintGap", "geometry", "Signed bumper-to-bumper longitudinal gap in the subject frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-FOOTPRINT-GEOMETRY-001", units="meter"),
    make_predicate("np:hasLateralFootprintGap", "geometry", "Signed lateral footprint gap in the subject frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-FOOTPRINT-GEOMETRY-001", units="meter"),
    make_predicate("np:hasTotalObservedFrameCount", "temporal", "Total available observations of a track in the processing window.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.INTEGER, rule_id="R-TEMPORAL-DIFFERENCE-001", temporal_scope="interval"),
    make_predicate("np:hasTotalObservedSpan", "temporal", "Elapsed span between first and current available observations, including gaps.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="second", temporal_scope="interval"),
    make_predicate("np:hasContinuousObservedFrameCount", "temporal", "Number of observations in the current continuity streak.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.INTEGER, rule_id="R-TEMPORAL-DIFFERENCE-001", temporal_scope="interval"),
    make_predicate("np:hasContinuousObservedDuration", "temporal", "Duration of the current continuity streak.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="second", temporal_scope="interval"),
    make_predicate("np:hasPrimaryLane", "map", "Conservative unambiguous primary lane assignment.", ["SpatialEntity"], ["Lane"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasPrimaryLaneConnector", "map", "Conservative unambiguous primary lane-connector assignment.", ["SpatialEntity"], ["LaneConnector"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasPrimaryMapOverlapRatio", "map", "Entity-footprint overlap ratio with the selected primary lane or lane-connector polygon.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasBaselineProgress", "map", "Nearest projected arc-length progress along the selected baseline path.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MEMBERSHIP-EXACT-001", units="meter"),
    make_predicate("np:hasBaselineLateralOffset", "map", "Signed lateral offset from the selected baseline; positive is left of baseline travel direction.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MEMBERSHIP-EXACT-001", units="meter"),
    make_predicate("np:hasAmbiguousMapMatch", "map", "Multiple map candidates are too similar for a reliable primary assignment.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasAvailableFutureDuration", "future_observation", "Available observed-future duration for both tracks.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-FUTURE-OBSERVATION-001", units="second", temporal_scope="observed_future"),
    make_predicate("np:hasFutureSampleCount", "future_observation", "Number of fixed-step observed-future samples used.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.INTEGER, rule_id="R-FUTURE-OBSERVATION-001", temporal_scope="observed_future"),
    make_predicate("np:hasFutureCoverageRatio", "future_observation", "Fraction of requested fixed-step future samples available for both tracks.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-FUTURE-OBSERVATION-001", temporal_scope="observed_future"),
    make_predicate("np:hasInsufficientFutureEvidence", "future_observation", "Observed-future coverage is insufficient for risk classification.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-FUTURE-OBSERVATION-001", temporal_scope="observed_future"),
])

PREDICATES.extend([
    make_predicate("np:hasVelocityHeading", "motion", "Direction of the global velocity vector when speed is sufficient.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MOTION-SPATIAL-001", units="radian"),
    make_predicate("np:hasDisplacementHeading", "temporal", "Direction from the previous continuous position to the current position when displacement is sufficient.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-TEMPORAL-DIFFERENCE-001", units="radian", temporal_scope="interval"),
    make_predicate("np:hasMapHeading", "map", "Local legal travel direction of the selected primary lane or connector baseline.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MEMBERSHIP-EXACT-001", units="radian"),
    make_predicate("np:hasEffectiveTravelHeading", "motion", "Conservative direction of travel selected from agreeing map, velocity, and displacement evidence.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MOTION-SPATIAL-001", units="radian"),
    make_predicate("np:hasTravelDirectionSource", "motion", "Evidence source used for the effective direction of travel.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.STRING, rule_id="R-MAP-MOTION-SPATIAL-001"),
    make_predicate("np:hasTravelLongitudinalDistanceTo", "geometry", "Object longitudinal displacement in the selected subject travel-direction frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MOTION-SPATIAL-001", units="meter"),
    make_predicate("np:hasTravelLateralDistanceTo", "geometry", "Object lateral displacement in the selected subject travel-direction frame.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MOTION-SPATIAL-001", units="meter"),
    make_predicate("np:hasTravelDirectionDifferenceTo", "motion", "Absolute difference between subject and object effective travel directions.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MOTION-SPATIAL-001", units="radian"),
    make_predicate("np:hasMapProgressDifferenceTo", "map", "Object minus subject progress on the same selected map baseline.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MOTION-SPATIAL-001", units="meter"),
    make_predicate("np:hasSpatialMapRelation", "map", "Topological map context used for spatial relevance: same, adjacent-left/right, successor, predecessor, or unrelated.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.STRING, rule_id="R-MAP-MOTION-SPATIAL-001"),
])

RULES.append(
    RuleDefinition(
        rule_id="R-EGO-RELEVANCE-001",
        name="Route- and interaction-aware agent relevance",
        description=(
            "Retain every tracked entity but create pair nodes only for agents supported "
            "by lane graph, expert route, intersection, traffic control, proximity, "
            "vulnerable-road-user, static-obstacle, or closest-approach evidence."
        ),
        inputs=[
            "expert_route_roadblocks", "primary_map_match", "lane_graph",
            "intersection_context", "traffic_lights", "agent_type",
            "relative_geometry", "relative_velocity", "temporal_motion",
        ],
        parameters={"built_in_relevance_threshold_profile": relevance_thresholds_as_dict()},
        expression="relevant(pair) iff at least one conservative relevance reason is established",
        implementation="evaluate_pair_relevance",
    )
)

PREDICATES.extend([
    make_predicate("np:isRelevantToEgo", "relevance", "Pair was selected as relevant to ego using route, topology, conflict, or proximity evidence.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-EGO-RELEVANCE-001"),
    make_predicate("np:isRelevantPair", "relevance", "Non-ego pair was retained by the same conservative interaction-relevance rules.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.BOOLEAN, rule_id="R-EGO-RELEVANCE-001"),
    make_predicate("np:hasRelevanceScore", "relevance", "Deterministic ranking score used only to limit non-critical ego connections.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-EGO-RELEVANCE-001"),
    make_predicate("np:hasRelevanceReason", "relevance", "One positive reason that made the pair relevant.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.STRING, rule_id="R-EGO-RELEVANCE-001"),
    make_predicate("np:hasPrimaryRelevanceReason", "relevance", "Highest-priority relevance reason for the pair.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.STRING, rule_id="R-EGO-RELEVANCE-001"),
    make_predicate("np:hasRouteRelation", "route", "Relation between subject and object roadblocks in the expert-route sequence.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.STRING, rule_id="R-EGO-RELEVANCE-001"),
    make_predicate("np:hasSignedPathDistanceTo", "map", "Signed along-path distance when a common lane-graph frame is available; positive means object downstream.", ["PairwiseState"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-EGO-RELEVANCE-001", units="meter"),
    make_predicate("np:hasParentRoadblock", "map", "Selected primary lane belongs to this parent roadblock.", ["SpatialEntity"], ["Roadblock"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasParentRoadblockConnector", "map", "Selected primary lane connector belongs to this parent roadblock connector.", ["SpatialEntity"], ["RoadblockConnector"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasRouteRoadblockRank", "route", "Zero-based position of the matched roadblock or roadblock connector in the expert route.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.INTEGER, rule_id="R-EGO-RELEVANCE-001"),
    make_predicate("np:inIntersection", "map", "Entity geometric center is covered by an intersection polygon.", ["SpatialEntity"], ["Intersection"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:intersectsIntersection", "map", "Entity footprint has positive-area intersection with an intersection polygon.", ["SpatialEntity"], ["Intersection"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasPrimaryMapIntersection", "map", "Selected primary lane or lane connector is topologically associated with this intersection.", ["SpatialEntity"], ["Intersection"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:inCrosswalk", "map", "Entity geometric center is covered by a crosswalk polygon.", ["SpatialEntity"], ["Crosswalk"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:intersectsCrosswalk", "map", "Entity footprint has positive-area intersection with a crosswalk polygon.", ["SpatialEntity"], ["Crosswalk"], DERIVED, ValueType.ENTITY, rule_id="R-MAP-MEMBERSHIP-EXACT-001"),
    make_predicate("np:hasMapSpeedLimit", "map", "Native speed limit of the selected lane or lane connector when available.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MEMBERSHIP-EXACT-001", units="m/s"),
    make_predicate("np:hasBaselineCurvature", "map", "Local curvature of the selected baseline at the entity projection.", ["SpatialEntity"], ["Literal"], DERIVED, ValueType.FLOAT, rule_id="R-MAP-MEMBERSHIP-EXACT-001", units="1/meter"),
])

DEFINITIONS_BY_ID = {p.predicate_id: p for p in PREDICATES}




def native_value(pid, subject_id, value, value_type, frame, source_fields):
    return PredicateAssertion(
        assertion_id=stable_id(pid, subject_id, repr(value), frame.timestamp_us),
        predicate_id=pid,
        subject_id=subject_id,
        value=value,
        value_type=value_type,
        valid_time_us=int(frame.timestamp_us),
        assertion_kind=NATIVE,
        provenance=provenance_for(frame, source_fields=source_fields),
    )


def native_relation(pid, subject_id, object_id, frame, source_fields):
    return PredicateAssertion(
        assertion_id=stable_id(pid, subject_id, object_id, frame.timestamp_us),
        predicate_id=pid,
        subject_id=subject_id,
        object_id=object_id,
        value_type=ValueType.ENTITY,
        valid_time_us=int(frame.timestamp_us),
        assertion_kind=NATIVE,
        provenance=provenance_for(frame, source_fields=source_fields),
    )






def wrap_signed(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

def normalize_filter_values(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return {str(value)}
    return {str(item) for item in value}


def scenario_map_name(scenario):
    for attribute in ("map_name", "_map_name"):
        value = getattr(scenario, attribute, None)
        if value is not None:
            return str(value)
    return ""


def filter_scenarios_from_yaml(scenarios, yaml_path, random_seed):
    if yaml_path is None:
        return list(scenarios)
    import random
    import yaml

    with Path(yaml_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    scenario_types = normalize_filter_values(config.get("scenario_types"))
    scenario_tokens = normalize_filter_values(config.get("scenario_tokens"))
    log_names = normalize_filter_values(config.get("log_names"))
    map_names = normalize_filter_values(config.get("map_names"))

    filtered = []
    for scenario in scenarios:
        if scenario_types is not None and str(scenario.scenario_type) not in scenario_types:
            continue
        if scenario_tokens is not None and str(scenario.token) not in scenario_tokens:
            continue
        if log_names is not None and str(scenario.log_name) not in log_names:
            continue
        if map_names is not None and scenario_map_name(scenario) not in map_names:
            continue
        filtered.append(scenario)

    if bool(config.get("shuffle", False)):
        random.Random(int(random_seed)).shuffle(filtered)

    num_per_type = config.get("num_scenarios_per_type")
    if num_per_type is not None:
        counts = Counter()
        selected = []
        for scenario in filtered:
            scenario_type = str(scenario.scenario_type)
            if counts[scenario_type] < int(num_per_type):
                selected.append(scenario)
                counts[scenario_type] += 1
        filtered = selected

    total_limit = config.get("limit_total_scenarios")
    if total_limit is not None:
        if isinstance(total_limit, float):
            keep = max(1, int(len(filtered) * total_limit))
        else:
            keep = int(total_limit)
        filtered = filtered[:keep]
    return filtered


def flatten_assertion(assertion):
    """Flatten an assertion for tabular output.

    Compact mode is the default because run-level metadata (dataset version,
    extractor version, generated_at, etc.) is already stored once in the run
    manifest/config.  We keep the columns required for filtering, Excel/video
    inspection, deterministic traceability, and semantic explanation.
    """
    record = assertion.model_dump()
    provenance = record.pop("provenance")
    evidence = record.pop("evidence")

    # Value is stored as JSON so entity-valued predicates can leave it null
    # while literal predicates retain their exact scalar/list representation.
    record["value_json"] = json.dumps(record.pop("value"), ensure_ascii=False)

    # Promote the provenance fields that are useful per assertion. Everything
    # else is constant/redundant at run level and is kept in run_config.json.
    if getattr(ARGS, "compact_parquet", True):
        compact = {
            "assertion_id": record.get("assertion_id"),
            "predicate_id": record.get("predicate_id"),
            "subject_id": record.get("subject_id"),
            "object_id": record.get("object_id"),
            "value_json": record.get("value_json"),
            "value_type": record.get("value_type"),
            "valid_time_us": record.get("valid_time_us"),
            "end_time_us": record.get("end_time_us"),
            "prov_scenario_token": provenance.get("scenario_token"),
            "prov_timestamp_us": provenance.get("timestamp_us"),
            "prov_rule_id": provenance.get("rule_id"),
        }
        record = compact
    else:
        for key, value in provenance.items():
            record[f"prov_{key}"] = value

    # Keep semantic evidence because it is used by the interaction visualizer
    # and is part of the explainability story.  To save space, evidence for
    # low-level native/measurement predicates is omitted in compact mode.
    keep_evidence = (
        not getattr(ARGS, "compact_parquet", True)
        or assertion.predicate_id in SEMANTIC_PREDICATE_IDS
    )

    if record.get("predicate_id") in {"np:mergesInFrontOf", "np:mergesBehind"} and isinstance(evidence, dict):
        for key in (
            "merge_strategy",
            "merge_event_id",
            "subject_track_token",
            "object_track_token",
            "source_lane_change_event_id",
            "source_lane_change_sequence_index",
            "source_lane_change_side",
            "source_map_entity_id",
            "target_map_entity_id",
            "source_map_kind",
            "target_map_kind",
            "subject_final_order",
            "merge_relation_label",
            "post_merge_bumper_gap_m",
            "post_merge_center_path_distance_m",
            "post_merge_order_stable_frames",
            "post_merge_order_status",
            "post_merge_order_confirmation_frame_index",
            "post_merge_order_confirmation_time_us",
            "other_vehicle_target_map_entity_id",
            "start_time_us",
            "boundary_time_us",
            "completion_time_us",
        ):
            record[key] = evidence.get(key)

    if record.get("predicate_id") == "np:controls" and isinstance(evidence, dict):
        for key in (
            "canonical_signal_id",
            "canonical_controlled_movement_id",
            "dataset_adapter",
            "native_controlled_object_type",
            "native_lane_connector_id",
            "map_name",
            "observed_signal_status",
            "control_relation_independent_of_signal_state",
        ):
            record[key] = evidence.get(key)

    if record.get("predicate_id") == "np:hasSignalState" and isinstance(evidence, dict):
        for key in (
            "canonical_signal_id",
            "canonical_signal_state_id",
            "canonical_signal_state",
            "dataset_adapter",
            "native_lane_connector_id",
            "map_name",
        ):
            record[key] = evidence.get(key)

    if record.get("predicate_id") == "np:isRelevantSignal" and isinstance(evidence, dict):
        for key in (
            "canonical_signal_id",
            "dataset_adapter",
            "native_lane_connector_id",
            "map_name",
            "relevance_basis",
            "agent_type",
            "agent_track_token",
            "agent_primary_map_kind",
            "agent_primary_map_object_id",
            "abstains_when_movement_is_ambiguous",
        ):
            record[key] = evidence.get(key)

    if record.get("predicate_id") == "np:crossesInFrontOf" and isinstance(evidence, dict):
        for key in (
            "crosses_in_front_strategy",
            "interaction_semantics",
            "temporal_assignment",
            "subject_track_token",
            "object_track_token",
            "subject_agent_type",
            "object_agent_type",
            "conflict_x_m",
            "conflict_y_m",
            "crossing_time_us",
            "subject_conflict_entry_time_us",
            "subject_conflict_clearance_time_us",
            "current_timestamp_us",
            "subject_conflict_duration_s",
            "crossing_angle_deg",
            "ray_length_m",
            "subject_ray_distance_m",
            "object_ray_distance_m",
            "subject_speed_mps",
            "subject_eta_s",
            "object_eta_s",
            "eta_gap_s",
            "reference_geometry",
            "map_context_filter",
            "subject_map_filter_reason",
            "object_map_filter_reason",
            "subject_crossing_speed_mps",
            "object_front_distance_m",
            "front_clearance_m",
            "minimum_interval_front_clearance_m",
            "corridor_half_width_m",
            "side_direction",
            "object_speed_mps",
            "object_stationary_mode",
            "object_reaches_conflict_in_observation",
            "object_arrival_time_us",
            "object_arrival_mode",
            "arrival_gap_s",
            "clearance_gap_s",
            "directional_candidate_score",
        ):
            record[key] = evidence.get(key)

    if record.get("predicate_id") == "np:yieldsTo" and isinstance(evidence, dict):
        for key in (
            "yield_strategy",
            "yield_event_id",
            "yield_mode",
            "interaction_semantics",
            "subject_track_token",
            "object_track_token",
            "subject_agent_type",
            "object_agent_type",
            "reaction_start_time_us",
            "reaction_start_frame_index",
            "yield_end_time_us",
            "object_clearance_time_us",
            "subject_proceed_time_us",
            "subject_baseline_speed_mps",
            "subject_minimum_speed_mps",
            "subject_speed_drop_mps",
            "initial_subject_eta_s",
            "initial_object_eta_s",
            "initial_eta_difference_s",
            "conflict_x_m",
            "conflict_y_m",
            "yield_reference_x_m",
            "yield_reference_y_m",
            "subject_remained_outside_conflict",
            "object_proceeded_first",
            "subject_proceeded_after_object",
            "red_signal_exclusion_applied",
            "red_signal_detected_during_yield_response",
            "supporting_predicates",
        ):
            record[key] = evidence.get(key)

    if record.get("predicate_id") == "np:overtakes" and isinstance(evidence, dict):
        for key in (
            "overtake_strategy",
            "overtake_case",
            "overtake_case_number",
            "overtake_case_label",
            "overtake_initial_lane_relation",
            "overtake_observation_mode",
            "overtake_assignment_strategy",
            "overtake_assignment_scope",
            "overtake_active",
            "overtake_completed",
            "overtake_phase",
            "overtake_phase_number",
            "overtake_phase_label",
            "overtake_current_time_us",
            "overtake_current_frame_index",
            "overtake_assignment_start_time_us",
            "overtake_assignment_start_frame_index",
            "overtake_assignment_start_center_longitudinal_m",
            "overtake_detector_start_time_us",
            "overtake_detector_start_frame_index",
            "overtake_total_event_duration_s",
            "relation_family",
            "initial_lane_id",
            "passing_lane_id",
            "final_lane_id",
            "passing_side",
            "start_time_us",
            "passing_lane_observation_start_time_us",
            "side_by_side_time_us",
            "order_reversal_time_us",
            "clearance_time_us",
            "return_time_us",
            "completion_time_us",
            "initial_bumper_gap_m",
            "final_clearance_gap_m",
            "maximum_relative_speed_mps",
            "same_flow_fraction",
            "final_stable_duration_s",
            "intersection_involved",
        ):
            record[key] = evidence.get(key)

    record["evidence_json"] = (
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)
        if keep_evidence and evidence
        else None
    )
    return record


def entity_uri(entity_id):
    return NP[f"entity/{entity_id}"]


def predicate_uri(predicate_id):
    return NP[predicate_id.replace("np:", "")]







# ============================================================================
# MAP-AND-MOTION SPATIAL CONSISTENCY OVERRIDES (v6.0.0)
# ============================================================================
# These definitions intentionally override the earlier compatibility
# implementations.  The catalogue remains backward compatible, while runtime
# emission becomes positive-only, continuity-aware, and mutually exclusive.

SEMANTIC_RULE_IDS = {
    "R-SEMANTIC-THRESHOLD-001",
    "R-FUTURE-OBSERVATION-001",
    "R-MANEUVER-TOPOLOGY-001",
    "R-TRAFFIC-CONTROL-001",
    "R-GEOMETRIC-VISIBILITY-001",
    "R-INTENT-INTERACTION-001",
    "R-FOLLOW-HIERARCHICAL-001",
    "R-LANE-CHANGE-INTERACTION-003",
    "R-MERGE-INTERACTION-001",
    "R-CROSSES-IN-FRONT-001",
    "R-INTERACTION-YIELD-001",
    "R-CONFLICT-RISK-CV-002",
    "R-OVERTAKE-NOTEBOOK-CASE1-006",
}

VEHICLE_TYPE_MARKERS = (
    "VEHICLE", "CAR", "TRUCK", "BUS", "MOTORCYCLE", "BICYCLE", "EGO",
)
STATIC_TYPE_MARKERS = ("STATIC", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT")


def _normalized_agent_type(record_or_value: Any) -> str:
    value = record_or_value.get("agent_type") if isinstance(record_or_value, dict) else record_or_value
    return str(value or "UNKNOWN").upper()


def is_static_record(record: dict[str, Any]) -> bool:
    agent_type = _normalized_agent_type(record)
    return any(marker in agent_type for marker in STATIC_TYPE_MARKERS)


def is_vehicle_like(record: dict[str, Any]) -> bool:
    if str(record.get("track_token")) == "ego":
        return True
    agent_type = _normalized_agent_type(record)
    return not is_static_record(record) and any(marker in agent_type for marker in VEHICLE_TYPE_MARKERS)



DYNAMIC_ROAD_USER_TYPE_MARKERS = (
    "VEHICLE",
    "CAR",
    "TRUCK",
    "BUS",
    "MOTORCYCLE",
    "BICYCLE",
    "CYCLIST",
    "PEDESTRIAN",
    "EGO",
)


def is_dynamic_road_user(record: dict) -> bool:
    """Return True for dynamic traffic participants.

    Includes vehicles, cyclists, bicycles, pedestrians, and ego.
    Static objects such as barriers and traffic cones are excluded.
    """
    if str(record.get("track_token")) == "ego":
        return True

    agent_type = _normalized_agent_type(record)

    return (
        not is_static_record(record)
        and any(
            marker in agent_type
            for marker in DYNAMIC_ROAD_USER_TYPE_MARKERS
        )
    )


def record_semantic_omission(state: Optional[dict[str, Any]], predicate_id: str, reason: str) -> None:
    if state is None:
        return
    state.setdefault("semantic_omissions", Counter())[(str(predicate_id), str(reason))] += 1


def derived_value(
    pid,
    subject_id,
    value,
    value_type,
    frame,
    rule_id,
    evidence,
    end_time_us=None,
    *,
    start_time_us=None,
):
    valid_time_us = int(frame.timestamp_us) if start_time_us is None else int(start_time_us)
    return PredicateAssertion(
        assertion_id=stable_id(pid, subject_id, repr(value), valid_time_us, end_time_us),
        predicate_id=pid,
        subject_id=subject_id,
        value=value,
        value_type=value_type,
        valid_time_us=valid_time_us,
        end_time_us=end_time_us,
        assertion_kind=DERIVED,
        provenance=provenance_for(frame, rule_id=rule_id),
        evidence=evidence,
    )


def derived_relation(
    pid,
    subject_id,
    object_id,
    frame,
    rule_id,
    evidence,
    end_time_us=None,
    *,
    start_time_us=None,
):
    valid_time_us = int(frame.timestamp_us) if start_time_us is None else int(start_time_us)
    return PredicateAssertion(
        assertion_id=stable_id(pid, subject_id, object_id, valid_time_us, end_time_us),
        predicate_id=pid,
        subject_id=subject_id,
        object_id=object_id,
        value_type=ValueType.ENTITY,
        valid_time_us=valid_time_us,
        end_time_us=end_time_us,
        assertion_kind=DERIVED,
        provenance=provenance_for(frame, rule_id=rule_id),
        evidence=evidence,
    )


def append_positive_boolean_semantic(
    assertions,
    pid,
    subject_id,
    condition,
    frame,
    rule_id,
    evidence,
    *,
    state=None,
    omission_reason="condition_not_established",
    start_time_us=None,
    end_time_us=None,
):
    """Append only a positively established Boolean semantic fact."""
    if condition is True:
        assertions.append(derived_value(
            pid, subject_id, True, ValueType.BOOLEAN, frame, rule_id, evidence,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
        ))
        return True
    record_semantic_omission(state, pid, omission_reason)
    return False

SEMANTIC_RELATION_PREDICATES = {
    "np:follows", "np:queuesBehind", "np:changesLane", "np:crossesInFrontOf", "np:yieldsTo", "np:overtakes",
    "np:entersLaneConnector", "np:exitsLaneConnector", "np:merges",
    "np:controls", "np:hasSignalState", "np:isRelevantSignal",
    "np:hasLineOfSightTo", "np:occludedByAgent",
    "np:yieldingTo", "np:waitingFor", "np:creatingGapFor", "np:competingForGapWith",
    "np:hasConflictRiskWith",
}
SEMANTIC_PREDICATE_IDS = set(SEMANTIC_BOOLEAN_PREDICATES) | {"np:isRelevantToEgo", "np:isRelevantPair"} | SEMANTIC_RELATION_PREDICATES | {
    "np:changesLaneLeft", "np:changesLaneRight", "np:keepsLane",
    "np:turnsLeft", "np:turnsRight", "np:goesStraight", "np:makesUTurn",
}


def provenance_for(frame: Any, *, source_fields=None, rule_id=None) -> Provenance:
    return Provenance(
        dataset=str(getattr(frame, "dataset", "nuPlan")),
        dataset_version=str(getattr(frame, "dataset_version", "v1.1")),
        split=str(getattr(frame, "split", "mini")),
        log_name=str(frame.log_name),
        scenario_token=str(frame.scenario_token),
        timestamp_us=int(frame.timestamp_us),
        source_fields=source_fields or [],
        rule_id=rule_id,
        rule_version="1.0.0" if rule_id else None,
        extractor_version="paper-release-1.0.0",
    )


ALL_PREDICATE_CATEGORIES = tuple(sorted({definition.category for definition in PREDICATES}))

CATEGORY_DEPENDENCIES = {
    "structure": set(),
    "scenario": {"structure"},
    "agent": {"structure"},
    "position_values": {"structure", "agent"},
    "geometry": {"structure", "agent", "position_values", "pairwise"},
    "motion": {"structure", "agent", "position_values"},
    "motion_state": {"motion"},
    "temporal": {"structure", "agent", "position_values", "motion", "pairwise"},
    "map": {"structure", "agent", "position_values", "geometry"},
    "route": {"map", "scenario"},
    "traffic_light": {"map"},
    "pairwise": {"structure", "agent"},
    "relevance": {"pairwise", "geometry", "motion", "map", "route"},
    "spatial": {"pairwise", "geometry", "motion", "map", "relevance"},
    "heading": {"pairwise", "motion", "map"},
    "interaction": {"structure", "pairwise", "geometry", "motion", "map", "spatial", "temporal"},
    "risk": {"pairwise", "geometry", "motion", "map"},
    "future_observation": {"pairwise", "geometry", "motion", "temporal"},
    "maneuver": {"map", "motion", "temporal"},
    "visibility": {"geometry", "position_values", "pairwise"},
    "sensor_visibility": {"visibility"},
    "intent": {"interaction", "risk", "future_observation", "temporal"},
}


def _parse_requested_categories() -> set[str]:
    """Categories explicitly requested by the user."""
    raw = str(getattr(ARGS, "categories", "all") or "all").strip().lower()
    if raw in {"all", "*"}:
        return set(ALL_PREDICATE_CATEGORIES)
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = selected - set(ALL_PREDICATE_CATEGORIES)
    if unknown:
        raise ValueError(
            f"Unknown categories: {sorted(unknown)}. Available: {list(ALL_PREDICATE_CATEGORIES)}"
        )
    return selected


def _resolve_category_dependencies(selected: set[str]) -> set[str]:
    """Return transitive dependencies needed to compute selected categories."""
    resolved = set(selected)
    changed = True
    while changed:
        changed = False
        for category in tuple(resolved):
            before = len(resolved)
            resolved.update(CATEGORY_DEPENDENCIES.get(category, set()))
            changed |= len(resolved) != before
    return resolved


# Three deliberately separate concepts:
# - REQUESTED_CATEGORIES: what the user asked for.
# - COMPUTE_CATEGORIES: requested categories plus internal dependencies.
# - WRITE_CATEGORIES: exactly what is persisted. Dependencies are written only
#   when --include-category-dependencies is explicitly enabled.
REQUESTED_CATEGORIES = _parse_requested_categories()
COMPUTE_CATEGORIES = _resolve_category_dependencies(REQUESTED_CATEGORIES)
WRITE_CATEGORIES = (
    set(COMPUTE_CATEGORIES)
    if getattr(ARGS, "include_category_dependencies", False)
    else set(REQUESTED_CATEGORIES)
)

# Backward-compatible internal name. Existing derivation code that checks
# ACTIVE_CATEGORIES should see everything it may need to compute.
ACTIVE_CATEGORIES = COMPUTE_CATEGORIES


def requested_categories() -> set[str]:
    """Backward-compatible accessor for explicitly requested categories."""
    return set(REQUESTED_CATEGORIES)


def assertion_category(assertion) -> str:
    definition = DEFINITIONS_BY_ID.get(assertion.predicate_id)
    return definition.category if definition is not None else "unknown"


def filter_assertions_by_category(assertions):
    """Strict persisted-output filter."""
    if set(WRITE_CATEGORIES) == set(ALL_PREDICATE_CATEGORIES):
        return list(assertions)
    return [a for a in assertions if assertion_category(a) in WRITE_CATEGORIES]


def unexpected_output_categories(assertions) -> set[str]:
    """Categories that must never reach serialized output."""
    return {
        assertion_category(a)
        for a in assertions
        if assertion_category(a) not in WRITE_CATEGORIES
    }


__all__ = [name for name in globals() if not name.startswith("__")]
