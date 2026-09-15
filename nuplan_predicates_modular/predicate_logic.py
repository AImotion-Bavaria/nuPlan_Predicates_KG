#!/usr/bin/env python3
"""Pure logic and built-in thresholds for the nuPlan predicate extractor.

The extractor uses this module as the single source of truth for all numerical
semantic thresholds.  CLI arguments may still override the defaults for an
experiment, but no threshold must be supplied by the user for a normal run.

Spatial relations use a lane-graph and route-aware classifier rather than equal angular
sectors.  Consequently, an object with clear positive longitudinal displacement
can be ``inFrontOf`` or ``frontLeftOf``/``frontRightOf``, but it cannot be
classified as purely ``leftOf`` merely because it is far away laterally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True)
class PredicateThresholds:
    """Built-in semantic configuration used by default by the extractor."""

    # Sampling and temporal continuity
    sample_interval_s: float = 0.5
    maximum_temporal_gap_factor: float = 1.5
    minimum_acceleration_samples: int = 3

    # Motion states
    stopped_speed_mps: float = 0.30
    moving_speed_mps: float = 0.80
    acceleration_threshold_mps2: float = 0.30
    closing_speed_threshold_mps: float = 0.20
    minimum_forward_speed_mps: float = 0.30

    # Exclusive spatial-position classifier
    # Pure left/right is allowed only when longitudinal separation is inside
    # this narrow band around the subject's lateral axis.
    spatial_side_longitudinal_band_m: float = 2.0
    # Minimum reliable longitudinal/lateral displacement.
    spatial_longitudinal_deadband_m: float = 1.0
    spatial_lateral_deadband_m: float = 0.75
    # The front/rear corridor widens with distance:
    # |lateral| <= base + ratio * |longitudinal|.
    spatial_front_lateral_base_m: float = 2.5
    spatial_front_lateral_ratio: float = 0.30
    colocated_epsilon_m: float = 1e-3

    # Map-and-motion-aware spatial relevance. Spatial predicates are omitted
    # when map context or travel direction is unreliable, rather than guessed.
    spatial_relevance_max_distance_m: float = 80.0
    spatial_adjacent_relevance_max_distance_m: float = 45.0
    spatial_diagonal_max_longitudinal_m: float = 20.0
    spatial_motion_only_fallback_max_distance_m: float = 25.0
    spatial_motion_only_max_lateral_m: float = 8.0
    spatial_min_direction_speed_mps: float = 0.75
    spatial_min_displacement_for_direction_m: float = 0.40
    spatial_direction_agreement_threshold_rad: float = 0.45
    spatial_map_motion_tolerance_rad: float = 0.60
    spatial_same_flow_threshold_rad: float = 0.45
    spatial_connected_sign_tolerance_m: float = 1.0
    spatial_require_map_context: bool = True

    # Footprint distance classes
    near_distance_m: float = 5.0
    very_near_distance_m: float = 2.0
    distance_epsilon_m: float = 1e-3
    overlap_area_epsilon_m2: float = 1e-4

    # Heading and following
    alignment_threshold_rad: float = 0.35
    opposite_threshold_rad: float = 0.35
    lateral_same_lane_tolerance_m: float = 2.5
    maximum_follow_headway_s: float = 5.0
    minimum_follow_duration_s: float = 1.0
    maximum_follow_distance_m: float = 80.0
    follow_queue_speed_threshold_mps: float = 2.0
    follow_queue_leader_speed_threshold_mps: float = 4.0
    maximum_queue_follow_gap_m: float = 12.0
    follow_reverse_speed_tolerance_mps: float = 0.30
    follow_leader_tie_tolerance_m: float = 0.50
    follow_max_path_hops: int = 3

    # Notebook-faithful two-stage completed overtake detector (v1.0.9)
    # Legacy phase knobs remain available for CLI compatibility, but the strict
    # detector is governed by the notebook thresholds below.
    minimum_overtake_approach_duration_s: float = 0.5
    minimum_overtake_passing_duration_s: float = 0.5
    minimum_overtake_completion_duration_s: float = 0.5
    minimum_overtake_relative_speed_mps: float = 0.30
    minimum_overtake_subject_speed_mps: float = 2.0
    minimum_overtaken_vehicle_speed_mps: float = 0.50
    minimum_overtake_clearance_m: float = 1.0
    maximum_overtake_entry_gap_m: float = 120.0
    overtake_departure_max_lead_m: float = 2.0
    overtake_same_flow_threshold_rad: float = 0.55
    overtake_lane_stability_frames: int = 2
    maximum_overtake_lane_transition_duration_s: float = 1.5
    maximum_overtake_duration_s: float = 30.0
    overtake_pair_query_radius_m: float = 80.0
    overtake_min_common_frames: int = 5
    overtake_search_padding_s: float = 4.0
    overtake_max_frame_gap_s: float = 1.5
    minimum_overtake_initial_behind_gap_m: float = 1.0
    minimum_overtake_side_by_side_lateral_m: float = 1.25
    minimum_overtake_same_flow_fraction: float = 0.80
    overtake_lane_query_radius_m: float = 8.0
    overtake_lane_min_overlap_ratio: float = 0.10
    overtake_lane_ambiguity_margin: float = 0.05
    overtake_max_lane_path_hops: int = 8
    overtake_positive_deduplication_tolerance_s: float = 2.0

    # Shared temporal-maneuver history gate. A confirmed maneuver must have
    # observed precondition history before its activation/onset frame.
    minimum_maneuver_history_frames: int = 1
    minimum_maneuver_history_s: float = 0.5

    # Vehicle-to-lane interaction: offline topology-constrained lane-state decoding
    lane_change_source_stable_frames: int = 1
    lane_change_target_stable_frames: int = 2
    lane_change_max_transition_unknown_frames: int = 4
    lane_change_minimum_subject_speed_mps: float = 0.10
    lane_change_maximum_transition_duration_s: float = 8.0
    lane_change_minimum_lateral_shift_m: float = 1.0
    lane_change_minimum_stable_lane_overlap_ratio: float = 0.45
    lane_change_require_dual_lane_overlap: bool = False
    lane_change_boundary_evidence_window_frames: int = 3
    lane_change_minimum_dual_lane_overlap_ratio: float = 0.03
    lane_change_onset_search_s: float = 6.0
    lane_change_lateral_onset_threshold_m: float = 0.15
    lane_change_lateral_velocity_onset_threshold_mps: float = 0.15
    lane_change_onset_future_gain_m: float = 0.40
    lane_change_onset_stable_frames: int = 2
    lane_change_target_complete_overlap_ratio: float = 0.60
    lane_change_source_remaining_overlap_ratio: float = 0.40
    lane_change_completion_stable_frames: int = 2
    lane_change_allow_single_frame_high_confidence_completion: bool = True
    lane_change_single_frame_target_overlap_ratio: float = 0.80
    lane_change_max_lane_path_hops: int = 5
    lane_change_adjacency_search_frames: int = 6
    lane_change_allow_geometric_adjacency_fallback: bool = True
    lane_change_geometric_max_polygon_distance_m: float = 1.50
    lane_change_geometric_min_centerline_distance_m: float = 2.0
    lane_change_geometric_max_centerline_distance_m: float = 6.5
    lane_change_geometric_max_direction_difference_rad: float = 0.30
    lane_change_local_min_centerline_distance_m: float = 1.25
    lane_change_local_max_centerline_distance_m: float = 6.5
    lane_change_local_max_direction_difference_rad: float = 0.60
    lane_change_allow_single_frame_primary_target: bool = True
    lane_change_primary_flicker_max_frames: int = 1
    lane_change_single_frame_primary_overlap_ratio: float = 0.70
    lane_change_recover_left_censored_events: bool = True
    lane_change_left_censored_min_source_overlap_ratio: float = 0.05
    lane_change_left_censored_min_overlap_change: float = 0.05
    lane_change_temporal_required_frames: int = 2
    lane_change_temporal_window_frames: int = 3
    lane_change_candidate_center_bonus: float = 0.35
    lane_change_candidate_primary_bonus: float = 0.10
    lane_change_candidate_ambiguity_penalty: float = 0.05
    lane_change_max_grouping_lane_ids: int = 128
    lane_change_decoder_max_unknown_gap_frames: int = 3
    lane_change_decoder_target_stable_frames: int = 2
    lane_change_decoder_emit_probable: bool = True
    lane_change_decoder_minimum_probable_score: float = 3.25
    lane_change_decoder_minimum_confirmed_score: float = 6.0
    lane_change_decoder_recover_left_censored: bool = True

    # Vehicle-to-vehicle merge interactions anchored to decoded lane changes
    merge_target_preexistence_window_s: float = 2.0
    merge_order_confirmation_window_s: float = 2.0
    merge_order_stable_frames: int = 2
    merge_minimum_preexisting_target_frames: int = 1
    merge_minimum_path_overlap_ratio: float = 0.20
    merge_same_flow_threshold_rad: float = 0.55
    merge_maximum_neighbor_gap_m: float = 40.0
    merge_maximum_headway_s: float = 5.0
    merge_max_target_path_hops: int = 6
    merge_allow_terminal_single_frame: bool = True

    # Dynamic-road-user crossing interaction.
    # The active detector uses exact 10 m finite forward segments plus crossing
    # angle and directional ETA order. Legacy corridor thresholds are kept in
    # the configuration object for backwards-compatible CLI/config parsing.
    crosses_in_front_min_observations: int = 3
    crosses_in_front_min_track_displacement_m: float = 0.30
    crosses_in_front_min_simultaneous_observation_s: float = 1.00
    crosses_in_front_max_pair_distance_m: float = 30.0
    crosses_in_front_max_front_distance_m: float = 20.0
    crosses_in_front_min_front_clearance_m: float = -0.25
    crosses_in_front_min_interval_front_clearance_m: float = -0.50
    crosses_in_front_side_buffer_m: float = 0.35
    crosses_in_front_side_search_window_s: float = 3.0
    crosses_in_front_min_crossing_angle_deg: float = 25.0
    crosses_in_front_max_crossing_angle_deg: float = 155.0
    crosses_in_front_min_subject_crossing_speed_mps: float = 0.30
    crosses_in_front_order_margin_s: float = 0.25
    crosses_in_front_max_arrival_time_difference_s: float = 6.0
    crosses_in_front_conflict_footprint_buffer_m: float = 0.35
    crosses_in_front_max_stationary_object_speed_mps: float = 1.0
    crosses_in_front_max_stationary_object_front_distance_m: float = 12.0

    # Map-context filters for crossesInFrontOf. These are intentionally local
    # to this predicate and do not change map/spatial predicates.
    crosses_in_front_map_query_radius_m: float = 12.0
    crosses_in_front_max_road_distance_m: float = 6.0
    crosses_in_front_crosswalk_near_distance_m: float = 5.0
    crosses_in_front_bicycle_road_distance_m: float = 3.0
    crosses_in_front_parked_max_speed_mps: float = 0.5
    crosses_in_front_parked_min_duration_s: float = 2.0
    crosses_in_front_parked_max_displacement_m: float = 1.0
    crosses_in_front_parked_max_lane_distance_m: float = 1.5

    crosses_in_front_include_pedestrian_pedestrian: bool = False

    # Verified behavioral yielding interaction, v9.5.39.
    # A yield is not inferred from crossing order alone.  The subject must
    # exhibit a measurable motion concession before a verified crossing or
    # merge event, remain outside the crossing conflict while the other road
    # user passes, and subsequently proceed.  Ordinary RED-signal stopping is
    # conservatively excluded.
    yield_pre_event_window_s: float = 4.0
    yield_post_event_window_s: float = 4.0
    yield_max_initial_eta_difference_s: float = 2.5
    yield_min_baseline_speed_mps: float = 1.0
    yield_min_speed_drop_mps: float = 0.75
    yield_slow_speed_mps: float = 1.25
    yield_max_subject_conflict_distance_m: float = 18.0
    yield_conflict_clearance_buffer_m: float = 0.25
    yield_min_proceed_speed_mps: float = 0.80
    yield_min_proceed_displacement_m: float = 1.50

    # Future observed evidence
    future_horizon_s: float = 6.0
    future_step_s: float = 0.5
    minimum_future_coverage_ratio: float = 0.75

    # Maneuvers and map matching
    turn_angle_threshold_rad: float = 0.35
    uturn_angle_threshold_rad: float = 2.5
    lane_change_lateral_threshold_m: float = 0.75
    lane_stability_frames: int = 2
    primary_lane_min_overlap_ratio: float = 0.20
    primary_lane_ambiguity_margin: float = 0.05
    primary_map_lateral_tiebreak_margin_m: float = 0.50

    # Traffic control
    stop_line_search_radius_m: float = 35.0
    stop_line_near_distance_m: float = 4.0
    minimum_red_wait_duration_s: float = 1.0

    # Geometric visibility
    ego_fov_deg: float = 120.0
    ego_fov_range_m: float = 80.0

    # Conservative inferred intent
    minimum_intent_duration_s: float = 1.0
    gap_increase_threshold_m: float = 2.0
    arrival_time_tolerance_s: float = 1.5


DEFAULT_THRESHOLDS = PredicateThresholds()


def thresholds_as_dict() -> dict[str, float | int]:
    """Return the exact built-in threshold profile for run metadata."""
    return asdict(DEFAULT_THRESHOLDS)


PRIMARY_SPATIAL_SECTORS = (
    "np:inFrontOf",
    "np:frontLeftOf",
    "np:leftOf",
    "np:rearLeftOf",
    "np:behind",
    "np:rearRightOf",
    "np:rightOf",
    "np:frontRightOf",
)

MUTUAL_EXCLUSION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"np:isStopped", "np:isSlow", "np:isMoving"}),
    frozenset({"np:isAccelerating", "np:isDecelerating"}),
    frozenset(PRIMARY_SPATIAL_SECTORS),
    frozenset({"np:alignedWith", "np:oppositeDirectionTo"}),
    frozenset({"np:isClosingWith", "np:isRecedingFrom"}),
    frozenset({"np:movingToward", "np:movingAwayFrom"}),
    frozenset({"np:overlapping", "np:touching", "np:veryNear", "np:near"}),
    frozenset({"np:keepsLane", "np:changesLaneLeft", "np:changesLaneRight"}),
    frozenset({"np:turnsLeft", "np:turnsRight", "np:goesStraight", "np:makesUTurn"}),
)

SEMANTIC_BOOLEAN_PREDICATES = frozenset().union(*MUTUAL_EXCLUSION_GROUPS) | frozenset({
    "np:near",
    "np:veryNear",
    "np:withinEgoFieldOfView",
    "np:geometricallyVisibleToEgo",
    "np:hasInsufficientFutureEvidence",
    "np:inSameLaneAs",
    "np:sharesIntersectionWith",
    "np:hasAmbiguousMapMatch",
})


def wrap_signed(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def parse_boolean(value: Any) -> Optional[bool]:
    """Parse booleans safely; return None for non-boolean/ambiguous values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def classify_speed_state(speed_mps: Optional[float], stopped: float, moving: float) -> Optional[str]:
    if speed_mps is None or not math.isfinite(float(speed_mps)):
        return None
    speed = max(0.0, float(speed_mps))
    if speed <= stopped:
        return "np:isStopped"
    if speed <= moving:
        return "np:isSlow"
    return "np:isMoving"


def classify_acceleration_state(acceleration_mps2: Optional[float], threshold: float) -> Optional[str]:
    if acceleration_mps2 is None or not math.isfinite(float(acceleration_mps2)):
        return None
    acceleration = float(acceleration_mps2)
    if acceleration >= threshold:
        return "np:isAccelerating"
    if acceleration <= -threshold:
        return "np:isDecelerating"
    return None


def classify_primary_spatial_sector(
    longitudinal_m: Optional[float],
    lateral_m: Optional[float],
    *,
    colocated_epsilon_m: float = DEFAULT_THRESHOLDS.colocated_epsilon_m,
    overlapping: bool = False,
    longitudinal_deadband_m: float = DEFAULT_THRESHOLDS.spatial_longitudinal_deadband_m,
    lateral_deadband_m: float = DEFAULT_THRESHOLDS.spatial_lateral_deadband_m,
    side_longitudinal_band_m: float = DEFAULT_THRESHOLDS.spatial_side_longitudinal_band_m,
    front_lateral_base_m: float = DEFAULT_THRESHOLDS.spatial_front_lateral_base_m,
    front_lateral_ratio: float = DEFAULT_THRESHOLDS.spatial_front_lateral_ratio,
) -> Optional[str]:
    """Return exactly one threshold-aware subject-frame spatial relation.

    The old implementation divided the full circle into equal 45-degree angular
    sectors.  That made a distant object with a large lateral offset become
    ``leftOf`` even when it also had a large positive longitudinal displacement.

    The revised logic has three explicit zones:

    * ``leftOf``/``rightOf``: only close to the subject's lateral axis, i.e.
      ``abs(longitudinal) <= side_longitudinal_band_m``;
    * ``inFrontOf``/``behind``: inside a distance-widening longitudinal corridor;
    * diagonal predicates: clear longitudinal displacement outside that corridor.

    Therefore any object beyond the side band with positive longitudinal
    displacement is front or front-diagonal, never purely left/right.
    """
    if longitudinal_m is None or lateral_m is None or overlapping:
        return None
    x, y = float(longitudinal_m), float(lateral_m)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if math.hypot(x, y) <= colocated_epsilon_m:
        return None

    ax, ay = abs(x), abs(y)

    # Pure lateral relations are restricted to a narrow longitudinal band.
    if ax <= side_longitudinal_band_m:
        if y >= lateral_deadband_m:
            return "np:leftOf"
        if y <= -lateral_deadband_m:
            return "np:rightOf"
        if x >= longitudinal_deadband_m:
            return "np:inFrontOf"
        if x <= -longitudinal_deadband_m:
            return "np:behind"
        return None

    if x >= longitudinal_deadband_m:
        corridor_half_width = front_lateral_base_m + front_lateral_ratio * x
        if ay <= corridor_half_width:
            return "np:inFrontOf"
        return "np:frontLeftOf" if y > 0 else "np:frontRightOf"

    if x <= -longitudinal_deadband_m:
        corridor_half_width = front_lateral_base_m + front_lateral_ratio * ax
        if ay <= corridor_half_width:
            return "np:behind"
        return "np:rearLeftOf" if y > 0 else "np:rearRightOf"

    # This branch is possible only with unusual custom thresholds.
    if y >= lateral_deadband_m:
        return "np:leftOf"
    if y <= -lateral_deadband_m:
        return "np:rightOf"
    return None


def classify_body_frame_spatial_relation(
    *,
    subject_xyh: Sequence[float],
    object_xy: Sequence[float],
    overlapping: bool,
    longitudinal_deadband_m: float,
    lateral_deadband_m: float,
    side_longitudinal_band_m: float,
    front_lateral_base_m: float,
    front_lateral_ratio: float,
    colocated_epsilon_m: float = 1e-3,
) -> "SpatialClassification":
    """Classify object position in the subject body frame.

    This classifier is deliberately independent of lane matching, route
    topology, travel-speed reliability, pair relevance, and same-flow checks.
    It answers only the geometric question: where is the object relative to the
    subject's current body heading?  Map-aware and motion-aware evidence remains
    available separately in the map, heading, relevance, and interaction
    categories.
    """
    if len(subject_xyh) < 3 or len(object_xy) < 2:
        return SpatialClassification(
            None, "omitted", None, False, False,
            "invalid_body_frame_pose", "body_frame",
        )

    heading = subject_xyh[2]
    longitudinal, lateral = project_relative_position(
        subject_xyh[:2], object_xy[:2], heading
    )
    predicate_id = classify_primary_spatial_sector(
        longitudinal,
        lateral,
        colocated_epsilon_m=colocated_epsilon_m,
        overlapping=overlapping,
        longitudinal_deadband_m=longitudinal_deadband_m,
        lateral_deadband_m=lateral_deadband_m,
        side_longitudinal_band_m=side_longitudinal_band_m,
        front_lateral_base_m=front_lateral_base_m,
        front_lateral_ratio=front_lateral_ratio,
    )
    return SpatialClassification(
        predicate_id,
        "subject_body_frame_geometry",
        None,
        False,
        predicate_id is not None,
        None if predicate_id is not None else "body_frame_inside_deadband_or_overlap",
        "body_frame",
        float(heading) if heading is not None and math.isfinite(float(heading)) else None,
        longitudinal,
        lateral,
        None,
    )


@dataclass(frozen=True)
class DirectionEvidence:
    """Selected direction of travel and the evidence supporting it."""

    heading_rad: Optional[float]
    source: str
    reliable: bool
    omission_reason: Optional[str]
    map_heading_rad: Optional[float]
    velocity_heading_rad: Optional[float]
    displacement_heading_rad: Optional[float]
    velocity_displacement_difference_rad: Optional[float]
    map_motion_difference_rad: Optional[float]


@dataclass(frozen=True)
class SpatialClassification:
    predicate_id: Optional[str]
    source: str
    map_progress_delta_m: Optional[float]
    map_progress_eligible: bool
    relevant: bool = False
    omission_reason: Optional[str] = None
    map_relation: str = "unrelated"
    reference_heading_rad: Optional[float] = None
    travel_longitudinal_m: Optional[float] = None
    travel_lateral_m: Optional[float] = None
    travel_direction_difference_rad: Optional[float] = None


def _finite_angle(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return wrap_signed(number) if math.isfinite(number) else None


def angular_difference(a: Optional[float], b: Optional[float]) -> Optional[float]:
    aa, bb = _finite_angle(a), _finite_angle(b)
    if aa is None or bb is None:
        return None
    return abs(wrap_signed(aa - bb))


def circular_mean_pair(a: float, b: float) -> float:
    """Circular mean of two headings, robust around the -pi/pi boundary."""
    x = math.cos(a) + math.cos(b)
    y = math.sin(a) + math.sin(b)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return wrap_signed(a)
    return wrap_signed(math.atan2(y, x))


def select_travel_direction(
    *,
    body_heading_rad: Optional[float],
    map_heading_rad: Optional[float],
    map_match_reliable: bool,
    velocity_heading_rad: Optional[float],
    speed_mps: Optional[float],
    displacement_heading_rad: Optional[float],
    displacement_m: Optional[float],
    minimum_speed_mps: float,
    minimum_displacement_m: float,
    velocity_displacement_tolerance_rad: float,
    map_motion_tolerance_rad: float,
) -> DirectionEvidence:
    """Select a conservative travel direction from map and motion evidence.

    Priority is not simply map > velocity > body heading. The map direction is
    accepted only when the moving entity agrees with it. Velocity and temporal
    displacement must also agree when both are informative. Body heading is
    retained only as evidence and is never used alone for semantic front/behind
    classification because tracked-box yaw can be noisy or opposite to motion.
    """
    body = _finite_angle(body_heading_rad)
    map_heading = _finite_angle(map_heading_rad) if map_match_reliable else None
    velocity = _finite_angle(velocity_heading_rad)
    displacement = _finite_angle(displacement_heading_rad)

    speed_ok = (
        speed_mps is not None
        and math.isfinite(float(speed_mps))
        and float(speed_mps) >= minimum_speed_mps
        and velocity is not None
    )
    displacement_ok = (
        displacement_m is not None
        and math.isfinite(float(displacement_m))
        and float(displacement_m) >= minimum_displacement_m
        and displacement is not None
    )

    vd_difference = angular_difference(velocity, displacement) if speed_ok and displacement_ok else None
    if speed_ok and displacement_ok:
        if vd_difference is None or vd_difference > velocity_displacement_tolerance_rad:
            return DirectionEvidence(
                None, "unreliable", False, "velocity_displacement_conflict",
                map_heading, velocity, displacement, vd_difference, None,
            )
        motion = circular_mean_pair(float(velocity), float(displacement))
        motion_source = "velocity_and_displacement"
    elif speed_ok:
        motion = float(velocity)
        motion_source = "velocity"
    elif displacement_ok:
        motion = float(displacement)
        motion_source = "temporal_displacement"
    else:
        motion = None
        motion_source = "none"

    map_motion_difference = angular_difference(map_heading, motion) if map_heading is not None and motion is not None else None
    if map_heading is not None and motion is not None:
        if map_motion_difference is None or map_motion_difference > map_motion_tolerance_rad:
            return DirectionEvidence(
                None, "unreliable", False, "map_motion_direction_conflict",
                map_heading, velocity, displacement, vd_difference, map_motion_difference,
            )
        return DirectionEvidence(
            map_heading,
            f"map_confirmed_by_{motion_source}",
            True,
            None,
            map_heading,
            velocity,
            displacement,
            vd_difference,
            map_motion_difference,
        )

    if map_heading is not None:
        # A stopped or very slow road user can still inherit the legal lane-flow
        # direction from an unambiguous map match.
        return DirectionEvidence(
            map_heading, "map_low_speed", True, None,
            map_heading, velocity, displacement, vd_difference, None,
        )

    if motion is not None:
        return DirectionEvidence(
            motion, motion_source, True, None,
            None, velocity, displacement, vd_difference, None,
        )

    return DirectionEvidence(
        None, "unreliable", False, "no_reliable_map_or_motion_direction",
        map_heading, velocity, displacement, vd_difference, map_motion_difference,
    )


def project_relative_position(
    subject_xy: Sequence[float],
    object_xy: Sequence[float],
    reference_heading_rad: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if reference_heading_rad is None or len(subject_xy) < 2 or len(object_xy) < 2:
        return None, None
    values = [*subject_xy[:2], *object_xy[:2], reference_heading_rad]
    if not all(math.isfinite(float(value)) for value in values):
        return None, None
    sx, sy = float(subject_xy[0]), float(subject_xy[1])
    ox, oy = float(object_xy[0]), float(object_xy[1])
    heading = float(reference_heading_rad)
    dx, dy = ox - sx, oy - sy
    c, s = math.cos(heading), math.sin(heading)
    return float(c * dx + s * dy), float(-s * dx + c * dy)


def classify_relevant_spatial_relation(
    *,
    subject_xy: Sequence[float],
    object_xy: Sequence[float],
    center_distance_m: Optional[float],
    overlapping: bool,
    subject_direction: DirectionEvidence,
    object_direction: DirectionEvidence,
    map_relation: str,
    map_progress_delta_m: Optional[float],
    require_map_context: bool,
    relevance_max_distance_m: float,
    adjacent_relevance_max_distance_m: float,
    diagonal_max_longitudinal_m: float,
    motion_only_fallback_max_distance_m: float,
    motion_only_max_lateral_m: float,
    same_flow_threshold_rad: float,
    longitudinal_deadband_m: float,
    lateral_deadband_m: float,
    side_longitudinal_band_m: float,
    front_lateral_base_m: float = 2.5,
    front_lateral_ratio: float = 0.30,
    colocated_epsilon_m: float = 1e-3,
) -> SpatialClassification:
    """Emit only map- and motion-relevant spatial semantics.

    Rules:
      * same lane/connector -> front or behind from map progress only;
      * adjacent parallel lane -> left/right only when longitudinally abreast,
        diagonal only at moderate longitudinal gaps, and front/behind at large
        gaps where lateral lane identity is no longer the dominant fact;
      * connected successor/predecessor -> front/behind when geometry agrees;
      * cross-traffic, opposite flow, ambiguous direction, unrelated map objects,
        excessive distance, and overlap -> no semantic spatial predicate;
      * optional motion-only fallback is disabled by default through
        ``require_map_context=True``.
    """
    relation = str(map_relation or "unrelated")
    if overlapping:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     "overlapping_pair", relation)
    if center_distance_m is None or not math.isfinite(float(center_distance_m)):
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     "invalid_center_distance", relation)
    distance = float(center_distance_m)
    if distance <= colocated_epsilon_m:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     "colocated_pair", relation)
    if distance > relevance_max_distance_m:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     "outside_spatial_relevance_range", relation)
    if not subject_direction.reliable or subject_direction.heading_rad is None:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     f"subject_{subject_direction.omission_reason or 'direction_unreliable'}", relation)
    if not object_direction.reliable or object_direction.heading_rad is None:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     f"object_{object_direction.omission_reason or 'direction_unreliable'}", relation)

    direction_difference = angular_difference(subject_direction.heading_rad, object_direction.heading_rad)
    if direction_difference is None or direction_difference > same_flow_threshold_rad:
        return SpatialClassification(
            None, "omitted", map_progress_delta_m, False, False,
            "cross_or_opposite_travel_direction", relation,
            subject_direction.heading_rad, None, None, direction_difference,
        )

    longitudinal, lateral = project_relative_position(
        subject_xy, object_xy, subject_direction.heading_rad
    )
    if longitudinal is None or lateral is None:
        return SpatialClassification(
            None, "omitted", map_progress_delta_m, False, False,
            "invalid_travel_frame_projection", relation,
            subject_direction.heading_rad, longitudinal, lateral, direction_difference,
        )

    common = dict(
        map_relation=relation,
        reference_heading_rad=subject_direction.heading_rad,
        travel_longitudinal_m=longitudinal,
        travel_lateral_m=lateral,
        travel_direction_difference_rad=direction_difference,
    )

    local_corridor_half_width = (
        float(front_lateral_base_m)
        + float(front_lateral_ratio) * abs(float(longitudinal))
    )
    local_order_reliable = bool(
        abs(float(lateral)) <= local_corridor_half_width
        and abs(float(longitudinal)) >= longitudinal_deadband_m
    )
    local_order_sign = 1 if longitudinal >= longitudinal_deadband_m else (
        -1 if longitudinal <= -longitudinal_deadband_m else 0
    )

    if relation == "same_map":
        if map_progress_delta_m is None or not math.isfinite(float(map_progress_delta_m)):
            return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                         "same_map_without_comparable_progress", **common)
        delta = float(map_progress_delta_m)
        map_order_sign = 1 if delta >= longitudinal_deadband_m else (
            -1 if delta <= -longitudinal_deadband_m else 0
        )
        # On a clearly local same-lane corridor, the subject travel frame is a
        # direct observable. It prevents an inverted map baseline/progress sign
        # from labeling a visibly forward vehicle as behind (or vice versa).
        if local_order_reliable and local_order_sign != 0 and local_order_sign != map_order_sign:
            pid = "np:inFrontOf" if local_order_sign > 0 else "np:behind"
            return SpatialClassification(
                pid, "local_travel_frame_overrode_map_progress_conflict", delta,
                True, True, None, **common
            )
        if map_order_sign > 0:
            return SpatialClassification("np:inFrontOf", "same_map_progress", delta, True, True, None, **common)
        if map_order_sign < 0:
            return SpatialClassification("np:behind", "same_map_progress", delta, True, True, None, **common)
        if local_order_reliable and local_order_sign != 0:
            pid = "np:inFrontOf" if local_order_sign > 0 else "np:behind"
            return SpatialClassification(pid, "local_travel_frame_resolved_progress_deadband", delta, True, True, None, **common)
        return SpatialClassification(None, "omitted", delta, True, False,
                                     "same_map_progress_inside_deadband", **common)

    if relation in {"adjacent_left", "adjacent_right"}:
        if distance > adjacent_relevance_max_distance_m:
            return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                         "adjacent_lane_outside_relevance_range", **common)
        side_left = relation == "adjacent_left"
        # The official lane topology fixes the side. Geometry is used only as a
        # sanity check, while along-road ordering comes from projection onto the
        # subject lane baseline. This remains correct on curved roads.
        if side_left and lateral < -lateral_deadband_m:
            return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                         "left_adjacency_geometry_conflict", **common)
        if not side_left and lateral > lateral_deadband_m:
            return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                         "right_adjacency_geometry_conflict", **common)
        if map_progress_delta_m is None or not math.isfinite(float(map_progress_delta_m)):
            return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                         "adjacent_lane_without_common_path_projection", **common)
        along = float(map_progress_delta_m)
        if abs(along) <= side_longitudinal_band_m:
            pid = "np:leftOf" if side_left else "np:rightOf"
        elif abs(along) <= diagonal_max_longitudinal_m:
            if along > 0:
                pid = "np:frontLeftOf" if side_left else "np:frontRightOf"
            else:
                pid = "np:rearLeftOf" if side_left else "np:rearRightOf"
        else:
            pid = "np:inFrontOf" if along > 0 else "np:behind"
        return SpatialClassification(pid, "adjacent_topology_and_common_path_progress",
                                     along, True, True, None, **common)

    if relation in {"successor", "predecessor"}:
        path_valid = map_progress_delta_m is not None and math.isfinite(float(map_progress_delta_m))
        path_sign = 0
        if path_valid:
            path_value = float(map_progress_delta_m)
            path_sign = 1 if path_value >= longitudinal_deadband_m else (
                -1 if path_value <= -longitudinal_deadband_m else 0
            )
        expected_topology_sign = 1 if relation == "successor" else -1

        if local_order_reliable and local_order_sign != 0 and local_order_sign != expected_topology_sign:
            pid = "np:inFrontOf" if local_order_sign > 0 else "np:behind"
            return SpatialClassification(
                pid, "local_travel_frame_overrode_connected_topology_conflict",
                map_progress_delta_m, path_valid, True, None, **common
            )
        if path_sign == expected_topology_sign:
            pid = "np:inFrontOf" if expected_topology_sign > 0 else "np:behind"
            return SpatialClassification(
                pid, "connected_lane_graph_distance", map_progress_delta_m,
                True, True, None, **common
            )
        if local_order_reliable and local_order_sign == expected_topology_sign:
            pid = "np:inFrontOf" if local_order_sign > 0 else "np:behind"
            return SpatialClassification(
                pid, "local_travel_frame_confirmed_connected_order",
                map_progress_delta_m, path_valid, True, None, **common
            )
        reason = (
            "successor_without_positive_path_distance"
            if relation == "successor"
            else "predecessor_without_negative_path_distance"
        )
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False, reason, **common)

    if require_map_context:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     "unrelated_or_ambiguous_map_context", **common)

    if distance > motion_only_fallback_max_distance_m:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     "motion_fallback_outside_range", **common)
    if abs(lateral) > motion_only_max_lateral_m:
        return SpatialClassification(None, "omitted", map_progress_delta_m, False, False,
                                     "motion_fallback_excessive_lateral_offset", **common)
    pid = classify_primary_spatial_sector(
        longitudinal,
        lateral,
        colocated_epsilon_m=colocated_epsilon_m,
        overlapping=False,
        longitudinal_deadband_m=longitudinal_deadband_m,
        lateral_deadband_m=lateral_deadband_m,
        side_longitudinal_band_m=side_longitudinal_band_m,
    )
    return SpatialClassification(pid, "motion_only_fallback", map_progress_delta_m, False,
                                 pid is not None, None if pid else "motion_fallback_deadband", **common)


# Backward-compatible wrapper retained for old imports. New extraction uses
# classify_relevant_spatial_relation and supplies explicit map/motion evidence.
def classify_map_aware_spatial_relation(
    longitudinal_m: Optional[float],
    lateral_m: Optional[float],
    *,
    overlapping: bool,
    shared_primary_map: bool,
    subject_progress_m: Optional[float],
    object_progress_m: Optional[float],
    heading_difference_rad: Optional[float],
    subject_map_lateral_offset_m: Optional[float],
    object_map_lateral_offset_m: Optional[float],
    alignment_threshold_rad: float,
    same_lane_lateral_tolerance_m: float,
    colocated_epsilon_m: float,
    longitudinal_deadband_m: float,
    lateral_deadband_m: float,
    side_longitudinal_band_m: float,
    front_lateral_base_m: float,
    front_lateral_ratio: float,
) -> SpatialClassification:
    map_eligible = bool(
        shared_primary_map
        and not overlapping
        and subject_progress_m is not None
        and object_progress_m is not None
        and heading_difference_rad is not None
        and float(heading_difference_rad) <= alignment_threshold_rad
        and abs(float(subject_map_lateral_offset_m or 0.0)) <= same_lane_lateral_tolerance_m
        and abs(float(object_map_lateral_offset_m or 0.0)) <= same_lane_lateral_tolerance_m
    )
    progress_delta = None
    if map_eligible:
        progress_delta = float(object_progress_m) - float(subject_progress_m)
        if progress_delta >= longitudinal_deadband_m:
            return SpatialClassification("np:inFrontOf", "shared_primary_map_progress", progress_delta, True, True)
        if progress_delta <= -longitudinal_deadband_m:
            return SpatialClassification("np:behind", "shared_primary_map_progress", progress_delta, True, True)
    pid = classify_primary_spatial_sector(
        longitudinal_m, lateral_m,
        colocated_epsilon_m=colocated_epsilon_m,
        overlapping=overlapping,
        longitudinal_deadband_m=longitudinal_deadband_m,
        lateral_deadband_m=lateral_deadband_m,
        side_longitudinal_band_m=side_longitudinal_band_m,
        front_lateral_base_m=front_lateral_base_m,
        front_lateral_ratio=front_lateral_ratio,
    )
    return SpatialClassification(pid, "subject_frame_thresholds", progress_delta, map_eligible, pid is not None)

def classify_distance_state(
    free_space_m: Optional[float],
    *,
    intersection_area_m2: Optional[float],
    touching: Optional[bool],
    overlap_area_epsilon_m2: float,
    distance_epsilon_m: float,
    very_near_m: float,
    near_m: float,
) -> Optional[str]:
    if intersection_area_m2 is not None and float(intersection_area_m2) > overlap_area_epsilon_m2:
        return "np:overlapping"
    if touching is True:
        return "np:touching"
    if free_space_m is None or not math.isfinite(float(free_space_m)):
        return None
    distance = max(0.0, float(free_space_m))
    if distance <= distance_epsilon_m:
        return "np:touching"
    if distance <= very_near_m:
        return "np:veryNear"
    if distance <= near_m:
        return "np:near"
    return None


def classify_heading_state(heading_difference_rad: Optional[float], aligned: float, opposite: float) -> Optional[str]:
    if heading_difference_rad is None or not math.isfinite(float(heading_difference_rad)):
        return None
    difference = min(math.pi, max(0.0, float(heading_difference_rad)))
    if difference <= aligned:
        return "np:alignedWith"
    if abs(math.pi - difference) <= opposite:
        return "np:oppositeDirectionTo"
    return None


def classify_signed_state(value: Optional[float], positive_pid: str, negative_pid: str, threshold: float) -> Optional[str]:
    if value is None or not math.isfinite(float(value)):
        return None
    number = float(value)
    if number > threshold:
        return positive_pid
    if number < -threshold:
        return negative_pid
    return None


def check_threshold_relationships(args: Any) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(0 <= args.stopped_speed_mps < args.moving_speed_mps,
            "Require 0 <= stopped_speed_mps < moving_speed_mps")
    require(0 <= args.very_near_distance_m < args.near_distance_m,
            "Require 0 <= very_near_distance_m < near_distance_m")
    require(args.maximum_follow_distance_m > args.maximum_queue_follow_gap_m > 0,
            "Require maximum_follow_distance_m > maximum_queue_follow_gap_m > 0")
    require(args.follow_queue_leader_speed_threshold_mps >= args.follow_queue_speed_threshold_mps >= 0,
            "Require queue leader speed threshold >= queue subject speed threshold >= 0")
    require(args.follow_reverse_speed_tolerance_mps >= 0,
            "Require follow_reverse_speed_tolerance_mps >= 0")
    require(args.follow_leader_tie_tolerance_m >= 0,
            "Require follow_leader_tie_tolerance_m >= 0")
    require(int(args.follow_max_path_hops) >= 1,
            "Require follow_max_path_hops >= 1")
    require(args.minimum_overtake_approach_duration_s >= 0,
            "Require minimum_overtake_approach_duration_s >= 0")
    require(args.minimum_overtake_passing_duration_s >= 0,
            "Require minimum_overtake_passing_duration_s >= 0")
    require(args.minimum_overtake_completion_duration_s > 0,
            "Require minimum_overtake_completion_duration_s > 0")
    require(args.minimum_overtake_relative_speed_mps > 0,
            "Require minimum_overtake_relative_speed_mps > 0")
    require(args.minimum_overtake_subject_speed_mps > 0,
            "Require minimum_overtake_subject_speed_mps > 0")
    require(args.minimum_overtaken_vehicle_speed_mps >= 0,
            "Require minimum_overtaken_vehicle_speed_mps >= 0")
    require(args.minimum_overtake_clearance_m > 0,
            "Require minimum_overtake_clearance_m > 0")
    require(args.maximum_overtake_entry_gap_m > args.minimum_overtake_clearance_m,
            "Require maximum_overtake_entry_gap_m > minimum_overtake_clearance_m")
    require(0 <= args.overtake_departure_max_lead_m < args.maximum_follow_distance_m,
            "Require 0 <= overtake_departure_max_lead_m < maximum_follow_distance_m")
    require(0 <= args.overtake_same_flow_threshold_rad < math.pi / 2,
            "Require 0 <= overtake_same_flow_threshold_rad < pi/2")
    require(int(args.overtake_lane_stability_frames) >= 2,
            "Require overtake_lane_stability_frames >= 2")
    require(args.maximum_overtake_lane_transition_duration_s >= args.sample_interval_s,
            "Require maximum_overtake_lane_transition_duration_s >= sample_interval_s")
    require(args.maximum_overtake_duration_s > (
                args.minimum_overtake_approach_duration_s
                + args.minimum_overtake_passing_duration_s
                + args.minimum_overtake_completion_duration_s
            ),
            "Require maximum_overtake_duration_s to exceed the sum of minimum phase durations")
    require(getattr(args, "overtake_pair_query_radius_m", 80.0) > 0,
            "Require overtake_pair_query_radius_m > 0")
    require(int(getattr(args, "overtake_min_common_frames", 5)) >= 2,
            "Require overtake_min_common_frames >= 2")
    require(getattr(args, "overtake_search_padding_s", 4.0) >= 0,
            "Require overtake_search_padding_s >= 0")
    require(getattr(args, "overtake_max_frame_gap_s", 1.5) > 0,
            "Require overtake_max_frame_gap_s > 0")
    require(getattr(args, "minimum_overtake_initial_behind_gap_m", 1.0) > 0,
            "Require minimum_overtake_initial_behind_gap_m > 0")
    require(getattr(args, "minimum_overtake_side_by_side_lateral_m", 1.25) > 0,
            "Require minimum_overtake_side_by_side_lateral_m > 0")
    require(0 < getattr(args, "minimum_overtake_same_flow_fraction", 0.80) <= 1,
            "Require minimum_overtake_same_flow_fraction in (0, 1]")
    require(getattr(args, "overtake_lane_query_radius_m", 8.0) > 0,
            "Require overtake_lane_query_radius_m > 0")
    require(0 <= getattr(args, "overtake_lane_min_overlap_ratio", 0.10) <= 1,
            "Require overtake_lane_min_overlap_ratio in [0, 1]")
    require(0 <= getattr(args, "overtake_lane_ambiguity_margin", 0.05) <= 1,
            "Require overtake_lane_ambiguity_margin in [0, 1]")
    require(int(getattr(args, "overtake_max_lane_path_hops", 8)) >= 1,
            "Require overtake_max_lane_path_hops >= 1")
    require(getattr(args, "overtake_positive_deduplication_tolerance_s", 2.0) >= 0,
            "Require overtake_positive_deduplication_tolerance_s >= 0")
    require(int(getattr(args, "minimum_maneuver_history_frames", 2)) >= 1,
            "Require minimum_maneuver_history_frames >= 1")
    require(getattr(args, "minimum_maneuver_history_s", 0.5) >= 0,
            "Require minimum_maneuver_history_s >= 0")
    require(0 <= args.alignment_threshold_rad < math.pi / 2,
            "Require 0 <= alignment_threshold_rad < pi/2")
    require(0 <= args.opposite_threshold_rad < math.pi / 2,
            "Require 0 <= opposite_threshold_rad < pi/2")
    require(0 < args.turn_angle_threshold_rad < args.uturn_angle_threshold_rad <= math.pi,
            "Require 0 < turn_angle_threshold_rad < uturn_angle_threshold_rad <= pi")
    require(0 < args.future_step_s <= args.future_horizon_s,
            "Require 0 < future_step_s <= future_horizon_s")
    require(0 < args.ego_fov_deg <= 360,
            "Require 0 < ego_fov_deg <= 360")
    require(0 <= args.spatial_front_lateral_ratio <= 1,
            "Require 0 <= spatial_front_lateral_ratio <= 1")
    require(args.spatial_side_longitudinal_band_m >= args.spatial_longitudinal_deadband_m,
            "Require spatial_side_longitudinal_band_m >= spatial_longitudinal_deadband_m")
    require(0 < args.spatial_same_flow_threshold_rad < math.pi / 2,
            "Require 0 < spatial_same_flow_threshold_rad < pi/2")
    require(0 < args.spatial_direction_agreement_threshold_rad < math.pi / 2,
            "Require 0 < spatial_direction_agreement_threshold_rad < pi/2")
    require(0 < args.spatial_map_motion_tolerance_rad < math.pi / 2,
            "Require 0 < spatial_map_motion_tolerance_rad < pi/2")
    require(args.spatial_adjacent_relevance_max_distance_m <= args.spatial_relevance_max_distance_m,
            "Require adjacent spatial range <= general spatial range")
    require(args.spatial_diagonal_max_longitudinal_m >= args.spatial_side_longitudinal_band_m,
            "Require diagonal max longitudinal distance >= side band")

    for name in (
        "minimum_follow_duration_s", "minimum_red_wait_duration_s",
        "minimum_intent_duration_s", "sample_interval_s",
        "maximum_temporal_gap_factor", "minimum_future_coverage_ratio",
    ):
        require(float(getattr(args, name)) > 0, f"Require {name} > 0")
    for name in (
        "distance_epsilon_m", "overlap_area_epsilon_m2", "colocated_epsilon_m",
        "lane_change_lateral_threshold_m", "spatial_longitudinal_deadband_m",
        "spatial_lateral_deadband_m", "spatial_side_longitudinal_band_m",
        "spatial_front_lateral_base_m", "spatial_relevance_max_distance_m",
        "spatial_adjacent_relevance_max_distance_m", "spatial_diagonal_max_longitudinal_m",
        "spatial_motion_only_fallback_max_distance_m", "spatial_motion_only_max_lateral_m",
        "spatial_min_direction_speed_mps", "spatial_min_displacement_for_direction_m",
        "spatial_direction_agreement_threshold_rad", "spatial_map_motion_tolerance_rad",
        "spatial_same_flow_threshold_rad", "spatial_connected_sign_tolerance_m",
    ):
        require(float(getattr(args, name)) >= 0, f"Require {name} >= 0")
    require(int(args.minimum_acceleration_samples) >= 2,
            "Require minimum_acceleration_samples >= 2")
    require(0 < args.minimum_future_coverage_ratio <= 1,
            "Require 0 < minimum_future_coverage_ratio <= 1")
    require(int(getattr(args, "crosses_in_front_min_observations", 3)) >= 2,
            "Require crosses_in_front_min_observations >= 2")
    require(getattr(args, "crosses_in_front_min_simultaneous_observation_s", 1.0) >= 0,
            "Require crosses_in_front_min_simultaneous_observation_s >= 0")
    require(getattr(args, "crosses_in_front_max_pair_distance_m", 30.0) > 0,
            "Require crosses_in_front_max_pair_distance_m > 0")
    require(getattr(args, "crosses_in_front_max_front_distance_m", 20.0) > 0,
            "Require crosses_in_front_max_front_distance_m > 0")
    require(getattr(args, "crosses_in_front_side_buffer_m", 0.35) >= 0,
            "Require crosses_in_front_side_buffer_m >= 0")
    require(getattr(args, "crosses_in_front_side_search_window_s", 3.0) > 0,
            "Require crosses_in_front_side_search_window_s > 0")
    require(
        0 <= getattr(args, "crosses_in_front_min_crossing_angle_deg", 25.0)
        < getattr(args, "crosses_in_front_max_crossing_angle_deg", 155.0)
        <= 180,
        "Require 0 <= crosses_in_front_min_crossing_angle_deg < "
        "crosses_in_front_max_crossing_angle_deg <= 180",
    )
    require(getattr(args, "crosses_in_front_min_subject_crossing_speed_mps", 0.30) >= 0,
            "Require crosses_in_front_min_subject_crossing_speed_mps >= 0")
    require(getattr(args, "crosses_in_front_max_arrival_time_difference_s", 6.0) > 0,
            "Require crosses_in_front_max_arrival_time_difference_s > 0")
    require(getattr(args, "crosses_in_front_order_margin_s", 0.25) >= 0,
            "Require crosses_in_front_order_margin_s >= 0")
    require(getattr(args, "crosses_in_front_conflict_footprint_buffer_m", 0.35) >= 0,
            "Require crosses_in_front_conflict_footprint_buffer_m >= 0")
    require(getattr(args, "crosses_in_front_max_stationary_object_speed_mps", 1.0) >= 0,
            "Require crosses_in_front_max_stationary_object_speed_mps >= 0")
    require(getattr(args, "crosses_in_front_max_stationary_object_front_distance_m", 12.0) > 0,
            "Require crosses_in_front_max_stationary_object_front_distance_m > 0")
    require(getattr(args, "crosses_in_front_map_query_radius_m", 12.0) > 0,
            "Require crosses_in_front_map_query_radius_m > 0")
    require(getattr(args, "crosses_in_front_max_road_distance_m", 6.0) >= 0,
            "Require crosses_in_front_max_road_distance_m >= 0")
    require(getattr(args, "crosses_in_front_crosswalk_near_distance_m", 5.0) >= 0,
            "Require crosses_in_front_crosswalk_near_distance_m >= 0")
    require(getattr(args, "crosses_in_front_bicycle_road_distance_m", 3.0) >= 0,
            "Require crosses_in_front_bicycle_road_distance_m >= 0")
    require(getattr(args, "crosses_in_front_parked_max_speed_mps", 0.5) >= 0,
            "Require crosses_in_front_parked_max_speed_mps >= 0")
    require(getattr(args, "yield_pre_event_window_s", 4.0) > 0,
            "Require yield_pre_event_window_s > 0")
    require(getattr(args, "yield_post_event_window_s", 4.0) > 0,
            "Require yield_post_event_window_s > 0")
    require(getattr(args, "yield_max_initial_eta_difference_s", 2.5) > 0,
            "Require yield_max_initial_eta_difference_s > 0")
    for name, default in (
        ("yield_min_baseline_speed_mps", 1.0),
        ("yield_min_speed_drop_mps", 0.75),
        ("yield_slow_speed_mps", 1.25),
        ("yield_max_subject_conflict_distance_m", 18.0),
        ("yield_conflict_clearance_buffer_m", 0.25),
        ("yield_min_proceed_speed_mps", 0.80),
        ("yield_min_proceed_displacement_m", 1.50),
    ):
        require(getattr(args, name, default) >= 0, f"Require {name} >= 0")
    require(getattr(args, "crosses_in_front_parked_min_duration_s", 2.0) >= 0,
            "Require crosses_in_front_parked_min_duration_s >= 0")
    require(getattr(args, "crosses_in_front_parked_max_displacement_m", 1.0) >= 0,
            "Require crosses_in_front_parked_max_displacement_m >= 0")
    require(getattr(args, "crosses_in_front_parked_max_lane_distance_m", 1.5) >= 0,
            "Require crosses_in_front_parked_max_lane_distance_m >= 0")

    return errors


def observations_are_continuous(
    previous_frame_index: int,
    current_frame_index: int,
    previous_timestamp_us: int,
    current_timestamp_us: int,
    *,
    sample_interval_s: float,
    maximum_gap_factor: float,
) -> bool:
    if current_frame_index != previous_frame_index + 1:
        return False
    dt_s = (int(current_timestamp_us) - int(previous_timestamp_us)) / 1e6
    return 0 < dt_s <= sample_interval_s * maximum_gap_factor


@dataclass
class StreakResult:
    active: bool
    start_timestamp_us: Optional[int]
    duration_s: float
    frame_count: int


def update_condition_streak(
    store: MutableMapping[Any, Mapping[str, Any]],
    key: Any,
    *,
    condition: bool,
    current_timestamp_us: int,
    current_frame_index: int,
    continuous_with_previous: bool,
) -> StreakResult:
    previous = store.get(key)
    if not condition:
        store.pop(key, None)
        return StreakResult(False, None, 0.0, 0)
    if previous is None or not continuous_with_previous:
        start = int(current_timestamp_us)
        count = 1
    else:
        start = int(previous["start_timestamp_us"])
        count = int(previous["frame_count"]) + 1
    duration = max(0.0, (int(current_timestamp_us) - start) / 1e6)
    store[key] = {
        "start_timestamp_us": start,
        "last_timestamp_us": int(current_timestamp_us),
        "last_frame_index": int(current_frame_index),
        "frame_count": count,
        "duration_s": duration,
    }
    return StreakResult(True, start, duration, count)


def canonical_symmetric_pair(token_a: str, token_b: str) -> tuple[str, str]:
    a, b = str(token_a), str(token_b)
    return (a, b) if a <= b else (b, a)
