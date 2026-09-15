"""Regression tests that keep the public repository aligned with the paper appendix."""
from types import SimpleNamespace

from nuplan_predicates_modular import base
from nuplan_predicates_modular.categories import motion, risk


PAPER_PREDICATES = {
    # Spatial (12)
    "inFrontOf", "behind", "leftOf", "rightOf", "frontLeftOf", "frontRightOf",
    "rearLeftOf", "rearRightOf", "overlapping", "touching", "veryNear", "near",
    # Motion (17)
    "hasVelocity", "hasVelocityX", "hasVelocityY", "hasSpeed", "hasAccelerationX",
    "hasAccelerationY", "hasAcceleration", "hasRelativeSpeedTo",
    "hasLongitudinalRelativeSpeedTo", "hasLateralRelativeSpeedTo", "hasClosingSpeedTo",
    "hasVelocityTowardTarget", "hasSubjectForwardSpeed", "hasVelocityHeading",
    "hasEffectiveTravelHeading", "hasTravelDirectionSource", "hasTravelDirectionDifferenceTo",
    # Temporal (17)
    "precedes", "hasDeltaTimeFromPrevious", "hasDisplacementFromPrevious",
    "hasHeadingChangeFromPrevious", "hasSpeedChangeFromPrevious", "hasEstimatedAcceleration",
    "hasDisplacementHeading", "hasObservedFrameCount", "hasObservedDuration",
    "hasTotalObservedFrameCount", "hasTotalObservedSpan", "hasContinuousObservedFrameCount",
    "hasContinuousObservedDuration", "hasPairObservedFrameCount", "hasPairObservedDuration",
    "hasCenterDistanceChangeFromPrevious", "hasFreeSpaceDistanceChangeFromPrevious",
    # Map (25)
    "inLane", "inLaneConnector", "intersectsLane", "intersectsLaneConnector", "hasPrimaryLane",
    "hasPrimaryLaneConnector", "hasPrimaryMapOverlapRatio", "hasAmbiguousMapMatch",
    "hasBaselineProgress", "hasBaselineLateralOffset", "hasMapHeading", "hasBaselineCurvature",
    "hasMapSpeedLimit", "hasParentRoadblock", "hasParentRoadblockConnector", "inIntersection",
    "intersectsIntersection", "hasPrimaryMapIntersection", "inCrosswalk", "intersectsCrosswalk",
    "hasSpatialMapRelation", "hasMapProgressDifferenceTo", "hasSignedPathDistanceTo",
    "inSameLaneAs", "sharesIntersectionWith",
    # Interaction (8)
    "follows", "queuesBehind", "changesLane", "mergesInFrontOf", "mergesBehind",
    "crossesInFrontOf", "yieldsTo", "overtakes",
    # Traffic control (3)
    "controls", "hasSignalState", "isRelevantSignal",
    # Risk (1)
    "hasConflictRiskWith",
}


def test_all_83_paper_predicates_are_registered():
    assert len(PAPER_PREDICATES) == 83
    paper_categories = {"spatial", "motion", "temporal", "map", "interaction", "traffic_light", "risk"}
    registered = {
        definition.predicate_id.removeprefix("np:")
        for definition in base.PREDICATES
        if definition.category in paper_categories
    }
    assert registered == PAPER_PREDICATES


def test_has_velocity_is_materialized_as_planar_vector():
    frame = SimpleNamespace(
        timestamp_us=123,
        log_name="log",
        scenario_token="scenario",
        dataset="nuPlan",
        dataset_version="v1.1",
        split="mini",
    )
    entity = SimpleNamespace(velocity=SimpleNamespace(x=3.0, y=4.0))
    record = {"entity_id": "agent:a", "agent_type": "VEHICLE"}
    assertions = []
    motion.append_entity_motion_measurements(
        assertions, frame, record, entity, "tracked_object", state={}
    )
    velocity = next(a for a in assertions if a.predicate_id == "np:hasVelocity")
    speed = next(a for a in assertions if a.predicate_id == "np:hasSpeed")
    assert velocity.value == [3.0, 4.0]
    assert velocity.value_type == base.ValueType.VECTOR2
    assert speed.value == 5.0


def test_un_r157_imminent_braking_gate():
    demand = risk._collision_avoidance_braking_demand_mps2(5.0, 1.5, 0.5)
    assert demand == 12.5
    ok, evidence = risk._safety_gate(
        {"map_relation": "same_map"},
        {"track_token": "a", "agent_type": "VEHICLE"},
        {"track_token": "b", "agent_type": "VEHICLE"},
        (5.0, 0.0),
        (0.0, 0.0),
        closing_speed_mps=5.0,
        current_clearance_m=1.5,
        predicted_min_clearance_m=0.2,
        protected_clearance_m=0.5,
    )
    assert ok is True
    assert evidence["imminent_collision_condition"] is True


def test_un_r157_lane_change_gap_gate_for_ego():
    ok, evidence = risk._safety_gate(
        {"map_relation": "adjacent_left"},
        {"track_token": "ego", "agent_type": "EGO"},
        {"track_token": "target", "agent_type": "VEHICLE"},
        (10.0, 0.0),
        (10.0, 0.0),
        closing_speed_mps=0.6,
        current_clearance_m=20.0,
        predicted_min_clearance_m=5.0,
        protected_clearance_m=0.5,
    )
    # Braking demand is small, but the 5 m predicted gap is below the 10 m
    # distance travelled by the ego in the configured 1 s R157 gap window.
    assert evidence["imminent_collision_condition"] is False
    assert evidence["lane_change_safety"]["applicable"] is True
    assert evidence["lane_change_safety"]["gap_below_one_second_travel"] is True
    assert ok is True
