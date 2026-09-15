"""Regression tests for v9.5.34 finite-10m-ray crossesInFrontOf."""

import math
from types import SimpleNamespace

from nuplan_predicates_modular.categories import interaction


def _record(token, x, y, heading, vx, vy, agent_type, *, width=2.0, length=4.5):
    return {
        "entity_id": f"entity_{token}_{x}_{y}",
        "track_token": token,
        "xyh": (float(x), float(y), float(heading)),
        "velocity": (float(vx), float(vy)),
        "speed": float(math.hypot(vx, vy)),
        "agent_type": agent_type,
        "dimensions": {
            "length_m": float(length),
            "width_m": float(width),
            "height_m": 1.5,
        },
    }


def _frame(timestamp_us):
    return SimpleNamespace(
        timestamp_us=timestamp_us,
        log_name="test_log",
        scenario_token="test_scenario",
    )


def _snapshots(
    *,
    object_x=4.0,
    object_y=-1.0,
    object_heading=math.pi / 2.0,
    object_velocity=(0.0, 0.0),
    subject_velocity=(2.0, 0.0),
    subject_type="PEDESTRIAN",
    object_type="VEHICLE",
    frames=5,
):
    result = []
    for index in range(frames):
        t_s = index * 0.5
        timestamp_us = index * 500_000
        sx = subject_velocity[0] * t_s
        sy = subject_velocity[1] * t_s
        ox = object_x + object_velocity[0] * t_s
        oy = object_y + object_velocity[1] * t_s
        subject = _record(
            "S", sx, sy, 0.0,
            subject_velocity[0], subject_velocity[1], subject_type,
            width=0.8 if subject_type == "PEDESTRIAN" else 2.0,
            length=0.8 if subject_type == "PEDESTRIAN" else 4.5,
        )
        obj = _record(
            "O", ox, oy, object_heading,
            object_velocity[0], object_velocity[1], object_type,
            width=0.8 if object_type == "PEDESTRIAN" else 2.0,
            length=0.8 if object_type == "PEDESTRIAN" else 4.5,
        )
        result.append({
            "timestamp_us": timestamp_us,
            "frame_index": index,
            "frame": _frame(timestamp_us),
            "entities": {"S": subject, "O": obj},
        })
    return result


def test_finite_forward_ray_intersection_enforces_10m_on_both_agents():
    hit = interaction._crosses_forward_ray_intersection(
        (0.0, 0.0), 0.0,
        (4.0, -1.0), math.pi / 2.0,
        max_distance_m=10.0,
    )
    assert hit is not None
    assert math.isclose(hit["subject_lambda_m"], 4.0, abs_tol=1e-9)
    assert math.isclose(hit["object_mu_m"], 1.0, abs_tol=1e-9)

    assert interaction._crosses_forward_ray_intersection(
        (0.0, 0.0), 0.0,
        (11.0, -1.0), math.pi / 2.0,
        max_distance_m=10.0,
    ) is None

    assert interaction._crosses_forward_ray_intersection(
        (0.0, 0.0), 0.0,
        (-1.0, -1.0), math.pi / 2.0,
        max_distance_m=10.0,
    ) is None


def test_stopped_nonparked_object_can_be_crossed_in_front_of_within_10m():
    events = interaction._crosses_find_events(_snapshots())
    assert len(events) == 1
    assert events[0]["subject_token"] == "S"
    assert events[0]["object_token"] == "O"
    assert events[0]["reference_geometry"] == "finite_forward_ray_segments_10m"
    assert all(
        row["subject_ray_distance_m"] <= 10.0
        and row["object_ray_distance_m"] <= 10.0
        for row in events[0]["active_frames"]
    )


def test_predicate_starts_only_when_previously_far_conflict_enters_10m_segments():
    events = interaction._crosses_find_events(_snapshots(object_x=11.0))
    assert len(events) == 1
    event = events[0]
    assert event["entry_time_us"] == 500_000
    assert event["active_frames"][0]["subject_ray_distance_m"] == 10.0


def test_parallel_rays_are_rejected():
    snapshots = _snapshots(object_heading=0.0)
    assert interaction._crosses_find_events(snapshots) == []


def test_shallow_crossing_angle_is_rejected():
    snapshots = _snapshots(object_heading=math.radians(10.0))
    assert interaction._crosses_find_events(snapshots) == []


def test_arrival_order_makes_relation_directional():
    # O moves north and reaches the conflict before S.  Therefore S->O must
    # not be emitted; the reverse direction is the only possible one.
    snapshots = _snapshots(object_velocity=(0.0, 2.0))
    events = interaction._crosses_find_events(snapshots)
    assert events
    assert all(
        not (e["subject_token"] == "S" and e["object_token"] == "O")
        for e in events
    )


def test_pedestrian_pedestrian_is_disabled_by_default():
    snapshots = _snapshots(object_type="PEDESTRIAN")
    assert interaction._crosses_find_events(snapshots) == []


def test_assertions_exist_only_on_frames_where_10m_rays_intersect():
    snapshots = _snapshots(object_x=11.0)
    previous = interaction.ARGS.derive_semantic_predicates
    interaction.ARGS.derive_semantic_predicates = True
    try:
        assertions = interaction.derive_crosses_in_front_predicates(snapshots)
    finally:
        interaction.ARGS.derive_semantic_predicates = previous

    assert [row.end_time_us for row in assertions] == [500_000, 1_000_000, 1_500_000, 2_000_000]
    assert all(row.valid_time_us == 500_000 for row in assertions)
    for row in assertions:
        assert row.evidence["ray_length_m"] == 10.0
        assert row.evidence["subject_ray_distance_m"] <= 10.0
        assert row.evidence["object_ray_distance_m"] <= 10.0


def test_motor_vehicle_in_carpark_is_rejected_before_emission():
    snapshots = _snapshots(subject_type="VEHICLE")
    tracks = interaction._crosses_build_tracks(snapshots)
    subject_item = tracks["S"][0]
    object_item = tracks["O"][0]

    original = interaction._crosses_map_context

    def fake_context(item):
        token = item["record"]["track_token"]
        return {
            "available": True,
            "road_distance_m": 0.0,
            "lane_distance_m": 0.0,
            "intersection_distance_m": 0.0,
            "crosswalk_distance_m": 0.0,
            "carpark_distance_m": 0.0 if token == "S" else None,
            "carpark_contains_center": token == "S",
            "carpark_footprint_overlap": token == "S",
            "carpark_footprint_overlap_area_m2": 1.0 if token == "S" else 0.0,
            "carpark_area_ids": ["park_1"] if token == "S" else [],
        }

    interaction._crosses_map_context = fake_context
    try:
        result = interaction._crosses_pair_map_filter(
            tracks["S"], tracks["O"], subject_item, object_item
        )
    finally:
        interaction._crosses_map_context = original

    assert result["accepted"] is False
    assert result["reason"] == "subject_in_carpark_area"
