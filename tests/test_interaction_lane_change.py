"""Synthetic tests of the vehicle-to-target-lane interaction predicate."""
from types import SimpleNamespace

from test_interaction_follow import (
    DirectionEvidence,
    Frame,
    MapAPI,
    MapObject,
    load_interaction_module,
)


def _lane_change_record(token, x, y, lane, *, ego=False):
    return {
        "entity_id": "ego" if ego else f"entity:{token}",
        "track_token": token,
        "agent_type": "EGO" if ego else "VEHICLE",
        "xyh": (float(x), float(y), 0.0),
        "velocity": (8.0, 0.0),
        "speed": 8.0,
        "dimensions": {"length_m": 4.0, "width_m": 2.0},
        "map_match_ambiguous": False,
        "lane_ids": {f"lane:{lane.id}"},
        "footprint_lane_ids": {f"lane:{lane.id}"},
        "primary_map_object_id": lane.id,
        "primary_lane_id": lane.id,
        "primary_map_kind": "lane",
        "primary_map_object": lane,
        "baseline_length_m": lane.length,
        "map_heading_rad": 0.0,
        "travel_direction": DirectionEvidence(heading_rad=0.0),
        "left_adjacent_object_ids": set(lane.left_adjacent_ids),
        "right_adjacent_object_ids": set(lane.right_adjacent_ids),
        "map_candidates": [],
    }


def _history(source, target, *, ego=False):
    api = MapAPI([source, target])
    y_values = (0.0, 0.0, 0.4, 1.5, 2.5, 3.6, 4.0, 4.0, 4.0)
    snapshots = []
    for frame_index, y in enumerate(y_values):
        lane = source if frame_index <= 3 else target
        record = _lane_change_record("ego" if ego else "vehicle", 10.0 + frame_index * 4.0, y, lane, ego=ego)
        frame = Frame(frame_index * 500_000, api)
        snapshots.append({
            "frame": frame,
            "timestamp_us": frame.timestamp_us,
            "frame_index": frame_index,
            "scenario_token": "lane-change-scenario",
            "entities": {record["track_token"]: record},
        })
    return snapshots


def test_emits_binary_vehicle_to_target_lane_relation_with_dynamic_subject_arrow():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)

    assertions = module.derive_lane_change_predicates(_history(source, target))

    assert assertions
    assert {item.predicate_id for item in assertions} == {"np:changesLane"}
    assert {item.subject_id for item in assertions} == {"entity:vehicle"}
    assert {item.object_id for item in assertions} == {"lane:L1"}
    assert {item.evidence["source_lane_entity_id"] for item in assertions} == {"lane:L0"}
    assert {item.evidence["target_lane_entity_id"] for item in assertions} == {"lane:L1"}
    assert {item.evidence["lane_change_side"] for item in assertions} == {"left"}
    assert {item.evidence["interaction_kind"] for item in assertions} == {"vehicle_to_structure"}
    assert all(item.evidence["has_dual_lane_overlap"] for item in assertions)

    arrow_endpoints = {
        (
            item.evidence["lane_change_arrow_end_x"],
            item.evidence["lane_change_arrow_end_y"],
        )
        for item in assertions
    }
    assert len(arrow_endpoints) == 1
    end_x, end_y = next(iter(arrow_endpoints))
    assert end_y == 4.0
    for item in assertions:
        assert item.evidence["lane_change_arrow_start_x"] == item.evidence[
            "lane_change_current_subject_x"
        ]
        assert item.evidence["lane_change_arrow_start_y"] == item.evidence[
            "lane_change_current_subject_y"
        ]
        assert end_x >= item.evidence["lane_change_arrow_start_x"]
    assert assertions[0].valid_time_us == assertions[0].evidence["start_time_us"]
    assert assertions[-1].end_time_us == assertions[-1].evidence["completion_time_us"]
    assert assertions[-1].evidence["lane_change_completed"] is True


def test_supports_ego_to_lane_structure_relation():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)

    assertions = module.derive_lane_change_predicates(_history(source, target, ego=True))

    assert assertions
    assert {item.subject_id for item in assertions} == {"ego"}
    assert {item.object_id for item in assertions} == {"lane:L1"}


def test_rejects_transition_between_non_adjacent_lanes():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, center_y=0.0)
    target = MapObject("L1", 0.0, center_y=8.0)

    assertions = module.derive_lane_change_predicates(_history(source, target))

    assert assertions == []


def test_accepts_conservative_geometric_adjacency_fallback():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, center_y=0.0)
    target = MapObject("L1", 0.0, center_y=4.0)

    assertions = module.derive_lane_change_predicates(_history(source, target))

    assert assertions
    assert {
        item.evidence["adjacency_source"] for item in assertions
    } == {"local_geometric_fallback"}


def _custom_lane_change_history(lanes, samples, *, token="vehicle"):
    """Build a synthetic track from (x, y, lane_id) samples."""
    api = MapAPI(list(lanes.values()))
    result = []
    for frame_index, (x, y, lane_id) in enumerate(samples):
        lane = lanes[lane_id]
        record = _lane_change_record(token, x, y, lane)
        frame = Frame(frame_index * 500_000, api)
        result.append({
            "frame": frame,
            "timestamp_us": frame.timestamp_us,
            "frame_index": frame_index,
            "scenario_token": "lane-change-regression",
            "entities": {token: record},
        })
    return result


def _event_starts(assertions):
    return sorted({
        (
            item.evidence["source_lane_id"],
            item.evidence["target_lane_id"],
            item.evidence["maneuver_onset_frame_index"],
            item.evidence["completion_frame_index"],
        )
        for item in assertions
    })


def test_detects_departure_and_return_lane_changes_during_overtake_shape():
    """One outward and one return transition must produce two events."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    passing = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    samples = [
        (0.0, 0.0, "L0"),
        (4.0, 0.0, "L0"),
        (8.0, 0.15, "L0"),
        (12.0, 0.60, "L0"),
        (16.0, 2.00, "L0"),
        (20.0, 3.50, "L1"),
        (24.0, 4.00, "L1"),
        (28.0, 4.00, "L1"),
        (32.0, 4.00, "L1"),
        (36.0, 3.80, "L1"),
        (40.0, 3.20, "L1"),
        (44.0, 2.00, "L0"),
        (48.0, 0.50, "L0"),
        (52.0, 0.00, "L0"),
        (56.0, 0.00, "L0"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history({"L0": source, "L1": passing}, samples)
    )

    events = _event_starts(assertions)
    assert len(events) == 2
    assert events[0][:2] == ("L0", "L1")
    assert events[1][:2] == ("L1", "L0")
    # Activation is backdated to the first sustained lateral departure,
    # before the primary lane assignment changes.
    assert events[0][2] <= 2
    assert events[1][2] <= 9


def test_accepts_sparse_sampled_crossing_without_dual_lane_overlap():
    """A real adjacent-lane assignment transition must not require a sampled straddle frame."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    samples = [
        (0.0, 0.0, "L0"),
        (4.0, 0.0, "L0"),
        (8.0, 0.0, "L0"),
        (12.0, 4.0, "L1"),
        (16.0, 4.0, "L1"),
        (20.0, 4.0, "L1"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history({"L0": source, "L1": target}, samples)
    )

    assert assertions
    assert all(not item.evidence["has_dual_lane_overlap"] for item in assertions)
    assert all(
        item.evidence["has_confirmed_lane_assignment_transition"]
        for item in assertions
    )
    assert {
        item.evidence["boundary_evidence_type"] for item in assertions
    } == {"decoded_topology_constrained_corridor_transition"}


def test_longitudinal_lane_tokens_are_collapsed_before_lateral_detection():
    """A0→A1 is map segmentation, while A1→B1 is the actual lane change."""
    module = load_interaction_module()
    source_0 = MapObject("A0", 0.0, length=20.0, outgoing=("A1",), center_y=0.0)
    source_1 = MapObject(
        "A1", 20.0, length=20.0, left_adjacent=("B1",), center_y=0.0
    )
    target_1 = MapObject(
        "B1", 20.0, length=20.0, outgoing=("B2",),
        right_adjacent=("A1",), center_y=4.0,
    )
    target_2 = MapObject("B2", 40.0, length=20.0, center_y=4.0)
    lanes = {lane.id: lane for lane in (source_0, source_1, target_1, target_2)}
    samples = [
        (8.0, 0.0, "A0"),
        (16.0, 0.0, "A0"),
        (23.0, 0.15, "A1"),
        (28.0, 0.70, "A1"),
        (33.0, 2.00, "A1"),
        (37.0, 3.60, "B1"),
        (42.0, 4.00, "B2"),
        (48.0, 4.00, "B2"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history(lanes, samples)
    )

    assert assertions
    assert {item.evidence["source_lane_id"] for item in assertions} == {"A1"}
    assert {item.evidence["target_entry_lane_id"] for item in assertions} == {"B1"}
    assert {item.evidence["target_lane_id"] for item in assertions} <= {"B1", "B2"}
    assert all(
        item.evidence["longitudinal_lane_successors_collapsed"]
        for item in assertions
    )
    assert min(
        item.evidence["maneuver_onset_frame_index"] for item in assertions
    ) <= 2


def _record_with_lane_candidates(
    token,
    x,
    y,
    primary_lane,
    candidates,
    *,
    ego=False,
):
    """Build a record whose primary lane may lag behind geometric occupancy."""
    record = _lane_change_record(token, x, y, primary_lane, ego=ego)
    record["map_candidates"] = [
        {
            "kind": "lane",
            "object_id": lane.id,
            "map_object": lane,
            "overlap_ratio": float(overlap),
            "center_covered": bool(center),
        }
        for lane, overlap, center in candidates
    ]
    return record


def _candidate_history(lanes, samples, *, token="vehicle", ego=False):
    """Build snapshots from explicit candidate-lane occupancy evidence."""
    api = MapAPI(list(lanes.values()))
    result = []
    for frame_index, sample in enumerate(samples):
        x, y, primary_lane_id, candidate_rows = sample
        record = _record_with_lane_candidates(
            token,
            x,
            y,
            lanes[primary_lane_id],
            [
                (lanes[lane_id], overlap, center)
                for lane_id, overlap, center in candidate_rows
            ],
            ego=ego,
        )
        frame = Frame(frame_index * 500_000, api)
        result.append({
            "frame": frame,
            "timestamp_us": frame.timestamp_us,
            "frame_index": frame_index,
            "scenario_token": "candidate-lane-state-machine",
            "entities": {record["track_token"]: record},
        })
    return result


def test_detects_two_changes_when_primary_lane_assignment_lags_geometry():
    """Persistent primary switches confirm both events; geometry backdates onset."""
    module = load_interaction_module()
    original = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    passing = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    lanes = {"L0": original, "L1": passing}
    samples = [
        (0.0, 0.0, "L0", [("L0", 1.00, True)]),
        (4.0, 0.0, "L0", [("L0", 1.00, True)]),
        (8.0, 0.2, "L0", [("L0", 0.95, True), ("L1", 0.05, False)]),
        (12.0, 1.0, "L0", [("L0", 0.75, True), ("L1", 0.25, False)]),
        (16.0, 2.0, "L0", [("L0", 0.50, True), ("L1", 0.50, True)]),
        # Primary token is deliberately late: geometry already supports L1.
        (20.0, 3.5, "L0", [("L0", 0.25, False), ("L1", 0.75, True)]),
        (24.0, 4.0, "L1", [("L1", 1.00, True)]),
        (28.0, 4.0, "L1", [("L1", 1.00, True)]),
        (32.0, 4.0, "L1", [("L1", 1.00, True)]),
        (36.0, 3.5, "L1", [("L1", 0.75, True), ("L0", 0.25, False)]),
        (40.0, 2.0, "L1", [("L1", 0.50, True), ("L0", 0.50, True)]),
        # Return geometry supports L0 before the primary token changes back.
        (44.0, 0.5, "L1", [("L1", 0.25, False), ("L0", 0.75, True)]),
        (48.0, 0.0, "L0", [("L0", 1.00, True)]),
        (52.0, 0.0, "L0", [("L0", 1.00, True)]),
    ]

    assertions = module.derive_lane_change_predicates(
        _candidate_history(lanes, samples)
    )
    events = _event_starts(assertions)

    assert len(events) == 2
    assert events[0][:2] == ("L0", "L1")
    assert events[1][:2] == ("L1", "L0")
    assert events[0][2] <= 3
    assert events[1][2] <= 9
    assert {
        item.evidence["lane_change_sequence_index"] for item in assertions
    } == {1, 2}
    assert all(
        item.evidence["multiple_lane_changes_per_subject_supported"]
        for item in assertions
    )


def test_detects_two_lane_changes_for_every_vehicle_like_subject():
    """The state machine must run independently for ego and other vehicles."""
    module = load_interaction_module()
    original = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    passing = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    samples = [
        (0.0, 0.0, "L0"),
        (4.0, 0.0, "L0"),
        (8.0, 0.2, "L0"),
        (12.0, 1.0, "L0"),
        (16.0, 2.0, "L0"),
        (20.0, 3.6, "L1"),
        (24.0, 4.0, "L1"),
        (28.0, 4.0, "L1"),
        (32.0, 3.8, "L1"),
        (36.0, 3.0, "L1"),
        (40.0, 2.0, "L0"),
        (44.0, 0.4, "L0"),
        (48.0, 0.0, "L0"),
        (52.0, 0.0, "L0"),
    ]
    lanes = {"L0": original, "L1": passing}
    agent_history = _custom_lane_change_history(lanes, samples, token="agent")
    ego_history = _custom_lane_change_history(lanes, samples, token="ego")
    combined = []
    for agent_snapshot, ego_snapshot in zip(agent_history, ego_history):
        ego_record = ego_snapshot["entities"]["ego"]
        ego_record["entity_id"] = "ego"
        ego_record["agent_type"] = "EGO"
        combined.append({
            **agent_snapshot,
            "entities": {
                "agent": agent_snapshot["entities"]["agent"],
                "ego": ego_record,
            },
        })

    assertions = module.derive_lane_change_predicates(combined)
    events_by_subject = {}
    for item in assertions:
        events_by_subject.setdefault(item.subject_id, set()).add(
            item.evidence["event_id"]
        )

    assert set(events_by_subject) == {"entity:agent", "ego"}
    assert all(len(event_ids) == 2 for event_ids in events_by_subject.values())


def test_nonadjacent_map_glitch_does_not_hide_later_real_lane_change():
    """A brief strong but nonadjacent state must be ignored, not adopted."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    glitch = MapObject("LX", 0.0, center_y=12.0)
    lanes = {"L0": source, "L1": target, "LX": glitch}
    samples = [
        (0.0, 0.0, "L0"),
        (4.0, 0.0, "L0"),
        (8.0, 12.0, "LX"),
        (12.0, 12.0, "LX"),
        (16.0, 1.0, "L0"),
        (20.0, 2.0, "L0"),
        (24.0, 3.6, "L1"),
        (28.0, 4.0, "L1"),
        (32.0, 4.0, "L1"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history(lanes, samples)
    )
    events = _event_starts(assertions)

    assert len(events) == 1
    assert events[0][:2] == ("L0", "L1")




def test_one_frame_primary_lane_flicker_is_rejected():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    samples = [
        (0.0, 0.0, "L0"),
        (4.0, 0.0, "L0"),
        (8.0, 2.0, "L1"),
        (12.0, 0.0, "L0"),
        (16.0, 0.0, "L0"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history({"L0": source, "L1": target}, samples)
    )

    assert assertions == []

def test_strong_target_occupancy_can_confirm_change_without_primary_switch():
    """Stable adjacent target occupancy must recover a delayed/missing primary switch."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    lanes = {"L0": source, "L1": target}
    samples = [
        (0.0, 0.0, "L0", [("L0", 1.00, True)]),
        (4.0, 0.2, "L0", [("L0", 0.95, True), ("L1", 0.05, False)]),
        (8.0, 1.2, "L0", [("L0", 0.70, True), ("L1", 0.30, False)]),
        (12.0, 2.0, "L0", [("L0", 0.50, True), ("L1", 0.50, True)]),
        (16.0, 3.4, "L0", [("L0", 0.20, False), ("L1", 0.80, True)]),
        (20.0, 4.0, "L0", [("L0", 0.05, False), ("L1", 0.95, True)]),
    ]

    assertions = module.derive_lane_change_predicates(
        _candidate_history(lanes, samples)
    )

    assert assertions
    assert {item.evidence["target_lane_id"] for item in assertions} == {"L1"}
    assert all(not item.evidence["has_confirmed_lane_assignment_transition"] for item in assertions)
    assert all(item.evidence["event_confidence_status"] == "confirmed" for item in assertions)


def test_offline_decoder_combines_primary_and_geometric_evidence():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    assertions = module.derive_lane_change_predicates(_history(source, target))

    assert assertions
    assert {item.evidence["lane_change_strategy"] for item in assertions} == {
        "offline_drivable_primitive_path_transfer_decoder_v9_5_25"
    }
    assert all(not item.evidence["primary_lane_is_event_confirmation"] for item in assertions)
    assert all(item.evidence["decoded_maneuver_state"] == "CHANGING_LEFT" for item in assertions)
    assert {item.evidence["boundary_frame_index"] for item in assertions} == {4}
    assert {item.evidence["completion_frame_index"] for item in assertions} == {5}

def test_outward_and_return_changes_survive_longitudinal_lane_segmentation():
    """A0/A1 and B0/B1 form two corridors and must yield two transitions."""
    module = load_interaction_module()
    a0 = MapObject("A0", 0.0, length=24.0, outgoing=("A1",), left_adjacent=("B0",), center_y=0.0)
    a1 = MapObject("A1", 24.0, length=40.0, left_adjacent=("B1",), center_y=0.0)
    b0 = MapObject("B0", 0.0, length=24.0, outgoing=("B1",), right_adjacent=("A0",), center_y=4.0)
    b1 = MapObject("B1", 24.0, length=40.0, right_adjacent=("A1",), center_y=4.0)
    lanes = {lane.id: lane for lane in (a0, a1, b0, b1)}
    samples = [
        (4.0, 0.0, "A0"),
        (12.0, 0.0, "A0"),
        (20.0, 0.4, "A0"),
        (26.0, 1.2, "A1"),
        (32.0, 2.2, "A1"),
        (38.0, 3.7, "B1"),
        (44.0, 4.0, "B1"),
        (50.0, 4.0, "B1"),
        (56.0, 3.6, "B1"),
        (60.0, 2.2, "B1"),
        (62.0, 0.8, "A1"),
        (63.0, 0.0, "A1"),
        (63.5, 0.0, "A1"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history(lanes, samples)
    )
    event_pairs = {
        (
            item.evidence["source_corridor_id"],
            item.evidence["target_corridor_id"],
        )
        for item in assertions
    }

    assert len({item.evidence["event_id"] for item in assertions}) == 2
    assert len(event_pairs) == 2
    assert all(
        item.evidence["longitudinal_lane_successors_collapsed"]
        for item in assertions
    )


def test_suppresses_first_change_without_observed_pre_onset_history():
    """A maneuver already underway at track start is left-censored, not confirmed."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    samples = [
        (0.0, 0.8, "L0"),
        (4.0, 2.6, "L1"),
        (8.0, 3.8, "L1"),
        (12.0, 4.0, "L1"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history({"L0": source, "L1": target}, samples)
    )

    assert assertions == []


def test_primary_lane_id_is_used_during_temporary_non_lane_map_kind():
    """A connector/ambiguous map-kind label must not hide an explicit lane token."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    history = _history(source, target)
    history[2]["entities"]["vehicle"]["primary_map_kind"] = "lane_connector"
    history[3]["entities"]["vehicle"]["primary_map_kind"] = "lane_connector"

    assertions = module.derive_lane_change_predicates(history)

    assert assertions
    assert {item.evidence["target_lane_id"] for item in assertions} == {"L1"}


def test_track_gap_inside_lane_change_does_not_cancel_event():
    """A temporarily missing agent frame must not delete the whole maneuver."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    history = _history(source, target)
    history[3]["entities"] = {}

    assertions = module.derive_lane_change_predicates(history)

    assert assertions
    frames = sorted({item.evidence["lane_change_current_frame_index"] for item in assertions})
    assert 3 not in frames
    assert frames[-1] >= 5
    assert all(item.evidence["track_gaps_tolerated"] for item in assertions)


def test_left_censored_first_change_is_not_exported_as_confirmed_predicate():
    """Track-start overlap decay can be diagnostic, but no normal predicate is emitted."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    lanes = {"L0": source, "L1": target}
    samples = [
        (0.0, 2.2, "L1", [("L0", 0.45, True), ("L1", 0.55, True)]),
        (4.0, 3.2, "L1", [("L0", 0.20, False), ("L1", 0.80, True)]),
        (8.0, 4.0, "L1", [("L1", 1.00, True)]),
        (12.0, 4.0, "L1", [("L1", 1.00, True)]),
    ]

    assertions = module.derive_lane_change_predicates(
        _candidate_history(lanes, samples)
    )

    assert assertions == []


def test_overtake_shape_detects_two_changes_when_first_source_is_short():
    """The outward event and return event must both survive asymmetric evidence."""
    module = load_interaction_module()
    original = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    passing = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    samples = [
        (0.0, 0.6, "L0"),
        (4.0, 2.4, "L1"),
        (8.0, 3.8, "L1"),
        (12.0, 4.0, "L1"),
        (16.0, 4.0, "L1"),
        (20.0, 3.2, "L1"),
        (24.0, 1.8, "L0"),
        (28.0, 0.3, "L0"),
        (32.0, 0.0, "L0"),
    ]

    assertions = module.derive_lane_change_predicates(
        _custom_lane_change_history({"L0": original, "L1": passing}, samples)
    )
    events = {
        (
            item.evidence["source_lane_id"],
            item.evidence["target_lane_id"],
            item.evidence["event_id"],
        )
        for item in assertions
    }

    # The outward transition is already underway at the observation boundary
    # and is therefore left-censored. The later return has sufficient history.
    assert len({event_id for _, _, event_id in events}) == 1
    assert {(source, target) for source, target, _ in events} == {("L1", "L0")}


def test_decoder_rejects_aborted_within_lane_lateral_drift():
    """A multi-frame drift that never establishes the adjacent lane is not a lane change."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    lanes = {"L0": source, "L1": target}
    samples = [
        (0.0, 0.0, "L0", [("L0", 1.00, True)]),
        (4.0, 0.4, "L0", [("L0", 0.90, True), ("L1", 0.10, False)]),
        (8.0, 1.0, "L0", [("L0", 0.75, True), ("L1", 0.25, False)]),
        (12.0, 1.4, "L0", [("L0", 0.65, True), ("L1", 0.35, False)]),
        (16.0, 0.8, "L0", [("L0", 0.80, True), ("L1", 0.20, False)]),
        (20.0, 0.0, "L0", [("L0", 1.00, True)]),
    ]

    assertions = module.derive_lane_change_predicates(_candidate_history(lanes, samples))

    assert assertions == []


def test_decoder_exports_confidence_and_hidden_state_evidence():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)

    assertions = module.derive_lane_change_predicates(_history(source, target))

    assert assertions
    assert {item.evidence["decoded_maneuver_state"] for item in assertions} == {
        "CHANGING_LEFT"
    }
    assert {item.evidence["event_confidence_status"] for item in assertions} <= {
        "confirmed", "probable", "censored"
    }
    assert all(item.evidence["event_confidence_score"] >= 0.0 for item in assertions)
    assert all(
        item.evidence["decoded_lane_state_model"]
        == "KEEPING_LANE|CHANGING_LEFT|CHANGING_RIGHT|UNKNOWN"
        for item in assertions
    )


def test_decoder_retains_right_censored_lane_change_at_track_end():
    """A track ending during target entry is retained as censored, not dropped."""
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    lanes = {"L0": source, "L1": target}
    samples = [
        (0.0, 0.0, "L0", [("L0", 1.00, True)]),
        (4.0, 0.2, "L0", [("L0", 0.95, True), ("L1", 0.05, False)]),
        (8.0, 1.2, "L0", [("L0", 0.70, True), ("L1", 0.30, False)]),
        (12.0, 2.4, "L0", [("L0", 0.40, False), ("L1", 0.60, True)]),
    ]

    assertions = module.derive_lane_change_predicates(_candidate_history(lanes, samples))

    assert assertions
    assert {item.evidence["target_lane_id"] for item in assertions} == {"L1"}
    assert all(item.evidence["completion_censored"] for item in assertions)
    assert {item.evidence["event_confidence_status"] for item in assertions} == {
        "censored"
    }


def _primitive_record(token, x, y, primitive, kind, *, primary=True):
    prefix = "lane_connector" if kind == "lane_connector" else "lane"
    record = {
        "entity_id": f"entity:{token}",
        "track_token": token,
        "agent_type": "VEHICLE",
        "xyh": (float(x), float(y), 0.0),
        "velocity": (8.0, 0.0),
        "speed": 8.0,
        "dimensions": {"length_m": 4.0, "width_m": 2.0},
        "map_match_ambiguous": False,
        "lane_ids": set(),
        "footprint_lane_ids": set(),
        "primary_map_object_id": primitive.id,
        "primary_lane_id": primitive.id if kind == "lane" else None,
        "primary_connector_id": primitive.id if kind == "lane_connector" else None,
        "primary_map_kind": kind,
        "primary_map_object": primitive,
        "baseline_length_m": primitive.length,
        "map_heading_rad": 0.0,
        "travel_direction": DirectionEvidence(heading_rad=0.0),
        "left_adjacent_object_ids": set(primitive.left_adjacent_ids) if kind == "lane" else set(),
        "right_adjacent_object_ids": set(primitive.right_adjacent_ids) if kind == "lane" else set(),
        "map_candidates": [{
            "kind": kind,
            "object_id": primitive.id,
            "entity_id": f"{prefix}:{primitive.id}",
            "map_object": primitive,
            "overlap_ratio": 1.0,
            "center_covered": True,
        }],
    }
    if kind == "lane":
        record["lane_ids"] = {f"lane:{primitive.id}"}
        record["footprint_lane_ids"] = {f"lane:{primitive.id}"}
    return record


def _primitive_history(primitives, samples, *, token="vehicle"):
    api = MapAPI([row[0] for row in primitives.values()])
    result = []
    for frame_index, (x, y, key) in enumerate(samples):
        primitive, kind = primitives[key]
        record = _primitive_record(token, x, y, primitive, kind)
        frame = Frame(frame_index * 500_000, api)
        result.append({
            "frame": frame,
            "timestamp_us": frame.timestamp_us,
            "frame_index": frame_index,
            "scenario_token": "primitive-lane-change-regression",
            "entities": {token: record},
        })
    return result


def test_lane_to_connector_to_lane_directed_continuation_is_not_lane_change():
    module = load_interaction_module()
    lane0 = MapObject("L0", 0.0, length=20.0, outgoing=("C0",), center_y=0.0)
    connector0 = MapObject("C0", 20.0, length=20.0, outgoing=("L1",), center_y=0.0)
    lane1 = MapObject("L1", 40.0, length=20.0, center_y=0.0)
    primitives = {
        "L0": (lane0, "lane"),
        "C0": (connector0, "lane_connector"),
        "L1": (lane1, "lane"),
    }
    samples = [
        (5.0, 0.0, "L0"), (15.0, 0.0, "L0"),
        (25.0, 0.0, "C0"), (35.0, 0.0, "C0"),
        (45.0, 0.0, "L1"), (55.0, 0.0, "L1"),
    ]
    assertions = module.derive_lane_change_predicates(
        _primitive_history(primitives, samples)
    )
    assert assertions == []


def test_parallel_lane_connector_transfer_is_detected_as_lane_change():
    module = load_interaction_module()
    connector0 = MapObject("C0", 0.0, length=80.0, center_y=0.0)
    connector1 = MapObject("C1", 0.0, length=80.0, center_y=4.0)
    primitives = {
        "C0": (connector0, "lane_connector"),
        "C1": (connector1, "lane_connector"),
    }
    samples = [
        (5.0, 0.0, "C0"), (10.0, 0.0, "C0"),
        (15.0, 0.4, "C0"), (20.0, 1.4, "C0"),
        (25.0, 2.8, "C1"), (30.0, 4.0, "C1"),
        (35.0, 4.0, "C1"), (40.0, 4.0, "C1"),
    ]
    assertions = module.derive_lane_change_predicates(
        _primitive_history(primitives, samples)
    )
    assert assertions
    assert {item.object_id for item in assertions} == {"lane_connector:C1"}
    assert {item.evidence["source_map_kind"] for item in assertions} == {"lane_connector"}
    assert {item.evidence["target_map_kind"] for item in assertions} == {"lane_connector"}
    assert {item.evidence["source_lane_id"] for item in assertions} == {"C0"}
    assert {item.evidence["target_lane_id"] for item in assertions} == {"C1"}
    assert all(item.evidence["lateral_lane_connector_transfer_supported"] for item in assertions)


def test_branched_lane_connectors_are_not_globally_collapsed_and_lateral_transfer_survives():
    module = load_interaction_module()
    lane0 = MapObject("L0", 0.0, length=20.0, outgoing=("C0", "C1"), center_y=0.0)
    connector0 = MapObject("C0", 20.0, length=80.0, center_y=0.0)
    connector1 = MapObject("C1", 20.0, length=80.0, center_y=4.0)
    primitives = {
        "L0": (lane0, "lane"),
        "C0": (connector0, "lane_connector"),
        "C1": (connector1, "lane_connector"),
    }
    samples = [
        (5.0, 0.0, "L0"), (15.0, 0.0, "L0"),
        (25.0, 0.0, "C0"), (30.0, 0.0, "C0"),
        (35.0, 0.4, "C0"), (40.0, 1.5, "C0"),
        (45.0, 2.8, "C1"), (50.0, 4.0, "C1"),
        (55.0, 4.0, "C1"),
    ]
    assertions = module.derive_lane_change_predicates(
        _primitive_history(primitives, samples)
    )
    assert assertions
    assert {item.object_id for item in assertions} == {"lane_connector:C1"}
    # L0 -> C0 is a directed topology continuation and must not create an event.
    assert all(item.evidence["source_lane_id"] == "C0" for item in assertions)


def test_lane_to_lateral_lane_connector_transfer_is_detected():
    module = load_interaction_module()
    lane0 = MapObject("L0", 0.0, length=80.0, center_y=0.0)
    connector1 = MapObject("C1", 0.0, length=80.0, center_y=4.0)
    primitives = {"L0": (lane0, "lane"), "C1": (connector1, "lane_connector")}
    samples = [
        (5.0, 0.0, "L0"), (10.0, 0.0, "L0"),
        (15.0, 0.4, "L0"), (20.0, 1.4, "L0"),
        (25.0, 2.8, "C1"), (30.0, 3.6, "C1"), (35.0, 4.0, "C1"),
    ]
    assertions = module.derive_lane_change_predicates(_primitive_history(primitives, samples))
    assert assertions
    assert {item.object_id for item in assertions} == {"lane_connector:C1"}
    assert {item.evidence["source_map_kind"] for item in assertions} == {"lane"}
    assert {item.evidence["target_map_kind"] for item in assertions} == {"lane_connector"}


def test_lane_connector_to_lateral_lane_transfer_is_detected():
    module = load_interaction_module()
    connector0 = MapObject("C0", 0.0, length=80.0, center_y=0.0)
    lane1 = MapObject("L1", 0.0, length=80.0, center_y=4.0)
    primitives = {"C0": (connector0, "lane_connector"), "L1": (lane1, "lane")}
    samples = [
        (5.0, 0.0, "C0"), (10.0, 0.0, "C0"),
        (15.0, 0.4, "C0"), (20.0, 1.4, "C0"),
        (25.0, 2.8, "L1"), (30.0, 3.6, "L1"), (35.0, 4.0, "L1"),
    ]
    assertions = module.derive_lane_change_predicates(_primitive_history(primitives, samples))
    assert assertions
    assert {item.object_id for item in assertions} == {"lane:L1"}
    assert {item.evidence["source_map_kind"] for item in assertions} == {"lane_connector"}
    assert {item.evidence["target_map_kind"] for item in assertions} == {"lane"}
