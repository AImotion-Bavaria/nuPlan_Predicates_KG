"""Synthetic tests for vehicle-to-vehicle merge interactions."""
from test_interaction_follow import Frame, MapAPI, MapObject, load_interaction_module
from test_interaction_lane_change import _primitive_record


def _merge_history(*, object_offset_m: float, target_kind: str = "lane", second_object_offset_m=None):
    source = MapObject("S0", 0.0, length=100.0, center_y=0.0)
    target = MapObject("T0", 0.0, length=100.0, center_y=4.0)
    api = MapAPI([source, target])
    samples = [
        (5.0, 0.0, source, "lane"),
        (10.0, 0.0, source, "lane"),
        (15.0, 0.4, source, "lane"),
        (20.0, 1.4, source, "lane"),
        (25.0, 2.8, target, target_kind),
        (30.0, 4.0, target, target_kind),
        (35.0, 4.0, target, target_kind),
        (40.0, 4.0, target, target_kind),
    ]
    snapshots = []
    for frame_index, (sx, sy, primitive, kind) in enumerate(samples):
        subject = _primitive_record("subject", sx, sy, primitive, kind)
        obj_x = sx + float(object_offset_m)
        obj = _primitive_record("object", obj_x, 4.0, target, target_kind)
        entities = {"subject": subject, "object": obj}
        if second_object_offset_m is not None:
            obj2 = _primitive_record(
                "object2", sx + float(second_object_offset_m), 4.0, target, target_kind
            )
            entities["object2"] = obj2
        frame = Frame(frame_index * 500_000, api)
        snapshots.append({
            "frame": frame,
            "timestamp_us": frame.timestamp_us,
            "frame_index": frame_index,
            "scenario_token": "merge-scenario",
            "entities": entities,
        })
    return snapshots


def _merge_predicates(module, snapshots):
    lane_changes = module.derive_lane_change_predicates(snapshots)
    return lane_changes, module.derive_merging_predicates(
        snapshots, lane_change_assertions=lane_changes
    )


def test_merge_in_front_of_vehicle_already_in_target_stream():
    module = load_interaction_module()
    lane_changes, merges = _merge_predicates(module, _merge_history(object_offset_m=-8.0))
    assert lane_changes
    assert merges
    assert {row.predicate_id for row in merges} == {"np:mergesInFrontOf"}
    assert {row.subject_id for row in merges} == {"entity:subject"}
    assert {row.object_id for row in merges} == {"entity:object"}
    assert all(row.evidence["subject_final_order"] == "subject_ahead" for row in merges)
    assert all(row.evidence["other_vehicle_preexists_in_target_stream"] for row in merges)


def test_merge_behind_vehicle_already_in_target_stream():
    module = load_interaction_module()
    lane_changes, merges = _merge_predicates(module, _merge_history(object_offset_m=10.0))
    assert lane_changes
    assert merges
    assert {row.predicate_id for row in merges} == {"np:mergesBehind"}
    assert all(row.evidence["subject_final_order"] == "subject_behind" for row in merges)


def test_entering_gap_emits_nearest_front_and_rear_relations():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=-7.0, second_object_offset_m=12.0)
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert {row.predicate_id for row in merges} == {
        "np:mergesInFrontOf", "np:mergesBehind"
    }
    event_signatures = {
        (row.predicate_id, row.object_id, row.evidence["merge_event_id"])
        for row in merges
    }
    assert any(pid == "np:mergesInFrontOf" and obj == "entity:object" for pid, obj, _ in event_signatures)
    assert any(pid == "np:mergesBehind" and obj == "entity:object2" for pid, obj, _ in event_signatures)


def test_does_not_merge_with_vehicle_that_appears_only_after_boundary():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=-8.0)
    # Remove the object until after the subject has entered the target stream.
    for snapshot in snapshots[:5]:
        snapshot["entities"].pop("object", None)
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert merges == []


def test_connector_target_supports_merge_in_front():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=-8.0, target_kind="lane_connector")
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert merges
    assert {row.predicate_id for row in merges} == {"np:mergesInFrontOf"}
    assert {row.evidence["target_map_kind"] for row in merges} == {"lane_connector"}
    assert {row.evidence["target_map_entity_id"] for row in merges} == {"lane_connector:T0"}


def test_nearest_rear_neighbor_only_is_selected():
    module = load_interaction_module()
    source = MapObject("S0", 0.0, length=100.0, center_y=0.0)
    target = MapObject("T0", 0.0, length=100.0, center_y=4.0)
    api = MapAPI([source, target])
    samples = [
        (5.0, 0.0, source, "lane"), (10.0, 0.0, source, "lane"),
        (15.0, 0.4, source, "lane"), (20.0, 1.4, source, "lane"),
        (25.0, 2.8, target, "lane"), (30.0, 4.0, target, "lane"),
        (35.0, 4.0, target, "lane"), (40.0, 4.0, target, "lane"),
    ]
    snapshots = []
    for i, (sx, sy, primitive, kind) in enumerate(samples):
        subject = _primitive_record("subject", sx, sy, primitive, kind)
        near = _primitive_record("near_rear", sx - 7.0, 4.0, target, "lane")
        far = _primitive_record("far_rear", sx - 20.0, 4.0, target, "lane")
        frame = Frame(i * 500_000, api)
        snapshots.append({
            "frame": frame, "timestamp_us": frame.timestamp_us,
            "frame_index": i, "scenario_token": "merge-nearest",
            "entities": {"subject": subject, "near_rear": near, "far_rear": far},
        })
    _, merges = _merge_predicates(module, snapshots)
    front_rows = [row for row in merges if row.predicate_id == "np:mergesInFrontOf"]
    assert front_rows
    assert {row.object_id for row in front_rows} == {"entity:near_rear"}


def test_no_merge_without_lane_change_for_longitudinal_lane_connector_progression():
    module = load_interaction_module()
    lane0 = MapObject("L0", 0.0, length=20.0, outgoing=("C0",), center_y=0.0)
    connector = MapObject("C0", 20.0, length=20.0, outgoing=("L1",), center_y=0.0)
    lane1 = MapObject("L1", 40.0, length=30.0, center_y=0.0)
    api = MapAPI([lane0, connector, lane1])
    sequence = [
        (5.0, lane0, "lane"), (15.0, lane0, "lane"),
        (25.0, connector, "lane_connector"), (35.0, connector, "lane_connector"),
        (45.0, lane1, "lane"), (55.0, lane1, "lane"),
    ]
    snapshots = []
    for i, (sx, primitive, kind) in enumerate(sequence):
        subject = _primitive_record("subject", sx, 0.0, primitive, kind)
        obj = _primitive_record("object", sx - 8.0, 0.0, primitive, kind)
        frame = Frame(i * 500_000, api)
        snapshots.append({
            "frame": frame, "timestamp_us": frame.timestamp_us,
            "frame_index": i, "scenario_token": "merge-longitudinal",
            "entities": {"subject": subject, "object": obj},
        })
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes == []
    assert merges == []


def test_ego_can_merge_in_front_of_target_stream_vehicle():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=-8.0)
    for snapshot in snapshots:
        subject = snapshot["entities"].pop("subject")
        subject["track_token"] = "ego"
        subject["entity_id"] = "ego"
        subject["agent_type"] = "EGO"
        snapshot["entities"]["ego"] = subject
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert merges
    assert {row.subject_id for row in merges} == {"ego"}
    assert {row.predicate_id for row in merges} == {"np:mergesInFrontOf"}


def test_merge_rejects_neighbor_beyond_40m_bumper_gap():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=50.0)
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert merges == []


def test_merge_in_front_rejects_large_time_headway_even_within_40m():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=-20.0)
    # For mergesInFrontOf, the object is the trailing vehicle.  A 16 m bumper
    # gap at 2 m/s is an 8 s headway, beyond the configured 5 s interaction
    # threshold even though the metric gap is below 40 m.
    for snapshot in snapshots:
        obj = snapshot["entities"]["object"]
        obj["velocity"] = (2.0, 0.0)
        obj["speed"] = 2.0
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert merges == []


def test_merge_behind_rejects_large_time_headway_even_within_40m():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=20.0)
    # For mergesBehind, the subject is the trailing vehicle.
    for snapshot in snapshots:
        subject = snapshot["entities"]["subject"]
        subject["velocity"] = (2.0, 0.0)
        subject["speed"] = 2.0
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert merges == []


def test_merge_keeps_queue_case_when_headway_is_not_meaningful():
    module = load_interaction_module()
    snapshots = _merge_history(object_offset_m=-20.0)
    # Below minimum_forward_speed_mps the time-headway gate is not applied;
    # the 40 m metric gap gate remains authoritative for stop-and-go traffic.
    for snapshot in snapshots:
        obj = snapshot["entities"]["object"]
        obj["velocity"] = (0.2, 0.0)
        obj["speed"] = 0.2
    lane_changes, merges = _merge_predicates(module, snapshots)
    assert lane_changes
    assert merges
    rows = [row for row in merges if row.predicate_id == "np:mergesInFrontOf"]
    assert rows
    assert all(row.evidence["post_merge_headway_gate_applied"] is False for row in rows)
    assert all(row.evidence["post_merge_headway_s"] is None for row in rows)
