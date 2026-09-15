"""Regression tests for Lane/LaneConnector corridor-aware overtaking."""

from test_interaction_follow import Frame, MapAPI, MapObject, load_interaction_module, record


def _primitive_record(token, x, primitive, *, kind, speed, y, intersection_id=None):
    item = record(token, x, primitive, speed=speed, y=y, intersection_id=intersection_id)
    item["primary_map_kind"] = kind
    item["primary_map_object_id"] = primitive.id
    item["primary_map_object"] = primitive
    item["map_candidates"] = [{
        "kind": kind,
        "object_id": primitive.id,
        "map_object": primitive,
        "overlap_ratio": 1.0,
        "center_covered": True,
    }]
    if kind == "lane_connector":
        item["lane_ids"] = set()
        item["footprint_lane_ids"] = set()
        item["primary_lane_id"] = None
        item["primary_connector_id"] = primitive.id
    else:
        item["primary_lane_id"] = primitive.id
        item["primary_connector_id"] = None
    return item


def _lane_connector_corridor_overtake_history():
    # Original stream: A0 -> AC -> A1
    # Passing stream:  B0 -> BC -> B1
    # The actual lateral departure occurs inside the connector region AC -> BC.
    a0 = MapObject("A0", 0.0, outgoing=("AC",), left_adjacent=("B0",))
    ac = MapObject("AC", 0.0, outgoing=("A1",), left_adjacent=("BC",))
    a1 = MapObject("A1", 0.0, left_adjacent=("B1",))
    b0 = MapObject("B0", 0.0, outgoing=("BC",), right_adjacent=("A0",), center_y=4.0)
    bc = MapObject("BC", 0.0, outgoing=("B1",), right_adjacent=("AC",), center_y=4.0)
    b1 = MapObject("B1", 0.0, right_adjacent=("A1",), center_y=4.0)
    api = MapAPI([a0, ac, a1, b0, bc, b1])

    subject_positions = [10, 16, 23, 30, 38, 47, 55, 61, 67, 73, 79]
    object_positions = [25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65]
    subject_primitives = [a0, a0, a0, bc, bc, b1, b1, a1, a1, a1, a1]
    subject_kinds = ["lane", "lane", "lane", "lane_connector", "lane_connector", "lane", "lane", "lane", "lane", "lane", "lane"]
    object_primitives = [a0, a0, a0, ac, ac, a1, a1, a1, a1, a1, a1]
    object_kinds = ["lane", "lane", "lane", "lane_connector", "lane_connector", "lane", "lane", "lane", "lane", "lane", "lane"]

    history = []
    for index, (sx, ox, sp, sk, op, ok) in enumerate(zip(
        subject_positions, object_positions,
        subject_primitives, subject_kinds,
        object_primitives, object_kinds,
    )):
        connector_frame = sk == "lane_connector" or ok == "lane_connector"
        subject = _primitive_record(
            "subject", sx, sp, kind=sk, speed=12.0,
            y=4.0 if sp in {b0, bc, b1} else 0.0,
            intersection_id="I1" if connector_frame else None,
        )
        obj = _primitive_record(
            "object", ox, op, kind=ok, speed=8.0, y=0.0,
            intersection_id="I1" if connector_frame else None,
        )
        time_s = index * 0.5
        history.append({
            "frame": Frame(int(time_s * 1e6), api),
            "timestamp_us": int(time_s * 1e6),
            "frame_index": index,
            "entities": {"subject": subject, "object": obj},
        })
    return history


def test_overtake_survives_lane_connector_corridor_progression():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(
        _lane_connector_corridor_overtake_history()
    )
    assert assertions
    evidence = assertions[-1].evidence
    assert evidence["overtake_corridor_model"] == "lane_and_lane_connector_directed_continuity_v9_5_31"
    assert evidence["lane_and_lane_connector_primitives_supported"] is True
    assert evidence["longitudinal_lane_connector_progression_preserves_corridor"] is True
    assert evidence["lateral_lane_connector_passing_supported"] is True
    assert evidence["passing_map_kind"] == "lane_connector"
    assert evidence["passing_map_entity_id"] == "lane_connector:BC"
    assert evidence["intersection_involved"] is True
    assert evidence["returned_to_original_lane"] is True
    assert evidence["final_subject_map_entity_id"] == "lane:A1"
    assert evidence["final_object_map_entity_id"] == "lane:A1"
