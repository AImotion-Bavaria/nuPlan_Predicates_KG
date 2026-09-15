"""Pure tests for follows and fully observed Case 1 full-maneuver overtakes."""
import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from shapely.affinity import rotate, translate
from shapely.geometry import LineString, box as shapely_box

ROOT = Path(__file__).resolve().parents[1]
INTERACTION_PATH = ROOT / "nuplan_predicates_modular" / "categories" / "interaction.py"


@dataclass
class DirectionEvidence:
    heading_rad: float = 0.0
    source: str = "map"
    reliable: bool = True
    omission_reason: object = None


class Layer:
    def __init__(self, name):
        self.name = name


class MapObject:
    def __init__(
        self, object_id, offset, length=100.0, outgoing=(),
        left_adjacent=(), right_adjacent=(), *, center_y=0.0, width=4.0,
    ):
        self.id = object_id
        self.offset = float(offset)
        self.length = float(length)
        self.outgoing_edges = [SimpleNamespace(id=value) for value in outgoing]
        self.left_adjacent_ids = set(map(str, left_adjacent))
        self.right_adjacent_ids = set(map(str, right_adjacent))
        self.polygon = shapely_box(
            self.offset,
            float(center_y) - float(width) / 2.0,
            self.offset + self.length,
            float(center_y) + float(width) / 2.0,
        )
        self.baseline_path = SimpleNamespace(
            discrete_path=[
                SimpleNamespace(x=self.offset, y=float(center_y)),
                SimpleNamespace(x=self.offset + self.length, y=float(center_y)),
            ]
        )


class MapAPI:
    def __init__(self, objects):
        self.objects = {obj.id: obj for obj in objects}

    def get_map_object(self, object_id, layer):
        return self.objects.get(str(object_id))


class Frame:
    def __init__(self, timestamp_us, map_api):
        self.timestamp_us = int(timestamp_us)
        self.scenario_token = "scenario"
        self.log_name = "log"
        self.map_api = map_api


def load_interaction_module():
    package = "follow_logic_testpkg"
    for name in list(sys.modules):
        if name == package or name.startswith(package + "."):
            sys.modules.pop(name)

    root = types.ModuleType(package)
    root.__path__ = []
    categories = types.ModuleType(f"{package}.categories")
    categories.__path__ = []
    sys.modules[package] = root
    sys.modules[f"{package}.categories"] = categories

    args = SimpleNamespace(
        derive_semantic_predicates=True,
        follow_max_path_hops=3,
        distance_epsilon_m=1e-3,
        overlap_area_epsilon_m2=1e-4,
        follow_reverse_speed_tolerance_mps=0.30,
        maximum_follow_distance_m=80.0,
        follow_queue_speed_threshold_mps=2.0,
        follow_queue_leader_speed_threshold_mps=4.0,
        maximum_queue_follow_gap_m=12.0,
        minimum_forward_speed_mps=0.30,
        maximum_follow_headway_s=5.0,
        follow_leader_tie_tolerance_m=0.50,
        minimum_follow_duration_s=1.0,
        sample_interval_s=0.5,
        maximum_temporal_gap_factor=1.5,
        minimum_overtake_approach_duration_s=0.5,
        minimum_overtake_passing_duration_s=0.5,
        minimum_overtake_completion_duration_s=0.5,
        minimum_overtake_relative_speed_mps=0.30,
        minimum_overtake_subject_speed_mps=2.0,
        minimum_overtaken_vehicle_speed_mps=0.5,
        minimum_overtake_clearance_m=1.0,
        maximum_overtake_entry_gap_m=120.0,
        overtake_departure_max_lead_m=2.0,
        overtake_same_flow_threshold_rad=0.55,
        overtake_lane_stability_frames=2,
        maximum_overtake_lane_transition_duration_s=1.5,
        maximum_overtake_duration_s=30.0,
        overtake_pair_query_radius_m=80.0,
        overtake_min_common_frames=5,
        overtake_search_padding_s=4.0,
        overtake_max_frame_gap_s=1.5,
        minimum_overtake_initial_behind_gap_m=1.0,
        minimum_overtake_side_by_side_lateral_m=1.25,
        minimum_overtake_same_flow_fraction=0.80,
        overtake_lane_query_radius_m=8.0,
        overtake_lane_min_overlap_ratio=0.10,
        overtake_lane_ambiguity_margin=0.05,
        overtake_max_lane_path_hops=8,
        overtake_positive_deduplication_tolerance_s=2.0,
    )

    def wrap_signed(angle):
        return (float(angle) + math.pi) % (2 * math.pi) - math.pi

    def angular_difference(a, b):
        return abs(wrap_signed(float(a) - float(b)))

    def observations_are_continuous(prev_i, curr_i, prev_t, curr_t, *, sample_interval_s, maximum_gap_factor):
        return curr_i == prev_i + 1 and 0 < (curr_t - prev_t) / 1e6 <= sample_interval_s * maximum_gap_factor

    def update_condition_streak(store, key, *, condition, current_timestamp_us, current_frame_index, continuous_with_previous):
        previous = store.get(key)
        if not condition:
            store.pop(key, None)
            return SimpleNamespace(duration_s=0.0, start_timestamp_us=None, frame_count=0)
        if previous is None or not continuous_with_previous:
            start, count = int(current_timestamp_us), 1
        else:
            start, count = previous["start"], previous["count"] + 1
        duration = (int(current_timestamp_us) - start) / 1e6
        store[key] = {"start": start, "count": count}
        return SimpleNamespace(duration_s=duration, start_timestamp_us=start, frame_count=count)

    def derived_relation(pid, subject_id, object_id, frame, rule_id, evidence, start_time_us=None, end_time_us=None):
        return SimpleNamespace(
            predicate_id=pid,
            subject_id=subject_id,
            object_id=object_id,
            valid_time_us=int(start_time_us if start_time_us is not None else frame.timestamp_us),
            end_time_us=end_time_us,
            evidence=evidence,
        )

    def normalized(record):
        return str(record.get("agent_type", "UNKNOWN")).upper()

    base = types.ModuleType(f"{package}.base")
    for name, value in {
        "math": math,
        "ARGS": args,
        "SemanticMapLayer": SimpleNamespace(LANE=Layer("LANE"), LANE_CONNECTOR=Layer("LANE_CONNECTOR")),
        "_normalized_agent_type": normalized,
        "wrap_signed": wrap_signed,
        "angular_difference": angular_difference,
        "observations_are_continuous": observations_are_continuous,
        "update_condition_streak": update_condition_streak,
        "derived_relation": derived_relation,
        "stable_id": lambda *parts: "stable:" + "|".join(map(str, parts)),
    }.items():
        setattr(base, name, value)
    base.__all__ = [name for name in vars(base) if not name.startswith("__")]
    sys.modules[f"{package}.base"] = base

    def polygon(record):
        length = record["dimensions"]["length_m"]
        width = record["dimensions"]["width_m"]
        x, y, heading = record["xyh"]
        result = shapely_box(-length / 2, -width / 2, length / 2, width / 2)
        result = rotate(result, math.degrees(heading), origin=(0, 0))
        return translate(result, xoff=x, yoff=y)

    def metrics(subject, obj):
        a, b = polygon(subject), polygon(obj)
        area = float(a.intersection(b).area)
        return {
            "free_space_distance_m": float(a.distance(b)),
            "intersection_area_m2": area,
            "overlapping": area > args.overlap_area_epsilon_m2,
            "touching": a.touches(b),
        }

    geometry = types.ModuleType(f"{package}.categories.geometry")
    geometry.oriented_box_polygon = polygon
    geometry.footprint_metrics = metrics
    sys.modules[f"{package}.categories.geometry"] = geometry

    map_module = types.ModuleType(f"{package}.categories.map")
    map_module._baseline_length = lambda obj: obj.length
    map_module._map_edge_ids = lambda obj, direction: {
        str(edge.id) for edge in (obj.outgoing_edges if direction == "outgoing" else [])
    }
    map_module._project_progress_on_map_object = lambda obj, x, y: max(0.0, min(obj.length, float(x) - obj.offset))
    map_module._record_direction = lambda record: record["travel_direction"]
    sys.modules[f"{package}.categories.map"] = map_module

    name = f"{package}.categories.interaction"
    spec = importlib.util.spec_from_file_location(name, INTERACTION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def record(
    token,
    x,
    map_object,
    speed=10.0,
    agent_type="VEHICLE",
    *,
    y=0.0,
    map_heading_rad=0.0,
    velocity=None,
    intersection_id=None,
):
    if velocity is None:
        velocity = (float(speed), 0.0)
    return {
        "entity_id": f"entity:{token}",
        "track_token": token,
        "agent_type": agent_type,
        "xyh": (float(x), float(y), float(map_heading_rad)),
        "velocity": (float(velocity[0]), float(velocity[1])),
        "dimensions": {"length_m": 4.0, "width_m": 2.0},
        "map_match_ambiguous": False,
        "lane_ids": {f"lane:{map_object.id}"},
        "footprint_lane_ids": {f"lane:{map_object.id}"},
        "primary_map_object_id": map_object.id,
        "primary_map_kind": "lane",
        "primary_map_object": map_object,
        "baseline_length_m": map_object.length,
        "map_heading_rad": float(map_heading_rad),
        "travel_direction": DirectionEvidence(heading_rad=float(map_heading_rad)),
        "left_adjacent_object_ids": set(map_object.left_adjacent_ids),
        "right_adjacent_object_ids": set(map_object.right_adjacent_ids),
        "primary_map_intersection_id": intersection_id,
    }


def snapshots(records_factory, map_api):
    result = []
    for index, time_s in enumerate((0.0, 0.5, 1.0)):
        entities = records_factory(index)
        result.append({
            "frame": Frame(int(time_s * 1e6), map_api),
            "timestamp_us": int(time_s * 1e6),
            "frame_index": index,
            "entities": {item["track_token"]: item for item in entities},
        })
    return result


def test_selects_only_nearest_leader_and_emits_after_one_second():
    module = load_interaction_module()
    lane = MapObject("L1", 0.0)
    api = MapAPI([lane])
    history = snapshots(
        lambda _: [record("ego", 10, lane), record("near", 30, lane), record("far", 50, lane)],
        api,
    )
    assertions = module.derive_following_predicates(history)
    ego_assertions = [item for item in assertions if item.subject_id == "entity:ego"]
    assert len(ego_assertions) == 1
    assertion = ego_assertions[0]
    assert assertion.predicate_id == "np:follows"
    assert assertion.object_id == "entity:near"
    assert assertion.valid_time_us == 0
    assert assertion.end_time_us == 1_000_000
    assert assertion.evidence["leader_rank"] == 1
    assert assertion.evidence["follow_mode"] == "moving_headway"


def test_supports_unique_lane_to_successor_lane_path():
    module = load_interaction_module()
    lane_a = MapObject("A", 0.0, outgoing=("B",))
    lane_b = MapObject("B", 100.0)
    api = MapAPI([lane_a, lane_b])
    history = snapshots(
        lambda _: [record("ego", 90, lane_a), record("leader", 110, lane_b)],
        api,
    )
    assertions = module.derive_following_predicates(history)
    ego_assertions = [item for item in assertions if item.subject_id == "entity:ego"]
    assert len(ego_assertions) == 1
    evidence = ego_assertions[0].evidence
    assert evidence["path_relation"] == "successor"
    assert evidence["path_object_ids"] == ["A", "B"]
    assert evidence["path_bumper_gap_m"] > 0


def test_rejects_unresolved_branch_before_turn_choice():
    module = load_interaction_module()
    lane_a = MapObject("A", 0.0, outgoing=("B", "C"))
    lane_b = MapObject("B", 100.0)
    lane_c = MapObject("C", 100.0)
    api = MapAPI([lane_a, lane_b, lane_c])
    history = snapshots(
        lambda _: [record("ego", 90, lane_a), record("leader", 110, lane_b)],
        api,
    )
    assertions = module.derive_following_predicates(history)
    assert not [item for item in assertions if item.subject_id == "entity:ego"]


def test_queue_mode_covers_stopped_and_stop_and_go_following():
    module = load_interaction_module()
    lane = MapObject("L1", 0.0)
    api = MapAPI([lane])
    history = snapshots(
        lambda index: [
            record("ego", 10, lane, speed=0.0),
            record("leader", 18, lane, speed=0.0 if index < 2 else 3.0),
        ],
        api,
    )
    assertions = module.derive_following_predicates(history)
    ego_assertions = [item for item in assertions if item.subject_id == "entity:ego"]
    assert {item.predicate_id for item in ego_assertions} == {"np:follows", "np:queuesBehind"}
    follows = next(item for item in ego_assertions if item.predicate_id == "np:follows")
    queue = next(item for item in ego_assertions if item.predicate_id == "np:queuesBehind")
    assert follows.evidence["follow_mode"] == "queue_or_stop_and_go"
    assert queue.evidence["queue_condition_duration_s"] >= 1.0


def test_nearer_overlapping_vehicle_blocks_farther_leader():
    module = load_interaction_module()
    lane = MapObject("L1", 0.0)
    api = MapAPI([lane])
    history = snapshots(
        lambda _: [record("ego", 10, lane), record("overlap", 12, lane), record("far", 30, lane)],
        api,
    )
    assertions = module.derive_following_predicates(history)
    assert not [item for item in assertions if item.subject_id == "entity:ego"]


def test_moving_headway_supports_ego_as_follower_on_nonzero_map_heading():
    """EGO moving follow must not depend on the map's absolute heading."""
    module = load_interaction_module()
    lane = MapObject("L1", 0.0)
    api = MapAPI([lane])
    northbound_velocity = (0.0, 10.0)
    history = snapshots(
        lambda _: [
            record(
                "ego", 10, lane, map_heading_rad=math.pi / 2,
                velocity=northbound_velocity,
            ),
            record(
                "leader", 30, lane, map_heading_rad=math.pi / 2,
                velocity=northbound_velocity,
            ),
        ],
        api,
    )

    assertions = module.derive_following_predicates(history)
    ego_assertions = [
        item for item in assertions if item.subject_id == "entity:ego"
    ]
    assert len(ego_assertions) == 1
    assert ego_assertions[0].object_id == "entity:leader"
    assert ego_assertions[0].evidence["follow_mode"] == "moving_headway"
    assert math.isclose(
        ego_assertions[0].evidence["subject_forward_speed_mps"],
        10.0,
        abs_tol=1e-9,
    )


def test_moving_headway_supports_ego_as_leader_on_nonzero_map_heading():
    """A moving agent must be able to follow EGO on any road heading."""
    module = load_interaction_module()
    lane = MapObject("L1", 0.0)
    api = MapAPI([lane])
    northbound_velocity = (0.0, 10.0)
    history = snapshots(
        lambda _: [
            record(
                "follower", 10, lane, map_heading_rad=math.pi / 2,
                velocity=northbound_velocity,
            ),
            record(
                "ego", 30, lane, map_heading_rad=math.pi / 2,
                velocity=northbound_velocity,
            ),
        ],
        api,
    )

    assertions = module.derive_following_predicates(history)
    follower_assertions = [
        item for item in assertions if item.subject_id == "entity:follower"
    ]
    assert len(follower_assertions) == 1
    assert follower_assertions[0].object_id == "entity:ego"
    assert follower_assertions[0].evidence["follow_mode"] == "moving_headway"
    assert math.isclose(
        follower_assertions[0].evidence["object_forward_speed_mps"],
        10.0,
        abs_tol=1e-9,
    )


def _overtake_history(
    *,
    return_to_original=True,
    intersection=False,
    ambiguous_transitions=False,
    subject_token="subject",
    object_token="object",
):
    lane_a = MapObject("A", 0.0, left_adjacent=("B",))
    lane_b = MapObject("B", 0.0, right_adjacent=("A",))
    api = MapAPI([lane_a, lane_b])
    subject_positions = [10, 16, 23, 30, 38, 47, 55, 61, 67, 73, 79]
    object_positions = [25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65]
    subject_lanes = [
        lane_a,
        lane_a,
        lane_b,
        lane_b,
        lane_b,
        lane_b,
        lane_b,
        lane_a if return_to_original else lane_b,
        lane_a if return_to_original else lane_b,
        lane_a if return_to_original else lane_b,
        lane_a if return_to_original else lane_b,
    ]
    history = []
    for index, (sx, ox, slane) in enumerate(
        zip(subject_positions, object_positions, subject_lanes)
    ):
        time_s = index * 0.5
        subject_y = 4.0 if slane is lane_b else 0.0
        intersection_id = "I1" if intersection and index == 3 else None
        subject_record = record(
            subject_token,
            sx,
            slane,
            speed=12.0,
            y=subject_y,
            intersection_id=intersection_id,
        )
        if ambiguous_transitions and index in {2, 7}:
            subject_record["map_match_ambiguous"] = True
            subject_record["primary_map_object_id"] = None
            subject_record["primary_map_kind"] = None
            subject_record["primary_map_object"] = None
            subject_record["lane_ids"] = {"lane:A", "lane:B"}
            subject_record["footprint_lane_ids"] = {"lane:A", "lane:B"}
            subject_record["xyh"] = (
                subject_record["xyh"][0],
                2.0,
                subject_record["xyh"][2],
            )
        entities = [
            subject_record,
            record(object_token, ox, lane_a, speed=8.0),
        ]
        history.append({
            "frame": Frame(int(time_s * 1e6), api),
            "timestamp_us": int(time_s * 1e6),
            "frame_index": index,
            "entities": {item["track_token"]: item for item in entities},
        })
    return history


def test_emits_one_completed_overtake_after_pass_and_return():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(_overtake_history())
    assert len(assertions) > 1
    assertion = assertions[-1]
    assert [item.end_time_us for item in assertions] == sorted(
        item.end_time_us for item in assertions
    )
    assert all(item.valid_time_us == assertions[0].valid_time_us for item in assertions)
    assert assertions[0].evidence["overtake_phase_number"] == 1
    assert assertions[-1].evidence["overtake_phase_number"] == 4
    assert assertions[-1].evidence["overtake_completed"] is True
    assert all(item.evidence["overtake_active"] is True for item in assertions)
    assert assertion.predicate_id == "np:overtakes"
    assert assertion.subject_id == "entity:subject"
    assert assertion.object_id == "entity:object"
    evidence = assertion.evidence
    assert evidence["overtake_entry_mode"] == "same_lane_departure"
    assert evidence["overtake_case"] == "case1_same_lane_start"
    assert evidence["overtake_case_number"] == 1
    assert evidence["overtake_initial_lane_relation"] == "same_lane"
    assert evidence["relation_family"] == "agent_to_agent"
    assert evidence["initial_lane_id"] == "A"
    assert evidence["passing_lane_id"] == "B"
    assert evidence["final_lane_id"] == "A"
    assert evidence["passing_side"] == "left"
    assert evidence["returned_to_original_lane"] is True
    assert evidence["overtake_completion_mode"] == "returned_to_original_lane"
    assert evidence["initial_follow_observed"] is False
    assert evidence["side_by_side_time_us"] is not None
    assert evidence["order_reversal_time_us"] is not None
    assert evidence["clearance_time_us"] is not None
    assert evidence["final_clearance_gap_m"] >= 1.0
    assert evidence["maximum_relative_speed_mps"] >= 0.3


def test_does_not_emit_overtake_when_subject_remains_in_passing_lane():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(
        _overtake_history(return_to_original=False)
    )
    assert assertions == []



def _direct_approach_overtake_history():
    lane_a = MapObject("A", 0.0, length=200.0, left_adjacent=("B",))
    lane_b = MapObject("B", 0.0, length=200.0, right_adjacent=("A",))
    api = MapAPI([lane_a, lane_b])
    subject_positions = [0, 8, 20, 35, 50, 65, 80, 90, 100, 110, 120, 130]
    object_positions = [65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109]
    history = []
    for index, (sx, ox) in enumerate(zip(subject_positions, object_positions)):
        lane = lane_a if index < 2 or index >= 9 else lane_b
        y = 0.0 if lane is lane_a else 4.0
        entities = [
            record("subject", sx, lane, speed=10.0, y=y),
            record("object", ox, lane_a, speed=6.0),
        ]
        time_s = index * 0.5
        history.append({
            "frame": Frame(int(time_s * 1e6), api),
            "timestamp_us": int(time_s * 1e6),
            "frame_index": index,
            "entities": {item["track_token"]: item for item in entities},
        })
    return history


def test_direct_approach_can_start_overtake_without_initial_follows():
    module = load_interaction_module()
    history = _direct_approach_overtake_history()
    follows = module.derive_following_predicates(history)
    assert not any(
        item.subject_id == "entity:subject"
        and item.object_id == "entity:object"
        for item in follows
    )

    overtakes = module.derive_overtaking_predicates(history)
    assert len(overtakes) > 1
    evidence = overtakes[-1].evidence
    assert evidence["overtake_entry_mode"] == "same_lane_departure"
    assert evidence["initial_follow_observed"] is False
    assert evidence["initial_bumper_gap_m"] > 50.0
    assert evidence["overtake_case"] == "case1_same_lane_start"
    assert evidence["overtake_case_number"] == 1
    assert evidence["overtake_completion_mode"] == "returned_to_original_lane"
    assert evidence["returned_to_original_lane"] is True

def _adjacent_lane_start_history(
    *,
    subject_token="subject",
    object_token="object",
    object_changes_to_passing_lane=False,
):
    lane_a = MapObject("A", 0.0, left_adjacent=("B",))
    lane_b = MapObject("B", 0.0, right_adjacent=("A",))
    api = MapAPI([lane_a, lane_b])
    subject_positions = [10, 18, 27, 37, 48, 58, 67, 75, 82, 89, 96]
    object_positions = [25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65]
    subject_lanes = [lane_b] * 7 + [lane_a] * 4
    history = []
    for index, (sx, ox, subject_lane) in enumerate(
        zip(subject_positions, object_positions, subject_lanes)
    ):
        time_s = index * 0.5
        object_lane = (
            lane_b if object_changes_to_passing_lane and index >= 7 else lane_a
        )
        subject_record = record(
            subject_token,
            sx,
            subject_lane,
            speed=12.0,
            y=4.0 if subject_lane is lane_b else 0.0,
        )
        object_record = record(
            object_token,
            ox,
            object_lane,
            speed=8.0,
            y=4.0 if object_lane is lane_b else 0.0,
        )
        history.append({
            "frame": Frame(int(time_s * 1e6), api),
            "timestamp_us": int(time_s * 1e6),
            "frame_index": index,
            "entities": {
                subject_record["track_token"]: subject_record,
                object_record["track_token"]: object_record,
            },
        })
    return history


def test_adjacent_lane_start_is_rejected():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(
        _adjacent_lane_start_history()
    )
    assert assertions == []


def test_adjacent_lane_start_with_object_lane_change_is_rejected():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(
        _adjacent_lane_start_history(object_changes_to_passing_lane=True)
    )
    assert assertions == []


def test_overtakes_support_all_three_relation_families_for_case1():
    cases = [
        ("ego", "object", "ego_to_agent"),
        ("subject", "ego", "agent_to_ego"),
        ("subject", "object", "agent_to_agent"),
    ]
    for subject_token, object_token, expected_family in cases:
        module = load_interaction_module()
        assertions = module.derive_overtaking_predicates(
            _overtake_history(
                subject_token=subject_token,
                object_token=object_token,
            )
        )
        assert len(assertions) > 1
        assert all(
            assertion.evidence["relation_family"] == expected_family
            for assertion in assertions
        )


def test_notebook_strategy_allows_transient_non_phase_intersection_sample():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(
        _overtake_history(intersection=True)
    )
    # The internal detector rejects intersection lane states as strict phase
    # anchors, but it does not globally reject a transient sample between them.
    assert len(assertions) > 1
    assert assertions[-1].evidence["intersection_involved"] is True


def test_completed_overtake_survives_bounded_lane_boundary_ambiguity():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(
        _overtake_history(ambiguous_transitions=True)
    )
    assert len(assertions) > 1
    evidence = assertions[-1].evidence
    # Ambiguous boundary samples are not treated as phases; the strict detector
    # waits until map matching resolves to one unambiguous lane.
    assert evidence["departure_transition_frame_count"] == 0
    assert evidence["return_transition_frame_count"] == 0
    assert evidence["passing_lane_id"] == "B"
    assert evidence["final_lane_id"] == "A"


def _queue_overtake_history():
    """One Case 1 maneuver that passes two vehicles before returning."""
    lane_a = MapObject("A", 0.0, left_adjacent=("B",))
    lane_b = MapObject("B", 0.0, right_adjacent=("A",))
    api = MapAPI([lane_a, lane_b])
    subject_positions = [0, 10, 20, 32, 45, 58, 72, 85, 98, 110, 122, 134, 146]
    rear_positions = [25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73]
    front_positions = [45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93]
    subject_lanes = [lane_a, lane_a] + [lane_b] * 8 + [lane_a] * 3
    history = []
    for index, (sx, rear_x, front_x, subject_lane) in enumerate(
        zip(
            subject_positions,
            rear_positions,
            front_positions,
            subject_lanes,
        )
    ):
        time_s = index * 0.5
        subject_y = 4.0 if subject_lane is lane_b else 0.0
        entities = [
            record("subject", sx, subject_lane, speed=14.0, y=subject_y),
            record("queue_rear", rear_x, lane_a, speed=8.0),
            record("queue_front", front_x, lane_a, speed=8.0),
        ]
        history.append({
            "frame": Frame(int(time_s * 1e6), api),
            "timestamp_us": int(time_s * 1e6),
            "frame_index": index,
            "entities": {item["track_token"]: item for item in entities},
        })
    return history


def test_case1_queue_overtake_emits_one_binary_relation_per_passed_vehicle():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(_queue_overtake_history())
    objects = {assertion.object_id for assertion in assertions}
    assert objects == {"entity:queue_rear", "entity:queue_front"}
    for object_id in objects:
        rows = [row for row in assertions if row.object_id == object_id]
        assert len(rows) > 1
        assert rows[0].evidence["overtake_case"] == "case1_same_lane_start"
        assert rows[-1].evidence["overtake_completed"] is True
