"""Regression tests for traffic-light isRelevantSignal in v9.5.38."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nuplan_predicates_modular" / "categories" / "traffic_control.py"


def load_module(valid_connectors=("C17", "C18")):
    package = "traffic_relevance_v9538_testpkg"
    for name in list(sys.modules):
        if name == package or name.startswith(package + "."):
            sys.modules.pop(name)

    root = types.ModuleType(package)
    root.__path__ = []
    categories = types.ModuleType(f"{package}.categories")
    categories.__path__ = []
    sys.modules[package] = root
    sys.modules[f"{package}.categories"] = categories

    omissions = []

    def derived_relation(pid, subject_id, object_id, frame, rule_id, evidence, **kwargs):
        return SimpleNamespace(
            predicate_id=pid,
            subject_id=subject_id,
            object_id=object_id,
            valid_time_us=frame.timestamp_us,
            evidence=evidence,
            rule_id=rule_id,
        )

    def record_semantic_omission(state, pid, reason):
        omissions.append((pid, reason))

    layer = SimpleNamespace(LANE_CONNECTOR="LANE_CONNECTOR")
    base = types.ModuleType(f"{package}.base")
    values = {
        "ARGS": SimpleNamespace(derive_semantic_predicates=True),
        "MAP_QUERY_AVAILABLE": True,
        "SemanticMapLayer": layer,
        "derived_relation": derived_relation,
        "record_semantic_omission": record_semantic_omission,
    }
    for name, value in values.items():
        setattr(base, name, value)
    base.__all__ = list(values)
    sys.modules[f"{package}.base"] = base

    spec = importlib.util.spec_from_file_location(
        f"{package}.categories.traffic_control", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._test_omissions = omissions
    module._test_valid_connectors = set(valid_connectors)
    return module


class Frame:
    def __init__(self, valid_connectors=("C17", "C18"), timestamp_us=1):
        self.timestamp_us = int(timestamp_us)
        self.scenario_token = "scenario"
        self.log_name = "log"
        objects = {cid: SimpleNamespace(id=cid) for cid in valid_connectors}
        self.map_api = SimpleNamespace(
            map_name="test_map",
            get_one_map_object=lambda object_id, layer: objects.get(str(object_id)),
        )


def entity(
    entity_id="agent:A:1",
    agent_type="VEHICLE",
    track_token="A",
    primary_kind=None,
    primary_object_id=None,
    primary_connector_id=None,
    footprint_lane_ids=(),
    lane_ids=(),
    outgoing=(),
):
    return {
        "entity_id": entity_id,
        "agent_type": agent_type,
        "track_token": track_token,
        "primary_map_kind": primary_kind,
        "primary_map_object_id": primary_object_id,
        "primary_connector_id": primary_connector_id,
        "footprint_lane_ids": set(footprint_lane_ids),
        "lane_ids": set(lane_ids),
        "outgoing_object_ids": set(outgoing),
    }


def snapshot(frame, entities, connectors=("C17",), status="RED"):
    return {
        "frame": frame,
        "traffic_lights": [
            {"connector_id": connector, "status": status}
            for connector in connectors
        ],
        "entities": {str(i): item for i, item in enumerate(entities)},
    }


def by_pid(assertions, pid):
    return [row for row in assertions if row.predicate_id == pid]


def test_agent_on_controlled_connector_is_no_longer_relevant():
    module = load_module()
    frame = Frame()
    agent = entity(
        primary_kind="lane_connector",
        primary_object_id="C17",
        primary_connector_id="lane_connector:C17",
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [agent]), state={}
    )
    assert by_pid(rows, "np:isRelevantSignal") == []
    assert (
        "np:isRelevantSignal",
        "agent_already_entered_controlled_movement_signal_no_longer_relevant",
    ) in module._test_omissions

def test_unique_signal_controlled_successor_is_relevant_before_connector_entry():
    module = load_module()
    frame = Frame()
    agent = entity(
        primary_kind="lane",
        primary_object_id="L1",
        outgoing=("C17", "UNCONTROLLED"),
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [agent]), state={}
    )
    relevant = by_pid(rows, "np:isRelevantSignal")
    assert len(relevant) == 1
    assert relevant[0].object_id == "traffic_signal:test_map:control:C17"
    assert relevant[0].evidence["relevance_basis"] == "pre_entry_unique_signal_controlled_immediate_successor"


def test_multiple_controlled_successors_abstains_instead_of_guessing():
    module = load_module()
    frame = Frame()
    agent = entity(
        primary_kind="lane",
        primary_object_id="L1",
        outgoing=("C17", "C18"),
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [agent], connectors=("C17", "C18")), state={}
    )
    assert by_pid(rows, "np:isRelevantSignal") == []
    assert (
        "np:isRelevantSignal",
        "multiple_signal_controlled_successors_intent_unknown",
    ) in module._test_omissions


def test_pedestrian_is_not_linked_to_lane_connector_vehicle_signal():
    module = load_module()
    frame = Frame()
    pedestrian = entity(
        agent_type="PEDESTRIAN",
        primary_kind="lane_connector",
        primary_object_id="C17",
        primary_connector_id="lane_connector:C17",
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [pedestrian]), state={}
    )
    assert by_pid(rows, "np:isRelevantSignal") == []


def test_bicycle_on_controlled_connector_is_no_longer_relevant_after_entry():
    module = load_module()
    frame = Frame()
    bicycle = entity(
        agent_type="BICYCLE",
        primary_kind="lane_connector",
        primary_object_id="C17",
        primary_connector_id="lane_connector:C17",
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [bicycle]), state={}
    )
    assert by_pid(rows, "np:isRelevantSignal") == []

def test_same_signal_identity_is_used_by_controls_state_and_relevance():
    module = load_module()
    frame = Frame()
    agent = entity(
        primary_kind="lane",
        primary_object_id="L1",
        outgoing=("C17",),
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [agent], status="GREEN"), state={}
    )
    control = by_pid(rows, "np:controls")[0]
    signal_state = by_pid(rows, "np:hasSignalState")[0]
    relevant = by_pid(rows, "np:isRelevantSignal")[0]
    assert control.subject_id == signal_state.subject_id == relevant.object_id
    assert signal_state.object_id == "signal_state:GREEN"


def test_footprint_entering_controlled_connector_terminates_relevance_even_if_primary_is_lane():
    module = load_module()
    frame = Frame()
    agent = entity(
        primary_kind="lane",
        primary_object_id="L1",
        footprint_lane_ids=("lane_connector:C17",),
        outgoing=("C17",),
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [agent]), state={}
    )
    assert by_pid(rows, "np:isRelevantSignal") == []


def test_signal_state_change_after_entry_does_not_recreate_relevance():
    module = load_module()
    frame = Frame(timestamp_us=2)
    agent = entity(
        primary_kind="lane_connector",
        primary_object_id="C17",
        primary_connector_id="lane_connector:C17",
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [agent], status="RED"), state={}
    )
    assert by_pid(rows, "np:hasSignalState")[0].object_id == "signal_state:RED"
    assert by_pid(rows, "np:isRelevantSignal") == []


def test_invalid_native_connector_produces_no_relevance_fact():
    module = load_module(valid_connectors=("C17",))
    frame = Frame(valid_connectors=("C17",))
    agent = entity(
        primary_kind="lane_connector",
        primary_object_id="MISSING",
        primary_connector_id="lane_connector:MISSING",
    )
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(frame, [agent], connectors=("MISSING",)), state={}
    )
    assert by_pid(rows, "np:controls") == []
    assert by_pid(rows, "np:hasSignalState") == []
    assert by_pid(rows, "np:isRelevantSignal") == []
