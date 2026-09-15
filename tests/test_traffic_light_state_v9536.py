"""Regression tests for traffic-light controls + hasSignalState in v9.5.36."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nuplan_predicates_modular" / "categories" / "traffic_control.py"


def load_module():
    package = "traffic_controls_v9536_testpkg"
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
    return module


class Frame:
    def __init__(self, timestamp_us=1):
        self.timestamp_us = int(timestamp_us)
        self.scenario_token = "scenario"
        self.log_name = "log"
        objects = {"C17": SimpleNamespace(id="C17")}
        self.map_api = SimpleNamespace(
            map_name="test_map",
            get_one_map_object=lambda object_id, layer: objects.get(str(object_id)),
        )


def snapshot(frame, status="RED", connector="C17"):
    return {
        "frame": frame,
        "traffic_lights": [{"connector_id": connector, "status": status}],
    }


def by_pid(assertions, pid):
    return [row for row in assertions if row.predicate_id == pid]


def test_emits_controls_and_signal_state_for_same_signal_identity():
    module = load_module()
    rows = module.derive_traffic_light_predicates_at_snapshot(snapshot(Frame(), "RED"))
    controls = by_pid(rows, "np:controls")
    states = by_pid(rows, "np:hasSignalState")
    assert len(controls) == 1
    assert len(states) == 1
    assert controls[0].subject_id == states[0].subject_id
    assert states[0].object_id == "signal_state:RED"
    assert states[0].evidence["canonical_signal_state"] == "RED"


def test_state_identity_changes_but_signal_identity_does_not():
    module = load_module()
    red = module.derive_traffic_light_predicates_at_snapshot(snapshot(Frame(1), "RED"))
    green = module.derive_traffic_light_predicates_at_snapshot(snapshot(Frame(2), "GREEN"))
    red_state = by_pid(red, "np:hasSignalState")[0]
    green_state = by_pid(green, "np:hasSignalState")[0]
    assert red_state.subject_id == green_state.subject_id
    assert red_state.object_id == "signal_state:RED"
    assert green_state.object_id == "signal_state:GREEN"


def test_enum_style_and_amber_are_normalized():
    module = load_module()
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(Frame(), "TrafficLightStatusType.GREEN")
    )
    assert by_pid(rows, "np:hasSignalState")[0].object_id == "signal_state:GREEN"

    rows = module.derive_traffic_light_predicates_at_snapshot(snapshot(Frame(), "AMBER"))
    assert by_pid(rows, "np:hasSignalState")[0].object_id == "signal_state:YELLOW"


def test_unrecognized_status_maps_to_unknown():
    module = load_module()
    rows = module.derive_traffic_light_predicates_at_snapshot(snapshot(Frame(), "FLASHING"))
    state = by_pid(rows, "np:hasSignalState")[0]
    assert state.object_id == "signal_state:UNKNOWN"


def test_duplicate_rows_emit_one_control_and_one_state():
    module = load_module()
    frame = Frame()
    snap = {
        "frame": frame,
        "traffic_lights": [
            {"connector_id": "C17", "status": "RED"},
            {"connector_id": "C17", "status": "RED"},
        ],
    }
    rows = module.derive_traffic_light_predicates_at_snapshot(snap)
    assert len(by_pid(rows, "np:controls")) == 1
    assert len(by_pid(rows, "np:hasSignalState")) == 1


def test_conflicting_rows_collapse_to_unknown_not_two_states():
    module = load_module()
    frame = Frame()
    snap = {
        "frame": frame,
        "traffic_lights": [
            {"connector_id": "C17", "status": "RED"},
            {"connector_id": "C17", "status": "GREEN"},
        ],
    }
    rows = module.derive_traffic_light_predicates_at_snapshot(snap, state={})
    states = by_pid(rows, "np:hasSignalState")
    assert len(states) == 1
    assert states[0].object_id == "signal_state:UNKNOWN"
    assert (
        "np:hasSignalState",
        "conflicting_native_signal_states_collapsed_to_unknown",
    ) in module._test_omissions


def test_missing_connector_emits_no_traffic_light_facts():
    module = load_module()
    rows = module.derive_traffic_light_predicates_at_snapshot(
        snapshot(Frame(), "RED", connector="MISSING"), state={}
    )
    assert rows == []
    assert ("np:controls", "native_lane_connector_not_found_in_map") in module._test_omissions
    assert ("np:hasSignalState", "native_lane_connector_not_found_in_map") in module._test_omissions
