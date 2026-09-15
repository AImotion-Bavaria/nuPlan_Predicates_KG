"""Regression tests for the first generic traffic-light predicate in v9.5.35."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nuplan_predicates_modular" / "categories" / "traffic_control.py"


def load_module():
    package = "traffic_controls_v9535_testpkg"
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

    map_module = types.ModuleType(f"{package}.categories.map")

    def _find_map_object(map_api, object_id, layer):
        return map_api.objects.get(str(object_id))

    map_module._find_map_object = _find_map_object
    sys.modules[f"{package}.categories.map"] = map_module

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
            objects=objects,
            get_one_map_object=lambda object_id, layer: objects.get(str(object_id)),
        )


def snapshot(frame, status="RED", connector="C17"):
    return {
        "frame": frame,
        "traffic_lights": [{"connector_id": connector, "status": status}],
    }


def test_controls_uses_canonical_dataset_independent_entities():
    module = load_module()
    assertions = module.derive_traffic_control_at_snapshot(snapshot(Frame()))
    assert len(assertions) == 1
    row = assertions[0]
    assert row.predicate_id == "np:controls"
    assert row.subject_id == "traffic_signal:test_map:control:C17"
    assert row.object_id == "controlled_movement:test_map:lane_connector:C17"
    assert row.evidence["dataset_adapter"] == "nuplan"
    assert row.evidence["native_lane_connector_id"] == "C17"
    assert row.evidence["native_controlled_object_type"] == "lane_connector"


def test_controls_identity_does_not_change_with_signal_state_or_timestamp():
    module = load_module()
    red = module.derive_traffic_control_at_snapshot(snapshot(Frame(1), "RED"))[0]
    green = module.derive_traffic_control_at_snapshot(snapshot(Frame(2), "GREEN"))[0]
    assert red.subject_id == green.subject_id
    assert red.object_id == green.object_id
    assert red.evidence["observed_signal_status"] == "RED"
    assert green.evidence["observed_signal_status"] == "GREEN"


def test_controls_rejects_unknown_native_lane_connector():
    module = load_module()
    assertions = module.derive_traffic_control_at_snapshot(
        snapshot(Frame(), status="GREEN", connector="MISSING")
    )
    assert assertions == []
    assert ("np:controls", "native_lane_connector_not_found_in_map") in module._test_omissions


def test_duplicate_status_rows_emit_one_control_relation_per_frame():
    module = load_module()
    frame = Frame()
    snap = {
        "frame": frame,
        "traffic_lights": [
            {"connector_id": "C17", "status": "RED"},
            {"connector_id": "C17", "status": "RED"},
        ],
    }
    assertions = module.derive_traffic_control_at_snapshot(snap)
    assert len(assertions) == 1
