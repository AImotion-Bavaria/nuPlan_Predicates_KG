"""Pure helper tests for the production map module without loading nuPlan."""
import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "nuplan_predicates_modular" / "categories" / "map.py"


def load_map_module():
    package = "map_logic_testpkg"
    for name in list(sys.modules):
        if name == package or name.startswith(package + "."):
            sys.modules.pop(name)

    root = types.ModuleType(package)
    root.__path__ = []
    categories = types.ModuleType(f"{package}.categories")
    categories.__path__ = []
    sys.modules[package] = root
    sys.modules[f"{package}.categories"] = categories

    @dataclass
    class DirectionEvidence:
        heading_rad: object = None
        source: str = "unreliable"
        reliable: bool = False
        omission_reason: object = None
        map_heading_rad: object = None
        velocity_heading_rad: object = None
        displacement_heading_rad: object = None
        velocity_displacement_difference_rad: object = None
        map_motion_difference_rad: object = None

    base = types.ModuleType(f"{package}.base")
    base.math = math
    base.ARGS = SimpleNamespace(
        primary_lane_min_overlap_ratio=0.20,
        primary_lane_ambiguity_margin=0.05,
        primary_map_lateral_tiebreak_margin_m=0.50,
    )
    base.DirectionEvidence = DirectionEvidence
    base.MAP_QUERY_AVAILABLE = False
    base.SHAPELY_AVAILABLE = False
    base.wrap_signed = lambda angle: (float(angle) + math.pi) % (2 * math.pi) - math.pi
    base.angular_difference = lambda a, b: abs(base.wrap_signed(float(a) - float(b)))
    sys.modules[f"{package}.base"] = base

    geometry = types.ModuleType(f"{package}.categories.geometry")
    geometry.oriented_box_polygon = lambda record: None
    sys.modules[f"{package}.categories.geometry"] = geometry

    motion = types.ModuleType(f"{package}.categories.motion")
    motion.append_effective_travel_direction = lambda *args, **kwargs: None
    sys.modules[f"{package}.categories.motion"] = motion

    name = f"{package}.categories.map"
    spec = importlib.util.spec_from_file_location(name, MAP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def candidate(object_id, *, overlap, lateral, center=True, kind="lane"):
    return {
        "kind": kind,
        "object_id": object_id,
        "entity_id": f"{kind}:{object_id}",
        "center_covered": center,
        "overlap_ratio": overlap,
        "lateral_offset_m": lateral,
        "heading_difference_rad": 0.0,
    }


def test_primary_selection_is_ambiguous_when_geometry_is_tied():
    module = load_map_module()
    selected, ambiguous, reason, _ = module._select_primary_candidate(
        [candidate("L1", overlap=0.70, lateral=0.10), candidate("L2", overlap=0.69, lateral=0.20)]
    )
    assert selected is None
    assert ambiguous is True
    assert reason == "top_candidates_indistinguishable"


def test_primary_selection_uses_clear_overlap_margin():
    module = load_map_module()
    selected, ambiguous, reason, _ = module._select_primary_candidate(
        [candidate("L1", overlap=0.80, lateral=0.20), candidate("L2", overlap=0.60, lateral=0.10)]
    )
    assert selected["object_id"] == "L1"
    assert ambiguous is False
    assert reason == "overlap_margin"


def test_pair_distances_have_distinct_domains():
    module = load_map_module()
    subject = {"baseline_progress_m": 80.0, "baseline_length_m": 100.0}
    obj = {"baseline_progress_m": 15.0, "baseline_length_m": 90.0}
    assert module._same_baseline_progress_difference(subject, obj, "successor") is None
    assert module._signed_path_distance(subject, obj, "successor") == 35.0
    assert module._signed_path_distance(subject, obj, "predecessor") == -155.0


def test_same_lane_excludes_connectors_but_same_map_does_not():
    module = load_map_module()
    subject = {
        "primary_map_id": "lane_connector:C1",
        "primary_map_kind": "lane_connector",
        "map_match_ambiguous": False,
    }
    obj = dict(subject)
    assert module._same_primary_map(subject, obj) == "lane_connector:C1"
    assert module._same_primary_lane(subject, obj) is None


def test_adjacent_projection_is_internal_not_serialized_path_distance():
    module = load_map_module()

    class MapObject:
        pass

    subject = {
        "baseline_progress_m": 10.0,
        "primary_map_object": MapObject(),
        "xyh": (0.0, 0.0, 0.0),
    }
    obj = {"baseline_progress_m": 12.0, "xyh": (5.0, 3.0, 0.0)}
    module._project_progress_on_map_object = (
        lambda map_object, x, y: float(x)
    )
    assert module._signed_path_distance(subject, obj, "adjacent_left") is None
    assert (
        module._map_longitudinal_reference_difference(subject, obj, "adjacent_left")
        == 5.0
    )
