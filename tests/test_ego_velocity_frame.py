"""Regression tests for EGO body-frame velocity conversion in v9.5.6."""
import importlib.util
import math
import sys
import types
from typing import Any, Optional
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MOTION_PATH = ROOT / "nuplan_predicates_modular" / "categories" / "motion.py"


def load_motion_module():
    package = "ego_velocity_testpkg"
    for name in list(sys.modules):
        if name == package or name.startswith(package + "."):
            sys.modules.pop(name)

    root = types.ModuleType(package)
    root.__path__ = []
    categories = types.ModuleType(f"{package}.categories")
    categories.__path__ = []
    sys.modules[package] = root
    sys.modules[f"{package}.categories"] = categories

    base = types.ModuleType(f"{package}.base")
    base.math = math
    base.Any = Any
    base.Optional = Optional
    base.__all__ = ["math", "Any", "Optional"]
    sys.modules[f"{package}.base"] = base

    geometry = types.ModuleType(f"{package}.categories.geometry")
    geometry.get_geometric_center_pose = lambda entity: (
        float(entity.pose.x),
        float(entity.pose.y),
        float(entity.pose.heading),
    )
    sys.modules[f"{package}.categories.geometry"] = geometry

    name = f"{package}.categories.motion"
    spec = importlib.util.spec_from_file_location(name, MOTION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def vector(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def ego(*, heading, center_velocity=None, rear_velocity=None, omega=0.0, rear_to_center=0.0):
    dynamic = SimpleNamespace(
        center_velocity_2d=center_velocity,
        rear_axle_velocity_2d=rear_velocity,
        angular_velocity=float(omega),
    )
    return SimpleNamespace(
        pose=SimpleNamespace(x=0.0, y=0.0, heading=float(heading)),
        dynamic_car_state=dynamic,
        car_footprint=SimpleNamespace(
            rear_axle_to_center_dist=float(rear_to_center)
        ),
    )


def test_ego_center_velocity_is_rotated_from_body_to_global_frame():
    module = load_motion_module()
    state = ego(
        heading=math.pi / 2,
        center_velocity=vector(10.0, 0.0),
    )
    vx, vy = module.get_velocity(state)
    assert math.isclose(vx, 0.0, abs_tol=1e-9)
    assert math.isclose(vy, 10.0, abs_tol=1e-9)


def test_ego_lateral_velocity_is_rotated_with_longitudinal_velocity():
    module = load_motion_module()
    state = ego(
        heading=math.pi,
        center_velocity=vector(8.0, 1.0),
    )
    vx, vy = module.get_velocity(state)
    assert math.isclose(vx, -8.0, abs_tol=1e-9)
    assert math.isclose(vy, -1.0, abs_tol=1e-9)


def test_ego_rear_axle_fallback_is_shifted_then_rotated():
    module = load_motion_module()
    state = ego(
        heading=math.pi / 2,
        rear_velocity=vector(10.0, 0.0),
        omega=1.0,
        rear_to_center=2.0,
    )
    vx, vy = module.get_velocity(state)
    assert math.isclose(vx, -2.0, abs_tol=1e-9)
    assert math.isclose(vy, 10.0, abs_tol=1e-9)


def test_tracked_agent_global_velocity_is_unchanged():
    module = load_motion_module()
    tracked_object = SimpleNamespace(velocity=vector(3.0, 4.0))
    assert module.get_velocity(tracked_object) == (3.0, 4.0)
