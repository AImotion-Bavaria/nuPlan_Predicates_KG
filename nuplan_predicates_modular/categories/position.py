"""Position-value predicates and the initial per-entity record."""
from ..base import *
from .geometry import get_geometric_center_pose, get_dimensions


def initialize_entity_record(entity_id, entity, agent_type="UNKNOWN"):
    x, y, heading = get_geometric_center_pose(entity)
    return {
        "entity_id": entity_id,
        "xyh": (x, y, heading),
        "velocity": None,
        "speed": None,
        "velocity_heading_rad": None,
        "displacement_heading_rad": None,
        "displacement_from_previous_m": None,
        "travel_direction": None,
        "dimensions": get_dimensions(entity),
        "lane_ids": set(),
        "agent_type": str(agent_type),
        "primary_lane_id": None,
        "primary_connector_id": None,
        "primary_map_id": None,
        "map_match_ambiguous": False,
    }


def append_position_measurements(assertions, frame, record, source_prefix):
    x, y, heading = record["xyh"]
    for pid, value, fields in (
        ("np:hasX", x, [f"{source_prefix}.center.x"]),
        ("np:hasY", y, [f"{source_prefix}.center.y"]),
        ("np:hasHeading", heading, [f"{source_prefix}.center.heading"]),
    ):
        assertions.append(
            native_value(pid, record["entity_id"], value, ValueType.FLOAT, frame, fields)
        )


def append_ego_rear_axle_position(assertions, frame, ego_id, ego_state):
    from .geometry import get_rear_axle_xy
    rear_axle = get_rear_axle_xy(ego_state)
    if rear_axle is None:
        return
    assertions.append(
        native_value(
            "np:hasRearAxleX", ego_id, rear_axle[0], ValueType.FLOAT, frame,
            ["ego_state.rear_axle.x"],
        )
    )
    assertions.append(
        native_value(
            "np:hasRearAxleY", ego_id, rear_axle[1], ValueType.FLOAT, frame,
            ["ego_state.rear_axle.y"],
        )
    )
