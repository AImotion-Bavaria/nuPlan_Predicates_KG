"""Thin per-entity coordinator; category logic lives in position/geometry/motion."""
from ..base import *
from .position import initialize_entity_record, append_position_measurements
from .geometry import append_entity_geometry_measurements
from .motion import append_entity_motion_measurements


def append_entity_measurements(
    assertions, frame, entity_id, entity, source_prefix, state=None, agent_type="UNKNOWN"
):
    record = initialize_entity_record(entity_id, entity, agent_type=agent_type)
    append_position_measurements(assertions, frame, record, source_prefix)
    append_entity_geometry_measurements(assertions, frame, record, source_prefix)
    append_entity_motion_measurements(
        assertions, frame, record, entity, source_prefix, state=state
    )
    return record
