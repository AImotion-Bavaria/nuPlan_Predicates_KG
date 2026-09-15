"""Observation/scenario/agent structural predicates."""
from ..base import *
from .geometry import get_track_token, get_agent_type


def append_observation_structure(assertions, frame):
    scenario_id = f"scenario:{frame.scenario_token}"
    observation_id = f"observation:{frame.scenario_token}:{frame.timestamp_us}"
    ego_id = f"ego:{frame.scenario_token}:{frame.timestamp_us}"
    assertions.append(
        native_value(
            "np:hasTimestamp", observation_id, int(frame.timestamp_us),
            ValueType.INTEGER, frame, ["lidar_pc.timestamp"],
        )
    )
    assertions.append(
        native_value(
            "np:hasScenarioType", scenario_id, str(frame.scenario_type),
            ValueType.STRING, frame, ["scenario.scenario_type"],
        )
    )
    assertions.append(
        native_value(
            "np:hasLogName", scenario_id, str(frame.log_name),
            ValueType.STRING, frame, ["scenario.log_name"],
        )
    )
    assertions.append(
        native_relation(
            "np:hasEgoState", observation_id, ego_id, frame, ["ego_state"]
        )
    )
    return scenario_id, observation_id, ego_id


def append_agent_structure(assertions, frame, observation_id, obj):
    token = get_track_token(obj)
    agent_id = f"agent:{token}:{frame.timestamp_us}"
    agent_type = get_agent_type(obj)
    assertions.append(
        native_relation(
            "np:hasTrackedObject", observation_id, agent_id, frame,
            ["tracked_objects"],
        )
    )
    assertions.append(
        native_value(
            "np:hasTrackToken", agent_id, token, ValueType.STRING, frame,
            ["tracked_object.metadata.track_token"],
        )
    )
    assertions.append(
        native_value(
            "np:hasAgentType", agent_id, agent_type, ValueType.STRING, frame,
            ["tracked_object.tracked_object_type"],
        )
    )
    return token, agent_id, agent_type
