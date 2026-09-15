"""Per-frame orchestration. Keep category-specific logic out of this file."""
from .base import *
from .categories.structure import append_observation_structure, append_agent_structure
from .categories.entity import append_entity_measurements
from .categories.position import append_ego_rear_axle_position
from .categories.motion import append_ego_acceleration_measurements
from .categories.temporal import append_temporal_measurements
from .categories.map import append_map_memberships
from .categories.route import append_expert_route
from .categories.pairwise import append_pairwise_measurements
from .categories.visibility import derive_visibility_predicates
from .categories.traffic_control import (
    derive_traffic_light_predicates_at_snapshot,
    append_native_traffic_light_status,
)
from .categories.relevance import (
    _select_relevant_ego_agents,
    _select_relevant_agent_pairs,
    reverse_selected_pair_decision,
)

def extract_frame_predicates(frame, state):
    assertions = []
    state.setdefault("semantic_streaks", {})
    state.setdefault("semantic_omissions", Counter())
    state.setdefault("temporal_continuity_violations", [])
    state["current_pair_measurements"] = {}
    if "route_ids" not in state:
        route_ids = tuple(str(value) for value in getattr(frame, "route_roadblock_ids", []))
        state["route_ids"] = route_ids
        state["route_id_set"] = set(route_ids)
        state["route_rank"] = {roadblock_id: index for index, roadblock_id in enumerate(route_ids)}
    route_ids = state["route_ids"]
    state["current_route_roadblock_rank"] = state["route_rank"]

    scenario_id, observation_id, ego_id = append_observation_structure(
        assertions, frame
    )

    ego_record = append_entity_measurements(
        assertions, frame, ego_id, frame.ego_state, "ego_state", state=state, agent_type="EGO"
    )
    ego_record["track_token"] = "ego"
    append_ego_rear_axle_position(assertions, frame, ego_id, frame.ego_state)
    append_ego_acceleration_measurements(assertions, frame, ego_id, frame.ego_state)

    append_temporal_measurements(assertions, frame, state, "ego", ego_record)
    ego_record["expert_route_roadblock_ids"] = state["route_id_set"]
    append_map_memberships(assertions, frame, ego_record, state=state)

    append_expert_route(assertions, frame)
    append_native_traffic_light_status(assertions, frame)

    agent_records = []
    for obj in frame.tracked_objects:
        token, agent_id, agent_type = append_agent_structure(
            assertions, frame, observation_id, obj
        )

        record = append_entity_measurements(
            assertions, frame, agent_id, obj, "tracked_object", state=state, agent_type=agent_type
        )
        record["track_token"] = token
        record["expert_route_roadblock_ids"] = set()
        append_temporal_measurements(assertions, frame, state, token, record)
        append_map_memberships(assertions, frame, record, state=state)
        agent_records.append(record)

    ego_direction_modes = {
        direction: ARGS.pair_selection_by_direction[direction]
        for direction in ("ego-agent", "agent-ego")
        if direction in ARGS.pair_directions
    }
    all_ego_agents, relevant_ego_agents, ego_relevance_decisions, relevance_contexts = _select_relevant_ego_agents(
        ego_record, agent_records, frame, state, direction_modes=ego_direction_modes
    )
    relevant_tokens = {str(record["track_token"]) for record, _ in relevant_ego_agents}
    relevant_agent_tokens = sorted(relevant_tokens)

    for record, decision in all_ego_agents:
        token = str(record["track_token"])
        if "ego-agent" in ARGS.pair_directions:
            mode = ARGS.pair_selection_by_direction["ego-agent"]
            if mode == "all" or token in relevant_tokens:
                append_pairwise_measurements(
                    assertions, frame, state, ego_record, record, "ego_to_agent", decision
                )
        if "agent-ego" in ARGS.pair_directions:
            mode = ARGS.pair_selection_by_direction["agent-ego"]
            if mode == "all" or token in relevant_tokens:
                reverse_decision = reverse_selected_pair_decision(
                    record, ego_record, frame, decision
                )
                append_pairwise_measurements(
                    assertions, frame, state, record, ego_record, "agent_to_ego", reverse_decision
                )

    relevant_agent_agent_pairs = []
    agent_pair_decisions = {}
    if "agent-agent" in ARGS.pair_directions:
        relevant_agent_agent_pairs, agent_pair_decisions = _select_relevant_agent_pairs(
            agent_records,
            frame,
            state,
            selection_mode=ARGS.pair_selection_by_direction["agent-agent"],
        )
        by_token = {str(record["track_token"]): record for record in agent_records}
        for first, second in relevant_agent_agent_pairs:
            a, b = by_token[first], by_token[second]
            decision = agent_pair_decisions[(first, second)]
            append_pairwise_measurements(assertions, frame, state, a, b, "agent_to_agent", decision)
            reverse_decision = reverse_selected_pair_decision(b, a, frame, decision)
            append_pairwise_measurements(
                assertions, frame, state, b, a, "agent_to_agent_reverse", reverse_decision
            )

    entities = {"ego": ego_record}
    entities.update({str(record["track_token"]): record for record in agent_records})
    snapshot = {
        "scenario_token": str(frame.scenario_token),
        "timestamp_us": int(frame.timestamp_us),
        "frame_index": int(state["frame_index"]),
        "frame": frame,
        "entities": entities,
        "traffic_lights": [
            {
                "connector_id": str(getattr(light, "lane_connector_id", "unknown")),
                "status": str(getattr(light, "status", "UNKNOWN")),
            }
            for light in frame.traffic_lights
        ],
        "pair_measurements": dict(state.get("current_pair_measurements", {})),
        "relevant_agent_tokens": relevant_agent_tokens,
        "ego_relevance_decisions": ego_relevance_decisions,
        "relevant_agent_agent_pairs": relevant_agent_agent_pairs,
    }
    # Retain frame snapshots only for categories that truly need temporal
    # scenario-level history. Spatial-only extraction should not keep every
    # full frame object in memory.
    snapshot_history_categories = {
        "future_observation", "maneuver", "interaction", "intent"
    }
    needs_snapshot_history = bool(ACTIVE_CATEGORIES & snapshot_history_categories)

    snapshots = state.setdefault("snapshots", []) if needs_snapshot_history else []
    previous_snapshot = snapshots[-1] if snapshots else None

    if ARGS.derive_semantic_predicates and "visibility" in ACTIVE_CATEGORIES:
        assertions.extend(derive_visibility_predicates(snapshot, state=state))

    if ARGS.derive_semantic_predicates and "traffic_light" in ACTIVE_CATEGORIES:
        assertions.extend(
            derive_traffic_light_predicates_at_snapshot(snapshot, previous_snapshot, state=state)
        )

    if needs_snapshot_history:
        snapshots.append(snapshot)

    return assertions
