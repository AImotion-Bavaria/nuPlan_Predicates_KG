"""Dataset-independent traffic-light predicate grounding.

v9.5.38 implements the first three predicates of the ``traffic_light`` family::

    controls(TrafficSignal, ControlledMovement)
    hasSignalState(TrafficSignal, SignalState)
    isRelevantSignal(Agent, TrafficSignal)

The public vocabulary is dataset-independent.  The nuPlan adapter grounds its
native ``traffic_light_status.lane_connector_id`` into a canonical
``ControlledMovement`` and maps the native status into one of four canonical
state entities: RED, YELLOW, GREEN, or UNKNOWN.

The same stable logical ``TrafficSignal`` identity is used by both predicates,
so a signal changing from RED to GREEN changes only the state relation, never
its identity or its ``controls`` relation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..base import *


TRAFFIC_LIGHT_CONTROL_RULE_ID = "R-TRAFFIC-LIGHT-CONTROLS-001"
TRAFFIC_LIGHT_STATE_RULE_ID = "R-TRAFFIC-LIGHT-STATE-001"
TRAFFIC_LIGHT_RELEVANCE_RULE_ID = "R-TRAFFIC-LIGHT-RELEVANCE-002"
CANONICAL_SIGNAL_STATES = {"RED", "YELLOW", "GREEN", "UNKNOWN"}


def _frame_map_name(frame: Any) -> str:
    """Return the most stable map identifier available on the current frame."""
    map_api = getattr(frame, "map_api", None)
    for attr in ("map_name", "_map_name"):
        value = getattr(map_api, attr, None)
        if value is None:
            continue
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(getattr(frame, "log_name", "unknown_map"))


def _canonical_signal_id(map_name: str, connector_id: str) -> str:
    """Stable logical traffic-signal entity for a controlled movement."""
    return f"traffic_signal:{map_name}:control:{connector_id}"


def _canonical_movement_id(map_name: str, connector_id: str) -> str:
    """Canonical controlled movement grounded by the dataset adapter."""
    return f"controlled_movement:{map_name}:lane_connector:{connector_id}"


def _canonical_signal_state_id(state_name: str) -> str:
    """Canonical signal-state entity shared across datasets."""
    return f"signal_state:{state_name}"


def _normalize_signal_state(value: Any) -> str:
    """Map a dataset/native status representation to the canonical state set."""
    if value is None:
        return "UNKNOWN"
    text = str(getattr(value, "name", value)).strip().upper()
    # Handles enum string forms such as ``TrafficLightStatusType.RED``.
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    aliases = {
        "AMBER": "YELLOW",
        "ORANGE": "YELLOW",
        "UNAVAILABLE": "UNKNOWN",
        "NONE": "UNKNOWN",
        "": "UNKNOWN",
    }
    text = aliases.get(text, text)
    return text if text in CANONICAL_SIGNAL_STATES else "UNKNOWN"


def _native_connector_exists(frame: Any, connector_id: str) -> bool:
    """Validate a nuPlan lane connector when the map API supports lookup."""
    if not MAP_QUERY_AVAILABLE:
        return True
    map_api = getattr(frame, "map_api", None)
    attempted = False
    for method_name in ("get_one_map_object", "get_map_object"):
        method = getattr(map_api, method_name, None)
        if method is None:
            continue
        attempted = True
        try:
            obj = method(str(connector_id), SemanticMapLayer.LANE_CONNECTOR)
        except Exception:
            continue
        if obj is not None:
            return True
    return True if not attempted else False


def _group_native_traffic_lights(snapshot):
    """Group native rows by controlled connector while preserving raw states."""
    grouped = defaultdict(list)
    for item in snapshot.get("traffic_lights", []):
        connector_id = str(item.get("connector_id", "")).strip()
        if not connector_id or connector_id.lower() == "unknown":
            grouped[None].append(item)
            continue
        grouped[connector_id].append(item)
    return grouped


def _resolve_group_state(items, state):
    """Return one deterministic canonical state for a connector in one frame."""
    normalized = [_normalize_signal_state(item.get("status")) for item in items]
    unique = sorted(set(normalized))
    if len(unique) == 1:
        return unique[0], normalized
    # Conflicting native rows should never create two states simultaneously.
    record_semantic_omission(
        state,
        "np:hasSignalState",
        "conflicting_native_signal_states_collapsed_to_unknown",
    )
    return "UNKNOWN", normalized




_LANE_USING_AGENT_MARKERS = (
    "EGO",
    "VEHICLE",
    "CAR",
    "BUS",
    "TRUCK",
    "TRAILER",
    "CONSTRUCTION_VEHICLE",
    "BICYCLE",
    "CYCLIST",
    "MOTORCYCLE",
)


def _lane_using_agent(record: dict[str, Any]) -> bool:
    """Return True for road users that may legitimately obey lane signals."""
    if str(record.get("track_token", "")) == "ego":
        return True
    text = str(record.get("agent_type", "UNKNOWN")).upper()
    return any(marker in text for marker in _LANE_USING_AGENT_MARKERS)


def _native_connector_id_from_entity_id(value: Any) -> str | None:
    text = str(value or "").strip()
    prefix = "lane_connector:"
    if text.startswith(prefix):
        native = text[len(prefix):].strip()
        return native or None
    return None


def _direct_controlled_connector_candidates(record: dict[str, Any], controlled_ids: set[str]) -> set[str]:
    """Controlled connectors currently occupied/intersected by the agent."""
    candidates: set[str] = set()

    primary_native = None
    if str(record.get("primary_map_kind", "")) == "lane_connector":
        value = record.get("primary_map_object_id")
        if value is not None:
            primary_native = str(value)
    if primary_native in controlled_ids:
        candidates.add(primary_native)

    native = _native_connector_id_from_entity_id(record.get("primary_connector_id"))
    if native in controlled_ids:
        candidates.add(native)

    for entity_id in record.get("footprint_lane_ids", set()) or set():
        native = _native_connector_id_from_entity_id(entity_id)
        if native in controlled_ids:
            candidates.add(native)

    for entity_id in record.get("lane_ids", set()) or set():
        native = _native_connector_id_from_entity_id(entity_id)
        if native in controlled_ids:
            candidates.add(native)

    return candidates


def _unique_controlled_successor_candidates(record: dict[str, Any], controlled_ids: set[str]) -> set[str]:
    """Signal-controlled immediate successors of the current lane."""
    if str(record.get("primary_map_kind", "")) != "lane":
        return set()
    outgoing = {str(value) for value in (record.get("outgoing_object_ids", set()) or set())}
    return outgoing & controlled_ids


def _derive_relevant_signal_assertions(snapshot, grouped, map_name, state):
    """Link an agent to a signal only while it still governs an upcoming movement.

    v9.5.38 pre-entry semantics:
      1. the agent must still be on an incoming lane;
      2. that lane must have exactly one currently signal-controlled immediate successor;
      3. the agent footprint must not already occupy/intersect the controlled connector.

    Once the agent reaches/enters the controlled lane connector, the signal has
    already governed the entry decision and ``isRelevantSignal`` is no longer
    emitted for that signal.  Later RED/GREEN changes therefore cannot be
    incorrectly attached to an agent that has already crossed into the movement.

    If several controlled movements are possible, intended movement is unknown and
    the predicate is intentionally omitted rather than guessed.
    """
    frame = snapshot["frame"]
    controlled_ids = {str(value) for value in grouped.keys() if value is not None}
    if not controlled_ids:
        return []

    assertions = []
    for record in snapshot.get("entities", {}).values():
        if not isinstance(record, dict) or not _lane_using_agent(record):
            continue

        # Crossing/occupying the controlled connector terminates relevance.
        # Footprint intersection is deliberately included so relevance stops as
        # the road user begins entering the controlled movement, not only after
        # its center has fully moved onto the connector.
        entered = _direct_controlled_connector_candidates(record, controlled_ids)
        if entered:
            record_semantic_omission(
                state,
                "np:isRelevantSignal",
                "agent_already_entered_controlled_movement_signal_no_longer_relevant",
            )
            continue

        successor = _unique_controlled_successor_candidates(record, controlled_ids)
        if len(successor) == 1:
            connector_id = next(iter(successor))
            basis = "pre_entry_unique_signal_controlled_immediate_successor"
            candidates = sorted(successor)
        elif len(successor) > 1:
            record_semantic_omission(
                state,
                "np:isRelevantSignal",
                "multiple_signal_controlled_successors_intent_unknown",
            )
            continue
        else:
            continue

        signal_id = _canonical_signal_id(map_name, connector_id)
        evidence = {
            "predicate_semantics": "TrafficSignal currently governs Agent before entry into the controlled movement",
            "canonical_signal_id": signal_id,
            "dataset_adapter": "nuplan",
            "native_lane_connector_id": connector_id,
            "map_name": map_name,
            "relevance_basis": basis,
            "agent_type": str(record.get("agent_type", "UNKNOWN")),
            "agent_track_token": str(record.get("track_token", "unknown")),
            "agent_primary_map_kind": record.get("primary_map_kind"),
            "agent_primary_map_object_id": record.get("primary_map_object_id"),
            "controlled_connector_candidates": candidates,
            "pre_entry_only": True,
            "terminates_when_controlled_connector_is_entered": True,
            "abstains_when_movement_is_ambiguous": True,
            "source_fields": [
                "agent_map_membership",
                "agent_footprint_map_intersection",
                "lane_topology.outgoing_edges",
                "traffic_light_status.lane_connector_id",
            ],
        }
        assertions.append(
            derived_relation(
                "np:isRelevantSignal",
                record["entity_id"],
                signal_id,
                frame,
                TRAFFIC_LIGHT_RELEVANCE_RULE_ID,
                evidence,
            )
        )

    return assertions


def derive_traffic_light_predicates_at_snapshot(snapshot, previous_snapshot=None, state=None):
    """Derive traffic-light facts for one frame: controls, state, and relevance.

    nuPlan grounding::

        traffic_light_status.lane_connector_id -> ControlledMovement
        traffic_light_status.status            -> SignalState

    Exactly one ``hasSignalState`` relation is emitted per valid logical signal
    per frame. Duplicate native rows do not duplicate canonical assertions.
    """
    assertions = []
    if not ARGS.derive_semantic_predicates:
        return assertions

    state = state if state is not None else {}
    frame = snapshot["frame"]
    map_name = _frame_map_name(frame)

    grouped = _group_native_traffic_lights(snapshot)
    if None in grouped:
        for _ in grouped.pop(None):
            record_semantic_omission(
                state, "np:controls", "missing_native_controlled_movement_id"
            )
            record_semantic_omission(
                state, "np:hasSignalState", "missing_native_controlled_movement_id"
            )

    valid_grouped = {}
    for connector_id, items in grouped.items():
        if not _native_connector_exists(frame, connector_id):
            record_semantic_omission(
                state, "np:controls", "native_lane_connector_not_found_in_map"
            )
            record_semantic_omission(
                state, "np:hasSignalState", "native_lane_connector_not_found_in_map"
            )
            continue

        valid_grouped[connector_id] = items
        signal_id = _canonical_signal_id(map_name, connector_id)
        movement_id = _canonical_movement_id(map_name, connector_id)
        canonical_state, normalized_rows = _resolve_group_state(items, state)
        state_id = _canonical_signal_state_id(canonical_state)
        raw_statuses = [str(item.get("status", "UNKNOWN")) for item in items]

        control_evidence = {
            "predicate_semantics": "TrafficSignal controls ControlledMovement",
            "canonical_signal_id": signal_id,
            "canonical_controlled_movement_id": movement_id,
            "dataset_adapter": "nuplan",
            "native_controlled_object_type": "lane_connector",
            "native_lane_connector_id": connector_id,
            "map_name": map_name,
            "observed_signal_status": canonical_state,
            "control_relation_independent_of_signal_state": True,
            "source_fields": [
                "traffic_light_status.lane_connector_id",
                "nuplan_map.lane_connector",
            ],
        }
        assertions.append(
            derived_relation(
                "np:controls",
                signal_id,
                movement_id,
                frame,
                TRAFFIC_LIGHT_CONTROL_RULE_ID,
                control_evidence,
            )
        )

        state_evidence = {
            "predicate_semantics": "TrafficSignal has current SignalState",
            "canonical_signal_id": signal_id,
            "canonical_signal_state_id": state_id,
            "canonical_signal_state": canonical_state,
            "dataset_adapter": "nuplan",
            "native_lane_connector_id": connector_id,
            "map_name": map_name,
            "native_signal_status_rows": raw_statuses,
            "normalized_signal_status_rows": normalized_rows,
            "state_vocabulary": ["RED", "YELLOW", "GREEN", "UNKNOWN"],
            "source_fields": [
                "traffic_light_status.lane_connector_id",
                "traffic_light_status.status",
            ],
        }
        assertions.append(
            derived_relation(
                "np:hasSignalState",
                signal_id,
                state_id,
                frame,
                TRAFFIC_LIGHT_STATE_RULE_ID,
                state_evidence,
            )
        )

    assertions.extend(
        _derive_relevant_signal_assertions(snapshot, valid_grouped, map_name, state)
    )
    return assertions


def derive_traffic_control_at_snapshot(snapshot, previous_snapshot=None, state=None):
    """Backward-compatible entry point returning only ``np:controls``."""
    return [
        assertion
        for assertion in derive_traffic_light_predicates_at_snapshot(
            snapshot, previous_snapshot=previous_snapshot, state=state
        )
        if getattr(assertion, "predicate_id", None) == "np:controls"
    ]


def append_native_traffic_light_status(assertions, frame):
    """Reserved compatibility hook; canonical traffic-light facts are emitted above."""
    return None
