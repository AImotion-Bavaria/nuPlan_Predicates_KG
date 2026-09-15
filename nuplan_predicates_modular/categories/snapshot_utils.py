"""Shared scenario-snapshot indexing helpers."""
from ..base import *


def _pair_id_for_snapshot(snapshot, subject, obj, pair_tag):
    return (
        f"pair:{snapshot['scenario_token']}:{snapshot['timestamp_us']}:"
        f"{pair_tag}:{stable_id(subject['entity_id'], obj['entity_id'])}"
    )


def _snapshot_pairs(snapshot):
    entities = snapshot["entities"]
    ego = entities["ego"]
    for token in snapshot.get("relevant_agent_tokens", []):
        agent = entities.get(token)
        if agent is not None:
            yield ego, agent, "ego_to_agent"
            yield agent, ego, "agent_to_ego"
    for first, second in snapshot.get("relevant_agent_agent_pairs", []):
        a = entities.get(first)
        b = entities.get(second)
        if a is not None and b is not None:
            yield a, b, "agent_to_agent"
            yield b, a, "agent_to_agent_reverse"


def _records_by_token(snapshots):
    result = defaultdict(list)
    for snapshot in snapshots:
        for token, record in snapshot["entities"].items():
            result[str(token)].append((snapshot["timestamp_us"], record))
    return result
