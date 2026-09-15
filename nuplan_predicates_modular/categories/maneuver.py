"""Lane keeping, lane changes, connector transitions, and turn predicates."""
from ..base import *
from .map import _adjacent_lane_ids
from .snapshot_utils import _records_by_token

def _primary_lane_token(record: dict[str, Any]) -> Optional[str]:
    value = record.get("primary_lane_id")
    return None if value is None else str(value).split(":", 1)[1]


def _primary_connector_token(record: dict[str, Any]) -> Optional[str]:
    value = record.get("primary_connector_id")
    return None if value is None else str(value).split(":", 1)[1]


def _stable_value(records: list[dict[str, Any]], end_index: int, frames: int, getter) -> Optional[str]:
    start = end_index - frames + 1
    if start < 0:
        return None
    values = [getter(records[i]) for i in range(start, end_index + 1)]
    if values[0] is None or any(value != values[0] for value in values):
        return None
    return values[0]


def derive_maneuver_predicates(snapshots):
    assertions = []
    if not ARGS.derive_semantic_predicates:
        return assertions
    tracks = _records_by_token(snapshots)
    stable_frames = int(ARGS.lane_stability_frames)
    for token, sequence in tracks.items():
        records = [record for _, record in sequence]
        connector_state = None
        for index in range(1, len(sequence)):
            previous_snapshot, previous = sequence[index - 1]
            snapshot, current = sequence[index]
            frame = snapshot["frame"]
            previous_lane = _primary_lane_token(previous)
            current_lane = _primary_lane_token(current)
            previous_connector = _primary_connector_token(previous)
            current_connector = _primary_connector_token(current)
            event_emitted = False

            if previous_lane is not None and current_connector is not None and current_connector != previous_connector:
                connector_state = {
                    "connector_id": current_connector,
                    "entry_timestamp_us": int(snapshot["timestamp_us"]),
                    "entry_heading_rad": float(current["xyh"][2]),
                    "last_heading_rad": float(current["xyh"][2]),
                    "accumulated_heading_change_rad": 0.0,
                    "incoming_lane_id": previous_lane,
                }
                assertions.append(derived_relation(
                    "np:entersLaneConnector", current["entity_id"], f"lane_connector:{current_connector}",
                    frame, "R-MANEUVER-TOPOLOGY-001",
                    {"incoming_lane_id": previous_lane, "connector_id": current_connector},
                    start_time_us=int(previous_snapshot["timestamp_us"]),
                    end_time_us=int(snapshot["timestamp_us"]),
                ))
                event_emitted = True

            if connector_state is not None:
                heading = float(current["xyh"][2])
                connector_state["accumulated_heading_change_rad"] += wrap_signed(
                    heading - float(connector_state["last_heading_rad"])
                )
                connector_state["last_heading_rad"] = heading

            if connector_state is not None and previous_connector is not None and current_lane is not None:
                assertions.append(derived_relation(
                    "np:exitsLaneConnector", current["entity_id"], f"lane:{current_lane}",
                    frame, "R-MANEUVER-TOPOLOGY-001",
                    {
                        "connector_id": connector_state["connector_id"],
                        "incoming_lane_id": connector_state["incoming_lane_id"],
                        "outgoing_lane_id": current_lane,
                    },
                    start_time_us=connector_state["entry_timestamp_us"],
                    end_time_us=int(snapshot["timestamp_us"]),
                ))
                change = float(connector_state["accumulated_heading_change_rad"])
                if abs(change) >= ARGS.uturn_angle_threshold_rad:
                    maneuver_pid = "np:makesUTurn"
                elif change >= ARGS.turn_angle_threshold_rad:
                    maneuver_pid = "np:turnsLeft"
                elif change <= -ARGS.turn_angle_threshold_rad:
                    maneuver_pid = "np:turnsRight"
                else:
                    maneuver_pid = "np:goesStraight"
                append_positive_boolean_semantic(
                    assertions, maneuver_pid, current["entity_id"], True,
                    frame, "R-MANEUVER-TOPOLOGY-001",
                    {
                        "connector_id": connector_state["connector_id"],
                        "incoming_lane_id": connector_state["incoming_lane_id"],
                        "outgoing_lane_id": current_lane,
                        "accumulated_heading_change_rad": change,
                        "exclusive_group": "connector_maneuver",
                    },
                    start_time_us=connector_state["entry_timestamp_us"],
                    end_time_us=int(snapshot["timestamp_us"]),
                )
                connector_state = None
                event_emitted = True

            # Completed stable lane-to-lane transition outside connectors.
            target_lane = _stable_value(records, index, stable_frames, _primary_lane_token)
            source_end = index - stable_frames
            source_lane = _stable_value(records, source_end, stable_frames, _primary_lane_token) if source_end >= 0 else None
            if (
                target_lane is not None
                and source_lane is not None
                and target_lane != source_lane
                and current_connector is None
                and previous_connector is None
            ):
                left_ids, right_ids = _adjacent_lane_ids(frame.map_api, source_lane)
                left_match = target_lane in left_ids
                right_match = target_lane in right_ids
                source_record = records[source_end]
                sx, sy, sh = source_record["xyh"]
                cx, cy, _ = current["xyh"]
                lateral_delta = -math.sin(sh) * (cx - sx) + math.cos(sh) * (cy - sy)
                if not left_match and not right_match and abs(lateral_delta) >= ARGS.lane_change_lateral_threshold_m:
                    left_match = lateral_delta > 0
                    right_match = lateral_delta < 0
                lane_change_pid = None
                if left_match and not right_match:
                    lane_change_pid = "np:changesLaneLeft"
                elif right_match and not left_match:
                    lane_change_pid = "np:changesLaneRight"
                if lane_change_pid is not None:
                    start_snapshot, _ = sequence[max(0, source_end - stable_frames + 1)]
                    append_positive_boolean_semantic(
                        assertions, lane_change_pid, current["entity_id"], True,
                        frame, "R-MANEUVER-TOPOLOGY-001",
                        {
                            "source_lane_id": source_lane,
                            "target_lane_id": target_lane,
                            "lateral_displacement_m": lateral_delta,
                            "source_stability_frames": stable_frames,
                            "target_stability_frames": stable_frames,
                            "exclusive_group": "lane_maneuver",
                        },
                        start_time_us=int(start_snapshot["timestamp_us"]),
                        end_time_us=int(snapshot["timestamp_us"]),
                    )
                    event_emitted = True

            if (
                not event_emitted
                and previous_lane is not None
                and current_lane == previous_lane
                and previous_connector is None
                and current_connector is None
            ):
                append_positive_boolean_semantic(
                    assertions, "np:keepsLane", current["entity_id"], True,
                    frame, "R-MANEUVER-TOPOLOGY-001",
                    {"lane_id": current_lane, "exclusive_group": "lane_maneuver"},
                    start_time_us=int(previous_snapshot["timestamp_us"]),
                    end_time_us=int(snapshot["timestamp_us"]),
                )
    return assertions
