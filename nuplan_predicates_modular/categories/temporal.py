"""Continuity-aware entity temporal measurements and acceleration states."""
from ..base import *

def _entity_continuity(previous: Optional[dict[str, Any]], frame, state) -> bool:
    if previous is None:
        return False
    return observations_are_continuous(
        int(previous["frame_index"]),
        int(state["frame_index"]),
        int(previous["timestamp_us"]),
        int(frame.timestamp_us),
        sample_interval_s=float(ARGS.sample_interval_s),
        maximum_gap_factor=float(ARGS.maximum_temporal_gap_factor),
    )

def append_temporal_measurements(assertions, frame, state, track_key, record):
    history = state.setdefault("entity_history", {})
    current_us = int(frame.timestamp_us)
    current_frame_index = int(state["frame_index"])
    previous = history.get(track_key)
    continuous = _entity_continuity(previous, frame, state)

    if previous is None:
        total_count = 1
        total_first_us = current_us
        continuous_count = 1
        continuous_first_us = current_us
        acceleration_history = []
    else:
        total_count = int(previous.get("total_count", 0)) + 1
        total_first_us = int(previous.get("total_first_timestamp_us", current_us))
        if continuous:
            continuous_count = int(previous.get("continuous_count", 0)) + 1
            continuous_first_us = int(previous.get("continuous_first_timestamp_us", current_us))
            acceleration_history = list(previous.get("acceleration_history", []))
        else:
            continuous_count = 1
            continuous_first_us = current_us
            acceleration_history = []

    if previous is not None and continuous:
        previous_us = int(previous["timestamp_us"])
        dt_s = (current_us - previous_us) / 1e6
        px, py, ph = previous["xyh"]
        cx, cy, ch = record["xyh"]
        displacement = math.hypot(cx - px, cy - py)
        heading_change = wrap_signed(ch - ph)
        displacement_heading = (
            wrap_signed(math.atan2(cy - py, cx - px))
            if displacement >= ARGS.spatial_min_displacement_for_direction_m
            else None
        )
        record["displacement_from_previous_m"] = float(displacement)
        record["displacement_heading_rad"] = displacement_heading
        evidence = {
            "track_key": str(track_key),
            "previous_entity_id": str(previous["entity_id"]),
            "current_entity_id": str(record["entity_id"]),
            "previous_timestamp_us": previous_us,
            "current_timestamp_us": current_us,
            "continuous": True,
            "previous_frame_index": int(previous["frame_index"]),
            "current_frame_index": current_frame_index,
            "previous_x_m": float(px),
            "previous_y_m": float(py),
            "previous_heading_rad": float(ph),
            "current_x_m": float(cx),
            "current_y_m": float(cy),
            "current_heading_rad": float(ch),
            "previous_speed_mps": None if previous.get("speed") is None else float(previous["speed"]),
            "current_speed_mps": None if record.get("speed") is None else float(record["speed"]),
        }
        assertions.append(derived_relation(
            "np:precedes", previous["entity_id"], record["entity_id"], frame,
            "R-TEMPORAL-DIFFERENCE-001", evidence,
            start_time_us=previous_us, end_time_us=current_us,
        ))
        for pid, value in (
            ("np:hasDeltaTimeFromPrevious", dt_s),
            ("np:hasDisplacementFromPrevious", displacement),
            ("np:hasHeadingChangeFromPrevious", heading_change),
        ):
            assertions.append(derived_value(
                pid, record["entity_id"], float(value), ValueType.FLOAT, frame,
                "R-TEMPORAL-DIFFERENCE-001", evidence,
                start_time_us=previous_us, end_time_us=current_us,
            ))
        if displacement_heading is not None:
            assertions.append(derived_value(
                "np:hasDisplacementHeading", record["entity_id"], displacement_heading,
                ValueType.FLOAT, frame, "R-TEMPORAL-DIFFERENCE-001",
                {**evidence, "displacement_m": displacement,
                 "minimum_displacement_m": ARGS.spatial_min_displacement_for_direction_m},
                start_time_us=previous_us, end_time_us=current_us,
            ))

        if previous.get("speed") is not None and record.get("speed") is not None and dt_s > 0:
            speed_change = float(record["speed"] - previous["speed"])
            instantaneous_acceleration = speed_change / dt_s
            acceleration_history.append(instantaneous_acceleration)
            acceleration_history = acceleration_history[-max(ARGS.minimum_acceleration_samples, 5):]
            robust_acceleration = float(median(acceleration_history))
            assertions.append(derived_value(
                "np:hasSpeedChangeFromPrevious", record["entity_id"], speed_change,
                ValueType.FLOAT, frame, "R-TEMPORAL-DIFFERENCE-001", evidence,
                start_time_us=previous_us, end_time_us=current_us,
            ))
            assertions.append(derived_value(
                "np:hasEstimatedAcceleration", record["entity_id"], instantaneous_acceleration,
                ValueType.FLOAT, frame, "R-TEMPORAL-DIFFERENCE-001",
                {**evidence, "estimator": "finite_difference_speed"},
                start_time_us=previous_us, end_time_us=current_us,
            ))
            record["estimated_acceleration_mps2"] = robust_acceleration
            record["instantaneous_acceleration_mps2"] = instantaneous_acceleration
            if ARGS.derive_semantic_predicates and len(acceleration_history) >= ARGS.minimum_acceleration_samples:
                acceleration_pid = classify_acceleration_state(
                    robust_acceleration, ARGS.acceleration_threshold_mps2
                )
                if acceleration_pid is not None:
                    append_positive_boolean_semantic(
                        assertions, acceleration_pid, record["entity_id"], True,
                        frame, "R-SEMANTIC-THRESHOLD-001",
                        {
                            **evidence,
                            "robust_acceleration_mps2": robust_acceleration,
                            "samples": list(acceleration_history),
                            "threshold_mps2": ARGS.acceleration_threshold_mps2,
                            "exclusive_group": "acceleration_state",
                        },
                        state=state,
                        start_time_us=continuous_first_us,
                        end_time_us=current_us,
                    )
    elif previous is not None:
        state.setdefault("temporal_continuity_violations", []).append({
            "track_key": str(track_key),
            "previous_timestamp_us": int(previous["timestamp_us"]),
            "current_timestamp_us": current_us,
            "previous_frame_index": int(previous["frame_index"]),
            "current_frame_index": current_frame_index,
            "action": "streak_reset_no_cross_gap_derivative",
        })

    total_span = (current_us - total_first_us) / 1e6
    continuous_duration = (current_us - continuous_first_us) / 1e6
    evidence_counts = {
        "track_key": str(track_key),
        "current_timestamp_us": current_us,
        "current_frame_index": current_frame_index,
        "total_first_timestamp_us": total_first_us,
        "continuous_first_timestamp_us": continuous_first_us,
        "total_count": total_count,
        "continuous_count": continuous_count,
        "continuous": continuous,
    }
    count_values = (
        ("np:hasObservedFrameCount", continuous_count, ValueType.INTEGER),
        ("np:hasObservedDuration", continuous_duration, ValueType.FLOAT),
        ("np:hasTotalObservedFrameCount", total_count, ValueType.INTEGER),
        ("np:hasTotalObservedSpan", total_span, ValueType.FLOAT),
        ("np:hasContinuousObservedFrameCount", continuous_count, ValueType.INTEGER),
        ("np:hasContinuousObservedDuration", continuous_duration, ValueType.FLOAT),
    )
    for pid, value, value_type in count_values:
        assertions.append(derived_value(
            pid, record["entity_id"], value, value_type, frame,
            "R-TEMPORAL-DIFFERENCE-001", evidence_counts,
            start_time_us=continuous_first_us if "Continuous" in pid or pid in {"np:hasObservedFrameCount", "np:hasObservedDuration"} else total_first_us,
            end_time_us=current_us,
        ))

    record["observed_frame_count"] = continuous_count
    record["observed_duration_s"] = continuous_duration
    record["total_observed_frame_count"] = total_count
    record["total_observed_span_s"] = total_span
    record["continuous_with_previous"] = continuous

    history[track_key] = {
        "entity_id": record["entity_id"],
        "timestamp_us": current_us,
        "frame_index": current_frame_index,
        "total_first_timestamp_us": total_first_us,
        "continuous_first_timestamp_us": continuous_first_us,
        "total_count": total_count,
        "continuous_count": continuous_count,
        "xyh": record["xyh"],
        "speed": record.get("speed"),
        "acceleration_history": acceleration_history,
    }


def append_pair_temporal_measurements(assertions, frame, state, pair_id, pair_key, context):
    """Emit pair continuity, distance changes, observed counts, and update history."""
    kin = context["kinematics"]
    geometry = context["geometry"]
    evidence = context["evidence"]
    history = state.setdefault("pair_history", {})
    current_us = int(frame.timestamp_us)
    previous = history.get(pair_key)
    continuous = False
    if previous is not None:
        continuous = observations_are_continuous(
            int(previous["frame_index"]), int(state["frame_index"]),
            int(previous["timestamp_us"]), current_us,
            sample_interval_s=ARGS.sample_interval_s,
            maximum_gap_factor=ARGS.maximum_temporal_gap_factor,
        )
    if continuous:
        frame_count = int(previous["frame_count"]) + 1
        first_us = int(previous["first_timestamp_us"])
        center_change = kin["center_distance_m"] - previous["center_distance_m"]
        assertions.append(
            derived_value(
                "np:hasCenterDistanceChangeFromPrevious", pair_id, center_change,
                ValueType.FLOAT, frame, "R-PAIR-HISTORY-001",
                {**evidence,
                 "previous_timestamp_us": int(previous["timestamp_us"]),
                 "current_timestamp_us": current_us,
                 "previous_frame_index": int(previous["frame_index"]),
                 "current_frame_index": int(state["frame_index"]),
                 "previous_center_distance_m": float(previous["center_distance_m"]),
                 "current_center_distance_m": float(kin["center_distance_m"])},
                start_time_us=int(previous["timestamp_us"]), end_time_us=current_us,
            )
        )
        if (
            geometry.get("free_space_distance_m") is not None
            and previous.get("free_space_distance_m") is not None
        ):
            free_change = (
                geometry["free_space_distance_m"] - previous["free_space_distance_m"]
            )
            assertions.append(
                derived_value(
                    "np:hasFreeSpaceDistanceChangeFromPrevious", pair_id, free_change,
                    ValueType.FLOAT, frame, "R-PAIR-HISTORY-001",
                    {**evidence,
                     "previous_timestamp_us": int(previous["timestamp_us"]),
                     "current_timestamp_us": current_us,
                     "previous_frame_index": int(previous["frame_index"]),
                     "current_frame_index": int(state["frame_index"]),
                     "previous_free_space_distance_m": float(previous["free_space_distance_m"]),
                     "current_free_space_distance_m": float(geometry["free_space_distance_m"])},
                    start_time_us=int(previous["timestamp_us"]), end_time_us=current_us,
                )
            )
    else:
        frame_count = 1
        first_us = current_us

    duration_s = (current_us - first_us) / 1e6
    assertions.append(
        derived_value(
            "np:hasPairObservedFrameCount", pair_id, frame_count, ValueType.INTEGER,
            frame, "R-PAIR-HISTORY-001", {**evidence, "first_timestamp_us": first_us, "current_timestamp_us": current_us, "current_frame_index": int(state["frame_index"]), "frame_count": frame_count},
            start_time_us=first_us, end_time_us=current_us,
        )
    )
    assertions.append(
        derived_value(
            "np:hasPairObservedDuration", pair_id, duration_s, ValueType.FLOAT,
            frame, "R-PAIR-HISTORY-001", {**evidence, "first_timestamp_us": first_us, "current_timestamp_us": current_us, "current_frame_index": int(state["frame_index"]), "frame_count": frame_count},
            start_time_us=first_us, end_time_us=current_us,
        )
    )
    history[pair_key] = {
        "frame_index": int(state["frame_index"]),
        "timestamp_us": current_us,
        "frame_count": frame_count,
        "first_timestamp_us": first_us,
        "center_distance_m": kin["center_distance_m"],
        "free_space_distance_m": geometry.get("free_space_distance_m"),
    }
