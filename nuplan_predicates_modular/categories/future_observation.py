"""Retrospective observed-future trajectories and risk predicates."""
from ..base import *
from .geometry import footprint_metrics
from .snapshot_utils import _pair_id_for_snapshot, _snapshot_pairs, _records_by_token

def _interpolate_angle(a: float, b: float, ratio: float) -> float:
    return wrap_signed(float(a) + ratio * wrap_signed(float(b) - float(a)))


def _interpolate_record(sequence, target_us: int) -> Optional[dict[str, Any]]:
    timestamps = [int(snapshot["timestamp_us"]) for snapshot, _ in sequence]
    pos = bisect_left(timestamps, int(target_us))
    if pos < len(timestamps) and timestamps[pos] == int(target_us):
        return sequence[pos][1]
    if pos == 0 or pos >= len(sequence):
        return None
    left_snapshot, left = sequence[pos - 1]
    right_snapshot, right = sequence[pos]
    left_us = int(left_snapshot["timestamp_us"])
    right_us = int(right_snapshot["timestamp_us"])
    dt_s = (right_us - left_us) / 1e6
    if dt_s <= 0 or dt_s > ARGS.sample_interval_s * ARGS.maximum_temporal_gap_factor:
        return None
    ratio = (int(target_us) - left_us) / (right_us - left_us)
    lx, ly, lh = left["xyh"]
    rx, ry, rh = right["xyh"]
    velocity = None
    if left.get("velocity") is not None and right.get("velocity") is not None:
        velocity = (
            left["velocity"][0] + ratio * (right["velocity"][0] - left["velocity"][0]),
            left["velocity"][1] + ratio * (right["velocity"][1] - left["velocity"][1]),
        )
    result = dict(left)
    result["entity_id"] = f"interpolated:{left['track_token']}:{target_us}"
    result["xyh"] = (
        lx + ratio * (rx - lx),
        ly + ratio * (ry - ly),
        _interpolate_angle(lh, rh, ratio),
    )
    result["velocity"] = velocity
    result["speed"] = None if velocity is None else math.hypot(*velocity)
    result["interpolated"] = True
    return result


def derive_future_predicates(snapshots):
    assertions = []
    if not ARGS.derive_observed_future_labels:
        return assertions
    tracks = _records_by_token(snapshots)
    sequence_by_token = {str(token): sequence for token, sequence in tracks.items()}
    horizon_us = int(round(ARGS.future_horizon_s * 1e6))
    step_us = int(round(ARGS.future_step_s * 1e6))
    expected_samples = max(1, int(math.floor(ARGS.future_horizon_s / ARGS.future_step_s)))

    # Retrospective links to actual later observations.
    for token, sequence in sequence_by_token.items():
        for i, (snapshot, record) in enumerate(sequence):
            frame = snapshot["frame"]
            for future_snapshot, future_record in sequence[i + 1:]:
                dt_us = int(future_snapshot["timestamp_us"]) - int(snapshot["timestamp_us"])
                if dt_us <= 0:
                    continue
                if dt_us > horizon_us:
                    break
                evidence = {
                    "future_delta_s": dt_us / 1e6,
                    "track_token": token,
                    "evidence_level": "retrospective",
                    "temporal_scope": "observed_future",
                }
                assertions.append(derived_relation(
                    "np:hasFuturePosition", record["entity_id"], future_record["entity_id"],
                    frame, "R-FUTURE-OBSERVATION-001", evidence,
                    start_time_us=int(snapshot["timestamp_us"]),
                    end_time_us=int(future_snapshot["timestamp_us"]),
                ))
                assertions.append(derived_relation(
                    "np:hasFutureFootprint", record["entity_id"], future_record["entity_id"],
                    frame, "R-FUTURE-OBSERVATION-001", evidence,
                    start_time_us=int(snapshot["timestamp_us"]),
                    end_time_us=int(future_snapshot["timestamp_us"]),
                ))

    for current_index, snapshot in enumerate(snapshots):
        current_time = int(snapshot["timestamp_us"])
        frame = snapshot["frame"]
        for subject, obj, pair_tag in _snapshot_pairs(snapshot):
            subject_sequence = sequence_by_token.get(str(subject["track_token"]), [])
            object_sequence = sequence_by_token.get(str(obj["track_token"]), [])
            pair_id = _pair_id_for_snapshot(snapshot, subject, obj, pair_tag)
            min_distance = math.inf
            min_time_s = None
            observed_ttc_s = None
            sample_count = 0
            for sample_index in range(1, expected_samples + 1):
                target_us = current_time + sample_index * step_us
                if target_us > current_time + horizon_us:
                    break
                future_subject = _interpolate_record(subject_sequence, target_us)
                future_object = _interpolate_record(object_sequence, target_us)
                if future_subject is None or future_object is None:
                    continue
                geometry = footprint_metrics(future_subject, future_object)
                free_space = geometry.get("free_space_distance_m")
                if free_space is None:
                    continue
                sample_count += 1
                dt_s = (target_us - current_time) / 1e6
                if free_space < min_distance:
                    min_distance = free_space
                    min_time_s = dt_s
                if geometry.get("overlapping") is True and observed_ttc_s is None:
                    observed_ttc_s = dt_s

            coverage = sample_count / expected_samples
            available_duration = sample_count * ARGS.future_step_s
            evidence = {
                "future_horizon_s": ARGS.future_horizon_s,
                "future_step_s": ARGS.future_step_s,
                "expected_samples": expected_samples,
                "future_samples": sample_count,
                "future_coverage_ratio": coverage,
                "subject_track_token": subject["track_token"],
                "object_track_token": obj["track_token"],
                "evidence_level": "retrospective",
                "temporal_scope": "observed_future",
                "current_frame_excluded_from_ttc": True,
            }
            assertions.append(derived_value(
                "np:hasAvailableFutureDuration", pair_id, float(available_duration),
                ValueType.FLOAT, frame, "R-FUTURE-OBSERVATION-001", evidence,
            ))
            assertions.append(derived_value(
                "np:hasFutureSampleCount", pair_id, int(sample_count),
                ValueType.INTEGER, frame, "R-FUTURE-OBSERVATION-001", evidence,
            ))
            assertions.append(derived_value(
                "np:hasFutureCoverageRatio", pair_id, float(coverage),
                ValueType.FLOAT, frame, "R-FUTURE-OBSERVATION-001", evidence,
            ))
            sufficient = sample_count > 0 and coverage >= ARGS.minimum_future_coverage_ratio
            if not sufficient:
                append_positive_boolean_semantic(
                    assertions, "np:hasInsufficientFutureEvidence", pair_id, True,
                    frame, "R-FUTURE-OBSERVATION-001", evidence,
                )
                continue

            # v9.5.44: risk semantics are intentionally no longer derived from
            # observed future trajectories.  The future_observation category keeps
            # its retrospective coverage/link facts only; the risk category is
            # rebuilt independently from current-state evidence.

    return assertions
