"""Conservative inferred intent candidates: yielding, waiting, gap creation, competition."""
from ..base import *
from .geometry import relative_kinematics
from .snapshot_utils import _snapshot_pairs, _records_by_token

def _future_records(sequence, current_us: int, horizon_us: int):
    return [
        (snapshot, record) for snapshot, record in sequence
        if current_us <= int(snapshot["timestamp_us"]) <= current_us + horizon_us
    ]


def _common_conflict_region(subject_future, object_future, subject, obj):
    if not SHAPELY_AVAILABLE or len(subject_future) < 2 or len(object_future) < 2:
        return None
    LineString = __import__("shapely.geometry", fromlist=["LineString"]).LineString
    subject_line = LineString([(r["xyh"][0], r["xyh"][1]) for _, r in subject_future])
    object_line = LineString([(r["xyh"][0], r["xyh"][1]) for _, r in object_future])
    subject_width = float(subject["dimensions"].get("width_m") or 1.0)
    object_width = float(obj["dimensions"].get("width_m") or 1.0)
    region = subject_line.buffer(subject_width / 2.0).intersection(object_line.buffer(object_width / 2.0))
    if region.is_empty:
        return None
    centroid = region.centroid
    return {
        "geometry": region,
        "point": (float(centroid.x), float(centroid.y)),
        "region_id": stable_id(round(float(centroid.x), 2), round(float(centroid.y), 2), round(float(region.area), 2)),
        "area_m2": float(region.area),
    }


def _arrival_time_to_conflict(future_sequence, current_us: int, conflict_point, radius_m: float) -> float:
    cx, cy = conflict_point
    for snapshot, record in future_sequence:
        x, y, _ = record["xyh"]
        if math.hypot(x - cx, y - cy) <= radius_m:
            return (int(snapshot["timestamp_us"]) - current_us) / 1e6
    return math.inf


def derive_intent_predicates(snapshots):
    assertions = []
    if not (ARGS.derive_semantic_predicates and ARGS.derive_intent_candidates):
        return assertions
    tracks = _records_by_token(snapshots)
    sequence_by_token = {str(token): sequence for token, sequence in tracks.items()}
    horizon_us = int(ARGS.future_horizon_s * 1e6)
    streaks = {}
    gap_starts = {}
    previous_snapshot = None

    for snapshot in snapshots:
        frame = snapshot["frame"]
        current_us = int(snapshot["timestamp_us"])
        current_frame_index = int(snapshot.get("frame_index", 0))
        continuous = False
        if previous_snapshot is not None:
            continuous = observations_are_continuous(
                int(previous_snapshot.get("frame_index", current_frame_index - 1)), current_frame_index,
                int(previous_snapshot["timestamp_us"]), current_us,
                sample_interval_s=ARGS.sample_interval_s,
                maximum_gap_factor=ARGS.maximum_temporal_gap_factor,
            )
        yielding_pairs = set()
        for subject, obj, pair_tag in _snapshot_pairs(snapshot):
            if not (is_vehicle_like(subject) and is_vehicle_like(obj)):
                continue
            subject_future = _future_records(sequence_by_token.get(str(subject["track_token"]), []), current_us, horizon_us)
            object_future = _future_records(sequence_by_token.get(str(obj["track_token"]), []), current_us, horizon_us)
            conflict = _common_conflict_region(subject_future, object_future, subject, obj)
            if conflict is None:
                continue
            radius = max(
                float(subject["dimensions"].get("width_m") or 1.0),
                float(obj["dimensions"].get("width_m") or 1.0),
            )
            subject_arrival = _arrival_time_to_conflict(subject_future, current_us, conflict["point"], radius)
            object_arrival = _arrival_time_to_conflict(object_future, current_us, conflict["point"], radius)
            subject_speed = float(subject.get("speed") or 0.0)
            object_speed = float(obj.get("speed") or 0.0)
            subject_accel = subject.get("estimated_acceleration_mps2")
            subject_decelerating = subject_accel is not None and subject_accel <= -ARGS.acceleration_threshold_mps2
            subject_stopped = subject_speed <= ARGS.stopped_speed_mps
            object_clears_first = math.isfinite(object_arrival) and (
                not math.isfinite(subject_arrival) or object_arrival + ARGS.arrival_time_tolerance_s < subject_arrival
            )
            evidence = {
                "evidence_level": "inferred",
                "confidence": 0.75,
                "conflict_region_id": conflict["region_id"],
                "conflict_region_area_m2": conflict["area_m2"],
                "subject_arrival_s": subject_arrival,
                "object_arrival_s": object_arrival,
                "subject_speed_mps": subject_speed,
                "object_speed_mps": object_speed,
                "subject_estimated_acceleration_mps2": subject_accel,
                "future_evidence_horizon_s": ARGS.future_horizon_s,
            }

            yield_key = (str(subject["track_token"]), str(obj["track_token"]), conflict["region_id"], "yield")
            yield_condition = bool((subject_decelerating or subject_stopped) and object_clears_first)
            yield_streak = update_condition_streak(
                streaks, yield_key,
                condition=yield_condition,
                current_timestamp_us=current_us,
                current_frame_index=current_frame_index,
                continuous_with_previous=continuous,
            )
            if yield_condition and yield_streak.duration_s >= ARGS.minimum_intent_duration_s:
                yielding_pairs.add((str(subject["track_token"]), str(obj["track_token"]), conflict["region_id"]))
                assertions.append(derived_relation(
                    "np:yieldingTo", subject["entity_id"], obj["entity_id"], frame,
                    "R-INTENT-INTERACTION-001",
                    {**evidence, "condition_duration_s": yield_streak.duration_s},
                    start_time_us=yield_streak.start_timestamp_us, end_time_us=current_us,
                ))

            wait_key = (str(subject["track_token"]), str(obj["track_token"]), conflict["region_id"], "wait")
            object_blocks_soon = math.isfinite(object_arrival) and object_arrival <= max(ARGS.minimum_intent_duration_s * 2.0, 3.0)
            wait_condition = bool(subject_stopped and object_blocks_soon)
            wait_streak = update_condition_streak(
                streaks, wait_key,
                condition=wait_condition,
                current_timestamp_us=current_us,
                current_frame_index=current_frame_index,
                continuous_with_previous=continuous,
            )
            if wait_condition and wait_streak.duration_s >= ARGS.minimum_intent_duration_s:
                assertions.append(derived_relation(
                    "np:waitingFor", subject["entity_id"], obj["entity_id"], frame,
                    "R-INTENT-INTERACTION-001",
                    {**evidence, "condition_duration_s": wait_streak.duration_s},
                    start_time_us=wait_streak.start_timestamp_us, end_time_us=current_us,
                ))

            # Gap creation is emitted only when the receiving object is observed
            # to enter the subject's current primary lane and the measured gap
            # increases continuously by the configured amount.
            kin = relative_kinematics(subject, obj)
            gap_key = (str(subject["track_token"]), str(obj["track_token"]), "gap")
            future_enters_subject_lane = bool(
                subject.get("primary_map_id")
                and any(r.get("primary_map_id") == subject.get("primary_map_id") for _, r in object_future[1:])
                and obj.get("primary_map_id") != subject.get("primary_map_id")
            )
            current_gap = float(kin["center_distance_m"])
            if not continuous or gap_key not in gap_starts or not future_enters_subject_lane:
                gap_starts[gap_key] = {"distance_m": current_gap, "timestamp_us": current_us, "frame_index": current_frame_index}
            gap_increase = current_gap - float(gap_starts[gap_key]["distance_m"])
            gap_duration = (current_us - int(gap_starts[gap_key]["timestamp_us"])) / 1e6
            if (
                future_enters_subject_lane
                and subject_decelerating
                and gap_increase >= ARGS.gap_increase_threshold_m
                and gap_duration >= ARGS.minimum_intent_duration_s
            ):
                assertions.append(derived_relation(
                    "np:creatingGapFor", subject["entity_id"], obj["entity_id"], frame,
                    "R-INTENT-INTERACTION-001",
                    {**evidence, "measured_gap_increase_m": gap_increase,
                     "gap_condition_duration_s": gap_duration,
                     "receiving_object_later_enters_subject_lane": True},
                    start_time_us=int(gap_starts[gap_key]["timestamp_us"]), end_time_us=current_us,
                ))

            canonical = canonical_symmetric_pair(subject["track_token"], obj["track_token"])
            object_accel = obj.get("estimated_acceleration_mps2")
            object_decelerating = object_accel is not None and object_accel <= -ARGS.acceleration_threshold_mps2
            object_stopped = object_speed <= ARGS.stopped_speed_mps
            subject_clears_first = math.isfinite(subject_arrival) and (
                not math.isfinite(object_arrival) or subject_arrival + ARGS.arrival_time_tolerance_s < object_arrival
            )
            reverse_yield_condition = bool((object_decelerating or object_stopped) and subject_clears_first)
            no_verified_yield = not yield_condition and not reverse_yield_condition
            competing_condition = bool(
                math.isfinite(subject_arrival)
                and math.isfinite(object_arrival)
                and abs(subject_arrival - object_arrival) <= ARGS.arrival_time_tolerance_s
                and no_verified_yield
            )
            compete_key = (*canonical, conflict["region_id"], "compete")
            compete_streak = update_condition_streak(
                streaks, compete_key,
                condition=competing_condition,
                current_timestamp_us=current_us,
                current_frame_index=current_frame_index,
                continuous_with_previous=continuous,
            )
            if (
                competing_condition
                and compete_streak.duration_s >= ARGS.minimum_intent_duration_s
                and str(subject["track_token"]) == canonical[0]
            ):
                assertions.append(derived_relation(
                    "np:competingForGapWith", subject["entity_id"], obj["entity_id"],
                    frame, "R-INTENT-INTERACTION-001",
                    {**evidence, "condition_duration_s": compete_streak.duration_s,
                     "canonical_symmetric_pair": list(canonical)},
                    start_time_us=compete_streak.start_timestamp_us, end_time_us=current_us,
                ))
        previous_snapshot = snapshot
    return assertions
