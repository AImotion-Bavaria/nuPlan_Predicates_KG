"""Entity and pairwise motion measurements plus entity motion-state semantics."""
from ..base import *
from .geometry import get_geometric_center_pose


def _body_vector_to_global(
    longitudinal: float, lateral: float, heading_rad: float
) -> tuple[float, float]:
    """Rotate a vector from the vehicle body frame into the global map frame."""
    c = math.cos(float(heading_rad))
    s = math.sin(float(heading_rad))
    return (
        float(c * longitudinal - s * lateral),
        float(s * longitudinal + c * lateral),
    )


def get_velocity(entity: Any) -> Optional[tuple[float, float]]:
    """Return geometric-center velocity in the global map frame.

    nuPlan stores ``EgoState.dynamic_car_state`` velocity components in the
    vehicle frame: x is longitudinal and y is lateral.  Tracked-object
    ``velocity`` components are already expressed in the global map frame.
    v9.5.5 treated the EGO body-frame components as global x/y values, which
    made EGO forward speed depend incorrectly on the road's absolute heading.
    That suppressed ``moving_headway`` for EGO on most road orientations while
    leaving stopped queue detections unaffected.
    """
    dynamic = getattr(entity, "dynamic_car_state", None)
    if dynamic is not None:
        try:
            _, _, heading = get_geometric_center_pose(entity)
        except Exception:
            heading = None

        center_velocity = getattr(dynamic, "center_velocity_2d", None)
        if center_velocity is not None:
            longitudinal = float(center_velocity.x)
            lateral = float(center_velocity.y)
            if heading is None:
                return longitudinal, lateral
            return _body_vector_to_global(longitudinal, lateral, heading)

        rear_velocity = getattr(dynamic, "rear_axle_velocity_2d", None)
        if rear_velocity is not None:
            longitudinal = float(rear_velocity.x)
            lateral = float(rear_velocity.y)
            omega = float(getattr(dynamic, "angular_velocity", 0.0) or 0.0)

            # DynamicCarState defines the rear-axle-to-center displacement in
            # the vehicle frame as (distance, 0).  Shift the local velocity to
            # the geometric center before rotating it into the global frame.
            rear_to_center = None
            footprint = getattr(entity, "car_footprint", None)
            if footprint is not None:
                rear_to_center = getattr(footprint, "rear_axle_to_center_dist", None)
            try:
                rear_to_center = float(rear_to_center)
            except (TypeError, ValueError):
                rear_to_center = None
            if rear_to_center is None:
                rear_axle = getattr(entity, "rear_axle", None)
                try:
                    cx, cy, _ = get_geometric_center_pose(entity)
                    rx, ry = float(rear_axle.x), float(rear_axle.y)
                    rear_to_center = math.hypot(cx - rx, cy - ry)
                except Exception:
                    rear_to_center = 0.0

            center_longitudinal = longitudinal
            center_lateral = lateral + omega * float(rear_to_center)
            if heading is None:
                return center_longitudinal, center_lateral
            return _body_vector_to_global(
                center_longitudinal, center_lateral, heading
            )

    velocity = getattr(entity, "velocity", None)
    if velocity is None:
        return None
    return float(velocity.x), float(velocity.y)


def get_ego_acceleration(entity: Any) -> Optional[tuple[float, float]]:
    dynamic = getattr(entity, "dynamic_car_state", None)
    if dynamic is None:
        return None
    center_acceleration = getattr(dynamic, "center_acceleration_2d", None)
    if center_acceleration is not None:
        return float(center_acceleration.x), float(center_acceleration.y)
    acceleration = getattr(dynamic, "rear_axle_acceleration_2d", None)
    if acceleration is None:
        return None
    return float(acceleration.x), float(acceleration.y)


def append_entity_motion_measurements(
    assertions, frame, record, entity, source_prefix, state=None
):
    velocity = get_velocity(entity)
    record["velocity"] = velocity
    if velocity is None:
        return

    entity_id = record["entity_id"]
    vx, vy = velocity
    speed = math.hypot(vx, vy)
    velocity_heading = None
    if speed >= ARGS.spatial_min_direction_speed_mps:
        velocity_heading = wrap_signed(math.atan2(vy, vx))

    record["speed"] = speed
    record["velocity_heading_rad"] = velocity_heading
    assertions.append(
        native_value(
            "np:hasVelocity", entity_id, [float(vx), float(vy)], ValueType.VECTOR2, frame,
            [f"{source_prefix}.velocity.x", f"{source_prefix}.velocity.y"],
        )
    )
    assertions.append(
        native_value(
            "np:hasVelocityX", entity_id, vx, ValueType.FLOAT, frame,
            [f"{source_prefix}.velocity.x"],
        )
    )
    assertions.append(
        native_value(
            "np:hasVelocityY", entity_id, vy, ValueType.FLOAT, frame,
            [f"{source_prefix}.velocity.y"],
        )
    )
    assertions.append(
        derived_value(
            "np:hasSpeed", entity_id, speed, ValueType.FLOAT, frame,
            "R-SPEED-MAGNITUDE-001",
            {
                "velocity_x_mps": vx,
                "velocity_y_mps": vy,
                "reference_point": "geometric_center",
                "coordinate_frame": "global_map",
                "ego_native_velocity_frame": "vehicle_body_rotated_to_global_map",
            },
        )
    )
    if velocity_heading is not None:
        assertions.append(
            derived_value(
                "np:hasVelocityHeading", entity_id, velocity_heading,
                ValueType.FLOAT, frame, "R-MAP-MOTION-SPATIAL-001",
                {
                    "velocity_x_mps": vx,
                    "velocity_y_mps": vy,
                    "minimum_direction_speed_mps": ARGS.spatial_min_direction_speed_mps,
                },
            )
        )

    agent_type = str(record.get("agent_type", "UNKNOWN")).upper()
    if ARGS.derive_semantic_predicates and not any(
        marker in agent_type for marker in STATIC_TYPE_MARKERS
    ):
        state_pid = classify_speed_state(
            speed, ARGS.stopped_speed_mps, ARGS.moving_speed_mps
        )
        if state_pid is not None:
            append_positive_boolean_semantic(
                assertions, state_pid, entity_id, True, frame,
                "R-SEMANTIC-THRESHOLD-001",
                {
                    "speed_mps": speed,
                    "stopped_speed_mps": ARGS.stopped_speed_mps,
                    "moving_speed_mps": ARGS.moving_speed_mps,
                    "exclusive_group": "speed_state",
                },
                state=state,
            )


def append_pair_motion_measurements(assertions, frame, pair_id, context):
    kin = context["kinematics"]
    spatial = context["spatial"]
    evidence = context["evidence"]
    values = (
        ("np:hasRelativeSpeedTo", kin.get("relative_speed_mps")),
        ("np:hasLongitudinalRelativeSpeedTo", kin.get("relative_longitudinal_speed_mps")),
        ("np:hasLateralRelativeSpeedTo", kin.get("relative_lateral_speed_mps")),
        ("np:hasClosingSpeedTo", kin.get("closing_speed_mps")),
        ("np:hasVelocityTowardTarget", kin.get("velocity_toward_target_mps")),
        ("np:hasSubjectForwardSpeed", kin.get("subject_forward_speed_mps")),
        ("np:hasTravelDirectionDifferenceTo", spatial.travel_direction_difference_rad),
    )
    for pid, value in values:
        if value is not None and math.isfinite(float(value)):
            rule_id = (
                "R-MAP-MOTION-SPATIAL-001"
                if pid == "np:hasTravelDirectionDifferenceTo"
                else "R-RELATIVE-KINEMATICS-001"
            )
            assertions.append(
                derived_value(pid, pair_id, float(value), ValueType.FLOAT, frame, rule_id, evidence)
            )


def append_ego_acceleration_measurements(assertions, frame, ego_id, ego_state):
    acceleration = get_ego_acceleration(ego_state)
    if acceleration is None:
        return
    ax, ay = acceleration
    assertions.append(
        native_value(
            "np:hasAccelerationX", ego_id, ax, ValueType.FLOAT, frame,
            ["ego_state.center_or_rear_axle_acceleration_2d.x"],
        )
    )
    assertions.append(
        native_value(
            "np:hasAccelerationY", ego_id, ay, ValueType.FLOAT, frame,
            ["ego_state.center_or_rear_axle_acceleration_2d.y"],
        )
    )
    assertions.append(
        derived_value(
            "np:hasAcceleration", ego_id, math.hypot(ax, ay), ValueType.FLOAT,
            frame, "R-ACCELERATION-MAGNITUDE-001",
            {
                "acceleration_x_mps2": ax,
                "acceleration_y_mps2": ay,
                "quantity": "unsigned_magnitude",
            },
        )
    )


def append_effective_travel_direction(assertions, frame, record, state=None):
    direction = select_travel_direction(
        body_heading_rad=record["xyh"][2],
        map_heading_rad=record.get("map_heading_rad"),
        map_match_reliable=(
            not record.get("map_match_ambiguous", False)
            and record.get("primary_map_id") is not None
        ),
        velocity_heading_rad=record.get("velocity_heading_rad"),
        speed_mps=record.get("speed"),
        displacement_heading_rad=record.get("displacement_heading_rad"),
        displacement_m=record.get("displacement_from_previous_m"),
        minimum_speed_mps=ARGS.spatial_min_direction_speed_mps,
        minimum_displacement_m=ARGS.spatial_min_displacement_for_direction_m,
        velocity_displacement_tolerance_rad=ARGS.spatial_direction_agreement_threshold_rad,
        map_motion_tolerance_rad=ARGS.spatial_map_motion_tolerance_rad,
    )
    record["travel_direction"] = direction
    if direction.reliable and direction.heading_rad is not None:
        direction_evidence = {
            "source": direction.source,
            "map_heading_rad": direction.map_heading_rad,
            "velocity_heading_rad": direction.velocity_heading_rad,
            "displacement_heading_rad": direction.displacement_heading_rad,
            "velocity_displacement_difference_rad": direction.velocity_displacement_difference_rad,
            "map_motion_difference_rad": direction.map_motion_difference_rad,
        }
        assertions.append(
            derived_value(
                "np:hasEffectiveTravelHeading", record["entity_id"],
                float(direction.heading_rad), ValueType.FLOAT, frame,
                "R-MAP-MOTION-SPATIAL-001", direction_evidence,
            )
        )
        assertions.append(
            derived_value(
                "np:hasTravelDirectionSource", record["entity_id"],
                direction.source, ValueType.STRING, frame,
                "R-MAP-MOTION-SPATIAL-001", direction_evidence,
            )
        )
    else:
        record_semantic_omission(
            state, "effective_travel_direction",
            direction.omission_reason or "unreliable",
        )
    return direction
