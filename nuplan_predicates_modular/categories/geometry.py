"""Geometry measurements and footprint calculations."""
from ..base import *

def get_geometric_center_pose(entity: Any) -> tuple[float, float, float]:
    car_footprint = getattr(entity, "car_footprint", None)
    if car_footprint is not None:
        oriented_box = getattr(car_footprint, "oriented_box", None)
        if oriented_box is not None:
            center = getattr(oriented_box, "center", None)
            if center is not None:
                return float(center.x), float(center.y), float(center.heading)

    center = getattr(entity, "center", None)
    if center is not None:
        return float(center.x), float(center.y), float(center.heading)

    box = getattr(entity, "box", None)
    if box is not None:
        center = getattr(box, "center", None)
        if center is not None:
            return float(center.x), float(center.y), float(center.heading)

    raise AttributeError(f"No geometric center pose for {type(entity)}")


def get_rear_axle_xy(entity: Any) -> Optional[tuple[float, float]]:
    rear_axle = getattr(entity, "rear_axle", None)
    if rear_axle is None:
        return None
    return float(rear_axle.x), float(rear_axle.y)






def get_dimensions(entity: Any) -> dict[str, Optional[float]]:
    candidates = [
        getattr(getattr(entity, "car_footprint", None), "oriented_box", None),
        entity,
        getattr(entity, "box", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        length = getattr(candidate, "length", None)
        width = getattr(candidate, "width", None)
        height = getattr(candidate, "height", None)
        if length is not None and width is not None:
            return {
                "length_m": float(length),
                "width_m": float(width),
                "height_m": float(height) if height is not None else None,
            }
    return {"length_m": None, "width_m": None, "height_m": None}


def get_track_token(obj: Any) -> str:
    metadata = getattr(obj, "metadata", None)
    token = getattr(metadata, "track_token", None)
    if token is None:
        token = getattr(obj, "track_token", None)
    if token is None:
        token = stable_id(type(obj).__name__, repr(get_geometric_center_pose(obj)))
    return str(token)


def get_agent_type(obj: Any) -> str:
    return str(getattr(obj, "tracked_object_type", "UNKNOWN"))


def oriented_box_polygon(record: dict[str, Any]):
    if not SHAPELY_AVAILABLE:
        return None
    dims = record["dimensions"]
    length = dims.get("length_m")
    width = dims.get("width_m")
    if length is None or width is None:
        return None
    x, y, heading = record["xyh"]
    polygon = shapely_box(-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)
    polygon = rotate(polygon, math.degrees(heading), origin=(0, 0))
    return translate(polygon, xoff=x, yoff=y)




def relative_kinematics(subject: dict[str, Any], obj: dict[str, Any]) -> dict[str, float]:
    sx, sy, sh = subject["xyh"]
    ox, oy, oh = obj["xyh"]
    dx, dy = ox - sx, oy - sy
    c, s = math.cos(sh), math.sin(sh)
    longitudinal = c * dx + s * dy
    lateral = -s * dx + c * dy
    center_distance = math.hypot(dx, dy)
    result = {
        "center_distance_m": float(center_distance),
        "longitudinal_distance_m": float(longitudinal),
        "lateral_distance_m": float(lateral),
        "bearing_rad": float(math.atan2(lateral, longitudinal)),
        "heading_difference_rad": abs(wrap_signed(oh - sh)),
    }

    sv = subject.get("velocity")
    ov = obj.get("velocity")
    if sv is not None and ov is not None:
        dvx, dvy = ov[0] - sv[0], ov[1] - sv[1]
        result["relative_longitudinal_speed_mps"] = float(c * dvx + s * dvy)
        result["relative_lateral_speed_mps"] = float(-s * dvx + c * dvy)
        result["relative_speed_mps"] = float(math.hypot(dvx, dvy))
        result["closing_speed_mps"] = (
            0.0 if center_distance == 0.0 else float(-(dx * dvx + dy * dvy) / center_distance)
        )

    if sv is not None:
        result["subject_forward_speed_mps"] = float(c * sv[0] + s * sv[1])
        if center_distance == 0.0:
            result["velocity_toward_target_mps"] = 0.0
        else:
            result["velocity_toward_target_mps"] = float((sv[0] * dx + sv[1] * dy) / center_distance)

        # Save the raw mathematical ratio whenever division is defined. It is
        # a candidate measurement, not a following or safety label.
        forward_speed = result["subject_forward_speed_mps"]
        if forward_speed != 0.0:
            result["time_headway_candidate_s"] = float(longitudinal / forward_speed)

    return result

def footprint_metrics(subject: dict[str, Any], obj: dict[str, Any]) -> dict[str, Optional[Any]]:
    subject_poly = oriented_box_polygon(subject)
    object_poly = oriented_box_polygon(obj)
    if subject_poly is None or object_poly is None:
        return {
            "free_space_distance_m": None,
            "intersection_area_m2": None,
            "overlapping": None,
            "touching": None,
            "longitudinal_footprint_gap_m": None,
            "lateral_footprint_gap_m": None,
        }
    free_space = float(subject_poly.distance(object_poly))
    intersection_area = float(subject_poly.intersection(object_poly).area)
    overlapping = intersection_area > ARGS.overlap_area_epsilon_m2
    touching = bool(
        not overlapping
        and (subject_poly.touches(object_poly) or free_space <= ARGS.distance_epsilon_m)
    )

    sx, sy, sh = subject["xyh"]
    local_subject = rotate(translate(subject_poly, xoff=-sx, yoff=-sy), -math.degrees(sh), origin=(0, 0))
    local_object = rotate(translate(object_poly, xoff=-sx, yoff=-sy), -math.degrees(sh), origin=(0, 0))
    sminx, sminy, smaxx, smaxy = local_subject.bounds
    ominx, ominy, omaxx, omaxy = local_object.bounds
    object_center_x = (ominx + omaxx) / 2.0
    object_center_y = (ominy + omaxy) / 2.0
    longitudinal_gap = ominx - smaxx if object_center_x >= 0 else omaxx - sminx
    lateral_gap = ominy - smaxy if object_center_y >= 0 else omaxy - sminy
    return {
        "free_space_distance_m": free_space,
        "intersection_area_m2": intersection_area,
        "overlapping": overlapping,
        "touching": touching,
        "longitudinal_footprint_gap_m": float(longitudinal_gap),
        "lateral_footprint_gap_m": float(lateral_gap),
    }


def append_entity_geometry_measurements(assertions, frame, record, source_prefix):
    """Emit dimensions and footprint area for one entity."""
    entity_id = record["entity_id"]
    dimensions = record["dimensions"]
    for pid, key in (
        ("np:hasLength", "length_m"),
        ("np:hasWidth", "width_m"),
        ("np:hasHeight", "height_m"),
    ):
        value = dimensions.get(key)
        if value is not None:
            assertions.append(
                native_value(
                    pid, entity_id, value, ValueType.FLOAT, frame,
                    [f"{source_prefix}.{key}"],
                )
            )
    if dimensions.get("length_m") is not None and dimensions.get("width_m") is not None:
        assertions.append(
            derived_value(
                "np:hasArea", entity_id,
                dimensions["length_m"] * dimensions["width_m"],
                ValueType.FLOAT, frame, "R-FOOTPRINT-GEOMETRY-001", dimensions,
            )
        )


def append_pair_geometry_measurements(assertions, frame, pair_id, context):
    """Emit pairwise values owned by the geometry category."""
    kin = context["kinematics"]
    geometry = context["geometry"]
    spatial = context["spatial"]
    evidence = context["evidence"]
    values = (
        ("np:hasCenterDistanceTo", kin.get("center_distance_m"), "R-RELATIVE-KINEMATICS-001"),
        ("np:hasLongitudinalDistanceTo", kin.get("longitudinal_distance_m"), "R-RELATIVE-KINEMATICS-001"),
        ("np:hasLateralDistanceTo", kin.get("lateral_distance_m"), "R-RELATIVE-KINEMATICS-001"),
        ("np:hasBearingTo", kin.get("bearing_rad"), "R-RELATIVE-KINEMATICS-001"),
        ("np:hasHeadingDifferenceTo", kin.get("heading_difference_rad"), "R-RELATIVE-KINEMATICS-001"),
        ("np:hasFreeSpaceDistanceTo", geometry.get("free_space_distance_m"), "R-FOOTPRINT-GEOMETRY-001"),
        ("np:hasIntersectionArea", geometry.get("intersection_area_m2"), "R-FOOTPRINT-GEOMETRY-001"),
        ("np:hasLongitudinalFootprintGap", geometry.get("longitudinal_footprint_gap_m"), "R-FOOTPRINT-GEOMETRY-001"),
        ("np:hasLateralFootprintGap", geometry.get("lateral_footprint_gap_m"), "R-FOOTPRINT-GEOMETRY-001"),
        ("np:hasTravelLongitudinalDistanceTo", spatial.travel_longitudinal_m, "R-MAP-MOTION-SPATIAL-001"),
        ("np:hasTravelLateralDistanceTo", spatial.travel_lateral_m, "R-MAP-MOTION-SPATIAL-001"),
    )
    for pid, value, rule_id in values:
        if value is not None and math.isfinite(float(value)):
            assertions.append(
                derived_value(pid, pair_id, float(value), ValueType.FLOAT, frame, rule_id, evidence)
            )
