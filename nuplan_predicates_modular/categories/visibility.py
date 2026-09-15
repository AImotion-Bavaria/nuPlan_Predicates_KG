"""Geometric field-of-view, line-of-sight, and occlusion predicates."""
from ..base import *
from .geometry import oriented_box_polygon

def _fov_polygon(ex: float, ey: float, heading: float, half_fov: float, radius: float):
    if not SHAPELY_AVAILABLE:
        return None
    Polygon = __import__("shapely.geometry", fromlist=["Polygon"]).Polygon
    steps = max(12, int(math.degrees(half_fov) / 3.0))
    angles = [heading - half_fov + (2.0 * half_fov * i / steps) for i in range(steps + 1)]
    points = [(ex, ey)] + [(ex + radius * math.cos(a), ey + radius * math.sin(a)) for a in angles]
    return Polygon(points)


def derive_visibility_predicates(snapshot, state=None):
    assertions = []
    frame = snapshot["frame"]
    ego = snapshot["entities"].get("ego")
    if ego is None or not SHAPELY_AVAILABLE or not ARGS.derive_semantic_predicates:
        return assertions
    ex, ey, eh = ego["xyh"]
    half_fov = math.radians(ARGS.ego_fov_deg) / 2.0
    fov_polygon = _fov_polygon(ex, ey, eh, half_fov, ARGS.ego_fov_range_m)
    all_agents = [record for token, record in snapshot["entities"].items() if token != "ego"]
    relevant_tokens = set(snapshot.get("relevant_agent_tokens", []))
    agents = [record for record in all_agents if str(record.get("track_token")) in relevant_tokens]
    polygons = {str(r["track_token"]): oriented_box_polygon(r) for r in all_agents}
    LineString = __import__("shapely.geometry", fromlist=["LineString"]).LineString

    for target in agents:
        target_poly = polygons.get(str(target["track_token"]))
        if target_poly is None or fov_polygon is None:
            record_semantic_omission(state, "np:geometricallyVisibleToEgo", "missing_geometry")
            continue
        within = bool(target_poly.intersects(fov_polygon))
        if not within:
            record_semantic_omission(state, "np:withinEgoFieldOfView", "outside_geometric_fov")
            continue
        tx, ty, _ = target["xyh"]
        evidence = {
            "fov_half_angle_rad": half_fov,
            "maximum_range_m": ARGS.ego_fov_range_m,
            "visibility_kind": "geometric_not_sensor_visibility",
        }
        append_positive_boolean_semantic(
            assertions, "np:withinEgoFieldOfView", target["entity_id"], True,
            frame, "R-GEOMETRIC-VISIBILITY-001", evidence, state=state,
        )

        representative_points = [(tx, ty)]
        try:
            coords = list(target_poly.exterior.coords)
            representative_points.extend(coords[::max(1, len(coords) // 4)][:4])
        except Exception:
            pass
        visible_rays = 0
        blocker_by_token = {}
        for point_x, point_y in representative_points:
            ray = LineString([(ex, ey), (float(point_x), float(point_y))])
            ray_length = float(ray.length)
            blocked = False
            for blocker in all_agents:
                if blocker["track_token"] == target["track_token"]:
                    continue
                blocker_poly = polygons.get(str(blocker["track_token"]))
                if blocker_poly is None:
                    continue
                bx, by, _ = blocker["xyh"]
                if math.hypot(bx - ex, by - ey) >= ray_length + ARGS.distance_epsilon_m:
                    continue
                if ray.intersects(blocker_poly):
                    blocked = True
                    blocker_by_token[str(blocker["track_token"])] = blocker
                    break
            if not blocked:
                visible_rays += 1
        visible_fraction = visible_rays / max(1, len(representative_points))
        visible = visible_rays > 0
        if visible:
            assertions.append(derived_relation(
                "np:hasLineOfSightTo", ego["entity_id"], target["entity_id"],
                frame, "R-GEOMETRIC-VISIBILITY-001",
                {**evidence, "visible_ray_fraction": visible_fraction},
            ))
            append_positive_boolean_semantic(
                assertions, "np:geometricallyVisibleToEgo", target["entity_id"], True,
                frame, "R-GEOMETRIC-VISIBILITY-001",
                {**evidence, "visible_ray_fraction": visible_fraction}, state=state,
            )
        for blocker in blocker_by_token.values():
            assertions.append(derived_relation(
                "np:occludedByAgent", target["entity_id"], blocker["entity_id"],
                frame, "R-GEOMETRIC-VISIBILITY-001",
                {**evidence, "blocker_id": blocker["entity_id"], "visible_ray_fraction": visible_fraction},
            ))
    return assertions
