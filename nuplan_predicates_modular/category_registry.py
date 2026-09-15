"""Human-readable registry for category-by-category review."""

CATEGORY_MODULES = {
    "structure": "frame_pipeline.py",
    "scenario": "frame_pipeline.py",
    "agent": "frame_pipeline.py",
    "position_values": "categories/position.py",
    "geometry": "categories/geometry.py",
    "motion": "categories/motion.py",
    "motion_state": "categories/motion.py and categories/temporal.py",
    "temporal": "categories/temporal.py",
    "map": "categories/map.py",
    "route": "categories/map.py and frame_pipeline.py",
    "traffic_light": "categories/traffic_control.py (controls + hasSignalState + isRelevantSignal)",
    "pairwise": "categories/pairwise.py (coordinator only)",
    "relevance": "categories/relevance.py",
    "spatial": "categories/spatial.py plus predicate_logic.py",
    "heading": "categories/heading.py plus predicate_logic.py",
    "interaction": "categories/interaction.py",
    "risk": "categories/risk.py (v9.5.44 current-state conflict risk)",
    "future_observation": "categories/future_observation.py",
    "maneuver": "categories/maneuver.py",
    "visibility": "categories/visibility.py",
    "sensor_visibility": "catalog only; no nuPlan sensor extraction implemented",
    "intent": "categories/intent.py",
}

RECOMMENDED_REVIEW_ORDER = [
    "structure", "scenario", "agent", "position_values", "geometry", "motion",
    "temporal", "map", "route", "pairwise", "relevance", "spatial", "heading",
    "interaction", "traffic_light", "visibility", "future_observation", "risk",
    "maneuver", "intent", "sensor_visibility",
]
