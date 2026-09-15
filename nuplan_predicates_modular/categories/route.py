"""Native expert-route structural predicates."""
from ..base import *


def append_expert_route(assertions, frame):
    route_id = f"route:{frame.scenario_token}"
    for roadblock_id in frame.route_roadblock_ids:
        assertions.append(
            native_relation(
                "np:onExpertRoute", f"roadblock:{roadblock_id}", route_id,
                frame, ["route_roadblock_ids"],
            )
        )
    return route_id
