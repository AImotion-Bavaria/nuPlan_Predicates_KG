"""Regression tests for v9.5.31 observed-history gating of temporal maneuvers."""

from test_interaction_follow import MapObject, load_interaction_module, _overtake_history
from test_interaction_lane_change import _history
from test_interaction_merge import _merge_history, _merge_predicates


def test_lane_change_never_activates_at_first_observed_frame():
    module = load_interaction_module()
    source = MapObject("L0", 0.0, left_adjacent=("L1",), center_y=0.0)
    target = MapObject("L1", 0.0, right_adjacent=("L0",), center_y=4.0)
    assertions = module.derive_lane_change_predicates(_history(source, target))
    assert assertions
    assert min(row.evidence["lane_change_current_frame_index"] for row in assertions) > 0
    assert all(row.evidence["maneuver_history_gate_passed"] for row in assertions)
    assert all(row.evidence["maneuver_history_frame_count"] >= 1 for row in assertions)
    assert all(row.evidence["maneuver_history_duration_s"] >= 0.5 for row in assertions)


def test_overtake_uses_initial_behind_frames_as_history_not_active_relation():
    module = load_interaction_module()
    assertions = module.derive_overtaking_predicates(_overtake_history())
    assert assertions
    first = assertions[0]
    # The red relation begins at verified departure, not at the first
    # same-lane/behind observation used to prove the maneuver precondition.
    assert first.evidence["overtake_current_frame_index"] > 0
    assert (
        first.evidence["overtake_current_frame_index"]
        == first.evidence["overtake_departure_frame_index"]
    )
    assert first.evidence["maneuver_history_gate_passed"] is True
    assert first.evidence["maneuver_history_frame_count"] >= 1
    assert first.evidence["maneuver_history_duration_s"] >= 0.5


def test_merge_inherits_nonzero_lane_change_onset_and_history():
    module = load_interaction_module()
    lane_changes, merges = _merge_predicates(
        module, _merge_history(object_offset_m=-8.0)
    )
    assert lane_changes
    assert merges
    assert min(row.evidence["merge_current_frame_index"] for row in merges) > 0
    assert all(row.evidence["maneuver_history_gate_passed"] for row in merges)
    assert all(row.evidence["maneuver_history_frame_count"] >= 1 for row in merges)
    assert all(row.evidence["maneuver_history_duration_s"] >= 0.5 for row in merges)
