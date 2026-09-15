from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_interaction_runner_does_not_require_overtake_yaml():
    script = (ROOT / "run_interaction.sh").read_text(encoding="utf-8")
    assert 'SCENARIO_FILTER_YAML="${SCENARIO_FILTER_YAML:-}"' in script
    assert "overtake_candidate_scenarios.yaml" not in script
    assert "complete nuPlan catalogue" in script
    assert "run_predicates.sh" in script


def test_video_config_has_no_required_scenario_filter():
    config = json.loads((ROOT / "batch_config.json").read_text(encoding="utf-8"))
    assert config["scenario_filter_yaml"] is None
    assert config["assertion_output_dir"].endswith("paper_release")


def test_external_overtake_search_notebook_removed():
    assert not list(ROOT.glob("*overtake*reference*.ipynb"))
    assert not list(ROOT.glob("*overtake*candidate*.yaml"))
