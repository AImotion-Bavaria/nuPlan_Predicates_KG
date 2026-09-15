"""Regression checks for exact-predicate visualization filtering."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_batch_config_defaults_to_all_predicates():
    cfg = json.loads((ROOT / "batch_config.json").read_text(encoding="utf-8"))
    assert cfg["semantic_predicates"] == "all"


def test_category_runner_supports_exact_predicate_options():
    code = (ROOT / "run_category_videos.sh").read_text(encoding="utf-8")
    assert "--predicate|--predicates" in code
    assert 'cfg["semantic_predicates"] = requested_predicates' in code


def test_worker_passes_predicate_filter_to_renderer():
    code = (ROOT / "run_scenario_export.py").read_text(encoding="utf-8")
    assert '"SEMANTIC_PREDICATES": config.get("semantic_predicates", "all")' in code
    assert 'semantic_predicates=config.get("semantic_predicates", "all")' in code


def test_video_renderer_applies_exact_predicate_filter():
    code = (ROOT / "semantic_map_video_cell_v9_5_31.py").read_text(encoding="utf-8")
    assert "filter_semantic_edges_by_predicates(" in code
    assert 'semantic_predicate_is_selected(\n                    "np:changesLane"' in code
    assert "Predicate filter:" in code


def test_notebook_exposes_user_predicate_selection():
    nb = json.loads((ROOT / "nuplan_visual_inspection_video_v9_5_31.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert 'SEMANTIC_PREDICATES = "all"' in code
    assert '# SEMANTIC_PREDICATES = ["follows"]' in code
    assert "def filter_semantic_edges_by_predicates" in code
    assert "def validate_semantic_predicate_selection" in code


def test_all_tracked_object_types_are_enabled_by_default():
    cfg = json.loads((ROOT / "batch_config.json").read_text(encoding="utf-8"))
    assert cfg["agent_types"]
    assert all(bool(value) for value in cfg["agent_types"].values())


def test_video_runner_saves_by_family_then_predicate():
    code = (ROOT / "run_category_videos.sh").read_text(encoding="utf-8")
    assert "export_root = base_export_root / fragment / predicate_fragment" in code


def test_exact_predicate_runner_filters_scenarios_from_parquet():
    code = (ROOT / "run_category_videos.sh").read_text(encoding="utf-8")
    assert 'if [[ "$ENABLED_PREDICATES" != "all" ]]' in code
    assert 'columns=[token_col, "predicate_id"]' in code


def test_crosses_in_front_uses_dedicated_teal_arrow_only():
    nb = json.loads(
        (ROOT / "nuplan_visual_inspection_video_v9_5_31.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
    )
    assert '"np:crossesInFrontOf": {"color": "#00897b"' in code
    assert '"arrowstyle": "-|>"' in code
    assert "SHOW_EDGE_IDS" in code

def test_parallel_runner_supports_exact_predicate_options():
    code = (ROOT / "run_parallel_exports.sh").read_text(encoding="utf-8")
    assert "--predicate)" in code
    assert "--predicates)" in code
    assert 'cfg["semantic_predicates"] = requested_predicates' in code
    assert 'echo "Semantic predicates: $ENABLED_PREDICATES"' in code

