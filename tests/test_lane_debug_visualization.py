from __future__ import annotations

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "nuplan_visual_inspection_video_v9_5_31.ipynb"
VIDEO_SCRIPT = ROOT / "semantic_map_video_cell_v9_5_31.py"
CONFIG = ROOT / "batch_config.json"


def _function_source(cell_source: str, name: str) -> str:
    tree = ast.parse(cell_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(cell_source, node) or ""
    raise AssertionError(f"Function {name} not found")


def test_lane_centerlines_are_not_drawn():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = nb.cells[9].source
    function = _function_source(source, "plot_local_map")
    assert "getattr(map_object, \"baseline_path\"" not in function
    assert "_plot_linestring_geometry(" not in function
    assert "boundary_only=lane_like" in function


def test_debug_switches_are_exposed_and_forwarded():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    config_cell = nb.cells[23].source
    assert "SHOW_AGENT_LANE_INFO = True" in config_cell
    assert "SHOW_MAP_STRUCTURE_IDS = True" in config_cell

    video = VIDEO_SCRIPT.read_text(encoding="utf-8")
    assert '"SHOW_AGENT_LANE_INFO"' in video
    assert '"SHOW_MAP_STRUCTURE_IDS"' in video
    assert "show_agent_lane_info=VIDEO_SHOW_AGENT_LANE_INFO" in video
    assert "show_map_structure_ids=VIDEO_SHOW_MAP_STRUCTURE_IDS" in video


def test_agent_debug_labels_show_primary_candidates_and_parent():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = nb.cells[10].source
    assert '"np:hasPrimaryLane"' in source
    assert '"np:hasPrimaryLaneConnector"' in source
    assert '"np:hasAmbiguousMapMatch"' in source
    assert '"candidate_summaries"' in source
    assert 'lines = [f"P={primary}"]' in source
    assert 'lines = ["P=AMBIG"]' in source
    assert 'lines.append("C=" + ",".join(candidates[:3]))' in source
    assert "lines.append(parent)" in source


def test_map_debug_labels_cover_native_lane_group_structures():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = nb.cells[9].source
    assert '"LANE": "L"' in source
    assert '"LANE_CONNECTOR": "LC"' in source
    assert '"ROADBLOCK": "RB"' in source
    assert '"ROADBLOCK_CONNECTOR": "RBC"' in source


def test_debug_visualization_does_not_import_extractor_cli_modules():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    imports = nb.cells[2].source
    assert "nuplan_predicates_modular.categories.map" not in imports
    assert "fall back to direct nuPlan map-API containment" in imports


def test_video_passes_frame_assertions_to_agent_lane_debug_labels():
    video = VIDEO_SCRIPT.read_text(encoding="utf-8")
    assert "frame_assertions=frame_assertions" in video


def test_agent_lane_debug_has_safe_map_api_fallback():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = nb.cells[10].source
    assert "def _debug_visual_map_assignment(map_api, row):" in source
    assert "map_api.get_all_map_objects(point, layer)" in source
    assert "map_api=map_api" in source
    assert '"source": "visual_map"' in source


def test_interaction_export_includes_map_debug_assertions():
    script = (ROOT / "run_interaction.sh").read_text(encoding="utf-8")
    assert "--categories interaction,map" in script
