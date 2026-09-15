from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import nbformat
import pandas as pd
from shapely.geometry import LineString, Point


ROOT = Path(__file__).resolve().parents[1]
VIDEO_SCRIPT = ROOT / "semantic_map_video_cell_v9_5_31.py"
VIDEO_NOTEBOOK = ROOT / "nuplan_visual_inspection_video_v9_5_31.ipynb"


def _function_nodes(source: str, names: set[str]):
    tree = ast.parse(source)
    return [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]


def test_full_scene_video_cell_is_synchronized_and_counts_lane_overlays():
    script = VIDEO_SCRIPT.read_text(encoding="utf-8")
    ast.parse(script)
    notebook = nbformat.read(VIDEO_NOTEBOOK, as_version=4)

    assert notebook.cells[26].source == script
    assert "def _video_draw_lane_change_overlays" in script
    assert 'label_counts["changesLane"]' in script
    assert 'summary["active_relation_count"] > 0' in script
    assert '"ego_to_structure": True' in notebook.cells[23].source
    assert '"agent_to_structure": True' in notebook.cells[23].source
    assert "SHOW_AGENT_LANE_INFO = True" in notebook.cells[23].source
    assert "SHOW_MAP_STRUCTURE_IDS = True" in notebook.cells[23].source


def test_lane_change_overlay_respects_vehicle_to_structure_switch():
    source = VIDEO_SCRIPT.read_text(encoding="utf-8")
    nodes = _function_nodes(source, {"_video_active_lane_change_overlays"})
    namespace = {
        "pd": pd,
        "classify_semantic_relation": lambda subject, obj: "agent_to_structure",
        "entity_id_to_track_token": lambda value: str(value).split(":")[-1],
        "_evidence_mapping": lambda value: json.loads(value),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(VIDEO_SCRIPT), "exec"),
        namespace,
    )
    frame = pd.DataFrame([
        {
            "predicate_id": "np:changesLane",
            "subject_id": "agent:vehicle",
            "object_id": "lane:L1",
            "evidence_json": json.dumps({"event_id": "event-1", "target_lane_id": "L1"}),
        }
    ])

    enabled = namespace["_video_active_lane_change_overlays"](
        frame,
        {"agent_to_structure": True},
    )
    disabled = namespace["_video_active_lane_change_overlays"](
        frame,
        {"agent_to_structure": False},
    )
    disabled_agent_type = namespace["_video_active_lane_change_overlays"](
        frame,
        {"agent_to_structure": True},
        entity_table=pd.DataFrame([{"track_token": "another-agent"}]),
    )

    assert len(enabled) == 1
    assert enabled[0]["object_id"] == "lane:L1"
    assert disabled == []
    assert disabled_agent_type == []


def test_video_runner_discovers_lane_changes_merges_and_overtakes():
    script = (ROOT / "run_videos.sh").read_text(encoding="utf-8")
    assert 'np:changesLane' in script
    assert 'np:mergesInFrontOf' in script
    assert 'np:mergesBehind' in script
    assert 'np:overtakes' in script
    assert "detected_interaction_tokens.txt" in script


def test_video_restores_entity_valued_merge_edges():
    source = VIDEO_SCRIPT.read_text(encoding="utf-8")
    nodes = _function_nodes(source, {"_video_restore_merge_edges"})
    namespace = {
        "pd": pd,
        "_excel_predicate_list": lambda values: values if isinstance(values, list) else [values],
        "entity_id_to_track_token": lambda value: str(value).replace("entity:", ""),
        "classify_semantic_relation": lambda subject, obj: "agent_to_agent",
        "short_token": lambda value: str(value),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(VIDEO_SCRIPT), "exec"),
        namespace,
    )
    frame = pd.DataFrame([
        {
            "predicate_id": "np:mergesInFrontOf",
            "subject_id": "entity:subject",
            "object_id": "entity:object",
        },
        {
            "predicate_id": "np:mergesBehind",
            "subject_id": "entity:subject",
            "object_id": "entity:front",
        },
    ])
    restored = namespace["_video_restore_merge_edges"](pd.DataFrame(), frame)
    assert len(restored) == 2
    assert set(restored["relation_labels"].astype(str)) == {
        "mergesInFrontOf", "mergesBehind"
    }
