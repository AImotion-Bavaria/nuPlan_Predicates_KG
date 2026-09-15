from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "nuplan_visual_inspection_video_v9_5_31.ipynb"


def _load_functions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "def collapse_edges_for_plot" in cell.source
    )
    module = ast.parse(source)
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"collapse_edges_for_plot", "prepare_plot_edges"}
    ]
    code = compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK), "exec")
    namespace = {
        "pd": pd,
        "_excel_predicate_list": lambda value: (
            value if isinstance(value, list) else [value]
        ),
        "entity_id_to_track_token": lambda value: str(value).split(":")[-1],
    }
    exec(code, namespace)
    return namespace["collapse_edges_for_plot"], namespace["prepare_plot_edges"]


def test_collapsed_edges_preserve_endpoint_columns():
    collapse, prepare = _load_functions()
    edges = pd.DataFrame([
        {
            "subject_id": "np:agent_subject",
            "object_id": "np:agent_object",
            "subject_token": "subject",
            "object_token": "object",
            "relation_type": "agent_to_agent",
            "category": "interaction",
            "predicates": ["np:overtakes"],
        }
    ])
    entities = pd.DataFrame([
        {"track_token": "subject", "display_id": "A1"},
        {"track_token": "object", "display_id": "A2"},
    ])

    collapsed = collapse(edges)
    assert {"subject_token", "object_token", "subject_id", "object_id"}.issubset(
        collapsed.columns
    )
    assert collapsed.iloc[0]["subject_token"] == "subject"
    assert collapsed.iloc[0]["object_token"] == "object"

    plotted = prepare(edges, entities)
    assert plotted.iloc[0]["subject_display_id"] == "A1"
    assert plotted.iloc[0]["object_display_id"] == "A2"
    assert plotted.iloc[0]["relation_labels"] == "overtakes"


def test_empty_frame_returns_stable_schema():
    collapse, prepare = _load_functions()
    entities = pd.DataFrame(columns=["track_token", "display_id"])
    plotted = prepare(pd.DataFrame(), entities)
    assert plotted.empty
    assert {"subject_token", "object_token"}.issubset(plotted.columns)
