import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_retains_pairwise_identity_dependency():
    text = (ROOT / "nuplan_predicates_modular" / "base.py").read_text()
    assert '"temporal": {"structure", "agent", "position_values", "motion", "pairwise"}' in text


def test_current_video_notebook_is_python39_compatible():
    notebook_path = ROOT / "nuplan_visual_inspection_video_v9_5_31.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        ast.parse(
            "".join(cell.get("source", [])),
            filename=f"notebook_cell_{index}",
            feature_version=(3, 9),
        )
    assert "def parquet_scenario_index" in code
    assert "def _evidence_mapping" in code
    assert "SEMANTIC_PREDICATE_ARROW_STYLES" in code
    assert '"np:follows": {"color": "#1565c0"' in code
    assert '"np:overtakes": {"color": "#e53935"' in code
    assert '"np:crossesInFrontOf": {"color": "#00897b"' in code
    assert '"label": "Crosses in front"' in code
    assert "pd.DataFrame | None" not in code


def test_package_has_one_canonical_video_notebook():
    notebooks = sorted(ROOT.glob("nuplan_visual_inspection_video_v*.ipynb"))
    assert [path.name for path in notebooks] == [
        "nuplan_visual_inspection_video_v9_5_31.ipynb"
    ]
