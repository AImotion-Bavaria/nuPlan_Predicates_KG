#!/usr/bin/env python3
"""Export one nuPlan scenario to an organized map-only MP4 and CSV-table folder."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Use the runtime bundled with this package; no external PROJECT_ROOT/src is required.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_BUNDLED_SRC = _PACKAGE_ROOT / "src"
if str(_BUNDLED_SRC) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_SRC))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _exec_notebook_definitions(notebook_path: Path, config: dict) -> dict:
    import nbformat

    nb = nbformat.read(notebook_path, as_version=4)
    ns: dict = {"__name__": "__nuplan_batch_worker__"}

    # Execute the path cell first, then override all machine-specific paths.
    exec(compile(nb.cells[1].source, f"{notebook_path}:cell1", "exec"), ns)
    ns.update({
        # Explicitly expose the package-local runtime so the notebook cannot
        # fall back to an older parent-project adapter.
        "PACKAGE_ROOT": _PACKAGE_ROOT,
        "BUNDLED_SRC": _BUNDLED_SRC,
        "PROJECT_ROOT": Path(config["project_root"]),
        "DATASET_ROOT": Path(config["dataset_root"]),
        "MAP_ROOT": Path(config["map_root"]),
        "MAP_VERSION": config["map_version"],
        "SCENARIO_FILTER_YAML": (
            Path(config["scenario_filter_yaml"])
            if config.get("scenario_filter_yaml")
            else None
        ),
        "OUTPUT_DIR": Path(config["assertion_output_dir"]),
    })

    # Definition/data-loading cells only. Skip master execution, plotting, and video calls.
    for idx in range(2, 20):
        cell = nb.cells[idx]
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        exec(compile(cell.source, f"{notebook_path}:cell{idx}", "exec"), ns)

    return ns


def _inject_plot_configuration(ns: dict, config: dict) -> None:
    ns.update({
        "SEMANTIC_CATEGORIES": dict(config["semantic_categories"]),
        "SEMANTIC_RELATIONS": dict(config["semantic_relations"]),
        "SEMANTIC_PREDICATES": config.get("semantic_predicates", "all"),
        "AGENT_TYPES": dict(config["agent_types"]),
        "INTERESTED_DISTANCE_THRESHOLD_M": float(config["interested_distance_threshold_m"]),
        "INCLUDE_WITHIN_DISTANCE": bool(config["include_within_distance"]),
        "INCLUDE_FORWARD_CORRIDOR": bool(config["include_forward_corridor"]),
        "FORWARD_CORRIDOR_LENGTH_M": float(config["forward_corridor_length_m"]),
        "FORWARD_CORRIDOR_HALF_WIDTH_M": float(config["forward_corridor_half_width_m"]),
        "INCLUDE_SAME_LANE": bool(config["include_same_lane"]),
        "INCLUDE_PREDICTED_PATH_INTERSECTION": bool(config["include_predicted_path_intersection"]),
        "PREDICTION_HORIZON_S": float(config["prediction_horizon_s"]),
        "PREDICTION_STEP_S": float(config["prediction_step_s"]),
        "PATH_INTERSECTION_CLEARANCE_M": float(config["path_intersection_clearance_m"]),
        "INCLUDE_EXISTING_SPATIAL_AGENTS": bool(config["include_existing_spatial_agents"]),
        "FILTER_EDGES_TO_SELECTED_AGENTS": bool(config["filter_edges_to_selected_agents"]),
        "MANUAL_HIGHLIGHT_TOKENS": list(config.get("manual_highlight_tokens", [])),
        "MAP_RADIUS_M": float(config["map_radius_m"]),
        "INCLUDE_ALL_AGENTS_IN_GRAPH": False,
        "DRAW_SEMANTIC_ARROWS_ON_MAP": bool(config["draw_semantic_arrows_on_map"]),
        "SHOW_AGENT_LABELS": bool(config["show_agent_labels"]),
        "SHOW_AGENT_LANE_INFO": bool(config.get("show_agent_lane_info", False)),
        "SHOW_MAP_STRUCTURE_IDS": bool(config.get("show_map_structure_ids", False)),
        "SHOW_EDGE_IDS": bool(config["show_edge_ids"]),
        "SHOW_HEADING_ARROWS": True,
        "SHOW_EDGE_LEGEND": False,
        "MAX_EDGE_LEGEND_ROWS": None,
        "SHOW_FILTERED_TABLES": False,
        "MAX_FILTERED_TABLE_ROWS": 200,
    })


def _exec_video_definitions(video_cell_path: Path, ns: dict, config: dict) -> None:
    source = video_cell_path.read_text(encoding="utf-8")
    marker = "VIDEO_RESULT = render_full_scene_configurable_semantic_video("
    if marker not in source:
        raise RuntimeError(f"Cannot find video invocation marker in {video_cell_path}")
    definitions = source.split(marker, 1)[0]

    # Variables referenced by the video-only configuration block.
    ns.update({
        "VIDEO_EXPORT_ROOT": Path(config["export_root"]),
        "VIDEO_SCENARIO_SELECTOR": "index",
        "VIDEO_SCENARIO_VALUE": 0,
        "VIDEO_SCENARIO_SEED": int(config.get("scenario_seed", 42)),
        "VIDEO_FILTER_SEED": int(config.get("filter_seed", 42)),
        "VIDEO_START_FRAME": int(config.get("video_start_frame", 0)),
        "VIDEO_END_FRAME": config.get("video_end_frame"),
        "VIDEO_FRAME_STEP": int(config.get("video_frame_step", 1)),
        "VIDEO_PLAYBACK_SPEED": float(config.get("video_playback_speed", 1.0)),
        "VIDEO_FPS": config.get("video_fps"),
        "VIDEO_FIGSIZE": tuple(config.get("video_figsize", [12, 12])),
        "VIDEO_DPI": int(config.get("video_dpi", 100)),
        "VIDEO_SHOW_STATUS_BOX": bool(config.get("show_status_box", True)),
    })
    exec(compile(definitions, str(video_cell_path), "exec"), ns)


def _safe_token(token: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in token)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--video-cell", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario-index", type=int)
    group.add_argument("--scenario-token", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.time()
    print(f"[1/6] Starting scenario export", flush=True)
    print(f"      config={args.config.resolve()}", flush=True)
    selector_preview = "index" if args.scenario_index is not None else "token"
    value_preview = args.scenario_index if args.scenario_index is not None else args.scenario_token
    print(f"      requested {selector_preview}={value_preview}", flush=True)
    config = _load_json(args.config.resolve())
    print("[2/6] Initializing notebook definitions", flush=True)

    # Notebook initialization prints the same catalog and helper messages for
    # every worker. Capture that common output so each task log stays focused
    # on the selected scenario. It is saved later inside the scenario folder.
    initialization_output = io.StringIO()
    with contextlib.redirect_stdout(initialization_output), contextlib.redirect_stderr(initialization_output):
        ns = _exec_notebook_definitions(args.notebook.resolve(), config)
        _inject_plot_configuration(ns, config)
        _exec_video_definitions(args.video_cell.resolve(), ns, config)

    print("[3/6] Notebook initialization complete", flush=True)

    selector = "index" if args.scenario_index is not None else "token"
    value = args.scenario_index if args.scenario_index is not None else args.scenario_token
    scenario, catalog_row = ns["select_scenario"](
        selector=selector,
        value=value,
        random_seed=int(config.get("scenario_seed", 42)),
        filter_seed=int(config.get("filter_seed", 42)),
    )
    token = str(scenario.token)
    print(f"[4/6] Resolved scenario token={token}", flush=True)

    # select_scenario() returns a catalog-row dictionary in this notebook.
    # Never stringify that dictionary into a directory name. For index-based
    # runs, the requested index is the stable folder index. For token-based
    # runs, use the catalog row's numeric index when available.
    if args.scenario_index is not None:
        resolved_index = int(args.scenario_index)
    elif isinstance(catalog_row, dict) and catalog_row.get("index") is not None:
        resolved_index = int(catalog_row["index"])
    else:
        resolved_index = None

    index_fragment = (
        f"{resolved_index:05d}"
        if resolved_index is not None
        else "token"
    )
    scenario_dir = Path(config["export_root"]) / (
        f"scenario_{index_fragment}_{_safe_token(token)}"
    )
    video_dir = scenario_dir / "video"
    tables_dir = scenario_dir / "tables"
    for directory in (video_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Scenario selector: {selector}")
    print(f"Scenario request: {value}")
    print(f"Resolved scenario index: {resolved_index}")
    print(f"Scenario token: {token}")
    print(f"Scenario directory: {scenario_dir}")

    done_marker = scenario_dir / "DONE.json"
    failed_marker = scenario_dir / "FAILED.json"

    if done_marker.exists() and not args.overwrite:
        print(f"SKIP complete scenario: {scenario_dir}")
        return 0

    if args.overwrite:
        done_marker.unlink(missing_ok=True)
        failed_marker.unlink(missing_ok=True)

    result_record = {
        "status": "running",
        "scenario_selector": selector,
        "scenario_value": value,
        "scenario_catalog_row": catalog_row,
        "resolved_scenario_index": resolved_index,
        "scenario_token": token,
        "scenario_directory": str(scenario_dir),
    }

    try:
        print("[5/6] Rendering video", flush=True)
        video_result = ns["render_full_scene_configurable_semantic_video"](
            scenario_selector="token",
            scenario_value=token,
            scenario_seed=int(config.get("scenario_seed", 42)),
            filter_seed=int(config.get("filter_seed", 42)),
            output_dir=Path(config["assertion_output_dir"]),
            export_root=video_dir,
            start_frame=int(config.get("video_start_frame", 0)),
            end_frame=config.get("video_end_frame"),
            frame_step=int(config.get("video_frame_step", 1)),
            fps=config.get("video_fps"),
            playback_speed=float(config.get("video_playback_speed", 1.0)),
            figsize=tuple(config.get("video_figsize", [12, 12])),
            dpi=int(config.get("video_dpi", 100)),
            semantic_categories=config["semantic_categories"],
            semantic_relations=config["semantic_relations"],
            semantic_predicates=config.get("semantic_predicates", "all"),
            agent_types=config["agent_types"],
            interested_distance_threshold_m=float(config["interested_distance_threshold_m"]),
            include_within_distance=bool(config["include_within_distance"]),
            include_forward_corridor=bool(config["include_forward_corridor"]),
            forward_corridor_length_m=float(config["forward_corridor_length_m"]),
            forward_corridor_half_width_m=float(config["forward_corridor_half_width_m"]),
            include_same_lane=bool(config["include_same_lane"]),
            include_predicted_path_intersection=bool(config["include_predicted_path_intersection"]),
            prediction_horizon_s=float(config["prediction_horizon_s"]),
            prediction_step_s=float(config["prediction_step_s"]),
            path_intersection_clearance_m=float(config["path_intersection_clearance_m"]),
            include_existing_spatial_agents=bool(config["include_existing_spatial_agents"]),
            filter_edges_to_selected_agents=bool(config["filter_edges_to_selected_agents"]),
            manual_highlight_tokens=list(config.get("manual_highlight_tokens", [])),
            map_radius_m=float(config["map_radius_m"]),
            draw_semantic_arrows_on_map=bool(config["draw_semantic_arrows_on_map"]),
            show_agent_labels=bool(config["show_agent_labels"]),
            show_agent_lane_info=bool(config.get("show_agent_lane_info", False)),
            show_map_structure_ids=bool(config.get("show_map_structure_ids", False)),
            show_edge_ids=bool(config["show_edge_ids"]),
            show_status_box=bool(config.get("show_status_box", True)),
        )

        summary = video_result.get("summary")
        if summary is not None:
            summary.to_csv(
                tables_dir / "video_frame_summary.csv",
                index=False,
            )

        edge_events = video_result.get("edge_events")
        if edge_events is not None:
            edge_events.to_csv(
                tables_dir / "video_edge_events.csv",
                index=False,
            )

        result_record.update({
            "status": "complete",
            "elapsed_seconds": round(time.time() - started, 3),
            "video_path": str(video_result["video_path"]),
            "video_frame_summary_path": str(
                tables_dir / "video_frame_summary.csv"
            ),
            "video_edge_events_path": str(
                tables_dir / "video_edge_events.csv"
            ),
            "rendered_frames": len(
                video_result.get("frame_indices", [])
            ),
            "active_frame_count": int(
                video_result.get("active_frame_count", 0)
            ),
            "fps": float(video_result.get("fps", 0.0)),
        })
        done_marker.write_text(
            json.dumps(result_record, indent=2, default=str),
            encoding="utf-8",
        )
        print("[6/6] Video export complete", flush=True)
        print(json.dumps(result_record, indent=2, default=str), flush=True)
        return 0

    except Exception as exc:
        result_record.update({
            "status": "failed",
            "elapsed_seconds": round(time.time() - started, 3),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        failed_marker.write_text(
            json.dumps(result_record, indent=2, default=str),
            encoding="utf-8",
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
