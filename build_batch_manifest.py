#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--export-root", type=Path, required=True)
args = parser.parse_args()
root = args.export_root
rows = []

for scenario_dir in sorted(root.glob("scenario_*")):
    candidates = [
        scenario_dir / "DONE.json",
        scenario_dir / "FAILED.json",
    ]
    data = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            break
        except Exception:
            pass

    if data is None:
        data = {
            "status": "unknown",
            "scenario_directory": str(scenario_dir),
        }

    rows.append({
        "scenario_catalog_row": data.get("scenario_catalog_row", ""),
        "scenario_token": data.get("scenario_token", ""),
        "status": data.get("status", ""),
        "elapsed_seconds": data.get("elapsed_seconds", ""),
        "rendered_frames": data.get("rendered_frames", ""),
        "active_frame_count": data.get("active_frame_count", ""),
        "fps": data.get("fps", ""),
        "video_path": data.get("video_path", ""),
        "video_frame_summary_path": data.get(
            "video_frame_summary_path", ""
        ),
        "video_edge_events_path": data.get(
            "video_edge_events_path", ""
        ),
        "scenario_directory": data.get(
            "scenario_directory", str(scenario_dir)
        ),
        "error": data.get("error", ""),
    })

out_dir = root / "_batch"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "batch_manifest.csv"
fields = list(rows[0]) if rows else [
    "scenario_catalog_row",
    "scenario_token",
    "status",
    "elapsed_seconds",
    "rendered_frames",
    "active_frame_count",
    "fps",
    "video_path",
    "video_frame_summary_path",
    "video_edge_events_path",
    "scenario_directory",
    "error",
]

with out_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(out_path)
