#!/usr/bin/env python3
"""Validate a predicate run without loading all Parquet rows into memory."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def find_parquet(root: Path):
    return sorted({p.resolve() for p in root.rglob("batch_*.parquet") if p.is_file()})


def find_definitions(root: Path):
    preferred = root / "predicate_definitions.csv"
    if preferred.exists():
        return preferred
    matches = list(root.rglob("predicate_definitions.csv"))
    return matches[0] if matches else None


def main():
    args = parse_args()
    root = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"ERROR: output directory not found: {root}")

    definition_path = find_definitions(root)
    if definition_path is None:
        raise SystemExit("ERROR: predicate_definitions.csv not found")

    defs = pd.read_csv(definition_path)
    definition_map = dict(zip(defs["predicate_id"].astype(str), defs["category"].astype(str)))

    active_path = root / "active_categories.json"
    if not active_path.exists():
        matches = list(root.rglob("active_categories.json"))
        active_path = matches[0] if matches else None

    requested = None
    written_expected = None
    if active_path and Path(active_path).exists():
        active = json.loads(Path(active_path).read_text())
        requested = set(active.get("requested", []))
        written_expected = set(active.get("write", []))

    parquet_files = find_parquet(root)
    if not parquet_files:
        raise SystemExit("ERROR: no batch_*.parquet files found")

    predicate_counts = Counter()
    category_counts = Counter()
    unknown_predicates = Counter()
    required_columns = {
        "predicate_id", "subject_id", "valid_time_us", "prov_scenario_token"
    }
    missing_required = {}

    for path in parquet_files:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        missing = sorted(required_columns - names)
        if missing:
            missing_required[str(path)] = missing
        for batch in pf.iter_batches(columns=["predicate_id"], batch_size=100_000):
            frame = batch.to_pandas()
            for pid, count in frame["predicate_id"].astype(str).value_counts().items():
                n = int(count)
                predicate_counts[pid] += n
                category = definition_map.get(pid)
                if category is None:
                    unknown_predicates[pid] += n
                    category = "unknown"
                category_counts[category] += n

    actual_categories = set(category_counts)
    unexpected = (
        sorted(actual_categories - written_expected)
        if written_expected is not None
        else []
    )

    print("VALIDATION")
    print("Output:", root)
    print("Parquet files:", len(parquet_files))
    print("Assertions:", f"{sum(predicate_counts.values()):,}")
    if requested is not None:
        print("Requested categories:", ",".join(sorted(requested)))
    if written_expected is not None:
        print("Expected written categories:", ",".join(sorted(written_expected)))
    print("Actual written categories:", ",".join(sorted(actual_categories)))
    print()
    for category, count in sorted(category_counts.items()):
        print(f"  {category:22s} {count:,}")

    problems = []
    if unexpected:
        problems.append(f"unexpected written categories: {unexpected}")
    if unknown_predicates:
        problems.append(f"unknown predicate ids: {sorted(unknown_predicates)}")
    if missing_required:
        problems.append(f"Parquet files missing required columns: {len(missing_required)}")

    if problems:
        print("\nFAILED")
        for problem in problems:
            print("-", problem)
        raise SystemExit(1)

    print("\nPASSED: output categories and compact Parquet schema are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
