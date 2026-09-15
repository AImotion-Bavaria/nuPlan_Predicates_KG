#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# run_category_videos.sh
#
# Wrapper around the existing run_parallel_exports.sh.
#
# It:
#   1. takes one predicate extraction directory;
#   2. selects one or more semantic categories;
#   3. discovers all scenarios automatically;
#   4. creates a temporary batch_config.json;
#   5. calls the existing run_parallel_exports.sh using --tokens-file.
#
# The original batch_config.json and run_parallel_exports.sh are untouched.
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_CONFIG="${SCRIPT_DIR}/batch_config.json"
OLD_RUNNER="${SCRIPT_DIR}/run_parallel_exports.sh"

PREDICATE_OUTPUT=""
CATEGORY_SPEC=""
PREDICATE_SPEC="all"
WORKERS="${WORKERS:-8}"
MAX_SCENARIOS="all"
OVERWRITE=0
CUSTOM_EXPORT_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  bash run_category_videos.sh \
    --predicate-output /path/to/outputs_python/RUN_NAME \
    --category interaction \
    --workers 3 \
    --overwrite

Options:
  --predicate-output DIR
      Predicate extraction directory.

  --category NAME
      Show one category only.
      Example: interaction

  --categories A,B,C
      Show several categories together.
      Example: interaction,map

  --predicate NAME
      Show one exact predicate inside the enabled category/categories.
      Examples: follows, np:follows, crossesInFrontOf

  --predicates A,B,C
      Show several exact predicates together.
      Example: follows,overtakes,crossesInFrontOf

      Default: all
      Use --predicates all to show every predicate in the enabled family.

  --workers N
      Parallel scenario workers. Default: 8

  --max-scenarios N
      Render at most N unique matching scenarios.
      Default: all

      When --predicate/--predicates is used, the script first finds only
      scenarios containing the selected predicate(s), then keeps the first N.

  --export-root DIR
      Optional base video output directory.
      Default is derived automatically as:
        <project_root>/visual_inspection/<run_name>

      Final videos are separated by family and predicate:
        <base>/<family>/all/
        <base>/<family>/<predicate>/

  --config FILE
      Base batch_config.json.
      Default: package batch_config.json

  --overwrite
      Re-render scenarios even if DONE.json exists.

  -h, --help
      Show this help.

Examples:

  bash run_category_videos.sh \
    --predicate-output ${PREDICATE_OUTPUT:-outputs_python/paper_release} \
    --category interaction \
    --workers 3 \
    --overwrite

  bash run_category_videos.sh \
    --predicate-output ${PREDICATE_OUTPUT:-outputs_python/paper_release} \
    --category interaction \
    --predicate follows \
    --max-scenarios 10 \
    --workers 3 \
    --overwrite

  bash run_category_videos.sh \
    --predicate-output ${PREDICATE_OUTPUT:-outputs_python/paper_release} \
    --category spatial \
    --predicates all \
    --workers 3 \
    --overwrite

  bash run_category_videos.sh \
    --predicate-output ${PREDICATE_OUTPUT:-outputs_python/paper_release} \
    --categories interaction,map \
    --workers 3 \
    --overwrite
EOF
}


while (($#)); do
  case "$1" in
    --predicate-output)
      [[ $# -ge 2 ]] || { echo "ERROR: --predicate-output requires DIR" >&2; exit 2; }
      PREDICATE_OUTPUT="$2"
      shift 2
      ;;

    --category|--categories)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      [[ -z "$CATEGORY_SPEC" ]] || {
        echo "ERROR: use only one of --category or --categories." >&2
        exit 2
      }
      CATEGORY_SPEC="$2"
      shift 2
      ;;

    --predicate|--predicates)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      PREDICATE_SPEC="$2"
      shift 2
      ;;

    --workers)
      [[ $# -ge 2 ]] || { echo "ERROR: --workers requires N" >&2; exit 2; }
      WORKERS="$2"
      shift 2
      ;;

    --max-scenarios)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-scenarios requires N or all" >&2; exit 2; }
      MAX_SCENARIOS="$2"
      shift 2
      ;;

    --export-root)
      [[ $# -ge 2 ]] || { echo "ERROR: --export-root requires DIR" >&2; exit 2; }
      CUSTOM_EXPORT_ROOT="$2"
      shift 2
      ;;

    --config)
      [[ $# -ge 2 ]] || { echo "ERROR: --config requires FILE" >&2; exit 2; }
      BASE_CONFIG="$2"
      shift 2
      ;;

    --overwrite)
      OVERWRITE=1
      shift
      ;;

    -h|--help)
      usage
      exit 0
      ;;

    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done


[[ -n "$PREDICATE_OUTPUT" ]] || {
  echo "ERROR: --predicate-output is required." >&2
  exit 2
}

[[ -n "$CATEGORY_SPEC" ]] || {
  echo "ERROR: --category or --categories is required." >&2
  exit 2
}

[[ -d "$PREDICATE_OUTPUT" ]] || {
  echo "ERROR: predicate output directory does not exist:" >&2
  echo "  $PREDICATE_OUTPUT" >&2
  exit 2
}

[[ -f "$BASE_CONFIG" ]] || {
  echo "ERROR: base config does not exist:" >&2
  echo "  $BASE_CONFIG" >&2
  exit 2
}

[[ -f "$OLD_RUNNER" ]] || {
  echo "ERROR: existing run_parallel_exports.sh not found:" >&2
  echo "  $OLD_RUNNER" >&2
  exit 2
}

[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --workers must be a positive integer." >&2
  exit 2
}

if [[ "$MAX_SCENARIOS" != "all" ]]; then
  [[ "$MAX_SCENARIOS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: --max-scenarios must be a positive integer or 'all'." >&2
    exit 2
  }
fi


PREDICATE_OUTPUT="$(cd -- "$PREDICATE_OUTPUT" && pwd)"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nuplan_category_video_XXXXXX")"
TMP_CONFIG="${TMP_DIR}/batch_config.json"
TOKENS_FILE="${TMP_DIR}/scenario_tokens.txt"

cleanup() {
  rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT


# ------------------------------------------------------------
# Create temporary configuration.
# ------------------------------------------------------------

CONFIG_RESULT="$(
python - "$BASE_CONFIG" "$TMP_CONFIG" "$PREDICATE_OUTPUT" "$CATEGORY_SPEC" "$PREDICATE_SPEC" "$CUSTOM_EXPORT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

base_config = Path(sys.argv[1]).expanduser().resolve()
target_config = Path(sys.argv[2]).expanduser().resolve()
predicate_output = Path(sys.argv[3]).expanduser().resolve()
category_spec = sys.argv[4].strip()
predicate_spec = sys.argv[5].strip()
custom_export_root = sys.argv[6].strip()

with base_config.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

semantic_categories = cfg.get("semantic_categories")

if not isinstance(semantic_categories, dict) or not semantic_categories:
    raise SystemExit(
        "ERROR: batch_config.json has no semantic_categories dictionary."
    )

requested = [
    item.strip()
    for item in category_spec.split(",")
    if item.strip()
]
requested = list(dict.fromkeys(requested))

if not requested:
    raise SystemExit("ERROR: no category was parsed.")

unknown = [
    item for item in requested
    if item not in semantic_categories
]

if unknown:
    raise SystemExit(
        "ERROR: unknown category/categories: "
        + ", ".join(unknown)
        + "\nAvailable: "
        + ", ".join(semantic_categories.keys())
    )

# Turn everything off, then enable only requested categories.
for key in semantic_categories:
    semantic_categories[key] = False

for key in requested:
    semantic_categories[key] = True

# Exact predicate filter. The family selection above remains the first gate.
# "all" keeps the old behavior. Short names are normalized to np:<name>.
if not predicate_spec or predicate_spec.lower() in {"all", "*"}:
    requested_predicates = "all"
    predicate_fragment = "all"
else:
    values = [
        item.strip()
        for item in predicate_spec.split(",")
        if item.strip()
    ]
    if not values:
        raise SystemExit("ERROR: no predicate was parsed.")

    normalized = []
    for value in values:
        if value.lower() in {"all", "*"}:
            normalized = []
            requested_predicates = "all"
            break
        predicate_id = value if value.startswith("np:") else f"np:{value}"
        if predicate_id not in normalized:
            normalized.append(predicate_id)
    else:
        requested_predicates = normalized

    if requested_predicates == "all":
        predicate_fragment = "all"
    else:
        predicate_fragment = "_".join(
            predicate.replace("np:", "", 1)
            for predicate in requested_predicates
        )
        predicate_fragment = re.sub(
            r"[^A-Za-z0-9_-]+", "_", predicate_fragment
        ).strip("_") or "selected"

cfg["semantic_predicates"] = requested_predicates
cfg["assertion_output_dir"] = str(predicate_output)

run_name = predicate_output.name
project_root = predicate_output.parent.parent

if custom_export_root:
    base_export_root = Path(custom_export_root).expanduser().resolve()
else:
    base_export_root = project_root / "visual_inspection" / run_name

fragment = "_".join(requested)
fragment = re.sub(r"[^A-Za-z0-9_-]+", "_", fragment).strip("_")
if not fragment:
    fragment = "semantic"

# Save first by family selection, then by exact predicate selection.
# Examples:
#   visual_inspection/RUN/interaction/all/
#   visual_inspection/RUN/interaction/follows/
#   visual_inspection/RUN/interaction/crossesInFrontOf/
#   visual_inspection/RUN/interaction/follows_crossesInFrontOf/
export_root = base_export_root / fragment / predicate_fragment
cfg["export_root"] = str(export_root)

target_config.parent.mkdir(parents=True, exist_ok=True)
with target_config.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print(",".join(requested))
print(export_root)
if requested_predicates == "all":
    print("all")
else:
    print(",".join(requested_predicates))
PY
)"

mapfile -t CONFIG_LINES <<< "$CONFIG_RESULT"

ENABLED_CATEGORIES="${CONFIG_LINES[0]}"
EXPORT_ROOT="${CONFIG_LINES[1]}"
ENABLED_PREDICATES="${CONFIG_LINES[2]}"


# ------------------------------------------------------------
# Discover scenario tokens.
#
# If an exact predicate filter is active, scan the Parquet assertions and keep
# ONLY scenarios containing at least one selected predicate. This makes
# predicate-specific video export fast and avoids empty videos.
#
# If predicates=all, preserve the normal family-wide behavior and prefer
# scenario_processing_times.csv, falling back to Parquet.
# ------------------------------------------------------------

SCENARIO_TIMES="${PREDICATE_OUTPUT}/scenario_processing_times.csv"

if [[ "$ENABLED_PREDICATES" != "all" ]]; then

  echo "Exact predicate filter active; selecting only scenarios containing:"
  echo "  $ENABLED_PREDICATES"

  python - "$PREDICATE_OUTPUT" "$TOKENS_FILE" "$ENABLED_PREDICATES" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
target = Path(sys.argv[2])
selected = {
    value.strip()
    for value in sys.argv[3].split(",")
    if value.strip()
}

files = sorted({
    p.resolve()
    for pattern in (
        "workers/worker_*/batches/batch_*.parquet",
        "batches/batch_*.parquet",
        "**/batches/batch_*.parquet",
    )
    for p in root.glob(pattern)
    if p.is_file()
})

if not files:
    raise SystemExit(
        f"ERROR: no batch_*.parquet files found under {root}"
    )

tokens = []
seen = set()

for path in files:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)

    token_col = None
    for candidate in ("scenario_token", "prov_scenario_token"):
        if candidate in names:
            token_col = candidate
            break

    if token_col is None or "predicate_id" not in names:
        continue

    for batch in parquet.iter_batches(
        batch_size=100000,
        columns=[token_col, "predicate_id"],
    ):
        token_values = batch.column(0).to_pylist()
        predicate_values = batch.column(1).to_pylist()

        for token_value, predicate_value in zip(
            token_values,
            predicate_values,
        ):
            if token_value is None or predicate_value is None:
                continue

            predicate_id = str(predicate_value).strip()
            if predicate_id not in selected:
                continue

            token = str(token_value).strip()
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)

with target.open("w", encoding="utf-8") as f:
    for token in tokens:
        f.write(token + "\n")

print(len(tokens))
PY

elif [[ -f "$SCENARIO_TIMES" ]]; then

  python - "$SCENARIO_TIMES" "$TOKENS_FILE" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])

tokens = []
seen = set()

with source.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    if not reader.fieldnames or "scenario_token" not in reader.fieldnames:
        raise SystemExit(
            "ERROR: scenario_processing_times.csv has no scenario_token column."
        )

    for row in reader:
        token = str(row.get("scenario_token", "")).strip()
        status = str(row.get("status", "")).strip().lower()

        # Keep successful rows. Older timing files may not have status.
        if status and status not in {"completed", "ok", "success"}:
            continue

        if token and token not in seen:
            seen.add(token)
            tokens.append(token)

with target.open("w", encoding="utf-8") as f:
    for token in tokens:
        f.write(token + "\n")

print(len(tokens))
PY

else

  echo "scenario_processing_times.csv not found; discovering tokens from Parquet."

  python - "$PREDICATE_OUTPUT" "$TOKENS_FILE" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
target = Path(sys.argv[2])

files = sorted({
    p.resolve()
    for pattern in (
        "workers/worker_*/batches/batch_*.parquet",
        "batches/batch_*.parquet",
        "**/batches/batch_*.parquet",
    )
    for p in root.glob(pattern)
    if p.is_file()
})

if not files:
    raise SystemExit(
        f"ERROR: no batch_*.parquet files found under {root}"
    )

tokens = []
seen = set()

for path in files:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)

    token_col = None
    for candidate in ("scenario_token", "prov_scenario_token"):
        if candidate in names:
            token_col = candidate
            break

    if token_col is None:
        continue

    for batch in parquet.iter_batches(
        batch_size=100000,
        columns=[token_col],
    ):
        for value in batch.column(0).to_pylist():
            if value is None:
                continue

            token = str(value).strip()

            if token and token not in seen:
                seen.add(token)
                tokens.append(token)

if not tokens:
    raise SystemExit(
        "ERROR: no scenario tokens were found in the predicate output."
    )

with target.open("w", encoding="utf-8") as f:
    for token in tokens:
        f.write(token + "\n")

print(len(tokens))
PY

fi


# ------------------------------------------------------------
# Optional limit on UNIQUE matching scenarios.
# Predicate filtering has already happened above, so when an exact
# predicate is selected every retained token contains that predicate.
# ------------------------------------------------------------

DISCOVERED_SCENARIO_COUNT="$(grep -cve '^[[:space:]]*$' "$TOKENS_FILE" || true)"

if [[ "$DISCOVERED_SCENARIO_COUNT" -eq 0 ]]; then
  echo "ERROR: no scenarios were found in:" >&2
  echo "  $PREDICATE_OUTPUT" >&2
  exit 1
fi

if [[ "$MAX_SCENARIOS" != "all" ]] &&    (( DISCOVERED_SCENARIO_COUNT > MAX_SCENARIOS )); then
  LIMITED_TOKENS_FILE="${TMP_DIR}/scenario_tokens_limited.txt"
  head -n "$MAX_SCENARIOS" "$TOKENS_FILE" > "$LIMITED_TOKENS_FILE"
  mv "$LIMITED_TOKENS_FILE" "$TOKENS_FILE"
fi

SCENARIO_COUNT="$(grep -cve '^[[:space:]]*$' "$TOKENS_FILE" || true)"

if [[ "$SCENARIO_COUNT" -eq 0 ]]; then
  echo "ERROR: no scenarios were found in:" >&2
  echo "  $PREDICATE_OUTPUT" >&2
  exit 1
fi


echo
echo "Category video export"
echo "---------------------"
echo "Predicate output:      $PREDICATE_OUTPUT"
echo "Semantic categories:   $ENABLED_CATEGORIES"
echo "Semantic predicates:   $ENABLED_PREDICATES"
echo "Scenarios discovered:  $DISCOVERED_SCENARIO_COUNT"
echo "Scenario limit:        $MAX_SCENARIOS"
echo "Scenarios to render:   $SCENARIO_COUNT"
echo "Workers:               $WORKERS"
echo "Video output:          $EXPORT_ROOT"
echo


RUN_ARGS=(
  --tokens-file "$TOKENS_FILE"
  --workers "$WORKERS"
  --config "$TMP_CONFIG"
)

if [[ "$OVERWRITE" -eq 1 ]]; then
  RUN_ARGS+=(--overwrite)
fi

exec bash "$OLD_RUNNER" "${RUN_ARGS[@]}"
