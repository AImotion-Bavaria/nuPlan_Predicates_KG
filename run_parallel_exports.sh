#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Parallel nuPlan video exporter with automatic category switch
# ============================================================
#
# Examples:
#
#   # One category
#   bash run_parallel_exports.sh \
#     --category interaction \
#     --indices 0:2 \
#     --workers 3 \
#     --overwrite
#
#   # Another category
#   bash run_parallel_exports.sh \
#     --category spatial \
#     --indices 0:2 \
#     --workers 3 \
#     --overwrite
#
#   # Several categories together in the same video
#   bash run_parallel_exports.sh \
#     --categories interaction,map \
#     --indices 0:2 \
#     --workers 3 \
#     --overwrite
#
#   # One category, selected predicates only
#   bash run_parallel_exports.sh \
#     --category interaction \
#     --predicates follows,crossesInFrontOf \
#     --indices 0:49 \
#     --workers 8 \
#     --overwrite
#
#   # Scenario tokens instead of indices
#   bash run_parallel_exports.sh \
#     --category interaction \
#     --tokens-file scenario_tokens.txt \
#     --workers 3 \
#     --overwrite
#
# If --category/--categories is supplied:
#   1. all semantic_categories in batch_config.json are first set to false;
#   2. only the requested category/categories are set to true;
#   3. the original batch_config.json is NOT modified;
#   4. a temporary runtime config is created automatically;
#   5. videos are written under:
#
#        <base export_root>/<category-fragment>/
#
#      Examples:
#        <export_root>/interaction/
#        <export_root>/spatial/
#        <export_root>/interaction_map/
#
# This avoids collisions between DONE.json/video files from different
# category visualizations.
# ============================================================


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_CONFIG="$SCRIPT_DIR/batch_config.json"
CONFIG="$BASE_CONFIG"

NOTEBOOK="$SCRIPT_DIR/nuplan_visual_inspection_video_v9_5_31.ipynb"
VIDEO_CELL="$SCRIPT_DIR/semantic_map_video_cell_v9_5_31.py"

WORKERS="${WORKERS:-8}"
OVERWRITE=0

MODE="indices"
VALUES=""
VALUES_FILE=""

CATEGORY_SPEC=""
CATEGORY_WAS_SET=0

PREDICATE_SPEC=""
PREDICATE_WAS_SET=0

CUSTOM_EXPORT_ROOT=""
TEMP_CONFIG=""


usage() {
  cat <<'EOF'
Usage:
  bash run_parallel_exports.sh --category CATEGORY --indices START:END [options]

Examples:
  bash run_parallel_exports.sh \
    --category interaction \
    --indices 0:49 \
    --workers 8

  bash run_parallel_exports.sh \
    --category spatial \
    --indices 0,3,8,12 \
    --workers 4

  bash run_parallel_exports.sh \
    --categories interaction,map \
    --indices 0:2 \
    --workers 3 \
    --overwrite

  bash run_parallel_exports.sh \
    --category interaction \
    --tokens-file scenario_tokens.txt \
    --workers 8

Category options:
  --category NAME
      Enable exactly one semantic category.
      Example:
        --category interaction

  --categories A,B,C
      Enable several semantic categories together.
      Example:
        --categories interaction,map

      All other semantic categories are automatically disabled.

Predicate options:
  --predicate NAME
      Show one exact predicate inside the enabled category/categories.
      Examples:
        --predicate follows
        --predicate np:crossesInFrontOf

  --predicates A,B,C
      Show several exact predicates together.
      Example:
        --predicates follows,crossesInFrontOf

      Short names are normalized to np:<name>.
      Use --predicates all to preserve the current all-predicate behavior.

Scenario options:
  --indices VALUE
      Inclusive range START:END or comma-separated scenario indices.

      Examples:
        --indices 0:49
        --indices 0,3,8,12

  --tokens-file FILE
      Text file containing one scenario token per line.

Execution options:
  --workers N
      Number of scenarios processed simultaneously.
      Default: 8

  --config FILE
      Base JSON configuration.
      Default: batch_config.json beside this script.

  --export-root DIR
      Override the base video export root.

      When --category/--categories is used, the script still creates:
          DIR/<category-fragment>/

  --overwrite
      Recreate scenarios already marked complete.

  -h, --help
      Show this help.

Notes:
  - The original JSON configuration is never modified.
  - If no --category or --categories is supplied, the category settings
    already present in batch_config.json are used unchanged.
  - If no --predicate or --predicates is supplied, semantic_predicates from
    the selected config are preserved unchanged.
  - Category filtering and predicate filtering are both visualization-only;
    predicate extraction/detection logic is not changed.
EOF
}


cleanup() {
  if [[ -n "${TEMP_CONFIG:-}" && -f "$TEMP_CONFIG" ]]; then
    rm -f -- "$TEMP_CONFIG"
  fi
}
trap cleanup EXIT


while (($#)); do
  case "$1" in

    --category)
      if (($# < 2)); then
        echo "ERROR: --category requires a category name." >&2
        exit 2
      fi

      if (( CATEGORY_WAS_SET == 1 )); then
        echo "ERROR: use only one of --category or --categories." >&2
        exit 2
      fi

      CATEGORY_SPEC="$2"
      CATEGORY_WAS_SET=1
      shift 2
      ;;

    --categories)
      if (($# < 2)); then
        echo "ERROR: --categories requires a comma-separated list." >&2
        exit 2
      fi

      if (( CATEGORY_WAS_SET == 1 )); then
        echo "ERROR: use only one of --category or --categories." >&2
        exit 2
      fi

      CATEGORY_SPEC="$2"
      CATEGORY_WAS_SET=1
      shift 2
      ;;

    --predicate)
      if (($# < 2)); then
        echo "ERROR: --predicate requires a predicate name." >&2
        exit 2
      fi

      if (( PREDICATE_WAS_SET == 1 )); then
        echo "ERROR: use only one of --predicate or --predicates." >&2
        exit 2
      fi

      PREDICATE_SPEC="$2"
      PREDICATE_WAS_SET=1
      shift 2
      ;;

    --predicates)
      if (($# < 2)); then
        echo "ERROR: --predicates requires a comma-separated list." >&2
        exit 2
      fi

      if (( PREDICATE_WAS_SET == 1 )); then
        echo "ERROR: use only one of --predicate or --predicates." >&2
        exit 2
      fi

      PREDICATE_SPEC="$2"
      PREDICATE_WAS_SET=1
      shift 2
      ;;

    --indices)
      if (($# < 2)); then
        echo "ERROR: --indices requires a value." >&2
        exit 2
      fi
      MODE="indices"
      VALUES="$2"
      shift 2
      ;;

    --tokens-file)
      if (($# < 2)); then
        echo "ERROR: --tokens-file requires a file." >&2
        exit 2
      fi
      MODE="tokens"
      VALUES_FILE="$2"
      shift 2
      ;;

    --workers)
      if (($# < 2)); then
        echo "ERROR: --workers requires a positive integer." >&2
        exit 2
      fi
      WORKERS="$2"
      shift 2
      ;;

    --config)
      if (($# < 2)); then
        echo "ERROR: --config requires a JSON file." >&2
        exit 2
      fi
      BASE_CONFIG="$2"
      CONFIG="$BASE_CONFIG"
      shift 2
      ;;

    --export-root)
      if (($# < 2)); then
        echo "ERROR: --export-root requires a directory." >&2
        exit 2
      fi
      CUSTOM_EXPORT_ROOT="$2"
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
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done


# ------------------------------------------------------------
# Validate basic arguments
# ------------------------------------------------------------

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --workers must be a positive integer." >&2
  exit 2
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: configuration file not found:" >&2
  echo "  $BASE_CONFIG" >&2
  exit 2
fi

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: notebook not found:" >&2
  echo "  $NOTEBOOK" >&2
  exit 2
fi

if [[ ! -f "$VIDEO_CELL" ]]; then
  echo "ERROR: video cell not found:" >&2
  echo "  $VIDEO_CELL" >&2
  exit 2
fi


# ------------------------------------------------------------
# Create an effective runtime configuration.
#
# IMPORTANT:
# batch_config.json itself is never changed.
# ------------------------------------------------------------

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/nuplan_video_config_XXXXXX.json")"

RUNTIME_FRAGMENT="$(
python - "$BASE_CONFIG" "$TEMP_CONFIG" \
  "$CATEGORY_WAS_SET" "$CATEGORY_SPEC" \
  "$PREDICATE_WAS_SET" "$PREDICATE_SPEC" \
  "$CUSTOM_EXPORT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

base_config_path = Path(sys.argv[1]).expanduser().resolve()
temp_config_path = Path(sys.argv[2]).expanduser().resolve()
category_was_set = bool(int(sys.argv[3]))
category_spec = sys.argv[4].strip()
predicate_was_set = bool(int(sys.argv[5]))
predicate_spec = sys.argv[6].strip()
custom_export_root = sys.argv[7].strip()

with base_config_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

semantic_categories = cfg.get("semantic_categories")
if not isinstance(semantic_categories, dict) or not semantic_categories:
    raise SystemExit(
        "ERROR: config does not contain a non-empty "
        "'semantic_categories' dictionary."
    )

available = list(semantic_categories.keys())

if category_was_set:
    requested_categories = [
        item.strip()
        for item in category_spec.split(",")
        if item.strip()
    ]

    if not requested_categories:
        raise SystemExit("ERROR: no category names were parsed.")

    requested_categories = list(dict.fromkeys(requested_categories))

    unknown = [
        item for item in requested_categories
        if item not in semantic_categories
    ]

    if unknown:
        raise SystemExit(
            "ERROR: unknown semantic category/categories: "
            + ", ".join(unknown)
            + "\nAvailable categories: "
            + ", ".join(available)
        )

    for key in semantic_categories:
        semantic_categories[key] = False

    for key in requested_categories:
        semantic_categories[key] = True
else:
    requested_categories = [
        key
        for key, enabled in semantic_categories.items()
        if bool(enabled)
    ]

    if not requested_categories:
        raise SystemExit(
            "ERROR: no semantic categories are enabled in the config."
        )


# Predicate selection is a second visualization gate.
# If it is not explicitly supplied, preserve the config exactly as-is.
if predicate_was_set:
    if not predicate_spec or predicate_spec.lower() in {"all", "*"}:
        requested_predicates = "all"
    else:
        raw_predicates = [
            item.strip()
            for item in predicate_spec.split(",")
            if item.strip()
        ]

        if not raw_predicates:
            raise SystemExit("ERROR: no predicate names were parsed.")

        normalized = []
        for value in raw_predicates:
            if value.lower() in {"all", "*"}:
                normalized = []
                requested_predicates = "all"
                break

            predicate_id = (
                value
                if value.startswith("np:")
                else f"np:{value}"
            )

            if predicate_id not in normalized:
                normalized.append(predicate_id)
        else:
            requested_predicates = normalized

        if not normalized and "requested_predicates" not in locals():
            requested_predicates = "all"

    cfg["semantic_predicates"] = requested_predicates
else:
    requested_predicates = cfg.get("semantic_predicates", "all")


if custom_export_root:
    base_export_root = Path(custom_export_root).expanduser()
else:
    if not cfg.get("export_root"):
        raise SystemExit(
            "ERROR: config does not contain 'export_root'."
        )
    base_export_root = Path(cfg["export_root"]).expanduser()


category_fragment = "_".join(requested_categories)
category_fragment = re.sub(
    r"[^A-Za-z0-9_-]+",
    "_",
    category_fragment,
).strip("_") or "semantic"


if requested_predicates == "all":
    predicate_fragment = "all"
else:
    predicate_fragment = "_".join(
        predicate.replace("np:", "", 1)
        for predicate in requested_predicates
    )
    predicate_fragment = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        predicate_fragment,
    ).strip("_") or "selected"


# Preserve the old output location when no new predicate selector is used.
# With explicit predicates, add one extra level to prevent DONE.json
# collisions between different predicate selections.
if category_was_set:
    effective_export_root = base_export_root / category_fragment
else:
    effective_export_root = base_export_root

if predicate_was_set:
    effective_export_root = effective_export_root / predicate_fragment

cfg["export_root"] = str(effective_export_root)

cfg["_video_runtime"] = {
    "selected_semantic_categories": requested_categories,
    "selected_semantic_predicates": requested_predicates,
    "category_fragment": category_fragment,
    "predicate_fragment": predicate_fragment,
    "base_config": str(base_config_path),
}

temp_config_path.parent.mkdir(parents=True, exist_ok=True)
with temp_config_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print(category_fragment)
print(predicate_fragment)
PY
)"

CONFIG="$TEMP_CONFIG"


# ------------------------------------------------------------
# Read effective values for display and setup
# ------------------------------------------------------------

readarray -t CONFIG_INFO < <(
python - "$CONFIG" <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))

enabled = [
    name
    for name, value in cfg["semantic_categories"].items()
    if bool(value)
]

predicates = cfg.get("semantic_predicates", "all")
if predicates == "all":
    predicates_text = "all"
elif isinstance(predicates, (list, tuple, set)):
    predicates_text = ",".join(str(x) for x in predicates)
else:
    predicates_text = str(predicates)

print(cfg["export_root"])
print(",".join(enabled))
print(predicates_text)
print(cfg.get("assertion_output_dir", ""))
PY
)

EXPORT_ROOT="${CONFIG_INFO[0]}"
ENABLED_CATEGORIES="${CONFIG_INFO[1]}"
ENABLED_PREDICATES="${CONFIG_INFO[2]}"
ASSERTION_OUTPUT_DIR="${CONFIG_INFO[3]}"

mkdir -p \
  "$EXPORT_ROOT/_batch/logs" \
  "$EXPORT_ROOT/_batch/manifests"


# Save a copy of the effective configuration with the videos.
EFFECTIVE_CONFIG_COPY="$EXPORT_ROOT/_batch/effective_video_config.json"
cp -- "$CONFIG" "$EFFECTIVE_CONFIG_COPY"


# ------------------------------------------------------------
# Optional friendly warning if Stable 3 category_counts.csv exists
# and a selected category was not written by predicate extraction.
# ------------------------------------------------------------

if [[ -n "$ASSERTION_OUTPUT_DIR" && -f "$ASSERTION_OUTPUT_DIR/category_counts.csv" ]]; then
  python - "$ASSERTION_OUTPUT_DIR/category_counts.csv" "$ENABLED_CATEGORIES" <<'PY'
import csv
import sys
from pathlib import Path

counts_file = Path(sys.argv[1])
requested = [x for x in sys.argv[2].split(",") if x]

present = set()

try:
    with counts_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (
                row.get("category")
                or row.get("family")
                or ""
            ).strip()
            if category:
                present.add(category)
except Exception:
    raise SystemExit(0)

missing = [x for x in requested if x not in present]

if missing:
    print(
        "WARNING: selected video category/categories are not listed "
        "in the predicate extraction category_counts.csv:"
    )
    for item in missing:
        print(f"  - {item}")
    print(
        "The video can still render map/agent context, but semantic "
        "edges for a missing family may be absent."
    )
PY
fi


# ------------------------------------------------------------
# Build task file
# ------------------------------------------------------------

TASK_FILE="$EXPORT_ROOT/_batch/manifests/tasks_$(date +%Y%m%d_%H%M%S).tsv"

if [[ "$MODE" == "indices" ]]; then

  if [[ -z "$VALUES" ]]; then
    echo "ERROR: provide --indices or --tokens-file." >&2
    exit 2
  fi

  if [[ "$VALUES" =~ ^([0-9]+):([0-9]+)$ ]]; then

    START="${BASH_REMATCH[1]}"
    END="${BASH_REMATCH[2]}"

    if (( END < START )); then
      echo "ERROR: invalid index range: $VALUES" >&2
      exit 2
    fi

    for ((i=START; i<=END; i++)); do
      printf 'index\t%s\n' "$i"
    done > "$TASK_FILE"

  else

    : > "$TASK_FILE"

    IFS=',' read -ra ITEMS <<< "$VALUES"

    for i in "${ITEMS[@]}"; do
      i="${i//[[:space:]]/}"

      if ! [[ "$i" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid scenario index: $i" >&2
        exit 2
      fi

      printf 'index\t%s\n' "$i" >> "$TASK_FILE"
    done
  fi

else

  if [[ ! -f "$VALUES_FILE" ]]; then
    echo "ERROR: token file not found:" >&2
    echo "  $VALUES_FILE" >&2
    exit 2
  fi

  awk '
    NF && $1 !~ /^#/ {
      print "token\t" $1
    }
  ' "$VALUES_FILE" > "$TASK_FILE"
fi


TASK_COUNT="$(wc -l < "$TASK_FILE")"

if (( TASK_COUNT == 0 )); then
  echo "ERROR: no scenarios were selected." >&2
  exit 2
fi


# ------------------------------------------------------------
# Print resolved run configuration
# ------------------------------------------------------------

echo
echo "Video export configuration"
echo "--------------------------"
echo "Semantic categories: $ENABLED_CATEGORIES"
echo "Semantic predicates: $ENABLED_PREDICATES"
echo "Assertion output:     $ASSERTION_OUTPUT_DIR"
echo "Video export root:    $EXPORT_ROOT"
echo "Parallel workers:     $WORKERS"
echo "Tasks:                $TASK_COUNT"
echo "Overwrite:            $OVERWRITE"
echo "Effective config:     $EFFECTIVE_CONFIG_COPY"
echo "Task list:            $TASK_FILE"
echo


# ------------------------------------------------------------
# Parallel rendering
# ------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

export \
  SCRIPT_DIR \
  CONFIG \
  NOTEBOOK \
  VIDEO_CELL \
  EXPORT_ROOT \
  OVERWRITE


run_one() {
  local kind="$1"
  local value="$2"

  local arg
  local log_name

  if [[ "$kind" == "index" ]]; then
    arg="--scenario-index"
    log_name="index_$(printf '%05d' "$value")"
  else
    arg="--scenario-token"
    log_name="token_${value//[^A-Za-z0-9_-]/_}"
  fi

  local overwrite_arg=()

  if (( OVERWRITE == 1 )); then
    overwrite_arg=(--overwrite)
  fi

  python -u "$SCRIPT_DIR/run_scenario_export.py" \
    --config "$CONFIG" \
    --notebook "$NOTEBOOK" \
    --video-cell "$VIDEO_CELL" \
    "$arg" "$value" \
    "${overwrite_arg[@]}" \
    > "$EXPORT_ROOT/_batch/logs/${log_name}.log" 2>&1
}

export -f run_one


# One independent Python process per scenario.
# A failed scenario does not prevent the other scenario jobs from running.
set +e

xargs \
  -P "$WORKERS" \
  -n 2 \
  bash -c 'run_one "$1" "$2"' _ \
  < "$TASK_FILE"

XARGS_STATUS=$?

set -e


# ------------------------------------------------------------
# Build manifest
# ------------------------------------------------------------

python "$SCRIPT_DIR/build_batch_manifest.py" \
  --export-root "$EXPORT_ROOT"


echo
echo "Batch finished."
echo
echo "Category/categories:"
echo "  $ENABLED_CATEGORIES"
echo
echo "Predicate/predicates:"
echo "  $ENABLED_PREDICATES"
echo
echo "Videos:"
echo "  $EXPORT_ROOT"
echo
echo "Batch manifest:"
echo "  $EXPORT_ROOT/_batch/batch_manifest.csv"
echo
echo "Logs:"
echo "  $EXPORT_ROOT/_batch/logs/"
echo

exit "$XARGS_STATUS"
