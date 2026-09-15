#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/batch_config.json}"
NOTEBOOK="${SCRIPT_DIR}/nuplan_visual_inspection_video_v9_5_31.ipynb"
VIDEO_CELL="${SCRIPT_DIR}/semantic_map_video_cell_v9_5_31.py"
EXPORTER="${SCRIPT_DIR}/run_scenario_export.py"

SCENARIO_TOKEN=""
FRAME_INDEX=""
OVERWRITE=1

usage() {
  cat <<'USAGE'
Usage:
  bash run_frame.sh SCENARIO_TOKEN FRAME_INDEX

or:
  bash run_frame.sh --scenario-token TOKEN --frame INDEX

Options:
  --scenario-token TOKEN   nuPlan scenario token.
  --frame INDEX            Zero-based frame index to render.
  --config FILE            batch_config.json to use.
  --no-overwrite           Keep an already completed single-frame export.
  -h, --help               Show this help.

Examples:
  bash run_frame.sh 1234567890abcdef 25

  bash run_frame.sh \
    --scenario-token 1234567890abcdef \
    --frame 25
USAGE
}

# Convenient positional form: run_frame.sh TOKEN FRAME
if (($# >= 2)) && [[ "$1" != --* ]] && [[ "$2" != --* ]]; then
  SCENARIO_TOKEN="$1"
  FRAME_INDEX="$2"
  shift 2
fi

while (($#)); do
  case "$1" in
    --scenario-token)
      (($# >= 2)) || { echo "ERROR: --scenario-token requires a value." >&2; exit 2; }
      SCENARIO_TOKEN="$2"
      shift 2
      ;;
    --frame)
      (($# >= 2)) || { echo "ERROR: --frame requires an index." >&2; exit 2; }
      FRAME_INDEX="$2"
      shift 2
      ;;
    --config)
      (($# >= 2)) || { echo "ERROR: --config requires a file." >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --no-overwrite)
      OVERWRITE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$SCENARIO_TOKEN" ]] || { echo "ERROR: scenario token is required." >&2; usage >&2; exit 2; }
[[ "$FRAME_INDEX" =~ ^[0-9]+$ ]] || { echo "ERROR: frame must be a non-negative integer." >&2; exit 2; }

for path in "$CONFIG" "$NOTEBOOK" "$VIDEO_CELL" "$EXPORTER"; do
  [[ -f "$path" ]] || { echo "ERROR: Required file not found: $path" >&2; exit 1; }
done

# Make a temporary copy of batch_config.json that renders exactly one frame.
TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/nuplan_frame_config.XXXXXX.json")"
trap 'rm -f -- "$TMP_CONFIG"' EXIT

readarray -t FRAME_PATHS < <("${PYTHON_BIN}" - "$CONFIG" "$TMP_CONFIG" "$SCENARIO_TOKEN" "$FRAME_INDEX" <<'PY'
import json
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
tmp_path = Path(sys.argv[2])
token = sys.argv[3]
frame = int(sys.argv[4])

cfg = json.loads(config_path.read_text(encoding="utf-8"))

base_export_root = Path(cfg["export_root"]).expanduser()
safe_token = re.sub(r"[^A-Za-z0-9_-]+", "_", token)
frame_root = base_export_root / "single_frames" / f"{safe_token}_frame_{frame:05d}"

cfg["video_start_frame"] = frame
cfg["video_end_frame"] = frame
cfg["video_frame_step"] = 1
cfg["export_root"] = str(frame_root)

tmp_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

print(frame_root)
print(cfg["assertion_output_dir"])
PY
)

FRAME_EXPORT_ROOT="${FRAME_PATHS[0]}"
ASSERTION_OUTPUT_DIR="${FRAME_PATHS[1]}"

echo "Scenario token: ${SCENARIO_TOKEN}"
echo "Frame index:    ${FRAME_INDEX}"
echo "Assertions:     ${ASSERTION_OUTPUT_DIR}"
echo "Frame output:   ${FRAME_EXPORT_ROOT}"

ARGS=(
  --config "$TMP_CONFIG"
  --notebook "$NOTEBOOK"
  --video-cell "$VIDEO_CELL"
  --scenario-token "$SCENARIO_TOKEN"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -u "$EXPORTER" "${ARGS[@]}"

MP4_PATH="$(find "$FRAME_EXPORT_ROOT" -type f -name '*.mp4' ! -name '*temporary*' | head -n 1 || true)"

if [[ -n "$MP4_PATH" ]]; then
  echo
  echo "One-frame MP4: $MP4_PATH"

  if command -v ffmpeg >/dev/null 2>&1; then
    PNG_PATH="${MP4_PATH%.mp4}.png"
    ffmpeg -y -loglevel error -i "$MP4_PATH" -frames:v 1 "$PNG_PATH"
    echo "PNG frame:     $PNG_PATH"
  else
    echo "ffmpeg not found; PNG extraction skipped."
  fi
fi
