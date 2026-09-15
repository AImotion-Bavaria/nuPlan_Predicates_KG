#!/usr/bin/env python3
"""Check that Python resolves the bundled repository modules from this checkout."""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
BUNDLED_SRC = ROOT
# Both public packages live directly under the repository root.
for _path in (ROOT,):
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

package_init = ROOT / "nuplan_predicates_modular" / "__init__.py"
pipeline_file = ROOT / "nuplan_predicates_modular" / "pipeline.py"

print("Repository root:", ROOT)
print("Package __init__ exists:", package_init.exists(), package_init)
print("Pipeline exists:", pipeline_file.exists(), pipeline_file)

package_spec = importlib.util.find_spec("nuplan_predicates_modular")
pipeline_spec = importlib.util.find_spec("nuplan_predicates_modular.pipeline")

print("Resolved package:", None if package_spec is None else package_spec.origin)
print("Resolved pipeline:", None if pipeline_spec is None else pipeline_spec.origin)

expected_pipeline = pipeline_file.resolve()

if package_spec is None:
    raise SystemExit("ERROR: nuplan_predicates_modular cannot be resolved.")

if pipeline_spec is None or pipeline_spec.origin is None:
    raise SystemExit("ERROR: nuplan_predicates_modular.pipeline cannot be resolved.")

resolved_pipeline = Path(pipeline_spec.origin).resolve()
if resolved_pipeline != expected_pipeline:
    raise SystemExit(
        "ERROR: Python is resolving pipeline.py from the wrong checkout:\n"
        f"  expected: {expected_pipeline}\n"
        f"  resolved: {resolved_pipeline}"
    )

print("OK: bundled pipeline resolves correctly.")


# Video-export runtime must resolve the bundled nuPlan adapter as well.
from nuplan_predicate_kg.adapters import nuplan_adapter as video_adapter
resolved_adapter = Path(video_adapter.__file__).resolve()
expected_adapter = (ROOT / "nuplan_predicate_kg" / "adapters" / "nuplan_adapter.py").resolve()
print("Resolved nuPlan adapter:", resolved_adapter)
if resolved_adapter != expected_adapter:
    raise SystemExit(
        "ERROR: Python is resolving nuplan_adapter.py from the wrong checkout:\n"
        f"  expected: {expected_adapter}\n"
        f"  resolved: {resolved_adapter}"
    )
print("OK: bundled nuPlan adapter resolves correctly.")
