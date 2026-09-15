"""Command-line entry point for the Predicate KG extractor."""
from __future__ import annotations

import json
import os
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Configure thresholds before importing the extraction implementation."""
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    os.environ["PREDICATE_KG_CLI_ARGS_JSON"] = json.dumps(cli_args)

    from nuplan_predicates_modular import base

    base.initialize_runtime_paths(validate_paths=True)

    # Import only after the configured base module exists. Several legacy
    # modules retain module-level aliases to the configured output paths.
    from nuplan_predicates_modular.pipeline import main as pipeline_main

    result = pipeline_main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
