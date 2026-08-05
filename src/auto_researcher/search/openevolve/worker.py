"""Fixed child-process entry point; never accepts commands or executable paths."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        return 64
    source, entry_point, input_path, output_path = map(Path, sys.argv[1:])
    spec = importlib.util.spec_from_file_location("bounded_candidate", source)
    if spec is None or spec.loader is None:
        return 65
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, entry_point.name)
    configuration = json.loads(input_path.read_text(encoding="utf-8"))
    result = function(configuration)
    output_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
