"""Fixed untrusted candidate child used only by the image-owned supervisor."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import resource
import sys
from pathlib import Path


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, dict):
        return all(isinstance(key, str) and finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool))


def main() -> None:
    (
        source,
        entry_point,
        input_path,
        control_fd_text,
        output_bytes_text,
        file_size_bytes_text,
        cpu_time_text,
    ) = sys.argv[1:]
    control_fd = int(control_fd_text)
    output_bytes = int(output_bytes_text)
    file_size_bytes = int(file_size_bytes_text)
    cpu_time_seconds = int(cpu_time_text)
    if output_bytes <= 0:
        raise ValueError("invalid output limit")
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_seconds, cpu_time_seconds))
    spec = importlib.util.spec_from_file_location("candidate", source)
    if spec is None or spec.loader is None:
        raise ValueError("candidate source unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = getattr(module, entry_point)(json.loads(Path(input_path).read_text()))
    if not isinstance(result, dict) or not finite(result):
        raise ValueError("invalid structured output")
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    os.write(control_fd, encoded)
    os.close(control_fd)


if __name__ == "__main__":
    main()
