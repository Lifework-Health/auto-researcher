import importlib.util
import json
import math
import sys
from pathlib import Path


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite output")
    if isinstance(value, dict):
        return all(isinstance(k, str) and finite(v) for k, v in value.items())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


source, entry_point, input_path, output_path, candidate_id, execution_identity = (
    sys.argv[1:]
)
spec = importlib.util.spec_from_file_location("candidate", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = getattr(module, entry_point)(json.loads(Path(input_path).read_text()))
if not isinstance(result, dict) or not finite(result):
    raise ValueError("invalid structured output")
payload = {
    "schema": "openevolve-candidate-output-v1",
    "candidate_id": candidate_id,
    "execution_identity": execution_identity,
    "configuration": result,
}
Path(output_path).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
