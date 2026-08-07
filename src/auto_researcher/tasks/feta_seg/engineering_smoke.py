"""Command-line entry point for the non-scientific FeTA CUDA engineering smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_researcher.tasks.feta_seg.runner import run_engineering_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = run_engineering_smoke(arguments.data_dir)
    except Exception as exc:
        safe_reason = str(exc)
        safe_codes = {
            "feta_cuda_unavailable_for_full_baseline",
            "feta_ml_dependencies_unavailable",
            "feta_dataset_identity_mismatch",
        }
        print(
            json.dumps(
                {
                    "success": False,
                    "scientific_baseline": False,
                    "error": safe_reason
                    if safe_reason in safe_codes
                    else f"feta_engineering_smoke_failed:{type(exc).__name__}",
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
