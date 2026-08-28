"""Operator entry point for the frozen VCC 2026 baseline preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from auto_researcher.benchmarks.vcc2026.models import dump_report, load_baseline_fixture
from auto_researcher.benchmarks.vcc2026.preflight import run_baseline_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("archival", "campaign"),
        default="campaign",
        help="archival verifies the known submitted state; campaign requires every gate",
    )
    args = parser.parse_args()

    fixture = load_baseline_fixture(args.manifest)
    report = run_baseline_preflight(args.checkout, fixture)
    print(dump_report(report), end="")
    if args.mode == "archival" and report.frozen_source_verified:
        print("VCC2026_FROZEN_BASELINE_VERIFIED")
        return 0
    if args.mode == "campaign" and report.campaign_ready:
        print("VCC2026_CAMPAIGN_PREFLIGHT_PASS")
        return 0
    print("PRE-RUN BLOCKED")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
