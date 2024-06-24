"""Verification and negative controls audit CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from nocturne.verification.verifier import SystemVerifier, SystemVerificationReport


def run_verify_command(
    output_path: Optional[str] = None,
    controls_only: bool = False,
    verbose: bool = True,
) -> int:
    """Execute complete system verification and negative controls suite."""
    if verbose:
        print("=== Nocturne Verification & Negative Controls Audit ===")

    report: SystemVerificationReport = SystemVerifier.run_all()

    if verbose:
        print(f"\nPlatform: {report.platform_info['system']} {report.platform_info['machine']} | Python {report.platform_info['python_version']}")
        print(f"Timestamp: {report.timestamp_utc}\n")

        if not controls_only:
            print("System Subsystem Integrity Checks:")
            for chk in report.checks:
                status_str = "PASS" if chk["passed"] else "FAIL"
                print(f"  [{status_str}] {chk['check_id']:8s} {chk['name']:35s} ({chk['duration_s'] * 1000:6.1f} ms)")

        print("\nNegative Control Invariant Gates (NC01 - NC12):")
        for ctrl in report.negative_controls:
            status_str = "PASS" if ctrl["caught"] else "FAIL"
            print(f"  [{status_str}] {ctrl['control_id']:6s} {ctrl['name']:40s} caught: {ctrl['exception_type']}")


        print("-" * 65)
        print(f"Summary: {report.passed_checks}/{report.total_checks} passed in {report.total_duration_s:.2f}s | Status: {'ALL CHECKS PASSED' if report.all_passed else 'VERIFICATION FAILED'}")

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        if verbose:
            print(f"Report saved to: {p}")

    return 0 if report.all_passed else 1


def add_verify_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the verify subcommand."""
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write verification report JSON",
    )
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Only display negative controls (NC01-NC12)",
    )
