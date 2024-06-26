"""Acceptance bundle generation and cryptographic manifest verification CLI subcommand."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

from nocturne.verification.acceptance import (
    AcceptanceBundleGenerator,
    AcceptanceBundleReport,
)


def verify_bundle_integrity(bundle_dir: str | Path, verbose: bool = True) -> bool:
    """Verify that all files listed in ACCEPTANCE_MANIFEST.json match their cryptographic hashes."""
    bpath = Path(bundle_dir)
    manifest_file = bpath / "ACCEPTANCE_MANIFEST.json"
    if not manifest_file.exists():
        if verbose:
            print(f"Error: Manifest not found at {manifest_file}")
        return False

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_matched = True
    if verbose:
        print(f"=== Verifying Acceptance Manifest at {bpath} ===")

    for fname, meta in manifest.items():
        target = bpath / fname
        if not target.exists():
            if verbose:
                print(f"  [MISSING] {fname}")
            all_matched = False
            continue

        with open(target, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()

        if actual_sha == meta["sha256"]:
            if verbose:
                print(f"  [VALID]   {fname:30s} {actual_sha[:16]}... ({meta['size_bytes']} bytes)")
        else:
            if verbose:
                print(f"  [CORRUPT] {fname:30s} Expected: {meta['sha256'][:16]}, Got: {actual_sha[:16]}")
            all_matched = False

    return all_matched


def run_acceptance_command(
    output_dir: str = "artifacts/acceptance_bundle",
    verify_only: bool = False,
    verbose: bool = True,
) -> int:
    """Build or verify cryptographic acceptance bundle."""
    if verify_only:
        valid = verify_bundle_integrity(output_dir, verbose=verbose)
        if verbose:
            print(f"\nIntegrity Status: {'VALID' if valid else 'INTEGRITY CHECK FAILED'}")
        return 0 if valid else 1

    if verbose:
        print(f"=== Building Acceptance Bundle in '{output_dir}' ===")

    gen = AcceptanceBundleGenerator(output_dir=output_dir)
    report: AcceptanceBundleReport = gen.build_bundle()

    if verbose:
        print(f"\nManifest SHA-256: {report.manifest_sha256}")
        print(f"Files Emitted ({len(report.files_emitted)}):")
        out_path = Path(output_dir)
        for f in report.files_emitted:
            p = out_path / f
            size_b = p.stat().st_size if p.exists() else 0
            print(f"  - {p.name:32s} ({size_b:7d} bytes)")

        print(f"\nStatus: {'ALL ACCEPTANCE CHECKS PASSED' if report.all_checks_passed else 'ACCEPTANCE FAILED'}")
        print(f"Duration: {report.total_duration_s:.2f}s")

    return 0 if report.all_checks_passed else 1


def add_acceptance_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the acceptance subcommand."""
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="artifacts/acceptance_bundle",
        help="Target directory for acceptance bundle artifacts",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify cryptographic checksums of an existing acceptance bundle without regenerating",
    )
