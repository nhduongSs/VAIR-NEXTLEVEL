"""Packaging and final checks for a submission.

Kept separate from :mod:`medical_coder.pipeline` on purpose: these helpers are
generic, but `pipeline` pulls in the whole generative LLM backend, so importing
them from there would drag ~43 KB of unrelated code into any consumer — notably
the predict-v2 notebook, which uses none of it.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from .validation import validate_output_directory


def create_submission_zip(output_dir: Path, zip_path: Path) -> None:
    """Write `output/<id>.json` members, then read the archive back to verify."""
    output_files = sorted(
        (path for path in output_dir.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not output_files:
        raise FileNotFoundError(f"No JSON files found in {output_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_files:
            archive.write(path, arcname=f"output/{path.name}")
    temporary.replace(zip_path)

    expected = [f"output/{path.name}" for path in output_files]
    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.namelist() != expected:
            raise RuntimeError(f"ZIP verification failed: {zip_path}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")


def validate_all(input_dir: Path, output_dir: Path) -> None:
    errors = validate_output_directory(input_dir, output_dir)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"Output validation failed:\n{formatted}")
