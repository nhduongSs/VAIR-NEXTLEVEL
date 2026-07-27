from __future__ import annotations

import json
import hashlib
import logging
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol

from .alignment import AlignmentIssue, align_mentions
from .local_llm import LocalMedicalExtractor
from .models import AlignedEntity, EntityType
from .validation import (
    sanitize_candidates,
    validate_output_directory,
    validate_submission_record,
)


LOGGER = logging.getLogger(__name__)


class MedicalExtractor(Protocol):
    def extract(self, record_id: str, raw_text: str): ...

    def normalize(
        self,
        record_id: str,
        raw_text: str,
        entities: list[AlignedEntity],
    ): ...


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    cache_dir: Path
    model: str
    reasoning_effort: str
    workers: int
    max_candidates: int
    overwrite: bool
    selected_ids: frozenset[str] | None
    icd_allowlist: set[str] | None
    rxnorm_allowlist: set[str] | None
    quantization: str = "4bit"
    icd_terminology: Path | None = None
    rxnorm_terminology: Path | None = None
    embedding_model: str | None = "models/multilingual-e5-small"
    embedding_device: str = "cpu"
    retrieval_top_k: int = 20
    max_input_tokens: int = 24_576
    max_new_tokens: int = 8_192
    run_signature: str | None = None


def load_allowlist(path: Path | None, uppercase: bool) -> set[str] | None:
    if path is None:
        return None
    values: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().split(",", 1)[0].strip()
        if not value or value.startswith("#"):
            continue
        values.add(value.upper() if uppercase else value)
    return values


def discover_inputs(
    input_dir: Path,
    selected_ids: frozenset[str] | None,
) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = [path for path in input_dir.glob("*.txt") if path.stem.isdigit()]
    files.sort(key=lambda path: int(path.stem))
    if selected_ids is not None:
        files = [path for path in files if path.stem in selected_ids]
        missing = selected_ids - {path.stem for path in files}
        if missing:
            raise FileNotFoundError(f"Input IDs not found: {sorted(missing, key=int)}")
    if not files:
        raise FileNotFoundError(f"No numeric .txt files found in {input_dir}")
    return files


def _apply_normalization(
    entities: list[AlignedEntity],
    mappings: Iterable,
    *,
    max_candidates: int,
    icd_allowlist: set[str] | None,
    rxnorm_allowlist: set[str] | None,
) -> None:
    seen_indices: set[int] = set()
    for mapping in mappings:
        index = mapping.entity_index
        if index in seen_indices or not (0 <= index < len(entities)):
            continue
        seen_indices.add(index)
        entity = entities[index]
        if entity.type == EntityType.DIAGNOSIS:
            allowlist = icd_allowlist
        elif entity.type == EntityType.MEDICATION:
            allowlist = rxnorm_allowlist
        else:
            continue
        entity.candidates = sanitize_candidates(
            entity.type,
            mapping.candidates,
            max_candidates=max_candidates,
            allowlist=allowlist,
        )


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_fingerprint(path: Path | None) -> str:
    if path is None:
        return "none"
    stat = path.stat()
    value = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def build_run_signature(config: PipelineConfig) -> str:
    settings = {
        "backend": "local-transformers",
        "model": config.model,
        "quantization": config.quantization,
        "icd_terminology": _file_fingerprint(config.icd_terminology),
        "rxnorm_terminology": _file_fingerprint(config.rxnorm_terminology),
        "embedding_model": config.embedding_model,
        "retrieval_top_k": config.retrieval_top_k,
        "max_candidates": config.max_candidates,
    }
    serialized = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def process_file(
    input_path: Path,
    config: PipelineConfig,
    extractor: MedicalExtractor,
) -> tuple[str, int, list[AlignmentIssue]]:
    record_id = input_path.stem
    output_path = config.output_dir / f"{record_id}.json"
    provenance_path = config.output_dir / ".provenance" / f"{record_id}.json"
    raw_text = input_path.read_text(encoding="utf-8")

    if output_path.exists() and not config.overwrite:
        if config.run_signature is not None:
            if not provenance_path.exists():
                raise RuntimeError(
                    f"{output_path} has no local-run provenance. Use --overwrite "
                    "to replace outputs created by an older/API pipeline."
                )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("run_signature") != config.run_signature:
                raise RuntimeError(
                    f"{output_path} was created with a different model/KB. "
                    "Use --overwrite to prevent a mixed submission."
                )
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        validate_submission_record(raw_text, existing)
        LOGGER.info("%s: valid output exists; skipping", record_id)
        return record_id, len(existing), []

    extraction = extractor.extract(record_id, raw_text)
    entities, issues = align_mentions(raw_text, extraction.entities)
    normalization = extractor.normalize(record_id, raw_text, entities)
    _apply_normalization(
        entities,
        normalization.mappings,
        max_candidates=config.max_candidates,
        icd_allowlist=config.icd_allowlist,
        rxnorm_allowlist=config.rxnorm_allowlist,
    )

    submission = [entity.to_submission_dict() for entity in entities]
    validate_submission_record(raw_text, submission)
    _atomic_write_json(output_path, submission)
    if config.run_signature is not None:
        _atomic_write_json(
            provenance_path,
            {
                "record_id": record_id,
                "run_signature": config.run_signature,
                "model": config.model,
            },
        )

    if issues:
        issue_path = config.cache_dir / f"{record_id}.alignment_issues.json"
        _atomic_write_json(
            issue_path,
            [
                {
                    "text": issue.text,
                    "type": issue.type,
                    "start_hint": issue.start_hint,
                    "reason": issue.reason,
                }
                for issue in issues
            ],
        )
    return record_id, len(submission), issues


def run_pipeline(config: PipelineConfig) -> None:
    inputs = discover_inputs(config.input_dir, config.selected_ids)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if config.workers != 1:
        raise ValueError(
            "Local GPU inference is intentionally sequential; set --workers 1. "
            "Parallel workers would duplicate model VRAM."
        )
    if config.icd_terminology is None or config.rxnorm_terminology is None:
        LOGGER.warning(
            "A terminology file is missing. Affected diagnosis/medication "
            "candidates will be empty; provide both --icd-kb and --rxnorm-kb."
        )

    if config.run_signature is None:
        config = replace(config, run_signature=build_run_signature(config))

    extractor = LocalMedicalExtractor(
        model_path=config.model,
        quantization=config.quantization,
        cache_dir=config.cache_dir,
        icd_terminology=config.icd_terminology,
        rxnorm_terminology=config.rxnorm_terminology,
        embedding_model=config.embedding_model,
        embedding_device=config.embedding_device,
        retrieval_top_k=config.retrieval_top_k,
        max_candidates=config.max_candidates,
        max_input_tokens=config.max_input_tokens,
        max_new_tokens=config.max_new_tokens,
    )

    completed = 0
    for input_path in inputs:
        record_id, count, issues = process_file(input_path, config, extractor)
        completed += 1
        LOGGER.info(
            "[%d/%d] %s: %d entities, %d dropped mentions",
            completed,
            len(inputs),
            record_id,
            count,
            len(issues),
        )


def create_submission_zip(output_dir: Path, zip_path: Path) -> None:
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
