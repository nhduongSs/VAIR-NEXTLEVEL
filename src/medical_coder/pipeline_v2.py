"""Precision-first pipeline: GLiNER spans + exact-alias linking + empty assertions.

    raw text
    -> GLiNER spans (per-type thresholds, verbatim offsets)
    -> overlap resolution
    -> header/junk rejection
    -> unique-exact-alias linking against the Vietnamese ICD-10 (TT06) and RxNorm
    -> schema validation
    -> output.zip

There is no generative step, so the whole run is CPU-only and finishes in about a
minute for 100 records. That matters beyond speed: every design choice below is
one the host metric rewards directly, and none of them depend on a GPU being
available at submission time.

Assertions are always emitted empty, and that is a measured choice rather than a
gap. A wrong assertion forfeits the whole Jaccard of its concept, while an empty
prediction against an empty gold list scores 1.0; on the reference solution's
split every negation / family / history rule they tried over-fired, and
`isNegated` separated at AUC 0.497 — chance. Submission 01 emitted 266 assertion
labels and scored 20.19 on the component; the reference emitted none and scored
35.27. Restoring assertions is worth revisiting only against labelled validation
data, which is why there is no flag to half-enable it here.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .exact_link import ExactAliasIndex, PrecisionFirstLinker, strip_diacritics
from .gliner_ner import (
    DEFAULT_THRESHOLDS,
    GlinerSpanExtractor,
    ScoredSpan,
    resolve_overlaps,
    trim_generic_prefix,
)
from .models import AlignedEntity, EntityType
from .validation import validate_submission_record

# The declared budget is on parameter count, not memory: quantization does not
# change what must be reported. GLiNER 0.289B + two 4B teachers = 8.517B.
PARAMETER_BUDGET = 9_000_000_000

LOGGER = logging.getLogger(__name__)

# Bare field labels and section headings. A span is rejected only when it is
# *exactly* one of these: the same words inside a longer mention ("kết quả xét
# nghiệm glucose cao") are part of a real concept and must survive, so this is an
# equality test rather than a substring blacklist.
HEADER_LABELS = frozenset(
    {
        "thuốc", "tên thuốc",
        "triệu chứng", "các triệu chứng",
        "chẩn đoán", "chẩn đoán sơ bộ", "chẩn đoán ra viện", "icd",
        "xét nghiệm", "tên xét nghiệm",
        "kết quả", "kết quả xét nghiệm",
        "khám lâm sàng", "cận lâm sàng",
        "tiền sử", "tiền sử bệnh", "bệnh sử",
        "điều trị", "tình trạng", "diễn biến",
        "lý do", "lý do vào viện", "hỏi bệnh",
        "bệnh nhân",
    }
)
_NUMBERED_HEADING = re.compile(r"^\d+\s*[.)]\s*$")


@dataclass(frozen=True)
class PipelineV2Config:
    input_dir: Path
    output_dir: Path
    model_path: str = "urchade/gliner_multi-v2.1"
    device: str = "cpu"
    icd_kb: Path | None = None
    rxnorm_kb: Path | None = None
    thresholds: dict[EntityType, float] | None = None
    raw_floor: float = 0.02
    max_chunk_chars: int = 800
    max_candidates: int = 1
    selected_ids: frozenset[str] | None = None
    # Optional Qwen teachers. Without them the run is CPU-only; with them the
    # TRIỆU_CHỨNG -> CHẨN_ĐOÁN corrector runs, and additions need both.
    primary_teacher: str | None = None
    secondary_teacher: str | None = None
    teacher_device: str = "cuda"
    teacher_quantization: str = "4bit"
    teacher_batch_size: int = 48
    addition_margin: float = 1.0
    reject_margin: float | None = None


def is_header_span(text: str) -> bool:
    """True when the mention is a bare heading rather than a clinical concept."""
    stripped = " ".join(text.split())
    if len(stripped) < 2:
        return True
    if _NUMBERED_HEADING.match(stripped):
        return True
    normalized = stripped.rstrip(":;.-").strip().lower()
    if normalized in HEADER_LABELS:
        return True
    return strip_diacritics(normalized) in {
        strip_diacritics(label) for label in HEADER_LABELS
    }


def discover_inputs(input_dir: Path, selected_ids: frozenset[str] | None) -> list[Path]:
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


def build_linker(config: PipelineV2Config) -> PrecisionFirstLinker:
    icd_index = ExactAliasIndex.from_path(config.icd_kb) if config.icd_kb else None
    rxnorm_index = ExactAliasIndex.from_path(config.rxnorm_kb) if config.rxnorm_kb else None
    if icd_index is None:
        LOGGER.warning("No --icd-kb given; CHẨN_ĐOÁN candidates will all be empty")
    if rxnorm_index is None:
        LOGGER.warning("No --rxnorm-kb given; THUỐC candidates will all be empty")
    return PrecisionFirstLinker(
        icd_index=icd_index,
        rxnorm_index=rxnorm_index,
        max_candidates=config.max_candidates,
    )


def build_entities(
    raw_text: str,
    spans: list[ScoredSpan],
    linker: PrecisionFirstLinker,
) -> list[AlignedEntity]:
    entities: list[AlignedEntity] = []
    for span in spans:
        mention = raw_text[span.start : span.end]
        if is_header_span(mention):
            continue
        entities.append(
            AlignedEntity(
                text=mention,
                type=span.type,
                assertions=[],
                position=(span.start, span.end),
                candidates=linker.link(mention, span.type),
            )
        )
    return entities


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_selector(config: PipelineV2Config, gliner_parameters: int):
    """Load the Qwen teachers, refusing to exceed the declared parameter budget."""
    if config.primary_teacher is None:
        return None
    from .selector import SpanSelector, Teacher, TeacherConfig

    def load(path: str):
        return Teacher(
            TeacherConfig(
                model_path=path,
                device=config.teacher_device,
                quantization=config.teacher_quantization,
            )
        )

    primary = load(config.primary_teacher)
    secondary = load(config.secondary_teacher) if config.secondary_teacher else None
    selector = SpanSelector(
        primary,
        secondary,
        batch_size=config.teacher_batch_size,
        addition_margin=config.addition_margin,
        reject_margin=config.reject_margin,
    )
    total = gliner_parameters + selector.total_parameters
    if total > PARAMETER_BUDGET:
        raise RuntimeError(
            f"Declared parameters {total / 1e9:.3f}B exceed the {PARAMETER_BUDGET / 1e9:.0f}B "
            "limit. Drop the secondary teacher or use smaller ones."
        )
    LOGGER.info("declared parameters: %.3fB / %.0fB", total / 1e9, PARAMETER_BUDGET / 1e9)
    if secondary is None:
        LOGGER.info("no secondary teacher: type correction only, no span additions")
    return selector


def run_pipeline_v2(config: PipelineV2Config) -> int:
    inputs = discover_inputs(config.input_dir, config.selected_ids)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    extractor = GlinerSpanExtractor(
        config.model_path,
        thresholds=config.thresholds or DEFAULT_THRESHOLDS,
        raw_floor=config.raw_floor,
        max_chunk_chars=config.max_chunk_chars,
        device=config.device,
    )
    linker = build_linker(config)
    selector = build_selector(config, extractor.parameters)

    total = 0
    for index, input_path in enumerate(inputs, start=1):
        raw_text = input_path.read_text(encoding="utf-8")
        raw_spans = extractor.scored_spans(raw_text)
        spans = extractor.gate(raw_spans)
        if selector is not None:
            spans = selector.select(raw_text, spans, raw_spans)
        spans = resolve_overlaps(
            [trim_generic_prefix(raw_text, span) for span in spans]
        )
        entities = build_entities(raw_text, spans, linker)
        submission = [entity.to_submission_dict() for entity in entities]
        validate_submission_record(raw_text, submission)
        _atomic_write_json(config.output_dir / f"{input_path.stem}.json", submission)
        total += len(submission)
        LOGGER.info(
            "[%d/%d] %s: %d concepts", index, len(inputs), input_path.stem, len(submission)
        )
    return total
