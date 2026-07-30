"""Zero-shot span detection with GLiNER, replacing free-form LLM generation.

Why this exists
---------------
Submission 01 asked Qwen3-8B to *write out* every mention as JSON, then aligned
the generated strings back onto the raw text. That design loses on two fronts the
host metric punishes hardest:

* every mention the model paraphrases is dropped by alignment, so recall is
  capped by the model's copying fidelity;
* generated mentions carry no score, so there is no dial between precision and
  recall — and the scorer counts each spurious concept **twice** in every
  denominator.

GLiNER returns character offsets with a calibrated score per span, which gives
both an exact `text[start:end]` guarantee and a per-type threshold to tune.

Long notes are chunked on line boundaries as verbatim slices, so a chunk offset
plus the local offset is always the true global offset.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .models import EntityType

LOGGER = logging.getLogger(__name__)

# Descriptive English label strings generalise better than the Vietnamese type
# names: GLiNER was trained on English label prompts.
DEFAULT_LABEL_MAP: dict[str, EntityType] = {
    "symptom": EntityType.SYMPTOM,
    "disease or diagnosis": EntityType.DIAGNOSIS,
    "medication or drug": EntityType.MEDICATION,
    "medical test or lab name": EntityType.TEST_NAME,
    "test result or measurement value": EntityType.TEST_RESULT,
}

# Tuned on the reference solution's dev split; the host double-penalty on
# spurious spans makes precision the right side to err on.
DEFAULT_THRESHOLDS: dict[EntityType, float] = {
    EntityType.SYMPTOM: 0.20,
    EntityType.DIAGNOSIS: 0.25,
    EntityType.MEDICATION: 0.30,
    EntityType.TEST_NAME: 0.15,
    EntityType.TEST_RESULT: 0.35,
}

# A trailing token that belongs inside a THUỐC span. The Vòng 1 example shows
# drug spans carrying strength, route and frequency ("amlodipine 10 mg po daily"),
# but GLiNER usually stops at the drug name.
_DOSE_TOKEN = re.compile(
    r"^(\d+([.,\-/]\d+)*%?"
    r"|\d+\s*(mg|mcg|ug|g|ml|iu|meq|mmol)\b"
    r"|mg|mcg|µg|ug|g|ml|iu|x|%"
    r"|po|iv|im|sc|sl|pr|tab"
    r"|(q\d+h|qd|qid|qod|bid|tid|qhs|qam|qpm|prn|daily)(:prn)?"
    r")$",
    re.IGNORECASE,
)
# Vietnamese indication markers that terminate a drug span.
_INDICATION_MARKERS = ("điều trị", "cho", "để", "khi", "nếu")

# A generic lead-in that adds no clinical content ("dấu hiệu điển hình" -> "điển
# hình" is wrong, but "dấu hiệu vàng da kéo dài" -> "vàng da kéo dài" is right).
_GENERIC_PREFIX = re.compile(
    r"^(?:dấu hiệu|biểu hiện|tình trạng|hội chứng)\b[\s:;,.-]*", re.IGNORECASE
)
_WORD_TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")


def trim_generic_prefix(text: str, span: "ScoredSpan") -> "ScoredSpan":
    """Drop a generic lead-in when a complete concept (>= 2 words) remains.

    This is the one boundary lever the reference solution kept after ablation;
    the others (symptom-edge trimming, drug re-cleaning) measured worse. A
    one-word remainder is left alone because it is unstable.
    """
    match = _GENERIC_PREFIX.match(text[span.start : span.end])
    if match is None:
        return span
    new_start = span.start + match.end()
    if len(_WORD_TOKEN.findall(text[new_start : span.end])) < 2:
        return span
    return ScoredSpan(new_start, span.end, span.type, span.score)


@dataclass(frozen=True)
class ScoredSpan:
    start: int
    end: int
    type: EntityType
    score: float


def iter_chunks(text: str, max_chunk_chars: int) -> list[tuple[int, str]]:
    """Split into verbatim slices on line boundaries, keeping global offsets."""
    chunks: list[tuple[int, str]] = []
    position = 0
    length = len(text)
    while position < length:
        end = min(position + max_chunk_chars, length)
        if end < length:
            newline = text.rfind("\n", position, end)
            if newline > position:
                end = newline + 1
        chunks.append((position, text[position:end]))
        position = end
    return chunks or [(0, "")]


def extend_medication_span(text: str, start: int, end: int) -> int:
    """Absorb trailing dosage/route/frequency tokens into a medication span."""
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    remainder = text[end:line_end]
    lowered = remainder.lower()
    for marker in _INDICATION_MARKERS:
        position = lowered.find(marker)
        if position != -1:
            remainder = remainder[:position]
            break
    new_end = end
    for match in re.finditer(r"\S+", remainder):
        if _DOSE_TOKEN.match(match.group(0).strip(".,;:")):
            new_end = end + match.end()
        else:
            break
    return new_end


def resolve_overlaps(spans: list[ScoredSpan]) -> list[ScoredSpan]:
    """Keep non-overlapping spans, preferring higher score then tighter bounds."""
    ordered = sorted(spans, key=lambda span: (-span.score, span.start - span.end))
    kept: list[ScoredSpan] = []
    for span in ordered:
        if any(not (span.end <= other.start or span.start >= other.end) for other in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda span: (span.start, span.end))
    return kept


class GlinerSpanExtractor:
    """Per-type-thresholded GLiNER span extraction over raw clinical text."""

    def __init__(
        self,
        model_path: str = "urchade/gliner_multi-v2.1",
        *,
        label_map: dict[str, EntityType] | None = None,
        thresholds: dict[EntityType, float] | None = None,
        raw_floor: float = 0.02,
        max_chunk_chars: int = 800,
        device: str = "cpu",
    ) -> None:
        try:
            from gliner import GLiNER
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "GLiNER backend needs the gliner package: python -m pip install gliner"
            ) from error

        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.labels = list(self.label_map)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.raw_floor = raw_floor
        self.max_chunk_chars = max_chunk_chars
        LOGGER.info("loading GLiNER %s on %s", model_path, device)
        self.model = GLiNER.from_pretrained(model_path)
        try:
            self.model = self.model.to(device)
        except Exception:  # pragma: no cover - some versions manage device internally
            LOGGER.debug("GLiNER manages its own device placement")
        self.model.eval()
        self.parameters = sum(
            parameter.numel() for parameter in self.model.parameters()
        )

    def scored_spans(self, text: str) -> list[ScoredSpan]:
        """All spans above ``raw_floor``, before per-type gating."""
        spans: list[ScoredSpan] = []
        for chunk_start, chunk in iter_chunks(text, self.max_chunk_chars):
            if not chunk.strip():
                continue
            predictions = self.model.predict_entities(
                chunk, self.labels, threshold=self.raw_floor
            )
            for prediction in predictions:
                entity_type = self.label_map.get(prediction["label"])
                if entity_type is None:
                    continue
                start = chunk_start + prediction["start"]
                end = chunk_start + prediction["end"]
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                if entity_type is EntityType.MEDICATION:
                    end = extend_medication_span(text, start, end)
                if start < end:
                    spans.append(
                        ScoredSpan(start, end, entity_type, float(prediction.get("score", 1.0)))
                    )
        return spans

    def gate(self, spans: list[ScoredSpan]) -> list[ScoredSpan]:
        """Apply per-type thresholds, then resolve overlaps."""
        return resolve_overlaps(
            [span for span in spans if span.score >= self.thresholds.get(span.type, 0.5)]
        )

    def extract(self, text: str) -> list[ScoredSpan]:
        return self.gate(self.scored_spans(text))
