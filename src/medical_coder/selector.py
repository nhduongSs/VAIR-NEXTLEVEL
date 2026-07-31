"""Qwen teachers that re-type and extend GLiNER spans using next-token logits.

Two roles, both decided by a *single forward pass* over a fixed label set — never
by generation. That matters: a generated answer can be anything, while a logit
over the digits 0-5 is a closed decision that cannot corrupt the schema.

1. **Corrector** — for every baseline ``TRIỆU_CHỨNG`` span, read five-way logits
   and re-type it to ``CHẨN_ĐOÁN`` when the teacher disagrees. This is GLiNER's
   single largest type error: it reads chronic diseases as symptoms. Our
   CPU-only run produced 529 diagnoses against the reference solution's 668, and
   this step is what closes that gap.

2. **Additions** — for low-scoring raw spans that do not overlap a baseline span,
   read six-way logits (0 = not a concept) in *both* teachers and add the span
   only when they agree on the same type, that type carries no candidates, and
   both clear a margin over ``NONE``.

Additions are restricted to ``TRIỆU_CHỨNG`` / ``TÊN_XÉT_NGHIỆM`` /
``KẾT_QUẢ_XÉT_NGHIỆM`` on purpose. A spurious diagnosis or drug is charged to the
candidate denominator as well (``2 * (n_codes + 1)``), so recall bought in those
two types is paid for twice.

Consensus is required for additions but not for correction: correction moves a
span that already exists, while an addition creates one, and only the second can
manufacture a spurious concept.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from .gliner_ner import ScoredSpan
from .models import EntityType

LOGGER = logging.getLogger(__name__)

# Digit order is fixed: the prompts below number the types 1-5 in this sequence.
ORDERED_TYPES: tuple[EntityType, ...] = (
    EntityType.SYMPTOM,
    EntityType.DIAGNOSIS,
    EntityType.MEDICATION,
    EntityType.TEST_NAME,
    EntityType.TEST_RESULT,
)

# Types safe to add: none of them carry candidates.
DEFAULT_ADDITION_TYPES = frozenset(
    {EntityType.SYMPTOM, EntityType.TEST_NAME, EntityType.TEST_RESULT}
)

FIVE_WAY_SYSTEM = (
    "Bạn là chuyên gia y lâm sàng Việt Nam. Phân loại ý niệm y khoa vào ĐÚNG MỘT loại:\n"
    "1 = TRIỆU_CHỨNG (triệu chứng: đau đầu, sốt, phù, mệt mỏi)\n"
    "2 = CHẨN_ĐOÁN (chẩn đoán bệnh: đái tháo đường, tăng huyết áp, viêm phổi, ung thư)\n"
    "3 = THUỐC (paracetamol, ceftriaxone, amlodipine)\n"
    "4 = TÊN_XÉT_NGHIỆM (công thức máu, MRI, nội soi, X-quang)\n"
    "5 = KẾT_QUẢ_XÉT_NGHIỆM (giá trị số: 120 mg/dL, HbA1c 7.2%, GCS 15)\n"
    "Quy tắc: 'tăng huyết áp/đái tháo đường/xơ gan' = 2; 'đau/phù/sốt/mệt' = 1."
)

SIX_WAY_SYSTEM = (
    "Bạn là chuyên gia y lâm sàng Việt Nam. Phân loại ý niệm y khoa:\n"
    "0 = KHÔNG PHẢI ý niệm y khoa hợp lệ (tiêu đề, từ chung chung, số/liều rời, hành chính)\n"
    "1 = TRIỆU_CHỨNG (triệu chứng: đau đầu, sốt, phù)\n"
    "2 = CHẨN_ĐOÁN (bệnh: đái tháo đường, tăng huyết áp, viêm phổi)\n"
    "3 = THUỐC (paracetamol, ceftriaxone)\n"
    "4 = TÊN_XÉT_NGHIỆM (công thức máu, MRI, nội soi)\n"
    "5 = KẾT_QUẢ_XÉT_NGHIỆM (giá trị số: 120 mg/dL, GCS 15)\n"
    "Quy tắc: 'tăng huyết áp/đái tháo đường'=2; 'đau/phù/sốt'=1; tiêu đề=0."
)


def line_context(text: str, start: int, end: int, line_limit: int = 300) -> str:
    """The mention's line, prefixed by the nearest non-empty line as a section hint."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end][:line_limit]

    previous = ""
    cursor = line_start - 1
    while cursor > 0:
        candidate_start = text.rfind("\n", 0, cursor) + 1
        candidate_end = text.find("\n", candidate_start)
        if candidate_end == -1:
            candidate_end = len(text)
        candidate = text[candidate_start:candidate_end].strip()
        if candidate:
            previous = candidate[:80]
            break
        cursor = candidate_start - 1
    return f"[mục: {previous}] {line}" if previous else line


def single_token_ids(tokenizer, surfaces: Iterable[str]) -> list[int]:
    """Token ids for surfaces that encode to exactly one token."""
    ids = []
    for surface in surfaces:
        encoded = tokenizer.encode(surface, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return ids


def digit_token_groups(tokenizer, digits: Iterable[int]) -> list[list[int]]:
    """One id group per digit, covering the bare and space-prefixed forms."""
    return [single_token_ids(tokenizer, (str(d), f" {d}")) for d in digits]


@dataclass
class TeacherConfig:
    model_path: str
    device: str = "cuda"
    quantization: str = "4bit"
    dtype: str = "bfloat16"


class Teacher:
    """A causal LM used only to score a fixed set of next tokens."""

    def __init__(self, config: TeacherConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(config.dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if config.quantization == "none":
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path, torch_dtype=torch_dtype
            ).to(config.device)
        else:
            from transformers import BitsAndBytesConfig

            if config.quantization == "4bit":
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif config.quantization == "8bit":
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"Unknown quantization mode: {config.quantization!r}")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                quantization_config=bnb,
                device_map={"": config.device},
            )
        self.model.eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.entity_digits = digit_token_groups(self.tokenizer, range(1, 6))
        self.none_digits = single_token_ids(self.tokenizer, ("0", " 0"))
        LOGGER.info(
            "loaded teacher %s (%.3fB params) on %s",
            config.model_path,
            self.parameters / 1e9,
            config.device,
        )

    def chat_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # older templates have no enable_thinking argument
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def group_scores(
        self,
        prompts: Sequence[str],
        groups: Sequence[Sequence[int]],
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]:
        """Log-sum-exp of each token group, from one next-token forward per batch."""
        import torch

        device = next(self.model.parameters()).device
        results: list[list[float]] = []
        for offset in range(0, len(prompts), batch_size):
            batch = list(prompts[offset : offset + batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length + 8,
            ).to(device)
            with torch.inference_mode():
                try:
                    logits = self.model(**encoded, logits_to_keep=1).logits[:, -1, :]
                except TypeError:  # transformers without logits_to_keep
                    logits = self.model(**encoded).logits[:, -1, :]
            for row in logits:
                results.append(
                    [
                        torch.logsumexp(row[list(ids)], -1).item() if ids else -1e9
                        for ids in groups
                    ]
                )
        return results


class SpanSelector:
    """Type correction, plus consensus additions when a second teacher is present."""

    def __init__(
        self,
        primary: Teacher,
        secondary: Teacher | None = None,
        *,
        batch_size: int = 48,
        max_length: int = 384,
        addition_margin: float = 1.0,
        addition_types: frozenset[EntityType] = DEFAULT_ADDITION_TYPES,
        reject_margin: float | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.batch_size = batch_size
        self.max_length = max_length
        self.addition_margin = addition_margin
        self.addition_types = addition_types
        # None disables rejection. A positive value is how far NONE must beat the
        # best entity label before a span is dropped, so larger is more cautious.
        self.reject_margin = reject_margin
        # Every margin the rejector has seen, so one run yields the whole
        # threshold curve instead of a single operating point. Picking the next
        # margin from this distribution beats guessing and re-submitting.
        self.margins_seen: list[float] = []

    @property
    def total_parameters(self) -> int:
        total = self.primary.parameters
        if self.secondary is not None:
            total += self.secondary.parameters
        return total

    def _five_way(self, teacher: Teacher, prompts: Sequence[str]) -> list[EntityType]:
        scores = teacher.group_scores(
            prompts, teacher.entity_digits, self.batch_size, self.max_length
        )
        return [ORDERED_TYPES[max(range(5), key=row.__getitem__)] for row in scores]

    def _six_way(
        self, teacher: Teacher, prompts: Sequence[str]
    ) -> list[tuple[EntityType, float]]:
        groups = list(teacher.entity_digits) + [teacher.none_digits]
        scores = teacher.group_scores(prompts, groups, self.batch_size, self.max_length)
        output = []
        for row in scores:
            entity_scores, none_score = row[:5], row[5]
            best = max(range(5), key=entity_scores.__getitem__)
            output.append((ORDERED_TYPES[best], entity_scores[best] - none_score))
        return output

    def correct_types(self, text: str, spans: list[ScoredSpan]) -> list[ScoredSpan]:
        """Re-type TRIỆU_CHỨNG spans the primary teacher reads as CHẨN_ĐOÁN."""
        indices = [i for i, span in enumerate(spans) if span.type is EntityType.SYMPTOM]
        if not indices:
            return list(spans)
        prompts = [
            self.primary.chat_prompt(
                FIVE_WAY_SYSTEM,
                f"Ngữ cảnh: {line_context(text, spans[i].start, spans[i].end)}\n"
                f'Ý niệm: "{text[spans[i].start : spans[i].end]}"\n'
                "Trả lời CHỈ MỘT chữ số 1-5.",
            )
            for i in indices
        ]
        predictions = self._five_way(self.primary, prompts)
        corrected = list(spans)
        for position, prediction in zip(indices, predictions):
            if prediction is EntityType.DIAGNOSIS:
                span = spans[position]
                corrected[position] = ScoredSpan(
                    span.start, span.end, EntityType.DIAGNOSIS, span.score
                )
        return corrected

    def reject_spans(
        self, text: str, spans: list[ScoredSpan], margin: float
    ) -> list[ScoredSpan]:
        """Drop baseline spans the teacher reads as "not a medical concept".

        Fixes an asymmetry that had no justification: adding a span required two
        teachers to agree, while keeping one required nothing beyond GLiNER's
        threshold. The six-way prompt already has a `0 = not a concept` option;
        this simply asks it about spans we were going to emit anyway.

        The arithmetic is favourable. Dropping a spurious span removes 2 units
        from every denominator; dropping a correct one only costs `alpha` from
        the numerator, because the ground-truth concept stays in the denominator
        as a miss either way. Rejection therefore pays off whenever it is right
        more than ``1 / (1 + 2 * text_score / alpha)`` of the time — about 57% at
        our current operating point, against a pipeline precision near 55%.
        """
        if not spans:
            return []
        prompts = [
            self.primary.chat_prompt(
                SIX_WAY_SYSTEM,
                f"Ngữ cảnh: {line_context(text, span.start, span.end)}\n"
                f'Ý niệm: "{text[span.start : span.end]}"\n'
                "Trả lời CHỈ MỘT chữ số 0-5.",
            )
            for span in spans
        ]
        votes = self._six_way(self.primary, prompts)
        if self.secondary is not None:
            # With a second opinion available, require both to call it junk.
            secondary_votes = self._six_way(self.secondary, prompts)
            votes = [
                (t, max(p, s))
                for (t, p), (_, s) in zip(votes, secondary_votes)
            ]
        self.margins_seen.extend(margin_value for _, margin_value in votes)
        kept = [span for span, (_, m) in zip(spans, votes) if m > -margin]
        LOGGER.info("rejector: dropped %d of %d spans", len(spans) - len(kept), len(spans))
        return kept

    def rejection_report(self) -> str:
        """How many spans each candidate margin would drop, over the whole corpus."""
        if not self.margins_seen:
            return "rejector: chưa chạy"
        total = len(self.margins_seen)
        ordered = sorted(self.margins_seen)
        lines = [f"rejector: đã chấm {total} span", "  margin  bỏ đi   tỉ lệ"]
        for candidate in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            dropped = sum(1 for value in ordered if value <= -candidate)
            lines.append(f"  {candidate:6.1f} {dropped:6d} {dropped / total:7.1%}")
        return "\n".join(lines)

    def propose_additions(
        self,
        text: str,
        baseline: list[ScoredSpan],
        raw: list[ScoredSpan],
    ) -> list[ScoredSpan]:
        """Add non-overlapping raw spans both teachers agree on."""
        if self.secondary is None:
            return []
        occupied = [(span.start, span.end) for span in baseline]

        def overlaps(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in occupied)

        candidates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for span in raw:
            key = (span.start, span.end)
            if key in seen or overlaps(*key):
                continue
            if len(text[span.start : span.end].strip()) < 2:
                continue
            seen.add(key)
            candidates.append(key)
        if not candidates:
            return []

        prompts = [
            self.primary.chat_prompt(
                SIX_WAY_SYSTEM,
                f"Ngữ cảnh: {line_context(text, start, end)}\n"
                f'Ý niệm: "{text[start:end]}"\n'
                "Trả lời CHỈ MỘT chữ số 0-5.",
            )
            for start, end in candidates
        ]
        primary_votes = self._six_way(self.primary, prompts)
        secondary_votes = self._six_way(self.secondary, prompts)

        additions: list[ScoredSpan] = []
        for (start, end), (p_type, p_margin), (s_type, s_margin) in zip(
            candidates, primary_votes, secondary_votes
        ):
            if p_type is not s_type or p_type not in self.addition_types:
                continue
            if p_margin < self.addition_margin or s_margin < self.addition_margin:
                continue
            additions.append(ScoredSpan(start, end, p_type, 0.0))
        return additions

    def select(
        self,
        text: str,
        baseline: list[ScoredSpan],
        raw: list[ScoredSpan],
    ) -> list[ScoredSpan]:
        # Reject before correcting: no point asking the corrector to re-type a
        # span that is about to be dropped.
        if self.reject_margin is not None:
            baseline = self.reject_spans(text, baseline, self.reject_margin)
        corrected = self.correct_types(text, baseline)
        corrected.extend(self.propose_additions(text, corrected, raw))
        corrected.sort(key=lambda span: (span.start, span.end))
        return corrected
