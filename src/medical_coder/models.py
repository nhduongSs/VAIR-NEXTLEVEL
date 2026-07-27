from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntityType(str, Enum):
    SYMPTOM = "TRIỆU_CHỨNG"
    TEST_NAME = "TÊN_XÉT_NGHIỆM"
    TEST_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
    DIAGNOSIS = "CHẨN_ĐOÁN"
    MEDICATION = "THUỐC"


class AssertionType(str, Enum):
    NEGATED = "isNegated"
    FAMILY = "isFamily"
    HISTORICAL = "isHistorical"


class ExtractedMention(StrictModel):
    """Mention returned by the LLM before deterministic offset alignment."""

    text: str = Field(min_length=1)
    type: EntityType
    assertions: list[AssertionType]
    start_hint: int = Field(
        ge=0,
        description="Estimated zero-based start position in the original text.",
    )


class ExtractionResponse(StrictModel):
    entities: list[ExtractedMention]


class CandidateRequest(StrictModel):
    entity_index: int = Field(ge=0)
    type: EntityType
    text: str = Field(min_length=1)
    context: str


class CandidatePrediction(StrictModel):
    entity_index: int = Field(ge=0)
    candidates: list[str]


class NormalizationResponse(StrictModel):
    mappings: list[CandidatePrediction]


class AlignedEntity(StrictModel):
    text: str
    type: EntityType
    assertions: list[AssertionType]
    position: tuple[int, int]
    candidates: list[str] = Field(default_factory=list)

    def to_submission_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "text": self.text,
            "type": self.type.value,
        }
        if self.type in {EntityType.DIAGNOSIS, EntityType.MEDICATION}:
            result["candidates"] = self.candidates
        result["assertions"] = [item.value for item in self.assertions]
        result["position"] = [self.position[0], self.position[1]]
        return result

