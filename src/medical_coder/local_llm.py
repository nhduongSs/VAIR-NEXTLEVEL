from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import (
    AlignedEntity,
    AssertionType,
    CandidatePrediction,
    EntityType,
    ExtractionResponse,
    NormalizationResponse,
)
from .prompts import (
    EXTRACTION_INSTRUCTIONS,
    EXTRACTION_PROMPT_VERSION,
    NORMALIZATION_INSTRUCTIONS,
    NORMALIZATION_PROMPT_VERSION,
)
from .terminology import RetrievedCandidate, TerminologyIndex


LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def _cache_key(
    *,
    stage: str,
    model: str,
    prompt_version: str,
    payload: str,
) -> str:
    value = "\n".join((stage, model, prompt_version, payload))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _clean_model_response(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()
    return cleaned


def extract_json_value(value: str) -> dict[str, Any] | list[Any]:
    """Read the first valid JSON object or array from a local model response."""

    cleaned = _clean_model_response(value)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            result, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, (dict, list)):
            return result
    raise ValueError("Local model response does not contain a valid JSON object or array")


def extract_json_object(value: str) -> dict[str, Any]:
    """Backward-compatible helper that requires the decoded value to be an object."""

    result = extract_json_value(value)
    if isinstance(result, dict):
        return result
    raise ValueError("Local model response contains a JSON array, not an object")


def _has_complete_top_level_json(value: str) -> bool:
    """Return True only when the first top-level JSON container is complete."""

    cleaned = _clean_model_response(value)
    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
    if not starts:
        return False
    try:
        result, _ = json.JSONDecoder().raw_decode(cleaned[min(starts) :])
    except json.JSONDecodeError:
        return False
    return isinstance(result, (dict, list))


def _truthy_assertion_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y", "có", "co"}
    return False


def _coerce_assertions(value: Any) -> list[str]:
    allowed = {item.value for item in AssertionType}
    result: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            if item in allowed and item not in result:
                result.append(item)
            return
        if isinstance(item, dict):
            for key, enabled in item.items():
                if key in allowed and _truthy_assertion_flag(enabled) and key not in result:
                    result.append(key)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return result


def _coerce_response_payload(
    value: dict[str, Any] | list[Any],
    response_type: type[T],
) -> dict[str, Any]:
    """Normalize common Qwen JSON variants before strict Pydantic validation."""

    if response_type is ExtractionResponse:
        if isinstance(value, list):
            rows: Any = value
        elif "entities" in value:
            rows = value["entities"]
        elif "entity" in value:
            rows = value["entity"]
        elif "text" in value and "type" in value:
            rows = [value]
        else:
            rows = []
        if isinstance(rows, dict):
            rows = [rows]

        entities: list[dict[str, Any]] = []
        allowed_entity_types = {item.value for item in EntityType}
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or not row.get("text") or not row.get("type"):
                    continue
                entity_type = str(row["type"]).strip()
                if entity_type not in allowed_entity_types:
                    LOGGER.debug(
                        "Dropping entity with unsupported type %r: %r",
                        entity_type,
                        row.get("text"),
                    )
                    continue
                try:
                    start_hint = max(0, int(row.get("start_hint", 0)))
                except (TypeError, ValueError):
                    start_hint = 0
                entities.append(
                    {
                        "text": row["text"],
                        "type": entity_type,
                        "assertions": _coerce_assertions(row.get("assertions", [])),
                        "start_hint": start_hint,
                    }
                )
        return {"entities": entities}

    if response_type is NormalizationResponse:
        if isinstance(value, list):
            rows = value
        elif "mappings" in value:
            rows = value["mappings"]
        elif "mapping" in value:
            rows = value["mapping"]
        elif "entity_index" in value:
            rows = [value]
        else:
            rows = []
        if isinstance(rows, dict):
            rows = [rows]

        mappings: list[dict[str, Any]] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or "entity_index" not in row:
                    continue
                candidates = row.get("candidates", [])
                if isinstance(candidates, (str, int)):
                    candidates = [str(candidates)]
                if not isinstance(candidates, list):
                    candidates = []
                mappings.append(
                    {
                        "entity_index": row["entity_index"],
                        "candidates": [str(code) for code in candidates if code is not None],
                    }
                )
        return {"mappings": mappings}

    if isinstance(value, dict):
        return value
    raise ValueError(f"Cannot coerce JSON array to {response_type.__name__}")


class LocalTransformersGenerator:
    """Lazy local Hugging Face generator; it never makes a network API call."""

    def __init__(
        self,
        *,
        model_path: str,
        quantization: str = "4bit",
        device_map: str = "auto",
        max_input_tokens: int = 24_576,
        max_new_tokens: int = 8_192,
        seed: int = 17,
    ) -> None:
        if quantization not in {"4bit", "8bit", "none"}:
            raise ValueError("quantization must be one of: 4bit, 8bit, none")
        self.model_path = model_path
        self.quantization = quantization
        self.device_map = device_map
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Local inference dependencies are missing. Install with: "
                "python -m pip install -e '.[local]'"
            ) from exc

        if self.quantization != "none" and not torch.cuda.is_available():
            raise RuntimeError(
                f"{self.quantization} inference requires a CUDA GPU. "
                "Use Kaggle GPU mode or --quantization none for a small CPU model."
            )

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        if self.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        LOGGER.info(
            "Loading local model %s (%s); no external inference API is used",
            self.model_path,
            self.quantization,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            **model_kwargs,
        )
        self._model.eval()

    def generate(self, *, instructions: str, input_text: str) -> str:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_text},
        ]
        response_prefix = ""
        if '{"entities":[' in instructions:
            response_prefix = '{"entities":['
        elif '{"mappings":[' in instructions:
            response_prefix = '{"mappings":['

        try:
            prefilled_messages = messages + [{"role": "assistant", "content": response_prefix}]
            prompt = self._tokenizer.apply_chat_template(
                prefilled_messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                prompt = self._tokenizer.apply_chat_template(
                    prefilled_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            except TypeError:
                response_prefix = ""
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        token_count = int(inputs["input_ids"].shape[-1])
        if token_count > self.max_input_tokens:
            raise RuntimeError(
                f"Prompt has {token_count} tokens, above --max-input-tokens "
                f"{self.max_input_tokens}. Split the record instead of truncating it."
            )

        model_device = next(self._model.parameters()).device
        inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

        tokenizer = self._tokenizer
        prompt_length = token_count

        class CompleteJsonStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                finished = []
                for row in input_ids:
                    generated_text = tokenizer.decode(
                        row[prompt_length:],
                        skip_special_tokens=True,
                    )
                    finished.append(
                        _has_complete_top_level_json(response_prefix + generated_text)
                    )
                return torch.tensor(finished, dtype=torch.bool, device=input_ids.device)

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            use_cache=True,
        )
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=StoppingCriteriaList([CompleteJsonStoppingCriteria()]),
            )
        response_ids = generated[0, token_count:]
        generated_text = self._tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()
        return response_prefix + generated_text


class LocalMedicalExtractor:
    def __init__(
        self,
        *,
        model_path: str,
        quantization: str,
        cache_dir: Path,
        icd_terminology: Path | None,
        rxnorm_terminology: Path | None,
        embedding_model: str | None,
        embedding_device: str,
        retrieval_top_k: int,
        max_candidates: int,
        max_input_tokens: int,
        max_new_tokens: int,
        generator: LocalTransformersGenerator | None = None,
    ) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_top_k = retrieval_top_k
        self.max_candidates = max_candidates
        self.generator = generator or LocalTransformersGenerator(
            model_path=model_path,
            quantization=quantization,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
        embedding_cache_dir = cache_dir / "terminology_embeddings"
        self.indexes: dict[EntityType, TerminologyIndex] = {}
        if icd_terminology is not None:
            self.indexes[EntityType.DIAGNOSIS] = TerminologyIndex(
                icd_terminology,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
                embedding_cache_dir=embedding_cache_dir,
            )
        if rxnorm_terminology is not None:
            self.indexes[EntityType.MEDICATION] = TerminologyIndex(
                rxnorm_terminology,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
                embedding_cache_dir=embedding_cache_dir,
            )

    def _load_cache(self, path: Path, response_type: type[T]) -> T | None:
        if not path.exists():
            return None
        try:
            return response_type.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Ignoring invalid cache %s: %s", path, exc)
            return None

    @staticmethod
    def _write_cache(path: Path, result: BaseModel) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def _parse(
        self,
        *,
        instructions: str,
        input_text: str,
        response_type: type[T],
    ) -> T:
        last_error: Exception | None = None
        retry_suffix = ""
        for attempt in range(2):
            output = self.generator.generate(
                instructions=instructions,
                input_text=input_text + retry_suffix,
            )
            try:
                payload = _coerce_response_payload(
                    extract_json_value(output),
                    response_type,
                )
                return response_type.model_validate(payload)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                preview = output[:1000].replace("\n", "\\n")
                LOGGER.warning(
                    "Invalid local JSON on attempt %d: %s; output preview=%s",
                    attempt + 1,
                    exc,
                    preview,
                )
                retry_suffix = (
                    "\n\nLần trả lời trước không hợp lệ. Hãy tạo lại từ đầu và chỉ "
                    "trả đúng một JSON object theo schema, không có văn bản khác."
                )
        raise RuntimeError(f"Local model failed to produce valid JSON: {last_error}")

    def extract(self, record_id: str, raw_text: str) -> ExtractionResponse:
        payload = f"<clinical_text>\n{raw_text}\n</clinical_text>"
        cache_key = _cache_key(
            stage="extraction",
            model=self.model_path,
            prompt_version=EXTRACTION_PROMPT_VERSION,
            payload=payload,
        )
        cache_path = self.cache_dir / f"{record_id}.extraction.{cache_key}.json"
        cached = self._load_cache(cache_path, ExtractionResponse)
        if cached is not None:
            LOGGER.info("%s: using extraction cache", record_id)
            return cached

        result = self._parse(
            instructions=EXTRACTION_INSTRUCTIONS,
            input_text=payload,
            response_type=ExtractionResponse,
        )
        self._write_cache(cache_path, result)
        return result

    @staticmethod
    def _candidate_payload(candidate: RetrievedCandidate) -> dict[str, Any]:
        return {
            "code": candidate.code,
            "label": candidate.label,
            "retrieval_score": round(candidate.score, 5),
            "exact_alias": candidate.exact,
        }

    def normalize(
        self,
        record_id: str,
        raw_text: str,
        entities: list[AlignedEntity],
    ) -> NormalizationResponse:
        request_rows: list[dict[str, Any]] = []
        allowed_codes: dict[int, set[str]] = {}
        retrieval_fallback: dict[int, list[RetrievedCandidate]] = {}

        for index, entity in enumerate(entities):
            terminology = self.indexes.get(entity.type)
            if terminology is None:
                continue
            start, end = entity.position
            context_start = max(0, start - 240)
            context_end = min(len(raw_text), end + 240)
            context = raw_text[context_start:context_end]
            retrieved = terminology.retrieve(
                entity.text,
                context=context,
                top_k=self.retrieval_top_k,
            )
            retrieval_fallback[index] = retrieved
            allowed_codes[index] = {candidate.code for candidate in retrieved}
            request_rows.append(
                {
                    "entity_index": index,
                    "type": entity.type.value,
                    "text": entity.text,
                    "context": context,
                    "choices": [self._candidate_payload(item) for item in retrieved],
                }
            )

        if not request_rows:
            return NormalizationResponse(mappings=[])

        serialized = json.dumps(
            {"entities": request_rows},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload = f"<normalization_input>\n{serialized}\n</normalization_input>"
        cache_key = _cache_key(
            stage="normalization",
            model=self.model_path,
            prompt_version=NORMALIZATION_PROMPT_VERSION,
            payload=payload,
        )
        cache_path = self.cache_dir / f"{record_id}.normalization.{cache_key}.json"
        cached = self._load_cache(cache_path, NormalizationResponse)
        if cached is not None:
            LOGGER.info("%s: using normalization cache", record_id)
            return cached

        raw_mappings: list[CandidatePrediction] = []
        # P100 uses the SDPA math fallback, whose attention allocation grows
        # quadratically. Two rows retain all retrieved choices while keeping the
        # normalization prompt safely below the VRAM spike seen with 12 rows.
        normalization_batch_size = 2
        batch_start = 0
        while batch_start < len(request_rows):
            batch = request_rows[
                batch_start : batch_start + normalization_batch_size
            ]
            batch_serialized = json.dumps(
                {"entities": batch},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            batch_payload = (
                f"<normalization_input>\n{batch_serialized}\n"
                "</normalization_input>"
            )
            try:
                batch_result = self._parse(
                    instructions=NORMALIZATION_INSTRUCTIONS,
                    input_text=batch_payload,
                    response_type=NormalizationResponse,
                )
            except RuntimeError as exc:
                if (
                    "out of memory" not in str(exc).casefold()
                    or normalization_batch_size == 1
                ):
                    raise
                LOGGER.warning(
                    "CUDA OOM with normalization batch=%d; retrying one entity at a time",
                    normalization_batch_size,
                )
                normalization_batch_size = 1
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                continue
            raw_mappings.extend(batch_result.mappings)
            batch_start += len(batch)

        selected_by_index: dict[int, list[str]] = {}
        for mapping in raw_mappings:
            allowed = allowed_codes.get(mapping.entity_index)
            if allowed is None or mapping.entity_index in selected_by_index:
                continue
            selected: list[str] = []
            for code in mapping.candidates:
                normalized = str(code).strip()
                if normalized in allowed and normalized not in selected:
                    selected.append(normalized)
                if len(selected) >= self.max_candidates:
                    break
            selected_by_index[mapping.entity_index] = selected

        mappings: list[CandidatePrediction] = []
        for row in request_rows:
            index = int(row["entity_index"])
            selected = selected_by_index.get(index)
            if selected is None:
                # Missing LLM mappings only fall back when retrieval found one exact alias.
                exact = [
                    candidate.code
                    for candidate in retrieval_fallback[index]
                    if candidate.exact
                ]
                selected = exact[:1]
            mappings.append(
                CandidatePrediction(entity_index=index, candidates=selected)
            )

        sanitized = NormalizationResponse(mappings=mappings)
        self._write_cache(cache_path, sanitized)
        return sanitized
