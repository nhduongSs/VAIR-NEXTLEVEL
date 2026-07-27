from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_EMBEDDER_CACHE: dict[tuple[str, str], object] = {}


def normalize_term(value: str) -> str:
    """Normalize a term for retrieval only; never use this text for offsets."""

    decomposed = unicodedata.normalize("NFD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return _NON_ALNUM_RE.sub(" ", without_marks).strip()


def _character_ngrams(value: str, size: int = 3) -> set[str]:
    compact = value.replace(" ", "_")
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


@dataclass(frozen=True)
class TerminologyEntry:
    code: str
    label: str
    aliases: tuple[str, ...]

    @property
    def search_text(self) -> str:
        values = (self.label, *self.aliases)
        return " ; ".join(value for value in values if value)


@dataclass(frozen=True)
class RetrievedCandidate:
    code: str
    label: str
    score: float
    exact: bool


def _iter_jsonl(path: Path) -> Iterable[TerminologyEntry]:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            code = str(item["code"]).strip()
            label = str(item["label"]).strip()
            raw_aliases = item.get("aliases", [])
            if isinstance(raw_aliases, str):
                raw_aliases = raw_aliases.split("|")
            aliases = tuple(
                str(alias).strip() for alias in raw_aliases if str(alias).strip()
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid terminology row") from exc
        if code and label:
            yield TerminologyEntry(code=code, label=label, aliases=aliases)


def _iter_delimited(path: Path) -> Iterable[TerminologyEntry]:
    sample = path.read_text(encoding="utf-8")[:8192]
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
    except csv.Error:
        pass

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = {str(name).strip().casefold(): name for name in (reader.fieldnames or [])}
        if "code" not in fieldnames or "label" not in fieldnames:
            raise ValueError(
                f"{path}: expected a header with at least 'code' and 'label' columns"
            )
        for line_number, row in enumerate(reader, start=2):
            code = str(row[fieldnames["code"]] or "").strip()
            label = str(row[fieldnames["label"]] or "").strip()
            aliases_value = (
                str(row.get(fieldnames.get("aliases", ""), "") or "").strip()
                if "aliases" in fieldnames
                else ""
            )
            aliases = tuple(
                alias.strip() for alias in aliases_value.split("|") if alias.strip()
            )
            if not code or not label:
                raise ValueError(f"{path}:{line_number}: code and label are required")
            yield TerminologyEntry(code=code, label=label, aliases=aliases)


def load_terminology(path: Path) -> list[TerminologyEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"Terminology file does not exist: {path}")
    iterator = _iter_jsonl(path) if path.suffix.lower() == ".jsonl" else _iter_delimited(path)
    entries: list[TerminologyEntry] = []
    seen: set[str] = set()
    for entry in iterator:
        if entry.code in seen:
            continue
        seen.add(entry.code)
        entries.append(entry)
    if not entries:
        raise ValueError(f"Terminology file is empty: {path}")
    return entries


class TerminologyIndex:
    """Fast lexical retrieval with an optional local E5 semantic reranker."""

    def __init__(
        self,
        path: Path,
        *,
        embedding_model: str | None = None,
        embedding_device: str = "cpu",
        embedding_cache_dir: Path | None = None,
        embedding_batch_size: int = 128,
    ) -> None:
        self.path = path
        self.entries = load_terminology(path)
        self._normalized_names: list[tuple[str, ...]] = []
        self._tokens: list[set[str]] = []
        self._ngrams: list[set[str]] = []
        self._exact: dict[str, set[int]] = {}
        self._token_postings: dict[str, set[int]] = {}
        self._ngram_postings: dict[str, set[int]] = {}

        for index, entry in enumerate(self.entries):
            names = tuple(
                dict.fromkeys(
                    normalized
                    for normalized in (
                        normalize_term(entry.label),
                        *(normalize_term(alias) for alias in entry.aliases),
                    )
                    if normalized
                )
            )
            token_set = set().union(*(set(name.split()) for name in names))
            ngram_set = set().union(*(_character_ngrams(name) for name in names))
            self._normalized_names.append(names)
            self._tokens.append(token_set)
            self._ngrams.append(ngram_set)
            for name in names:
                self._exact.setdefault(name, set()).add(index)
            for token in token_set:
                if len(token) >= 2:
                    self._token_postings.setdefault(token, set()).add(index)
            for ngram in ngram_set:
                self._ngram_postings.setdefault(ngram, set()).add(index)

        self._embedder = None
        self._embeddings = None
        if embedding_model:
            self._prepare_embeddings(
                embedding_model=embedding_model,
                device=embedding_device,
                cache_dir=embedding_cache_dir,
                batch_size=embedding_batch_size,
            )

    def _prepare_embeddings(
        self,
        *,
        embedding_model: str,
        device: str,
        cache_dir: Path | None,
        batch_size: int,
    ) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval requires the local extra: "
                "python -m pip install -e '.[local]'"
            ) from exc

        embedder_key = (embedding_model, device)
        if embedder_key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[embedder_key] = SentenceTransformer(
                embedding_model,
                device=device,
                local_files_only=True,
            )
        self._embedder = _EMBEDDER_CACHE[embedder_key]
        fingerprint_value = (
            f"{self.path.resolve()}:{self.path.stat().st_size}:"
            f"{self.path.stat().st_mtime_ns}:{embedding_model}"
        )
        fingerprint = hashlib.sha256(fingerprint_value.encode("utf-8")).hexdigest()[:20]
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{self.path.stem}.{fingerprint}.embeddings.npy"

        if cache_path is not None and cache_path.exists():
            embeddings = np.load(cache_path, mmap_mode="r")
            if embeddings.shape[0] != len(self.entries):
                raise RuntimeError(f"Stale semantic index: {cache_path}")
            self._embeddings = embeddings
            return

        passages = [f"passage: {entry.search_text}" for entry in self.entries]
        embeddings = self._embedder.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        embeddings = embeddings.astype("float32")
        if cache_path is not None:
            np.save(cache_path, embeddings)
            self._embeddings = np.load(cache_path, mmap_mode="r")
        else:
            self._embeddings = embeddings

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def _lexical_score(
        self,
        index: int,
        query: str,
        query_tokens: set[str],
        query_ngrams: set[str],
    ) -> tuple[float, bool]:
        names = self._normalized_names[index]
        exact = query in names
        token_score = self._jaccard(query_tokens, self._tokens[index])
        ngram_score = self._jaccard(query_ngrams, self._ngrams[index])
        contains = any(query in name or name in query for name in names)
        score = (
            (1.0 if exact else 0.0)
            + (0.18 if contains else 0.0)
            + 0.55 * token_score
            + 0.27 * ngram_score
        )
        return min(score, 1.0), exact

    def retrieve(
        self,
        mention: str,
        *,
        context: str = "",
        top_k: int = 20,
        lexical_pool_size: int = 160,
    ) -> list[RetrievedCandidate]:
        query = normalize_term(mention)
        if not query:
            return []
        query_tokens = set(query.split())
        query_ngrams = _character_ngrams(query)

        pool: set[int] = set(self._exact.get(query, set()))
        for token in query_tokens:
            pool.update(self._token_postings.get(token, set()))
        for ngram in query_ngrams:
            pool.update(self._ngram_postings.get(ngram, set()))

        lexical_by_index = {
            index: self._lexical_score(
                index,
                query,
                query_tokens,
                query_ngrams,
            )
            for index in pool
        }

        semantic_scores: dict[int, float] = {}
        if self._embedder is not None and self._embeddings is not None:
            import numpy as np

            semantic_query = normalize_term(f"{mention} {context[:240]}")
            query_embedding = self._embedder.encode(
                [f"query: {semantic_query}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            # Add a global semantic pool. This is necessary for Vietnamese
            # mentions whose official ICD/RxNorm labels are only in English.
            all_scores = np.asarray(self._embeddings) @ query_embedding
            semantic_pool_size = min(lexical_pool_size, len(self.entries))
            if semantic_pool_size:
                semantic_indices = np.argpartition(
                    all_scores,
                    -semantic_pool_size,
                )[-semantic_pool_size:]
                pool.update(int(index) for index in semantic_indices)
            semantic_scores = {
                int(index): float((all_scores[index] + 1.0) / 2.0)
                for index in pool
            }

        if not pool:
            return []

        lexical = [
            (*lexical_by_index.get(index, (0.0, False)), index)
            for index in pool
        ]
        lexical.sort(key=lambda item: (-item[0], not item[1], self.entries[item[2]].code))
        lexical = lexical[:lexical_pool_size]

        if semantic_scores:
            # Do not discard semantic-only hits just because their lexical
            # score is zero; rerank the union by the final combined score.
            lexical = [
                (*lexical_by_index.get(index, (0.0, False)), index)
                for index in pool
            ]

        ranked: list[RetrievedCandidate] = []
        for lexical_score, exact, index in lexical:
            semantic_score = semantic_scores.get(index)
            combined = (
                lexical_score
                if semantic_score is None
                else 0.52 * lexical_score + 0.48 * semantic_score
            )
            if not math.isfinite(combined):
                continue
            entry = self.entries[index]
            ranked.append(
                RetrievedCandidate(
                    code=entry.code,
                    label=entry.label,
                    score=combined,
                    exact=exact,
                )
            )
        ranked.sort(key=lambda item: (-item.score, not item.exact, item.code))
        return ranked[:top_k]
