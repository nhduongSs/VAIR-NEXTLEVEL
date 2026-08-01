from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .icd_vn import build as build_icd_kb
from .models import EntityType
from .pipeline import (
    PipelineConfig,
    create_submission_zip,
    load_allowlist,
    run_pipeline,
    validate_all,
)
from .pipeline_v2 import PipelineV2Config, run_pipeline_v2
from .scoring import load_records, score_corpus


def _parse_ids(raw: str | None) -> frozenset[str] | None:
    if raw is None:
        return None
    values: set[str] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token.isdigit():
            raise argparse.ArgumentTypeError(f"Invalid record ID: {token!r}")
        values.add(str(int(token)))
    return frozenset(values)


def _parse_thresholds(raw: list[str] | None) -> dict[EntityType, float] | None:
    if not raw:
        return None
    from .gliner_ner import DEFAULT_THRESHOLDS

    thresholds = dict(DEFAULT_THRESHOLDS)
    for item in raw:
        name, separator, value = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"Expected TYPE=VALUE, got {item!r}")
        try:
            entity_type = EntityType(name.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"Unknown entity type: {name!r}") from None
        try:
            thresholds[entity_type] = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid threshold: {value!r}") from None
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical-coder",
        description="Self-hosted clinical extraction and ICD-10/RxNorm linking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run local self-hosted inference")
    predict.add_argument("--input-dir", type=Path, default=Path("input"))
    predict.add_argument("--output-dir", type=Path, default=Path("output"))
    predict.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".medical_coder_cache"),
    )
    predict.add_argument(
        "--model-path",
        "--model",
        dest="model",
        default="models/Qwen3-8B",
        help="Local snapshot directory (default: models/Qwen3-8B)",
    )
    predict.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
        help="Weight loading mode; 4bit is recommended for a Kaggle 16 GB GPU",
    )
    predict.add_argument("--workers", type=int, default=1)
    predict.add_argument("--max-candidates", type=int, default=3)
    predict.add_argument(
        "--retrieval-top-k",
        type=int,
        default=20,
        help="Number of valid local codes offered to the LLM per entity",
    )
    predict.add_argument(
        "--icd-kb",
        type=Path,
        help="Local ICD terminology CSV/TSV/JSONL with code, label, aliases",
    )
    predict.add_argument(
        "--rxnorm-kb",
        type=Path,
        help="Local RxNorm terminology CSV/TSV/JSONL with code, label, aliases",
    )
    predict.add_argument(
        "--embedding-model",
        default="models/multilingual-e5-small",
        help="Local semantic retrieval model; pass 'none' for lexical-only",
    )
    predict.add_argument(
        "--embedding-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    predict.add_argument("--max-input-tokens", type=int, default=24_576)
    predict.add_argument("--max-new-tokens", type=int, default=8_192)
    predict.add_argument(
        "--ids",
        help="Comma-separated IDs for a small test run, e.g. 1,2,3",
    )
    predict.add_argument("--overwrite", action="store_true")
    predict.add_argument("--icd-allowlist", type=Path)
    predict.add_argument("--rxnorm-allowlist", type=Path)
    predict.add_argument(
        "--zip-path",
        type=Path,
        default=Path("output.zip"),
    )
    predict.add_argument("--no-zip", action="store_true")

    predict_v2 = subparsers.add_parser(
        "predict-v2",
        help="Precision-first CPU pipeline: GLiNER spans + exact-alias linking",
    )
    predict_v2.add_argument("--input-dir", type=Path, default=Path("input"))
    predict_v2.add_argument("--output-dir", type=Path, default=Path("output"))
    predict_v2.add_argument(
        "--model-path",
        "--model",
        dest="model",
        default="urchade/gliner_multi-v2.1",
        help="GLiNER snapshot directory or hub id",
    )
    predict_v2.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    predict_v2.add_argument("--icd-kb", type=Path, help="Vietnamese ICD-10 terminology")
    predict_v2.add_argument("--rxnorm-kb", type=Path, help="RxNorm terminology")
    predict_v2.add_argument(
        "--max-candidates",
        type=int,
        default=1,
        help="Codes per concept; >1 raises the spurious-candidate penalty",
    )
    predict_v2.add_argument("--raw-floor", type=float, default=0.02)
    predict_v2.add_argument("--max-chunk-chars", type=int, default=800)
    predict_v2.add_argument(
        "--threshold",
        action="append",
        metavar="TYPE=VALUE",
        help="Override a per-type GLiNER threshold, e.g. --threshold CHẨN_ĐOÁN=0.30",
    )
    predict_v2.add_argument(
        "--primary-teacher",
        help="Qwen model for TRIỆU_CHỨNG->CHẨN_ĐOÁN type correction (needs a GPU)",
    )
    predict_v2.add_argument(
        "--secondary-teacher",
        help="Second Qwen model; required before any span is added by consensus",
    )
    predict_v2.add_argument(
        "--teacher-device", default="cuda", help="Device for the Qwen teachers"
    )
    predict_v2.add_argument(
        "--teacher-quantization", choices=("4bit", "8bit", "none"), default="4bit"
    )
    predict_v2.add_argument("--teacher-batch-size", type=int, default=48)
    predict_v2.add_argument(
        "--addition-margin",
        type=float,
        default=1.0,
        help="Minimum logit margin over NONE before a span may be added",
    )
    predict_v2.add_argument(
        "--reject-margin",
        type=float,
        default=None,
        help=(
            "Drop baseline spans the teacher calls 'not a concept' by this logit "
            "margin. Omit to keep every span (current behaviour)."
        ),
    )
    predict_v2.add_argument(
        "--teacher-decides",
        action="store_true",
        help="Teacher chọn span và gán type; GLiNER chỉ đề xuất ứng viên",
    )
    predict_v2.add_argument("--decide-margin", type=float, default=0.0)
    predict_v2.add_argument(
        "--emit-assertions",
        action="store_true",
        help="Bật luật phát hiện isNegated/isHistorical (xem assertions.py)",
    )
    predict_v2.add_argument("--ids", help="Comma-separated IDs for a smoke run")
    predict_v2.add_argument("--zip-path", type=Path, default=Path("output.zip"))
    predict_v2.add_argument("--no-zip", action="store_true")

    label = subparsers.add_parser(
        "label-gpt",
        help="CÔNG CỤ PHÁT TRIỂN: gán nhãn giả bằng LLM ngoài, KHÔNG thuộc pipeline nộp bài",
    )
    label.add_argument("--input-dir", type=Path, default=Path("input"))
    label.add_argument("--output-dir", type=Path, default=Path("data/pseudo_gt"))
    label.add_argument("--model", default=None, help="mặc định xem DEFAULT_MODEL trong gpt_labeler")
    label.add_argument("--cache-dir", type=Path, default=Path(".gpt_label_cache"))
    label.add_argument("--ids", help="Chỉ gán nhãn một số bản ghi, ví dụ 1,2,3")

    validate = subparsers.add_parser("validate", help="Validate an output directory")
    validate.add_argument("--input-dir", type=Path, default=Path("input"))
    validate.add_argument("--output-dir", type=Path, default=Path("output"))
    validate.add_argument("--zip-path", type=Path)

    build_kb = subparsers.add_parser(
        "build-icd-kb",
        help="Build the Vietnamese ICD-10 terminology TSV from the MoH TT06 catalog",
    )
    build_kb.add_argument(
        "--xlsx",
        type=Path,
        default=Path("data/kb/raw/Phu_luc_Bang_danh_muc_ICD10_FINAL_TT06_2026.xlsx"),
    )
    build_kb.add_argument(
        "--output", type=Path, default=Path("data/terminology/icd10_vn.tsv")
    )

    score = subparsers.add_parser(
        "score", help="Score predictions against labelled ground truth"
    )
    score.add_argument("--output-dir", type=Path, default=Path("output"))
    score.add_argument("--truth-dir", type=Path, required=True)
    score.add_argument(
        "--per-record", action="store_true", help="Also print the worst records"
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "predict":
        if args.workers < 1:
            parser.error("--workers must be >= 1")
        if args.max_candidates < 1:
            parser.error("--max-candidates must be >= 1")
        if args.retrieval_top_k < args.max_candidates:
            parser.error("--retrieval-top-k must be >= --max-candidates")
        if args.max_input_tokens < 1024 or args.max_new_tokens < 128:
            parser.error("token limits are too small")
        try:
            selected_ids = _parse_ids(args.ids)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

        config = PipelineConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            model=args.model,
            reasoning_effort="none",
            workers=args.workers,
            max_candidates=args.max_candidates,
            overwrite=args.overwrite,
            selected_ids=selected_ids,
            icd_allowlist=load_allowlist(args.icd_allowlist, uppercase=True),
            rxnorm_allowlist=load_allowlist(args.rxnorm_allowlist, uppercase=False),
            quantization=args.quantization,
            icd_terminology=args.icd_kb,
            rxnorm_terminology=args.rxnorm_kb,
            embedding_model=(
                None if args.embedding_model.casefold() == "none" else args.embedding_model
            ),
            embedding_device=args.embedding_device,
            retrieval_top_k=args.retrieval_top_k,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        run_pipeline(config)

        # A selected subset is useful for smoke tests but is not a valid final submission.
        if selected_ids is None:
            validate_all(args.input_dir, args.output_dir)
            if not args.no_zip:
                create_submission_zip(args.output_dir, args.zip_path)
                logging.info("Created %s", args.zip_path)
        else:
            logging.info(
                "Subset run completed; skipping full-directory validation and ZIP creation"
            )
        return

    if args.command == "predict-v2":
        if args.max_candidates < 0:
            parser.error("--max-candidates must be >= 0")
        try:
            selected_ids = _parse_ids(args.ids)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        try:
            thresholds = _parse_thresholds(args.threshold)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

        total = run_pipeline_v2(
            PipelineV2Config(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_path=args.model,
                device=args.device,
                icd_kb=args.icd_kb,
                rxnorm_kb=args.rxnorm_kb,
                thresholds=thresholds,
                raw_floor=args.raw_floor,
                max_chunk_chars=args.max_chunk_chars,
                max_candidates=args.max_candidates,
                selected_ids=selected_ids,
                emit_assertions=args.emit_assertions,
                primary_teacher=args.primary_teacher,
                secondary_teacher=args.secondary_teacher,
                teacher_device=args.teacher_device,
                teacher_quantization=args.teacher_quantization,
                teacher_batch_size=args.teacher_batch_size,
                addition_margin=args.addition_margin,
                reject_margin=args.reject_margin,
                teacher_decides=args.teacher_decides,
                decide_margin=args.decide_margin,
            )
        )
        logging.info("Wrote %d concepts", total)
        if selected_ids is None:
            validate_all(args.input_dir, args.output_dir)
            if not args.no_zip:
                create_submission_zip(args.output_dir, args.zip_path)
                logging.info("Created %s", args.zip_path)
        else:
            logging.info("Subset run completed; skipping validation and ZIP creation")
        return

    if args.command == "label-gpt":
        from .gpt_labeler import DEFAULT_MODEL, LabelConfig, label_corpus

        totals = label_corpus(
            LabelConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model=args.model or DEFAULT_MODEL,
                cache_dir=args.cache_dir,
                selected_ids=_parse_ids(args.ids),
            )
        )
        logging.info(
            "%d tài liệu, %d khái niệm, %d không căn được, %d lấy từ cache",
            totals["documents"], totals["concepts"], totals["dropped"], totals["cached"],
        )
        logging.info("Bước tiếp: python tools/calibrate_pseudo_gt.py %s", args.output_dir)
        return

    if args.command == "build-icd-kb":
        count = build_icd_kb(args.xlsx, args.output)
        logging.info("Wrote %d ICD-10 codes to %s", count, args.output)
        return

    if args.command == "score":
        truth = load_records(args.truth_dir)
        if not truth:
            parser.error(f"No labelled records found in {args.truth_dir}")
        predictions = load_records(args.output_dir)
        result = score_corpus(predictions, truth)
        print(result.format_report())
        if args.per_record:
            worst = sorted(result.per_record, key=lambda record: record.text)[:10]
            print("\nweakest records by text score:")
            for record in worst:
                print(
                    f"  {record.record_id:>4}  text={record.text:.3f} "
                    f"assert={record.assertions:.3f} cand={record.candidates:.3f} "
                    f"pred={record.predicted} gold={record.expected}"
                )
        return

    validate_all(args.input_dir, args.output_dir)
    if args.zip_path is not None:
        create_submission_zip(args.output_dir, args.zip_path)
        logging.info("Created %s", args.zip_path)
    logging.info("Validation passed")


if __name__ == "__main__":
    main()
