from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import (
    PipelineConfig,
    create_submission_zip,
    load_allowlist,
    run_pipeline,
    validate_all,
)


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

    validate = subparsers.add_parser("validate", help="Validate an output directory")
    validate.add_argument("--input-dir", type=Path, default=Path("input"))
    validate.add_argument("--output-dir", type=Path, default=Path("output"))
    validate.add_argument("--zip-path", type=Path)

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

    validate_all(args.input_dir, args.output_dir)
    if args.zip_path is not None:
        create_submission_zip(args.output_dir, args.zip_path)
        logging.info("Created %s", args.zip_path)
    logging.info("Validation passed")


if __name__ == "__main__":
    main()
