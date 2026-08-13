from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from src.core.rag import (
    run_rag_query,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_NEIGHBOR_EXPANSION,
    DEFAULT_CONTEXT_MAX_TOKENS_RECHERCHE,
    DEFAULT_CONTEXT_MAX_TOKENS_RESUME,
    DEFAULT_MAX_PARENTS,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer questions using local Chroma index + OpenAI.")
    parser.add_argument("--query", type=str, required=True, help="User question.")
    parser.add_argument("--mode", choices=("recherche", "resume"), default="recherche")
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=DEFAULT_GENERATION_MODEL,
        help=f"Generation model (default: {DEFAULT_GENERATION_MODEL}).",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Override top-k retrieval (0 uses defaults).")
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Max chunks per source (0 uses defaults).",
    )
    parser.add_argument(
        "--neighbor-expansion",
        type=int,
        default=DEFAULT_NEIGHBOR_EXPANSION,
        help=f"Include prev/next neighbors (default: {DEFAULT_NEIGHBOR_EXPANSION}).",
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Enable MMR re-ranking (disabled by default).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-parent", action="store_true", help="Include parent chunks (summary mode).")
    group.add_argument("--no-parent", action="store_true", help="Disable parent chunks even in summary mode.")
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=0,
        help=(
            "Total prompt budget in tokens (0 uses mode default: "
            f"{DEFAULT_CONTEXT_MAX_TOKENS_RECHERCHE} recherche, {DEFAULT_CONTEXT_MAX_TOKENS_RESUME} resume)."
        ),
    )
    parser.add_argument(
        "--max-parents",
        type=int,
        default=DEFAULT_MAX_PARENTS,
        help=f"Max parent chunks to include (default: {DEFAULT_MAX_PARENTS}).",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata filter in key=value format (repeatable).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for retrieval and model outputs.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "").strip(),
        help="OpenAI API key (default: OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=60.0,
        help="OpenAI request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for OpenAI calls (default: 5).",
    )
    parser.add_argument(
        "--retry-base-delay-sec",
        type=float,
        default=1.0,
        help="Base delay for retry backoff (default: 1.0s).",
    )
    args = parser.parse_args()
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if args.max_per_source < 0:
        raise ValueError("--max-per-source must be >= 0")
    if args.neighbor_expansion < 0:
        raise ValueError("--neighbor-expansion must be >= 0")
    if args.context_max_tokens < 0:
        raise ValueError("--context-max-tokens must be >= 0")
    if args.max_parents < 0:
        raise ValueError("--max-parents must be >= 0")
    if args.request_timeout_sec <= 0:
        raise ValueError("--request-timeout-sec must be > 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.retry_base_delay_sec <= 0:
        raise ValueError("--retry-base-delay-sec must be > 0")
    return args


def main() -> None:
    args = parse_args()
    output = run_rag_query(
        query=args.query,
        mode=args.mode,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        top_k=args.top_k,
        max_per_source=args.max_per_source,
        neighbor_expansion=args.neighbor_expansion,
        mmr=args.mmr,
        include_parent=args.include_parent,
        no_parent=args.no_parent,
        context_max_tokens=args.context_max_tokens,
        max_parents=args.max_parents,
        filters=args.filter,
        debug=args.debug,
        openai_api_key=args.openai_api_key,
        request_timeout_sec=args.request_timeout_sec,
        max_retries=args.max_retries,
        retry_base_delay_sec=args.retry_base_delay_sec,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
