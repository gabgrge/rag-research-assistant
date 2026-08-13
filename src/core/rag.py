from __future__ import annotations

import re
import json
import os
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Mapping

from dotenv import load_dotenv

from src.utils.common_utils import to_int, utc_now_iso
from src.utils.logging_utils import build_script_logger
from src.integrations.llm.openai_utils import build_openai_client, call_with_retries
from src.utils.type_hints import ChromaCollection, OpenAIClient
from src.integrations.vector.chroma_utils import build_chroma_collection
from src.utils.paths import INDEX_DIR, LOGS_DIR

load_dotenv()

INDEX_DIR = INDEX_DIR.resolve()
LOG_PATH = (LOGS_DIR / "rag_answer.log").resolve()

DEFAULT_COLLECTION_NAME = "rag_leaf_chunks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GENERATION_MODEL = "gpt-5-mini"

DEFAULT_TOP_K_RECHERCHE = 6
DEFAULT_TOP_K_RESUME = 12
DEFAULT_MAX_PER_SOURCE_RECHERCHE = 2
DEFAULT_MAX_PER_SOURCE_RESUME = 3
DEFAULT_NEIGHBOR_EXPANSION = 1
DEFAULT_CONTEXT_MAX_TOKENS_RECHERCHE = 5000
DEFAULT_CONTEXT_MAX_TOKENS_RESUME = 8000
DEFAULT_MAX_PARENTS = 6
DEFAULT_MIN_CONTEXT_TOKENS = 300

NO_SOURCE_SENTINEL = "UNKNOWN"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOGGER = build_script_logger("rag_answer", LOG_PATH)

NO_SOURCE_MESSAGE = (
    "Je ne trouve pas de passage pertinent dans les sources actuelles. "
    "Essayez de reformuler, de préciser, ou de filtrer."
)

SYSTEM_PROMPT = (
    "Tu es un assistant RAG.\n"
    "Règle stricte: tu n'utilises QUE les informations présentes dans les Sources.\n"
    f"Si les Sources ne contiennent pas la réponse, réponds exactement: \"{NO_SOURCE_SENTINEL}\".\n"
    "Tu réponds en français, de façon concise et factuelle.\n"
    "Chaque point important doit être suivi d'au moins une citation [C#].\n"
    "Ne cite jamais un [C#] qui n'existe pas dans les Sources."
)
USER_PROMPT_PREFIX = "Question:\n{query}\n\nSources:\n"
USER_PROMPT_SUFFIX = (
    "Retourne un JSON strict avec:\n"
    "- answer: soit la réponse en français avec citations [C#], soit exactement \"UNKNOWN\"\n"
    "- citation_ids: liste unique des labels utilisés dans answer (ex: [\"C1\", \"C3\"]).\n"
    "Contraintes:\n"
    "- citation_ids doit correspondre aux [C#] présents dans answer\n"
    "- si answer == \"UNKNOWN\", citation_ids doit être []"
)


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    text: str
    meta: Dict[str, object]
    distance: Optional[float]
    score: float
    origin: str
    retrieval_score: Optional[float]
    expanded_from: Optional[str]
    expanded_depth: Optional[int]


def embed_query(
    client: OpenAIClient,
    text: str,
    embedding_model: str,
    max_retries: int,
    retry_base_delay_sec: float,
) -> List[float]:
    def _call() -> object:
        return client.embeddings.create(model=embedding_model, input=[text])

    response = call_with_retries(
        _call,
        max_retries=max_retries,
        retry_base_delay_sec=retry_base_delay_sec,
        label="embedding",
        logger=LOGGER,
    )
    data = getattr(response, "data", None)
    if not isinstance(data, list) or not data:
        raise RuntimeError("Embedding response missing data.")
    embedding = getattr(data[0], "embedding", None)
    if not isinstance(embedding, list):
        raise RuntimeError("Embedding response missing embedding.")
    return embedding


def parse_filters(raw_filters: Iterable[str]) -> Dict[str, object]:
    filters: Dict[str, object] = {}
    for raw in raw_filters:
        if "=" not in raw:
            raise ValueError(f"Invalid filter '{raw}' (expected key=value).")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid filter '{raw}' (missing key).")
        if key in filters:
            existing = filters[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                filters[key] = [existing, value]
        else:
            filters[key] = value
    return filters


def build_where_clause(filters: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not filters:
        return None
    where: Dict[str, object] = {}
    for key, value in filters.items():
        if isinstance(value, list):
            where[key] = {"$in": value}
        else:
            where[key] = value
    return where


def flatten_query_results(results: Mapping[str, object]) -> List[Candidate]:
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    candidates: List[Candidate] = []
    for idx, chunk_id in enumerate(ids):
        if not isinstance(chunk_id, str):
            continue
        text = documents[idx] if idx < len(documents) else ""
        meta = metadatas[idx] if idx < len(metadatas) else {}
        distance_raw = distances[idx] if idx < len(distances) else 0.0
        try:
            distance = float(distance_raw)
        except (TypeError, ValueError):
            distance = 0.0
        score = -distance
        if not isinstance(meta, dict):
            meta = {}
        candidates.append(
            Candidate(
                chunk_id=chunk_id,
                text=str(text or ""),
                meta=meta,
                distance=distance,
                score=score,
                origin="seed",
                retrieval_score=score,
                expanded_from=None,
                expanded_depth=None,
            )
        )
    return candidates


def apply_max_per_source(candidates: List[Candidate], max_per_source: int, top_k: int) -> List[Candidate]:
    if max_per_source <= 0:
        return candidates[:top_k]
    counts: Dict[str, int] = {}
    filtered: List[Candidate] = []
    for cand in candidates:
        source_id = str(cand.meta.get("source_id", ""))
        counts.setdefault(source_id, 0)
        if counts[source_id] >= max_per_source:
            continue
        counts[source_id] += 1
        filtered.append(cand)
        if len(filtered) >= top_k:
            break
    return filtered


def apply_soft_cap_per_source(candidates: List[Candidate], max_per_source: int) -> List[Candidate]:
    if max_per_source <= 0:
        return candidates
    counts: Dict[str, int] = {}
    primary: List[Candidate] = []
    overflow: List[Candidate] = []
    for cand in candidates:
        source_id = str(cand.meta.get("source_id", ""))
        count = counts.get(source_id, 0)
        if count < max_per_source:
            primary.append(cand)
            counts[source_id] = count + 1
        else:
            overflow.append(cand)
    return primary + overflow


def redundancy_ratio(candidates: List[Candidate]) -> float:
    if not candidates:
        return 0.0
    counts: Dict[str, int] = {}
    for cand in candidates:
        source_id = str(cand.meta.get("source_id", ""))
        counts[source_id] = counts.get(source_id, 0) + 1
    max_count = max(counts.values()) if counts else 0
    return max_count / max(1, len(candidates))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if math.isclose(norm_a, 0.0, abs_tol=1e-12) or math.isclose(norm_b, 0.0, abs_tol=1e-12):
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def apply_mmr(
    candidates: List[Candidate],
    query_embedding: Sequence[float],
    *,
    top_k: int,
    lambda_mult: float = 0.7,
    collection: ChromaCollection,
) -> List[Candidate]:
    ids = [cand.chunk_id for cand in candidates]
    if not ids:
        return []
    payload = collection.get(ids=ids, include=["embeddings"])
    embeddings = payload.get("embeddings", [])
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError("MMR embeddings unavailable.")
    id_to_embedding: Dict[str, Sequence[float]] = {}
    for chunk_id, embedding in zip(ids, embeddings):
        if isinstance(chunk_id, str) and isinstance(embedding, list):
            id_to_embedding[chunk_id] = embedding
    if not id_to_embedding:
        raise RuntimeError("MMR embeddings empty.")

    remaining = [cand for cand in candidates if cand.chunk_id in id_to_embedding]
    if not remaining:
        raise RuntimeError("MMR has no candidates after embedding filter.")
    selected: List[Candidate] = []
    while remaining and len(selected) < top_k:
        best = None
        best_score = None
        for cand in remaining:
            emb = id_to_embedding.get(cand.chunk_id)
            if emb is None:
                continue
            sim_to_query = cosine_similarity(query_embedding, emb)
            if not selected:
                score = sim_to_query
            else:
                sim_to_selected = max(
                    cosine_similarity(emb, id_to_embedding.get(s.chunk_id, [])) for s in selected
                )
                score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
            if best is None or score > best_score:
                best = cand
                best_score = score
        if best is None:
            break
        selected.append(best)
        remaining = [cand for cand in remaining if cand.chunk_id != best.chunk_id]
    return selected


def apply_rank_scores(candidates: List[Candidate]) -> List[Candidate]:
    ranked: List[Candidate] = []
    for idx, cand in enumerate(candidates):
        score = max(0.0, 1.0 - idx * 0.001)
        ranked.append(
            Candidate(
                chunk_id=cand.chunk_id,
                text=cand.text,
                meta=cand.meta,
                distance=cand.distance,
                score=score,
                origin=cand.origin,
                retrieval_score=cand.retrieval_score,
                expanded_from=cand.expanded_from,
                expanded_depth=cand.expanded_depth,
            )
        )
    return ranked


def fetch_by_ids(collection: ChromaCollection, ids: Iterable[str]) -> Dict[str, Candidate]:
    ids_list = [cid for cid in ids if isinstance(cid, str) and cid]
    if not ids_list:
        return {}
    payload = collection.get(ids=ids_list, include=["documents", "metadatas"])
    returned_ids = payload.get("ids", [])
    documents = payload.get("documents", [])
    metadatas = payload.get("metadatas", [])
    out: Dict[str, Candidate] = {}
    for idx, chunk_id in enumerate(returned_ids):
        text = documents[idx] if idx < len(documents) else ""
        meta = metadatas[idx] if idx < len(metadatas) else {}
        if not isinstance(meta, dict):
            meta = {}
        out[str(chunk_id)] = Candidate(
            chunk_id=str(chunk_id),
            text=str(text or ""),
            meta=meta,
            distance=None,
            score=0.0,
            origin="expanded",
            retrieval_score=None,
            expanded_from=None,
            expanded_depth=None,
        )
    return out


def expand_candidates(
    selected: List[Candidate],
    collection: ChromaCollection,
    neighbor_expansion: int,
    include_parent: bool,
    max_parents: int,
) -> List[Candidate]:
    if neighbor_expansion <= 0 and not include_parent:
        return selected

    by_id: Dict[str, Candidate] = {cand.chunk_id: cand for cand in selected}
    frontier = [cand.chunk_id for cand in selected]

    for depth in range(neighbor_expansion):
        extra_ids: set[str] = set()
        neighbor_scores: Dict[str, float] = {}
        neighbor_from: Dict[str, str] = {}
        for cid in frontier:
            cand = by_id.get(cid)
            if cand is None:
                continue
            meta = cand.meta
            prev_id = str(meta.get("prev_id", "")).strip()
            next_id = str(meta.get("next_id", "")).strip()
            if prev_id:
                extra_ids.add(prev_id)
                if cand.score >= neighbor_scores.get(prev_id, float("-inf")):
                    neighbor_scores[prev_id] = cand.score
                    neighbor_from[prev_id] = cand.chunk_id
            if next_id:
                extra_ids.add(next_id)
                if cand.score >= neighbor_scores.get(next_id, float("-inf")):
                    neighbor_scores[next_id] = cand.score
                    neighbor_from[next_id] = cand.chunk_id
        extra_ids = {cid for cid in extra_ids if cid not in by_id}
        if not extra_ids:
            break
        extra = fetch_by_ids(collection, extra_ids)
        for cid, cand in extra.items():
            base_score = neighbor_scores.get(cid, 0.0)
            adjusted = max(0.0, base_score - (depth + 1) * 0.05)
            by_id[cid] = Candidate(
                chunk_id=cand.chunk_id,
                text=cand.text,
                meta=cand.meta,
                distance=cand.distance,
                score=adjusted,
                origin="expanded",
                retrieval_score=None,
                expanded_from=neighbor_from.get(cid),
                expanded_depth=depth + 1,
            )
        frontier = list(extra_ids)

    if include_parent and max_parents != 0:
        parent_scores: Dict[str, float] = {}
        parent_by_source: Dict[str, Tuple[str, float, str]] = {}
        for cand in by_id.values():
            parent_id = str(cand.meta.get("parent_id", "")).strip()
            source_id = str(cand.meta.get("source_id", "")).strip()
            if not parent_id or parent_id in by_id or not source_id:
                continue
            existing = parent_by_source.get(source_id)
            if existing is None or cand.score > existing[1]:
                parent_by_source[source_id] = (parent_id, cand.score, cand.chunk_id)
                parent_scores[parent_id] = cand.score

        parent_ids = {parent_id for parent_id, _, _ in parent_by_source.values()}
        parent_from: Dict[str, str] = {parent_id: child_id for parent_id, _, child_id in parent_by_source.values()}
        if 0 < max_parents < len(parent_ids):
            ranked = sorted(parent_scores.items(), key=lambda item: item[1], reverse=True)
            allowed = {parent_id for parent_id, _ in ranked[:max_parents]}
            parent_ids = {pid for pid in parent_ids if pid in allowed}
        extra = fetch_by_ids(collection, parent_ids)
        for cid, cand in extra.items():
            base_score = parent_scores.get(cid, 0.0)
            adjusted = max(0.0, base_score - 0.03)
            by_id[cid] = Candidate(
                chunk_id=cand.chunk_id,
                text=cand.text,
                meta=cand.meta,
                distance=cand.distance,
                score=adjusted,
                origin="expanded",
                retrieval_score=None,
                expanded_from=parent_from.get(cid),
                expanded_depth=1,
            )

    return list(by_id.values())


def token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken
    except ImportError:
        return len(text.split())
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except (ValueError, KeyError):
        return len(text.split())
    return len(encoding.encode(text, disallowed_special=()))


def resolve_path(meta: Mapping[str, object]) -> str:
    return str(meta.get("origin_path") or "")


def retrieval_fields(cand: Candidate) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if cand.origin != "seed":
        return None, None, None
    score = round(cand.score, 6)
    retrieval_score = round(cand.retrieval_score, 6) if cand.retrieval_score is not None else None
    distance = round(cand.distance, 6) if cand.distance is not None else None
    return score, retrieval_score, distance


def expansion_fields(
    cand: Candidate, label_by_id: Mapping[str, str]
) -> Tuple[Optional[str], Optional[int]]:
    if cand.origin != "expanded":
        return None, None
    if cand.expanded_from is None:
        return None, cand.expanded_depth
    return label_by_id.get(cand.expanded_from), cand.expanded_depth


def add_optional_meta_fields(payload: Dict[str, object], meta: Mapping[str, object]) -> None:
    for key in ("unit_index_end", "unit_type_end", "char_start", "char_end"):
        value = meta.get(key)
        if value is None or value == "":
            continue
        payload[key] = value


def add_origin_fields(payload: Dict[str, object], cand: Candidate, label_by_id: Mapping[str, str]) -> None:
    if cand.origin == "seed":
        score, retrieval_score, distance = retrieval_fields(cand)
        payload["score"] = score
        payload["retrieval_score"] = retrieval_score
        payload["retrieval_distance"] = distance
    else:
        expanded_from, expanded_depth = expansion_fields(cand, label_by_id)
        payload["expanded_from_label"] = expanded_from
        payload["expanded_depth"] = expanded_depth


def build_context_blocks(
    candidates: List[Candidate], context_max_tokens: int
) -> Tuple[List[str], List[Candidate], int, int, bool]:
    blocks: List[str] = []
    used: List[Candidate] = []
    budget = context_max_tokens
    used_tokens = 0
    stopped_by_budget = False
    for idx, cand in enumerate(candidates, start=1):
        meta = cand.meta
        unit_type = str(meta.get("unit_type_start", ""))
        unit_index = to_int(meta.get("unit_index_start"), 0)
        header = (
            f"[C{idx}]\n"
            f"source_id: {meta.get('source_id','')}\n"
            f"filename: {meta.get('filename','')}\n"
            f"nature: {meta.get('nature','')}\n"
            f"ext: {meta.get('ext','')}\n"
            f"unit: {unit_type} {unit_index}\n"
            f"chunk_id: {cand.chunk_id}\n"
            f"text:\n{cand.text}\n"
        )
        block_tokens = token_count(header)
        if block_tokens > budget and blocks:
            stopped_by_budget = True
            break
        blocks.append(header)
        used.append(cand)
        budget -= block_tokens
        used_tokens += block_tokens
        if budget <= 0:
            stopped_by_budget = True
            break
    return blocks, used, used_tokens, budget, stopped_by_budget


def estimate_prompt_overhead_tokens(query: str) -> int:
    overhead_text = (
        SYSTEM_PROMPT
        + "\n"
        + USER_PROMPT_PREFIX.format(query=query)
        + "\n"
        + USER_PROMPT_SUFFIX
    )
    return token_count(overhead_text)


def build_generation_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citation_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["answer", "citation_ids"],
        "additionalProperties": False,
    }


def extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        texts: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
        if texts:
            return "\n".join(texts)
    raise RuntimeError("Could not extract text from response.")


def generate_answer(
    client: OpenAIClient,
    *,
    model: str,
    query: str,
    context_blocks: List[str],
    max_retries: int,
    retry_base_delay_sec: float,
) -> Dict[str, object]:
    if not context_blocks:
        return {"answer": NO_SOURCE_SENTINEL, "citation_ids": []}

    system_prompt = SYSTEM_PROMPT
    user_prompt = (
        USER_PROMPT_PREFIX.format(query=query)
        + "\n".join(context_blocks)
        + "\n\n"
        + USER_PROMPT_SUFFIX
    )

    def _call() -> object:
        return client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "rag_answer",
                    "strict": True,
                    "schema": build_generation_schema(),
                }
            },
        )

    response = call_with_retries(
        _call,
        max_retries=max_retries,
        retry_base_delay_sec=retry_base_delay_sec,
        label="generation",
        logger=LOGGER,
    )
    raw_text = extract_response_text(response)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON: {raw_text}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Model JSON output is not an object.")
    if "answer" not in parsed or "citation_ids" not in parsed:
        raise RuntimeError("Model JSON missing required keys.")
    return parsed


def build_final_output(
    *,
    query: str,
    mode: str,
    model: str,
    used_blocks: List[Candidate],
    block_labels: Dict[str, Candidate],
    model_json: Dict[str, object],
) -> Dict[str, object]:
    answer_text = str(model_json.get("answer", "")).strip()
    if answer_text == NO_SOURCE_SENTINEL:
        return {
            "query": query,
            "mode": mode,
            "model": model,
            "answer": NO_SOURCE_MESSAGE,
            "citations": [],
            "used_chunks": [],
            "generated_at": utc_now_iso(),
        }
    labels_raw = re.findall(r"\[C(\d+)]", answer_text)
    ordered_labels: List[str] = []
    seen: set[str] = set()
    for label in labels_raw:
        normalized = f"C{label}"
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_labels.append(normalized)
    if not ordered_labels:
        LOGGER.warning("Answer missing citations; forcing no-source response.")
        return {
            "query": query,
            "mode": mode,
            "model": model,
            "answer": NO_SOURCE_MESSAGE,
            "citations": [],
            "used_chunks": [],
            "generated_at": utc_now_iso(),
        }
    valid_labels: List[str] = []
    for label in ordered_labels:
        if label in block_labels:
            valid_labels.append(label)
    if not valid_labels:
        LOGGER.warning("Citations not found in sources; forcing no-source response.")
        return {
            "query": query,
            "mode": mode,
            "model": model,
            "answer": NO_SOURCE_MESSAGE,
            "citations": [],
            "used_chunks": [],
            "generated_at": utc_now_iso(),
        }

    remap = {old: f"C{idx}" for idx, old in enumerate(valid_labels, start=1)}

    def normalize_citations(match: re.Match[str]) -> str:
        old_label = f"C{match.group(1)}"
        new_label = remap.get(old_label)
        return f"[{new_label}]" if new_label else ""

    answer_text = re.sub(r"\[C(\d+)]", normalize_citations, answer_text)
    answer_text = re.sub(r"\s{2,}", " ", answer_text).strip()
    answer_text = re.sub(r"\s++([.,;:!?])", r"\1", answer_text)

    label_by_id = {block_labels[old].chunk_id: remap[old] for old in valid_labels}

    citations: List[Dict[str, object]] = []
    for label in valid_labels:
        cand = block_labels.get(label)
        if cand is None:
            continue
        new_label = remap[label]
        meta = cand.meta
        path = resolve_path(meta)
        payload: Dict[str, object] = {
            "label": new_label,
            "origin": cand.origin,
            "chunk_id": cand.chunk_id,
            "source_id": meta.get("source_id", ""),
            "nature": meta.get("nature", ""),
            "path": path,
            "filename": meta.get("filename", ""),
            "ext": meta.get("ext", ""),
            "unit_type": meta.get("unit_type_start", ""),
            "unit_index": to_int(meta.get("unit_index_start"), 0),
        }
        add_optional_meta_fields(payload, meta)
        add_origin_fields(payload, cand, label_by_id)
        citations.append(payload)

    used_chunks = []
    for cand in used_blocks:
        meta = cand.meta
        path = resolve_path(meta)
        payload: Dict[str, object] = {
            "origin": cand.origin,
            "chunk_id": cand.chunk_id,
            "source_id": meta.get("source_id", ""),
            "nature": meta.get("nature", ""),
            "path": path,
            "filename": meta.get("filename", ""),
            "ext": meta.get("ext", ""),
            "unit_type": meta.get("unit_type_start", ""),
            "unit_index": to_int(meta.get("unit_index_start"), 0),
        }
        add_optional_meta_fields(payload, meta)
        add_origin_fields(payload, cand, label_by_id)
        used_chunks.append(payload)

    return {
        "query": query,
        "mode": mode,
        "model": model,
        "answer": answer_text,
        "citations": citations,
        "used_chunks": used_chunks,
        "generated_at": utc_now_iso(),
    }


def run_rag_query(
    *,
    query: str,
    mode: str = "recherche",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    top_k: int = 0,
    max_per_source: int = 0,
    neighbor_expansion: int = DEFAULT_NEIGHBOR_EXPANSION,
    mmr: bool = False,
    include_parent: bool = False,
    no_parent: bool = False,
    context_max_tokens: int = 0,
    max_parents: int = DEFAULT_MAX_PARENTS,
    filters: Optional[Iterable[str]] = None,
    debug: bool = False,
    openai_api_key: str = "",
    request_timeout_sec: float = 60.0,
    max_retries: int = 5,
    retry_base_delay_sec: float = 1.0,
    client: Optional[OpenAIClient] = None,
) -> Dict[str, object]:
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if max_per_source < 0:
        raise ValueError("max_per_source must be >= 0")
    if neighbor_expansion < 0:
        raise ValueError("neighbor_expansion must be >= 0")
    if context_max_tokens < 0:
        raise ValueError("context_max_tokens must be >= 0")
    if max_parents < 0:
        raise ValueError("max_parents must be >= 0")
    if request_timeout_sec <= 0:
        raise ValueError("request_timeout_sec must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if retry_base_delay_sec <= 0:
        raise ValueError("retry_base_delay_sec must be > 0")

    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    selected_top_k = top_k or (DEFAULT_TOP_K_RESUME if mode == "resume" else DEFAULT_TOP_K_RECHERCHE)
    selected_max_per_source = max_per_source or (
        DEFAULT_MAX_PER_SOURCE_RESUME if mode == "resume" else DEFAULT_MAX_PER_SOURCE_RECHERCHE
    )
    selected_include_parent = include_parent or (mode == "resume" and not no_parent)
    selected_context_max_tokens = context_max_tokens or (
        DEFAULT_CONTEXT_MAX_TOKENS_RESUME if mode == "resume" else DEFAULT_CONTEXT_MAX_TOKENS_RECHERCHE
    )

    raw_filters = list(filters or [])
    parsed_filters = parse_filters(raw_filters)
    where = build_where_clause(parsed_filters)

    if client is None:
        client = build_openai_client(api_key=openai_api_key, timeout_sec=request_timeout_sec)
    collection = build_chroma_collection(index_dir=INDEX_DIR, collection_name=collection_name)

    query_embedding = embed_query(
        client=client,
        text=query,
        embedding_model=embedding_model,
        max_retries=max_retries,
        retry_base_delay_sec=retry_base_delay_sec,
    )

    initial_k = min(max(selected_top_k * 4, selected_top_k), 200)
    query_kwargs: Dict[str, object] = {
        "query_embeddings": [query_embedding],
        "n_results": initial_k,
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    raw_results = collection.query(**query_kwargs)
    if debug:
        LOGGER.info("Debug: initial_k=%d, where=%s", initial_k, where)
    candidates = flatten_query_results(raw_results)
    candidates.sort(key=lambda c: c.distance if c.distance is not None else 0.0)
    if debug:
        LOGGER.info("Debug: retrieved_candidates=%d", len(candidates))

    if mmr and redundancy_ratio(candidates[:selected_top_k]) >= 0.4:
        try:
            mmr_pool = min(initial_k, 50)
            mmr_k = min(mmr_pool, max(selected_top_k * 2, selected_top_k))
            mmr_candidates = candidates[:mmr_pool]
            candidates = apply_mmr(
                mmr_candidates,
                query_embedding,
                top_k=mmr_k,
                lambda_mult=0.7,
                collection=collection,
            )
            candidates = apply_rank_scores(candidates)
        except (RuntimeError, ValueError, TypeError) as exc:
            LOGGER.warning("MMR failed, using distance ranking. %s", exc)
    elif debug and not mmr:
        LOGGER.info("Debug: mmr=disabled")
    if debug:
        LOGGER.info("Debug: candidates_after_mmr=%d", len(candidates))

    candidates = apply_max_per_source(
        candidates, max_per_source=selected_max_per_source, top_k=selected_top_k
    )
    if debug:
        LOGGER.info("Debug: candidates_after_cap=%d", len(candidates))
    expanded = expand_candidates(
        candidates,
        collection=collection,
        neighbor_expansion=neighbor_expansion,
        include_parent=selected_include_parent,
        max_parents=max_parents,
    )
    if debug:
        LOGGER.info("Debug: expanded_candidates=%d", len(expanded))

    expanded.sort(key=lambda c: c.score, reverse=True)
    if mode == "recherche":
        expanded = apply_soft_cap_per_source(expanded, max_per_source=selected_max_per_source)
        if debug:
            LOGGER.info("Debug: expanded_after_soft_cap=%d", len(expanded))
    overhead = estimate_prompt_overhead_tokens(query)
    min_context = min(DEFAULT_MIN_CONTEXT_TOKENS, selected_context_max_tokens)
    context_budget = max(min_context, selected_context_max_tokens - overhead)
    context_blocks, used_candidates, used_tokens, budget_left, stopped_by_budget = build_context_blocks(
        expanded, context_budget
    )
    if debug:
        LOGGER.info(
            "Debug: overhead=%d, context_budget=%d, context_blocks=%d",
            overhead,
            context_budget,
            len(context_blocks),
        )
        LOGGER.info(
            "Debug: context_used_tokens=%d, budget_left=%d, stopped_by_budget=%s",
            used_tokens,
            budget_left,
            stopped_by_budget,
        )

    block_labels: Dict[str, Candidate] = {}
    for idx, cand in enumerate(used_candidates, start=1):
        block_labels[f"C{idx}"] = cand

    model_json = generate_answer(
        client=client,
        model=generation_model,
        query=query,
        context_blocks=context_blocks,
        max_retries=max_retries,
        retry_base_delay_sec=retry_base_delay_sec,
    )

    return build_final_output(
        query=query,
        mode=mode,
        model=generation_model,
        used_blocks=used_candidates,
        block_labels=block_labels,
        model_json=model_json,
    )
