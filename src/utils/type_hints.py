from __future__ import annotations

from typing import Any, Mapping, Protocol


class OpenAIEmbeddingsClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class OpenAIResponsesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class OpenAIClient(Protocol):
    embeddings: OpenAIEmbeddingsClient
    responses: OpenAIResponsesClient


class ChromaCollection(Protocol):
    def get(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def query(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def upsert(self, **kwargs: Any) -> Any: ...
    def update(self, **kwargs: Any) -> Any: ...
    def delete(self, **kwargs: Any) -> Any: ...
