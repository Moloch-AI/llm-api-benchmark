"""The adapter interface between the runner and a provider's wire format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import ProviderConfig


class InvalidResponse(ValueError):
    """The provider answered with something that is not a valid response of its format."""


@dataclass(frozen=True)
class ChatRequest:
    url: str
    headers: dict[str, str]
    query: dict[str, str]
    body: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ErrorInfo:
    type: str | None
    message: str


@dataclass(frozen=True)
class ChunkInfo:
    """What one streamed event contributed."""

    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str | None = None
    error: ErrorInfo | None = None


@dataclass(frozen=True)
class CompletionInfo:
    content: str
    reasoning: str
    finish_reason: str | None
    usage: Usage | None
    model: str | None


class Adapter(ABC):
    """Builds requests for one provider endpoint and parses its responses."""

    def __init__(
        self, provider: ProviderConfig, api_key: str | None, extra_headers: dict[str, str]
    ):
        self.provider = provider
        self._api_key = api_key
        self._extra_headers = extra_headers

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(provider={self.provider.name!r})"

    @abstractmethod
    def build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        max_tokens: int | None,
        temperature: float | None,
    ) -> ChatRequest: ...

    @abstractmethod
    def parse_chunk(self, data: Any) -> ChunkInfo: ...

    @abstractmethod
    def parse_completion(self, data: Any) -> CompletionInfo: ...

    @abstractmethod
    def parse_error(self, status: int, body: bytes, content_type: str) -> ErrorInfo: ...
