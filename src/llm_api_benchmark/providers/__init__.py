"""Provider adapters and presets.

An adapter turns a benchmark prompt into an HTTP request and turns the provider's response
(streamed or not) into the fields the runner measures. Adapters are selected by a preset's
``api`` value. To support a different wire format, implement :class:`base.Adapter` and register
it in :data:`ADAPTERS`.
"""

from __future__ import annotations

from .base import Adapter, ChatRequest, ChunkInfo, CompletionInfo, ErrorInfo
from .openai_chat import OpenAIChatAdapter
from .presets import PRESETS, Preset

ADAPTERS: dict[str, type[Adapter]] = {
    "openai-chat": OpenAIChatAdapter,
    # The synthetic endpoint speaks the same wire format through an in-process transport.
    "synthetic": OpenAIChatAdapter,
}

__all__ = [
    "ADAPTERS",
    "PRESETS",
    "Adapter",
    "ChatRequest",
    "ChunkInfo",
    "CompletionInfo",
    "ErrorInfo",
    "OpenAIChatAdapter",
    "Preset",
]
