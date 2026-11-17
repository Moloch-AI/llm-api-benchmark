"""Incremental server-sent events parser.

Follows the SSE format (``data:`` lines joined until a blank line, ``event:`` names, comment lines
starting with ``:``), with one tolerance for servers that omit the blank line between events: a
new ``data:`` line after a complete JSON payload starts a new event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    data: str
    event: str | None = None


@dataclass(frozen=True)
class Comment:
    text: str


class SSEParser:
    def __init__(self) -> None:
        self._data: list[str] = []
        self._event: str | None = None

    def feed(self, line: str) -> list[Event | Comment]:
        """Feed one line (without its line terminator). Returns what it completed."""
        line = line.rstrip("\r\n")
        out: list[Event | Comment] = []
        if line == "":
            ev = self._dispatch()
            if ev:
                out.append(ev)
            return out
        if line.startswith(":"):
            out.append(Comment(line[1:].strip()))
            return out
        field, sep, value = line.partition(":")
        if not sep:
            field, value = line, ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            if self._data and _complete("\n".join(self._data)):
                ev = self._dispatch()
                if ev:
                    out.append(ev)
            self._data.append(value)
        elif field == "event":
            self._event = value or None
        # "id" and "retry" are irrelevant for a single request; unknown fields are ignored.
        return out

    def flush(self) -> list[Event | Comment]:
        ev = self._dispatch()
        return [ev] if ev else []

    def _dispatch(self) -> Event | None:
        if not self._data:
            self._event = None
            return None
        ev = Event("\n".join(self._data), self._event)
        self._data = []
        self._event = None
        return ev


def _complete(payload: str) -> bool:
    if payload.strip() == "[DONE]":
        return True
    try:
        json.loads(payload)
    except ValueError:
        return False
    return True
