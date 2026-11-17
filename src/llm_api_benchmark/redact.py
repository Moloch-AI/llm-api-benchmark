"""Keep credentials out of terminal output, logs and result files.

Keys are read from environment variables and registered here. Every string that reaches the
terminal, a log record or a result file (error messages, response bodies, URLs) passes through
:func:`redact`, which replaces registered secrets and anything that looks like a bearer token.
Result files never contain any part of a key, not even the last characters.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit, urlunsplit

_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")
# Common key shapes: sk-..., sk-proj-..., and long opaque tokens after "key=" in query strings.
_SK = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")
_QUERY_KEY = re.compile(r"(?i)\b((?:api[_-]?key|access[_-]?token|token|key)=)[^&\s\"']{6,}")


def mask(secret: str | None) -> str:
    """Short, safe description of a secret for the terminal: never more than the last 4 characters."""
    if not secret:
        return "(not set)"
    if len(secret) < 12:
        return "****"
    return "****" + secret[-4:]


class Redactor:
    """Replaces registered secrets and token-shaped substrings in text."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def add(self, secret: str | None) -> None:
        if secret and len(secret) >= 4:
            self._secrets.add(secret)

    def __call__(self, text: str) -> str:
        for secret in sorted(self._secrets, key=len, reverse=True):
            if secret in text:
                text = text.replace(secret, "[redacted]")
        text = _BEARER.sub(r"\1[redacted]", text)
        text = _SK.sub("[redacted]", text)
        return _QUERY_KEY.sub(r"\1[redacted]", text)


REDACTOR = Redactor()


def redact(text: str) -> str:
    return REDACTOR(text)


def redact_url(url: str) -> str:
    """Drop user:password from a URL and redact key-like query values."""
    parts = urlsplit(url)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "[redacted]@" + netloc.rsplit("@", 1)[1]
    return redact(urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)))


class RedactingFilter(logging.Filter):
    """Logging filter that formats the message and redacts it before any handler sees it."""

    def __init__(self, redactor: Redactor = REDACTOR) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redactor(record.getMessage())
        record.args = None
        return True
