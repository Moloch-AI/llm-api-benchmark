from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from tests.helpers import TEST_KEY

from llm_api_benchmark.redact import REDACTOR


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FAKE_API_KEY", TEST_KEY)
    for name in ("MOLOCHAI_API_KEY", "MOLOCHAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    yield
    REDACTOR._secrets.clear()


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args: Any) -> None:  # keep test output clean
        pass

    def _send(
        self,
        status: int,
        body: bytes,
        ctype: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.server.seen.append(("GET", self.path, dict(self.headers)))
        if self.path.endswith("/models"):
            if self.headers.get("Authorization") != f"Bearer {TEST_KEY}":
                self._send(401, b'{"message":"Unauthenticated."}')
                return
            data = {
                "object": "list",
                "data": [
                    {"id": "local-a", "available": True},
                    {"id": "local-b", "available": False},
                ],
            }
            self._send(200, json.dumps(data).encode())
            return
        self._send(404, b'{"message":"Not found."}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.seen.append(("POST", self.path, dict(self.headers)))
        last = body["messages"][-1]["content"]
        if "[slow]" in last:
            time.sleep(1.5)
        if self.headers.get("Authorization") != f"Bearer {TEST_KEY}":
            # Echo the token back, as a careless server might: the tool must redact it.
            token = self.headers.get("Authorization", "")
            self._send(
                401,
                json.dumps(
                    {"error": {"type": "invalid_key", "message": f"bad token {token}"}}
                ).encode(),
            )
            return
        if not body.get("stream"):
            reply = {
                "id": "x",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Canberra"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            }
            self._send(200, json.dumps(reply).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("Can", "berra"):
            ev = {
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                "model": body["model"],
            }
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(
            b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}\n\n'
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[tuple[str, str, dict[str, str]]] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


@pytest.fixture
def local_server() -> Iterator[_Server]:
    """A real OpenAI-compatible HTTP server on localhost (real sockets, real streaming)."""
    srv = _Server()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
