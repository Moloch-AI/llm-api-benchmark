from __future__ import annotations

from llm_api_benchmark.sse import Comment, Event, SSEParser


def feed_all(lines: list[str]) -> list[Event | Comment]:
    p = SSEParser()
    out: list[Event | Comment] = []
    for line in lines:
        out.extend(p.feed(line))
    out.extend(p.flush())
    return out


def test_events_comments_and_done() -> None:
    items = feed_all([": ka", "", 'data: {"a":1}', "", "data: [DONE]", ""])
    assert items == [Comment("ka"), Event('{"a":1}'), Event("[DONE]")]


def test_multiline_data_and_event_name_and_crlf() -> None:
    items = feed_all(["event: error\r", 'data: {"x":\r', "data: 1}\r", "\r"])
    assert items == [Event('{"x":\n1}', "error")]


def test_field_without_space_and_unknown_fields() -> None:
    items = feed_all(["id: 7", "retry: 100", 'data:{"a":2}', ""])
    assert items == [Event('{"a":2}')]


def test_tolerates_missing_blank_lines_between_events() -> None:
    items = feed_all(['data: {"a":1}', 'data: {"a":2}', "data: [DONE]"])
    assert items == [Event('{"a":1}'), Event('{"a":2}'), Event("[DONE]")]


def test_flush_at_eof_and_empty_dispatch() -> None:
    p = SSEParser()
    assert p.feed("") == []
    assert p.feed('data: {"a":1}') == []
    assert p.flush() == [Event('{"a":1}')]
    assert p.flush() == []


def test_queue_comment_text_is_kept() -> None:
    [c] = feed_all([": queued · position 2 of 3 · waiting 4s"])
    assert isinstance(c, Comment)
    assert c.text.startswith("queued")
