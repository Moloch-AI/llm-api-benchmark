"""Report renderers: terminal table, Markdown, static HTML, and run comparisons."""

from __future__ import annotations

from .compare import (
    CompareError,
    Comparison,
    compare,
    comparison_json,
    comparison_markdown,
    comparison_text,
)
from .html import html_report
from .markdown import markdown_report
from .table import render_table, terminal_report

__all__ = [
    "CompareError",
    "Comparison",
    "compare",
    "comparison_json",
    "comparison_markdown",
    "comparison_text",
    "html_report",
    "markdown_report",
    "render_table",
    "terminal_report",
]
