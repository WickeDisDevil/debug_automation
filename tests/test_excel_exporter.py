"""Smoke tests for the Excel exporter's sheet-name sanitizer.

Excel enforces two hard rules on worksheet names:
  * length <= 31 characters
  * none of the characters in `: \ / ? * [ ]`

`_sanitize_sheet_name` is the single chokepoint that protects every
write path (showcase report, per-category tabs) from those rules, so
these tests pin its contract directly.
"""

from __future__ import annotations

from bugfix_ai.categorization.excel_exporter import _sanitize_sheet_name


def test_truncates_to_31_chars():
    name = "a" * 50
    out = _sanitize_sheet_name(name)
    assert len(out) <= 31


def test_strips_forbidden_characters():
    out = _sanitize_sheet_name("audio:codec/wm[8804]?")
    for ch in r":\/?*[]":
        assert ch not in out


def test_non_empty_input_yields_non_empty_output():
    assert _sanitize_sheet_name("display/dp/link.c")


def test_preserves_safe_characters():
    out = _sanitize_sheet_name("audio_codec-wm8804")
    assert "audio_codec-wm8804" in out
