from __future__ import annotations

import re
from dataclasses import dataclass

from utils.content_normalizer import normalize_whitespace

_L1_RE = re.compile(r"^第\s*[0-9零〇一二两三四五六七八九十百千万]+\s*节(?:\s+.*)?$")
_L2_RE = re.compile(r"^[零〇一二两三四五六七八九十百千万]+\s*[、.．]\s*.+$")
_L3_RE = re.compile(r"^[（(]\s*[0-9零〇一二两三四五六七八九十百千万]+\s*[)）]\s*.+$")
_ARABIC_ENUM_RE = re.compile(r"^\d+\s*[、.．]\s*.+$")


@dataclass(frozen=True)
class HeadingState:
    level1_title: str = ""
    level2_title: str = ""
    level3_title: str = ""



def normalize_heading_text(text: str) -> str:
    return normalize_whitespace(text, preserve_newlines=False)


def detect_heading_level(text: str) -> int:
    normalized = normalize_heading_text(text)
    if not normalized:
        return 0

    if _ARABIC_ENUM_RE.match(normalized):
        return 0
    if _L1_RE.match(normalized):
        return 1
    if _L2_RE.match(normalized):
        return 2
    if _L3_RE.match(normalized):
        return 3
    return 0


def apply_heading(state: HeadingState, heading_text: str, level: int | None = None) -> HeadingState:
    normalized = normalize_heading_text(heading_text)
    level_value = int(level) if level is not None else detect_heading_level(normalized)
    if level_value <= 0:
        return state

    if level_value == 1:
        return HeadingState(level1_title=normalized)
    if level_value == 2:
        return HeadingState(
            level1_title=state.level1_title,
            level2_title=normalized,
        )
    return HeadingState(
        level1_title=state.level1_title,
        level2_title=state.level2_title,
        level3_title=normalized,
    )


def build_heading_metadata(state: HeadingState) -> dict[str, str]:
    parts = [item for item in [state.level1_title, state.level2_title, state.level3_title] if item]
    return {
        "level1_title": state.level1_title,
        "level2_title": state.level2_title,
        "level3_title": state.level3_title,
        "heading_path": " > ".join(parts) if parts else "front_matter",
    }

