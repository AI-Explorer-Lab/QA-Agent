from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


L1_RE = re.compile(
    r"^\s*"
    r"第"
    r"\s*[0-9〇零两一-龥]+"
    r"\s*节"
    r"(?:\s*[:：]?\s*.*)?$"
)
L2_RE_CN = re.compile(
    r"^\s*[零〇一二两三四五六七八九十百千万]+\s*[、.．]\s*.+$"
)
L3_RE_PAREN = re.compile(
    r"^\s*[\(（]\s*[0-9零〇一二两三四五六七八九十百千万]+\s*[\)）]\s*.+$"
)
ARABIC_ENUM_RE = re.compile(r"^\s*\d+\s*[、.．]\s*.+$")
NOISE_RE = re.compile(
    r"^\s*[√✓\\/\s]*适用.*不适用\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RecoveredHeading:
    level: int
    text: str
    confidence: float
    page_idx: int = -1
    block_index: int = -1
    match_key: str = ""


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_heading_match_key(text: Any) -> str:
    normalized = normalize_text(text)
    normalized = (
        normalized.replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
        .replace("．", ".")
        .replace("。", ".")
        .replace("，", ",")
        .replace("、", ".")
    )
    normalized = normalized.replace(" ", "")
    return normalized


def is_noise_title(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    if len(normalized) <= 1:
        return True
    return bool(NOISE_RE.match(normalized))


def detect_heading_level(text: str) -> Tuple[int, float, List[str]]:
    reasons: List[str] = []
    normalized = normalize_text(text)

    if L1_RE.match(normalized):
        reasons.append("regex:L1")
        return 1, 0.93, reasons

    if ARABIC_ENUM_RE.match(normalized):
        reasons.append("exclude:arabic_enum_plain")
        return 0, 0.0, reasons

    if L2_RE_CN.match(normalized):
        reasons.append("regex:L2_chinese_enum")
        return 2, 0.90, reasons

    if L3_RE_PAREN.match(normalized):
        reasons.append("regex:L3_parenthesized_enum")
        return 3, 0.88, reasons

    return 0, 0.0, reasons


def _extract_block_text(block: Dict[str, Any]) -> str:
    fragments: List[str] = []
    for line in block.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            content = normalize_text(span.get("content"))
            if content:
                fragments.append(content)
    return normalize_text("".join(fragments))


def _layout_adjustment(level: int, bbox: List[float], page_size: List[float]) -> Tuple[float, List[str]]:
    if len(bbox) != 4 or len(page_size) < 2:
        return 0.0, []

    x0, y0, x1, y1 = bbox
    page_width, page_height = page_size[0], page_size[1]
    center_x = (x0 + x1) / 2.0
    box_h = max(0.0, y1 - y0)
    delta = 0.0
    reasons: List[str] = []

    if level == 1:
        if abs(center_x - page_width / 2.0) <= page_width * 0.26:
            delta += 0.03
            reasons.append("layout:L1_centered")
        if box_h >= page_height * 0.014:
            delta += 0.02
            reasons.append("layout:L1_tall")
    elif level == 2:
        if x0 <= page_width * 0.26:
            delta += 0.03
            reasons.append("layout:L2_left")
        else:
            delta -= 0.03
            reasons.append("layout:L2_not_left")
    elif level == 3:
        if page_width * 0.08 <= x0 <= page_width * 0.45:
            delta += 0.03
            reasons.append("layout:L3_indented")

    return delta, reasons


def _is_high_conf(level: int, confidence: float) -> bool:
    if level == 1:
        return confidence >= 0.92
    if level == 2:
        return confidence >= 0.88
    if level == 3:
        return confidence >= 0.86
    return False


def _dedupe_headings(headings: List[RecoveredHeading]) -> List[RecoveredHeading]:
    seen = set()
    deduped: List[RecoveredHeading] = []
    for item in headings:
        key = (item.level, item.text, item.page_idx)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def recover_high_conf_headings_from_mineru_payload(payload: Dict[str, Any]) -> List[RecoveredHeading]:
    recovered: List[RecoveredHeading] = []

    for page in payload.get("pdf_info", []) or []:
        page_idx = int(page.get("page_idx", -1))
        page_size = page.get("page_size", []) or []
        for block in page.get("para_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "title":
                continue

            text = _extract_block_text(block)
            if is_noise_title(text):
                continue

            level, base_confidence, _ = detect_heading_level(text)
            if level <= 0:
                continue

            delta, _ = _layout_adjustment(level, block.get("bbox", []) or [], page_size)
            confidence = max(0.0, min(1.0, base_confidence + delta))
            if not _is_high_conf(level, confidence):
                continue

            recovered.append(
                RecoveredHeading(
                    level=level,
                    text=normalize_text(text),
                    confidence=confidence,
                    page_idx=page_idx,
                    block_index=int(block.get("index", -1)),
                    match_key=normalize_heading_match_key(text),
                )
            )

    recovered.sort(key=lambda item: (item.page_idx, item.block_index))
    return _dedupe_headings(recovered)


def recover_headings_from_paragraph_texts(paragraph_texts: Iterable[str]) -> List[RecoveredHeading]:
    recovered: List[RecoveredHeading] = []
    for index, paragraph_text in enumerate(paragraph_texts):
        text = normalize_text(paragraph_text)
        if is_noise_title(text):
            continue

        level, confidence, _ = detect_heading_level(text)
        if level <= 0:
            continue

        recovered.append(
            RecoveredHeading(
                level=level,
                text=text,
                confidence=confidence,
                page_idx=-1,
                block_index=index,
                match_key=normalize_heading_match_key(text),
            )
        )

    return _dedupe_headings(recovered)
