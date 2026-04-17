from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# L1/L2/L3 recovery (preserve good L2 behavior from current script).
L1_RE = re.compile(
    r"^\s*"
    r"\u7b2c"  # 第
    r"\s*[0-9\u3007\u96f6\u4e24\u4e00-\u9fa5]+"
    r"\s*\u8282"  # 节
    r"(?:\s*[:\uff1a]?\s*.*)?$"
)
L2_RE_CN = re.compile(
    r"^\s*[\u96f6\u3007\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07]+\s*[\u3001\.\uff0e]\s*.+$"
)
L3_RE_PAREN = re.compile(
    r"^\s*[\(\uff08]\s*[0-9\u96f6\u3007\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07]+\s*[\)\uff09]\s*.+$"
)

# Keep this exclusion to avoid the known false-structure issue:
# "1、xxx / 2、xxx" should not be treated as L2.
ARABIC_ENUM_RE = re.compile(r"^\s*\d+\s*[\u3001\.\uff0e]\s*.+$")

NOISE_RE = re.compile(
    r"^\s*[\u221a\u2713\u005c\/surd\s]*\u9002\u7528.*\u4e0d\u9002\u7528\s*$",
    flags=re.IGNORECASE,
)


@dataclass
class TitleCandidate:
    page_idx: int
    block_index: int
    text: str
    bbox: List[float]
    page_size: List[float]
    level: str
    confidence: float
    reasons: List[str]


@dataclass
class HeadingNode:
    node_id: str
    level: str
    text: str
    page_idx: int
    block_index: int
    confidence: float
    reasons: List[str]
    children: List["HeadingNode"]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_block_text(block: Dict[str, Any]) -> str:
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


def is_noise_title(text: str) -> bool:
    if not text:
        return True
    if len(text) <= 1:
        return True
    if NOISE_RE.match(text):
        return True
    return False


def detect_level(text: str) -> Tuple[str, float, List[str]]:
    reasons: List[str] = []
    if L1_RE.match(text):
        reasons.append("regex:L1_第X节")
        return "L1", 0.93, reasons

    # Keep Arabic plain enum excluded to preserve current L2 quality.
    if ARABIC_ENUM_RE.match(text):
        reasons.append("exclude:arabic_enum_plain")
        return "OTHER", 0.0, reasons

    if L2_RE_CN.match(text):
        reasons.append("regex:L2_chinese_enum")
        return "L2", 0.90, reasons

    if L3_RE_PAREN.match(text):
        reasons.append("regex:L3_parenthesized_enum")
        return "L3", 0.88, reasons

    return "OTHER", 0.0, reasons


def layout_adjustment(level: str, bbox: List[float], page_size: List[float]) -> Tuple[float, List[str]]:
    if len(bbox) != 4 or len(page_size) < 2:
        return 0.0, []

    x0, y0, x1, y1 = bbox
    page_width, page_height = page_size[0], page_size[1]
    center_x = (x0 + x1) / 2.0
    box_h = max(0.0, y1 - y0)
    delta = 0.0
    reasons: List[str] = []

    if level == "L1":
        if abs(center_x - page_width / 2.0) <= page_width * 0.26:
            delta += 0.03
            reasons.append("layout:L1_centered")
        if box_h >= page_height * 0.014:
            delta += 0.02
            reasons.append("layout:L1_relatively_tall")
    elif level == "L2":
        if x0 <= page_width * 0.26:
            delta += 0.03
            reasons.append("layout:L2_left_aligned")
        else:
            delta -= 0.03
            reasons.append("layout:L2_not_left")
    elif level == "L3":
        if page_width * 0.08 <= x0 <= page_width * 0.45:
            delta += 0.03
            reasons.append("layout:L3_indented")

    return delta, reasons


def iter_candidates(payload: Dict[str, Any]) -> Tuple[List[TitleCandidate], Dict[str, int]]:
    out: List[TitleCandidate] = []
    stats = {
        "total_title_blocks": 0,
        "noise_filtered": 0,
        "excluded_arabic_enum_plain": 0,
        "other_filtered": 0,
    }

    for page in payload.get("pdf_info", []) or []:
        page_idx = int(page.get("page_idx", -1))
        page_size = page.get("page_size", []) or []
        for block in page.get("para_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "title":
                continue
            stats["total_title_blocks"] += 1

            text = extract_block_text(block)
            if is_noise_title(text):
                stats["noise_filtered"] += 1
                continue

            level, score, reasons = detect_level(text)
            if level == "OTHER":
                if "exclude:arabic_enum_plain" in reasons:
                    stats["excluded_arabic_enum_plain"] += 1
                else:
                    stats["other_filtered"] += 1
                continue

            delta, layout_reasons = layout_adjustment(level, block.get("bbox", []) or [], page_size)
            score = max(0.0, min(1.0, score + delta))
            reasons = reasons + layout_reasons + ["type:title"]

            out.append(
                TitleCandidate(
                    page_idx=page_idx,
                    block_index=int(block.get("index", -1)),
                    text=text,
                    bbox=block.get("bbox", []) or [],
                    page_size=page_size,
                    level=level,
                    confidence=score,
                    reasons=reasons,
                )
            )

    out.sort(key=lambda c: (c.page_idx, c.block_index))
    return out, stats


def is_high_conf(candidate: TitleCandidate) -> bool:
    if candidate.level == "L1":
        return candidate.confidence >= 0.92
    if candidate.level == "L2":
        return candidate.confidence >= 0.88
    if candidate.level == "L3":
        return candidate.confidence >= 0.86
    return False


def dedupe_candidates(candidates: List[TitleCandidate]) -> List[TitleCandidate]:
    seen = set()
    deduped: List[TitleCandidate] = []
    for c in candidates:
        key = (c.level, c.text, c.page_idx)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def candidate_to_node(candidate: TitleCandidate, node_id: str) -> HeadingNode:
    return HeadingNode(
        node_id=node_id,
        level=candidate.level,
        text=candidate.text,
        page_idx=candidate.page_idx,
        block_index=candidate.block_index,
        confidence=round(candidate.confidence, 4),
        reasons=candidate.reasons,
        children=[],
    )


def recover_hierarchy(candidates: List[TitleCandidate]) -> List[HeadingNode]:
    roots: List[HeadingNode] = []
    front_matter = HeadingNode(
        node_id="root_front_matter",
        level="ROOT",
        text="front_matter",
        page_idx=-1,
        block_index=-1,
        confidence=1.0,
        reasons=["synthetic_root"],
        children=[],
    )
    roots.append(front_matter)

    current_l1: Optional[HeadingNode] = None
    current_l2: Optional[HeadingNode] = None
    seq = 1

    for c in candidates:
        node = candidate_to_node(c, node_id=f"h_{seq}")
        seq += 1

        if c.level == "L1":
            roots.append(node)
            current_l1 = node
            current_l2 = None
            continue

        if c.level == "L2":
            if current_l1 is not None:
                current_l1.children.append(node)
            else:
                front_matter.children.append(node)
            current_l2 = node
            continue

        # L3
        if current_l2 is not None:
            current_l2.children.append(node)
        elif current_l1 is not None:
            current_l1.children.append(node)
        else:
            front_matter.children.append(node)

    return roots


def serialize_nodes(nodes: List[HeadingNode]) -> List[Dict[str, Any]]:
    def _to_dict(node: HeadingNode) -> Dict[str, Any]:
        raw = asdict(node)
        raw["children"] = [_to_dict(child) for child in node.children]
        return raw

    return [_to_dict(node) for node in nodes]


def render_tree(nodes: List[HeadingNode]) -> str:
    lines: List[str] = []

    def _walk(node: HeadingNode, depth: int) -> None:
        indent = "  " * depth
        lines.append(
            f"{indent}- [{node.level}] p{node.page_idx} #{node.block_index} "
            f"(conf={node.confidence:.2f}) {node.text}"
        )
        for child in node.children:
            _walk(child, depth + 1)

    for node in nodes:
        _walk(node, 0)
    return "\n".join(lines)


def find_default_json() -> Path:
    candidates = sorted(Path("docs/json_docs").glob("*.json"))
    if not candidates:
        raise FileNotFoundError("No json files found under docs/json_docs")
    return candidates[-1]


def main() -> int:
    source_json = find_default_json()
    payload = json.loads(source_json.read_text(encoding="utf-8"))

    candidates, stats = iter_candidates(payload)
    high_conf = [c for c in candidates if is_high_conf(c)]
    high_conf = dedupe_candidates(high_conf)
    roots = recover_hierarchy(high_conf)

    out_dir = Path("test_folder")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "mineru_heading_hierarchy_l123_summary.txt"
    tree_txt_path = out_dir / "mineru_heading_hierarchy_l123_tree.txt"
    tree_json_path = out_dir / "mineru_heading_hierarchy_l123_tree.json"
    candidates_tsv_path = out_dir / "mineru_heading_hierarchy_l123_candidates.tsv"

    l1_count = sum(1 for c in high_conf if c.level == "L1")
    l2_count = sum(1 for c in high_conf if c.level == "L2")
    l3_count = sum(1 for c in high_conf if c.level == "L3")

    summary_lines = [
        "High-confidence hierarchy recovery summary (L1/L2/L3)",
        f"source={source_json}",
        f"total_title_blocks={stats['total_title_blocks']}",
        f"noise_filtered={stats['noise_filtered']}",
        f"excluded_arabic_enum_plain={stats['excluded_arabic_enum_plain']}",
        f"other_filtered={stats['other_filtered']}",
        f"total_candidates_after_filter={len(candidates)}",
        f"high_conf_candidates={len(high_conf)}",
        f"L1={l1_count}, L2={l2_count}, L3={l3_count}",
        "rule=type:title + regex(L1/L2-cn/L3-paren) + bbox layout; arabic plain enum excluded",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    tree_txt_path.write_text(render_tree(roots), encoding="utf-8")
    tree_json_path.write_text(json.dumps(serialize_nodes(roots), ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_lines = ["level\tpage_idx\tblock_index\tconfidence\ttext\treasons"]
    for c in high_conf:
        tsv_lines.append(
            f"{c.level}\t{c.page_idx}\t{c.block_index}\t{c.confidence:.4f}\t{c.text}\t{'|'.join(c.reasons)}"
        )
    candidates_tsv_path.write_text("\n".join(tsv_lines), encoding="utf-8")

    print(f"summary={summary_path}")
    print(f"tree_txt={tree_txt_path}")
    print(f"tree_json={tree_json_path}")
    print(f"candidates_tsv={candidates_tsv_path}")
    print(f"excluded_arabic_enum_plain={stats['excluded_arabic_enum_plain']}")
    print(f"high_conf_candidates={len(high_conf)}")
    print(f"L1={l1_count}, L2={l2_count}, L3={l3_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
