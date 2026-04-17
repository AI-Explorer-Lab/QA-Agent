from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


LEVEL1_RE = re.compile(
    r"^\s*第\s*[0-9\u3007\u96f6\u4e00-\u9fa5\u4e24]+\s*节(?:\s*[:：]?\s*.*)?\s*$"
)
LEVEL2_RE = re.compile(
    r"^\s*[0-9\u4e00-\u9fa5\u4e24]+\s*[、.．]\s*.+$"
)
LEVEL3_RE = re.compile(
    r"^\s*[\(（]\s*[0-9\u4e00-\u9fa5\u4e24]+\s*[\)）]\s*.+$"
)


@dataclass
class TitleBlock:
    page_idx: int
    block_index: int
    text: str
    bbox: List[float]
    raw_keys: List[str]
    sub_type: Any


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_block_text(block: Dict[str, Any]) -> str:
    fragments: List[str] = []
    for line in block.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            content = _safe_text(span.get("content"))
            if content:
                fragments.append(content)
    return "".join(fragments).strip()


def _iter_title_blocks(pdf_info: Iterable[Dict[str, Any]]) -> Iterable[TitleBlock]:
    for page in pdf_info:
        page_idx = int(page.get("page_idx", -1))
        for block in page.get("para_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "title":
                continue
            text = _extract_block_text(block)
            if not text:
                continue
            yield TitleBlock(
                page_idx=page_idx,
                block_index=int(block.get("index", -1)),
                text=text,
                bbox=block.get("bbox", []),
                raw_keys=list(block.keys()),
                sub_type=block.get("sub_type"),
            )


def _classify_level(text: str) -> str:
    if LEVEL1_RE.match(text):
        return "L1"
    if LEVEL2_RE.match(text):
        return "L2"
    if LEVEL3_RE.match(text):
        return "L3+"
    return "OTHER"


def analyze_json(json_path: Path) -> Dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    pdf_info = payload.get("pdf_info", [])
    title_blocks = list(_iter_title_blocks(pdf_info))

    explicit_level_keys = {"level", "heading_level", "title_level", "outline_level", "sub_type"}
    explicit_level_supported = False
    explicit_level_evidence: List[str] = []

    for block in title_blocks:
        keys = set(block.raw_keys)
        present = sorted(keys & explicit_level_keys)
        if present:
            # sub_type exists but is often None; only count as explicit if it has a usable value.
            if "sub_type" in present and block.sub_type is None and len(present) == 1:
                continue
            explicit_level_supported = True
            explicit_level_evidence.append(
                f"page={block.page_idx}, index={block.block_index}, keys={present}, sub_type={block.sub_type}"
            )
            if len(explicit_level_evidence) >= 10:
                break

    by_level: Dict[str, List[TitleBlock]] = {"L1": [], "L2": [], "L3+": [], "OTHER": []}
    for block in title_blocks:
        by_level[_classify_level(block.text)].append(block)

    return {
        "file": str(json_path),
        "total_pages": len(pdf_info),
        "total_title_blocks": len(title_blocks),
        "explicit_level_supported": explicit_level_supported,
        "explicit_level_evidence": explicit_level_evidence,
        "l1": by_level["L1"],
        "l2": by_level["L2"],
        "l3_plus": by_level["L3+"],
        "other": by_level["OTHER"],
    }


def render_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("MinerU JSON 标题层级分析报告")
    lines.append("")
    lines.append(f"文件: {result['file']}")
    lines.append(f"总页数: {result['total_pages']}")
    lines.append(f"title 块数量: {result['total_title_blocks']}")
    lines.append("")
    lines.append("一、是否存在“显式一级/二级标题字段”")
    lines.append(f"- explicit_level_supported: {result['explicit_level_supported']}")
    if result["explicit_level_supported"]:
        lines.append("- 证据:")
        for item in result["explicit_level_evidence"]:
            lines.append(f"  - {item}")
    else:
        lines.append("- 结论: 当前 JSON 未提供可直接使用的标题层级字段（如 level/heading_level）。")
        lines.append("- 说明: `type=title` 存在，但只表示“标题块”，不直接给出一级/二级。")
    lines.append("")
    lines.append("二、是否可通过规则识别出一级/二级")
    lines.append(f"- L1（第X节）数量: {len(result['l1'])}")
    lines.append(f"- L2（一、二、三…）数量: {len(result['l2'])}")
    lines.append(f"- L3+（(一)(二)…）数量: {len(result['l3_plus'])}")
    lines.append(f"- 其他 title 数量: {len(result['other'])}")
    lines.append("")
    lines.append("三、样本（前 20 条）")
    for label, key in [("L1", "l1"), ("L2", "l2"), ("L3+", "l3_plus"), ("OTHER", "other")]:
        lines.append(f"- {label}:")
        for item in result[key][:20]:
            lines.append(f"  - p{item.page_idx} #{item.block_index}: {item.text}")
    lines.append("")
    lines.append("四、结论")
    lines.append("- 不能直接拿到“真一级/二级”（缺少显式层级字段）。")
    lines.append("- 可以高置信用规则拿到“业务可用的一/二级标题候选”。")
    lines.append("- 推荐策略: `type=title` + 正则分层 + 页码顺序重建目录。")
    return "\n".join(lines)


def find_default_json() -> Path:
    candidates = sorted(Path("docs/json_docs").glob("*.json"))
    if not candidates:
        raise FileNotFoundError("No json files found under docs/json_docs")
    return candidates[-1]


def main() -> int:
    json_path = find_default_json()
    output_dir = Path("test_folder")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = analyze_json(json_path)
    report_text = render_report(result)

    report_path = output_dir / "mineru_json_heading_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    # Also export flat heading list for quick manual check.
    list_path = output_dir / "mineru_json_headings_flat.txt"
    flat_lines = []
    for key in ("l1", "l2", "l3_plus", "other"):
        for item in result[key]:
            flat_lines.append(f"{key}\tp{item.page_idx}\t{item.block_index}\t{item.text}")
    list_path.write_text("\n".join(flat_lines), encoding="utf-8")

    print(f"report={report_path}")
    print(f"flat={list_path}")
    print(f"explicit_level_supported={result['explicit_level_supported']}")
    print(f"l1={len(result['l1'])}, l2={len(result['l2'])}, l3_plus={len(result['l3_plus'])}, other={len(result['other'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
