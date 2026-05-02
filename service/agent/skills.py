from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills"
_MARKDOWN_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+([^\n]+)\n(.*?)(?=^##\s+|\Z)", re.MULTILINE | re.DOTALL)
_REQUIRED_METADATA_FIELDS: Tuple[str, ...] = (
    "skill_name",
    "query_types",
    "required_slots",
    "tool_chain",
    "guardrails",
    "slot_schema",
    "execution_config",
    "few_shot_examples",
    "tool_constraints",
)


@dataclass(frozen=True)
class SkillPackage:
    task_description: str
    prompt_template: str
    few_shot_examples: Tuple[Mapping[str, Any], ...] = tuple()
    slot_schema: Dict[str, Any] = field(default_factory=dict)
    tool_constraints: Dict[str, Any] = field(default_factory=dict)
    execution_config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillDefinition:
    skill_name: str
    query_types: Tuple[str, ...]
    required_slots: Tuple[str, ...]
    input_schema: Dict[str, Any]
    tool_chain: Tuple[str, ...]
    output_schema: Dict[str, Any]
    guardrails: Dict[str, Any]
    trace_fields: Tuple[str, ...]
    package: SkillPackage

    def get_missing_slots(self, slots: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        for slot in self._effective_required_slots():
            value = slots.get(slot)
            if isinstance(value, list):
                if len(value) == 0:
                    missing.append(slot)
                elif slot == "compare_targets" and len(value) < 2:
                    missing.append(slot)
                continue
            if not value:
                missing.append(slot)
        return missing

    def _effective_required_slots(self) -> Tuple[str, ...]:
        if self.required_slots:
            return self.required_slots
        raw_required = self.package.slot_schema.get("required", [])
        if not isinstance(raw_required, list):
            return tuple()
        return tuple(str(item) for item in raw_required if str(item).strip())

    def package_metadata(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "query_types": list(self.query_types),
            "task_description": self.package.task_description,
            "prompt_template": self.package.prompt_template,
            "few_shot_examples": list(self.package.few_shot_examples),
            "slot_schema": dict(self.package.slot_schema),
            "tool_constraints": dict(self.package.tool_constraints),
            "execution_config": dict(self.package.execution_config),
            "tool_chain": list(self.tool_chain),
            "guardrails": dict(self.guardrails),
        }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _to_tuple(values: Iterable[Any]) -> Tuple[str, ...]:
    result: List[str] = []
    for item in values:
        text = _clean_text(item)
        if text:
            result.append(text)
    return tuple(result)


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _extract_metadata(markdown_text: str, source: Path) -> Dict[str, Any]:
    match = _MARKDOWN_CODE_BLOCK_RE.search(markdown_text)
    if not match:
        raise ValueError(f"Skill markdown is missing a JSON metadata block: {source}")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata in skill markdown: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Skill metadata must be a JSON object: {source}")
    return payload


def _extract_sections(markdown_text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for title, body in _SECTION_RE.findall(markdown_text):
        sections[_clean_text(title).lower()] = _clean_text(body)
    return sections


def _require_fields(metadata: Mapping[str, Any], source: Path) -> None:
    missing = [field for field in _REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(
            f"Skill markdown metadata is missing required fields {missing}: {source}"
        )


def _ensure_non_empty(value: str, field_name: str, source: Path) -> str:
    if value:
        return value
    raise ValueError(f"Skill markdown is missing required field '{field_name}': {source}")


def load_skill_definition(path: str | Path) -> SkillDefinition:
    source = Path(path)
    markdown_text = source.read_text(encoding="utf-8")
    metadata = _extract_metadata(markdown_text, source)
    sections = _extract_sections(markdown_text)
    _require_fields(metadata, source)

    task_description = sections.get("task description") or _clean_text(metadata.get("task_description"))
    prompt_template = sections.get("prompt template") or _clean_text(metadata.get("prompt_template"))
    task_description = _ensure_non_empty(task_description, "task_description", source)
    prompt_template = _ensure_non_empty(prompt_template, "prompt_template", source)

    package = SkillPackage(
        task_description=task_description,
        prompt_template=prompt_template,
        few_shot_examples=tuple(metadata.get("few_shot_examples") or []),
        slot_schema=_to_dict(metadata.get("slot_schema")),
        tool_constraints=_to_dict(metadata.get("tool_constraints")),
        execution_config=_to_dict(metadata.get("execution_config")),
    )

    return SkillDefinition(
        skill_name=_ensure_non_empty(_clean_text(metadata.get("skill_name")), "skill_name", source),
        query_types=_to_tuple(metadata.get("query_types") or []),
        required_slots=_to_tuple(metadata.get("required_slots") or []),
        input_schema=_to_dict(metadata.get("input_schema")),
        tool_chain=_to_tuple(metadata.get("tool_chain") or []),
        output_schema=_to_dict(metadata.get("output_schema")),
        guardrails=_to_dict(metadata.get("guardrails")),
        trace_fields=_to_tuple(metadata.get("trace_fields") or []),
        package=package,
    )


def load_all_skills(skill_dir: str | Path | None = None) -> Tuple[SkillDefinition, ...]:
    base_dir = Path(skill_dir) if skill_dir is not None else SKILL_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Skill directory does not exist: {base_dir}")

    skills = [load_skill_definition(path) for path in sorted(base_dir.glob("*.md"))]
    if not skills:
        raise ValueError(f"No markdown skills found in {base_dir}")

    names = [skill.skill_name for skill in skills]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate skill_name found in markdown skills under {base_dir}")
    return tuple(skills)


ALL_SKILLS = load_all_skills()
_SKILLS_BY_NAME = {skill.skill_name: skill for skill in ALL_SKILLS}

FactLookupSkill = _SKILLS_BY_NAME["FactLookupSkill"]
TableQASkill = _SKILLS_BY_NAME["TableQASkill"]
CitationLocateSkill = _SKILLS_BY_NAME["CitationLocateSkill"]
SummarizationSkill = _SKILLS_BY_NAME["SummarizationSkill"]
ReportGenerationSkill = _SKILLS_BY_NAME["ReportGenerationSkill"]
MultiDocCompareSkill = _SKILLS_BY_NAME["MultiDocCompareSkill"]
